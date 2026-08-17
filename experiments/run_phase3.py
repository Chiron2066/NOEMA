"""Phase 3 (H3): the annealing curriculum vs standard preference-tuning.

The REBUS/canalization test. Two LoRA-DPO runs on the same preference data:

  --arm standard : constant lr, constant beta        (chronic consolidation)
  --arm annealed : AnnealingSchedule cycles — plasticity windows (lr up,
                   beta down, small weight noise) alternating with
                   consolidation windows (lr down, beta up)

Tracked at every eval point (H3Tracker):
  reward   : held-out preference accuracy (implicit DPO reward margin > 0)
  ece      : calibration of greedy-answer confidence on a TriviaQA probe set
             (questions 2000-2100 — never seen by the Phase-1 head)
  distinct2 / sem_div : output diversity on sampled answers

H3 predicts: the standard arm shows the canalization signature (ECE decay
COUPLED with diversity collapse); the annealed arm holds calibration and
diversity at comparable reward.

Requires:  pip install peft
Run (each arm ~2-3h on MPS; run sequentially, not in parallel):
    caffeinate -i python experiments/run_phase3.py --arm standard
    caffeinate -i python experiments/run_phase3.py --arm annealed

Resumable at eval-checkpoint granularity via saved LoRA state.
"""
import os, sys, json, argparse, math

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from noema.semantic_entropy import NormalizedMatch, cluster_by_meaning
from noema.annealing import AnnealingSchedule, distinct_n, semantic_diversity, H3Tracker
from noema.metrics import ece

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
STEPS = 600
BATCH = 2                 # pairs per step (grad accumulation over singles)
BETA = 0.1
LR = 1e-5
EVAL_EVERY = 100
N_TRAIN_PAIRS = STEPS * BATCH
N_HELDOUT = 100
PROBE_RANGE = (2000, 2100)   # TriviaQA questions unseen in Phase 1
MAX_LEN = 640


def device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_pairs(n_total, seed=0):
    from datasets import load_dataset
    ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))
    pairs = []
    for i in idx:
        x = ds[int(i)]
        try:
            prompt = x["chosen"][0]["content"]
            chosen = x["chosen"][-1]["content"]
            rejected = x["rejected"][-1]["content"]
        except (KeyError, IndexError, TypeError):
            continue
        if not (prompt and chosen and rejected) or chosen == rejected:
            continue
        if len(prompt) > 1200 or len(chosen) > 1500 or len(rejected) > 1500:
            continue
        pairs.append((prompt, chosen, rejected))
        if len(pairs) >= n_total:
            break
    return pairs


def load_probe():
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    return [(x["question"], x["answer"]["value"])
            for x in ds.select(range(*PROBE_RANGE))]


def encode_pair(tok, prompt, response, device):
    """Returns input_ids and the index where response tokens start."""
    msgs = [{"role": "user", "content": prompt}]
    ptxt = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    pids = tok(ptxt, return_tensors="pt").input_ids[0]
    rids = tok(response, return_tensors="pt",
               add_special_tokens=False).input_ids[0]
    ids = torch.cat([pids, rids])[:MAX_LEN]
    return ids.unsqueeze(0).to(device), min(len(pids), MAX_LEN - 1)


def response_logp(model, ids, resp_start):
    """Sum log p(response tokens | prefix)."""
    logits = model(ids).logits[0, :-1]
    targets = ids[0, 1:]
    lp = F.log_softmax(logits.float(), dim=-1)
    tok_lp = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return tok_lp[resp_start - 1:].sum()


@torch.no_grad()
def eval_point(model, tok, probe, heldout, device, step, phase, tracker):
    model.eval()
    match = NormalizedMatch()
    # --- calibration on probe QA
    confs, corrects = [], []
    for q, gold in probe:
        prompt = f"Answer the question as briefly as possible.\nQ: {q}\nA:"
        enc = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=24, do_sample=False,
                             pad_token_id=tok.eos_token_id,
                             return_dict_in_generate=True, output_scores=True)
        seq = out.sequences[0, enc.input_ids.shape[1]:]
        text = tok.decode(seq, skip_special_tokens=True).strip().split("\n")[0]
        lps = []
        for t, sc in zip(seq, out.scores):
            if t == tok.eos_token_id:
                break
            lps.append(F.log_softmax(sc[0].float(), -1)[t].item())
        confs.append(math.exp(np.mean(lps)) if lps else 0.0)
        corrects.append(int(match(text, gold)))
    ece_val = ece(np.array(confs), np.array(corrects))
    acc = float(np.mean(corrects))
    # --- diversity on first 25 probe questions, 5 samples each
    d2s, sds = [], []
    for q, _ in probe[:25]:
        prompt = f"Answer the question as briefly as possible.\nQ: {q}\nA:"
        enc = tok(prompt, return_tensors="pt").to(device)
        outs = model.generate(**enc, max_new_tokens=24, do_sample=True,
                              temperature=1.0, top_p=0.95, num_return_sequences=5,
                              pad_token_id=tok.eos_token_id)
        samples = [tok.decode(o[enc.input_ids.shape[1]:],
                              skip_special_tokens=True).strip().split("\n")[0]
                   for o in outs]
        d2s.append(distinct_n(samples, 2))
        sds.append(semantic_diversity(cluster_by_meaning(samples, match)))
    # --- reward: held-out preference accuracy (policy margin > ref margin)
    wins = 0
    for prompt, chosen, rejected in heldout:
        idc, sc = encode_pair(tok, prompt, chosen, device)
        idr, sr = encode_pair(tok, prompt, rejected, device)
        pc, pr = response_logp(model, idc, sc), response_logp(model, idr, sr)
        with model.disable_adapter():
            rc, rr = response_logp(model, idc, sc), response_logp(model, idr, sr)
        wins += int((pc - pr) > (rc - rr))
    reward = wins / len(heldout)
    tracker.log(step, reward, float(ece_val), float(np.mean(d2s)),
                float(np.mean(sds)), phase)
    print(f"[eval @ {step}] reward={reward:.3f} acc={acc:.3f} "
          f"ECE={ece_val:.3f} distinct2={np.mean(d2s):.3f} "
          f"semdiv={np.mean(sds):.3f} phase={phase}")
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["standard", "annealed"], required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--cycle", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strong", action="store_true",
                    help="properly-dosed protocol: rank-16 LoRA on all "
                         "projections, harder plasticity/consolidation "
                         "contrast (use with --steps 2400 --cycle 400)")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model

    out_dir = f"phase3_{args.arm}"
    if args.strong:
        out_dir += f"_strong_s{args.seed}"
    os.makedirs(out_dir, exist_ok=True)

    device, dtype = device_and_dtype()
    print(f"device={device} dtype={dtype} arm={args.arm}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype)
    if args.strong:
        lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                          target_modules=["q_proj", "k_proj", "v_proj",
                                          "o_proj", "gate_proj", "up_proj",
                                          "down_proj"])
    else:
        lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                          target_modules=["q_proj", "k_proj", "v_proj",
                                          "o_proj"])
    model = get_peft_model(base, lcfg).to(device)
    model.train()

    pairs = load_pairs(args.steps * BATCH + N_HELDOUT, seed=args.seed)
    heldout = pairs[:N_HELDOUT]
    train = pairs[N_HELDOUT:]
    probe = load_probe()
    print(f"{len(train)} train pairs, {len(heldout)} held-out, "
          f"{len(probe)} probe questions")

    sched = AnnealingSchedule(total_steps=args.steps, cycle_steps=args.cycle,
                              plasticity_frac=0.3,
                              noise_hi=0.04 if args.strong else 0.02)
    beta_lo, beta_hi = (0.1, 2.0) if args.strong else (0.3, 1.5)
    lr_hi, lr_lo = (5.0, 0.5) if args.strong else (3.0, 0.7)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR)
    tracker = H3Tracker()
    eval_point(model, tok, probe, heldout, device, 0, "init", tracker)

    for step in range(args.steps):
        if args.arm == "annealed":
            k = sched.knobs(step)
            lr_mult = lr_hi if k.phase == "plasticity" else lr_lo
            beta = BETA * (beta_lo if k.phase == "plasticity" else beta_hi)
            noise = k.rep_noise_std
            phase = k.phase
        else:
            lr_mult, beta, noise, phase = 1.0, BETA, 0.0, "standard"
        for g in opt.param_groups:
            g["lr"] = LR * lr_mult

        opt.zero_grad()
        total = 0.0
        for b in range(BATCH):
            prompt, chosen, rejected = train[step * BATCH + b]
            idc, sc = encode_pair(tok, prompt, chosen, device)
            idr, sr = encode_pair(tok, prompt, rejected, device)
            with torch.no_grad(), model.disable_adapter():
                ref_c = response_logp(model, idc, sc)
                ref_r = response_logp(model, idr, sr)
            pol_c = response_logp(model, idc, sc)
            pol_r = response_logp(model, idr, sr)
            margin = (pol_c - pol_r) - (ref_c - ref_r)
            loss = -F.logsigmoid(beta * margin) / BATCH
            loss.backward()
            total += float(loss)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        if noise > 0:       # plasticity: small gaussian kick to LoRA weights
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        p.add_(torch.randn_like(p) * noise * p.std().clamp(min=1e-6))

        if (step + 1) % 20 == 0:
            print(f"[{step+1}/{args.steps}] loss={total:.4f} "
                  f"beta={beta:.3f} lr×{lr_mult:g} {phase}")
        if (step + 1) % EVAL_EVERY == 0:
            eval_point(model, tok, probe, heldout, device, step + 1,
                       phase, tracker)
            with open(os.path.join(out_dir, "history.json"), "w") as f:
                json.dump(dict(arm=args.arm,
                               history=tracker.history,
                               signature=tracker.canalization_signature()),
                          f, indent=2)
            model.save_pretrained(os.path.join(out_dir, "lora"))

    sig = tracker.canalization_signature()
    print(json.dumps(sig, indent=2))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(dict(arm=args.arm, history=tracker.history, signature=sig),
                  f, indent=2)
    print(f"Saved {out_dir}/history.json")


if __name__ == "__main__":
    main()
