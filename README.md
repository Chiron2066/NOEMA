# NOEMA — Intrinsic Precision Weighting for Hallucination-Resistant Generation

Prototype implementation of the NOEMA research proposal: distill sampled
semantic entropy (Farquhar et al., Nature 2024) into an intrinsic **precision
head**, route it into a **generate/retrieve/hedge/abstain gate**, and protect
its calibration during preference-tuning with an **annealing curriculum**.

## Hypothesis → experiment map

| Hypothesis | Test | Code | Endpoint |
|---|---|---|---|
| **H2** Amortization: a probe distilled from sampled SE detects confabulation at 1 forward pass | Train head on SE targets; compare AUROC vs sampled SE, in- and out-of-distribution | `experiments/run_phase1.py` | probe AUROC ≥ 90% of sampled-SE AUROC at ≤1.05× inference cost |
| **H1** Core: routing precision into decoding reduces confident errors | Compare confident-error rate: no gate vs gate vs gate+retrieval, ablations | `noema/gate.py` + Phase-2 harness | monotone drop in confident-error at fixed answer rate |
| **H3** Canalization: RLHF degrades calibration; annealing prevents it | Track ECE + output diversity across preference-tuning with/without entropy schedule | `noema/annealing.py` | annealed run preserves ECE & distinct-n at equal reward |
| **H4** Worldlessness limit: gating converts error→abstention, not →truth | Evidence-free split: accuracy-on-answered vs abstention rate | `gate_outcomes()` | errors abstained, accuracy-if-forced unchanged |

## Quick start (no GPU needed — synthetic validation)

```bash
python experiments/mock_demo.py     # end-to-end pipeline on a simulated LM
python tests/test_pipeline.py       # unit tests
```

## Real Phase-1 run (GPU box)

```bash
pip install torch transformers datasets accelerate
python experiments/run_phase1.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset triviaqa --n 2000 --k 10 --layer -6
```

Notes: `--layer -6` follows the SEP finding that middle-late layers encode
semantic uncertainty best — sweep `--layer` as a first ablation. Use
`--dataset simpleqa` for the OOD generalization check (train probe on
TriviaQA arrays, evaluate on SimpleQA arrays; both are cached in `--out`).

## Layout

```
noema/semantic_entropy.py   sampling → entailment clustering → SE
noema/precision_head.py     numpy probe + torch head, distillation +
                            asymmetric calibration loss (Kalai-style)
noema/gate.py               4-way decoding policy, threshold fitting,
                            selective-prediction analysis
noema/annealing.py          plasticity/consolidation schedule for tuning (H3)
noema/metrics.py            AUROC, ECE, risk-coverage/AURC (numpy only)
experiments/mock_demo.py    synthetic end-to-end validation (runs anywhere)
experiments/run_phase1.py   real H2 experiment on an open-weight model
tests/test_pipeline.py      unit tests
```

## Roadmap after Phase 1

1. **Phase 2 (H1)** — precision-gated attention: register forward hooks on
   attention modules of an open-weight model; add a pre-softmax bias toward
   context keys scaled by `g(pi_prior, pi_evidence)` (COMPASS-style but driven
   by the learned head). Evaluate on knowledge-conflict suites + FActScore.
2. **Phase 3 (H3)** — implement `annealing.AnnealingSchedule` inside a DPO/PPO
   loop; log ECE + semantic diversity per phase.
3. **Phase 4 (H4)** — evidence-free and poisoned-retrieval splits.
```
