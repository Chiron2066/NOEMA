# NOEMA — Paper Notes: Discussion Material & Reviewer Pressure-Test

Working notes for the paper's Discussion / Limitations sections.
Status: awaiting NQ-Open OOD results to close Phase 1.

---

## 1. Results ledger (fill in as runs complete)

| Experiment | Dataset | n | Key result | Status |
|---|---|---|---|---|
| Smoke test | TriviaQA | 100 | SE 0.811 / probe 0.745 | done |
| Main in-domain | TriviaQA | 2000 | SE 0.820 [.794–.847] / ridge head **0.819** [.793–.844], ratio 0.998, ECE **0.031** (Platt) | done |
| Gate (in-domain) | TriviaQA | 2000 | @5% target risk: coverage 5.5%, risk 3.6%, 99.7% errors→abstention (base error 59.1%) | done |
| OOD attempt 1 | SimpleQA | 500 | **Inconclusive**: model accuracy 3% (15 positives) → CIs too wide (SE .714 [.55–.86], frozen head .587 [.43–.76]). Behavioral note: frozen head correctly abstained ~everywhere; transferred ECE 0.016 | done |
| OOD attempt 2 | NQ-Open | 500 | **TRANSFER CONFIRMED.** Accuracy 22.4% (112 positives). In-domain SE 0.762 [.710–.813]; **frozen** TriviaQA head 0.762 [.713–.807] — transfer ratio 0.999, H2 OOD PASS. Transferred ECE 0.189 (discrimination transfers perfectly; calibration needs light per-domain refit — standard domain-shift result). SE correct 1.43 vs wrong 2.01. | done |
| Phase 2 γ-sweep | SQuAD+2 distr. | 50×3 | Dose-response inverted-U (same items): γ=1 low-π̂ 0.824→0.882; **γ=2 0.824→0.941 (optimum)**; γ=4 0.824→0.824 with mid-π̂ collapse 0.81→0.69. High-π̂ = 1.0 untouched in ALL runs (no competence cost). Earlier γ=4 no-distractor run: overall 0.92→0.84 (overdose confirmed twice). | done |
| Phase 2 main | SQuAD+2 distr. | 500 | **Routing effect ironclad**: conflict evidence-following 0.302→0.364, McNemar 42 vs 11 flips, p<0.0001. **Accuracy effect directional, underpowered**: overall 0.820→0.836 (p=.24); lowest-π̂ quartile 0.776→0.824 (8 vs 2 flips, p=.11) — gain concentrated where predicted, but base ceiling (0.82) leaves few fixable failures. No fluency damage (0 empty outputs). | done |
| Phase 2 enriched | SQuAD+3 distr., π̂<0.3 only | 685 scanned / 218 kept (stopped early, trend clear) | **Accuracy null confirmed**: base 0.821 = gated 0.821 (6 vs 6 flips). Key diagnostic: on blind-spot items the base model sticks to wrong parametric memory only ~10% under conflict — evidence-use is NOT the bottleneck at 1.5B on short contexts. Nothing for the gate to fix. | done |
| Phase 2b lost-in-middle | SQuAD, 8 passages, gold @ first/mid/last | 42 (stopped early) | **Base is FLAT across positions (0.810/0.810/0.810)** — no lost-in-the-middle effect at 1.5B / ~3k tokens; gating mildly perturbs (middle 0.81→0.74, 0 vs 3 flips). Second confirmation: no headroom ⇒ no gain. | done |
| Phase 2c adaptive retrieval | TriviaQA rc.wikipedia | 800 | **POSITIVE.** Closed-book 0.490, always-retrieve 0.666, oracle 0.744. π̂-routed retrieval: at a 50% retrieval budget, 0.649 vs 0.578 for random routing at the same budget (+7.1pts); 97.5% of always-retrieve accuracy for half the retrieval cost. Curve is strongly concave — first retrievals go exactly where needed (30% budget already yields 0.601). Bonus transfer check: frozen head AUROC 0.802 on this fresh question set. | done |
| Phase 2d two-knob gating | SQuAD+3 distr., 3 arms | 400 | Theory-consistent ordering — base 0.808 < block 0.810 < evidence-gated 0.823 — but sub-significant (evid vs base +20/−14, p=.39). Decisive diagnostic: base extracted from a WRONG passage 0.0% of the time (scorer top-1 0.98). The failure mode evidence-precision fixes does not exist at 1.5B/4 passages. Two-knob machinery validated as harmless & directionally right; payoff regime is elsewhere (scale, messier retrieval). | done |
| Phase 3 pilot (600 steps, LoRA r8, 3 cycles) | UltraFeedback DPO + TriviaQA probe | 2 arms | **Disease confirmed, cure not (at this dose).** Both arms: ECE 0.333→~0.39 (calibration decays within 600 LoRA steps of preference-tuning — canalization is fast and cheap to induce). Arms indistinguishable on ECE (0.388 std vs 0.390 ann); annealed slightly better reward (0.63 vs 0.62). Diversity drifts down, no collapse yet. H3 unresolved at pilot dose → strong protocol (2400 steps, 6 cycles, r16 all-proj, β 0.1×–2×) queued. | done |
| Phase 3 strong (2400 steps, r16 all-proj, 6 cycles, β 0.1×–2×) | UltraFeedback DPO + TriviaQA probe | 2 arms | **H3 not supported as operationalized.** Standard: reward plateaus 0.62–0.65 by step 300; ECE 0.333→~0.42 band and stays (chronic decay at zero further reward — the disease, robust). Annealed: acute harm at every plasticity eval (ECE spikes 0.49–0.63, diversity crashes), partial repair each consolidation; ends ECE 0.523 / reward 0.56 — worse than standard on both. Cure dose-response is itself inverted-U: pilot dose inert, strong dose toxic. Glimmer: deep-consolidation checkpoints (steps 1800–2000) hit the run's best state — semdiv 0.83 vs standard's 0.61 at comparable ECE (0.42) — suggesting the untested refinement: end on long consolidation, evaluate after integration, not mid-plasticity. | done |
| Layer sweep | TriviaQA | 2000 | Layer 0 = 0.500 (chance); rises through depth; jump at L16 (0.809); plateau L21–25; **peak L22 = 0.822 [.797–.848]**, matching the SE ceiling (0.820). Phase 1's layer −6 (=L23, 0.819) was near-optimal. Last layers dip slightly (0.804–0.812). | done |

Setup: Qwen2.5-1.5B-Instruct, layer −6 last-prompt-token hidden state,
K=10 samples at T=1.0, DeBERTa-large-MNLI bidirectional entailment,
ridge regression on exp(−SE), λ=1e4 by 5-fold CV inside train split,
Platt calibration on train correctness. Head weights: `phase1_triviaqa_n2000/ridge_head.npz`.

## 2. Findings worth a paragraph each

- **Amortization saturates, not just passes.** One forward pass matches ten
  (ratio 0.998). Frame vs SEPs (Kossen et al. 2024): same conclusion, but via
  soft-target ridge + Platt rather than logistic probing; note the recipe
  sensitivity below.
- **Recipe sensitivity as a finding.** Hard median-split labels + weak L2
  (1e-3) gave 0.722–0.733 and "H2 fail"; soft exp(−SE) targets + λ=1e4 gave
  0.819. The signal was fully present; the readout decided the verdict.
  Methodological warning for the field: probe-based negative results may be
  regularization artifacts. (Also a pleasing reflexive note: an
  under-regularized probe overfitting its history is canalization in
  miniature; the cure was loosening the prior.)
- **The competence-zone requirement.** SimpleQA (3% accuracy) shows knowledge
  detection is only measurable where knowledge exists. Deeper: the model's few
  correct SimpleQA answers had HIGH SE (1.77) — correct-by-luck, not known.
  Detection targets *knowing*, not correctness. OOD benchmarks must sit in the
  model's competence zone (20–50% accuracy).
- **Honest transfer of caution.** The TriviaQA-trained head, frozen, assigned
  low confidence to essentially all of SimpleQA (transferred ECE 0.016; gate
  abstained ~everywhere at 97% base error). Discrimination unmeasurable, but
  calibrated humility transferred — the desirable failure mode.
- **Asymmetric loss miscalibration.** α=4 pushed the torch head's outputs to
  extremes (ECE 0.41). α/β is an operating point to be tuned on the
  risk–coverage curve, not a constant.

- **The headline finding: domain-general knowing-signal.** The head trained
  only on TriviaQA, applied frozen (no retraining, no recalibration) to
  Natural Questions, matches the in-domain sampled-SE ceiling exactly
  (0.762 vs 0.762, ratio 0.999). It did not learn TriviaQA surface features;
  it learned to read a domain-general "epistemic status" direction in the
  model's hidden state. This is the paper's central claim: LLMs carry an
  intrinsic, linearly-readable, domain-general representation of their own
  knowledge reliability — a machine analog of precision-weighting — and it
  can be extracted for the cost of a dot product.
- **Knowing is computed, not lexical — and it lives late-mid, not last.**
  Layer sweep (all 29 hidden states, same 2000 items): the embedding layer is
  exactly chance (0.500) — the signal cannot be read off the words of the
  question; it is *constructed* by the network. AUROC climbs monotonically
  through the stack, jumps at L16, and plateaus at L21–25, peaking at L22
  (0.822) — reaching the sampled-SE ceiling (0.820) several layers *before*
  the output layers, which slightly degrade it (busy with token prediction).
  Interpretation: the model completes its self-assessment of knowledge
  reliability ~3/4 of the way through its depth, before it begins committing
  to an answer — exactly where a precision signal must be available if it is
  to gate downstream processing (Phase 2 hooks should read from, and act
  after, this plateau). Also validates Phase 1's layer −6 heuristic (L23,
  0.819 ≈ optimal).
- **Discrimination transfers; calibration doesn't (fully).** Frozen Platt
  parameters gave ECE 0.189 on NQ (vs 0.031 in-domain) because base accuracy
  shifted 41%→22%. Ranking is domain-general; probability mapping is
  domain-specific. Deployment recipe: freeze the head, recalibrate the last
  two scalars on ~100 labeled examples per domain.

- **Precision as action policy — where the accuracy gain actually lives
  (2c).** π̂ deciding WHETHER to retrieve (rather than HOW to attend) converts
  directly into accuracy-per-cost: +7.1pts over random routing at a fixed 50%
  retrieval budget; 97.5% of always-retrieve accuracy at half the cost. The
  brain analogy sharpens: precision's primary role in biology is not
  modulating perception mid-stream but selecting actions (look again, feel
  with feet, ask). In LLMs likewise: the signal's cash value is in the
  policy layer (retrieve/abstain/hedge), not in perturbing an attention
  process that already works. Together with the Phase-2 attention nulls,
  this is one coherent finding, not a win and a loss.

- **Phase 3 synthesis — the disease is cheap, the cure is not (yet).**
  (i) Preference-tuning degrades calibration fast, chronically, and past the
  point of any reward gain: all learning by step 300, ECE keeps worsening to
  step 2400. Canalization in LLM training is not hypothetical; it is the
  default. (ii) The REBUS-inspired annealing cure failed in both tested doses
  — inert when gentle, toxic when strong (acute ECE/diversity damage during
  plasticity, incomplete repair). (iii) The honest psychedelic parallel now
  cuts the other way too: dosing and integration timing dominate outcomes,
  and measuring mid-perturbation measures the perturbation. The
  deep-consolidation glimmer (diversity 0.83 vs 0.61 at matched ECE) points
  to the specific next protocol: front-load plasticity, end on extended
  consolidation, evaluate only post-integration. State as tested-and-open,
  not as failure of the framing.

- **Phase 2 synthesis — the intervention-headroom law.** Across five runs a
  single pattern: precision-gated attention helps exactly in proportion to how
  broken prior/evidence arbitration is, and at 1.5B on ≤3k-token contexts it
  is barely broken. (i) Routing effect ironclad (p<0.0001): the gate DOES
  re-route processing toward evidence, proportional to 1−π̂. (ii) Dose-response
  inverted-U (γ=1 helps, γ=2 optimal, γ=4 harms) — Yerkes–Dodson for
  neuromodulation, empirically. (iii) Zero competence cost: high-π̂ accuracy
  untouched in every run. (iv) Accuracy conversion requires headroom that this
  regime lacks: the base model follows evidence ~90% even on blind-spot items,
  and shows NO lost-in-the-middle sag at 8 passages. Scale-conditional
  prediction for future work: gains should appear precisely where parametric
  override grows (larger models) and where attention demonstrably fails
  (10k+ contexts). This is H4's shape at the attention level: the gate
  reallocates trust; it cannot manufacture either knowledge or failure to fix.

## 3. Reviewer pressure-test (external evaluation) + our responses

### Objection 1 — Credit assignment in SE distillation
*Sequence-level SE → token-level π̂ targets is ill-posed; one bad token can
corrupt a whole chain-of-thought.*

**Response.** Conceded for long-form; Phase 1 deliberately avoided it
(short-form QA, one pooled state, one target — hence the clean result; the
validated claim is question-level). Long-form plan: decompose generations into
atomic claims (FActScore-style), compute SE per claim, and supervise π̂ only at
semantic commit points (entity/assertion-binding tokens), not uniformly.
Neuroscience warrant: cortical precision operates over population events at
~100ms–1s (semantic-unit granularity), not per-spike; match supervision
granularity to semantic granularity. Flag as its own workstream.

### Objection 2 — Attention logit instability
*Dynamic per-token biases into pre-softmax logits may destabilize
autoregression if π̂ fluctuates across adjacent tokens.*

**Response.** Valid; biology suggests the fix: neuromodulators are SLOW.
ACh/5-HT adjust gain over hundreds of ms to seconds — the brain solved this
instability with low-pass gain control. Mitigations: (a) EMA-smooth π̂ across
tokens; (b) bound bias magnitude (COMPASS-style PID clamping preserves
fluency); (c) bias only the minority of context-copying heads; (d) fallback =
pure neuromodulation variant (π̂ → global temperature/layer gains), which
cannot break syntax by construction.

### Objection 3 — The timidity trap (over-abstention)
*α ≫ β may push the model to hedge/abstain on moderately hard tasks, hurting
utility and win-rates.*

**Response.** Valid — and empirically we hit its mirror image first (α=4 →
overconfident extremes, ECE 0.41). Both failures teach: α/β is a tunable
operating point on the risk–coverage curve, selected against a utility target
(`fit_generate_threshold`). Guards: hedging as intermediate output; per-intent
gating (factual queries gated, creative tasks explicitly ungated —
precision-gated mode switching turns the objection into a feature); report
coverage-at-fixed-risk alongside win-rate. Note Kalai et al.: benchmarks that
score confident-wrong above honest-hedge measure the disease.

### Reviewer's conditional verdict — resolved
*"If Phase 1 succeeds in proving a lightweight precision head can distill
semantic entropy at near-zero FLOP cost…"* — resolved affirmatively:
0.819 vs 0.820, ECE 0.031, one forward pass (TriviaQA n=2000, in-domain).
Open: OOD transfer (NQ-Open, pending) and Phase 2 integration.

## 4. Limitations to state plainly in the paper

1. Single model (Qwen2.5-1.5B), single layer (−6), short-form QA only.
2. In-domain result strong; OOD transfer not yet established (SimpleQA
   inconclusive by design mismatch; NQ pending).
3. SE ground truth depends on the NLI clusterer; entailment errors propagate.
4. Gate coverage inherently bounded by the base model's knowledge (a 1.5B
   model knows ~41% of TriviaQA; the gate cannot manufacture knowledge — H4).
5. Entropy-as-medicine (annealing curriculum, Phase 3) is untested; only
   entropy-as-probe is validated.
