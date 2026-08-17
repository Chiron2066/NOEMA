# The Knowing Signal: Precision-Weighting as an Architectural Principle for Hallucination Control in Language Models

**Chiron Caravella**
*Independent researcher*

---

## Abstract

Large language models confabulate: they assert fabricated content with the same fluent confidence as knowledge. Drawing on predictive-processing accounts of the brain — in which every representation carries a *precision* weight encoding its own reliability — this paper asks whether language models contain a latent analog of this mechanism, and whether it can be read, acted upon, and preserved. Four sets of findings are reported on Qwen2.5-1.5B-Instruct. **(1) Detection.** A ridge probe on a single hidden state matches 10-sample semantic entropy at detecting confabulation (AUROC 0.819 vs 0.820; 0.998 ratio), calibrates to ECE 0.031, transfers *frozen* across datasets with no loss of discrimination (0.762 vs 0.762 on NQ-Open; 0.802 on a third question set), reads at chance from the embedding layer (0.500), and peaks at three-quarters network depth (layer 22/28, 0.822) — before generation begins. Models compute a domain-general, linearly readable estimate of their own knowledge reliability; it costs one dot product to extract. **(2) Action.** Routed into policy, the signal converts 99.7% of would-be confident errors into abstentions at a 5% risk target, and governs *when to retrieve*: at a fixed 50% retrieval budget, precision-routed retrieval outperforms random routing by 7.1 points and attains 97.5% of always-retrieve accuracy at half the cost. **(3) Attention.** Routed into attention as a pre-softmax bias toward evidence, the signal reliably re-routes processing (evidence-adherence 0.302→0.364 under conflict, McNemar p<0.0001) with an inverted-U dose-response and zero cost to known items — but yields no significant accuracy gain across five experimental settings, because a 1.5B model on ≤3k-token contexts almost never mis-arbitrates between memory and evidence (0/400 wrong-passage extractions). Intervention benefit tracks arbitration failure, which grows with scale; this is stated as a falsifiable prediction. **(4) Training.** Preference-tuning (DPO/LoRA) degrades calibration rapidly, chronically, and past the point of reward gain (ECE 0.333→0.43 while reward plateaus by step 300) — canalization as the default outcome of alignment training. An annealing curriculum inspired by the REBUS model of psychedelic therapy failed to rescue calibration at both tested doses (inert when gentle; acutely harmful when strong), with a post-consolidation diversity benefit (0.83 vs 0.61 at matched ECE) pointing to integration-timed protocols as the specific open question. Code and per-item artifacts are released.

---

## 1. Introduction

The dominant failure mode of large language models is not ignorance but *unflagged* ignorance. A model asked a question beyond its knowledge does not report uncertainty; it produces a fluent, well-formed, wrong answer. This failure — confabulation, commonly "hallucination" — is the principal barrier to deployment in any setting where errors carry cost.

Neuroscience offers a specific diagnosis. In predictive-processing accounts (Friston, 2010; Clark, 2016), a brain is a generative model whose every prediction carries a second quantity: *precision*, an estimate of that prediction's reliability, which determines how much weight it receives against incoming evidence. Precision-weighting is what makes uncertainty *felt*: degraded input arrives pre-marked as untrustworthy, and behavior — hesitation, checking, asking — follows from the marking. Psychiatric conditions have been productively modeled as precision pathologies: canalization — priors grown too rigid to revise (Carhart-Harris et al., 2023) — and the REBUS model (Carhart-Harris & Friston, 2019) interprets psychedelic therapy as a controlled, temporary relaxation of over-weighted priors that permits their revision.

Language models, this paper argues, possess knowledge without possessing calibrated *access* to its reliability — and standard training actively erodes what access exists. The precision-weighting framework is operationalized here as an architecture (NOEMA) with four testable hypotheses:

- **H2 (amortization):** sampled semantic entropy (Farquhar et al., 2024) — the current gold-standard confabulation detector, requiring K forward passes — can be distilled into an intrinsic *precision head* read from hidden states in one pass.
- **H1 (routing):** routing the precision estimate into inference (attention biases toward evidence; action selection among generate/retrieve/hedge/abstain) reduces confident errors.
- **H3 (annealing):** alternating plasticity/consolidation phases during preference-tuning prevents the calibration decay that standard training causes.
- **H4 (worldlessness limit):** for questions where the model has neither knowledge nor evidence, no internal mechanism can substitute for world-contact; gating converts errors to abstentions, not to truth.

Complete results — positive, null, and negative — are reported for all four.

## 2. Related work

**Sampling-based uncertainty.** Semantic entropy (Kuhn et al., 2023; Farquhar et al., 2024) samples K answers, clusters them by bidirectional entailment, and scores the entropy over meaning-clusters; high entropy predicts confabulation. It is effective and expensive (K× inference cost).

**Probing internal states.** Semantic entropy probes (Kossen et al., 2024) and truthfulness probes (Azaria & Mitchell, 2023; Marks & Tegmark, 2023) show hidden states carry uncertainty-relevant information. The contributions here beyond this line: the frozen cross-dataset transfer result, the layer-depth geography, the calibration/discrimination decomposition under shift, and a demonstration that probe *recipes* can silently decide scientific verdicts (§5.1).

**Retrieval and its failures.** Retrieval-augmented generation presumes the model uses evidence; documented failures include positional neglect ("lost in the middle," Liu et al., 2023) and parametric override under knowledge conflict, which *increases* with model scale (Longpre et al., 2021). Adaptive-retrieval methods (e.g., FLARE, Self-RAG) gate retrieval on model signals; the present results show a frozen linear head suffices.

**Training-induced miscalibration.** RLHF-style tuning is known to degrade calibration (OpenAI, 2023 GPT-4 report; Kalai et al., 2025 argue benchmarks that reward confident guessing measure the disease). The contribution here is a controlled small-scale demonstration and a first explicit test of an entropy-cycling countermeasure.

## 3. The NOEMA architecture

NOEMA (from *noēma*: the object of thought) attaches three components to a frozen LM:

1. **Intrinsic precision head π̂.** A linear readout on a mid-network hidden state (last prompt token), trained to predict exp(−SE) where SE is sampled semantic entropy — i.e., trained to *amortize* the expensive detector. Recipe (found necessary, §5.1): standardize features; ridge regression with λ selected by 5-fold CV inside the training split; Platt-calibrate on training correctness.
2. **Policy gate.** π̂ thresholds select among actions: generate (high π̂), retrieve-then-answer (low π̂, evidence available), hedge (moderate π̂, no evidence), abstain (low π̂, no evidence). Thresholds are operating points on the risk–coverage curve.
3. **Gated attention (two knobs).** With retrieved context present, a bounded pre-softmax bias b = γ·(1−π̂) is added to attention logits at evidence-token positions in post-readout layers (16–27). *Prior* precision (π̂) sets the strength; *evidence* precision (per-passage relevance) sets the target — biasing the most question-relevant passage rather than the whole block.

The annealing curriculum (H3) operates at training time: cycles alternating plasticity (reduced preference pressure β, raised learning rate, weight noise) with consolidation (raised β, lowered learning rate), against a standard constant-knob control.

## 4. Methods

**Models.** Generator: Qwen2.5-1.5B-Instruct (28 layers, d=1536); readout at layer −6 (=22/23) chosen a priori and validated by sweep. NLI clusterer: DeBERTa-large-MNLI, bidirectional entailment with the question as context.

**Data.** TriviaQA rc.nocontext validation (n=2000; K=10 samples at T=1.0, top-p 0.95); NQ-Open validation (n=500, multi-alias gold matching); SimpleQA (n=500); SQuAD v1.1 validation with distractor passages and counterfactual (answer-swapped) conflict variants; TriviaQA rc.wikipedia (n=800) for adaptive retrieval; UltraFeedback-binarized for preference-tuning.

**Metrics.** AUROC (tie-corrected Mann–Whitney) for confabulation detection; ECE (10 equal-width bins); risk–coverage/AURC; McNemar exact tests on paired arms; bootstrap 95% CIs. Correctness: normalized match against gold aliases, NLI fallback for Phase-1 sets.

**Statistical hygiene.** Test halves touched once; hyperparameters (λ, γ) selected on training splits or separate sweep runs; all runs, including failed recipes and stopped experiments, reported.

## 5. Results

### 5.1 One forward pass matches ten (H2, in-domain)

On TriviaQA (n=2000; model accuracy 40.9%), sampled semantic entropy detects confabulation at AUROC 0.820 [0.794–0.847]. The ridge head, reading one hidden state, achieves **0.819** [0.793–0.844] — ratio 0.998 — with ECE **0.031** after Platt calibration. Amortization does not merely pass a threshold; it saturates.

**The recipe is the result's gatekeeper.** The first readout attempted — hard median-split labels, weak L2 (10⁻³), and an aggressive asymmetric penalty (α=4) — yielded 0.722–0.733 and ECE 0.41: an apparent *failure* of H2. The signal was fully present; the readout was under-regularized (train 0.806 vs test 0.733) and miscalibrated by design. Soft targets exp(−SE) with λ=10⁴ (CV-selected) recovered 0.819. This stands as a methodological warning: probe-based negative results in the interpretability literature may be regularization artifacts.

### 5.2 The signal is domain-general (frozen transfer)

The head trained on TriviaQA, applied **frozen** — no retraining, no recalibration — to NQ-Open (accuracy 22.4%): AUROC **0.762** [0.713–0.807], exactly matching in-domain sampled SE on that set (0.762 [0.710–0.813]; ratio 0.999). On a third, independently drawn TriviaQA-with-evidence set: 0.802. The head did not learn dataset surface features; it reads a domain-general epistemic-status direction in activation space.

Discrimination transfers; calibration does not fully: frozen Platt parameters give ECE 0.189 under the 41%→22% base-rate shift. Deployment recipe: freeze the head, refit two scalars on ~100 labeled per-domain examples.

Two boundary lessons. On SimpleQA (model accuracy 3%) detection is unmeasurable — 15 positives yield uninformative CIs — and the model's few correct answers had *high* SE (1.77): correct-by-luck, not known. Knowledge detection requires a competence zone (roughly 20–50% accuracy). Notably, the frozen head assigned low confidence to essentially all of SimpleQA (transferred ECE 0.016; the gate abstained nearly everywhere at 97% base error): calibrated humility transferred even where discrimination could not be measured.

### 5.3 Where knowing lives (layer geography)

Reading the same 2000 items at every layer: the embedding layer is exactly chance (**0.500**) — self-knowledge cannot be predicted from the question's words; the network computes it. AUROC climbs through the stack, jumps at layer 16 (0.809), plateaus over layers 21–25, peaks at layer 22 (**0.822** [0.797–0.848], reaching the sampled-SE ceiling), and dips slightly in the final layers. The model completes its self-assessment ~¾ through its depth, before committing to words — precisely where a precision signal must exist to gate downstream processing. The a-priori layer −6 choice (0.819) was one layer from optimal.

### 5.4 Precision selecting actions (the payoff)

**Abstention.** At a 5% target risk on answered questions, the gate answers the most-confident 5.5% at 3.6% risk, converting **99.7%** of would-be errors into abstentions (base error 59.1%). Across thresholds: the raw model asserts errors at 59%; with the gate at τ=0.55 it answers 31% of questions at 75% accuracy — the asserted-error rate falls to 25% with calibrated confidence attached (ECE 0.031).

**Adaptive retrieval.** On TriviaQA with real Wikipedia evidence (closed-book 49.0%, always-retrieve 66.6%, per-item oracle 74.4%), the policy *retrieve iff π̂<τ* dominates budget-matched random routing everywhere: at a 50% budget, **64.9% vs 57.8%** (+7.1 points); at 30%, 60.1% vs 54.2%. π̂-routing attains 97.5% of always-retrieve accuracy at half its cost; the accuracy-budget curve is strongly concave — the first retrievals purchased are exactly the needed ones.

### 5.5 Precision gating attention: mechanism confirmed, payoff regime absent (H1 at small scale)

On SQuAD with distractor passages (n=500, γ=2 from a three-point sweep), gating measurably re-routes processing: under counterfactual evidence, adherence to context rises 0.302→0.364 (McNemar 42 vs 11 flips, **p<0.0001**), scaled by (1−π̂). Dose-response is inverted-U — γ=1 helps (+5.9 points on low-π̂ items), γ=2 is optimal (+11.8 on the sweep), γ=4 harms (mid-π̂ accuracy 0.81→0.69) — and high-π̂ items were never damaged in any run.

But routing does not convert to significant accuracy gains: overall 0.820→0.836 (p=0.24); on blind-spot items (π̂<0.3, enriched n=218) exactly null (0.821 both, 6 vs 6 flips). The diagnosis is headroom, not mechanism: the base model sticks to wrong parametric memory only ~10% under conflict; shows *no* positional sag with the gold passage buried mid-context (0.810 at first/middle/last, n=42); and in 400 items with the two-knob gate (per-passage relevance scoring, top-1 accuracy 0.98) *never once* extracted its answer from a wrong passage. Arms ordered as theory predicts (base 0.808 < block-gated 0.810 < evidence-gated 0.823) but sub-significant (p=0.39).

**The intervention-headroom law.** Across five settings: precision-gated attention helps in proportion to how broken prior/evidence arbitration is — and at 1.5B on ≤3k-token contexts it is barely broken. Since parametric override grows with scale (Longpre et al., 2021) and positional neglect with context length (Liu et al., 2023), the falsifiable prediction follows that the same intervention yields significant gains on larger models and longer contexts. What the small model establishes is a boundary, not a defeat: this is H4 operating at the attention level. The gate reallocates trust; it cannot manufacture a failure to fix.

### 5.6 Training erodes the signal; naive annealing does not save it (H3)

**The disease.** DPO on UltraFeedback (LoRA r16 all-projections, 2400 steps): held-out preference accuracy plateaus at 0.62–0.65 by step 300, yet ECE degrades from 0.333 to a chronic ~0.42 band through step 2400 — the model continues becoming *more wrongly confident* long after it has stopped becoming better. A 600-step pilot reproduces the pattern (ECE→0.39). Canalization under preference-tuning is fast, cheap, and chronic.

**The cure, as operationalized, fails.** The annealed arm (6 cycles; plasticity: β×0.1, lr×5, weight noise; consolidation: β×2, lr×0.5) shows acute damage at every plasticity-phase evaluation (ECE spikes 0.49–0.63; diversity crashes to 0.31–0.52) with only partial repair during consolidation, ending *worse* than standard on both endpoints (ECE 0.523 vs 0.426; reward 0.56 vs 0.62). The gentle pilot dose (3 cycles, β 0.3×–1.5×) was inert (final ECE 0.390 vs 0.388). The cure exhibits its own inverted-U: sub-threshold or toxic, with the therapeutic window — if it exists — untested between.

**The glimmer.** The annealed run's best state occurred at deep-consolidation checkpoints (steps 1800–2000): ECE falling to 0.421 with semantic diversity **0.83 versus the standard arm's 0.61** at comparable calibration — the only condition in either arm combining preserved variety with recovering calibration. The run then ended mid-plasticity, wrecking it. The specific protocol this licenses for future work: front-loaded plasticity, terminal extended consolidation, evaluation only post-integration — the direct analog of not assessing therapy outcomes mid-session.

## 6. Discussion

**What the precision signal is.** Chance-level at the embeddings, saturating at ¾ depth, linear, domain-general, and cheap: the knowing-signal behaves like an architectural feature, not a dataset artifact. Its existence reframes hallucination: the model *has* the information needed to flag its own confabulations; the deployment stack simply never reads it.

**Where its value lies.** The positive and null results agree on one lesson: precision's cash value in LLMs — as in biology — is at the level of *action selection*, not mid-stream perceptual editing. Biological precision-weighting principally selects policies (look again, feel with the feet, ask); the gate's wins here (abstention, adaptive retrieval) are policy wins, and its nulls arose where the perceptual analog (attention over evidence) was already functioning.

**Anticipated objections, addressed with data.** *Credit assignment* (sequence-level SE to token-level targets is ill-posed): conceded; Phase 1 validates question-level supervision only, and claim-level supervision at semantic commit points is the stated path for long-form. *Attention-logit instability from fluctuating π̂*: the per-question π̂ used here is constant across generated tokens; the bounded, subset-of-layers bias produced zero degenerate outputs across all runs, and the biological answer (neuromodulators are slow) prescribes EMA smoothing for token-level extensions. *The timidity trap* (over-abstention destroying utility): the operating point is explicitly tunable on the risk–coverage curve; at every tested threshold, high-π̂ competence was untouched — and the trap's mirror (overconfidence from an aggressive asymmetric loss) was hit empirically first, confirming both walls of the corridor are real.

**Worldlessness (H4).** Nothing in these results manufactures knowledge. The gate's accuracy-on-answered rises because errors are abstained; retrieval helps only when evidence exists; a 1.5B model knows 41% of TriviaQA and no internal signal changes that. A system without world-contact can at best be *honest about its blindness* — the philosophical claim this project operationalizes, and, it is argued, the correct near-term target for hallucination work: not zero error, but zero unflagged error.

## 7. Limitations

Single model family and scale for all experiments (1.5B); replication at larger scale — where the headroom analysis predicts the attention-gating results should reverse — is the most important item of future work. Short-form QA only; long-form claim-level supervision is designed but untested. SE ground truth inherits NLI clusterer errors. LoRA-DPO is a small-scale proxy for full RLHF, and diversity collapse may require full-parameter training to manifest. The annealing search covered two doses of one schedule family. Adaptive-retrieval evidence was oracle-provided (dataset-supplied passages), not a live retriever.

## 8. Conclusion

A language model already computes a private, linear, domain-general estimate of whether it knows — finished three-quarters of the way through its forward pass, extractable for the cost of a dot product, and robust enough to govern abstention and retrieval policy unchanged across datasets. Standard preference-tuning steadily corrodes the calibration of that signal while buying nothing after its earliest steps, and the first transplant of psychedelic therapy's plasticity-cycling into training did not save it — inert at one dose, harmful at another, with one integration-timed condition pointing the way. Detection: proven. Policy: proven. Attention-routing: mechanism proven, payoff scale-conditional. Therapy: open, with the protocol specified. The brain's oldest trick for living with an unreliable world — knowing how much to trust itself — is latent in these machines; the work is to read it, act on it, and stop training it away.

## Data and code availability

All code (probe, gate, gated attention, annealing curriculum, experiment scripts), per-item JSONL artifacts, trained head weights, and the complete results ledger including failed recipes are available at [repository URL].

## References

*(to be completed at formatting time)*

- Azaria, A. & Mitchell, T. (2023). The internal state of an LLM knows when it's lying. *EMNLP Findings*.
- Carhart-Harris, R. & Friston, K. (2019). REBUS and the anarchic brain. *Pharmacological Reviews*, 71(3).
- Carhart-Harris, R. et al. (2023). Canalization and plasticity in psychopathology. *Neuropharmacology*, 226.
- Clark, A. (2016). *Surfing Uncertainty*. Oxford University Press.
- Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting hallucinations in large language models using semantic entropy. *Nature*, 630.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11.
- Kalai, A. et al. (2025). Why language models hallucinate. *(as cited)*.
- Kossen, J. et al. (2024). Semantic entropy probes. *arXiv*.
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic uncertainty. *ICLR*.
- Liu, N. et al. (2023). Lost in the middle: How language models use long contexts. *TACL*.
- Longpre, S. et al. (2021). Entity-based knowledge conflicts in question answering. *EMNLP*.
- Marks, S. & Tegmark, M. (2023). The geometry of truth. *arXiv*.
