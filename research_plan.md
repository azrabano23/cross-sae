# Reliable Cross-Domain SAE Feature Matching: Vision Models ↔ Human Visual Cortex

**A significance-tested, stability-controlled method for asking which interpretable
features are *shared* between an artificial vision model and the human brain.**

Azra Bano · working research plan · v0.1

---

## 1. One-paragraph statement

Sparse autoencoders (SAEs) extract interpretable, monosemantic features from
neural networks, and a 2025 result ([SAE-BrainMap](https://arxiv.org/abs/2506.11123))
showed that model-side SAE features align with human visual cortex. But that
alignment is asserted by raw cosine similarity with **no statistical control and
no handling of SAE non-determinism**, and it never trains SAEs on the brain side.
This project builds the missing rigor: train SAEs on **both** a vision transformer
and human brain responses to the **same** images (THINGS shared-stimulus design),
match features across domains with **false-discovery-rate (FDR) control** via
Model-X knockoffs, and gate every claim through a **seed-stability filter** so a
"shared feature" is reproducible, not a lucky initialization. The deliverable is a
method (and open-source tool) that says, with a calibrated error rate, *which*
interpretable concepts a model and a brain genuinely share.

## 2. Why this matters for AI safety / interpretability

The central open problem in mechanistic interpretability right now is not "can we
find features" but **"are the features we find real, reproducible, and not
artifacts of our method?"** (cf. the 2025 wave on SAE non-canonicity and
reliability). This project is a direct attack on that problem: it imports formal
error-rate control (FDR) and reproducibility testing into SAE feature analysis,
and uses the brain as an *external validity testbed* — biological visual cortex is
an independent representation that did not come from our training pipeline, so
agreement that survives FDR + stability filtering is strong evidence a feature is
a real unit of analysis, not a dictionary artifact. The methods transfer directly
back to LLM interpretability (the same knockoff + stability machinery applies to
matching SAE features across two LLMs, or across model checkpoints).

## 3. Prior work and the exact gap (researched + adversarially verified)

| Work | What it does | What it does **not** do |
|---|---|---|
| **SAE-BrainMap** ([2506.11123](https://arxiv.org/abs/2506.11123), Jun 2025) | Model-side SAE features → voxel fMRI, cosine sim ≤ 0.76, ROI selectivity, layer→ventral-stream map | No **brain-side** SAE (raw voxels); **no FDR / significance test** (greedy cosine argmax); no stability control |
| **Knockoffs-for-SAE** ([2511.11711](https://arxiv.org/abs/2511.11711), Nov 2025) | FDR-controlled selection of SAE latents inside **one LLM** | Single model; no brain; no cross-domain matching |
| **Universal SAEs** ([2502.03714](https://arxiv.org/html/2502.03714)) | Align SAE features **model↔model** (ImageNet ViTs) | No brain; no statistical guarantee (cosine threshold only) |
| **Superposition disentanglement** ([2510.03186](https://arxiv.org/abs/2510.03186)) | SAE disentangling improves DNN↔brain **linear-regression alignment score** | Model-side SAE only; no per-feature significance test |
| **BrainExplore** ([2512.08560](https://arxiv.org/abs/2512.08560)) | Discovers + describes brain visual representations | Brain-only; never bridges to a model; reliability scores, not FDR |

**Genuinely unclaimed intersection (this project):**
`{SAEs on BOTH model AND brain}` × `{FDR-controlled per-feature cross-domain matching}` × `{seed-stability controls}`.

> ⚠️ **Novelty is months-fresh.** SAE-BrainMap is Jun 2025; knockoffs-for-SAE is
> Nov 2025. Re-run a targeted preprint search ("dual-SAE THINGS/NSD FDR") at
> project start — the gap could close. This is logged as the day-0 go/no-go check.

## 4. Honesty boundary on the statistics (do not soften)

Model-X knockoffs give **exact, finite-sample** FDR control only when the
covariate distribution is *known* (Candès et al. 2018). We use second-order
Gaussian knockoffs estimated from data; on sparse, non-negative, heavy-tailed SAE
latents this is a known misspecification, so our guarantee is **approximate /
asymptotic** ([Fan et al. 2025](https://arxiv.org/abs/2502.05969)), not exact.
**Empirically validating that control holds on real SAE/brain latents is itself a
contribution, not an assumption.** The synthetic calibration in §6 is the first
step of that validation.

## 5. Contributions

1. **Dual-SAE** on THINGS shared stimuli — brain-side SAEs that SAE-BrainMap lacks.
2. **FDR-controlled cross-domain matching** (knockoffs), honestly framed as
   approximate, with an empirical calibration check on real latents.
3. **Seed-stability gate** (PW-MCC ensembling) — turns the field's biggest SAE
   critique into a built-in defense; we report what fraction of "matches" survive.
4. **Baseline bake-off** vs RSA and voxelwise encoding models — SAE matching must
   demonstrate it adds *interpretable, nameable, steerable* units RSA cannot.
5. **Stretch (high-risk, explicitly future work):** causal steering — perturb a
   matched model feature, predict the change in the brain-aligned axis via an
   encoding model. Not a core deliverable for a 3-month timeline.

## 6. Status — what already runs (today)

**(a) Statistical engine — validated on synthetic ground truth.**
`crosssae/knockoffs.py` (Gaussian Model-X knockoffs + Lasso-signed-max +
knockoff⁺), `crosssae/stability.py` (PW-MCC seed core), `crosssae/sae.py`
(Top-k SAE). `experiments/demo_fdr_matching.py` → `results/fdr_calibration.png`:
empirical FDR tracks at/below nominal across q ∈ [0.10, 0.30] with full recovery
of planted matches. The engine is calibrated.

**(b) Phase 1 — real model-side pipeline, end to end.**
`experiments/phase1_vit_sae.py` runs a **real pretrained ViT** over **real
images**, trains a real Top-k SAE (R²≈0.84, 182 live features), and matches its
features to semantic concepts under FDR control — **277 feature↔concept matches
across 10 concepts (q=0.2)** with structured selectivity
(`results/phase1_vit_sae.png`, `results/phase1_concept_features.csv`).

**(c) MVP — real human-brain-side pipeline, end to end.**
`experiments/mvp_brain_sae.py` runs the brain half on **real human fMRI** (Haxby
2001, ventral-temporal cortex): trains a **brain-side** Top-k SAE (R²≈0.97) and
FDR-matches brain features to the viewed object category — **73 matches**, with
unsupervised features recovering known VT organization (strong, cleanly-separated
scene/house selectivity) under knockoff control (`results/mvp_brain_sae.png`).

**(d) Cross-domain matcher + headline join.**
`crosssae/matching.py` implements the feature↔feature matcher; it recovers all
300 planted model↔brain pairs at full power under FDR control on synthetic
shared-stimulus data (`experiments/cross_domain_validation.py`).
`experiments/headline_model_brain.py` runs the **real** join — real ViT-SAE vs
real human-EEG-SAE over 200 shared THINGS-EEG2 images, matched at the independent
image level with a permutation null. The matcher fires on a synthetic positive
control (176 vs null 0) but finds **no above-chance matches** on real EEG, even
after pooling all 4 sessions (80 reps/image).

**(e) The diagnostic — and the real scientific finding.**
`experiments/rsa_diagnostic.py` (RSA, permutation-tested) localizes the null:
**raw ViT↔EEG representations share significant structure (rho=0.155, p=0.0005),
but the SAE step attenuates it below significance (rho=0.067, p=0.058).** The
model↔brain null is therefore *not* a failure of alignment (it exists) but
evidence that the sparse-autoencoder basis discards the shared cross-domain
structure — a controlled result on this project's central question (are SAE
features the right canonical unit?). Open next questions: do larger /
stability-gated SAEs recover it; does a spatially-resolved-fMRI join (stronger
per-image structure) let SAE matching beat raw RSA. Full log: `FINDINGS.md`.

## 7. Milestone plan (the long-term journey)

| Phase | Weeks | Deliverable | Standalone value |
|---|---|---|---|
| 0. Kill-check | 0 | Fresh preprint search; confirm gap open | de-risk before investing |
| 1. Real model side | 1 | Cache ViT activations over THINGS images; train model-side Top-k SAE | reproducible SAE on real model |
| 2. **MVP** | 2–4 | Train **brain-side** SAE on THINGS-EEG2; first FDR-controlled model↔brain matches | workshop-paper-shaped result alone |
| 3. Rigor | 5–7 | Seed-stability gate; empirical FDR calibration on real latents | the methods contribution |
| 4. So-what | 8–10 | Beat RSA / encoding baselines; name the matched concepts | the result that convinces |
| 5. Scale + write | 11–13 | THINGS-fMRI/MEG; arXiv preprint; open-source release | the full artifact |
| 6. Stretch | post | Causal steering experiment | follow-on paper |

## 8. Datasets

- **THINGS-EEG2** — 10 subjects, 82,160 trials, 16,740 image conditions, on
  [HuggingFace](https://huggingface.co/datasets/gasparyanartur/things-eeg2) (fast start).
- **THINGS-MEG / fMRI** — [OpenNeuro ds004212](https://openneuro.org/datasets/ds004212) (richer, stretch).
- **NSD** — 7T fMRI gold standard, [Nature Neuro 2022](https://www.nature.com/articles/s41593-021-00962-x) (validation, data agreement required).

## 9. Top risks

1. **Gap closes** before/while we work → day-0 kill-check + fast MVP.
2. **FDR control fails empirically** on heavy-tailed brain latents → that *is* a
   finding; escalate to SDP knockoffs / conditional-randomization tests.
3. **SAE instability dominates** → stability gate is built in from the start.
4. **RSA already suffices** → §5.4 baseline bake-off is the explicit test; if SAE
   adds nothing nameable/steerable over RSA, we report that honestly.
5. **Causal steering infeasible solo/3-mo** → pre-scoped as future work, not core.
