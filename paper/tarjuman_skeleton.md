# Tarjuman: Do Neural Foundation Models Make Sparse Features a Viable Interlingua Between Brains and Vision Transformers?

*Paper skeleton — v0.1. Azra Bano. Target: NeurIPS/ICLR (main or interpretability/NeuroAI workshop as a fast first release).*

> **Working honesty banner.** Numbers marked ✅ are real, logged results
> (FINDINGS.md §6–10); numbers in [brackets] are still target slots. The repo's
> integrity rule holds: report what the data says, with the control that proves it,
> and never tune a pipeline until a null becomes positive. The empirical arc so far is
> a chain of *honest negatives that each redirect the work* — that is the paper's spine,
> not a weakness.

---

## Abstract (target shape)

Sparse autoencoders (SAEs) extract monosemantic features from neural networks, and
recent work aligns model-side SAE features to human visual cortex — but does so
without statistical control, and never trains SAEs on a *foundation-model*
representation of the brain. A vision transformer and human EEG share significant
representational structure (RSA ρ=+0.155, p=5×10⁻⁴), and our prior capacity ablation
(FINDINGS §7) showed the apparent SAE attenuation of that structure is a **capacity**
effect, not a sparsity cost: SAE-RSA climbs from ρ≈0.06 (k=8) toward ρ≈0.12 (k=64)
and matches dense PCA at equal capacity — yet even the best stays well under the
split-half **EEG noise ceiling (ρ≈0.21)**. The open question is therefore not "does
the basis destroy structure" (it does not, beyond capacity) but **whether a better
brain-side *substrate* raises the achievable cross-domain alignment toward that
ceiling**. We ask whether a **neural foundation model** (CBraMod/LaBraM) supplies
that substrate: training the brain-side SAE on FM latents rather than raw EEG. Holding
the model side, stimuli, SAE recipe, capacity, and the FDR-controlled matching engine
fixed, we compare brain-side representations — raw EEG, a capacity-matched PCA control,
and FM latents — across a capacity sweep, against the noise ceiling, on (i) RSA before
vs after the sparse bottleneck and (ii) FDR-controlled ViT↔brain feature matching with
a permutation null. **Findings.** (1) ✅ On THINGS-EEG2, a real pretrained EEG FM
(CBraMod) does *not* lift cross-domain alignment — but for a diagnosable reason, not a
refutation: CBraMod's 1 s patch over 200 ms RSVP collapses the per-image noise ceiling
to ρ≈0.009 (each FM input spans ~5 images), so the testbed, not the hypothesis, fails.
(2) ✅ Moving the brain side to spatially-resolved THINGS-fMRI raises the visual-cortex
ceiling to ρ≈0.46 (2.1× scalp EEG) and the SAE *preserves* that structure (RDM-pres
0.83, image-decoding 38% > raw voxels 20% via denoising) — the brain-side dictionary is
viable there. (3) ✅ But at matched capacity dense PCA out-preserves the SAE (60% vs
38% decode), so on rich substrates **sparsity trades fidelity for interpretability** —
the case for SAE-over-PCA rests on monosemantic, nameable, FDR-matchable features, which
we make explicit. We additionally extend Model-X knockoff feature selection from
artificial-LM latents toward *biological* latents, characterizing where its Gaussian
approximation holds. The headline ViT-SAE↔fMRI-SAE join is method-ready and gated only
on access-controlled stimulus images, not on the approach.

---

## 1. Introduction

- **The dream and the wall.** Neural foundation models (POYO/POYO+, NDT3, CBraMod,
  LaBraM) decode brain activity but are black boxes; mech-interp (SAEs) opens up
  *artificial* networks. Almost no work sits at the intersection: SAEs were applied
  to a transformer trained on biological recordings for the *first time* in mid-2025
  ("Beyond Black Boxes", POYO+ TopK-SAE). Cross-system, statistically-controlled
  feature alignment between an FM's brain dictionary and a model's dictionary does
  not exist.
- **Our prior result chain** (the hook): raw ViT↔EEG structure is real and
  temporally localized to ~120 ms (FINDINGS §8, object-recognition latency — so it is
  genuine visual-object representation, not artifact); the SAE "attenuation" is a
  capacity effect, not sparsity (§7); and scalp EEG caps the achievable alignment at
  ρ≈0.21. This paper asks whether an FM *substrate* — not more SAE capacity — is what
  finally moves cross-domain alignment toward that ceiling.
- **Contributions** (see §3).
- **Framing for safety/interpretability:** the brain is an external validity testbed
  no training pipeline produced; a feature correspondence that survives FDR +
  stability + a permutation null is strong evidence the SAE feature is a real unit,
  not a dictionary artifact. The machinery transfers verbatim to LLM↔LLM and
  checkpoint↔checkpoint feature matching.

## 2. Related work (extends cross-sae's table)

| Work | Brain-side rep | Brain SAE? | FDR control? | FM backbone? | Cross-system match? |
|---|---|---|---|---|---|
| SAE-BrainMap (2506.11123) | raw voxels | ✗ | ✗ (cosine argmax) | ✗ | model→voxel only |
| Beyond Black Boxes (2506.14014) | POYO+ latents | ✓ (TopK-SAE) | ✗ | ✓ (decoder) | ✗ (within-model) |
| Knockoffs-for-SAE (2511.11711) | — (LM only) | n/a | ✓ (approx) | ✗ | ✗ (single LLM) |
| Universal SAEs (2502.03714) | — | n/a | ✗ | ✗ | model↔model |
| Superposition disentangle (2510.03186) | — | ✗ | ✗ | ✗ | regression score |
| **Tarjuman (this work)** | **FM latents** | **✓** | **✓ (approx, validated)** | **✓** | **✓ FM-brain ↔ ViT** |

**Unclaimed cell:** {brain-side SAE on an FM latent space} × {FDR-controlled
cross-system matching} × {seed-stability gate}. **Day-0 kill-check required** — an
EEG-SAE preprint landed ~May 2026; verify this exact combination is still open
before submission.

## 3. Contributions

1. **A substrate/basis decomposition of cross-domain SAE matching** — we separate
   three things usually conflated: the *substrate* (EEG vs FM-latents vs fMRI), the
   *basis* (sparse SAE vs dense PCA), and *capacity*. Result: capacity (not sparsity)
   governs recovery on EEG (§7); substrate governs the ceiling (EEG ρ≈0.21 vs fMRI
   ρ≈0.46, §10); and on rich substrates the sparse basis trades fidelity for
   interpretability (PCA > SAE at matched capacity, §10).
2. **A diagnosable FM-substrate negative** — the first attempt to use a real pretrained
   EEG FM (CBraMod) as a brain-side SAE substrate, with the *mechanism* of its failure
   pinned quantitatively (1 s-patch vs 200 ms-RSVP → per-image ceiling collapse), which
   redirects the field to matched-SOA / fMRI substrates rather than abandoning the idea.
3. **Knockoff FDR toward biological latents** — extending Model-X knockoffs from
   artificial-LM SAE latents to brain latents; empirical calibration of where the
   Gaussian-surrogate guarantee holds on non-Gaussian neural latents.
4. **The Rosetta dictionary (method-ready)** — a significance-tested, stability-gated
   set of features shared between an artificial vision model and human visual cortex,
   with auto-generated NL cards; gated only on access-controlled images, not method.
5. **Open-source tool** — `crosssae` + `backbone` + the experiments; reproducible,
   dependency-gated, honest-fallback, and ethics-respecting (no gated-data bypass).

## 4. Method

- **Shared-stimulus design.** THINGS-EEG2: the same 200 (→16,740) images shown to
  humans and fed to a ViT. (§4.1)
- **Model side.** Pretrained ViT block-6 activations → Top-k SAE. (unchanged)
- **Brain side (the variable).** raw EEG epochs → **backbone** ∈ {identity, PCA,
  EEG-FM} → Top-k SAE. (`crosssae/backbone.py`) (§4.2)
- **Matching.** Per-target Gaussian Model-X knockoffs + Lasso-signed-max + knockoff⁺
  at FDR q; per model feature select the matched brain features. (`crosssae/matching`)
- **Stability gate.** PW-MCC seed core on both sides before matching. (`crosssae/stability`)
- **Two diagnostics, identical across arms.** (i) RSA before vs after the SAE
  bottleneck (does the SAE preserve structure?); (ii) FDR matches vs permutation null.

## 5. Experiments & results

- **E1 — EEG substrate × capacity sweep vs noise ceiling.** ✅ `tarjuman_fm_join.py`
  reproduces §7 (ceiling 0.214; raw ViT↔EEG 0.155, p=5e-4; brain-SAE RSA 0.071→0.116
  over k=8→64; SAE≈PCA). Harness validated; best raw-EEG substrate uses only ~54% of
  available signal. [`results/tarjuman_fm_join.png`]
- **E2 — Real CBraMod FM substrate (EEG).** ✅ `tarjuman_fm_cbramod.py`. Identical 1 s
  windows, substrate the only variable, k=32: raw-EEG SAE 0.028 (ceiling 0.100) vs
  eeg-fm SAE 0.043 (**ceiling 0.009**). Thesis NOT supported on THINGS-EEG2 — confound,
  not refutation: 1 s-patch/200 ms-RSVP mismatch collapses the FM per-image ceiling.
  [`results/tarjuman_fm_cbramod.png`]
- **E3 — fMRI testbed ceiling.** ✅ `fmri_join_thingsfmri.py`. THINGS-fMRI visual cortex
  (10,481 voxels): image-decoding 20% (chance 1%), RDM ceiling **0.457 = 2.1× EEG**;
  whole-brain dilution artifact caught (0.04) and excluded. [`results/fmri_ceiling_roi.png`]
- **E4 — fMRI brain-side SAE preservation.** ✅ `fmri_brain_sae.py`. SAE preserves
  structure (k=64: RDM-pres 0.83, decode 38% > raw 20%) → dictionary viable; honest
  caveat: PCA > SAE at matched capacity (60% vs 38%), R²=0.61 → sparsity-fidelity cost.
  [`results/fmri_brain_sae.png`]
- **E5 — Knockoff calibration on neural latents** (synthetic planted structure; does
  empirical FDR ≤ nominal on heavy-tailed brain latents?). [slot, extends `fdr_calibration`]
- **E6 — The headline join + dictionary** (needs `THINGS_IMAGES_DIR`): ViT-SAE↔fMRI-SAE
  FDR matches + permutation null at the 0.46 ceiling; top matched pairs, NL cards,
  ROI/known-organization sanity, stability fractions. [method-ready; gated on images]
- **E7 — Ablations:** SAE width/k, subjects, ROI subsets; stability gate (PW-MCC).

## 6. Honesty / limitations (do not soften)

- Knockoff FDR on SAE latents is **approximate** (Gaussian surrogate; neural latents
  are sparse, non-negative, heavy-tailed) — validating control empirically is a
  contribution, not an assumption (mirrors `crosssae/knockoffs.py`).
- EEG is low-spatial-resolution; a null on EEG does not refute alignment — NSD fMRI
  (v2) is the stronger testbed. State data-limits explicitly.
- The FM arm is only as good as the released checkpoint; report the exact weights,
  layer, and preprocessing; never substitute an untrained network.
- The "Platonic universal axes drive alignment" strong causal claim was refuted in
  the literature (0-3 in our adversarial review) — we make only the weaker
  empirical claim.
- **Sparsity has a fidelity cost on rich substrates** (§10): on fMRI, dense PCA
  out-preserves the SAE at matched capacity. We do not claim SAEs are the most faithful
  basis; we claim they are the most *interpretable* one (monosemantic, nameable,
  FDR-matchable), and we report the PCA gap rather than hiding it.
- **Dataset↔FM matching matters** (§9b): a foundation model's input granularity must
  match the data's presentation rate (CBraMod's 1 s patch vs 200 ms RSVP). We report
  the per-image noise-ceiling collapse as the diagnostic, not a raw RSA number.
- **Ethics:** THINGS object images are access-controlled; all results here use only
  openly-licensed betas/ROI metadata read in place — the join awaits user-supplied
  `THINGS_IMAGES_DIR`, never a bypass.

## 7. Broader impact / the bridge to Tarjuman (product)

The dictionary is a *legible control interface*: it names what a circuit represents
with a calibrated error rate, which is the prerequisite for any auditable read/write
neural system. See `TARJUMAN.md` for the read-out agent and the simulation-only,
feature-targeted write policy that build on this artifact. We deliberately keep all
write-side work in simulation in this paper.

## 8. Reproducibility

`python experiments/tarjuman_fm_join.py` (offline arms); `TARJUMAN_FM_WEIGHTS=...`
adds the FM arm. Engine validated on synthetic ground truth
(`results/fdr_calibration.png`). Seeds fixed; permutation nulls reported.

---

### Author TODO before submission
- [x] Day-0 kill-check (combo still unclaimed — June 2026)
- [x] Run real CBraMod FM arm (EEG) — diagnosable negative (E2)
- [x] Establish fMRI as higher-ceiling testbed + SAE viability there (E3/E4)
- [ ] **Headline join (E6):** obtain `THINGS_IMAGES_DIR` (credentialed) → ViT-SAE↔fMRI-SAE
      FDR matches at the 0.46 ceiling — the result that closes the paper
- [ ] Knockoff FDR calibration on neural latents (E5)
- [ ] Stability fractions (PW-MCC) for every reported match
- [ ] Widen/better-train the fMRI SAE to narrow the PCA gap (or argue interpretability-only)
- [ ] Multi-subject fMRI; consider NSD as second substrate
- [ ] Decide venue: workshop fast-release (current results) vs main-track (with E6)
