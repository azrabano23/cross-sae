# Tarjuman: Do Neural Foundation Models Make Sparse Features a Viable Interlingua Between Brains and Vision Transformers?

*Paper skeleton — v0.1. Azra Bano. Target: NeurIPS/ICLR (main or interpretability/NeuroAI workshop as a fast first release).*

> **Working honesty banner (keep in the draft until results are in):** every
> headline number below is a *target slot*, not a result. The repo's integrity rule
> (FINDINGS.md) holds: report what the data says, with the control that proves it,
> and never tune a pipeline until a null becomes positive.

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
a permutation null. The FM arm must beat the capacity-matched PCA control, not merely
raw EEG, to justify itself. We report [RESULT]. We
additionally extend Model-X knockoff feature selection from artificial-LM latents to
*biological* FM latents, characterizing where its Gaussian approximation holds.
[If positive:] the result yields the first statistically-controlled, FM-grounded
"Rosetta dictionary" of features shared between an artificial vision model and the
human brain. [If negative:] it bounds when sparse features can serve as a
cross-system interlingua at all — itself a load-bearing result for interpretability.

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

1. **The FM-backbone hypothesis test** — a clean, single-variable comparison
   (raw / PCA / FM brain representation) of whether an FM latent space lets sparse
   features survive cross-domain matching. Resolves our prior negative.
2. **Knockoff FDR on biological FM latents** — first extension of Model-X knockoffs
   from artificial-LM SAE latents to a *brain* FM's SAE latents; empirical
   calibration of where the Gaussian-surrogate guarantee holds on non-Gaussian
   neural latents.
3. **The Rosetta dictionary (conditional on a positive result)** — a
   significance-tested, stability-gated set of features shared between an artificial
   vision model and human visual cortex, with auto-generated natural-language cards.
4. **Open-source tool** — `crosssae` + `backbone` + the FM-join experiment;
   reproducible, dependency-gated, honest-fallback.

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

## 5. Experiments & results [SLOTS]

- **E1 — Substrate × capacity sweep against the noise ceiling.** For each backbone
  {identity, PCA, FM} sweep capacity k∈{8,16,32,64} (multi-seed), plot cross-domain
  RSA vs the split-half EEG noise ceiling — the exact protocol of `sae_vs_pca_rsa.py`
  (§7), with the brain substrate as the new axis. *Hypothesis:* the FM curve sits
  above the capacity-matched PCA/identity curves and closer to the ceiling. A null
  here (FM ≈ PCA) is a real, reportable bound. [`results/tarjuman_fm_join.png`]
- **E2 — FDR-controlled matches + permutation null** per arm. [slot]
- **E3 — Knockoff calibration on FM latents** (synthetic planted structure in FM
  latent space; does empirical FDR ≤ nominal?). [slot, extends `fdr_calibration`]
- **E4 — The dictionary** (if E1/E2 positive): top matched feature pairs, NL cards,
  ROI/known-organization sanity. [slot]
- **E5 — Ablations:** SAE width/k, FM layer, pooling, subjects; stability fraction.

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
- [ ] Day-0 kill-check (dual-SAE-on-FM × FDR still unclaimed?)
- [ ] Run FM arm with real CBraMod + LaBraM weights; pick layer by E5
- [ ] NSD fMRI replication (v2 / stronger testbed)
- [ ] Stability fractions for every reported match
- [ ] Decide venue: workshop fast-release vs main-track with NSD
