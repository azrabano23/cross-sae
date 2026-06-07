# Findings (running log)

Honest, dated record of what each experiment showed — including the negative
results. The integrity rule for this repo: report what the data says, with the
control that proves it, and never tune a pipeline until a null result becomes
positive.

## 1. The statistical engine is calibrated (synthetic)
`experiments/demo_fdr_matching.py` → `results/fdr_calibration.png`
Knockoff matching with planted ground truth: empirical FDR tracks at/below the
nominal level across q ∈ [0.10, 0.30] with full recovery of true matches. The
engine controls error as advertised.

## 2. The cross-domain matcher recovers planted model↔brain pairs (synthetic)
`experiments/cross_domain_validation.py` → `results/cross_domain_validation.png`
Two SAE feature banks over shared stimuli with planted 1-to-many correspondences:
the matcher recovers all 300 planted pairs at full power with per-target FDR at/
below nominal. The headline method works *when cross-domain structure exists*.

## 3. Model side works on real data (real ViT)
`experiments/phase1_vit_sae.py` → `results/phase1_vit_sae.png`
Real pretrained ViT over real CIFAR-10 → Top-k SAE (R²≈0.84) → **277**
FDR-controlled feature↔concept matches across 10 concepts, with structured
selectivity. The model-side SAE produces nameable, significance-tested features.

## 4. Brain side works on real data (real human fMRI)
`experiments/mvp_brain_sae.py` → `results/mvp_brain_sae.png`
Real Haxby fMRI (ventral-temporal cortex) → brain-side Top-k SAE (R²≈0.97) →
**73** FDR-controlled feature↔category matches. Unsupervised brain-SAE features
recover known organization — strong, cleanly separated scene/house selectivity.
Object categories that don't clear threshold (face, bottle, …) reflect
conservative FDR + single-subject diffuse coding; reported, not hidden.

## 5. Real model↔brain join: honest negative on single-session EEG
`experiments/headline_model_brain.py` → `results/headline_model_brain.png`
Real ViT-SAE features vs real human-EEG-SAE features (THINGS-EEG2 sub-01,
200 shared natural images, matched at the independent image level, n=200).

- **Positive control** (synthetic data with planted structure, same matcher):
  176 matches vs permutation-null max 0 — the method detects cross-domain
  structure when present.
- **Real data:** 0 FDR-significant matches; permutation null (shuffled image↔brain
  alignment) mean ≈ 0.2 → permutation **p = 1.0**. No above-chance signal.

**Why this is the correct result, not a failure.** An earlier pseudo-trial
version (n=1000, 5 correlated pseudo-trials/image) reported 23 "matches" — but
those rows are not independent, so the knockoff filter was overconfident. Moving
to independent image-level rows + a permutation null exposed that as an artifact:
the honest answer is that single-subject, single-session **scalp EEG** averaged to
200 images does not carry above-chance model↔brain SAE-feature correspondence.
This is consistent with EEG's lack of the spatial structure fMRI has. The
bottleneck is brain-data SNR, **not** the method (which the positive control and
§1–2 validate).

**Indicated next step for a positive cross-domain result:** move the join to
spatially-resolved fMRI on a shared-stimulus set (NSD or THINGS-fMRI), where the
brain-side SAE already works well (§4), and/or pool all 4 EEG sessions × 80
repetitions per image for substantially higher SNR.
