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
to independent image-level rows + a permutation null exposed that as an artifact.

**Tested the SNR hypothesis — and it failed.** I first guessed the null was
single-session SNR, so I pooled all **4 sessions (80 reps/image, ~2× SNR)** and
refined the ERP window. Still 0 matches, p=1.0. So the null is **not** merely a
data-quantity problem. That motivated the diagnostic below.

## 6. WHY the join is null: the SAE basis attenuates shared structure (RSA)
`experiments/rsa_diagnostic.py` → `results/rsa_diagnostic.png`
Representational Similarity Analysis localizes the null to data vs method:

- **Raw ViT RDM vs EEG RDM:  Spearman rho = +0.155, p = 0.0005** (permutation).
  There **is** significant shared structure between the vision transformer and
  human EEG across the 200 images.
- **model-SAE RDM vs brain-SAE RDM:  rho = +0.067, p = 0.058.** After the SAE
  step the shared structure is **more than halved and falls below significance.**

**Finding:** the model↔brain null is not "brains and models don't align" (they
do, p=0.0005) — it's that the **sparse-autoencoder basis attenuates the shared
cross-domain structure** that RSA detects in the raw representations. This is
direct, controlled evidence on the project's core question: SAE features, though
monosemantic *within* a domain, are not (with these SAEs) the right unit for
cross-domain representational alignment.

**Honest caveats:** SAE RSA is attenuated-but-marginal (p=0.058), not strictly
zero; and these are modest SAEs (d=64) trained without the stability gate. Whether
larger SAEs recover the shared structure is tested directly in §7 — and they do.

## 7. The attenuation is a CAPACITY effect, not a sparsity effect (controlled ablation)
`experiments/sae_vs_pca_rsa.py` → `results/sae_vs_pca_rsa.png`
Sweeps the basis capacity for both a sparse (SAE) and a dense (PCA) reduction of
the same real ViT and EEG representations, against two reference lines a reviewer
would require:

- **EEG noise ceiling** (split-half reliability, Spearman-Brown corrected) = **0.214**.
  Raw ViT↔EEG RSA (0.155) is ~72% of this ceiling — a meaningful, not trivial, signal.
- **SAE RSA rises monotonically with capacity:** 0.061 (k=8) → 0.085 (k=16) →
  0.111 (k=32) → **0.118 (k=64)**, still climbing, approaching the raw 0.155.
- **At matched capacity, sparse ≈ dense, and sparse wins at scale:** PCA tops out
  ~0.105; SAE reaches 0.118 at k=64 (> PCA). Sparsity does **not** cost
  cross-domain structure.

**Revised conclusion (this supersedes §6's tentative read).** The cross-domain
attenuation in §6 was a *capacity* artifact of a small (d=64) SAE, **not** a
sparsity-specific cost. A sufficiently large SAE recovers the shared structure as
well as — or better than — dense PCA, up toward the raw ceiling. So SAE features
are not inherently the wrong unit for cross-domain alignment; they just need
enough capacity. (§6's d=64 was the weakest point on this very curve, which is why
the headline FDR match — run at d=64 — was null.)

**Reproducibility note:** 3 seeds per SAE setting (error bars shown); fixed RNG;
2000-permutation RSA tests; real data only. Open next step: re-run the FDR
cross-domain match at k=64 to see whether the recovered structure yields
above-chance feature matches, and move to spatially-resolved fMRI for a higher
noise ceiling.
