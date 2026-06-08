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

## 8. The shared structure is genuine object recognition: it peaks at 120 ms
`experiments/temporal_rsa.py` → `results/temporal_rsa.png`
Time-resolved RSA (Cichy et al. 2014 style) between the fixed ViT RDM and the EEG
RDM in sliding 40 ms windows, with a per-window split-half noise ceiling and a
1000-permutation test per window:

- **Flat / non-significant before ~80 ms** (no spurious low-level or pre-onset alignment).
- **Rises sharply and becomes significant at 100 ms** (p=0.022), **peaks at 120 ms
  (RSA=0.125, p=0.001)**, stays significant through 150 ms, then decays.
- 6/17 windows significant at p<0.05; the peak sits well under the noise ceiling.

**Interpretation.** 120 ms is the canonical latency at which object/category
information becomes decodable in human visual cortex, and matches published
CNN↔brain RSA timing (Cichy et al. 2014). So the ViT↔EEG structure this project
matches is **genuine visual-object representation** — it appears at the
object-recognition moment, not before (ruling out low-level/onset confounds) and
fades with the RSVP cycle. This anchors the whole cross-domain question: there is
a real, temporally-localized, semantically-meaningful signal to be matched; the
open problem is the *unit* (raw vs SAE vs FM-latent) and the *brain modality*
(scalp EEG's low ceiling vs fMRI) that let a feature-level matcher recover it.

**Caveat:** analysis restricted to 0–200 ms because the RSVP SOA is 200 ms
(later windows mix the next image); single subject; the per-window noise ceiling is
itself noisy (split-half on single-window amplitudes).

## 9. Tarjuman — brain-side SUBSTRATE as the new axis (harness validated; FM arm open)
`experiments/tarjuman_fm_join.py` → `results/tarjuman_fm_join.png`
Tarjuman (see `TARJUMAN.md`, `paper/tarjuman_skeleton.md`) asks the §8 "unit"
question one level down: not raw-vs-SAE basis (settled in §7: capacity, not sparsity)
but the brain-side *substrate* the SAE is trained on — raw EEG vs a pretrained EEG
foundation-model (CBraMod/LaBraM) latent space. The experiment reuses the §7 protocol
verbatim (split-half noise ceiling, capacity sweep k∈{8,16,32,64}, multi-seed SAE,
dense-PCA control) and adds substrate as the new axis.

- **Built-in cross-check PASSED.** The raw-EEG substrate reproduces §7: noise
  ceiling = 0.214, raw ViT↔EEG = 0.155 (p=0.0005), brain-SAE RSA climbs 0.071 (k=8) →
  0.116 (k=64), SAE ≈ dense PCA at matched capacity. The harness is trustworthy.
- **Quantified headroom (the target).** Best raw-EEG substrate reaches RSA 0.116 vs a
  0.214 ceiling — only ~54% of the available signal. That ~0.10 gap is the room a
  better substrate would have to close, and is the bar the FM arm must clear.
- **Day-0 kill-check (June 2026) PASSED** — {brain-SAE-on-FM-substrate × FDR
  cross-system match × stability} still unclaimed.

## 9b. Tarjuman — real CBraMod FM arm: honest negative, and WHY (testbed mismatch)
`experiments/tarjuman_fm_cbramod.py` → `results/tarjuman_fm_cbramod.png`
Ran the actual hypothesis with **real pretrained CBraMod** (ICLR 2025; HF
`weighting666/CBraMod`, loads with 0 missing keys). Design: identical 1 s single-trial
windows for both arms (CBraMod's patch is hard-locked to 200 pts = 1 s @ 200 Hz, the
minimum valid input), substrate the only variable; k=32, 2 seeds; per-substrate
split-half noise ceiling.

| arm (1 s windows, k=32) | SAE RSA | PCA RSA | rep-RSA | substrate ceiling |
|---|---|---|---|---|
| raw-EEG @1 s | +0.028 | +0.085 | +0.135 (p<.001) | **0.100** |
| eeg-fm (CBraMod) | +0.043 | +0.014 | +0.068 (p=.043) | **0.009** |

**Verdict: THESIS NOT SUPPORTED on THINGS-EEG2 — but this is a CONFOUND, not a clean
refutation, and the evidence says exactly why:**
- **FM noise ceiling collapses to 0.009** — CBraMod's per-image latents are
  trial-*un*reliable here. With a 1 s window over 200 ms RSVP, each FM input spans ~5
  images, so almost no trial-reproducible per-image signal survives. (FM-SAE 0.043
  even *exceeds* its 0.009 ceiling → the absolute FM numbers are unreliable, not real.)
- **The 1 s window alone wrecks the signal:** raw-EEG SAE falls 0.116 (clean 300 ms,
  §7) → 0.028 here. Window mismatch, not the FM, drives most of the degradation.

**What this proves:** the full pipeline runs end-to-end with a *real* foundation model
and yields an honest, mechanistically-explained negative. **Decisive redirect:**
THINGS-EEG2 (200 ms RSVP) is the wrong testbed for a 1 s-patch EEG FM. A clean test of
the FM-substrate hypothesis needs **slower-SOA visual EEG (≥1 s/stimulus)** or **fMRI
(NSD — clean per-image betas)**. Preprocessing was also only a generic stand-in for
CBraMod's pretraining pipeline (second-order caveat). Next: pick a matched-SOA testbed
before re-testing; do NOT tune this dataset to force a positive.
