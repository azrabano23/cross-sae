"""
TARJUMAN — does a neural FOUNDATION-MODEL substrate raise cross-domain alignment
toward the EEG noise ceiling, beyond what SAE capacity alone buys?

This supersedes the naive "FM rescues a null" framing. cross-sae's capacity ablation
(experiments/sae_vs_pca_rsa.py, FINDINGS §7) already established the correct picture:

  * raw ViT<->EEG share real structure (RSA rho~0.155, p=5e-4), temporally localized
    to ~120 ms (§8) — genuine visual-object representation.
  * the SAE "attenuation" of that structure is a CAPACITY effect, not sparsity:
    SAE-RSA climbs with k and matches dense PCA at equal capacity.
  * scalp EEG caps any model's achievable RSA at the split-half noise ceiling (~0.21).

So the open question Tarjuman tests is about the brain-side *substrate*, not the basis:

    Holding the SAE recipe, capacity, model side, and stimuli fixed, does feeding the
    SAE a pretrained EEG FOUNDATION-MODEL latent space (instead of raw EEG) raise the
    achievable cross-domain RSA toward the noise ceiling — and does it beat a
    capacity-matched PCA control on the SAME substrate?

It reuses the §7 protocol verbatim (noise ceiling, capacity sweep, multi-seed SAE,
PCA control) and adds SUBSTRATE as the new axis:

    substrate ∈ { raw-EEG , eeg-fm(CBraMod/LaBraM) }
        × basis ∈ { SAE(sparse) , PCA(dense) }
        × capacity k ∈ {8,16,32,64}

The raw-EEG substrate runs offline and should REPRODUCE §7 (a built-in cross-check).
The eeg-fm substrate needs real weights; absent them it is SKIPPED, never faked.

Run:
    python experiments/tarjuman_fm_join.py                       # raw-EEG substrate only
    TARJUMAN_FM_WEIGHTS=/path/to/cbramod.pth \
        python experiments/tarjuman_fm_join.py                   # + the FM substrate
Out:
    results/tarjuman_fm_join.png / .csv
"""
from __future__ import annotations

import os
import sys
import csv
import glob
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crosssae.sae import train_sae, encode_dataset
from crosssae.backbone import get_backbone, FMConfig
from experiments.headline_model_brain import (
    load_epochs, vit_features, DATA, SESSIONS, EPOCH_MS,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

CAPACITIES = [8, 16, 32, 64]      # SAE top-k / PCA components (d_hidden = 8*k, as §7)
SEEDS = [0, 1]                    # SAE non-determinism; mean +/- std (2 for tractability)
N_PERM = 2000                     # RSA permutation test


# --------------------------------------------------------------------------- #
#  data                                                                         #
# --------------------------------------------------------------------------- #
def load_trials_raw():
    """RAW (channel x time) single-trial windows for the FM tokenizer, aligned to
    load_epochs() trial ordering. Returns (Xraw (n_trials,n_ch,n_time), trial_img)."""
    per_code = {c: [] for c in range(1, 201)}
    for s in SESSIONS:
        a = np.load(s, allow_pickle=True).item()
        data = a["raw_eeg_data"]; ch = list(a["ch_names"]); sf = int(a["sfreq"])
        eeg_idx = [i for i, t in enumerate(a["ch_types"]) if t == "eeg"]
        stim = data[ch.index("stim")]
        X = data[eeg_idx]
        X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
        onset = np.where((stim[1:] > 0) & (stim[:-1] == 0))[0] + 1
        codes = stim[onset].astype(int)
        keep = codes <= 200
        onset, codes = onset[keep], codes[keep]
        s0 = int(EPOCH_MS[0] * sf / 1000); s1 = int(EPOCH_MS[1] * sf / 1000)
        for o, c in zip(onset, codes):
            if o + s1 < X.shape[1]:
                per_code[int(c)].append(X[:, o + s0:o + s1])
    trials, trial_img = [], []
    for code in range(1, 201):
        for w in per_code[code]:
            trials.append(w); trial_img.append(code - 1)
    return np.asarray(trials, dtype=np.float32), np.asarray(trial_img)


# --------------------------------------------------------------------------- #
#  RSA helpers (identical to §7)                                                #
# --------------------------------------------------------------------------- #
def rdm(X):
    Xz = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    return 1.0 - (Xz @ Xz.T) / X.shape[1]


def _upper(M):
    return M[np.triu_indices(M.shape[0], k=1)]


def rsa(Ra, Rb):
    return spearmanr(_upper(Ra), _upper(Rb)).correlation


def rsa_p(Ra, Rb, n_perm=N_PERM, seed=0):
    obs = rsa(Ra, Rb); a = _upper(Ra); n = Ra.shape[0]
    rng = np.random.default_rng(seed)
    null = np.array([spearmanr(a, _upper(Rb[np.ix_(p, p)])).correlation
                     for p in (rng.permutation(n) for _ in range(n_perm))])
    return float(obs), float((1 + np.sum(null >= obs)) / (1 + n_perm))


def eeg_noise_ceiling(Xtr, trial_img, n_splits=10, seed=0):
    """Split-half reliability of the EEG RDM (Spearman-Brown corrected) = ceiling."""
    rng = np.random.default_rng(seed)
    rels = []
    for _ in range(n_splits):
        A = np.zeros((200, Xtr.shape[1])); B = np.zeros((200, Xtr.shape[1]))
        for i in range(200):
            idx = np.where(trial_img == i)[0]; rng.shuffle(idx)
            h = len(idx) // 2
            A[i] = Xtr[idx[:h]].mean(0); B[i] = Xtr[idx[h:2 * h]].mean(0)
        r = rsa(rdm(A), rdm(B)); rels.append(2 * r / (1 + r))
    return float(np.mean(rels)), float(np.std(rels))


# --------------------------------------------------------------------------- #
#  one substrate: sweep capacity for SAE (sparse) and PCA (dense)               #
# --------------------------------------------------------------------------- #
def sweep_substrate(name, Brain_trials, trial_img, Vimg_t, R_vit_by_k_sae, R_vit_pca):
    """Returns dict with per-capacity SAE (mean/std) and PCA cross-domain RSA, where
    the brain side is `Brain_trials` (this substrate) and the model side is the SHARED
    ViT SAE/PCA precomputed in main (so model side is identical across substrates)."""
    Bt = torch.from_numpy(Brain_trials.astype(np.float32))
    Brain_img = np.stack([Brain_trials[trial_img == i].mean(0) for i in range(200)])

    sae_mean, sae_std, pca_rho = [], [], []
    for c in CAPACITIES:
        d = 8 * c
        rs = []
        for s in SEEDS:
            bsae = train_sae(Bt, d, c, seed=s, epochs=120)
            Ztr = encode_dataset(bsae, Bt).numpy()
            Zb = np.stack([Ztr[trial_img == i].mean(0) for i in range(200)])
            rs.append(rsa(R_vit_by_k_sae[(c, s)], rdm(Zb)))
        sae_mean.append(float(np.mean(rs))); sae_std.append(float(np.std(rs)))
        # dense PCA control on THIS substrate's per-image rep
        Pe = PCA(n_components=min(c, Brain_img.shape[1]), random_state=0).fit_transform(Brain_img)
        pca_rho.append(rsa(R_vit_pca[c], rdm(Pe)))
        print(f"  [{name}] k={c:3d}: SAE {sae_mean[-1]:+.3f}+/-{sae_std[-1]:.3f}  "
              f"PCA {pca_rho[-1]:+.3f}")
    return dict(name=name, sae_mean=sae_mean, sae_std=sae_std, pca=pca_rho)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(0)

    print("Loading EEG (binned trials) + ViT over 200 shared images...")
    Xtr, trial_img, Xb_img, img_paths = load_epochs()
    Vimg = vit_features(img_paths)
    Vimg_t = torch.from_numpy(Vimg)
    print(f"  trials {Xtr.shape}, per-image EEG {Xb_img.shape}, ViT {Vimg.shape}")

    # noise ceiling + raw cross-domain RSA (substrate-independent reference lines)
    nc_mean, nc_std = eeg_noise_ceiling(Xtr, trial_img)
    raw_rho, raw_p = rsa_p(rdm(Vimg), rdm(Xb_img))
    print(f"  EEG noise ceiling (SB) = {nc_mean:.3f} +/- {nc_std:.3f}")
    print(f"  RAW ViT<->EEG RSA = {raw_rho:.3f} (p={raw_p:.4f})")

    # SHARED model side: ViT SAE RDM per (capacity, seed) and ViT PCA RDM per capacity.
    print("Precomputing shared model-side ViT SAE/PCA RDMs across capacities...")
    R_vit_sae, R_vit_pca = {}, {}
    for c in CAPACITIES:
        d = 8 * c
        for s in SEEDS:
            msae = train_sae(Vimg_t, d, c, seed=s, epochs=200)
            R_vit_sae[(c, s)] = rdm(encode_dataset(msae, Vimg_t).numpy())
        R_vit_pca[c] = rdm(PCA(n_components=c, random_state=0).fit_transform(Vimg))

    substrates = []

    # --- substrate 1: raw EEG (should reproduce §7 — built-in cross-check) -----
    print("\n[substrate: raw-EEG]  (reproduces sae_vs_pca_rsa.py / §7)")
    substrates.append(sweep_substrate("raw-EEG", Xtr, trial_img, Vimg_t,
                                      R_vit_sae, R_vit_pca))

    # --- substrate 2: eeg-fm latents (the hypothesis; needs real weights) ------
    weights = os.environ.get("TARJUMAN_FM_WEIGHTS")
    if weights:
        print(f"\n[substrate: eeg-fm]   CBraMod latents from {weights}")
        try:
            Xraw_tr, fm_trial_img = load_trials_raw()
            fm = get_backbone("cbramod", cfg=FMConfig(weights_path=weights))
            Xtr_fm = fm.transform(Xraw_tr)
            substrates.append(sweep_substrate("eeg-fm", Xtr_fm, fm_trial_img, Vimg_t,
                                              R_vit_sae, R_vit_pca))
        except Exception as e:
            print(f"  FM substrate FAILED (reported, not hidden): {e}")
    else:
        print("\n[substrate: eeg-fm]   SKIPPED — no weights. This is the hypothesis")
        print("  test and is never faked. Run it with:")
        print("    TARJUMAN_FM_WEIGHTS=/path/to/cbramod.pth python experiments/tarjuman_fm_join.py")
        print("  (CBraMod: github.com/wjq-learning/CBraMod ; LaBraM: github.com/935963004/LaBraM)")

    # --- figure ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.axhspan(nc_mean - nc_std, nc_mean + nc_std, color="#c8e6c9", alpha=0.6,
               label=f"EEG noise ceiling ({nc_mean:.2f})")
    ax.axhline(raw_rho, color="#37474f", ls=":", lw=1.8,
               label=f"raw ViT<->EEG ({raw_rho:.2f}, p={raw_p:.3f})")
    colors = {"raw-EEG": "#c62828", "eeg-fm": "#6a1b9a"}
    for sub in substrates:
        col = colors.get(sub["name"], "#1565c0")
        ax.errorbar(CAPACITIES, sub["sae_mean"], yerr=sub["sae_std"], fmt="o-",
                    color=col, lw=2, ms=7, capsize=4, label=f"{sub['name']} · SAE (sparse)")
        ax.plot(CAPACITIES, sub["pca"], "s--", color=col, lw=1.5, ms=6, alpha=0.7,
                label=f"{sub['name']} · PCA (dense)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xscale("log", base=2); ax.set_xticks(CAPACITIES); ax.set_xticklabels(CAPACITIES)
    ax.set_xlabel("active dimensions per sample (SAE top-k / PCA components)")
    ax.set_ylabel("cross-domain RSA  (ViT RDM vs brain RDM)")
    ax.set_title("TARJUMAN — does an EEG foundation-model SUBSTRATE raise cross-domain\n"
                 "alignment toward the noise ceiling? (FM line above raw-EEG = the thesis)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    out = os.path.join(RESULTS, "tarjuman_fm_join.png")
    fig.savefig(out, dpi=150); print(f"\nSaved figure -> {out}")

    with open(os.path.join(RESULTS, "tarjuman_fm_join.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["substrate", "capacity", "sae_rsa_mean", "sae_rsa_std", "pca_rsa"])
        for sub in substrates:
            for i, c in enumerate(CAPACITIES):
                w.writerow([sub["name"], c, f"{sub['sae_mean'][i]:.4f}",
                            f"{sub['sae_std'][i]:.4f}", f"{sub['pca'][i]:.4f}"])
        w.writerow(["raw_rho", "", f"{raw_rho:.4f}", "noise_ceiling", f"{nc_mean:.4f}"])

    # --- verdict --------------------------------------------------------------
    print("\n=== VERDICT ===")
    raw_sub = next((s for s in substrates if s["name"] == "raw-EEG"), None)
    fm_sub = next((s for s in substrates if s["name"] == "eeg-fm"), None)
    if raw_sub:
        print(f"raw-EEG best SAE = {max(raw_sub['sae_mean']):.3f} "
              f"(ceiling {nc_mean:.3f}) — cross-check vs §7.")
    if fm_sub and raw_sub:
        d_sae = max(fm_sub["sae_mean"]) - max(raw_sub["sae_mean"])
        beats_pca = max(fm_sub["sae_mean"]) - max(fm_sub["pca"])
        print(f"eeg-fm best SAE = {max(fm_sub['sae_mean']):.3f}  "
              f"(Δ vs raw-EEG = {d_sae:+.3f}; FM-SAE − FM-PCA = {beats_pca:+.3f})")
        if d_sae > 0.03 and beats_pca > -0.02:
            print("THESIS SUPPORTED: FM substrate lifts cross-domain RSA beyond raw EEG,")
            print("not explained by the capacity-matched PCA control.")
        else:
            print("THESIS NOT SUPPORTED (reported honestly): FM substrate does not beat")
            print("raw-EEG / PCA on this data. The substrate is not the bottleneck here.")
    else:
        print("FM substrate not run (no weights) — raw-EEG cross-check only.")


if __name__ == "__main__":
    main()
