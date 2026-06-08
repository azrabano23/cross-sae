"""
TARJUMAN — the actual FM-substrate test with REAL pretrained CBraMod (ICLR 2025).

Tests the central hypothesis (TARJUMAN.md): does training the brain-side SAE on a
pretrained EEG foundation-model latent space raise cross-domain ViT<->brain RSA above
a raw-EEG substrate (and above a capacity-matched PCA control), toward the EEG noise
ceiling?

DESIGN — identical windows, substrate is the only variable:
  Both arms use the SAME 1-second single-trial windows (CBraMod's patch is hard-locked
  to 200 pts = 1 s @ 200 Hz, so 1 s is the minimum valid input). Within those windows:
    raw-EEG : bin each 1 s window to (63 ch x BINS)                -> brain SAE
    eeg-fm  : feed each 1 s window through pretrained CBraMod -> 200-d latent -> brain SAE
  Model side (ViT-SAE) and capacity are held fixed. Per arm we report cross-domain RSA
  (SAE-feature and dense-PCA), the substrate's split-half noise ceiling, vs the shared
  ViT RDM.

HONESTY CAVEATS (loud on purpose — read before trusting any number):
  1. RSVP CONTAMINATION. THINGS-EEG2 SOA is 200 ms, but CBraMod needs a 1 s window.
     Each 1 s window therefore spans ~5 successive images, so per-image FM features are
     contaminated by the next ~4 stimuli. THINGS-EEG2 is a POOR testbed for a 1 s-patch
     EEG foundation model; a clean test needs slower-presentation EEG or fMRI. A null
     here does NOT cleanly refute the FM-substrate hypothesis — it may reflect this
     mismatch. We run it anyway, transparently, because it is the available shared-
     stimulus data and the result is still informative about the mismatch itself.
  2. PREPROCESSING MISMATCH. CBraMod's pretraining preprocessing is not reproduced
     exactly here (per-channel z-score + per-patch standardization is a generic stand-in),
     so the FM operates somewhat off-distribution. Another reason to read a null cautiously.

Run:
    CBRAMOD_REPO=/path/to/CBraMod \
    TARJUMAN_FM_WEIGHTS=/path/to/CBraMod/pretrained_weights/pretrained_weights.pth \
        python experiments/tarjuman_fm_cbramod.py
Out:
    results/tarjuman_fm_cbramod.png / .csv
"""
from __future__ import annotations

import os
import sys
import csv
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crosssae.sae import train_sae, encode_dataset
from crosssae.backbone import get_backbone, FMConfig
from experiments.headline_model_brain import vit_features, DATA, SESSIONS
from experiments.tarjuman_fm_join import rdm, rsa, rsa_p, eeg_noise_ceiling
import glob

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

WIN_MS = (0, 1000)        # 1 s window (CBraMod minimum). Spans ~5 RSVP images — see caveat 1.
BINS = 20                 # raw-EEG arm: time bins for the 1 s window (63 x 20 = 1260-d)
K, SEEDS = 32, [0, 1]     # near the §7 capacity plateau; multi-seed for SAE non-determinism
D_HID = 8 * K


def load_trials_1s():
    """Single-trial 1 s windows. Returns raw (n,63,T@1000Hz) for the FM, binned
    (n,63*BINS) for the raw arm, trial->image index, and the 200 image paths."""
    meta = np.load(os.path.join(DATA, "image_metadata.npy"), allow_pickle=True).item()
    files = meta["test_img_files"]
    by_base = {os.path.basename(p): p
               for p in glob.glob(os.path.join(DATA, "test_images", "**", "*.jpg"), recursive=True)}
    img_paths = [by_base.get(files[k]) for k in range(200)]

    raw, binned, trial_img = [], [], []
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
        s0 = int(WIN_MS[0] * sf / 1000); s1 = int(WIN_MS[1] * sf / 1000)
        edges = np.linspace(s0, s1, BINS + 1).astype(int)
        for o, c in zip(onset, codes):
            if o + s1 >= X.shape[1]:
                continue
            w = X[:, o + s0:o + s1]                       # (63, T)
            raw.append(w.astype(np.float32))
            binned.append(np.stack([w[:, edges[b]-s0:edges[b+1]-s0].mean(1)
                                    for b in range(BINS)], axis=1).reshape(-1))
            trial_img.append(int(c) - 1)
    return (np.asarray(raw, dtype=np.float32), np.asarray(binned, dtype=np.float32),
            np.asarray(trial_img), img_paths)


def arm(name, Brain_trials, trial_img, R_vit_sae, R_vit_pca, Vimg):
    Bt = torch.from_numpy(Brain_trials.astype(np.float32))
    rs = []
    for s in SEEDS:
        bsae = train_sae(Bt, D_HID, K, seed=s, epochs=120)
        Ztr = encode_dataset(bsae, Bt).numpy()
        Zb = np.stack([Ztr[trial_img == i].mean(0) for i in range(200)])
        rs.append(rsa(R_vit_sae[s], rdm(Zb)))
    sae_m, sae_s = float(np.mean(rs)), float(np.std(rs))
    Brain_img = np.stack([Brain_trials[trial_img == i].mean(0) for i in range(200)])
    Pe = PCA(n_components=min(K, Brain_img.shape[1]), random_state=0).fit_transform(Brain_img)
    pca_r = rsa(R_vit_pca, rdm(Pe))
    rep_r, rep_p = rsa_p(rdm(Vimg), rdm(Brain_img))
    nc_m, nc_s = eeg_noise_ceiling(Brain_trials, trial_img)
    print(f"  [{name}] SAE {sae_m:+.3f}+/-{sae_s:.3f} | PCA {pca_r:+.3f} | "
          f"rep-RSA {rep_r:+.3f}(p={rep_p:.3f}) | ceiling {nc_m:.3f}")
    return dict(name=name, sae_m=sae_m, sae_s=sae_s, pca=pca_r,
                rep_r=rep_r, rep_p=rep_p, nc=nc_m, nc_s=nc_s)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(0)
    weights = os.environ.get("TARJUMAN_FM_WEIGHTS")
    if not weights:
        print("No TARJUMAN_FM_WEIGHTS — this script REQUIRES real CBraMod weights.")
        print("  CBRAMOD_REPO=/path/to/CBraMod TARJUMAN_FM_WEIGHTS=.../pretrained_weights.pth \\")
        print("    python experiments/tarjuman_fm_cbramod.py")
        return

    print(f"Loading 1 s single-trial windows {WIN_MS} ms (spans ~5 RSVP images — see caveat)...")
    Xraw, Xbin, trial_img, img_paths = load_trials_1s()
    print(f"  {Xraw.shape[0]} trials | raw {Xraw.shape} | binned {Xbin.shape}")
    Vimg = vit_features(img_paths)
    Vimg_t = torch.from_numpy(Vimg)

    # shared model side
    R_vit_sae = {}
    for s in SEEDS:
        msae = train_sae(Vimg_t, D_HID, K, seed=s, epochs=200)
        R_vit_sae[s] = rdm(encode_dataset(msae, Vimg_t).numpy())
    R_vit_pca = rdm(PCA(n_components=K, random_state=0).fit_transform(Vimg))

    rows = []
    print("\n[arm: raw-EEG @1s]")
    rows.append(arm("raw-EEG", Xbin, trial_img, R_vit_sae, R_vit_pca, Vimg))

    print("[arm: eeg-fm (CBraMod)]  encoding trials through the foundation model...")
    fm = get_backbone("cbramod", cfg=FMConfig(weights_path=weights))
    Xfm = fm.transform(Xraw)                                  # (n_trials, 200)
    print(f"  CBraMod latents {Xfm.shape}")
    rows.append(arm("eeg-fm", Xfm, trial_img, R_vit_sae, R_vit_pca, Vimg))

    # --- report ---------------------------------------------------------------
    raw_row = rows[0]; fm_row = rows[1]
    d_sae = fm_row["sae_m"] - raw_row["sae_m"]
    beats_pca = fm_row["sae_m"] - fm_row["pca"]
    print("\n" + "=" * 70)
    print(f"raw-EEG@1s  SAE={raw_row['sae_m']:+.3f}  (ceiling {raw_row['nc']:.3f})")
    print(f"eeg-fm      SAE={fm_row['sae_m']:+.3f}  (ceiling {fm_row['nc']:.3f})")
    print(f"Δ(FM − raw) = {d_sae:+.3f} | (FM-SAE − FM-PCA) = {beats_pca:+.3f}")
    if d_sae > 0.03 and beats_pca > -0.02:
        verdict = "THESIS SUPPORTED on THINGS-EEG2 (FM substrate lifts cross-domain RSA)."
    else:
        verdict = ("THESIS NOT SUPPORTED on THINGS-EEG2 (FM does not beat raw/PCA here). "
                   "Confounded by RSVP contamination + preprocessing mismatch — see caveats; "
                   "needs slower-SOA EEG or fMRI for a clean test.")
    print(verdict)
    print("=" * 70)

    # --- figure ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 5))
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [r["sae_m"] for r in rows], 0.4, yerr=[r["sae_s"] for r in rows],
           capsize=4, color="#6a1b9a", label=f"SAE (k={K})")
    ax.bar(x + 0.2, [r["pca"] for r in rows], 0.4, color="#b39ddb", label=f"PCA (k={K})")
    for i, r in enumerate(rows):
        ax.hlines(r["nc"], i - 0.4, i + 0.4, color="#2e7d32", lw=2,
                  label="substrate noise ceiling" if i == 0 else None)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{n}\n@1s window" for n in names])
    ax.set_ylabel("cross-domain RSA (ViT RDM vs brain RDM)")
    ax.set_title("TARJUMAN — real CBraMod substrate vs raw EEG (THINGS-EEG2, k=%d)\n"
                 "identical 1 s windows; CAVEAT: 200 ms RSVP contaminates 1 s FM inputs" % K)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(RESULTS, "tarjuman_fm_cbramod.png")
    fig.savefig(out, dpi=150); print(f"\nSaved figure -> {out}")

    with open(os.path.join(RESULTS, "tarjuman_fm_cbramod.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "sae_rsa_mean", "sae_rsa_std", "pca_rsa", "rep_rsa",
                    "rep_p", "noise_ceiling"])
        for r in rows:
            w.writerow([r["name"], f"{r['sae_m']:.4f}", f"{r['sae_s']:.4f}",
                        f"{r['pca']:.4f}", f"{r['rep_r']:.4f}", f"{r['rep_p']:.4f}",
                        f"{r['nc']:.4f}"])
        w.writerow(["VERDICT", verdict, "", "", "", "", ""])


if __name__ == "__main__":
    main()
