"""
EXPERIMENT — WHEN does the vision model's representation align with the brain?

The capacity ablation (FINDINGS §7) established that raw ViT<->EEG share real
representational structure (RSA up to the EEG noise ceiling). This experiment asks
a different, interpretable question: at what *post-stimulus latency* does that
shared structure emerge? If model<->brain alignment peaks at the object-recognition
window (~100-200 ms), the shared structure is genuine visual-object representation,
not a low-level or artifactual confound.

Method (time-resolved RSA, after Cichy et al. 2014):
  - For each sliding post-stimulus time window, build the 200x200 EEG RDM from the
    per-image response in that window, and correlate it (Spearman) with the fixed
    ViT RDM.
  - Per-window EEG noise ceiling (split-half, Spearman-Brown) bounds what is
    achievable. Per-window permutation test gives significance.

Real data only (THINGS-EEG2 sub-01, all sessions; pretrained ViT). RSVP SOA is
200 ms, so we analyze 0-200 ms (later windows are contaminated by the next image)
and say so.

Run:  python experiments/temporal_rsa.py
Out:  results/temporal_rsa.png / .csv
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.headline_model_brain import vit_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data", "things_eeg2")
SESSIONS = sorted(glob.glob(os.path.join(DATA, "raw-eeg/sub-01/ses-*/raw_eeg_test.npy")))

WIN_MS = 40                            # sliding window width
STEP_MS = 10                           # window step
T_RANGE = (0, 200)                     # analyze 0-200 ms (RSVP SOA = 200 ms)
N_PERM = 1000


def load_time_resolved():
    """Return image-averaged, time-resolved EEG: (200, 63, T) plus sfreq, and the
    per-trial tensor grouped by image for the split-half noise ceiling."""
    meta = np.load(os.path.join(DATA, "image_metadata.npy"), allow_pickle=True).item()
    files = meta["test_img_files"]
    by_base = {os.path.basename(p): p for p in glob.glob(os.path.join(DATA, "test_images", "**", "*.jpg"), recursive=True)}
    img_paths = [by_base.get(files[k]) for k in range(200)]

    T0, T1 = T_RANGE
    per_code = {c: [] for c in range(1, 201)}
    sf = None
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
        s0 = int(T0 * sf / 1000); s1 = int(T1 * sf / 1000)
        for o, c in zip(onset, codes):
            if o + s1 < X.shape[1]:
                per_code[int(c)].append(X[:, o + s0:o + s1])    # (63, T)
    trials = [np.array(per_code[c], dtype=np.float32) for c in range(1, 201)]
    avg = np.stack([t.mean(0) for t in trials])                 # (200, 63, T)
    return avg, trials, sf, img_paths


def rdm(F):
    Fz = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-8)
    return 1.0 - (Fz @ Fz.T) / F.shape[1]


def upper(M):
    return M[np.triu_indices(M.shape[0], k=1)]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(0)

    print("Loading time-resolved EEG (all sessions) + ViT...")
    avg, trials, sf, img_paths = load_time_resolved()
    T = avg.shape[2]
    Vrdm = rdm(vit_features(img_paths))
    v_upper = upper(Vrdm)
    print(f"  EEG {avg.shape}, sfreq {sf}")

    w = int(WIN_MS * sf / 1000); step = int(STEP_MS * sf / 1000)
    centers, rsa_t, ceil_t, p_t = [], [], [], []
    rng = np.random.default_rng(0)
    for start in range(0, T - w + 1, step):
        sl = slice(start, start + w)
        feat = avg[:, :, sl].mean(2)                            # (200, 63) mean amplitude
        Er = rdm(feat); e_upper = upper(Er)
        rho = spearmanr(v_upper, e_upper).correlation

        # split-half noise ceiling at this window
        A = np.zeros((200, 63)); B = np.zeros((200, 63))
        for i in range(200):
            tr = trials[i][:, :, sl].mean(2)
            idx = rng.permutation(len(tr)); h = len(idx) // 2
            A[i] = tr[idx[:h]].mean(0); B[i] = tr[idx[h:2*h]].mean(0)
        rel = spearmanr(upper(rdm(A)), upper(rdm(B))).correlation
        ceil = 2 * rel / (1 + rel) if rel > -1 else 0.0

        # permutation test (shuffle image labels of the EEG RDM)
        null = np.array([spearmanr(v_upper, upper(Er[np.ix_(p, p)])).correlation
                         for p in (rng.permutation(200) for _ in range(N_PERM))])
        pval = (1 + np.sum(null >= rho)) / (1 + N_PERM)

        c_ms = (start + w / 2) * 1000 / sf
        centers.append(c_ms); rsa_t.append(rho); ceil_t.append(ceil); p_t.append(pval)
        print(f"  t={c_ms:5.0f}ms  RSA={rho:+.3f}  ceiling={ceil:.3f}  p={pval:.3f}")

    centers = np.array(centers); rsa_t = np.array(rsa_t); ceil_t = np.array(ceil_t); p_t = np.array(p_t)
    peak = int(np.argmax(rsa_t))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(centers, 0, ceil_t, color="#c8e6c9", alpha=0.7, label="EEG noise ceiling")
    ax.plot(centers, rsa_t, "-o", color="#c62828", lw=2, ms=5, label="ViT↔EEG RSA")
    sig = p_t < 0.05
    ax.plot(centers[sig], rsa_t[sig], "o", color="#000", ms=7, mfc="none", label="p<0.05")
    ax.axvline(centers[peak], color="#1565c0", ls="--", lw=1.5,
               label=f"peak {centers[peak]:.0f}ms (RSA={rsa_t[peak]:.2f})")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("time post-stimulus (ms)"); ax.set_ylabel("cross-domain RSA (Spearman ρ)")
    ax.set_title("WHEN does the vision model align with the brain?\n"
                 "time-resolved ViT↔EEG RSA (THINGS-EEG2, 200 shared images)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(RESULTS, "temporal_rsa.png")
    fig.savefig(out, dpi=150); print(f"\nSaved figure -> {out}")

    with open(os.path.join(RESULTS, "temporal_rsa.csv"), "w", newline="") as f:
        wtr = csv.writer(f); wtr.writerow(["t_ms", "rsa", "noise_ceiling", "p"])
        for c, r, ce, p in zip(centers, rsa_t, ceil_t, p_t):
            wtr.writerow([f"{c:.0f}", f"{r:.4f}", f"{ce:.4f}", f"{p:.4f}"])
    print(f"VERDICT: ViT↔EEG alignment peaks at {centers[peak]:.0f} ms "
          f"(RSA={rsa_t[peak]:.3f}, p={p_t[peak]:.3f}); "
          f"{int(sig.sum())}/{len(centers)} windows significant at p<0.05.")


if __name__ == "__main__":
    main()
