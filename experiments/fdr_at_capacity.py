"""
EXPERIMENT — does the FDR cross-domain match recover above-chance pairs once the
SAE has enough capacity to preserve the shared structure (FINDINGS §7)?

The headline join (FINDINGS §5) was run at the WEAKEST point of the capacity curve
(k=8, RSA=0.061) and was null. §7 showed RSA climbs to 0.118 at k=64. This closes
the loop: re-run the exact FDR matcher + permutation null at k=8 vs k=64 on the
same real data, and report whether higher capacity yields above-chance matches.

Honest expectation: FDR feature-pair matching is much stricter than RSA, and the
EEG noise ceiling is only 0.214, so even k=64 may stay at/near chance. Either way
the result is reported with its permutation null, not tuned.

Real data only (THINGS-EEG2 sub-01, all sessions; pretrained ViT).
Run:  python experiments/fdr_at_capacity.py
Out:  results/fdr_at_capacity.png / .csv
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crosssae.sae import train_sae, encode_dataset
from crosssae.matching import cross_domain_match
from experiments.headline_model_brain import load_epochs, vit_features

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

CAPS = [8, 64]            # SAE top-k (d_hidden = 8*k); k=8 reproduces §5, k=64 is §7's recovered point
FDR_Q = 0.2
N_PERM = 15
MIN_ACTIVE = 0.05
# At n=200 images the knockoff filter needs candidates << n, so cap each side to
# its most-active features (otherwise d=512 yields >100 candidates and the filter
# is both ill-posed and slow — itself the honest reason feature-matching wants
# more stimuli than 200).
MAX_FEATS = 50


def live(Z):
    frac = (Z > 0).mean(0)
    idx = np.where(frac >= MIN_ACTIVE)[0]
    if len(idx) > MAX_FEATS:
        idx = idx[np.argsort(-frac[idx])[:MAX_FEATS]]   # keep the most-active
    return idx


def run_capacity(k, Xtr, trial_img, Vimg):
    d = 8 * k
    bsae = train_sae(torch.from_numpy(Xtr), d, k, seed=0, epochs=150)
    Ztr = encode_dataset(bsae, torch.from_numpy(Xtr)).numpy()
    Zb = np.stack([Ztr[trial_img == i].mean(0) for i in range(200)])
    msae = train_sae(torch.from_numpy(Vimg), d, k, seed=0, epochs=300)
    Zm = encode_dataset(msae, torch.from_numpy(Vimg)).numpy()
    Zm, Zb = Zm[:, live(Zm)], Zb[:, live(Zb)]

    M, stats = cross_domain_match(Zm, Zb, q=FDR_Q, seed=0)
    real = stats["total_matches"]
    null = []
    for pi in range(N_PERM):
        perm = np.random.default_rng(500 + pi).permutation(Zb.shape[0])
        _, sp = cross_domain_match(Zm, Zb[perm], q=FDR_Q, seed=0)
        null.append(sp["total_matches"])
    null = np.array(null)
    p = (1 + np.sum(null >= real)) / (1 + N_PERM)
    return dict(k=k, n_model=Zm.shape[1], n_brain=Zb.shape[1],
               real=real, null_mean=float(null.mean()), null_max=int(null.max()), p=float(p))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(0)
    print("Loading real EEG (all sessions) + ViT...")
    Xtr, trial_img, Xb_img, img_paths = load_epochs()
    Vimg = vit_features(img_paths)

    rows = []
    for k in CAPS:
        print(f"\n=== capacity k={k} (d={8*k}) ===")
        r = run_capacity(k, Xtr, trial_img, Vimg)
        print(f"  model feats {r['n_model']}, brain feats {r['n_brain']} | "
              f"real {r['real']} vs null {r['null_mean']:.1f} (max {r['null_max']}) -> p={r['p']:.3f}")
        rows.append(r)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r["real"] for r in rows], 0.36, color="#c62828", label="real matches")
    ax.bar(x + 0.18, [r["null_mean"] for r in rows], 0.36, color="#b0bec5", label="permutation null (mean)")
    for i, r in enumerate(rows):
        ax.text(i, max(r["real"], r["null_mean"]) + 0.3, f"p={r['p']:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"k={r['k']}\nRSA≈{'0.06' if r['k']==8 else '0.12'}" for r in rows])
    ax.set_ylabel(f"# FDR-significant model↔brain matches (q={FDR_Q})")
    ax.set_title("Does recovered SAE capacity yield above-chance FDR matches?\n"
                 "real ViT-SAE ↔ human-EEG-SAE (THINGS-EEG2, 200 shared images)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(RESULTS, "fdr_at_capacity.png")
    fig.savefig(out, dpi=150); print(f"\nSaved figure -> {out}")

    with open(os.path.join(RESULTS, "fdr_at_capacity.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["k", "n_model_feats", "n_brain_feats", "real_matches", "null_mean", "null_max", "p"])
        for r in rows:
            w.writerow([r["k"], r["n_model"], r["n_brain"], r["real"], f"{r['null_mean']:.2f}", r["null_max"], f"{r['p']:.3f}"])

    best = max(rows, key=lambda r: r["real"])
    print("\n=== VERDICT ===")
    if best["p"] < 0.05:
        print(f"Recovered capacity helps: k={best['k']} gives {best['real']} matches, p={best['p']:.3f} (above chance).")
    else:
        print(f"Even at recovered capacity, FDR matching stays at chance (best k={best['k']}: "
              f"{best['real']} vs null {best['null_mean']:.1f}, p={best['p']:.3f}). "
              f"RSA-level structure exists but is too diffuse for feature-pair matching at the EEG noise ceiling -> fMRI needed.")


if __name__ == "__main__":
    main()
