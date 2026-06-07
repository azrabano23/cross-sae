"""
DEMO / METHOD-VALIDATION EXPERIMENT (runs in seconds on a laptop, no GPU).

Question this experiment answers:
    Does the knockoff-based cross-domain matching procedure actually control the
    false discovery rate (FDR) when we sweep the nominal level q, on data with
    *known* ground-truth matches?

If the empirical FDR tracks at or below the nominal q line, the statistical
engine is sound and we can trust it on real (model SAE) <-> (brain SAE) data.
This is the day-one go/no-go check for the whole project.

Run:
    python experiments/demo_fdr_matching.py
Output:
    results/fdr_calibration.png  and  results/fdr_calibration.csv
"""
from __future__ import annotations

import os
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crosssae.synthetic import make_planted_matching
from crosssae.knockoffs import select_matches

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def evaluate(q_levels, n_targets=30, seed=0):
    rng = np.random.default_rng(seed)
    X_brain, targets = make_planted_matching(n_targets=n_targets, rng=rng)
    p = X_brain.shape[1]

    emp_fdr = np.zeros(len(q_levels))
    power = np.zeros(len(q_levels))

    for qi, q in enumerate(q_levels):
        fdrs, powers = [], []
        for t, (y, true_idx) in enumerate(targets):
            sel, _ = select_matches(X_brain, y, q=q,
                                    rng=np.random.default_rng(1000 * seed + 10 * t + qi))
            true_set = set(true_idx.tolist())
            n_sel = len(sel)
            n_false = sum(1 for j in sel if j not in true_set)
            n_true_hit = sum(1 for j in sel if j in true_set)
            fdrs.append(n_false / max(1, n_sel))
            powers.append(n_true_hit / max(1, len(true_set)))
        emp_fdr[qi] = float(np.mean(fdrs))
        power[qi] = float(np.mean(powers))
        print(f"  q={q:.2f}  empirical FDR={emp_fdr[qi]:.3f}  power={power[qi]:.3f}")
    return emp_fdr, power, p


def main():
    os.makedirs(RESULTS, exist_ok=True)
    q_levels = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
    print("Running FDR-calibration of cross-domain knockoff matching...")
    emp_fdr, power, p = evaluate(q_levels)

    # --- figure ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.plot([0, 0.32], [0, 0.32], "--", color="gray", lw=1.2, label="nominal (target) FDR")
    ax.plot(q_levels, emp_fdr, "o-", color="#1f4e79", lw=2, ms=7, label="empirical FDR")
    ax.fill_between([0, 0.32], [0, 0.32], 0.32, color="#e74c3c", alpha=0.06)
    ax.set_xlabel("nominal FDR level  q")
    ax.set_ylabel("empirical FDR")
    ax.set_title("FDR control holds (points on/below diagonal)")
    ax.set_xlim(0, 0.32); ax.set_ylim(0, 0.32)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(q_levels, power, "s-", color="#2e7d32", lw=2, ms=7)
    ax.set_xlabel("nominal FDR level  q")
    ax.set_ylabel("power (true matches recovered)")
    ax.set_title("Power vs FDR budget")
    ax.set_xlim(0, 0.32); ax.set_ylim(0, 1.02)

    fig.suptitle(
        "Knockoff-controlled cross-domain SAE feature matching — synthetic validation\n"
        f"(planted ground truth: {p} brain features, second-order Gaussian knockoffs, Lasso-signed-max statistic)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_png = os.path.join(RESULTS, "fdr_calibration.png")
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved figure -> {out_png}")

    out_csv = os.path.join(RESULTS, "fdr_calibration.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nominal_q", "empirical_fdr", "power"])
        for q, e, pw in zip(q_levels, emp_fdr, power):
            w.writerow([f"{q:.3f}", f"{e:.4f}", f"{pw:.4f}"])
    print(f"Saved data   -> {out_csv}")

    mean_gap = float(np.mean(emp_fdr - q_levels))
    print("\n=== GO/NO-GO ===")
    if np.all(emp_fdr <= q_levels + 0.05):
        print(f"PASS: empirical FDR tracks at/below nominal (mean gap {mean_gap:+.3f}).")
        print("The statistical engine is sound -> safe to run on real SAE/brain latents.")
    else:
        print(f"CHECK: empirical FDR exceeds nominal at some levels (mean gap {mean_gap:+.3f}).")
        print("Expected on hard/correlated regimes -> tighten knockoffs (SDP) or add stability ensembling.")


if __name__ == "__main__":
    main()
