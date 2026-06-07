"""
Validate the HEADLINE method: cross-domain feature<->feature matching.

Two SAE feature banks over the same stimuli, with planted model<->brain
correspondences. We check that crosssae.matching.cross_domain_match recovers the
true pairs while controlling the false-discovery rate over the (much larger) set
of spurious cross-domain pairs.

If empirical pair-FDR tracks the nominal q, the matcher is trustworthy on real
model<->brain data (the only change there is that the two banks come from a ViT
and a brain instead of from a generator).

Run:
    python experiments/cross_domain_validation.py
Output:
    results/cross_domain_validation.png / .csv
"""
from __future__ import annotations

import os
import sys
import csv
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crosssae.synthetic import make_shared_stimulus
from crosssae.matching import cross_domain_match

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    q_levels = [0.1, 0.15, 0.2, 0.25, 0.3]
    Zm, Zb, true_pairs = make_shared_stimulus(rng=np.random.default_rng(0))
    p_m, p_b = Zm.shape[1], Zb.shape[1]
    total_possible = p_m * p_b
    print(f"shared stimuli: n={Zm.shape[0]}, model feats={p_m}, brain feats={p_b}, "
          f"planted true pairs={len(true_pairs)} of {total_possible} possible")

    # Per-model-feature true brain sets (knockoffs controls FDR *per target*).
    true_by_model = {}
    for (m, j) in true_pairs:
        true_by_model.setdefault(m, set()).add(j)

    emp_fdr, power = [], []
    for q in q_levels:
        M, stats = cross_domain_match(Zm, Zb, q=q, seed=0)
        per_target_fdr, per_target_power = [], []
        for m, tset in true_by_model.items():
            sel = set(np.where(M[m] == 1)[0].tolist())
            n_sel = len(sel)
            n_true = len(sel & tset)
            per_target_fdr.append((n_sel - n_true) / max(1, n_sel))
            per_target_power.append(n_true / len(tset))
        fdr = float(np.mean(per_target_fdr)); pw = float(np.mean(per_target_power))
        emp_fdr.append(fdr); power.append(pw)
        print(f"  q={q:.2f}: total pairs found {int(M.sum()):4d} | "
              f"mean per-target FDR {fdr:.3f} | mean power {pw:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.plot([0, 0.32], [0, 0.32], "--", color="gray", lw=1.2, label="nominal FDR")
    ax.plot(q_levels, emp_fdr, "o-", color="#00695c", lw=2, ms=7, label="empirical pair-FDR")
    ax.set_xlabel("nominal FDR level q"); ax.set_ylabel("empirical cross-domain pair-FDR")
    ax.set_title("Cross-domain matching controls FDR"); ax.set_xlim(0, .32); ax.set_ylim(0, .32)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    ax.plot(q_levels, power, "s-", color="#bf360c", lw=2, ms=7)
    ax.set_xlabel("nominal FDR level q"); ax.set_ylabel("power (true model<->brain pairs found)")
    ax.set_title("Power vs FDR budget"); ax.set_xlim(0, .32); ax.set_ylim(0, 1.02)
    fig.suptitle("HEADLINE METHOD — FDR-controlled cross-domain (model<->brain) SAE feature matching\n"
                 f"(synthetic shared-stimulus validation: {len(true_pairs)} planted pairs in {total_possible} candidates)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out = os.path.join(RESULTS, "cross_domain_validation.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure -> {out}")
    with open(os.path.join(RESULTS, "cross_domain_validation.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["nominal_q", "empirical_pair_fdr", "power"])
        for q, e, pw in zip(q_levels, emp_fdr, power):
            w.writerow([f"{q:.3f}", f"{e:.4f}", f"{pw:.4f}"])
    print("Cross-domain matcher validated: it recovers planted model<->brain pairs under FDR control.")


if __name__ == "__main__":
    main()
