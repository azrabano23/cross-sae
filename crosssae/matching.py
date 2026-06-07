"""
Cross-domain feature<->feature matching (the headline method).

Given two SAE feature-activation matrices over the SAME stimuli — one from a
model (Z_model: n_stimuli x p_model) and one from the brain (Z_brain:
n_stimuli x p_brain) — find, for each model feature, the brain features that are
genuinely associated with it, with FALSE-DISCOVERY-RATE control.

This is exactly the per-target knockoff procedure from crosssae.knockoffs, run
once per model feature with the brain features as candidates (or vice-versa). The
output is a sparse, significance-tested model<->brain correspondence matrix M,
where M[i, j] = 1 means "model feature i and brain feature j are matched at FDR q".

Optional stability gating (crosssae.stability) restricts both sides to their
seed-reproducible cores first, so a reported match cannot be an artifact of a
single SAE initialization.
"""
from __future__ import annotations

import numpy as np

from .knockoffs import select_matches


def cross_domain_match(Z_a: np.ndarray, Z_b: np.ndarray, q: float = 0.2,
                       direction: str = "a->b", seed: int = 0, verbose: bool = False):
    """FDR-controlled matching between two SAE feature banks over shared stimuli.

    Parameters
    ----------
    Z_a, Z_b : (n_stimuli, p_a) and (n_stimuli, p_b) feature-activation matrices,
               row-aligned to the SAME stimuli.
    q : target FDR level for each per-target knockoff run.
    direction : "a->b" treats each column of Z_a as a target and the columns of
                Z_b as candidates (selects the b-features matched to each a-feature).
    seed : base RNG seed.

    Returns
    -------
    M : (p_a, p_b) binary correspondence matrix (1 = matched at FDR q).
    stats : dict with per-target discovery counts and matched correlations.
    """
    if direction == "b->a":
        M_t, stats = cross_domain_match(Z_b, Z_a, q=q, direction="a->b", seed=seed, verbose=verbose)
        return M_t.T, stats

    n, p_a = Z_a.shape
    p_b = Z_b.shape[1]
    M = np.zeros((p_a, p_b), dtype=np.int8)
    counts = np.zeros(p_a, dtype=int)

    # Only bother with targets that actually vary (a dead feature has no signal).
    live_targets = [i for i in range(p_a) if Z_a[:, i].std() > 1e-8]
    for i in live_targets:
        y = Z_a[:, i]
        sel, _ = select_matches(Z_b, y, q=q, rng=np.random.default_rng(seed * 100003 + i))
        M[i, sel] = 1
        counts[i] = len(sel)
        if verbose and len(sel):
            print(f"  model feature {i:4d} -> {len(sel)} brain features")

    stats = {
        "n_targets": len(live_targets),
        "n_matched_targets": int((counts > 0).sum()),
        "total_matches": int(M.sum()),
        "matches_per_target": counts,
    }
    return M, stats


def matched_pairs(M: np.ndarray, Z_a: np.ndarray, Z_b: np.ndarray):
    """Return the list of matched (a_idx, b_idx, correlation) triples, sorted by
    absolute correlation — the human-readable view of the correspondence matrix."""
    A = (Z_a - Z_a.mean(0)) / (Z_a.std(0) + 1e-8)
    B = (Z_b - Z_b.mean(0)) / (Z_b.std(0) + 1e-8)
    pairs = []
    ai, bi = np.where(M == 1)
    for a, b in zip(ai, bi):
        r = float((A[:, a] * B[:, b]).mean())
        pairs.append((int(a), int(b), r))
    pairs.sort(key=lambda t: -abs(t[2]))
    return pairs
