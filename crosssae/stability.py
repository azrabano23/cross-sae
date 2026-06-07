"""
SAE seed-stability layer.

This module exists to neutralize the single strongest objection to the whole
project. The 2025 literature (arXiv:2501.16615; arXiv:2505.20254) shows SAEs are
seed-dependent: identical data + architecture, different init -> as little as
~30% feature overlap, with rare features the most unstable. If we matched ONE
model SAE to ONE brain SAE, a "discovered" model<->brain correspondence could be
an artifact of a single lucky seed.

Defense: train an ENSEMBLE of SAEs per domain across seeds, keep only features
that are *reproducible* across seeds (high pairwise matching correlation, PW-MCC),
and run cross-domain matching only on this stable core. We report the stability
spectrum so the reader sees exactly how much was discarded.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_dictionaries(Za: np.ndarray, Zb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian-match features of two feature-activation matrices by correlation.

    Za, Zb : (n_samples, d) activation matrices from two SAEs over the SAME inputs.
    Returns (col_b_for_a, corr) : for each feature in Za, the matched Zb index and
    the Pearson correlation of the matched pair.
    """
    A = (Za - Za.mean(0)) / (Za.std(0) + 1e-8)
    B = (Zb - Zb.mean(0)) / (Zb.std(0) + 1e-8)
    corr = (A.T @ B) / A.shape[0]                # (d_a, d_b) cross-correlation
    # Maximize total matched correlation == minimize negative correlation.
    row, col = linear_sum_assignment(-corr)
    matched_corr = corr[row, col]
    order = np.argsort(row)
    return col[order], matched_corr[order]


def stability_scores(feature_mats: list[np.ndarray]) -> np.ndarray:
    """Pairwise-matched correlation consistency (PW-MCC) per feature.

    feature_mats : list of (n_samples, d) activation matrices, one per seed,
                   all over the SAME inputs.
    Returns a (d,) array: for each feature of the first SAE, its mean matched
    correlation to the corresponding feature across all other seeds. High = the
    feature reliably reappears regardless of initialization.
    """
    ref = feature_mats[0]
    d = ref.shape[1]
    accum = np.zeros(d)
    for other in feature_mats[1:]:
        _, corr = match_dictionaries(ref, other)
        accum += np.abs(corr)
    return accum / max(1, len(feature_mats) - 1)


def stable_core(feature_mats: list[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """Indices of features whose PW-MCC >= threshold (the reproducible core)."""
    scores = stability_scores(feature_mats)
    return np.where(scores >= threshold)[0]
