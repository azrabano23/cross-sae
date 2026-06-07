"""
Synthetic cross-domain data with *planted ground-truth matches*.

The point of this module is falsifiability: because we know which brain-side
features are truly associated with each model-side target feature, we can
measure the empirical FDR and power of the knockoff matching procedure and
check whether the advertised control actually holds. This is the sanity check
that must pass on synthetic data before any claim is made on real brain data.
"""
from __future__ import annotations

import numpy as np


def make_planted_matching(
    n: int = 3000,
    p_brain: int = 80,
    n_targets: int = 25,
    matches_per_target: int = 12,
    n_shared_factors: int = 6,
    signal: float = 2.2,
    noise: float = 1.0,
    rng: np.random.Generator | None = None,
):
    """Generate correlated, sparse, non-negative brain-side features plus a set
    of model-side target features, each with a known true-match set.

    Returns
    -------
    X_brain : (n, p_brain) non-negative brain-side feature matrix (covariates).
    targets : list of (y, true_idx) where y is (n,) and true_idx is an int array
              of the brain features genuinely driving y.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Shared latent factors induce realistic correlation among brain features
    # (knockoffs are only interesting when covariates are correlated).
    F = rng.standard_normal((n, n_shared_factors))
    loadings = rng.standard_normal((n_shared_factors, p_brain)) * 0.7
    base = F @ loadings + rng.standard_normal((n, p_brain))
    # Rectify to mimic the non-negative, sparse character of SAE latents.
    X_brain = np.maximum(base, 0.0)

    targets = []
    for _ in range(n_targets):
        true_idx = rng.choice(p_brain, size=matches_per_target, replace=False)
        w = rng.uniform(0.7, 1.3, size=matches_per_target)
        y = X_brain[:, true_idx] @ (signal * w)
        y = y + noise * rng.standard_normal(n)
        targets.append((y, np.sort(true_idx)))

    return X_brain, targets


def make_shared_stimulus(
    n: int = 3000,
    n_model_features: int = 25,
    p_brain: int = 80,
    brain_per_model: int = 12,
    signal: float = 2.2,
    noise: float = 1.0,
    rng: np.random.Generator | None = None,
):
    """Two SAE feature banks over the SAME stimuli with planted *distributed*
    cross-domain correspondences: each model feature maps to a SET of brain
    features (1-to-many), which is the biologically realistic regime — a single
    model concept is read out by a population of brain features.

    Built on make_planted_matching so the per-target signal matches the validated
    knockoff regime. Returns Z_model (the model SAE features), Z_brain (brain SAE
    features), and the set of true (model_idx, brain_idx) pairs.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    X_brain, targets = make_planted_matching(
        n=n, p_brain=p_brain, n_targets=n_model_features,
        matches_per_target=brain_per_model, signal=signal, noise=noise, rng=rng)

    Z_model = np.maximum(np.stack([y for y, _ in targets], axis=1), 0.0)  # (n, n_model_features)
    true_pairs = set()
    for m, (_, true_idx) in enumerate(targets):
        for j in true_idx:
            true_pairs.add((m, int(j)))
    return Z_model, X_brain, true_pairs
