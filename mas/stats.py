"""Small stats helpers: bootstrap CIs and paired bootstrap differences."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0, clusters=None):
    """Percentile bootstrap CI for the mean.

    clusters: optional cluster label per observation (e.g. task id). When given, whole
    clusters are resampled, because repeated runs of the same task (several seeds,
    several methods) are NOT independent; iid resampling would give intervals that are
    too narrow. The statistic is the ratio estimator sum(cluster sums)/sum(cluster sizes).
    """
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if clusters is None:
        means = rng.choice(v, (n_boot, len(v)), replace=True).mean(axis=1)
    else:
        _, inv = np.unique(np.asarray(clusters), return_inverse=True)
        k = inv.max() + 1
        sums = np.bincount(inv, weights=v, minlength=k)
        cnts = np.bincount(inv, minlength=k).astype(float)
        idx = rng.integers(0, k, (n_boot, k))
        means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_diff_ci(a, b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0, clusters=None):
    """CI for mean(a - b) on paired observations; significant if CI excludes 0."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bootstrap_ci(d, n_boot, alpha, seed, clusters)


def fmt(ci) -> str:
    m, lo, hi = ci
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
