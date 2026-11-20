"""Small stats helpers: bootstrap CIs and paired bootstrap differences."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, (n_boot, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_diff_ci(a, b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """CI for mean(a - b) on paired observations; significant if CI excludes 0."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bootstrap_ci(d, n_boot, alpha, seed)


def fmt(ci) -> str:
    m, lo, hi = ci
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
