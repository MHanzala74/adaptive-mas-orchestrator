import numpy as np
from mas.stats import bootstrap_ci, paired_diff_ci

def test_ci_contains_mean_and_shrinks():
    rng = np.random.default_rng(0)
    small = bootstrap_ci(rng.normal(1, 1, 30)); big = bootstrap_ci(rng.normal(1, 1, 3000))
    assert small[1] <= small[0] <= small[2]
    assert (big[2] - big[1]) < (small[2] - small[1])

def test_paired_diff_detects_shift():
    a = np.ones(100); b = np.zeros(100)
    m, lo, hi = paired_diff_ci(a, b)
    assert lo > 0.9
