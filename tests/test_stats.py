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

def test_cluster_bootstrap_is_wider_when_clusters_are_correlated():
    rng = np.random.default_rng(1)
    task_effect = rng.normal(0, 1, 40)
    v = np.concatenate([[e] * 3 for e in task_effect]) + rng.normal(0, 0.05, 120)   # 3 near-copies per task
    cl = np.repeat(np.arange(40), 3)
    iid, clu = bootstrap_ci(v), bootstrap_ci(v, clusters=cl)
    assert (clu[2] - clu[1]) > 1.4 * (iid[2] - iid[1])
    assert abs(clu[0] - iid[0]) < 1e-9

def test_cluster_ci_equals_iid_when_clusters_singletons():
    v = np.random.default_rng(2).normal(size=50)
    a, b = bootstrap_ci(v), bootstrap_ci(v, clusters=np.arange(50))
    assert abs((a[2] - a[1]) - (b[2] - b[1])) < 0.1
