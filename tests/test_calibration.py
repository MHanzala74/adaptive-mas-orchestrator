import os
import numpy as np
from mas.benchmarks.tasks import Task
from mas.calibration import shared_wrong_rate, task_stats, ratio_ci, per_task_pairs, fit_mock_corr, simulate_rate

T = [Task("a", "1+1", "2"), Task("b", "2+2", "4")]

def test_shared_wrong_rate_handcrafted():
    ans = [["3", "3", "2", "5"], ["4", "4", "4", "4"]]          # task a: wrong 3,3,5 -> 3 pairs, 1 same; task b all right
    rate, pairs = shared_wrong_rate(ans, T)
    assert pairs == 3 and abs(rate - 1 / 3) < 1e-9
    assert shared_wrong_rate([["3", "9"], ["5", "6"]], T)[0] == 0.0
    assert shared_wrong_rate([["3", "3"], ["5", "5"]], T)[0] == 1.0

def test_unparseable_answers_are_excluded():
    assert shared_wrong_rate([[None, "3", "3"], ["4", "4"]], T)[1] == 1

def test_task_stats_and_oracle():
    st = task_stats([["2", "3", "3"], ["4", "4", "5"]], T)
    assert st["acc_any_correct"] == 1.0 and st["acc_majority_vote"] == 0.5 and abs(st["acc_per_sample"] - 0.5) < 1e-9

def test_ratio_ci_brackets_estimate():
    num, den = np.array([1., 0, 2, 0, 1]), np.array([3., 3, 3, 3, 3])
    m, lo, hi = ratio_ci(num, den)
    assert lo <= m <= hi and abs(m - 4 / 15) < 1e-9

def test_simulated_rate_is_monotone_in_corr():
    r = [simulate_rate(c, n_tasks=200) for c in (0.0, 0.5, 1.0)]
    assert r[0] < r[1] < r[2] and r[0] < 0.15 and r[2] > 0.9

def test_fit_recovers_known_corr():
    for c in (0.3, 0.7):
        target = simulate_rate(c, n_tasks=300, seed=7)           # a different seed than the fit uses
        assert abs(fit_mock_corr(target)["mock_corr"] - c) <= 0.15

def test_end_to_end_synthetic_with_cache_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_CORR", "0.6")
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("rec", pathlib.Path("experiments/real_error_correlation.py"))
    rec = importlib.util.module_from_spec(spec); spec.loader.exec_module(rec)
    from mas.benchmarks.tasks import generate_synthetic
    r = rec.run(generate_synthetic(150, seed=500), k=5)
    assert r["shared_wrong"] is not None and 0.3 <= r["fit"]["mock_corr"] <= 0.9
