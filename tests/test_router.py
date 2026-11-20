import numpy as np
from mas.benchmarks.tasks import generate_synthetic, split
from mas.router.bandit import LinUCBRouter
from mas.router.features import featurize, DIM
from mas.router.train import collect_log, fit_full_info, evaluate_policy, oracle_choice

ARCHS = ["single", "debate", "planner_worker"]

def test_features_shape_and_range():
    x = featurize("(3 + 4) * 5 - 2")
    assert x.shape == (DIM,) and x.min() >= 0 and x.max() <= 1.0

def test_router_beats_worst_and_near_best_fixed():
    lam = 0.5
    tr, te = split(generate_synthetic(160, seed=7))
    log_tr = collect_log(tr, ARCHS, seeds=(0, 1))
    log_te = collect_log(te, ARCHS, seeds=(2,))
    r = fit_full_info(LinUCBRouter(ARCHS, alpha=0.0), log_tr, lam)
    learned = evaluate_policy(lambda rec: r.select(np.array(rec["x"]), explore=False), log_te, lam)
    fixed = {a: evaluate_policy(lambda rec, a=a: a, log_te, lam)["reward"] for a in ARCHS}
    oracle = evaluate_policy(oracle_choice(lam), log_te, lam)["reward"]
    assert learned["reward"] >= min(fixed.values())
    assert learned["reward"] >= max(fixed.values()) - 0.03
    assert oracle >= learned["reward"] - 1e-9

def test_budget_filter():
    r = LinUCBRouter(ARCHS)
    r.update("single", np.ones(DIM), 0.5, 40); r.update("debate", np.ones(DIM), 0.9, 300)
    assert r.select(np.ones(DIM), budget=100, explore=False) != "debate"
