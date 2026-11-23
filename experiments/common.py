"""Shared helpers for baseline / Pareto experiments."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from mas.architectures.impl import BASELINES, REGISTRY
from mas.benchmarks.tasks import generate_synthetic
from mas.router.bandit import LinUCBRouter
from mas.router.train import collect_log, fit_full_info

CORE = list(REGISTRY)                 # the router's pool: single, debate, planner_worker, solver_critic, debate_judge
BASE = list(BASELINES)                # comparison methods: sc3, sc5, sc7, adaptive_sc, cascade
ALLM = CORE + BASE

# Three simulated environments, fixed in advance (not chosen after seeing results):
ENVS = {
    "homogeneous, independent errors": {"MOCK_HETERO": "0", "MOCK_CORR": "0"},
    "heterogeneous, independent errors": {"MOCK_HETERO": "1", "MOCK_CORR": "0"},
    "heterogeneous, correlated errors (0.5)": {"MOCK_HETERO": "1", "MOCK_CORR": "0.5"},
}
LAM = 0.3


def set_env(name: str) -> None:
    os.environ.update(ENVS[name])


def restrict(log, archs):
    return [{**r, "outcomes": {a: r["outcomes"][a] for a in archs}} for r in log]


def build_logs(budget=None, n_train=200, n_test=200, seeds=(10, 11, 12)):
    train = generate_synthetic(n_train, seed=1)
    test = generate_synthetic(n_test, seed=100)
    return (collect_log(train, ALLM, seeds=(0, 1), budget=budget),
            collect_log(test, ALLM, seeds=seeds, budget=budget))


def train_router(log_train, archs, lam=LAM):
    return fit_full_info(LinUCBRouter(archs, alpha=0.0), restrict(log_train, archs), lam)


def router_policy(router, budget=None):
    return lambda rec: router.select(np.array(rec["x"]), budget, explore=False)
