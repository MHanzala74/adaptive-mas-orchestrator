"""Log collection, router training and counterfactual-replay evaluation."""
from __future__ import annotations

import random

import numpy as np

from mas.router.bandit import LinUCBRouter
from mas.router.features import featurize
from mas.testbed import run_once

REF_TOKENS = 300.0


def reward(correct: bool | float, tokens: float, lam: float) -> float:
    return float(correct) - lam * tokens / REF_TOKENS


def collect_log(tasks, archs, seeds=(0,), budget=None) -> list[dict]:
    """Full-information log: every architecture is run on every task."""
    log = []
    for s in seeds:
        for i, t in enumerate(tasks):
            out = {}
            for a in archs:
                r = run_once(t, a, seed=s * 1_000_003 + i, budget=budget)
                out[a] = {"correct": int(r.correct), "tokens": r.tokens}
            log.append({"task_id": t.id, "seed": s, "question": t.question,
                        "difficulty": t.difficulty, "x": featurize(t.question).tolist(), "outcomes": out})
    return log


def fit_full_info(router: LinUCBRouter, log, lam: float) -> LinUCBRouter:
    for rec in log:
        x = np.array(rec["x"])
        for a, o in rec["outcomes"].items():
            router.update(a, x, reward(o["correct"], o["tokens"], lam), o["tokens"])
    return router


def fit_bandit(router: LinUCBRouter, log, lam: float, seed: int = 0, epochs: int = 3) -> LinUCBRouter:
    """Partial feedback: only the chosen arm's outcome is observed (online learning)."""
    rng = random.Random(seed)
    for _ in range(epochs):
        order = log[:]
        rng.shuffle(order)
        for rec in order:
            x = np.array(rec["x"])
            a = router.select(x, explore=True)
            o = rec["outcomes"][a]
            router.update(a, x, reward(o["correct"], o["tokens"], lam), o["tokens"])
    return router


def evaluate_policy(choose, log, lam: float) -> dict:
    """Replay a policy on a held-out full-information log."""
    n = len(log)
    c = t = r = 0.0
    picks: dict[str, int] = {}
    for rec in log:
        a = choose(rec)
        o = rec["outcomes"][a]
        c += o["correct"]; t += o["tokens"]; r += reward(o["correct"], o["tokens"], lam)
        picks[a] = picks.get(a, 0) + 1
    return {"acc": c / n, "tokens": t / n, "reward": r / n, "picks": picks}


def oracle_choice(lam: float):
    return lambda rec: max(rec["outcomes"], key=lambda a: reward(rec["outcomes"][a]["correct"], rec["outcomes"][a]["tokens"], lam))
