"""Measure how correlated a model's errors are, and map it onto the mock's `MOCK_CORR` knob.

Key statistic: among pairs of WRONG samples drawn for the same task, the fraction that give the
SAME wrong answer ("shared-wrong rate"). Independent noise gives a rate near 0; a shared
misconception gives a high rate.
"""
from __future__ import annotations

import os
from collections import Counter

import numpy as np

from mas.benchmarks.tasks import Task, check, generate_synthetic
from mas.llm import MockLLM, parse_answer

GRID = [round(0.05 * i, 2) for i in range(21)]


def _norm(a: str | None):
    try:
        return None if a is None else round(float(a), 6)
    except ValueError:
        return None


def per_task_pairs(answers: list[list[str | None]], tasks: list[Task]):
    """Per task: (# wrong pairs that agree, # wrong pairs). Unparseable answers are excluded."""
    num, den = [], []
    for ans, t in zip(answers, tasks):
        wrong = [_norm(a) for a in ans if a is not None and not check(t, a) and _norm(a) is not None]
        pairs = len(wrong) * (len(wrong) - 1) // 2
        same = sum(c * (c - 1) // 2 for c in Counter(wrong).values())
        num.append(same); den.append(pairs)
    return np.array(num, float), np.array(den, float)


def shared_wrong_rate(answers, tasks) -> tuple[float, int]:
    num, den = per_task_pairs(answers, tasks)
    return (float(num.sum() / den.sum()) if den.sum() else float("nan")), int(den.sum())


def ratio_ci(num, den, n_boot: int = 2000, seed: int = 0):
    """Cluster (task) bootstrap CI for sum(num)/sum(den)."""
    rng = np.random.default_rng(seed)
    k = len(num)
    idx = rng.integers(0, k, (n_boot, k))
    d = den[idx].sum(axis=1)
    est = np.where(d > 0, num[idx].sum(axis=1) / np.where(d > 0, d, 1), np.nan)
    return float(num.sum() / den.sum()), float(np.nanquantile(est, 0.025)), float(np.nanquantile(est, 0.975))


def task_stats(answers, tasks) -> dict:
    n = len(tasks)
    samp = np.mean([[check(t, a) for a in ans] for ans, t in zip(answers, tasks)])
    maj = np.mean([check(t, Counter([a for a in ans if a is not None]).most_common(1)[0][0] if any(a is not None for a in ans) else None)
                   for ans, t in zip(answers, tasks)])
    orc = np.mean([any(check(t, a) for a in ans) for ans, t in zip(answers, tasks)])
    return {"n_tasks": n, "k": len(answers[0]), "acc_per_sample": float(samp), "acc_majority_vote": float(maj), "acc_any_correct": float(orc)}


def simulate_rate(c: float, k: int = 5, n_tasks: int = 300, seed: int = 0) -> float:
    """Shared-wrong rate the mock produces at MOCK_CORR = c (accuracy forced low so wrong pairs are plentiful)."""
    tasks = generate_synthetic(n_tasks, seed=900 + seed)
    llm = MockLLM(seed=seed, base_acc=0.0, error_corr=c)
    ans = [[parse_answer(llm.complete("s", f"QUESTION: {t.question}\nDIFFICULTY: {t.difficulty}").text) for _ in range(k)] for t in tasks]
    return shared_wrong_rate(ans, tasks)[0]


def fit_mock_corr(target: float, k: int = 5, n_tasks: int = 300) -> dict:
    """Pick the MOCK_CORR whose simulated shared-wrong rate is closest to `target`."""
    curve = {c: simulate_rate(c, k, n_tasks) for c in GRID}
    best = min(curve, key=lambda c: abs(curve[c] - target))
    lo, hi = min(curve.values()), max(curve.values())
    return {"mock_corr": best, "curve": curve, "in_range": bool(lo - 0.02 <= target <= hi + 0.02)}
