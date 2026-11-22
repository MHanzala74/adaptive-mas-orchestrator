"""Failure taxonomy: classify why a run produced a wrong answer (needs ground truth)."""
from __future__ import annotations

from collections import Counter

from mas.attribution import NON_ANSWERING, is_degenerate
from mas.benchmarks.tasks import check
from mas.core import RunResult

CATEGORIES = ["ok", "budget_exceeded", "degenerate_output", "no_answer", "injected_fault_propagated",
              "unanimous_wrong", "vote_lost_correct", "split_all_wrong", "natural_error"]


def classify(res: RunResult, task) -> str:
    if res.correct:
        return "ok"
    if res.budget_exceeded:
        return "budget_exceeded"
    if any(is_degenerate(s.output) for s in res.trace):
        return "degenerate_output"
    if res.answer is None:
        return "no_answer"
    if any(s.faulted for s in res.trace):
        return "injected_fault_propagated"
    if res.architecture == "debate":
        last = {}
        for s in res.trace:
            if s.agent not in NON_ANSWERING:
                last[s.agent] = s.answer
        vals = list(last.values())
        if len(set(vals)) == 1:
            return "unanimous_wrong"
        if any(check(task, v) for v in vals):
            return "vote_lost_correct"
        return "split_all_wrong"
    return "natural_error"


def taxonomy_counts(results: list[RunResult], tasks) -> Counter:
    return Counter(classify(r, t) for r, t in zip(results, tasks))
