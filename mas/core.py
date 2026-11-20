"""Shared runtime: trace recording, per-agent faults, repair overrides, budget."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Callable

from mas.llm import BaseLLM, TokenMeter, BudgetExceeded, parse_answer


@dataclass
class Step:
    idx: int
    agent: str
    prompt: str
    output: str
    answer: str | None
    faulted: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class RunResult:
    task_id: str
    architecture: str
    answer: str | None
    correct: bool
    tokens: int
    calls: int
    trace: list[Step]
    budget_exceeded: bool = False
    repaired: bool = False
    attempts: int = 1
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d["trace"] = [s.to_dict() for s in self.trace]
        return d


# A fault maps (agent, clean_output) -> corrupted_output. Defined in mas/faults.py
FaultFn = Callable[[str, str], str]


@dataclass
class Context:
    llm: BaseLLM
    meter: TokenMeter
    faults: dict[str, FaultFn] = field(default_factory=dict)
    # repair overrides: agent -> {"action": "reprompt"|"replace"|"bypass", "note": str}
    overrides: dict[str, dict] = field(default_factory=dict)
    trace: list[Step] = field(default_factory=list)

    def bypassed(self, agent: str) -> bool:
        return self.overrides.get(agent, {}).get("action") == "bypass"

    def ask(self, agent: str, system: str, prompt: str, temperature: float = 0.7) -> str:
        ov = self.overrides.get(agent, {})
        if ov.get("action") == "reprompt":
            system += "\nIMPORTANT CORRECTION: " + ov.get("note", "Double-check your work.")
        resp = self.llm.call(self.meter, f"AGENT:{agent}\n{system}", prompt, temperature)
        text, faulted = resp.text, False
        fault = self.faults.get(agent)
        if fault and ov.get("action") not in ("replace", "bypass"):
            corrupted = fault(agent, text)
            faulted = corrupted != text
            text = corrupted
        self.trace.append(Step(len(self.trace), agent, prompt, text, parse_answer(text), faulted))
        return text


class Architecture:
    name = "base"
    agents: list[str] = []

    def solve(self, task, ctx: Context) -> str | None:
        raise NotImplementedError

    def run(self, task, ctx: Context) -> RunResult:
        from mas.benchmarks.tasks import check
        exceeded, ans = False, None
        try:
            ans = self.solve(task, ctx)
        except BudgetExceeded:
            exceeded = True
            answers = [s.answer for s in ctx.trace if s.answer is not None]
            ans = Counter(answers).most_common(1)[0][0] if answers else None
        return RunResult(task.id, self.name, ans, check(task, ans), ctx.meter.total,
                         ctx.meter.calls, list(ctx.trace), budget_exceeded=exceeded)


def majority(answers: list[str | None]) -> str | None:
    vals = [a for a in answers if a is not None]
    if not vals:
        return None
    c = Counter(vals).most_common()
    return c[0][0]
