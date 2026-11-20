"""Self-healing orchestrator: route -> run -> detect -> attribute -> repair -> retry."""
from __future__ import annotations

from dataclasses import dataclass, field

from mas.architectures.impl import make
from mas.attribution import RuleAttributor
from mas.benchmarks.tasks import check
from mas.core import Context, RunResult
from mas.faults import FaultSpec, build_faults
from mas.llm import BaseLLM, BudgetExceeded, MockLLM, TokenMeter, parse_answer
from mas.repair import RepairPolicy, detect_failure
from mas.router.features import featurize


@dataclass
class Orchestrator:
    router: object | None = None            # LinUCBRouter or None
    default_arch: str = "single"
    attributor: object | None = field(default_factory=RuleAttributor)
    policy: RepairPolicy = field(default_factory=RepairPolicy)
    max_repairs: int = 2
    trigger: str = "detector"               # detector | oracle | never
    verify: bool = False                    # extra independent re-solve; mismatch => failure

    def _failed(self, res: RunResult, task=None, llm=None, meter=None) -> tuple[bool, str]:
        if self.trigger == "never":
            return False, "off"
        if self.trigger == "oracle":
            return (not res.correct), "oracle"
        if detect_failure(res):
            return True, "detector"
        if self.verify and task is not None and not res.budget_exceeded:
            try:   # independent verifier: a fresh solver call outside the traced topology
                out = llm.call(meter, "AGENT:verifier\nSolve independently. End with 'ANSWER: <number>'.",
                               f"QUESTION: {task.question}\nDIFFICULTY: {task.difficulty}", 0.2)
            except BudgetExceeded:
                return False, "verifier skipped (budget)"
            if parse_answer(out.text) != res.answer:
                return True, "verifier mismatch"
        return False, "ok"

    def run(self, task, *, budget: int | None = None, faults: list[FaultSpec] | None = None,
            seed: int = 0, llm: BaseLLM | None = None) -> RunResult:
        llm = llm or MockLLM(seed=seed)
        meter = TokenMeter(budget)             # one shared budget across all attempts
        fault_fns = build_faults(faults or [], salt=str(seed))
        arch_name = (self.router.select(featurize(task.question), budget, explore=False)
                     if self.router else self.default_arch)
        overrides: dict[str, dict] = {}
        history, attr_tokens, res = [], 0, None
        for attempt in range(1, self.max_repairs + 2):
            ctx = Context(llm, meter, faults=fault_fns, overrides=dict(overrides))
            res = make(arch_name).run(task, ctx)
            failed, signal = self._failed(res, task, llm, meter)
            history.append({"attempt": attempt, "arch": arch_name, "answer": res.answer,
                            "detected_failure": failed, "signal": signal})
            if not failed or attempt > self.max_repairs or res.budget_exceeded:
                break
            if self.attributor is None:      # ablation: blind retry, nothing changes
                history[-1]["repair"] = "retry"
                continue
            at = self.attributor.attribute(res)
            attr_tokens += at.tokens
            act = self.policy.plan(at, arch_name, attempt, overrides)
            history[-1].update({"blamed": at.agent, "reason": at.reason, "repair": act.kind})
            if act.kind == "none":
                break
            if act.kind == "switch":
                arch_name, overrides = act.switch_to, {}      # new topology = fresh agents
            else:
                overrides[act.agent] = {"action": act.kind, "note": act.note}
        res.tokens = meter.total + attr_tokens
        res.calls = meter.calls
        res.attempts = len(history)
        res.repaired = len(history) > 1 and res.correct
        res.meta = {"history": history, "final_arch": arch_name}
        return res
