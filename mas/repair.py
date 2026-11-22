"""Repair policy: turn an attribution into a concrete topology/agent change."""
from __future__ import annotations

from dataclasses import dataclass

from mas.attribution import Attribution
from mas.core import RunResult

# agents that can be dropped without the architecture becoming meaningless
BYPASSABLE = {"debater_0", "debater_1", "debater_2", "planner"}
ESCALATE = {"single": "debate", "planner_worker": "debate", "debate": "planner_worker"}


@dataclass
class RepairAction:
    kind: str                  # reprompt | replace | bypass | switch | none
    agent: str | None = None
    switch_to: str | None = None
    note: str = ""


class RepairPolicy:
    def __init__(self, allow_switch: bool = True):
        self.allow_switch = allow_switch

    def plan(self, at: Attribution, arch: str, attempt: int, already: dict[str, dict]) -> RepairAction:
        agent = at.agent
        if agent is None:
            return self._switch(arch, "no culprit identified")
        prior = already.get(agent, {}).get("action")
        if prior is None:
            if at.kind == "degenerate":
                return RepairAction("replace", agent, note="degenerate output -> fresh agent")
            return RepairAction("reprompt", agent, note="Your previous answer was inconsistent. Recompute carefully.")
        if prior in ("reprompt", "replace") and agent in BYPASSABLE:
            return RepairAction("bypass", agent, note="repeat offender -> drop from topology")
        return self._switch(arch, f"{agent} already {prior}")

    def _switch(self, arch: str, why: str) -> RepairAction:
        if self.allow_switch and arch in ESCALATE:
            return RepairAction("switch", switch_to=ESCALATE[arch], note=why)
        return RepairAction("none", note=why)


def detect_failure(res: RunResult, min_agreement: float = 1.0) -> bool:
    """Ground-truth-free failure signals available at inference time.

    min_agreement: for debate, minimum share of agents that must agree on the final
    answer. 1.0 = any disagreement is a failure (original behaviour); 0.66 = a 2-of-3
    majority is accepted (consensus-margin aware).
    """
    from mas.attribution import is_degenerate, NON_ANSWERING
    if res.answer is None or res.budget_exceeded:
        return True
    if any(is_degenerate(s.output) for s in res.trace):
        return True
    if res.architecture == "debate":
        last: dict[str, str | None] = {}
        for s in res.trace:
            if s.agent not in NON_ANSWERING:
                last[s.agent] = s.answer
        if last:
            top = max(list(last.values()).count(v) for v in last.values())
            if top / len(last) < min_agreement - 1e-9:
                return True
    return False
