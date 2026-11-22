"""Failure attribution: which agent / step caused a failed run?

Two attributors share one interface:
  - RuleAttributor: zero-token heuristics (degenerate output, outlier vs majority).
  - LLMAttributor: reads the full trace and names the culprit (Who&When style).
Ground truth comes from the fault-injection testbed (Step.faulted).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from mas.core import RunResult, Step
from mas.llm import BaseLLM, TokenMeter, get_llm

NON_ANSWERING = {"planner", "judge"}
FINAL_ANSWERER = {"solver_critic": "solver"}   # who owns the final answer when no outlier signal exists   # agents whose output is not a final numeric answer


@dataclass
class Attribution:
    agent: str | None
    step: int | None
    reason: str
    kind: str = "unknown"      # degenerate | outlier | unknown
    tokens: int = 0


def is_degenerate(text: str) -> bool:
    words = text.split()
    if not text.strip():
        return True
    if len(words) >= 20 and len(set(words)) / len(words) < 0.2:
        return True
    return False


def _last_answers(trace: list[Step]) -> dict[str, Step]:
    last = {}
    for s in trace:
        if s.agent not in NON_ANSWERING:
            last[s.agent] = s
    return last


def _first_answers(trace: list[Step]) -> dict[str, Step]:
    first = {}
    for s in trace:
        if s.agent not in NON_ANSWERING and s.agent not in first:
            first[s.agent] = s
    return first


def _anomalous_step(trace: list[Step], agent: str) -> int | None:
    steps = [s for s in trace if s.agent == agent]
    for s in steps:
        if is_degenerate(s.output) or (s.answer is None and agent not in NON_ANSWERING):
            return s.idx
    return steps[0].idx if steps else None


class RuleAttributor:
    name = "rule"

    def attribute(self, res: RunResult) -> Attribution:
        last = _last_answers(res.trace)
        # 1) degenerate / empty / unparsable output
        for agent, s in last.items():
            if is_degenerate(s.output) or s.answer is None:
                return Attribution(agent, _anomalous_step(res.trace, agent), "degenerate output", "degenerate")
        # 2) outlier versus the per-round majority (needs >= 3 answering agents):
        #    blame the agent that disagrees with the majority in the most rounds.
        by_agent: dict[str, list[Step]] = {}
        for st in res.trace:
            if st.agent not in NON_ANSWERING:
                by_agent.setdefault(st.agent, []).append(st)
        if len(by_agent) >= 3:
            rounds = min(len(v) for v in by_agent.values())
            dis = {a: 0 for a in by_agent}
            for r in range(rounds):
                vals = Counter(v[r].answer for v in by_agent.values())
                maj, cnt = vals.most_common(1)[0]
                if cnt >= 2:
                    for a, v in by_agent.items():
                        dis[a] += v[r].answer != maj
            worst = max(dis, key=dis.get)
            if dis[worst] > 0:
                return Attribution(worst, _anomalous_step(res.trace, worst) , "most rounds in disagreement", "outlier")
        # 3) fall back: last answering agent (the one that produced the final answer)
        if last:
            agent = FINAL_ANSWERER.get(res.architecture) or list(last)[-1]
            return Attribution(agent, last[agent].idx, "default: final answerer", "unknown")
        return Attribution(None, None, "empty trace")


def format_trace(trace: list[Step]) -> str:
    lines = []
    for s in trace:
        snippet = " ".join(s.output.split())[:120]
        lines.append(f"[{s.agent}] ANSWER: {s.answer if s.answer is not None else 'none'} | step={s.idx} | out: {snippet}")
    return "\n".join(lines)


ATTR_SYSTEM = ("ROLE:ATTRIBUTOR\nYou audit multi-agent runs that produced a wrong or suspicious result. "
               "Identify the single agent most responsible. Reply exactly as:\nBLAME: <agent>\nREASON: <short reason>")


class LLMAttributor:
    name = "llm"

    def __init__(self, llm: BaseLLM | None = None):
        self.llm = llm or get_llm()

    def attribute(self, res: RunResult) -> Attribution:
        meter = TokenMeter()
        prompt = f"ARCHITECTURE: {res.architecture}\nFINAL ANSWER: {res.answer}\nTRACE:\n{format_trace(res.trace)}"
        out = self.llm.call(meter, ATTR_SYSTEM, prompt, 0.0).text
        m = re.search(r"BLAME:\s*(\w+)", out)
        agent = m.group(1) if m and m.group(1) != "unknown" else None
        reason = (re.search(r"REASON:\s*(.+)", out) or [None, ""])[1] if m else "unparsed"
        known = {s.agent for s in res.trace}
        if agent not in known:
            agent = None
        step = _anomalous_step(res.trace, agent) if agent else None
        return Attribution(agent, step, reason.strip() if isinstance(reason, str) else "", "unknown", meter.total)


def truth_of(res: RunResult) -> tuple[set[str], set[int]]:
    faulted = [s for s in res.trace if s.faulted]
    return {s.agent for s in faulted}, {s.idx for s in faulted}


def score_attribution(results: list[RunResult], attributor) -> dict:
    """Agent-level and step-level accuracy on runs that contain an injected fault."""
    n = a_ok = s_ok = tokens = 0
    for r in results:
        agents, steps = truth_of(r)
        if not agents:
            continue
        at = attributor.attribute(r)
        n += 1; tokens += at.tokens
        a_ok += at.agent in agents
        s_ok += at.step in steps
    return {"n": n, "agent_acc": a_ok / n if n else 0.0,
            "step_acc": s_ok / n if n else 0.0, "tokens": tokens / n if n else 0.0}
