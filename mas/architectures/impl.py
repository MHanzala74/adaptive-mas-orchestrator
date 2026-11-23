from __future__ import annotations

from mas.core import Architecture, Context, majority
from mas.llm import parse_answer

SOLVER = "You are a careful problem solver. End with 'ANSWER: <number>'."


def _q(task) -> str:
    return f"QUESTION: {task.question}\nDIFFICULTY: {task.difficulty}"


class SingleAgent(Architecture):
    name = "single"
    agents = ["solo"]

    def solve(self, task, ctx: Context):
        return parse_answer(ctx.ask("solo", SOLVER, _q(task), 0.3))


class Debate(Architecture):
    name = "debate"

    def __init__(self, n_agents: int = 3, rounds: int = 2):
        self.n, self.rounds = n_agents, rounds
        self.agents = [f"debater_{i}" for i in range(n_agents)]

    def solve(self, task, ctx: Context):
        active = [a for a in self.agents if not ctx.bypassed(a)] or self.agents[:1]
        answers: dict[str, str | None] = {}
        for a in active:  # round 1: independent
            answers[a] = parse_answer(ctx.ask(a, SOLVER, _q(task), 0.8))
        for _ in range(self.rounds - 1):  # later rounds: see peers
            prev = dict(answers)
            for a in active:
                others = "; ".join(f"{k}={v}" for k, v in prev.items() if k != a)
                prompt = f"{_q(task)}\nOTHER ANSWERS: {others}\nReconsider and give your final answer."
                answers[a] = parse_answer(ctx.ask(a, SOLVER, prompt, 0.5))
        return majority(list(answers.values()))


class DebateJudge(Debate):
    """Debate, but when the vote has no majority a judge re-solves and breaks the tie
    (motivated by the taxonomy: many failures were 'vote lost correct')."""
    name = "debate_judge"

    def __init__(self, n_agents: int = 3, rounds: int = 2):
        super().__init__(n_agents, rounds)
        self.agents = self.agents + ["judge"]

    def solve(self, task, ctx: Context):
        active = [a for a in self.agents[:-1] if not ctx.bypassed(a)] or self.agents[:1]
        answers: dict[str, str | None] = {a: parse_answer(ctx.ask(a, SOLVER, _q(task), 0.8)) for a in active}
        for _ in range(self.rounds - 1):
            prev = dict(answers)
            for a in active:
                others = "; ".join(f"{k}={v}" for k, v in prev.items() if k != a)
                answers[a] = parse_answer(ctx.ask(a, SOLVER, f"{_q(task)}\nOTHER ANSWERS: {others}\nReconsider and give your final answer.", 0.5))
        vals = [v for v in answers.values() if v is not None]
        if not vals:
            return None
        top = max(vals.count(v) for v in vals)
        if top >= 2 or len(vals) < 2 or ctx.bypassed("judge"):
            return majority(list(answers.values()))
        j = parse_answer(ctx.ask("judge", "You are a judge. Solve independently; end with 'ANSWER: <number>'.", _q(task), 0.2))
        return j if j in vals else majority(list(answers.values()))


class PlannerWorker(Architecture):
    name = "planner_worker"
    agents = ["planner", "worker"]

    def solve(self, task, ctx: Context):
        plan = ""
        if not ctx.bypassed("planner"):
            plan = ctx.ask("planner", "Write a short numbered plan. Do not solve.", _q(task), 0.3)
        prompt = _q(task) + (f"\nPLAN: {plan}" if plan else "")
        return parse_answer(ctx.ask("worker", SOLVER, prompt, 0.3))


class SolverCritic(Architecture):
    """solver answers -> critic re-solves independently -> on mismatch solver revises."""
    name = "solver_critic"
    agents = ["solver", "critic"]

    def solve(self, task, ctx: Context):
        a1 = parse_answer(ctx.ask("solver", SOLVER, _q(task), 0.3))
        if ctx.bypassed("critic"):
            return a1
        crit = ctx.ask("critic", "You are an independent reviewer. Solve it yourself and end with 'ANSWER: <number>'.",
                       f"{_q(task)}\nPROPOSED: {a1}", 0.2)
        a2 = parse_answer(crit)
        if a2 is not None and a2 == a1:
            return a1
        prompt = f"{_q(task)}\nCRITIQUE: a reviewer got {a2}, you got {a1}. Re-solve carefully."
        return parse_answer(ctx.ask("solver", SOLVER, prompt, 0.3)) or a1


# ---------------- baselines (comparison methods; not part of the router's core pool) ----------------
class SelfConsistency(Architecture):
    """k independent samples from one solver, plurality vote (equal-budget baseline for debate)."""

    def __init__(self, k: int = 5):
        self.k = k
        self.name = f"sc{k}"
        self.agents = [f"sampler_{i}" for i in range(k)]

    def solve(self, task, ctx: Context):
        active = [a for a in self.agents if not ctx.bypassed(a)] or self.agents[:1]
        return majority([parse_answer(ctx.ask(a, SOLVER, _q(task), 0.8)) for a in active])


class AdaptiveSC(Architecture):
    """Sample sequentially and stop as soon as two samples agree (early-stopping self-consistency)."""
    name = "adaptive_sc"

    def __init__(self, kmax: int = 7):
        self.kmax = kmax
        self.agents = [f"sampler_{i}" for i in range(kmax)]

    def solve(self, task, ctx: Context):
        got: list[str | None] = []
        for a in self.agents:
            if ctx.bypassed(a):
                continue
            got.append(parse_answer(ctx.ask(a, SOLVER, _q(task), 0.8)))
            vals = [g for g in got if g is not None]
            if vals and max(vals.count(v) for v in vals) >= 2:
                break
        return majority(got)


class Cascade(Architecture):
    """Cheap first: two samples; if they agree accept, otherwise escalate to a full debate."""
    name = "cascade"
    agents = ["cascade_0", "cascade_1", "debater_0", "debater_1", "debater_2"]

    def solve(self, task, ctx: Context):
        a = parse_answer(ctx.ask("cascade_0", SOLVER, _q(task), 0.8))
        b = parse_answer(ctx.ask("cascade_1", SOLVER, _q(task), 0.8))
        if a is not None and a == b:
            return a
        return Debate().solve(task, ctx)


REGISTRY = {
    "single": SingleAgent,
    "debate": Debate,
    "planner_worker": PlannerWorker,
    "solver_critic": SolverCritic,
    "debate_judge": DebateJudge,
}


BASELINES = {
    "sc3": lambda: SelfConsistency(3),
    "sc5": lambda: SelfConsistency(5),
    "sc7": lambda: SelfConsistency(7),      # ~ same token budget as debate
    "adaptive_sc": AdaptiveSC,
    "cascade": Cascade,
}
ALL_NAMES = list(REGISTRY) + list(BASELINES)


def make(name: str) -> Architecture:
    if name in REGISTRY:
        return REGISTRY[name]()
    return BASELINES[name]()
