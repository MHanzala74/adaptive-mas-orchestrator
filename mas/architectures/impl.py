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


REGISTRY = {
    "single": SingleAgent,
    "debate": Debate,
    "planner_worker": PlannerWorker,
    "solver_critic": SolverCritic,
}


def make(name: str) -> Architecture:
    return REGISTRY[name]()
