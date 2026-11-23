"""Run tasks x architectures x fault scenarios and summarise recovery."""
from __future__ import annotations

from dataclasses import dataclass

from mas.architectures.impl import make
from mas.core import Context, RunResult
from mas.faults import FaultSpec, build_faults
from mas.llm import get_llm, TokenMeter, BaseLLM


def run_once(task, arch: str, *, llm: BaseLLM | None = None, seed: int = 0,
             budget: int | None = None, faults: list[FaultSpec] | None = None,
             overrides: dict | None = None) -> RunResult:
    ctx = Context(llm or get_llm(seed), TokenMeter(budget),
                  faults=build_faults(faults or [], salt=str(seed)), overrides=overrides or {})
    return make(arch).run(task, ctx)


def default_scenarios(arch: str) -> dict[str, list[FaultSpec]]:
    victim = {"single": "solo", "debate": "debater_0", "planner_worker": "worker", "solver_critic": "solver", "debate_judge": "debater_0",
              "sc3": "sampler_0", "sc5": "sampler_0", "sc7": "sampler_0", "adaptive_sc": "sampler_0", "cascade": "cascade_0"}[arch]
    sc = {"clean": []}
    for kind in ("wrong", "malicious", "loop", "silent"):
        sc[kind] = [FaultSpec(victim, kind)]
    return sc


def fault_sweep(tasks, archs=("single", "debate", "planner_worker"), seed=0) -> list[dict]:
    rows = []
    for arch in archs:
        for name, specs in default_scenarios(arch).items():
            res = [run_once(t, arch, seed=seed * 100003 + i, faults=specs) for i, t in enumerate(tasks)]
            rows.append({"arch": arch, "scenario": name,
                         "acc": sum(r.correct for r in res) / len(res),
                         "tokens": sum(r.tokens for r in res) / len(res)})
    return rows
