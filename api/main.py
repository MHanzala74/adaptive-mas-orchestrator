"""FastAPI service exposing the orchestrator, router training and fault sweeps."""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mas.architectures.impl import REGISTRY
from mas.attribution import LLMAttributor, RuleAttributor
from mas.benchmarks.tasks import Task, check, generate_synthetic, split
from mas.faults import FAULT_KINDS, FaultSpec
from mas.llm import get_llm
from mas.orchestrator import Orchestrator
from mas.repair import RepairPolicy
from mas.router.bandit import LinUCBRouter
from mas.router.train import collect_log, fit_full_info
from mas.testbed import fault_sweep

app = FastAPI(title="Adaptive Multi-Agent Orchestrator", version="0.1.0")
STATE: dict = {"router": None, "router_info": None}
ARCHS = list(REGISTRY)


class FaultIn(BaseModel):
    agent: str
    kind: Literal["wrong", "loop", "malicious", "silent"]
    prob: float = Field(1.0, ge=0, le=1)


class SolveIn(BaseModel):
    question: str
    answer: str | None = None
    difficulty: int = Field(1, ge=1, le=4)
    budget: int | None = Field(None, gt=0)
    architecture: str | None = None            # force one; otherwise router/default
    use_router: bool = True
    trigger: Literal["detector", "oracle", "never"] = "detector"
    attributor: Literal["rule", "llm"] = "rule"
    allow_switch: bool = True
    max_repairs: int = Field(2, ge=0, le=5)
    faults: list[FaultIn] = []
    seed: int = 0


class TrainIn(BaseModel):
    n_tasks: int = Field(120, ge=20, le=2000)
    lam: float = Field(0.5, ge=0)
    seed: int = 0


class SweepIn(BaseModel):
    n_tasks: int = Field(60, ge=10, le=500)
    seed: int = 0


@app.get("/health")
def health():
    return {"status": "ok", "router_trained": STATE["router"] is not None}


@app.get("/architectures")
def architectures():
    return {"architectures": ARCHS, "fault_kinds": list(FAULT_KINDS)}


@app.post("/solve")
def solve(req: SolveIn):
    if req.architecture and req.architecture not in REGISTRY:
        raise HTTPException(422, f"unknown architecture {req.architecture}")
    task = Task("api", req.question, req.answer or "", req.difficulty)
    llm = get_llm(req.seed)
    router = STATE["router"] if (req.use_router and not req.architecture) else None
    orch = Orchestrator(
        router=router, default_arch=req.architecture or "single",
        attributor=LLMAttributor(llm) if req.attributor == "llm" else RuleAttributor(),
        policy=RepairPolicy(req.allow_switch), max_repairs=req.max_repairs, trigger=req.trigger)
    res = orch.run(task, budget=req.budget, faults=[FaultSpec(**f.model_dump(), seed=req.seed) for f in req.faults],
                   seed=req.seed, llm=llm)
    out = res.to_dict()
    out["correct"] = check(task, res.answer) if req.answer is not None else None
    return out


@app.post("/router/train")
def train_router(req: TrainIn):
    tr, _ = split(generate_synthetic(req.n_tasks, req.seed), 1.0, req.seed)
    log = collect_log(tr, ARCHS, seeds=(0,))
    STATE["router"] = fit_full_info(LinUCBRouter(ARCHS, alpha=0.0), log, req.lam)
    STATE["router_info"] = {"n_tasks": len(tr), "lam": req.lam,
                            "mean_cost": {a: STATE["router"].expected_cost(a) for a in ARCHS}}
    return STATE["router_info"]


@app.get("/router/status")
def router_status():
    return STATE["router_info"] or {"trained": False}


@app.post("/faults/sweep")
def sweep(req: SweepIn):
    return {"rows": fault_sweep(generate_synthetic(req.n_tasks, req.seed), seed=req.seed)}
