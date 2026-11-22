from mas.architectures.impl import make, REGISTRY
from mas.benchmarks.tasks import generate_synthetic
from mas.core import Context
from mas.llm import MockLLM, TokenMeter

def run(arch, task, budget=None, seed=0, **kw):
    ctx = Context(MockLLM(seed=seed), TokenMeter(budget), **kw)
    return make(arch).run(task, ctx)

def test_all_architectures_run_and_cost_order():
    t = generate_synthetic(1)[0]
    r = {a: run(a, t) for a in REGISTRY}
    assert all(x.answer is not None for x in r.values())
    assert r["single"].tokens < r["planner_worker"].tokens < r["debate"].tokens

def test_budget_enforced():
    t = generate_synthetic(1)[0]
    r = run("debate", t, budget=60)
    assert r.budget_exceeded and r.tokens > 0

def test_bypass_planner():
    t = generate_synthetic(1)[0]
    r = run("planner_worker", t, overrides={"planner": {"action": "bypass"}})
    assert [s.agent for s in r.trace] == ["worker"]

def test_debate_beats_single_on_average():
    ts = generate_synthetic(120, seed=3)
    acc = {a: sum(run(a, t, seed=i).correct for i, t in enumerate(ts)) / len(ts) for a in ("single", "debate")}
    assert acc["debate"] >= acc["single"]

def test_solver_critic_flow_and_bypass():
    t = generate_synthetic(1)[0]
    r = run("solver_critic", t)
    assert [s.agent for s in r.trace][:2] == ["solver", "critic"] and 2 <= r.calls <= 3
    b = run("solver_critic", t, overrides={"critic": {"action": "bypass"}})
    assert [s.agent for s in b.trace] == ["solver"]

def test_solver_critic_beats_single_on_average():
    ts = generate_synthetic(150, seed=4)
    acc = {a: sum(run(a, t, seed=i).correct for i, t in enumerate(ts)) / len(ts) for a in ("single", "solver_critic")}
    assert acc["solver_critic"] > acc["single"]

def test_debate_judge_only_calls_judge_on_split_and_helps():
    ts = generate_synthetic(300, seed=8)
    rd = [run("debate", t, seed=i) for i, t in enumerate(ts)]
    rj = [run("debate_judge", t, seed=i) for i, t in enumerate(ts)]
    called = [any(s.agent == "judge" for s in r.trace) for r in rj]
    assert 0 < sum(called) < len(ts) // 2
    assert sum(r.correct for r in rj) >= sum(r.correct for r in rd)
    assert sum(r.tokens for r in rj) < 1.3 * sum(r.tokens for r in rd)
