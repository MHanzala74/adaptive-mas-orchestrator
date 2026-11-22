from mas.benchmarks.tasks import generate_synthetic
from mas.faults import FaultSpec
from mas.taxonomy import classify, CATEGORIES
from mas.testbed import run_once

def test_categories_cover_fault_kinds():
    t = generate_synthetic(30, 3)
    seen = set()
    for i, task in enumerate(t):
        for kind in ("loop", "wrong"):
            r = run_once(task, "single", seed=i, faults=[FaultSpec("solo", kind)])
            c = classify(r, task); assert c in CATEGORIES; seen.add(c)
        seen.add(classify(run_once(task, "single", seed=i), task))
    assert {"degenerate_output", "injected_fault_propagated", "ok"} <= seen

def test_budget_category():
    t = generate_synthetic(1)[0]
    r = run_once(t, "debate", budget=60)
    assert classify(r, t) in ("budget_exceeded", "ok")
