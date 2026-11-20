from mas.benchmarks.tasks import generate_synthetic
from mas.faults import FaultSpec
from mas.testbed import run_once, fault_sweep

def test_wrong_fault_hurts_single_not_debate():
    rows = {(r["arch"], r["scenario"]): r["acc"] for r in fault_sweep(generate_synthetic(80, 5))}
    assert rows[("single", "wrong")] < rows[("single", "clean")] - 0.3
    # debate has redundancy: one bad voter should hurt much less
    drop_single = rows[("single", "clean")] - rows[("single", "wrong")]
    drop_debate = rows[("debate", "clean")] - rows[("debate", "wrong")]
    assert drop_debate < drop_single

def test_fault_marks_trace():
    t = generate_synthetic(1)[0]
    r = run_once(t, "single", faults=[FaultSpec("solo", "wrong")])
    assert r.trace[0].faulted

def test_loop_and_silent_give_no_answer():
    t = generate_synthetic(1)[0]
    for k in ("loop", "silent"):
        assert run_once(t, "single", faults=[FaultSpec("solo", k)]).answer is None
