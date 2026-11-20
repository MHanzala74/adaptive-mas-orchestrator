from mas.benchmarks.tasks import generate_synthetic
from mas.faults import FaultSpec
from mas.orchestrator import Orchestrator
from mas.attribution import RuleAttributor

TASKS = generate_synthetic(100, 21)

def acc(orch, faults, budget=None):
    rs = [orch.run(t, faults=faults, seed=i, budget=budget) for i, t in enumerate(TASKS)]
    return sum(r.correct for r in rs) / len(rs), sum(r.tokens for r in rs) / len(rs), rs

def test_repair_recovers_loop_fault():
    f = [FaultSpec("solo", "loop")]
    base, _, _ = acc(Orchestrator(trigger="never"), f)
    healed, _, rs = acc(Orchestrator(trigger="detector"), f)
    assert base == 0.0 and healed > 0.5
    assert any(r.repaired for r in rs)

def test_no_repair_on_clean_costs_nothing_extra():
    a, t_never, _ = acc(Orchestrator(trigger="never"), [])
    b, t_det, _ = acc(Orchestrator(trigger="detector"), [])
    assert t_det == t_never and a == b      # single agent never self-detects -> no overhead

def test_budget_is_shared_and_respected():
    _, _, rs = acc(Orchestrator(trigger="detector"), [FaultSpec("solo", "loop")], budget=120)
    assert all(r.tokens <= 120 + 5 for r in rs)

def test_oracle_trigger_upper_bound_ge_detector():
    f = [FaultSpec("solo", "wrong")]
    det, _, _ = acc(Orchestrator(trigger="detector"), f)
    ora, _, _ = acc(Orchestrator(trigger="oracle"), f)
    assert ora >= det

def test_blind_retry_cannot_fix_persistent_fault_but_attribution_can():
    f = [FaultSpec("solo", "loop")]
    blind, _, _ = acc(Orchestrator(trigger="detector", attributor=None), f)
    attr, _, _ = acc(Orchestrator(trigger="detector"), f)
    assert blind == 0.0 and attr > 0.5

def test_verifier_detects_confident_wrong_and_switch_recovers():
    f = [FaultSpec("solo", "wrong")]
    plain, _, _ = acc(Orchestrator(trigger="detector"), f)
    ver, tok, rs = acc(Orchestrator(trigger="detector", verify=True), f)
    assert plain < 0.05 and ver > 0.3
    assert any(h.get("repair") == "switch" for r in rs for h in r.meta["history"])
