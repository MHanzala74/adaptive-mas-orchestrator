from mas.attribution import RuleAttributor, LLMAttributor, score_attribution, is_degenerate
from mas.benchmarks.tasks import generate_synthetic
from mas.faults import FaultSpec
from mas.testbed import run_once, default_scenarios

def runs(arch, kind, n=60):
    victim = default_scenarios(arch)[kind]
    return [run_once(t, arch, seed=i, faults=victim) for i, t in enumerate(generate_synthetic(n, 11))]

def test_degenerate():
    assert is_degenerate("") and is_degenerate("Let me re-check. " * 25)
    assert not is_degenerate("ANSWER: 4")

def test_rule_attributes_loop_and_silent_perfectly():
    for arch in ("single", "debate", "planner_worker"):
        for kind in ("loop", "silent"):
            s = score_attribution(runs(arch, kind), RuleAttributor())
            assert s["agent_acc"] == 1.0, (arch, kind, s)

def test_rule_finds_outlier_in_debate():
    s = score_attribution(runs("debate", "malicious"), RuleAttributor())
    assert s["agent_acc"] > 0.5

def test_llm_attributor_runs_and_costs_tokens():
    s = score_attribution(runs("debate", "wrong", 30), LLMAttributor())
    assert s["n"] == 30 and s["tokens"] > 0
