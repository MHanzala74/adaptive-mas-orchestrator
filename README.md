# Adaptive Multi-Agent Orchestrator with Failure Attribution

## Research question
Can a system decide, per task and within a token budget, whether to use a
**single agent**, **debate**, or **planner-worker** — and, when it fails,
identify the responsible agent/step and repair its own topology?

## Components
1. **Learned router** (contextual bandit) - picks the cheapest architecture likely to succeed.
2. **Failure attribution agent** - reads a trace, blames an agent/step, triggers a repair.
3. **Fault-injection testbed** - deliberately breaks agents to measure recovery.

## Status
- [x] Project skeleton
- [x] LLM layer with token/cost accounting (`mas/llm.py`: `TokenMeter`, hard budget, Mock + Anthropic backends)
- [x] Benchmarks (`mas/benchmarks/tasks.py`: synthetic difficulty-graded arithmetic, GSM8K jsonl loader, train/test split)
- [x] Architectures (`mas/core.py` runtime with trace + repair hooks; `mas/architectures/impl.py`: single, debate, planner-worker)
- [x] Router (`mas/router/`: observable task features, budget-aware LinUCB, full-info + bandit training, counterfactual replay evaluation, oracle)
- [x] Fault injection (`mas/faults.py`: wrong, malicious, loop, silent; `mas/testbed.py`: fault sweeps)
- [x] Failure attribution (`mas/attribution.py`: zero-token `RuleAttributor`, `LLMAttributor`, agent/step-level scoring against injected-fault ground truth)
- [ ] Repair policy + self-healing orchestrator
- [ ] FastAPI service
- [ ] Experiment grid, ablations, stats
- [ ] Technical report

## Early observations (mock backend, 100 tasks, rule attributor)
| Architecture | Fault | Agent-level attribution |
|---|---|---|
| single | any | 1.00 (trivial: one agent) |
| planner_worker | any | 1.00 (trivial: fault sits on last answerer) |
| debate | loop / silent | 1.00 |
| debate | wrong / malicious | 0.68 (n=200; was 0.54 using final-round only, 0.41 using round-1 only) |

Debate is the informative case: in round 2 the healthy debaters read the faulty
peer's answer and often converge to it, so the *outlier signal disappears*.
Scoring disagreement across *all* rounds beat both final-round-only and round-1-only voting; the remaining gap motivates the LLM attributor.
Numbers come from the offline mock solver; re-run with `LLM_BACKEND=anthropic` for real results.
