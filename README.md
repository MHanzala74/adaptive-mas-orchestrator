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
- [x] Repair policy + self-healing orchestrator (`mas/repair.py`, `mas/orchestrator.py`: route -> run -> detect -> attribute -> reprompt/replace/bypass/switch topology, shared token budget)
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

## Design assumptions (read before interpreting results)
- **Shared budget:** all attempts (initial run + repairs + LLM attribution) draw from one token budget per task.
- **Failure trigger:** `detector` uses only inference-time signals (no answer, degenerate output, debate disagreement, budget overrun). `oracle` uses ground truth and is reported only as an upper bound. A single agent that is *confidently wrong* is undetectable by the detector - a known limitation.
- **Topology switch = fresh agents:** faults are bound to agent names, so switching architecture escapes a broken agent. The `allow_switch=False` ablation isolates this effect.
