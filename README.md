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
- [ ] Router
- [ ] Fault injection
- [ ] Failure attribution + repair
- [ ] FastAPI service
- [ ] Experiment grid, ablations, stats
- [ ] Technical report
