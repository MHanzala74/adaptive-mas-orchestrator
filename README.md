# Adaptive Multi-Agent Orchestrator

A multi-agent system that **chooses its own agent architecture per task, detects when a run fails, finds the agent responsible, and repairs itself**, all inside a token budget.

<p align="center"><img src="docs/fig_architecture.png" width="900" alt="Figure 1: system architecture"></p>

## What it does
- **Routes** each task to the cheapest architecture likely to succeed (single, debate, debate+judge, planner-worker, solver-critic) using a learned contextual bandit.
- **Detects** failures at inference time (no answer, degenerate output, low agreement, budget overrun, optional verifier).
- **Attributes** a failure to a specific agent and step.
- **Repairs** by reprompting, replacing or bypassing the agent, or switching topology.
- **Tests itself** with a fault-injection testbed and reports bootstrap confidence intervals.

## Quick start
```bash
pip install -r requirements.txt
python -m pytest -q                  # 45 tests, offline
uvicorn api.main:app --reload        # API docs at http://127.0.0.1:8000/docs
```
Real models: set `LLM_BACKEND=anthropic` or `openai` plus the API key in your environment (never commit keys), and start with a small run.

## Documentation
**Full documentation: [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)**: components, API reference, configuration, experiments, results and limitations.
Also: [technical report draft](docs/report.md) and generated tables in [`results/`](results/).

## Status and honesty note
All reported numbers come from an **offline simulated solver**, not a real language model, so they validate the machinery and suggest hypotheses but are not evidence about real LLMs. Real-model runs are the next step; see the [limitations](docs/DOCUMENTATION.md#9-limitations-and-threats-to-validity) and [roadmap](docs/DOCUMENTATION.md#12-roadmap).
