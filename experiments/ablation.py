"""Ablation of the self-healing loop under injected faults.

Configs: never | blind_retry | attribution (no switch) | attribution+switch | oracle-trigger (upper bound)
Scenarios: persistent faults (wrong, malicious, loop, silent) and intermittent ones (@0.5).
Usage: python experiments/ablation.py [--n 100] [--seeds 0 1 2]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mas.attribution import RuleAttributor
from mas.benchmarks.tasks import generate_synthetic
from mas.faults import FaultSpec
from mas.orchestrator import Orchestrator
from mas.repair import RepairPolicy
from mas.stats import bootstrap_ci

VICTIM = {"single": "solo", "debate": "debater_0", "planner_worker": "worker"}
SCENARIOS = [("wrong", 1.0), ("malicious", 1.0), ("loop", 1.0), ("silent", 1.0), ("wrong", 0.5), ("loop", 0.5)]
CONFIGS = {
    "never": lambda a: Orchestrator(default_arch=a, trigger="never"),
    "blind_retry": lambda a: Orchestrator(default_arch=a, trigger="detector", attributor=None),
    "attr (no switch)": lambda a: Orchestrator(default_arch=a, attributor=RuleAttributor(), policy=RepairPolicy(False)),
    "attr + switch": lambda a: Orchestrator(default_arch=a, attributor=RuleAttributor(), policy=RepairPolicy(True)),
    "oracle trigger": lambda a: Orchestrator(default_arch=a, trigger="oracle", policy=RepairPolicy(True)),
}


def main(n, seeds):
    tasks = generate_synthetic(n, seed=200)
    hdr = "| arch | fault | " + " | ".join(CONFIGS) + " |"
    lines = ["## Ablation: accuracy (mean ± CI half-width) / tokens per task",
             f"\n{len(seeds)} seeds x {n} tasks per cell. Fault `kind@p`: p = per-call trigger probability.\n",
             hdr, "|" + "---|" * (2 + len(CONFIGS))]
    for arch in VICTIM:
        for kind, p in SCENARIOS:
            cells = []
            for name, mk in CONFIGS.items():
                orch, c, tk = mk(arch), [], []
                for sd in seeds:
                    f = [FaultSpec(VICTIM[arch], kind, p, seed=sd)]
                    for i, t in enumerate(tasks):
                        r = orch.run(t, faults=f, seed=sd * 7919 + i)
                        c.append(int(r.correct)); tk.append(r.tokens)
                m, lo, hi = bootstrap_ci(c)
                cells.append(f"{m:.2f}±{(hi - lo) / 2:.2f} / {sum(tk) / len(tk):.0f}")
            lines.append(f"| {arch} | {kind}@{p} | " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    Path("results").mkdir(exist_ok=True)
    Path("results/ablation.md").write_text(md)
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    main(a.n, a.seeds)
