"""Failure taxonomy across architectures and fault scenarios -> results/taxonomy.md + docs/fig_taxonomy.png/pdf.
Usage: python experiments/taxonomy.py [--n 200]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mas.benchmarks.tasks import generate_synthetic
from mas.taxonomy import CATEGORIES, taxonomy_counts
from mas.testbed import default_scenarios, run_once

ARCHS = ["single", "debate", "planner_worker", "solver_critic"]
SCEN = ["clean", "wrong", "loop"]
FAIL = [c for c in CATEGORIES if c != "ok"]
HATCH = ["////", "\\\\", "xxxx", "....", "++++", "**", "---", "||"]
GREYS = ["0.2", "0.4", "0.55", "0.7", "0.8", "0.88", "0.94", "1.0"]


def main(n):
    tasks = generate_synthetic(n, seed=300)
    rows, labels = [], []
    for arch in ARCHS:
        for sc in SCEN:
            specs = default_scenarios(arch)[sc]
            res = [run_once(t, arch, seed=i, faults=specs) for i, t in enumerate(tasks)]
            c = taxonomy_counts(res, tasks)
            rows.append([c.get(k, 0) / n for k in FAIL])
            labels.append(f"{arch.replace('_', '-')} / {sc}")
    # markdown
    md = ["## Failure taxonomy (share of all tasks; simulated solver)", "", "| arch / fault | " + " | ".join(FAIL) + " | total fail |",
          "|---|" + "---|" * (len(FAIL) + 1)]
    for lab, r in zip(labels, rows):
        md.append(f"| {lab} | " + " | ".join(f"{v:.2f}" for v in r) + f" | {sum(r):.2f} |")
    Path("results").mkdir(exist_ok=True)
    Path("results/taxonomy.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    # figure
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 9})
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    left = [0.0] * len(labels)
    y = list(range(len(labels)))[::-1]
    for j, k in enumerate(FAIL):
        vals = [r[j] for r in rows]
        ax.barh(y, vals, left=left, color=GREYS[j], edgecolor="black", linewidth=0.6,
                hatch=HATCH[j], label=k.replace("_", " "))
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Fraction of tasks ending in failure"); ax.set_xlim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig("docs/fig_taxonomy.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig("docs/fig_taxonomy.pdf", bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200)
    main(ap.parse_args().n)
