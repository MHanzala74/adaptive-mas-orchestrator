"""Accuracy-cost trade-offs: (a) accuracy vs mean tokens with Pareto frontier, (b) accuracy under a hard token budget.
Three pre-specified simulated environments (columns). Writes docs/fig_pareto.png/pdf and results/pareto.md.
Usage: python experiments/pareto.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *          # noqa: F401,F403
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mas.stats import bootstrap_ci

BUDGETS = [60, 100, 150, 200, 300, 500]
STYLE = {  # marker, filled?
    "single": ("o", 1), "planner_worker": ("s", 1), "solver_critic": ("^", 1), "debate": ("v", 1), "debate_judge": ("D", 1),
    "sc3": ("<", 0), "sc5": (">", 0), "sc7": ("X", 0), "adaptive_sc": ("P", 0), "cascade": ("d", 0),
    "router (core)": ("*", 1), "router (all)": ("*", 0),
}
LINES = ["single", "adaptive_sc", "sc7", "debate", "debate_judge", "router (core)", "router (all)"]


def acc_ci(choose, log):
    v = [rec["outcomes"][choose(rec)]["correct"] for rec in log]
    return bootstrap_ci(v, clusters=[r["task_id"] for r in log])


def tokens(choose, log):
    return float(np.mean([rec["outcomes"][choose(rec)]["tokens"] for rec in log]))


def frontier(pts):
    best, out = -1, []
    for x, y, n in sorted(pts):
        if y > best:
            out.append((x, y)); best = y
    return out


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 8.5, "mathtext.fontset": "stix"})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6))
    md = ["# Accuracy-cost trade-offs (simulated solver)", "", "Hard-budget sweep: accuracy when each method (or the router, using its cost estimates) must stay within B tokens; "
          "over-budget runs return the best answer so far.", ""]
    test_tasks = generate_synthetic(200, seed=100)
    for col, env in enumerate(ENVS):
        set_env(env)
        tr, te = build_logs()
        r_core, r_all = train_router(tr, CORE), train_router(tr, ALLM)
        pol = {a: (lambda rec, a=a: a) for a in ALLM}
        pol["router (core)"], pol["router (all)"] = router_policy(r_core), router_policy(r_all)
        ax = axes[0, col]
        pts = []
        for n, c in pol.items():
            m, lo, hi = acc_ci(c, te); t = tokens(c, te)
            mk, filled = STYLE[n]
            ax.errorbar(t, m, yerr=[[m - lo], [hi - m]], fmt=mk, mfc="black" if filled else "white", mec="black", color="black",
                        ms=9 if mk == "*" else 6, capsize=2, lw=0.8, label=n)
            pts.append((t, m, n))
        fr = frontier(pts)
        ax.step([p[0] for p in fr], [p[1] for p in fr], where="post", color="0.55", lw=1, ls="--", zorder=0)
        ax.set_title(env, fontsize=9.5); ax.set_xlabel("mean tokens per task"); ax.set_ylabel("accuracy" if col == 0 else "")
        ax.set_ylim(0.45, 1.0); ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, fontweight="bold", va="top")
        # (b) hard-budget sweep
        bx = axes[1, col]
        rows = {n: [] for n in LINES}
        for B in BUDGETS:
            _, teB = build_logs(budget=B, n_train=1)
            cl = {a: (lambda rec, a=a: a) for a in ALLM}
            cl["router (core)"], cl["router (all)"] = router_policy(r_core, B), router_policy(r_all, B)
            for n in LINES:
                rows[n].append(float(np.mean([rec["outcomes"][cl[n](rec)]["correct"] for rec in teB])))
        for n in LINES:
            mk, filled = STYLE[n]
            bx.plot(BUDGETS, rows[n], marker=mk, mfc="black" if filled else "white", mec="black", color="black",
                    ls="-" if "router" in n else ":", lw=1.4 if "router" in n else 0.9, ms=8 if mk == "*" else 5)
        bx.set_xlabel("hard token budget B"); bx.set_ylabel("accuracy" if col == 0 else ""); bx.set_ylim(0.5, 0.95)
        bx.spines[["top", "right"]].set_visible(False)
        bx.text(0.02, 0.97, "(b)", transform=bx.transAxes, fontweight="bold", va="top")
        md += [f"## {env}", "", "| method | " + " | ".join(f"B={b}" for b in BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
        md += [f"| {n} | " + " | ".join(f"{v:.3f}" for v in rows[n]) + " |" for n in LINES] + [""]
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.085))
    fig.tight_layout(rect=[0, 0.17, 1, 1])
    cap = ("$\\bf{Figure\\ 2.}$ Accuracy\u2013cost trade-offs in three pre-specified simulated environments (columns). (a) Accuracy versus mean tokens per task; error bars are 95%\n"
           "cluster-bootstrap intervals over tasks, the dashed line is the Pareto frontier; filled markers are core architectures, hollow markers are baselines (self-consistency\n"
           "with k samples, adaptive self-consistency, cascade), stars are the learned router over the core pool and over all methods. (b) Accuracy when each method must stay\n"
           "within a hard token budget B (routers use their cost estimates to exclude methods that do not fit). All results use a simulated solver, not a real model.")
    fig.text(0.012, 0.062, cap, ha="left", va="top", fontsize=8.8, linespacing=1.4)
    fig.savefig("docs/fig_pareto.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig("docs/fig_pareto.pdf", bbox_inches="tight", facecolor="white")
    Path("results/pareto.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
