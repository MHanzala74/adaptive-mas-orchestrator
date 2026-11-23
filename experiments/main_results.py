"""Main comparison: fixed architectures vs learned router vs oracle, with and without
an equal hard token budget. Seeds are separate runs; CIs bootstrap over (task, seed).

Usage: python experiments/main_results.py [--n-train 200] [--n-test 200] [--lam 0.3] [--budget 150]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mas.benchmarks.tasks import generate_synthetic
from mas.router.bandit import LinUCBRouter
from mas.router.train import collect_log, fit_full_info, fit_bandit, reward, oracle_choice
from mas.stats import bootstrap_ci, paired_diff_ci, fmt

ARCHS = ["single", "debate", "planner_worker"]


def per_item(choose, log, lam):
    rows = []
    for rec in log:
        o = rec["outcomes"][choose(rec)]
        rows.append((o["correct"], o["tokens"], reward(o["correct"], o["tokens"], lam)))
    return np.array(rows)


def ids(log):
    return [rec["task_id"] for rec in log]   # cluster = task (seeds of one task are not independent)


def run(n_train, n_test, lam, budget, seeds):
    train = generate_synthetic(n_train, seed=1)
    test = generate_synthetic(n_test, seed=100)
    out = []
    for bname, b in (("unconstrained", None), (f"budget={budget}", budget)):
        log_tr = collect_log(train, ARCHS, seeds=(0, 1), budget=b)
        log_te = collect_log(test, ARCHS, seeds=seeds, budget=b)
        full = fit_full_info(LinUCBRouter(ARCHS, alpha=0.0), log_tr, lam)
        bandit = fit_bandit(LinUCBRouter(ARCHS, alpha=0.3), log_tr, lam, epochs=3)
        pols = {a: (lambda rec, a=a: a) for a in ARCHS}
        pols["router (full-info)"] = lambda rec: full.select(np.array(rec["x"]), b, explore=False)
        pols["router (bandit)"] = lambda rec: bandit.select(np.array(rec["x"]), b, explore=False)
        pols["oracle"] = oracle_choice(lam)
        res = {n: per_item(c, log_te, lam) for n, c in pols.items()}
        cl = ids(log_te)
        best_fixed = max(ARCHS, key=lambda a: res[a][:, 2].mean())
        lines = [f"### {bname}  (lam={lam}, {len(seeds)} seeds x {n_test} tasks)", "",
                 "| method | accuracy [95% CI] | tokens/task | reward [95% CI] |", "|---|---|---|---|"]
        for n, r in res.items():
            lines.append(f"| {n} | {fmt(bootstrap_ci(r[:, 0], clusters=cl))} | {r[:, 1].mean():.0f} | {fmt(bootstrap_ci(r[:, 2], clusters=cl))} |")
        for n in ("router (full-info)", "router (bandit)"):
            d = paired_diff_ci(res[n][:, 2], res[best_fixed][:, 2], clusters=cl)
            lines.append(f"\n{n} - best fixed ({best_fixed}) reward: {fmt(d)}"
                         f" -> {'significant' if d[1] > 0 or d[2] < 0 else 'not significant'}")
        out.append("\n".join(lines))
    return "\n\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--tag", default="homogeneous")
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    a = ap.parse_args()
    md = run(a.n_train, a.n_test, a.lam, a.budget, tuple(a.seeds))
    Path("results").mkdir(exist_ok=True)
    Path(f"results/main_results_{a.tag}.md").write_text(f"## Mock environment: {a.tag}\n\n" + md + "\n")
    print(md)
