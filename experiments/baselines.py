"""Equal-budget comparison of orchestrated architectures against strong baselines
(self-consistency, adaptive self-consistency, cascade), in three pre-specified mock environments.
Usage: python experiments/baselines.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *          # noqa: F401,F403
from mas.router.train import evaluate_policy, oracle_choice, reward
from mas.stats import bootstrap_ci, paired_diff_ci, fmt


def per_item(choose, log):
    return np.array([[(o := rec["outcomes"][choose(rec)])["correct"], o["tokens"], reward(o["correct"], o["tokens"], LAM)]
                     for rec in log])


def run():
    out = ["# Baselines vs orchestrated architectures (simulated solver)", "",
           f"3 seeds x 200 test tasks; reward = correct - {LAM} * tokens/300; CIs: 95% cluster bootstrap over tasks. "
           "`router (core)` chooses among the 5 core architectures; `router (all)` also among the baselines.", ""]
    summary = {}
    for env in ENVS:
        set_env(env)
        tr, te = build_logs()
        cl = [r["task_id"] for r in te]
        pols = {a: (lambda rec, a=a: a) for a in ALLM}
        pols["router (core)"] = router_policy(train_router(tr, CORE))
        pols["router (all)"] = router_policy(train_router(tr, ALLM))
        res = {n: per_item(c, te) for n, c in pols.items()}
        fixed_best = max(ALLM, key=lambda a: res[a][:, 2].mean())
        out += [f"## Environment: {env}", "", "| method | accuracy [95% CI] | tokens/task | reward [95% CI] |", "|---|---|---|---|"]
        for n, r in sorted(res.items(), key=lambda kv: kv[1][:, 1].mean()):
            out.append(f"| {n} | {fmt(bootstrap_ci(r[:, 0], clusters=cl))} | {r[:, 1].mean():.0f} | {fmt(bootstrap_ci(r[:, 2], clusters=cl))} |")
        out.append("")

        def diff(a, b, col=0):
            return paired_diff_ci(res[a][:, col], res[b][:, col], clusters=cl)

        def tag(d):
            return "significant" if d[1] > 0 or d[2] < 0 else "n.s."
        lines = [("accuracy", "debate", "sc7", 0), ("accuracy", "debate_judge", "sc7", 0), ("accuracy", "adaptive_sc", "debate", 0),
                 ("reward", "router (core)", fixed_best, 2), ("reward", "router (all)", fixed_best, 2)]
        out.append(f"Paired differences (best fixed by reward = `{fixed_best}`):")
        for what, a, b, col in lines:
            d = diff(a, b, col)
            out.append(f"- {what}: `{a}` - `{b}` = {fmt(d)} ({tag(d)})")
        out.append("")
        summary[env] = (fixed_best, diff("debate", "sc7"), diff("router (core)", fixed_best, 2))
    Path("results").mkdir(exist_ok=True)
    Path("results/baselines.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    run()
