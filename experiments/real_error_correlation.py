"""Measure error correlation of a real model (or the mock, with --synthetic) and calibrate MOCK_CORR.

For each task, draw k samples from the single-solver prompt at temperature T, then report accuracy,
majority-vote accuracy, oracle (any-correct) accuracy and the shared-wrong rate with a cluster-bootstrap CI.

  python experiments/real_error_correlation.py --data data/gsm8k_test.jsonl --n 100 --k 5 --dry-run
  LLM_BACKEND=anthropic ANTHROPIC_API_KEY=... MAX_API_TOKENS=400000 python experiments/real_error_correlation.py --data data/gsm8k_test.jsonl --n 100
Re-running is free for finished tasks (replay cache) and stops cleanly at the spend limit."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mas.architectures.impl import SOLVER, _q
from mas.benchmarks.tasks import generate_synthetic, load_gsm8k_jsonl
from mas.calibration import fit_mock_corr, per_task_pairs, ratio_ci, task_stats
from mas.llm import TokenMeter, count_tokens, get_llm, parse_answer


def estimate(tasks, k, out_tokens=250):
    tin = sum(count_tokens(SOLVER + _q(t)) for t in tasks) * k
    return tin, len(tasks) * k * out_tokens


def collect(tasks, k, temperature):
    answers = []
    for i, t in enumerate(tasks):
        llm = get_llm(seed=i)                       # seed i -> cached, resumable, distinct per task
        answers.append([parse_answer(llm.call(TokenMeter(), SOLVER, _q(t), temperature).text) for _ in range(k)])
    return answers


def run(tasks, k=5, temperature=0.8):
    answers = collect(tasks, k, temperature)
    st = task_stats(answers, tasks)
    num, den = per_task_pairs(answers, tasks)
    if den.sum() == 0:
        return {**st, "shared_wrong": None}
    m, lo, hi = ratio_ci(num, den)
    fit = fit_mock_corr(m, k=k)
    return {**st, "shared_wrong": (m, lo, hi), "wrong_pairs": int(den.sum()), "fit": fit}


def report(r, source, k, temperature, guard_summary=""):
    md = [f"# Error correlation ({source})", "", f"{r['n_tasks']} tasks x k={k} samples, temperature {temperature}.", "",
          "| statistic | value |", "|---|---|",
          f"| accuracy per sample | {r['acc_per_sample']:.3f} |",
          f"| accuracy of k-sample majority vote | {r['acc_majority_vote']:.3f} |",
          f"| accuracy if any sample is right (oracle) | {r['acc_any_correct']:.3f} |"]
    if r["shared_wrong"] is None:
        md.append("| shared-wrong rate | n/a (no task had two wrong samples) |")
    else:
        m, lo, hi = r["shared_wrong"]
        md += [f"| shared-wrong rate [95% cluster CI] | {m:.3f} [{lo:.3f}, {hi:.3f}] ({r['wrong_pairs']} wrong pairs) |",
               f"| calibrated `MOCK_CORR` | **{r['fit']['mock_corr']}**{'' if r['fit']['in_range'] else ' (outside the mock range; treat as a bound)'} |"]
        md += ["", "Next step: re-run the simulated comparison at the measured correlation:", "",
               f"```bash\nMOCK_CORR={r['fit']['mock_corr']} MOCK_HETERO=1 python experiments/baselines.py\n```", "",
               "Caveat: the mock's wrong answers come from a small synthetic distribution, so the mapping is approximate; real wrong answers "
               "also collide by chance on small numbers. Treat the calibrated value as a plausible range, not a precise estimate."]
    if guard_summary:
        md += ["", f"API usage: {guard_summary}"]
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="GSM8K-format jsonl")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic arithmetic (works with the mock backend)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--dry-run", action="store_true", help="estimate tokens/cost only; make no calls")
    ap.add_argument("--out", default="results/real_error_correlation.md")
    a = ap.parse_args()
    backend = os.getenv("LLM_BACKEND", "mock")
    if a.synthetic:
        tasks, source = generate_synthetic(a.n, seed=500), "synthetic tasks"
    elif a.data:
        tasks, source = load_gsm8k_jsonl(a.data, limit=a.n), f"{Path(a.data).name}"
    else:
        sys.exit("give --data <gsm8k.jsonl> or --synthetic")
    tin, tout = estimate(tasks, a.k)
    from mas.llm_wrappers import GUARD
    pi, po = GUARD.price_in, GUARD.price_out
    print(f"Estimate: {len(tasks)} tasks x {a.k} samples = {len(tasks) * a.k} calls, ~{tin} input + ~{tout} output tokens"
          + (f", ~${(tin * pi + tout * po) / 1e6:.2f}" if pi or po else " (set PRICE_IN_PER_M / PRICE_OUT_PER_M for a $ estimate)"))
    if a.dry_run:
        sys.exit(0)
    if backend == "mock" and not a.synthetic:
        sys.exit("The mock backend cannot solve word problems. Set LLM_BACKEND=anthropic|openai, or use --synthetic.")
    r = run(tasks, a.k, a.temperature)
    md = report(r, f"{backend} on {source}", a.k, a.temperature, GUARD.summary() if backend != "mock" else "")
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(md)
    print(md)
