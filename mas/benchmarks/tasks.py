"""Task definition, checking, synthetic generator and GSM8K-style loader."""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from mas.llm import safe_eval


@dataclass
class Task:
    id: str
    question: str          # for synthetic tasks: an arithmetic expression
    answer: str            # canonical numeric answer as string
    difficulty: int = 1    # 1 (easy) .. 4 (hard)
    domain: str = "arith"

    def to_dict(self):
        return asdict(self)


def _norm(x: str) -> float | None:
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return None


def check(task: Task, predicted: str | None) -> bool:
    if predicted is None:
        return False
    a, b = _norm(task.answer), _norm(predicted)
    return a is not None and b is not None and abs(a - b) < 1e-6


def _expr(rng: random.Random, n_ops: int) -> str:
    parts = [str(rng.randint(2, 30))]
    for _ in range(n_ops):
        parts.append(rng.choice("+-*"))
        parts.append(str(rng.randint(2, 20)))
    e = " ".join(parts)
    if n_ops >= 3:  # add parentheses for harder items
        i = e.find(" ")
        j = e.find(" ", e.find(" ", i + 1) + 1)
        e = f"({e[:j]}){e[j:]}"
    return e


def generate_synthetic(n: int, seed: int = 0) -> list[Task]:
    """Arithmetic expressions; difficulty d uses d+1 operations."""
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        d = 1 + i % 4
        e = _expr(rng, d + 1)
        ans = safe_eval(e)
        tasks.append(Task(f"syn-{seed}-{i}", e, f"{ans:g}", d, "arith"))
    return tasks


def load_gsm8k_jsonl(path: str | Path, limit: int | None = None) -> list[Task]:
    """Load GSM8K-format jsonl: {"question": ..., "answer": "... #### 42"}."""
    tasks = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if limit and i >= limit:
            break
        r = json.loads(line)
        m = re.search(r"####\s*(-?[\d,\.]+)", r["answer"])
        if not m:
            continue
        q = r["question"]
        diff = min(4, 1 + len(re.findall(r"\d+", q)) // 3)
        tasks.append(Task(f"gsm-{i}", q, m.group(1).replace(",", ""), diff, "gsm8k"))
    return tasks


def split(tasks: list[Task], train_frac: float = 0.5, seed: int = 0):
    rng = random.Random(seed)
    t = tasks[:]
    rng.shuffle(t)
    k = int(len(t) * train_frac)
    return t[:k], t[k:]
