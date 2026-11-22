"""Append-only JSONL log of every orchestrated run (for audit and offline analysis)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from mas.core import RunResult


def append_run(path: str | Path, res: RunResult, extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.time(), **res.to_dict(), **(extra or {})}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def read_runs(path: str | Path) -> list[dict]:
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
