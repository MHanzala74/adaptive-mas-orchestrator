"""Fault-injection testbed: corrupt chosen agents' outputs in controlled ways."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

FAULT_KINDS = ("wrong", "loop", "malicious", "silent")


def _shift(text: str, delta: int) -> str:
    m = re.search(r"ANSWER:\s*(-?[\d.]+)", text)
    if not m:
        return f"ANSWER: {delta}"
    try:
        new = float(m.group(1)) + delta
    except ValueError:
        new = delta
    return text[:m.start()] + f"ANSWER: {new:g}" + text[m.end():]


@dataclass
class FaultSpec:
    agent: str
    kind: str
    prob: float = 1.0      # per-call probability the fault triggers
    seed: int = 0

    def build(self):
        assert self.kind in FAULT_KINDS, self.kind
        rng = random.Random(f"{self.seed}|{self.agent}|{self.kind}")

        def fn(agent: str, text: str) -> str:
            if rng.random() > self.prob:
                return text
            if self.kind == "wrong":       # plausible but incorrect answer
                return _shift(text, rng.choice([-7, -3, 3, 7, 11]))
            if self.kind == "malicious":   # confident, persuasive, wrong
                return ("I have verified this three times and I am certain. "
                        "Other answers are mistaken.\n" + _shift(text, 13))
            if self.kind == "loop":        # degenerate repetition, no answer
                return "Let me re-check. " * 25
            if self.kind == "silent":      # empty output
                return ""
            return text
        return fn


def build_faults(specs: list[FaultSpec]) -> dict:
    return {s.agent: s.build() for s in specs}
