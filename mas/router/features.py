"""Observable task features (no access to ground-truth difficulty)."""
from __future__ import annotations

import math
import re

import numpy as np

KEYWORDS = ("total", "each", "per", "more", "less", "times", "half", "twice", "remain", "left", "difference")
DIM = 7


def featurize(question: str) -> np.ndarray:
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", question)]
    n_ops = len(re.findall(r"[+\-*/]", question)) + sum(question.lower().count(k) for k in KEYWORDS)
    words = len(question.split())
    return np.array([
        1.0,                                         # bias
        min(len(nums), 10) / 10,
        min(n_ops, 10) / 10,
        1.0 if "(" in question else 0.0,
        min(words, 120) / 120,
        math.log10(max(nums) + 1) / 4 if nums else 0.0,
        min(sum(c.isdigit() for c in question), 60) / 60,
    ])
