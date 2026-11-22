"""Budget-aware LinUCB router over architectures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mas.router.features import DIM


class LinUCBRouter:
    def __init__(self, arms: list[str], alpha: float = 0.5, reg: float = 1.0, dim: int = DIM):
        self.arms, self.alpha = list(arms), alpha
        self.A = {a: reg * np.eye(dim) for a in arms}
        self.b = {a: np.zeros(dim) for a in arms}
        self.cost_sum = {a: 0.0 for a in arms}
        self.cost_n = {a: 0 for a in arms}

    def theta(self, a: str) -> np.ndarray:
        return np.linalg.solve(self.A[a], self.b[a])

    def expected_cost(self, a: str) -> float:
        return self.cost_sum[a] / self.cost_n[a] if self.cost_n[a] else 0.0

    def scores(self, x: np.ndarray, explore: bool = True) -> dict[str, float]:
        out = {}
        for a in self.arms:
            mean = float(x @ self.theta(a))
            bonus = self.alpha * float(np.sqrt(x @ np.linalg.solve(self.A[a], x))) if explore else 0.0
            out[a] = mean + bonus
        return out

    def select(self, x: np.ndarray, budget: int | None = None, explore: bool = True) -> str:
        sc = self.scores(x, explore)
        if budget is not None:   # exclude arms whose average cost exceeds the budget
            feasible = [a for a in self.arms if self.expected_cost(a) <= budget]
            if feasible:
                sc = {a: sc[a] for a in feasible}
        return max(sc, key=sc.get)

    def update(self, arm: str, x: np.ndarray, reward: float, tokens: float) -> None:
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.cost_sum[arm] += tokens
        self.cost_n[arm] += 1

    # ---- persistence ----
    def to_dict(self) -> dict:
        return {"arms": self.arms, "alpha": self.alpha,
                "A": {a: self.A[a].tolist() for a in self.arms}, "b": {a: self.b[a].tolist() for a in self.arms},
                "cost_sum": self.cost_sum, "cost_n": self.cost_n}

    @classmethod
    def from_dict(cls, d: dict) -> "LinUCBRouter":
        dim = len(d["b"][d["arms"][0]])
        r = cls(d["arms"], alpha=d["alpha"], dim=dim)
        r.A = {a: np.array(d["A"][a]) for a in r.arms}
        r.b = {a: np.array(d["b"][a]) for a in r.arms}
        r.cost_sum, r.cost_n = dict(d["cost_sum"]), dict(d["cost_n"])
        return r

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "LinUCBRouter":
        return cls.from_dict(json.loads(Path(path).read_text()))
