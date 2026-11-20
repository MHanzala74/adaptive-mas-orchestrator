"""Budget-aware LinUCB router over architectures."""
from __future__ import annotations

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
