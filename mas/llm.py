"""LLM abstraction with strict token accounting and budget enforcement.

Backends:
  - MockLLM: deterministic, offline, simulates a noisy solver (for tests/dev).
  - AnthropicLLM: real API calls (set LLM_BACKEND=anthropic).
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
import operator as op
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    pass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


def count_tokens(text: str) -> int:
    """Cheap, backend-independent token estimate (~1.3 tokens/word)."""
    return max(1, int(len(text.split()) * 1.3))


@dataclass
class TokenMeter:
    """Tracks usage for one task run; raises when the budget is exhausted."""
    budget: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def charge(self, resp: LLMResponse) -> None:
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        self.calls += 1
        if self.budget is not None and self.total > self.budget:
            raise BudgetExceeded(f"used {self.total} > budget {self.budget}")

    def remaining(self) -> int | None:
        return None if self.budget is None else max(0, self.budget - self.total)


class BaseLLM:
    def complete(self, system: str, prompt: str, temperature: float = 0.7) -> LLMResponse:
        raise NotImplementedError

    def call(self, meter: TokenMeter, system: str, prompt: str, temperature: float = 0.7) -> LLMResponse:
        resp = self.complete(system, prompt, temperature)
        meter.charge(resp)
        return resp


# ---------- Mock backend ----------
_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}


def safe_eval(expr: str) -> float:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        raise ValueError("unsafe expression")
    return ev(ast.parse(expr, mode="eval"))


@dataclass
class MockLLM(BaseLLM):
    """Simulated solver. Accuracy depends on task difficulty and on whether the
    prompt carries extra signal (peer answers for debate, a plan for planner-worker).
    This lets the full pipeline be developed and tested without API cost.
    """
    base_acc: float = 0.85
    difficulty_penalty: float = 0.18      # per difficulty level above 1
    debate_bonus: float = 0.10
    plan_bonus: float = 0.08
    seed: int = 0
    _n: int = field(default=0, init=False)

    def _rand(self, key: str) -> float:
        self._n += 1
        h = hashlib.sha256(f"{self.seed}|{self._n}|{key}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def complete(self, system: str, prompt: str, temperature: float = 0.7) -> LLMResponse:
        m = re.search(r"QUESTION:\s*(.+)", prompt)
        d = re.search(r"DIFFICULTY:\s*(\d+)", prompt)
        diff = int(d.group(1)) if d else 1
        if "ROLE:ATTRIBUTOR" in system:
            text = self._attribute(prompt)
        elif not m:
            text = "ANSWER: 0"
        else:
            expr = m.group(1).strip()
            try:
                truth = safe_eval(expr)
            except Exception:
                truth = 0
            p = self.base_acc - self.difficulty_penalty * (diff - 1)
            if "OTHER ANSWERS" in prompt:
                p += self.debate_bonus
            if "PLAN:" in prompt:
                p += self.plan_bonus
            p = min(0.99, max(0.02, p))
            if self._rand(expr) < p:
                ans = truth
            else:
                ans = truth + int(self._rand(expr + "x") * 20) - 10 or truth + 1
            text = f"Reasoning: evaluate step by step.\nANSWER: {ans:g}"
        return LLMResponse(text, count_tokens(system + prompt), count_tokens(text))

    def _attribute(self, prompt: str) -> str:
        # naive heuristic attributor for offline dev: blame the first agent whose
        # answer disagrees with the majority, else the last agent.
        agents = re.findall(r"\[(\w+)\] ANSWER: (\S+)", prompt)
        if not agents:
            return "BLAME: unknown\nREASON: no trace"
        vals = [a[1] for a in agents]
        maj = max(set(vals), key=vals.count)
        for name, v in agents:
            if v != maj:
                return f"BLAME: {name}\nREASON: disagrees with majority"
        return f"BLAME: {agents[-1][0]}\nREASON: default"


# ---------- Anthropic backend ----------
class AnthropicLLM(BaseLLM):
    def __init__(self, model: str | None = None):
        import anthropic  # lazy import
        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("MODEL_NAME", "claude-haiku-5-5")

    def complete(self, system: str, prompt: str, temperature: float = 0.7) -> LLMResponse:
        r = self.client.messages.create(
            model=self.model, max_tokens=600, temperature=temperature,
            system=system, messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in r.content if b.type == "text")
        return LLMResponse(text, r.usage.input_tokens, r.usage.output_tokens)


def get_llm(seed: int = 0) -> BaseLLM:
    if os.getenv("LLM_BACKEND", "mock") == "anthropic":
        return AnthropicLLM()
    return MockLLM(seed=seed)


def parse_answer(text: str) -> str | None:
    m = re.search(r"ANSWER:\s*(-?[\d.,]+)", text)
    return m.group(1).replace(",", "").rstrip(".") if m else None
