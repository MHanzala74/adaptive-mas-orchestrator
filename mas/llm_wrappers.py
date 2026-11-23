"""Safety wrappers for real LLM backends: retry with backoff, hard spend limit, on-disk cache.

Stack (outermost first):  CachedLLM -> LimitedLLM -> RetryingLLM -> real backend
  * Cache hits cost nothing and skip the limit/retry layers, so re-running a script resumes for free.
  * The limit counts only real (non-cached) API usage and stops the process before it overspends.
"""
from __future__ import annotations

import hashlib
import os
import random
import sqlite3
import threading
import time
from pathlib import Path

from mas.llm import BaseLLM, LLMResponse

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
RETRYABLE_NAMES = {"APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError",
                   "ServiceUnavailableError", "Timeout", "ConnectionError", "TimeoutError", "OverloadedError"}


class SpendLimitExceeded(Exception):
    pass


def is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in RETRYABLE_STATUS          # 400/401/403/404 are never retried
    return type(exc).__name__ in RETRYABLE_NAMES


class RetryingLLM(BaseLLM):
    def __init__(self, inner: BaseLLM, max_attempts: int = 5, base_delay: float = 1.0, max_delay: float = 30.0, sleep=time.sleep):
        self.inner, self.max_attempts, self.base_delay, self.max_delay, self.sleep = inner, max_attempts, base_delay, max_delay, sleep
        self.retries = 0

    def complete(self, system: str, prompt: str, temperature: float = 0.7) -> LLMResponse:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.inner.complete(system, prompt, temperature)
            except Exception as e:                       # noqa: BLE001
                if attempt == self.max_attempts or not is_retryable(e):
                    raise
                self.retries += 1
                self.sleep(min(self.max_delay, self.base_delay * 2 ** (attempt - 1)) * (0.5 + random.random() / 2))
        raise RuntimeError("unreachable")


class SpendGuard:
    """Process-wide counter of real API usage with hard caps (tokens and/or calls)."""

    def __init__(self, max_tokens: int | None = None, max_calls: int | None = None,
                 price_in: float = 0.0, price_out: float = 0.0):
        self.max_tokens, self.max_calls = max_tokens, max_calls
        self.price_in, self.price_out = price_in, price_out          # USD per million tokens
        self.tokens_in = self.tokens_out = self.calls = 0
        self._lock = threading.Lock()

    def check(self) -> None:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise SpendLimitExceeded(f"call limit reached ({self.calls}/{self.max_calls})")
        if self.max_tokens is not None and self.tokens_in + self.tokens_out >= self.max_tokens:
            raise SpendLimitExceeded(f"token limit reached ({self.tokens_in + self.tokens_out}/{self.max_tokens})")

    def record(self, r: LLMResponse) -> None:
        with self._lock:
            self.tokens_in += r.input_tokens; self.tokens_out += r.output_tokens; self.calls += 1

    @property
    def cost_usd(self) -> float:
        return (self.tokens_in * self.price_in + self.tokens_out * self.price_out) / 1e6

    def summary(self) -> str:
        return f"{self.calls} real calls, {self.tokens_in} in + {self.tokens_out} out tokens, ~${self.cost_usd:.4f}"

    @classmethod
    def from_env(cls) -> "SpendGuard":
        f = lambda k: float(os.environ[k]) if os.environ.get(k) else None      # noqa: E731
        mt, mc = f("MAX_API_TOKENS"), f("MAX_API_CALLS")
        return cls(int(mt) if mt else None, int(mc) if mc else None, f("PRICE_IN_PER_M") or 0.0, f("PRICE_OUT_PER_M") or 0.0)


GUARD = SpendGuard.from_env()          # shared by every wrapped LLM in this process


class LimitedLLM(BaseLLM):
    def __init__(self, inner: BaseLLM, guard: SpendGuard | None = None):
        self.inner, self.guard = inner, guard or GUARD

    def complete(self, system: str, prompt: str, temperature: float = 0.7) -> LLMResponse:
        self.guard.check()
        r = self.inner.complete(system, prompt, temperature)
        self.guard.record(r)
        return r


class _Store:
    """Tiny SQLite key-value store (thread-safe via a lock)."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, text TEXT, tin INT, tout INT)")
        self.lock = threading.Lock()

    def get(self, k):
        with self.lock:
            return self.db.execute("SELECT text, tin, tout FROM c WHERE k=?", (k,)).fetchone()

    def put(self, k, r: LLMResponse):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO c VALUES (?,?,?,?)", (k, r.text, r.input_tokens, r.output_tokens))
            self.db.commit()


_STORES: dict[str, _Store] = {}


class CachedLLM(BaseLLM):
    """Deterministic replay cache.

    The key includes the run seed and an occurrence counter, so k identical sampling calls inside
    one run get k *different* stored samples (self-consistency stays meaningful), while re-running
    the same seed replays them exactly (resume / reproducibility)."""

    def __init__(self, inner: BaseLLM, path: str, model_tag: str, seed: int = 0):
        self.inner, self.model_tag, self.seed = inner, model_tag, seed
        self.store = _STORES.setdefault(path, _Store(path))
        self._seen: dict[str, int] = {}
        self.hits = self.misses = 0

    def complete(self, system: str, prompt: str, temperature: float = 0.7) -> LLMResponse:
        base = hashlib.sha256(f"{self.model_tag}|{system}|{prompt}|{temperature}".encode()).hexdigest()
        n = self._seen[base] = self._seen.get(base, -1) + 1
        key = hashlib.sha256(f"{base}|{self.seed}|{n}".encode()).hexdigest()
        row = self.store.get(key)
        if row:
            self.hits += 1
            return LLMResponse(row[0], row[1], row[2])
        self.misses += 1
        r = self.inner.complete(system, prompt, temperature)
        self.store.put(key, r)
        return r


def wrap(base: BaseLLM, model_tag: str, seed: int = 0) -> BaseLLM:
    """Apply retry, spend limit and cache (disable the cache with LLM_CACHE=0)."""
    llm: BaseLLM = LimitedLLM(RetryingLLM(base))
    if os.getenv("LLM_CACHE", "1") != "0":
        llm = CachedLLM(llm, os.getenv("LLM_CACHE_PATH", "results/cache/llm_cache.sqlite"), model_tag, seed)
    return llm
