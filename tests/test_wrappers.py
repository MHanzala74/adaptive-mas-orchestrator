import pytest
from types import SimpleNamespace as NS
from mas.llm import BaseLLM, LLMResponse, TokenMeter, parse_answer
from mas.llm_wrappers import (CachedLLM, LimitedLLM, RetryingLLM, SpendGuard, SpendLimitExceeded, is_retryable)


class Flaky(BaseLLM):
    def __init__(self, fail_times, exc):
        self.n, self.fail, self.exc = 0, fail_times, exc

    def complete(self, system, prompt, temperature=0.7):
        self.n += 1
        if self.n <= self.fail:
            raise self.exc
        return LLMResponse(f"ANSWER: {self.n}", 10, 5)


class E429(Exception):
    status_code = 429

class E401(Exception):
    status_code = 401


def test_retry_recovers_from_transient_and_sleeps():
    sleeps = []
    llm = RetryingLLM(Flaky(2, E429()), sleep=sleeps.append)
    assert llm.complete("s", "p").text == "ANSWER: 3" and len(sleeps) == 2 and llm.retries == 2

def test_retry_does_not_retry_auth_errors():
    inner = Flaky(5, E401())
    with pytest.raises(E401):
        RetryingLLM(inner, sleep=lambda s: None).complete("s", "p")
    assert inner.n == 1 and not is_retryable(E401()) and is_retryable(E429())

def test_retry_gives_up_after_max_attempts():
    inner = Flaky(99, E429())
    with pytest.raises(E429):
        RetryingLLM(inner, max_attempts=3, sleep=lambda s: None).complete("s", "p")
    assert inner.n == 3

def test_spend_limit_stops_before_overspend():
    g = SpendGuard(max_tokens=40, price_in=1.0, price_out=2.0)
    llm = LimitedLLM(Flaky(0, None), g)
    llm.complete("s", "p"); llm.complete("s", "p")           # 15 tokens each = 30
    llm.complete("s", "p")                                   # 45 >= 40
    with pytest.raises(SpendLimitExceeded):
        llm.complete("s", "p")
    assert g.calls == 3 and g.tokens_in == 30 and g.tokens_out == 15
    assert g.cost_usd == pytest.approx((30 * 1.0 + 15 * 2.0) / 1e6)

def test_cache_replays_and_keeps_samples_distinct(tmp_path):
    inner = Flaky(0, None)
    path = str(tmp_path / "c.sqlite")
    a = CachedLLM(inner, path, "m", seed=1)
    first = [a.complete("s", "same prompt", 0.8).text for _ in range(3)]
    assert len(set(first)) == 3 and inner.n == 3             # three distinct samples, three real calls
    b = CachedLLM(inner, path, "m", seed=1)                  # "resume": same seed replays for free
    assert [b.complete("s", "same prompt", 0.8).text for _ in range(3)] == first and inner.n == 3 and b.hits == 3
    c = CachedLLM(inner, path, "m", seed=2)                  # different seed -> fresh samples
    c.complete("s", "same prompt", 0.8)
    assert inner.n == 4

def test_cache_hit_still_charges_budget_meter(tmp_path):
    llm = CachedLLM(Flaky(0, None), str(tmp_path / "c.sqlite"), "m")
    m = TokenMeter(); llm.call(m, "s", "p"); assert m.total == 15

def test_parse_answer_is_robust():
    assert parse_answer("so\nANSWER: $1,234.50") == "1234.50"
    assert parse_answer("**ANSWER: 18**") == "18"
    assert parse_answer("ANSWER: 7 dollars\nANSWER: 8") == "8"
    assert parse_answer("ANSWER: 5.0") == "5"
    assert parse_answer("no answer here") is None
