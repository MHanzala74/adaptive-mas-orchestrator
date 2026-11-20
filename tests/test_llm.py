import pytest
from mas.llm import MockLLM, TokenMeter, BudgetExceeded, parse_answer, safe_eval

def test_safe_eval():
    assert safe_eval("2+3*4") == 14
    with pytest.raises(ValueError):
        safe_eval("__import__('os')")

def test_meter_budget():
    llm, meter = MockLLM(), TokenMeter(budget=30)
    with pytest.raises(BudgetExceeded):
        for _ in range(10):
            llm.call(meter, "sys", "QUESTION: 2+2\nDIFFICULTY: 1")
    assert meter.calls >= 1

def test_parse_answer():
    assert parse_answer("blah\nANSWER: 1,234.") == "1234"

def test_mock_is_deterministic():
    a = MockLLM(seed=1).complete("s", "QUESTION: 5*5\nDIFFICULTY: 1").text
    b = MockLLM(seed=1).complete("s", "QUESTION: 5*5\nDIFFICULTY: 1").text
    assert a == b
