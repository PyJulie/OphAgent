"""Tests for OphAgent Verifier."""
import pytest
from unittest.mock import MagicMock
from ophagent.core.verifier import Verifier, Verdict
from ophagent.core.executor import StepResult


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_json.return_value = {
        "valid": True,
        "confidence": 0.92,
        "conflicts": [],
        "resolution": "",
        "verified_result": {"label": "DR", "confidence": 0.92},
    }
    return llm


@pytest.fixture
def mock_kb():
    kb = MagicMock()
    kb.retrieve.return_value = "No relevant context."
    return kb


@pytest.fixture
def verifier(mock_llm, mock_kb):
    v = Verifier(llm=mock_llm)
    v._kb = mock_kb
    return v


def make_results(tool_name="cfp_disease", output=None, error=None):
    return {1: StepResult(1, tool_name, output=output or {"label": "DR"}, error=error)}


def test_verify_returns_verdict(verifier):
    results = make_results()
    verdict = verifier.verify(results, query="Check for DR")
    assert isinstance(verdict, Verdict)
    assert verdict.valid is True
    assert verdict.confidence > 0


def test_low_confidence_flags_human_review(verifier):
    verifier.llm.chat_json.return_value = {
        "valid": True, "confidence": 0.3,
        "conflicts": [], "resolution": "",
        "verified_result": {},
    }
    results = make_results()
    verdict = verifier.verify(results, query="test")
    assert verdict.needs_human_review is True


def test_quality_gate_passes_good_image():
    assert Verifier.check_quality_gate({"quality_score": 0.9, "quality_label": "Good"}) is True


def test_quality_gate_fails_ungradable():
    assert Verifier.check_quality_gate({"quality_score": 0.2, "quality_label": "ungradable"}) is False


def test_quality_gate_passes_none():
    assert Verifier.check_quality_gate(None) is True


def test_verdict_from_dict():
    d = {
        "valid": True, "confidence": 0.85,
        "conflicts": ["minor conflict"],
        "resolution": "resolved",
        "verified_result": {"label": "AMD"},
    }
    v = Verdict.from_dict(d)
    assert v.valid is True
    assert v.confidence == 0.85
    assert v.conflicts == ["minor conflict"]
