"""Tests for OphAgent Executor."""
import pytest
from unittest.mock import MagicMock
from ophagent.core.executor import Executor, StepResult
from ophagent.core.planner import ExecutionPlan, PlanStep


@pytest.fixture
def mock_scheduler():
    sched = MagicMock()
    sched.run.return_value = {"label": "Normal", "confidence": 0.95}
    return sched


@pytest.fixture
def executor(mock_scheduler):
    return Executor(scheduler=mock_scheduler)


def make_plan(steps):
    return ExecutionPlan(query="test", steps=steps)


def test_execute_single_step(executor):
    plan = make_plan([PlanStep(1, "cfp_quality", {"image_path": "img.jpg"}, "Check quality", [])])
    results = executor.execute(plan)
    assert 1 in results
    assert results[1].success


def test_execute_skips_synthesise_step(executor):
    plan = make_plan([
        PlanStep(1, "cfp_quality", {"image_path": "img.jpg"}, "quality", []),
        PlanStep(2, "synthesise", {"summary": "report"}, "synth", [1]),
    ])
    results = executor.execute(plan)
    assert 1 in results
    assert 2 not in results


def test_failed_dependency_skips_step(executor):
    executor.scheduler.run.side_effect = [RuntimeError("Tool failed"), None]
    plan = make_plan([
        PlanStep(1, "cfp_quality", {}, "quality", []),
        PlanStep(2, "cfp_disease", {}, "disease", [1]),
    ])
    results = executor.execute(plan)
    assert not results[1].success
    assert not results[2].success


def test_resolve_inputs_static(executor):
    inputs = {"image_path": "test.jpg", "threshold": 0.5}
    resolved = executor._resolve_inputs(inputs, {})
    assert resolved == inputs


def test_format_results(executor):
    results = {
        1: StepResult(1, "cfp_quality", output={"label": "Good"}),
        2: StepResult(2, "cfp_disease", error="Timeout"),
    }
    formatted = Executor.format_results(results)
    assert "cfp_quality" in formatted
    assert "ERROR" in formatted
