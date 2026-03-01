"""Tests for OphAgent Planner."""
import pytest
from unittest.mock import MagicMock, patch
from ophagent.core.planner import Planner, ExecutionPlan, PlanStep


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_json.return_value = {
        "steps": [
            {"step_id": 1, "tool_name": "cfp_quality", "inputs": {"image_path": "test.jpg"},
             "description": "Assess quality", "depends_on": []},
            {"step_id": 2, "tool_name": "cfp_disease", "inputs": {"image_path": "test.jpg"},
             "description": "Classify disease", "depends_on": [1]},
            {"step_id": 3, "tool_name": "synthesise", "inputs": {"summary": "Summarise"},
             "description": "Generate report", "depends_on": [1, 2]},
        ]
    }
    return llm


@pytest.fixture
def planner(mock_llm):
    p = Planner(llm=mock_llm)
    p._tool_descriptions = "mock tool descriptions"
    return p


def test_plan_returns_execution_plan(planner):
    plan = planner.plan("Analyse this fundus image", image_paths=["test.jpg"])
    assert isinstance(plan, ExecutionPlan)
    assert len(plan.steps) == 3


def test_plan_step_types(planner):
    plan = planner.plan("Test query")
    for step in plan.steps:
        assert isinstance(step, PlanStep)
        assert isinstance(step.step_id, int)
        assert isinstance(step.tool_name, str)
        assert isinstance(step.inputs, dict)
        assert isinstance(step.depends_on, list)


def test_topological_order_respects_dependencies(planner):
    plan = planner.plan("Test query")
    ordered = plan.topological_order()
    # synthesise (depends on 1,2) should come last
    non_synth = [s for s in ordered if s.tool_name != "synthesise"]
    synth = [s for s in ordered if s.tool_name == "synthesise"]
    assert len(synth) == 1
    assert all(s.step_id in [1, 2] for s in non_synth)


def test_plan_with_no_steps_returns_fallback(planner):
    planner.llm.chat_json.return_value = {"steps": []}
    plan = planner.plan("Empty plan test")
    assert len(plan.steps) >= 1
    assert plan.steps[0].tool_name == "web_search"


def test_replan_amends_query(planner):
    original_plan = ExecutionPlan(query="Original", steps=[])
    planner.replan(original_plan, "conflicting grades")
    call_args = planner.llm.chat_json.call_args
    user_msg = call_args[0][0][0]["content"]
    assert "conflicting grades" in user_msg
