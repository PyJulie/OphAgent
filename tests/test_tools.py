"""Tests for OphAgent Tool layer (registry, scheduler, base)."""
import pytest
from unittest.mock import MagicMock, patch
from ophagent.tools.base import ToolMetadata, BaseTool


class ConcreteTestTool(BaseTool):
    """Minimal concrete tool for testing."""
    def run(self, inputs):
        return {"result": "ok", "inputs": inputs}


@pytest.fixture
def sample_metadata():
    return ToolMetadata(
        tool_id="test_tool",
        name="Test Tool",
        description="A test tool",
        modality="CFP",
        task="classification",
        scheduling_mode="inline",
        input_type="image",
        output_type="dict",
    )


@pytest.fixture
def tool(sample_metadata):
    return ConcreteTestTool(sample_metadata)


# --- BaseTool tests ---

def test_tool_run_returns_dict(tool):
    result = tool.run({"image_path": "img.jpg"})
    assert isinstance(result, dict)
    assert result["result"] == "ok"


def test_tool_call_adds_defaults(tool):
    result = tool({"image_path": "img.jpg"})
    assert "tool_id" in result
    assert "success" in result
    assert result["tool_id"] == "test_tool"
    assert result["success"] is True


def test_tool_repr(tool):
    r = repr(tool)
    assert "test_tool" in r
    assert "inline" in r


# --- ToolMetadata tests ---

def test_metadata_from_yaml_dict():
    d = {
        "name": "CFP Quality",
        "description": "Assess quality",
        "modality": "CFP",
        "task": "quality_assessment",
        "scheduling_mode": "fastapi",
        "input_type": "image",
        "output_type": "dict",
        "fastapi_port": 8110,
        "newly_developed": True,
    }
    meta = ToolMetadata.from_yaml_dict("cfp_quality", d)
    assert meta.tool_id == "cfp_quality"
    assert meta.fastapi_port == 8110
    assert meta.newly_developed is True


# --- ToolRegistry tests ---

def test_registry_loads_yaml(tmp_path):
    import yaml
    from ophagent.tools.registry import ToolRegistry
    registry_data = {
        "tools": {
            "test_tool": {
                "name": "Test", "description": "Test tool",
                "modality": "CFP", "task": "classification",
                "scheduling_mode": "inline",
                "input_type": "image", "output_type": "dict",
            }
        }
    }
    yaml_file = tmp_path / "registry.yaml"
    yaml_file.write_text(yaml.dump(registry_data))
    registry = ToolRegistry(registry_path=yaml_file)
    assert registry.exists("test_tool")
    meta = registry.get("test_tool")
    assert meta.name == "Test"


def test_registry_raises_on_unknown():
    from ophagent.tools.registry import ToolRegistry
    registry = ToolRegistry.__new__(ToolRegistry)
    registry._tools = {}
    with pytest.raises(KeyError):
        registry.get("nonexistent_tool")
