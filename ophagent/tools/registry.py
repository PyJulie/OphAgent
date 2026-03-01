"""
Tool Registry for OphAgent.

Loads tool metadata from tool_registry.yaml and provides lookup by tool_id,
modality, or task type.  The registry is the single source of truth for
which tools are available and how they should be scheduled.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ophagent.tools.base import ToolMetadata
from ophagent.utils.logger import get_logger

logger = get_logger("tools.registry")


class ToolRegistry:
    """
    Loads and caches tool metadata from the YAML registry.

    Usage::

        registry = ToolRegistry()
        meta = registry.get("cfp_disease")
        all_cfp = registry.get_by_modality("CFP")
    """

    def __init__(self, registry_path: Optional[Path] = None):
        from config.settings import get_settings
        self._path = registry_path or get_settings().tools.tool_registry_path
        self._tools: Dict[str, ToolMetadata] = {}
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            logger.warning(f"Tool registry not found at {self._path}")
            return
        with open(self._path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for tool_id, d in data.get("tools", {}).items():
            self._tools[tool_id] = ToolMetadata.from_yaml_dict(tool_id, d)
        logger.info(f"Loaded {len(self._tools)} tools from registry.")

    def reload(self) -> None:
        """Reload registry from disk (useful after hot-updates)."""
        self._tools.clear()
        self._load()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, tool_id: str) -> ToolMetadata:
        if tool_id not in self._tools:
            raise KeyError(f"Tool '{tool_id}' not found in registry.")
        return self._tools[tool_id]

    def get_all(self) -> Dict[str, ToolMetadata]:
        return dict(self._tools)

    def get_by_modality(self, modality: str) -> List[ToolMetadata]:
        return [m for m in self._tools.values() if m.modality == modality]

    def get_by_task(self, task: str) -> List[ToolMetadata]:
        return [m for m in self._tools.values() if m.task == task]

    def get_newly_developed(self) -> List[ToolMetadata]:
        return [m for m in self._tools.values() if m.newly_developed]

    def list_ids(self) -> List[str]:
        return list(self._tools.keys())

    def exists(self, tool_id: str) -> bool:
        return tool_id in self._tools

    # ------------------------------------------------------------------
    # Description string for the Planner
    # ------------------------------------------------------------------

    def get_tool_descriptions(self, newly_developed_only: bool = False) -> str:
        tools = (
            self.get_newly_developed()
            if newly_developed_only
            else list(self._tools.values())
        )
        lines = []
        for meta in tools:
            lines.append(
                f"- {meta.tool_id}: {meta.description} "
                f"[modality={meta.modality}, task={meta.task}]"
            )
        return "\n".join(lines)
