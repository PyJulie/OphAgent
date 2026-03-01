"""UWF Multi-Abnormality Screening (reused model)."""
from __future__ import annotations
from typing import Any, Dict
from ophagent.tools.base import BaseTool, CondaToolMixin, ToolMetadata


class UWFMultiTool(CondaToolMixin, BaseTool):
    """
    Ultra-widefield multi-abnormality screening model.
    Input:  {"image_path": str}
    Output: {"labels": list[str], "probabilities": dict[str, float]}
    """
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from config.settings import get_settings
        script = str(get_settings().project_root / self.metadata.entry_script)
        return self._run_conda_script(
            script_path=script,
            inputs=inputs,
            conda_env=self.metadata.conda_env,
        )
