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
        from pathlib import Path
        script = str(get_settings().project_root / self.metadata.entry_script)
        if Path(script).resolve() == Path(__file__).resolve():
            raise RuntimeError("No standalone UWF multi-abnormality conda entrypoint configured.")
        return self._run_conda_script(
            script_path=script,
            inputs=inputs,
            conda_env=self.metadata.conda_env,
        )

    def fallback_run(self, inputs: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        from ophagent.utils.fallback_inference import uwf_multi_abnormality_prediction
        return uwf_multi_abnormality_prediction(inputs["image_path"], error=error)
