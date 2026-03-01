"""AutoMorph - CFP vasculature segmentation and quantification (reused model)."""
from __future__ import annotations
from typing import Any, Dict
from ophagent.tools.base import BaseTool, CondaToolMixin, ToolMetadata


class AutoMorphTool(CondaToolMixin, BaseTool):
    """
    Segments retinal vasculature and computes morphological metrics (fractal
    dimension, vessel density, A/V ratio, etc.) via the AutoMorph pipeline.
    Runs in an isolated Conda environment.
    Input:  {"image_path": str, "output_dir": str (optional)}
    Output: {"vessel_mask_path": str, "metrics": dict}
    """
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from config.settings import get_settings
        script = str(get_settings().project_root / self.metadata.entry_script)
        return self._run_conda_script(
            script_path=script,
            inputs=inputs,
            conda_env=self.metadata.conda_env,
        )
