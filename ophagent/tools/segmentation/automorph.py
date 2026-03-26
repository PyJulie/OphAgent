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
        from pathlib import Path
        script = str(get_settings().project_root / self.metadata.entry_script)
        if Path(script).resolve() == Path(__file__).resolve():
            raise RuntimeError("No standalone AutoMorph conda entrypoint configured.")
        return self._run_conda_script(
            script_path=script,
            inputs=inputs,
            conda_env=self.metadata.conda_env,
        )

    def fallback_run(self, inputs: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        from ophagent.utils.fallback_inference import automorph_prediction
        return automorph_prediction(inputs["image_path"], error=error)
