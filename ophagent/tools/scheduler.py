"""
Tool Scheduler for OphAgent.

Routes tool execution requests to the correct backend:
  - "inline"  -> instantiate and call the tool's Python class directly
  - "fastapi" -> HTTP POST to the pre-loaded Docker/FastAPI service
  - "conda"   -> launch an on-demand Conda subprocess

The scheduler also manages a simple in-process tool instance cache
to avoid re-loading large models on every call.
"""
from __future__ import annotations

import importlib
import threading
from typing import Any, Dict, Optional

from ophagent.tools.base import BaseTool, ToolMetadata
from ophagent.tools.registry import ToolRegistry
from ophagent.utils.logger import get_logger

logger = get_logger("tools.scheduler")

# Mapping from tool_id to the Python module path of its implementation class
# Format: "tools/<subdir>/<module>" -> "ophagent.tools.<subdir>.<module>:<ClassName>"
_TOOL_CLASS_MAP: Dict[str, str] = {
    # --- Reused ---
    "fmue":                  "ophagent.tools.classification.oct_fmue:OCTFMUETool",
    "uwf_mdd":               "ophagent.tools.classification.uwf_mdd:UWFMDDTool",
    "uwf_multi_abnormality": "ophagent.tools.classification.uwf_multi_abnormality:UWFMultiTool",
    "retizero":              "ophagent.tools.clip_models.retizero:RetiZeroTool",
    "vilref":                "ophagent.tools.clip_models.vilref:ViLReFTool",
    "fundus_expert":         "ophagent.tools.vqa.fundus_expert:FundusExpertTool",
    "vision_unite":          "ophagent.tools.vqa.vision_unite:VisionUniteTool",
    "automorph":             "ophagent.tools.segmentation.automorph:AutoMorphTool",
    "retsam":                "ophagent.tools.segmentation.retsam:RetSAMTool",
    # --- Newly developed ---
    "cfp_quality":           "ophagent.tools.classification.cfp_quality:CFPQualityTool",
    "cfp_disease":           "ophagent.tools.classification.cfp_disease:CFPDiseaseTool",
    "cfp_ffa_multimodal":    "ophagent.tools.classification.cfp_ffa_multimodal:CFPFFATool",
    "uwf_quality_disease":   "ophagent.tools.classification.uwf_quality_disease:UWFQualityDiseaseTool",
    "cfp_glaucoma":          "ophagent.tools.classification.cfp_glaucoma:CFPGlaucomaTool",
    "cfp_pdr":               "ophagent.tools.classification.cfp_pdr:CFPPDRTool",
    "ffa_lesion":            "ophagent.tools.detection.ffa_lesion:FFALesionTool",
    "disc_fovea":            "ophagent.tools.detection.disc_fovea:DiscFoveaTool",
    # --- Auxiliary ---
    "gradcam":               "ophagent.tools.auxiliary.gradcam:GradCAMTool",
    "roi_cropping":          "ophagent.tools.auxiliary.roi_cropping:ROICroppingTool",
    "ocr_detector":          "ophagent.tools.auxiliary.ocr_detector:OCRDetectorTool",
    "web_search":            "ophagent.tools.auxiliary.web_search:WebSearchTool",
}


class ToolScheduler:
    """
    Central dispatcher for all tool calls.

    Thread-safe: uses a lock per tool_id to prevent concurrent model loading.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or ToolRegistry()
        self._instances: Dict[str, BaseTool] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    def run(self, tool_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute *tool_id* with *inputs*, routing to the appropriate backend.
        """
        if not self.registry.exists(tool_id):
            raise KeyError(f"Unknown tool: {tool_id!r}")

        meta = self.registry.get(tool_id)

        if meta.scheduling_mode == "conda":
            return self._run_conda(meta, inputs)
        elif meta.scheduling_mode == "fastapi":
            return self._run_fastapi(meta, inputs)
        else:
            return self._run_inline(meta, inputs)

    # ------------------------------------------------------------------
    # Inline execution (in-process)
    # ------------------------------------------------------------------

    def _run_inline(self, meta: ToolMetadata, inputs: Dict[str, Any]) -> Dict[str, Any]:
        tool = self._get_or_load_tool(meta)
        return tool(inputs)

    def _get_or_load_tool(self, meta: ToolMetadata) -> BaseTool:
        tool_id = meta.tool_id
        # Create per-tool lock lazily
        with self._global_lock:
            if tool_id not in self._locks:
                self._locks[tool_id] = threading.Lock()

        with self._locks[tool_id]:
            if tool_id not in self._instances:
                logger.info(f"Loading tool {tool_id} into memory.")
                self._instances[tool_id] = self._instantiate_tool(meta)
        return self._instances[tool_id]

    def _instantiate_tool(self, meta: ToolMetadata) -> BaseTool:
        class_path = _TOOL_CLASS_MAP.get(meta.tool_id)
        if class_path is None:
            raise NotImplementedError(
                f"No implementation class registered for tool '{meta.tool_id}'."
            )
        module_path, class_name = class_path.rsplit(":", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(meta)

    # ------------------------------------------------------------------
    # FastAPI execution (HTTP to pre-loaded service)
    # ------------------------------------------------------------------

    def _run_fastapi(self, meta: ToolMetadata, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import httpx
        from config.settings import get_settings
        cfg = get_settings().scheduler
        base_url = cfg.fastapi_base_url
        port = meta.fastapi_port

        # Wait for service readiness
        self._wait_for_service(base_url, port, cfg.service_startup_timeout)

        url = f"{base_url}:{port}/run"
        logger.debug(f"FastAPI call -> {url}")
        response = httpx.post(url, json=inputs, timeout=120)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _wait_for_service(base_url: str, port: int, timeout: int = 30) -> None:
        import time
        import httpx
        deadline = time.monotonic() + timeout
        health_url = f"{base_url}:{port}/health"
        while time.monotonic() < deadline:
            try:
                r = httpx.get(health_url, timeout=2)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
        raise TimeoutError(f"Service at {health_url} not ready after {timeout}s.")

    # ------------------------------------------------------------------
    # Conda subprocess execution (on-demand)
    # ------------------------------------------------------------------

    def _run_conda(self, meta: ToolMetadata, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import json
        import subprocess
        from config.settings import get_settings
        cfg = get_settings()
        conda_base = cfg.scheduler.conda_base
        conda_env = meta.conda_env
        script = str(cfg.project_root / meta.entry_script)

        input_json = json.dumps(inputs)
        cmd = [
            "bash", "-c",
            f"source {conda_base}/etc/profile.d/conda.sh && "
            f"conda activate {conda_env} && "
            f"python {script} '{input_json}'"
        ]
        logger.debug(f"Conda call: env={conda_env}, script={script}")
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=cfg.scheduler.conda_timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Conda tool '{meta.tool_id}' failed:\n{proc.stderr}"
            )
        try:
            return json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            return {"raw_output": proc.stdout.strip()}

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def unload_tool(self, tool_id: str) -> None:
        """Release a cached inline tool instance (free GPU memory)."""
        with self._global_lock:
            if tool_id in self._instances:
                self._instances[tool_id].unload_model()
                del self._instances[tool_id]
                logger.info(f"Unloaded tool {tool_id}.")

    def list_loaded(self):
        return list(self._instances.keys())
