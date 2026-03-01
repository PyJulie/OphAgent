"""
Base classes for all OphAgent tools.

Every tool wrapper (both reused and newly developed) must inherit from BaseTool
and implement the run() method.  The ToolMetadata dataclass holds registry info.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------

@dataclass
class ToolMetadata:
    """Static description of a tool, loaded from tool_registry.yaml."""
    tool_id: str
    name: str
    description: str
    modality: str                         # "CFP" | "OCT" | "UWF" | "FFA" | "any"
    task: str                             # "classification" | "segmentation" | …
    scheduling_mode: str                  # "inline" | "fastapi" | "conda"
    input_type: str                       # "image" | "image+text" | "multi_image"
    output_type: str
    model_weight: Optional[str] = None
    conda_env: Optional[str] = None
    fastapi_port: Optional[int] = None
    entry_script: Optional[str] = None
    newly_developed: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml_dict(cls, tool_id: str, d: dict) -> "ToolMetadata":
        return cls(
            tool_id=tool_id,
            name=d.get("name", tool_id),
            description=d.get("description", ""),
            modality=d.get("modality", "any"),
            task=d.get("task", ""),
            scheduling_mode=d.get("scheduling_mode", "inline"),
            input_type=d.get("input_type", "image"),
            output_type=d.get("output_type", "dict"),
            model_weight=d.get("model_weight"),
            conda_env=d.get("conda_env"),
            fastapi_port=d.get("fastapi_port"),
            entry_script=d.get("entry_script"),
            newly_developed=bool(d.get("newly_developed", False)),
            extra={k: v for k, v in d.items() if k not in {
                "name", "description", "modality", "task", "scheduling_mode",
                "input_type", "output_type", "model_weight", "conda_env",
                "fastapi_port", "entry_script", "newly_developed",
            }},
        )


# ---------------------------------------------------------------------------
# Abstract base tool
# ---------------------------------------------------------------------------

class BaseTool(ABC):
    """
    Abstract base class for every OphAgent tool.

    Subclasses must implement:
      - run(inputs) -> dict

    Optionally override:
      - load_model()  – called once on first use
      - unload_model() – release GPU memory
      - health_check() – used by scheduler for FastAPI services
    """

    def __init__(self, metadata: ToolMetadata):
        self.metadata = metadata
        self._model_loaded = False

    @property
    def tool_id(self) -> str:
        return self.metadata.tool_id

    @property
    def name(self) -> str:
        return self.metadata.name

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load model weights into memory. Override in subclasses."""
        self._model_loaded = True

    def unload_model(self) -> None:
        """Release model from memory. Override in subclasses."""
        self._model_loaded = False

    def health_check(self) -> bool:
        """Return True if tool is ready to process requests."""
        return True

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool with *inputs* and return a result dict.

        The result dict should always include at least:
          - "tool_id": str
          - "success": bool
        Plus task-specific keys (e.g. "label", "confidence", "mask_path").
        """

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self._model_loaded:
            self.load_model()
        result = self.run(inputs)
        result.setdefault("tool_id", self.tool_id)
        result.setdefault("success", True)
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.tool_id!r}, mode={self.metadata.scheduling_mode!r})"


# ---------------------------------------------------------------------------
# FastAPI client mixin
# ---------------------------------------------------------------------------

class FastAPIToolMixin:
    """
    Mixin for tools that run as pre-loaded FastAPI microservices.
    Provides a _post() helper for sending requests.
    """

    def _post(
        self,
        port: int,
        endpoint: str,
        payload: dict,
        timeout: int = 60,
        base_url: Optional[str] = None,
    ) -> dict:
        import httpx
        from config.settings import get_settings
        if base_url is None:
            base_url = get_settings().scheduler.fastapi_base_url
        url = f"{base_url}:{port}{endpoint}"
        response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def health_check(self) -> bool:
        try:
            import httpx
            from config.settings import get_settings
            base_url = get_settings().scheduler.fastapi_base_url
            url = f"{base_url}:{self.metadata.fastapi_port}/health"
            r = httpx.get(url, timeout=5)
            return r.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Conda subprocess mixin
# ---------------------------------------------------------------------------

class CondaToolMixin:
    """
    Mixin for tools that run in isolated Conda environments.
    Provides a _run_conda_script() helper.
    """

    def _run_conda_script(
        self,
        script_path: str,
        inputs: Dict[str, Any],
        conda_env: str,
        timeout: int = 120,
    ) -> dict:
        import json
        import shlex
        import subprocess
        from config.settings import get_settings

        conda_base = str(get_settings().scheduler.conda_base)
        input_json = json.dumps(inputs)

        # Quote all path/env arguments to prevent shell injection.
        # input_json is passed via stdin so no shell quoting is needed for it.
        conda_init_q = shlex.quote(f"{conda_base}/etc/profile.d/conda.sh")
        conda_env_q = shlex.quote(str(conda_env))
        script_path_q = shlex.quote(str(script_path))

        cmd = [
            "bash", "-c",
            f"source {conda_init_q} && "
            f"conda activate {conda_env_q} && "
            f"python {script_path_q}"
        ]
        proc = subprocess.run(
            cmd,
            input=input_json,   # tool scripts read JSON from stdin
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Conda script failed (env={conda_env}): {proc.stderr}"
            )
        try:
            return json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            return {"raw_output": proc.stdout.strip()}
