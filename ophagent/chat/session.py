"""
ChatSession: stateful, multi-turn OCT analysis assistant.

Design:
  - LLM is the orchestrator across turns (OpenAI-compatible tool-use API).
  - Session keeps a single `messages` list — all user/assistant/tool messages.
  - Session also tracks a `context` of registered images/volumes and cached analyses so
    the LLM can refer back to them by id ("the AMD case from earlier").
  - Sessions persist to JSON on disk; reload to continue a conversation.

Independent of the one-shot demo_agent — but shares the OphAgentToolKit / OphPredictor /
model registry so model weights are loaded only once.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .prompts import CHAT_SYSTEM_PROMPT
from ..agent.tools.oct_tools import OphAgentToolKit
from ..inference.model_registry import create_default_registry
from ..inference.predictor import OphPredictor
from ..models.caption.caption_model import OphCaptionModel
from ..utils.paths import output_path


@dataclass
class SessionContext:
    """Tracks state that survives between turns."""
    current_image: str | None = None
    current_volume: str | None = None
    analyses: dict[str, Any] = field(default_factory=dict)   # path -> last per-tool result
    volume_analyses: dict[str, Any] = field(default_factory=dict)  # path -> summary
    last_report: dict[str, str] | None = None  # {"html": ..., "pdf": ...}


@dataclass
class ChatSession:
    """A persistent OphAgent OCT session."""
    session_id: str
    backend: str = "openrouter"
    model: str = "openai/gpt-5.5-pro"
    caption_model: str = "openai/gpt-5.4"
    max_tokens: int = 8000
    messages: list[dict] = field(default_factory=list)
    context: SessionContext = field(default_factory=SessionContext)
    created_at: float = field(default_factory=time.time)
    workspace: str = str(output_path("chat_sessions"))

    # transient (not persisted)
    _client: Any = field(default=None, repr=False)
    _toolkit: OphAgentToolKit | None = field(default=None, repr=False)
    _registry: Any = field(default=None, repr=False)
    _predictor: Any = field(default=None, repr=False)
    _caption: Any = field(default=None, repr=False)

    # ── persistence ─────────────────────────────────────────────────────────

    @classmethod
    def new(cls, **kw) -> "ChatSession":
        return cls(session_id=uuid.uuid4().hex[:12], **kw)

    @classmethod
    def load(cls, path: str | Path) -> "ChatSession":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        ctx_data = data.pop("context", {})
        s = cls(**{k: v for k, v in data.items() if not k.startswith("_")})
        s.context = SessionContext(**ctx_data)
        return s

    def save(self, path: str | Path | None = None) -> Path:
        if path is None:
            Path(self.workspace).mkdir(parents=True, exist_ok=True)
            path = Path(self.workspace) / f"session_{self.session_id}.json"
        path = Path(path)
        data = {
            "session_id": self.session_id,
            "backend": self.backend,
            "model": self.model,
            "caption_model": self.caption_model,
            "max_tokens": self.max_tokens,
            "messages": self.messages,
            "context": asdict(self.context),
            "created_at": self.created_at,
            "workspace": self.workspace,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    # ── lazy initialisation of heavyweight resources ────────────────────────

    def _ensure_models(self):
        if self._registry is None:
            self._registry = create_default_registry()
            self._predictor = OphPredictor(self._registry)
        if self._caption is None:
            try:
                self._caption = OphCaptionModel(
                    backend=self.backend if self.backend != "local" else "openrouter",
                    model=self.caption_model,
                )
            except Exception:
                self._caption = None
        if self._toolkit is None:
            run_dir = Path(self.workspace) / self.session_id
            run_dir.mkdir(parents=True, exist_ok=True)
            self._toolkit = OphAgentToolKit(
                self._predictor,
                caption_model=self._caption,
                report_output_root=str(run_dir),
            )
            self._install_session_tools()

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.backend == "openrouter":
            import openai
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY env var required")
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={"X-Title": "oct-chat"},
            )
        elif self.backend == "aigcbest":
            import openai
            key = os.environ.get("AIGCBEST_API_KEY")
            if not key:
                raise RuntimeError("AIGCBEST_API_KEY env var required")
            self._client = openai.OpenAI(
                api_key=key, base_url="https://api2.aigcbest.top/v1",
            )
        elif self.backend == "dashscope":
            import openai
            key = os.environ.get("DASHSCOPE_API_KEY")
            if not key:
                raise RuntimeError("DASHSCOPE_API_KEY env var required")
            self._client = openai.OpenAI(
                api_key=key,
                base_url=os.environ.get(
                    "DASHSCOPE_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
            )
        elif self.backend == "openai":
            import openai
            self._client = openai.OpenAI()
        elif self.backend == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ── session-aware tool additions ────────────────────────────────────────

    def _install_session_tools(self):
        """Add tools that mutate self.context: register current image/volume, slice extraction, volume analysis."""
        from ..agent.tools.oct_tools import Tool, ToolParameter

        def _set_current_image(path: str) -> dict[str, Any]:
            p = Path(path)
            if not p.exists():
                return {"status": "error", "error": f"File not found: {path}"}
            self.context.current_image = str(p.resolve())
            return {"status": "ok", "current_image": str(p.resolve()),
                    "exists": True}

        def _set_current_volume(path: str) -> dict[str, Any]:
            p = Path(path)
            if not p.exists():
                return {"status": "error", "error": f"File not found: {path}"}
            self.context.current_volume = str(p.resolve())
            return {"status": "ok", "current_volume": str(p.resolve())}

        def _get_slice(index: int, volume_path: str | None = None,
                       save_as: str | None = None) -> dict[str, Any]:
            from ..data.volume_loader import load_volume
            import cv2
            vp = volume_path or self.context.current_volume
            if not vp:
                return {"status": "error", "error": "no current volume; call set_current_volume first"}
            vol = load_volume(vp)
            if index < 0 or index >= vol.n_slices:
                return {"status": "error",
                        "error": f"index out of range; volume has {vol.n_slices} slices"}
            img = vol.slice(index)
            out_dir = Path(self._toolkit.report_output_root) / "slices"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = Path(save_as) if save_as else (
                out_dir / f"{Path(vp).stem}_slice{index:03d}.png"
            )
            cv2.imwrite(str(out_path), img)
            # auto-register as current image for follow-up analyses
            self.context.current_image = str(out_path.resolve())
            return {"status": "ok", "saved_to": str(out_path),
                    "n_slices_total": vol.n_slices,
                    "registered_as_current_image": True}

        def _analyze_volume(volume_path: str | None = None, stride: int = 1,
                            classifier: str = "oct_classifier_octdl") -> dict[str, Any]:
            from ..agent.volume_processor import analyze_volume as _av
            vp = volume_path or self.context.current_volume
            if not vp:
                return {"status": "error", "error": "no current volume; call set_current_volume first"}
            result = _av(volume_path=vp, registry=self._registry, predictor=self._predictor,
                         classifier_model=classifier, slice_stride=stride, progress=False)
            summary = {
                "n_slices": result.volume.n_slices,
                "slices_with_fluid": result.slice_with_fluid_count,
                "foveal_slice_idx": result.foveal_slice_idx,
                "classification_consensus": result.classification_consensus,
                "total_fluid_voxels": result.total_fluid_voxels,
                "classifier_used": result.classifier_name,
            }
            self.context.volume_analyses[vp] = summary
            return {"status": "ok", **summary}

        self._toolkit.tools["set_current_image"] = Tool(
            name="set_current_image",
            description="Register an OCT B-scan image as the current focus. Subsequent tools "
                        "can omit image_path and use this one. Returns the resolved absolute path.",
            parameters=[ToolParameter("path", "string", "Filesystem path to the image")],
            function=_set_current_image,
        )
        self._toolkit.tools["set_current_volume"] = Tool(
            name="set_current_volume",
            description="Register an OCT volume (DICOM/NIfTI/NPY/folder) as the current cube. "
                        "Subsequent volume tools can omit volume_path.",
            parameters=[ToolParameter("path", "string", "Path to volume file/folder")],
            function=_set_current_volume,
        )
        self._toolkit.tools["get_slice"] = Tool(
            name="get_slice",
            description="Extract a single B-scan from the registered volume and save as PNG. "
                        "Auto-registers the extracted slice as the current image so you can "
                        "immediately call other tools on it.",
            parameters=[
                ToolParameter("index", "integer", "Slice index (0-based)"),
                ToolParameter("volume_path", "string", "Optional explicit volume path",
                              required=False, default=""),
            ],
            function=_get_slice,
        )
        self._toolkit.tools["analyze_volume"] = Tool(
            name="analyze_volume",
            description="Run the full per-slice pipeline (quality, classification, fluid+layer seg) "
                        "on every slice of the current volume and aggregate. Returns consensus, "
                        "fluid burden, foveal-like slice index.",
            parameters=[
                ToolParameter("volume_path", "string", "Optional explicit path",
                              required=False, default=""),
                ToolParameter("stride", "integer", "Process every N-th slice",
                              required=False, default=1),
                ToolParameter("classifier", "string", "Classifier model name",
                              required=False, default="oct_classifier_octdl",
                              enum=["oct_classifier_octdl", "oct_classifier_kermany",
                                    "oct_classifier_broad"]),
            ],
            function=_analyze_volume,
        )

        # Wrap analysis tools to default image_path to context.current_image
        for name in ("assess_quality", "classify_disease", "segment_fluid",
                     "segment_layers", "denoise_image", "super_resolve",
                     "caption_image"):
            if name not in self._toolkit.tools:
                continue
            self._wrap_with_default_image(name)

    def _wrap_with_default_image(self, tool_name: str):
        """Patch a tool so image_path defaults to self.context.current_image."""
        tool = self._toolkit.tools[tool_name]
        original_fn = tool.function

        def wrapped(**kwargs):
            if not kwargs.get("image_path") and self.context.current_image:
                kwargs["image_path"] = self.context.current_image
            return original_fn(**kwargs)

        tool.function = wrapped
        # mark image_path as optional now
        for p in tool.parameters:
            if p.name == "image_path":
                p.required = False
                p.default = ""

    # ── conversation API ────────────────────────────────────────────────────

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def chat(self, user_text: str | None = None, max_tool_steps: int = 12,
             on_event=None) -> str:
        """Send (optionally) a user message and run the tool-use loop until the
        assistant produces text. Returns the final assistant text.

        on_event: optional callable taking a dict event. Useful for streaming.
            Events emitted:
              {"type": "thinking"}                  — about to call the LLM
              {"type": "tool_call", "name": ...,    — model wants to run a tool
                                   "arguments": ...}
              {"type": "tool_result", "name": ...,  — tool finished
                                     "preview": ...}
              {"type": "text", "content": ...}      — final assistant text
              {"type": "error", "message": ...}     — something failed
        """
        self._ensure_models()
        self._ensure_client()
        if user_text is not None:
            self.add_user_message(user_text)

        def emit(ev: dict):
            if on_event:
                try:
                    on_event(ev)
                except Exception:
                    pass

        # construct the request message list with system prompt at the top
        sys_msg = {"role": "system", "content": CHAT_SYSTEM_PROMPT}
        ctx_note = self._context_note()
        if ctx_note:
            sys_msg["content"] = sys_msg["content"] + "\n\n# Session context\n" + ctx_note
        request_messages = [sys_msg] + self.messages

        tools = self._toolkit.get_all_schemas()

        for _step in range(max_tool_steps):
            emit({"type": "thinking"})
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=request_messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=self.max_tokens,
            )
            msg = resp.choices[0].message
            # Persist the assistant message (tool calls and/or text)
            asst_record: dict[str, Any] = {"role": "assistant",
                                           "content": msg.content or ""}
            if msg.tool_calls:
                asst_record["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            self.messages.append(asst_record)
            request_messages.append(asst_record)

            if not msg.tool_calls:
                emit({"type": "text", "content": msg.content or ""})
                return msg.content or ""

            # Execute each tool call
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                emit({"type": "tool_call", "name": tc.function.name,
                      "arguments": args})
                try:
                    result = self._toolkit.execute(tc.function.name, **args)
                except Exception as e:
                    result = {"error": str(e)}
                # cache analysis findings under the relevant image path
                if isinstance(result, dict) and self.context.current_image:
                    self.context.analyses.setdefault(self.context.current_image, {})[
                        tc.function.name
                    ] = self._serialize_for_history(result)
                tool_record = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps(self._serialize_for_history(result),
                                          default=str),
                }
                self.messages.append(tool_record)
                request_messages.append(tool_record)
                # tiny preview for the UI
                summary = self._brief_tool_summary(tc.function.name, result)
                emit({"type": "tool_result", "name": tc.function.name,
                      "preview": summary})

        emit({"type": "error", "message": "reached max tool steps"})
        return "(reached max tool steps without a final assistant message)"

    @staticmethod
    def _brief_tool_summary(name: str, result: dict) -> str:
        """Compose a one-line summary of a tool result for live display."""
        if not isinstance(result, dict):
            return str(result)[:120]
        if "error" in result:
            return f"error: {result['error']}"[:120]
        if name == "assess_quality":
            q = result.get("quality")
            c = result.get("confidence")
            return f"quality={q} ({c:.0%})" if isinstance(c, (int, float)) else f"quality={q}"
        if name == "classify_disease":
            return f"{result.get('predicted_class','?')} ({result.get('confidence',0):.0%})"
        if name == "segment_fluid":
            areas = result.get("class_areas", {})
            nonbg = {k: v for k, v in areas.items() if k.lower() != "background"}
            return ", ".join(f"{k}={v}px" for k, v in nonbg.items()) or "no fluid"
        if name == "segment_layers":
            return f"{result.get('num_classes_detected', 0)} layers"
        if name == "analyze_volume":
            cons = result.get("classification_consensus", {})
            top = next(iter(cons.items()), None)
            return (f"{result.get('n_slices','?')} slices, "
                    f"{result.get('slices_with_fluid','?')} fluid+"
                    + (f", top={top[0]}({top[1]})" if top else ""))
        if name == "get_slice":
            return f"saved slice → {Path(result.get('saved_to','')).name}"
        if name == "set_current_image":
            return f"→ {Path(result.get('current_image','')).name}"
        if name == "set_current_volume":
            return f"→ {Path(result.get('current_volume','')).name}"
        if name == "caption_image":
            cap = result.get("caption", "")
            return cap[:120] + ("…" if len(cap) > 120 else "")
        if name == "build_visual_report":
            return f"PDF: {Path(result.get('report_pdf','')).name}"
        # generic fallback
        return ", ".join(f"{k}={v}" for k, v in list(result.items())[:3])[:120]

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _serialize_for_history(obj: Any) -> Any:
        """Strip numpy arrays and other huge fields before saving."""
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in ("mask", "denoised_image", "super_resolved_image"):
                    out[k] = f"<omitted {type(v).__name__}>"
                else:
                    out[k] = ChatSession._serialize_for_history(v)
            return out
        if isinstance(obj, list):
            return [ChatSession._serialize_for_history(v) for v in obj]
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return f"<{type(obj).__name__}>"

    def _context_note(self) -> str:
        bits = []
        if self.context.current_image:
            bits.append(f"- Current image: `{self.context.current_image}`")
        if self.context.current_volume:
            bits.append(f"- Current volume: `{self.context.current_volume}`")
        if self.context.analyses:
            paths = ", ".join(Path(p).name for p in self.context.analyses)
            bits.append(f"- Images already analysed this session: {paths}")
        if self.context.volume_analyses:
            for path, summary in self.context.volume_analyses.items():
                bits.append(
                    f"- Volume `{Path(path).name}` analyzed: "
                    f"{summary.get('n_slices')} slices, "
                    f"{summary.get('slices_with_fluid')} with fluid, "
                    f"consensus={summary.get('classification_consensus')}"
                )
        if self.context.last_report:
            bits.append(
                f"- Last visual report HTML: `{self.context.last_report.get('html')}`"
            )
        return "\n".join(bits)
