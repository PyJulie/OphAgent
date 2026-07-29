"""
OphAgent OCT engine: LLM-powered orchestration of OCT analysis models.

Supports Claude (Anthropic) and OpenAI as backend LLMs.
Implements multi-step reasoning with tool use for end-to-end OCT analysis.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .prompts.system_prompt import SYSTEM_PROMPT, ANALYSIS_PROMPT_TEMPLATE
from .tools.oct_tools import OphAgentToolKit


@dataclass
class AgentMessage:
    role: str  # "user", "assistant", "tool"
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass
class AnalysisResult:
    query: str
    steps: list[dict]
    final_report: str
    findings: dict[str, Any]
    image_paths: list[str]


class OphAgent:
    """LLM-powered agent that orchestrates OCT analysis pipelines.

    The agent:
    1. Receives a clinical question + OCT image(s)
    2. Plans an analysis strategy
    3. Calls tools (models) in sequence
    4. Synthesizes results into a clinical report
    """

    def __init__(
        self,
        toolkit: OphAgentToolKit,
        backend: str = "anthropic",
        model: str | None = None,
        max_steps: int = 10,
    ):
        self.toolkit = toolkit
        self.backend = backend
        self.max_steps = max_steps
        self.conversation: list[AgentMessage] = []
        self.steps: list[dict] = []

        if backend == "anthropic":
            self.model = model or "claude-sonnet-4-20250514"
            self._client = self._init_anthropic()
        elif backend == "openai":
            self.model = model or "gpt-4o"
            self._client = self._init_openai()
        elif backend == "openrouter":
            self.model = model or os.environ.get(
                "OPENROUTER_MODEL", "anthropic/claude-sonnet-4"
            )
            self._client = self._init_openrouter()
        else:
            self._client = None
            self.model = "local"

    def _init_anthropic(self):
        try:
            import anthropic
            return anthropic.Anthropic()
        except ImportError:
            print("[WARN] anthropic package not installed. Install with: pip install anthropic")
            return None

    def _init_openai(self):
        try:
            import openai
            return openai.OpenAI()
        except ImportError:
            print("[WARN] openai package not installed. Install with: pip install openai")
            return None

    def _init_openrouter(self):
        """OpenRouter uses an OpenAI-compatible endpoint."""
        try:
            import openai
        except ImportError:
            print("[WARN] openai package not installed. Install with: pip install openai")
            return None

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("[WARN] OPENROUTER_API_KEY env var not set.")
            return None

        default_headers = {}
        if os.environ.get("OPENROUTER_HTTP_REFERER"):
            default_headers["HTTP-Referer"] = os.environ["OPENROUTER_HTTP_REFERER"]
        if os.environ.get("OPENROUTER_TITLE"):
            default_headers["X-Title"] = os.environ["OPENROUTER_TITLE"]
        else:
            default_headers["X-Title"] = "ophagent"

        return openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=default_headers,
        )

    def analyze(
        self,
        query: str,
        image_paths: list[str | Path],
        patient_context: str = "",
    ) -> AnalysisResult:
        """Run a complete OCT analysis pipeline.

        Args:
            query: Clinical question (e.g., "Check for AMD signs")
            image_paths: Paths to OCT images
            patient_context: Optional patient info (age, history, etc.)

        Returns:
            AnalysisResult with steps taken and final report.
        """
        image_paths = [str(p) for p in image_paths]
        self.steps = []
        self.conversation = []

        user_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            user_request=query,
            image_paths="\n".join(f"- {p}" for p in image_paths),
            patient_context=f"Patient context: {patient_context}" if patient_context else "",
        )

        if self._client and self.backend == "anthropic":
            return self._run_anthropic(query, user_prompt, image_paths)
        elif self._client and self.backend in ("openai", "openrouter"):
            return self._run_openai(query, user_prompt, image_paths)
        else:
            return self._run_local(query, user_prompt, image_paths)

    def _run_anthropic(
        self, query: str, user_prompt: str, image_paths: list[str]
    ) -> AnalysisResult:
        import anthropic

        tools = self._convert_tools_to_anthropic()
        messages = [{"role": "user", "content": user_prompt}]
        all_findings = {}

        for step in range(self.max_steps):
            response = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_uses = [b for b in assistant_content if b.type == "tool_use"]
            if not tool_uses:
                final_text = "".join(
                    b.text for b in assistant_content if hasattr(b, "text")
                )
                return AnalysisResult(
                    query=query,
                    steps=self.steps,
                    final_report=final_text,
                    findings=all_findings,
                    image_paths=image_paths,
                )

            tool_results = []
            for tool_use in tool_uses:
                result = self._execute_tool(tool_use.name, tool_use.input)
                all_findings[tool_use.name] = result
                self.steps.append({
                    "step": step + 1,
                    "tool": tool_use.name,
                    "input": tool_use.input,
                    "output": self._serialize_result(result),
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(self._serialize_result(result)),
                })

            messages.append({"role": "user", "content": tool_results})

        return AnalysisResult(
            query=query,
            steps=self.steps,
            final_report="Analysis reached maximum steps.",
            findings=all_findings,
            image_paths=image_paths,
        )

    def _run_openai(
        self, query: str, user_prompt: str, image_paths: list[str]
    ) -> AnalysisResult:
        tools = self.toolkit.get_all_schemas()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        all_findings = {}

        for step in range(self.max_steps):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=2048,
            )

            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                return AnalysisResult(
                    query=query,
                    steps=self.steps,
                    final_report=msg.content or "",
                    findings=all_findings,
                    image_paths=image_paths,
                )

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = self._execute_tool(tc.function.name, args)
                all_findings[tc.function.name] = result
                self.steps.append({
                    "step": step + 1,
                    "tool": tc.function.name,
                    "input": args,
                    "output": self._serialize_result(result),
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(self._serialize_result(result)),
                })

        return AnalysisResult(
            query=query,
            steps=self.steps,
            final_report="Analysis reached maximum steps.",
            findings=all_findings,
            image_paths=image_paths,
        )

    def _run_local(
        self, query: str, user_prompt: str, image_paths: list[str]
    ) -> AnalysisResult:
        """Rule-based fallback when no LLM API is available.

        Executes a fixed pipeline: quality → classify → segment fluid → segment layers.
        """
        all_findings = {}

        pipeline = [
            ("classify_disease", {"image_path": image_paths[0], "model_variant": "broad"}),
            ("classify_disease", {"image_path": image_paths[0], "model_variant": "octdl"}),
            ("denoise_image", {"image_path": image_paths[0]}),
        ]

        for step_idx, (tool_name, args) in enumerate(pipeline):
            print(f"[Step {step_idx + 1}] Running {tool_name}...")
            try:
                result = self._execute_tool(tool_name, args)
                finding_key = tool_name
                if finding_key in all_findings:
                    variant = args.get("model_variant", str(step_idx))
                    finding_key = f"{tool_name}_{variant}"
                all_findings[finding_key] = result
                self.steps.append({
                    "step": step_idx + 1,
                    "tool": tool_name,
                    "input": args,
                    "output": self._serialize_result(result),
                })
            except Exception as e:
                self.steps.append({
                    "step": step_idx + 1,
                    "tool": tool_name,
                    "input": args,
                    "error": str(e),
                })

        report_lines = ["OCT Analysis Report", "=" * 40]
        for tool_name, findings in all_findings.items():
            report_lines.append(f"\n## {tool_name}")
            for k, v in findings.items():
                if k not in ("mask", "denoised_image", "super_resolved_image"):
                    report_lines.append(f"  {k}: {v}")

        report = "\n".join(report_lines)

        return AnalysisResult(
            query=query,
            steps=self.steps,
            final_report=report,
            findings=all_findings,
            image_paths=image_paths,
        )

    def _execute_tool(self, tool_name: str, args: dict) -> dict[str, Any]:
        print(f"  → Executing tool: {tool_name}({args})")
        return self.toolkit.execute(tool_name, **args)

    def _convert_tools_to_anthropic(self) -> list[dict]:
        anthropic_tools = []
        for tool in self.toolkit.tools.values():
            schema = tool.to_schema()
            fn = schema["function"]
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn["description"],
                "input_schema": fn["parameters"],
            })
        return anthropic_tools

    @staticmethod
    def _serialize_result(result: dict) -> dict:
        serializable = {}
        for k, v in result.items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                serializable[k] = v
            else:
                serializable[k] = f"<{type(v).__name__}: shape={getattr(v, 'shape', 'N/A')}>"
        return serializable
