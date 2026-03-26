"""
LLM Backbone for OphAgent.

Provides a unified interface over OpenAI (primary, default GPT-5),
Google Gemini, and local models (via Ollama).
All agents (Planner, Verifier, etc.) call this module.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from tenacity import retry, stop_after_attempt, wait_exponential

from ophagent.utils.logger import get_logger

logger = get_logger("llm.backbone")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseLLM(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> str:
        """Send a chat conversation and return the assistant text."""

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> dict:
        """Like *chat* but parse the response as JSON, retrying on failure."""
        from ophagent.utils.text_utils import extract_json_block
        raw = self.chat(messages, system=system, max_tokens=max_tokens,
                        temperature=temperature, **kwargs)
        try:
            return json.loads(extract_json_block(raw))
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed ({e}); returning raw text in dict.")
            return {"raw": raw}


# ---------------------------------------------------------------------------
# OpenAI-compatible (GPT-5, GPT-4o, and local Ollama)
# ---------------------------------------------------------------------------

class OpenAILLM(BaseLLM):
    """
    Supports any OpenAI-compatible endpoint:
      - OpenAI API  (GPT-5, GPT-4o, …)
      - Local Ollama (Qwen, LLaMA, …) via base_url override
    """

    def __init__(
        self,
        model_id: str = "gpt-5",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        from openai import OpenAI
        self.model_id = model_id
        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> str:
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

class GeminiLLM(BaseLLM):
    """
    Google Gemini via the `google-generativeai` SDK.
    Supports Gemini-3.0-Pro, Gemini-1.5-Pro, Gemini-2.0-Flash, etc.
    """

    def __init__(self, model_id: str = "gemini-3.0-pro", api_key: Optional[str] = None):
        import google.generativeai as genai
        self.model_id = model_id
        if api_key:
            genai.configure(api_key=api_key)
        self._genai = genai

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> str:
        generation_config = self._genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        model = self._genai.GenerativeModel(
            model_name=self.model_id,
            system_instruction=system or "",
            generation_config=generation_config,
        )
        # Convert OpenAI-style messages to Gemini format
        history = []
        last_user_content = ""
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            if role == "user":
                last_user_content = msg["content"]
                history.append({"role": "user", "parts": [msg["content"]]})
            else:
                history.append({"role": "model", "parts": [msg["content"]]})

        # Use the last user message as the prompt; prior turns as history
        if len(history) > 1:
            chat_session = model.start_chat(history=history[:-1])
            response = chat_session.send_message(last_user_content)
        else:
            response = model.generate_content(last_user_content)

        return response.text


class HeuristicLLM(BaseLLM):
    """Offline fallback LLM used when remote providers are unavailable."""

    def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> str:
        user_content = "\n".join(msg["content"] for msg in messages if msg.get("role") == "user")
        if system and "Planner component of OphAgent" in system:
            return json.dumps(self._plan(user_content), indent=2)
        if system and "Verifier component of OphAgent" in system:
            return json.dumps(self._verify(user_content), indent=2)
        if system and "Memory Manager of OphAgent" in system:
            return json.dumps(self._memory_entry(user_content), indent=2)
        if system and "senior ophthalmologist" in system:
            return self._report(user_content)
        return "HeuristicLLM fallback response."

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> dict:
        user_content = "\n".join(msg["content"] for msg in messages if msg.get("role") == "user")
        if system and "Planner component of OphAgent" in system:
            return self._plan(user_content)
        if system and "Verifier component of OphAgent" in system:
            return self._verify(user_content)
        if system and "Memory Manager of OphAgent" in system:
            return self._memory_entry(user_content)
        raw = self.chat(messages, system=system, max_tokens=max_tokens, temperature=temperature, **kwargs)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}

    @staticmethod
    def _extract_query(user_content: str) -> str:
        match = re.search(r"Clinical query:\s*(.+?)(?:\n\n|\Z)", user_content, flags=re.S)
        if match:
            return match.group(1).strip()
        match = re.search(r"Patient query:\s*(.+?)(?:\n\n|\Z)", user_content, flags=re.S)
        if match:
            return match.group(1).strip()
        match = re.search(r"Query:\s*(.+?)(?:\n\n|\Z)", user_content, flags=re.S)
        if match:
            return match.group(1).strip()
        return user_content.strip()

    @staticmethod
    def _extract_images(user_content: str) -> List[str]:
        match = re.search(r"Available images:\n(.+?)(?:\n\n|\Z)", user_content, flags=re.S)
        if not match:
            return []
        lines = [line.strip() for line in match.group(1).splitlines() if line.strip()]
        return [line for line in lines if line.lower() != "no images provided."]

    def _plan(self, user_content: str) -> dict:
        query = self._extract_query(user_content)
        images = self._extract_images(user_content)
        query_l = query.lower()
        steps: List[Dict[str, Any]] = []
        image_path = images[0] if images else ""

        def add_step(tool_name: str, inputs: Dict[str, Any], description: str, depends_on: Optional[List[int]] = None):
            steps.append(
                {
                    "step_id": len(steps) + 1,
                    "tool_name": tool_name,
                    "inputs": inputs,
                    "description": description,
                    "depends_on": depends_on or [],
                }
            )

        if not images:
            add_step("web_search", {"query": query}, "Retrieve background ophthalmic guidance")
        elif "ffa" in query_l and len(images) >= 2:
            add_step("cfp_ffa_multimodal", {"cfp_path": images[0], "ffa_path": images[1]}, "Analyse paired CFP and FFA images")
        elif "disc" in query_l and "fovea" in query_l:
            add_step("disc_fovea", {"image_path": image_path}, "Localise optic disc and fovea")
        elif "segment" in query_l or "mask" in query_l:
            add_step("retsam", {"image_path": image_path}, "Generate retinal segmentation mask")
        else:
            if "uwf" in query_l:
                add_step("uwf_quality_disease", {"image_path": image_path}, "Assess UWF image quality and disease pattern")
            else:
                add_step("cfp_quality", {"image_path": image_path}, "Assess CFP image quality")
                quality_step = steps[-1]["step_id"]
                if "glaucoma" in query_l:
                    add_step("cfp_glaucoma", {"image_path": image_path}, "Evaluate glaucoma-related findings", [quality_step])
                elif "pdr" in query_l or "proliferative" in query_l:
                    add_step("cfp_pdr", {"image_path": image_path}, "Grade proliferative diabetic retinopathy activity", [quality_step])
                elif "question" in query_l or "what" in query_l or "describe" in query_l:
                    add_step("vision_unite", {"image_path": image_path, "question": query}, "Answer image question", [quality_step])
                else:
                    add_step("cfp_disease", {"image_path": image_path}, "Classify retinal disease patterns", [quality_step])

        dependency_ids = [step["step_id"] for step in steps]
        add_step(
            "synthesise",
            {"summary": "Summarise findings into a clinical report"},
            "Generate final report",
            dependency_ids,
        )
        return {"steps": steps}

    def _verify(self, user_content: str) -> dict:
        tool_outputs_match = re.search(r"Tool outputs:\n(.+?)(?:\n\nRelevant knowledge base context:|\Z)", user_content, flags=re.S)
        verified_result: Dict[str, Any] = {}
        confidence = 0.72
        if tool_outputs_match:
            for line in tool_outputs_match.group(1).splitlines():
                line = line.strip()
                if not line or ": " not in line:
                    continue
                prefix, payload = line.split(": ", 1)
                if payload.startswith("ERROR"):
                    confidence = min(confidence, 0.45)
                    continue
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    verified_result[prefix] = parsed
                    if parsed.get("quality_label", "").lower() == "bad":
                        confidence = min(confidence, 0.5)
        return {
            "valid": True,
            "confidence": confidence,
            "conflicts": [],
            "resolution": "Fallback verifier accepted the available tool outputs.",
            "verified_result": verified_result,
            "needs_human_review": confidence < 0.6,
        }

    def _report(self, user_content: str) -> str:
        query = self._extract_query(user_content)
        findings_match = re.search(r"Verified findings:\n(.+?)(?:\n\nKnowledge base context:|\Z)", user_content, flags=re.S)
        findings = findings_match.group(1).strip() if findings_match else "{}"
        return (
            "1. Image Quality Assessment\n"
            "Structured fallback synthesis based on the available tool outputs.\n\n"
            "2. Detected Findings\n"
            f"{findings}\n\n"
            "3. Differential Diagnosis\n"
            "Findings should be interpreted together with clinical context and formal review.\n\n"
            "4. Recommended Next Steps\n"
            f"Correlate the automated summary with the original query: {query}\n\n"
            "5. References / Supporting Evidence\n"
            "No live literature retrieval was required for the fallback report."
        )

    def _memory_entry(self, user_content: str) -> dict:
        query = self._extract_query(user_content)
        tools_match = re.search(r"Tools used:\s*(.+?)(?:\n\n|\Z)", user_content, flags=re.S)
        tools = [tool.strip() for tool in tools_match.group(1).split(",")] if tools_match else []
        return {
            "summary": f"Fallback memory entry for query: {query}",
            "key_findings": ["Automated heuristic execution completed"],
            "tools_used": [tool for tool in tools if tool],
            "modalities": ["CFP"] if any("cfp" in tool for tool in tools) else [],
            "tags": ["fallback"],
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class LLMBackbone:
    """
    Unified LLM entry-point.  Instantiate once and pass around.

    Supported providers
    -------------------
    - ``openai``  : OpenAI GPT-5 / GPT-4o (default)
    - ``gemini``  : Google Gemini-3.0-Pro / Gemini-1.5-Pro
    - ``local``   : Any Ollama-served model (Qwen, LLaMA, …)

    Usage::

        llm = LLMBackbone()           # uses Settings defaults (GPT-5)
        answer = llm.chat([{"role": "user", "content": "Hello"}])
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        from config.settings import get_settings
        settings = get_settings()
        cfg = settings.llm
        provider = provider or cfg.provider
        model_id = model_id or cfg.model_id
        api_key = api_key or cfg.api_key
        self._fallback_llm: BaseLLM = HeuristicLLM()
        self._allow_fallbacks = settings.allow_fallbacks

        try:
            if provider == "openai":
                self._llm = OpenAILLM(model_id=model_id, api_key=api_key)
            elif provider == "gemini":
                self._llm = GeminiLLM(model_id=model_id, api_key=api_key)
            elif provider == "local":
                base_url = getattr(cfg, "local_model_url", "http://localhost:11434/v1")
                self._llm = OpenAILLM(
                    model_id=model_id,
                    api_key=api_key or "ollama",
                    base_url=base_url,
                )
            else:
                raise ValueError(
                    f"Unknown LLM provider: {provider!r}. "
                    "Valid options: 'openai', 'gemini', 'local'."
                )
            logger.info(f"LLMBackbone initialised: provider={provider}, model={model_id}")
        except Exception as exc:
            if not self._allow_fallbacks:
                raise RuntimeError(
                    f"LLM provider initialisation failed in strict mode: {exc}"
                ) from exc
            logger.warning(f"LLM provider initialisation failed; using fallback LLM: {exc}")
            self._llm = self._fallback_llm

    # Delegate
    def chat(self, messages, **kw) -> str:
        try:
            return self._llm.chat(messages, **kw)
        except Exception as exc:
            if not self._allow_fallbacks:
                raise RuntimeError(
                    f"Primary LLM chat failed in strict mode: {exc}"
                ) from exc
            logger.warning(f"Primary LLM chat failed; using fallback LLM: {exc}")
            return self._fallback_llm.chat(messages, **kw)

    def chat_json(self, messages, **kw) -> dict:
        try:
            return self._llm.chat_json(messages, **kw)
        except Exception as exc:
            if not self._allow_fallbacks:
                raise RuntimeError(
                    f"Primary LLM JSON call failed in strict mode: {exc}"
                ) from exc
            logger.warning(f"Primary LLM JSON call failed; using fallback LLM: {exc}")
            return self._fallback_llm.chat_json(messages, **kw)
