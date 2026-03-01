"""
LLM Backbone for OphAgent.

Provides a unified interface over OpenAI (primary, default GPT-5),
Google Gemini, and local models (via Ollama).
All agents (Planner, Verifier, etc.) call this module.
"""
from __future__ import annotations

import json
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
        cfg = get_settings().llm
        provider = provider or cfg.provider
        model_id = model_id or cfg.model_id
        api_key = api_key or cfg.api_key

        if provider == "openai":
            self._llm: BaseLLM = OpenAILLM(model_id=model_id, api_key=api_key)
        elif provider == "gemini":
            self._llm = GeminiLLM(model_id=model_id, api_key=api_key)
        elif provider == "local":
            base_url = getattr(cfg, "local_model_url", "http://localhost:11434/v1")
            self._llm = OpenAILLM(model_id=model_id, api_key=api_key or "ollama",
                                   base_url=base_url)
        else:
            raise ValueError(
                f"Unknown LLM provider: {provider!r}. "
                "Valid options: 'openai', 'gemini', 'local'."
            )

        logger.info(f"LLMBackbone initialised: provider={provider}, model={model_id}")

    # Delegate
    def chat(self, messages, **kw) -> str:
        return self._llm.chat(messages, **kw)

    def chat_json(self, messages, **kw) -> dict:
        return self._llm.chat_json(messages, **kw)
