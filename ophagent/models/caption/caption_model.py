"""
OCT image caption model.

Wraps a vision-language LLM (via OpenRouter, OpenAI, or Anthropic) and
produces a clinical-style textual description of an OCT B-scan.

The caption is grounded by a clinical prompt that asks the model to
describe:
  - overall image quality / artifacts
  - retinal layer integrity
  - any visible lesions, fluid, drusen, PED, atrophy, etc.
  - anatomical landmarks (fovea, RPE, choroid)

This is a "soft" perception module that complements the trained
discriminative models (classifier / segmentor): it gives a
free-text description the agent can quote in its final report.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "You are an OCT image-reading assistant. Describe this retinal OCT B-scan in "
    "clinical language. In 4-6 short sentences, comment on: (1) overall image "
    "quality and any artifacts (shadow, motion, low signal), (2) retinal layer "
    "integrity and visibility of major bands (ILM, ELM, IS/OS, RPE), (3) any "
    "abnormal findings (fluid, drusen, PED, atrophy, hyper-reflective foci, "
    "epiretinal membrane, etc.), and (4) the appearance of the fovea and choroid "
    "if visible. Be specific about location (e.g. 'subfoveal', 'temporal to "
    "fovea'). Do not assign a final diagnosis."
)


def _image_to_data_url(image_path: str | Path) -> str:
    """Encode a local image file as a base64 data URL (PNG/JPEG)."""
    path = Path(image_path)
    suffix = path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix or 'png'}"
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


class OphCaptionModel:
    """Caption generator for OCT images using a vision-language LLM.

    Backends:
      - "openrouter": OpenAI-compatible client pointed at https://openrouter.ai/api/v1
      - "openai":     official OpenAI API
      - "anthropic":  Claude messages API

    The same backend / model can be reused by the agent itself, but typically
    you'd pick a cheaper-but-still-multimodal model for captions.
    """

    def __init__(
        self,
        backend: str = "openrouter",
        model: str | None = None,
        max_tokens: int = 320,
        prompt: str | None = None,
    ):
        self.backend = backend
        self.model = model or self._default_model(backend)
        self.max_tokens = max_tokens
        self.prompt = prompt or DEFAULT_PROMPT
        self._client = self._init_client()

    @staticmethod
    def _default_model(backend: str) -> str:
        if backend == "openrouter":
            return os.environ.get("OPENROUTER_CAPTION_MODEL", "openai/gpt-5.5")
        if backend == "openai":
            return "gpt-5.5"
        if backend == "anthropic":
            return "claude-sonnet-4-20250514"
        return "openai/gpt-5.5"

    def _init_client(self):
        if self.backend in ("openai", "openrouter"):
            try:
                import openai
            except ImportError:
                raise RuntimeError(
                    "openai package required. Install: pip install openai"
                )
            if self.backend == "openrouter":
                api_key = os.environ.get("OPENROUTER_API_KEY")
                if not api_key:
                    raise RuntimeError("OPENROUTER_API_KEY env var required")
                headers = {"X-Title": os.environ.get("OPENROUTER_TITLE", "ophagent")}
                ref = os.environ.get("OPENROUTER_HTTP_REFERER")
                if ref:
                    headers["HTTP-Referer"] = ref
                return openai.OpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    default_headers=headers,
                )
            return openai.OpenAI()  # uses OPENAI_API_KEY
        if self.backend == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise RuntimeError(
                    "anthropic package required. Install: pip install anthropic"
                )
            return anthropic.Anthropic()
        raise ValueError(f"Unknown caption backend: {self.backend}")

    def caption(self, image_path: str | Path, extra_context: str = "") -> str:
        """Generate a clinical caption for an OCT image."""
        prompt = self.prompt
        if extra_context:
            prompt = f"{prompt}\n\nExtra context from earlier analysis:\n{extra_context}"

        if self.backend in ("openai", "openrouter"):
            return self._caption_openai_compatible(image_path, prompt)
        if self.backend == "anthropic":
            return self._caption_anthropic(image_path, prompt)
        raise RuntimeError(f"unsupported backend {self.backend}")

    def _caption_openai_compatible(self, image_path: str | Path, prompt: str) -> str:
        data_url = _image_to_data_url(image_path)
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
        )
        return resp.choices[0].message.content.strip()

    def _caption_anthropic(self, image_path: str | Path, prompt: str) -> str:
        path = Path(image_path)
        suffix = path.suffix.lower().lstrip(".")
        mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix or 'png'}"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime, "data": data},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
