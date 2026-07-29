"""
Bilingual FFA report generator (paired CFP + FFA → structured clinical
caption in EN / ZH / JA / KO / ES / DE / FR).

Strategy: NO additional training. We use the trained joint classifier
(`cross_cfp_ffa_paired`, 98% val acc) to fix the diagnosis, then prompt the
chat LLM with class-conditioned few-shot exemplars from the 120 GPT-5
captions in `ophagent_eval.csv`. Output schema mirrors GPT-5's columns so
this tool is a drop-in replacement.

Tool: `paired_bilingual_report`
"""

from __future__ import annotations

import csv
import json
import logging
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from ..base import AdapterBase, ToolMetadata, AdapterResult, register
from ...utils.paths import external_file


log = logging.getLogger(__name__)

EVAL_CSV = external_file(
    "OPHAGENT_PAIRED_REPORT_EXEMPLARS", "demos", "paired_report_exemplars.csv"
)
LANGS = ["en", "zh", "ja", "ko", "es", "de", "fr"]
CAPTION_COL = {
    "en": "ophagent_caption", "zh": "oph_caption_zh", "ja": "oph_caption_ja",
    "ko": "oph_caption_ko", "es": "oph_caption_es", "de": "oph_caption_de",
    "fr": "oph_caption_fr",
}


@lru_cache(maxsize=1)
def _load_exemplars() -> dict:
    """Return {class_name: {lang: example_text}} — first row per class."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    if not EVAL_CSV.exists():
        return out
    with open(EVAL_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cls = row.get("gt_level1", "").split("-")[-1] or row.get("gt_level1")
            if not cls or cls in out:
                continue
            for lang in LANGS:
                col = CAPTION_COL[lang]
                if row.get(col):
                    out[cls][lang] = row[col]
    return dict(out)


def _build_prompt(diagnosis: str, prob: float,
                  ranked_diffs: list[dict], languages: list[str],
                  exemplars: dict) -> tuple[str, str]:
    """Return (system, user) prompts for the chat LLM."""
    system = (
        "You are a senior retinal specialist writing concise, structured "
        "clinical narratives for paired CFP + FFA findings.\n\n"
        "Strict output rules:\n"
        "- Return a single JSON object with one field per requested language "
        f"code: {languages}.\n"
        "- Each value is a 4-6 sentence clinical narrative, ~60-80 words, "
        "matching the **style** of the exemplar below (factual, structured: "
        "vessel filling → optic disc → macula → lesions → integrative line).\n"
        "- The narrative must be CONSISTENT with the diagnosis "
        f"`{diagnosis}` (probability {prob:.0%}).\n"
        "- Do NOT mention model names, probabilities, or AI reasoning.\n"
        "- Do NOT include the JSON keys' English label inside non-English "
        "narratives — translate everything.\n"
    )

    # Build exemplar block (use English; the model will translate)
    ex_en = exemplars.get(diagnosis, {}).get("en", "")
    if ex_en:
        system += f"\nExemplar EN narrative for {diagnosis}:\n{ex_en}\n"

    user_payload = {
        "primary_diagnosis": diagnosis,
        "primary_probability": round(prob, 3),
        "top3_differentials": ranked_diffs[:3],
        "languages_requested": languages,
    }
    user = (
        "Generate the bilingual narrative for the following case:\n"
        + json.dumps(user_payload, ensure_ascii=False, indent=2)
        + '\n\nReturn ONLY a JSON object like {"en": "...", "zh": "...", ...}.'
    )
    return system, user


def _call_llm(session, system: str, user: str) -> str:
    """Use the session's chat client to generate the JSON response."""
    if session is None:
        raise RuntimeError("paired_bilingual_report requires an active session")
    session._ensure_client()
    client = session._client
    resp = client.chat.completions.create(
        model=session.model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=1800,
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or "{}"


@register
class BilingualReportAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="paired_bilingual_report",
        modality="multi",
        task="captioning",
        description=(
            "Generate a structured multilingual FFA clinical report (EN / ZH / "
            "JA / KO / ES / DE / FR) for a paired CFP + FFA case. Internally: "
            "(1) runs the joint 5-class classifier (`cross_cfp_ffa_paired`) "
            "to fix the diagnosis; (2) prompts the chat LLM with a class-"
            "conditioned few-shot exemplar drawn from a 120-case GPT-5 "
            "reference corpus, in the requested languages. The output matches "
            "the GPT-5 caption schema used in the Paired_600 benchmark. Use "
            "this when the user asks for a 'report', '报告', '所見', or "
            "explicitly requests multilingual output."
        ),
        input_size=None,
        labels=[],
        confidence_threshold=0.5,
        limitations=[
            "Restricted to the 5 Paired_600 classes "
            "(Normal/DR/RVO/AMD/CSC) — fed to the LLM as a fact.",
            "Quality of the language varies with the underlying LLM; "
            "request `model=gpt-5.5-pro` or similar for clinical-grade prose.",
        ],
        cost_class="slow",
        source_dir="(composite + LLM)",
    )

    def __init__(self, device: str = "cuda", session=None):
        super().__init__(device)
        self.session = session

    def _load_impl(self) -> None:
        self._impl = "composite"

    def _predict_impl(self, image_path: str, ffa_path: str = "",
                      languages: str = "en,zh", session=None,
                      **_) -> AdapterResult:
        from ..base import GLOBAL_REGISTRY
        sess = session or self.session
        if not ffa_path:
            return AdapterResult(
                success=False, tool=self.metadata.name, modality="multi",
                task="captioning",
                error="paired_bilingual_report requires both image_path (CFP) and ffa_path",
            )
        # Step 1: diagnosis from joint classifier (fallback to softvote / ffa_paired5)
        for tool_name in ("cross_cfp_ffa_paired",
                          "cross_cfp_ffa_softvote", "ffa_paired5"):
            try:
                r = (GLOBAL_REGISTRY.predict(tool_name, image_path, ffa_path=ffa_path)
                     if tool_name.startswith("cross_")
                     else GLOBAL_REGISTRY.predict(tool_name, ffa_path))
                if r.success:
                    primary = r.predictions.get("primary_diagnosis", "Normal")
                    prob = float(r.predictions.get("primary_probability", 0))
                    ranked = r.predictions.get("ranked") or r.predictions.get("top_3") or [
                        {"label": primary, "probability": prob}]
                    diagnosis_tool = tool_name
                    break
            except Exception as e:
                log.warning(f"paired_bilingual_report: {tool_name} failed: {e}")
                r = None
        else:
            return AdapterResult(
                success=False, tool=self.metadata.name, modality="multi",
                task="captioning", error="all classifiers failed",
            )

        # Step 2: build LLM prompt
        langs = [l.strip().lower() for l in languages.split(",")
                 if l.strip().lower() in LANGS] or ["en"]
        exemplars = _load_exemplars()
        system, user = _build_prompt(primary, prob, ranked, langs, exemplars)

        # Step 3: call LLM
        try:
            raw = _call_llm(sess, system, user)
            parsed = json.loads(raw)
        except Exception as e:
            return AdapterResult(
                success=False, tool=self.metadata.name, modality="multi",
                task="captioning",
                error=f"LLM call/parse failed: {type(e).__name__}: {e}",
                predictions={"primary_diagnosis": primary,
                             "primary_probability": prob},
            )

        captions = {lang: parsed.get(lang, "") for lang in langs}

        return AdapterResult(
            success=True, tool=self.metadata.name, modality="multi",
            task="captioning",
            predictions={
                "primary_diagnosis": primary,
                "primary_probability": prob,
                "top3_differentials": ranked[:3],
                "diagnosis_source": diagnosis_tool,
                "languages": langs,
                "captions": captions,
            },
            confidence=prob,
            metadata={"diagnosis_tool": diagnosis_tool, "languages": langs},
        )
