"""Dynamic candidate-set CLIP for CFP.

This tool is deliberately separate from the fixed `cfp_clip_ensemble`.
The planner/LLM supplies task-specific candidate text, and the tool scores
those candidates with the English CFP CLIPs (RetiZero and FLAIR by default).

Use case:
  - DR severity grading: map ICDR grades to natural-language descriptions.
  - Focused differential: compare a small, user/task-defined candidate set.

The LLM only maps the clinical taxonomy to text. Image scoring remains CLIP.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..base import AdapterBase, ToolMetadata, AdapterResult, GLOBAL_REGISTRY, register
from ._flair_arch import classify


_DEFAULT_MODELS = ("retizero", "flair")
_MODEL_TO_TOOL = {
    "retizero": "cfp_retizero",
    "flair": "cfp_flair",
}


def _clean_text(text: Any) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:240]


def _split_texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        chunks = re.split(r"\s*(?:\||;|\n)\s*", value)
        return [_clean_text(c) for c in chunks if _clean_text(c)]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                out.extend(_split_texts(item))
            else:
                s = _clean_text(item)
                if s:
                    out.append(s)
        return out
    s = _clean_text(value)
    return [s] if s else []


def _parse_candidates(candidates_json: Any = "", candidate_texts: str = "") -> tuple[list[dict[str, Any]], list[str]]:
    """Parse a flexible candidate format into [{label, texts}]."""
    warnings: list[str] = []
    raw: Any = None

    if isinstance(candidates_json, (list, dict)):
        raw = candidates_json
    elif isinstance(candidates_json, str) and candidates_json.strip():
        try:
            raw = json.loads(candidates_json)
        except json.JSONDecodeError as exc:
            warnings.append(f"candidates_json could not be parsed as JSON: {exc}")

    candidates: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        raw = [
            {"label": label, "texts": texts}
            for label, texts in raw.items()
        ]
    if isinstance(raw, list):
        for idx, item in enumerate(raw, 1):
            if isinstance(item, str):
                label = _clean_text(item)
                texts = [label] if label else []
            elif isinstance(item, dict):
                label = _clean_text(
                    item.get("label") or item.get("name") or item.get("id") or f"candidate_{idx}"
                )
                texts = _split_texts(
                    item.get("texts")
                    or item.get("prompts")
                    or item.get("synonyms")
                    or item.get("descriptions")
                    or item.get("text")
                    or item.get("description")
                    or label
                )
            else:
                label = _clean_text(item)
                texts = [label] if label else []
            if label and texts:
                candidates.append({"label": label, "texts": texts[:8]})

    if not candidates and candidate_texts.strip():
        for idx, line in enumerate(candidate_texts.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                label, rest = line.split(":", 1)
                label = _clean_text(label)
                texts = _split_texts(rest)
            else:
                label = _clean_text(line)
                texts = [label] if label else []
            if label and texts:
                candidates.append({"label": label, "texts": texts[:8]})

    # De-duplicate labels and exact text duplicates. Exact duplicate texts in
    # different labels make the softmax ambiguous, so keep the first and warn.
    label_seen: set[str] = set()
    text_seen: dict[str, str] = {}
    cleaned: list[dict[str, Any]] = []
    for cand in candidates[:32]:
        label = _clean_text(cand.get("label"))
        if not label:
            continue
        label_key = label.lower()
        if label_key in label_seen:
            warnings.append(f"duplicate candidate label ignored: {label}")
            continue
        label_seen.add(label_key)
        texts: list[str] = []
        for text in _split_texts(cand.get("texts")):
            key = text.lower()
            if key in text_seen and text_seen[key] != label:
                warnings.append(
                    f"duplicate prompt text '{text}' kept for {text_seen[key]}, skipped for {label}"
                )
                continue
            text_seen[key] = label
            if text not in texts:
                texts.append(text)
        if texts:
            cleaned.append({"label": label, "texts": texts[:8]})

    return cleaned, warnings


def _parse_models(models: str | None) -> list[str]:
    out: list[str] = []
    for token in re.split(r"\s*,\s*|\s+", models or ""):
        token = token.strip().lower()
        if not token:
            continue
        if token in _MODEL_TO_TOOL and token not in out:
            out.append(token)
    return out or list(_DEFAULT_MODELS)


def _score_with_model(
    model_key: str,
    image_path: str,
    candidates: list[dict[str, Any]],
    *,
    use_domain_knowledge: bool,
) -> dict[str, Any]:
    tool_name = _MODEL_TO_TOOL[model_key]
    adapter = GLOBAL_REGISTRY.get(tool_name)
    adapter.load()

    panel: list[str] = []
    text_to_label: dict[str, str] = {}
    for cand in candidates:
        for text in cand["texts"]:
            panel.append(text)
            text_to_label[text] = cand["label"]
    ranked_text = classify(
        adapter._impl,
        image_path,
        panel,
        use_domain_knowledge=use_domain_knowledge,
        device=adapter.device,
    )
    text_scores = dict(ranked_text)

    candidate_rows = []
    for cand in candidates:
        entries = [
            {
                "text": text,
                "probability": float(text_scores.get(text, 0.0)),
            }
            for text in cand["texts"]
        ]
        best = max(entries, key=lambda e: e["probability"])
        candidate_rows.append({
            "label": cand["label"],
            "score_raw_max_text": best["probability"],
            "best_text": best["text"],
            "texts": entries,
        })

    denom = sum(c["score_raw_max_text"] for c in candidate_rows) or 1.0
    for c in candidate_rows:
        c["score"] = c["score_raw_max_text"] / denom
    candidate_rows.sort(key=lambda c: -c["score"])

    return {
        "tool": tool_name,
        "model": model_key,
        "candidate_scores": candidate_rows,
        "top1": candidate_rows[0] if candidate_rows else None,
        "top_texts": [
            {
                "text": text,
                "candidate": text_to_label.get(text),
                "probability": float(prob),
            }
            for text, prob in ranked_text[:8]
        ],
    }


@register
class CFPDynamicCLIPAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="cfp_dynamic_clip",
        modality="CFP",
        task="dynamic_classification",
        description=(
            "Dynamic candidate-set CFP CLIP. The planner supplies a small "
            "task-specific candidate list as JSON, with each candidate mapped "
            "to one or more English prompt texts. The tool scores those texts "
            "with RetiZero and/or FLAIR, aggregates synonyms per candidate, "
            "and returns fused top-k. Use this for DR severity / ICDR grading "
            "or focused differentials where fixed 11-class CLIP labels are "
            "too coarse. Do not call without explicit candidates."
        ),
        input_size=(224, 224),
        labels=[],
        confidence_threshold=0.0,
        limitations=[
            "Requires explicit candidates; not suitable for routine no-argument analyze_image.",
            "Scores are softmax over the supplied candidate text panel, so they are relative, not calibrated probabilities.",
            "First version uses English RetiZero/FLAIR text encoders only; ViLReF Chinese dynamic prompts can be added later.",
        ],
        cost_class="slow",
        source_dir="(dynamic composite: cfp_retizero + cfp_flair)",
    )

    def _load_impl(self) -> None:
        self._impl = "dynamic_clip"

    def _predict_impl(
        self,
        image_path: str,
        candidates_json: Any = "",
        candidate_texts: str = "",
        task_hint: str = "",
        models: str = "retizero,flair",
        use_domain_knowledge: bool | str = True,
        top_k: int | str = 5,
        **_,
    ) -> AdapterResult:
        candidates, warnings = _parse_candidates(candidates_json, candidate_texts)
        if len(candidates) < 2:
            return AdapterResult(
                success=False,
                tool=self.metadata.name,
                modality="CFP",
                task=self.metadata.task,
                error=(
                    "cfp_dynamic_clip requires at least two candidates. Pass "
                    "candidates_json as a JSON list of strings or objects "
                    "{label, texts:[...]}; include an explicit normal/negative "
                    "candidate when appropriate."
                ),
                predictions={"warnings": warnings},
            )

        try:
            top_k_int = max(1, min(10, int(top_k)))
        except Exception:
            top_k_int = 5
        use_dk = (
            use_domain_knowledge
            if isinstance(use_domain_knowledge, bool)
            else str(use_domain_knowledge).strip().lower() not in {"0", "false", "no"}
        )
        model_keys = _parse_models(models)

        per_model: dict[str, Any] = {}
        for model_key in model_keys:
            try:
                per_model[model_key] = _score_with_model(
                    model_key,
                    image_path,
                    candidates,
                    use_domain_knowledge=use_dk,
                )
            except Exception as exc:
                per_model[model_key] = {
                    "model": model_key,
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                warnings.append(f"{model_key} failed: {type(exc).__name__}: {exc}")

        usable = {
            k: v for k, v in per_model.items()
            if isinstance(v, dict) and v.get("candidate_scores")
        }
        if not usable:
            return AdapterResult(
                success=False,
                tool=self.metadata.name,
                modality="CFP",
                task=self.metadata.task,
                predictions={
                    "task_hint": task_hint,
                    "candidates": candidates,
                    "per_model": per_model,
                    "warnings": warnings,
                },
                error="No dynamic CLIP backend produced scores.",
            )

        fused_rows = []
        for cand in candidates:
            label = cand["label"]
            scores = {}
            best_text_by_model = {}
            for model_key, result in usable.items():
                match = next(
                    (c for c in result["candidate_scores"] if c["label"] == label),
                    None,
                )
                if match:
                    scores[model_key] = float(match["score"])
                    best_text_by_model[model_key] = match["best_text"]
            if scores:
                fused_rows.append({
                    "label": label,
                    "score": sum(scores.values()) / len(scores),
                    "model_scores": scores,
                    "best_text_by_model": best_text_by_model,
                })
        fused_rows.sort(key=lambda c: -c["score"])

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="CFP",
            task=self.metadata.task,
            predictions={
                "task_hint": task_hint,
                "candidate_count": len(candidates),
                "text_count": sum(len(c["texts"]) for c in candidates),
                "candidates": candidates,
                "models_requested": model_keys,
                "models_used": list(usable),
                "aggregation": "per_model softmax over candidate texts; max text per candidate; mean normalized candidate score across models",
                "fused_topk": fused_rows[:top_k_int],
                "fused_top1": fused_rows[0] if fused_rows else None,
                "per_model": per_model,
                "warnings": warnings,
                "interpretation_note": (
                    "Scores are relative to the supplied candidate panel. "
                    "Use them as a CLIP prior and integrate with quantitative "
                    "tools such as cfp_retsam_segmentation/cfp_dr_421_assessment "
                    "and cfp_pdr_cascade."
                ),
            },
            confidence=float(fused_rows[0]["score"]) if fused_rows else None,
            metadata={
                "method": "llm_mapped_dynamic_candidate_clip_v1",
                "use_domain_knowledge": use_dk,
            },
        )
