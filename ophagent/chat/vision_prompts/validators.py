"""
Generic validators for vision-LLM stage outputs.

Splits into:
  • Robust JSON parsing (handles markdown fences and trailing text)
  • Schema + self-consistency validators (provided by per-modality module)
  • Cross-tool comparators that consult the session's already-run
    classifier results
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterable


# ───────────────────────────────────────────────────────────────────────────
# 1. Lenient JSON parsing — many vision models wrap their JSON in fences.
# ───────────────────────────────────────────────────────────────────────────
_FENCE_PATTERNS = [
    re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE),
    re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL | re.IGNORECASE),
]


def parse_json_lenient(text: str) -> tuple[dict | list | None, str | None]:
    """Try hard to extract a JSON value from a possibly-wrapped string.

    Returns (parsed, error_message). On success, error_message is None.
    """
    if text is None:
        return None, "no content"
    s = text.strip()
    last: str | None = None
    # 1. Direct parse
    try:
        return json.loads(s), None
    except Exception:
        pass
    # 2. Markdown-fenced JSON
    for pat in _FENCE_PATTERNS:
        m = pat.search(s)
        if m:
            try:
                return json.loads(m.group(1)), None
            except Exception as e:
                last = f"fenced-block parse failed: {e}"
    # 3. Substring from first { to matching } (or [ to ])
    for opener, closer in (("{", "}"), ("[", "]")):
        i = s.find(opener)
        if i < 0:
            continue
        depth = 0
        for j in range(i, len(s)):
            if s[j] == opener:
                depth += 1
            elif s[j] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[i:j + 1]), None
                    except Exception as e:
                        last = f"substring parse failed: {e}"
                        break
    return None, last or "could not parse JSON from response"


# ───────────────────────────────────────────────────────────────────────────
# 2. Run a list of validators (each: stage1_json -> list[str])
# ───────────────────────────────────────────────────────────────────────────
def run_validators(stage1: dict, validators: Iterable[Callable]) -> list[str]:
    issues: list[str] = []
    for v in validators:
        try:
            out = v(stage1) or []
            issues.extend(out)
        except Exception as e:
            issues.append(f"validator {getattr(v, '__name__', v)!r} crashed: "
                          f"{type(e).__name__}: {e}")
    return issues


# ───────────────────────────────────────────────────────────────────────────
# 3. Cross-tool comparator — consults session.context.analyses
# ───────────────────────────────────────────────────────────────────────────
def _get_deep(d: Any, dotted: str) -> Any:
    cur = d
    for p in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(p)
        elif isinstance(cur, list):
            try:
                cur = cur[int(p)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if cur is None:
            return None
    return cur


def cross_check(
    stage1: dict,
    session_analyses: dict[str, dict],
    cross_fields: list[dict],
) -> list[str]:
    """Compare specific stage1 fields against classifier outputs.

    `session_analyses` shape: {image_path: {tool_name: jsonified_result}}.
    Only the *currently focused* image's tool results are used by callers.

    `cross_fields` schema (from per-modality module):
      [
        {
          "vision_path": "image_quality",
          "tool": "cfp_efiqa",
          "tool_path": "predictions.quality",
          "compare": fn(v, c) -> str | None,
        },
        ...
      ]
    """
    issues: list[str] = []
    if not session_analyses:
        return issues
    # Search for each tool across every analysed image (we don't know which
    # image the vision call was made for from inside this validator).
    for spec in cross_fields:
        v_val = _get_deep(stage1, spec["vision_path"])
        tool = spec["tool"]
        # Look across all analyses for this tool's most recent result
        tool_result = None
        for img_path, by_tool in session_analyses.items():
            if tool in by_tool:
                tool_result = by_tool[tool]
                # Keep iterating — last write wins for "most recent"
        if tool_result is None:
            continue
        c_val = _get_deep(tool_result, spec["tool_path"])
        try:
            msg = spec["compare"](v_val, c_val)
        except Exception as e:
            msg = f"comparator for {spec['vision_path']} crashed: {e}"
        if msg:
            issues.append(f"[{tool}] {msg}")
    return issues
