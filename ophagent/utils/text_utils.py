"""Text utility functions for OphAgent."""
from __future__ import annotations

import re
from typing import List


def clean_text(text: str) -> str:
    """Strip extra whitespace and normalise newlines."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    sep: str = "\n",
) -> List[str]:
    """Split *text* into overlapping chunks of roughly *chunk_size* chars."""
    segments = text.split(sep)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for seg in segments:
        seg_len = len(seg) + len(sep)
        if current_len + seg_len > chunk_size and current:
            chunks.append(sep.join(current))
            # Keep overlap
            overlap_chars = 0
            overlap_segs: List[str] = []
            for s in reversed(current):
                overlap_chars += len(s) + len(sep)
                overlap_segs.insert(0, s)
                if overlap_chars >= chunk_overlap:
                    break
            current = overlap_segs
            current_len = sum(len(s) + len(sep) for s in current)
        current.append(seg)
        current_len += seg_len

    if current:
        chunks.append(sep.join(current))
    return [c for c in chunks if c.strip()]


def truncate_text(text: str, max_chars: int = 2000, suffix: str = "...") -> str:
    """Truncate *text* to *max_chars*, appending *suffix* if truncated."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


def extract_json_block(text: str) -> str:
    """Extract the first JSON block from LLM output (handles markdown fences).

    Uses a brace-balanced walk instead of a greedy regex so that nested
    objects are handled correctly and the search cannot backtrack over the
    entire document.
    """
    # Try ```json ... ``` first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)

    # Brace-balanced extraction: find the first complete {...} block
    start = text.find("{")
    if start != -1:
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    return text


def format_findings(findings: dict) -> str:
    """Format a findings dict into a human-readable report string."""
    lines = []
    for key, value in findings.items():
        label = key.replace("_", " ").title()
        if isinstance(value, list):
            lines.append(f"{label}: {', '.join(str(v) for v in value)}")
        elif isinstance(value, float):
            lines.append(f"{label}: {value:.4f}")
        else:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def parse_label_probabilities(text: str) -> dict:
    """Parse 'Label: 0.92' style strings from model outputs into a dict."""
    result = {}
    for match in re.finditer(r"([\w\s\-]+):\s*([0-9]*\.?[0-9]+)", text):
        label = match.group(1).strip()
        prob = float(match.group(2))
        result[label] = prob
    return result
