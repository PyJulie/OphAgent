"""Vision-LLM prompt library — per-modality two-stage prompts (morphology
description → evidence-grounded differential), plus validators and a
capability probe.

Public entry points:
    from .cfp import STAGE1_SYSTEM, STAGE2_SYSTEM, build_user_prompt,
                     STAGE1_SCHEMA_SUMMARY, VALIDATORS, CROSS_TOOL_FIELDS
    from .capability_probe import vision_capability
    from .validators import run_validators, cross_check
"""
from . import cfp
from . import oct
from . import uwf
from . import vision_only
from .capability_probe import vision_capability, VISION_CAPABLE, TEXT_ONLY
from .validators import run_validators, cross_check, parse_json_lenient

PROMPTS = {"CFP": cfp, "OCT": oct, "UWF": uwf}

__all__ = [
    "PROMPTS",
    "cfp",
    "oct",
    "uwf",
    "vision_only",
    "vision_capability",
    "VISION_CAPABLE",
    "TEXT_ONLY",
    "run_validators",
    "cross_check",
    "parse_json_lenient",
]
