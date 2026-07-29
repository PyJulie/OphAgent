"""Deterministic execution policies for OphAgent effort tiers.

The language model proposes tools, but it does not own the lifecycle of a
clinical run.  This module is the single source of truth for tool-round budgets,
verification requirements, and component roles.  Provider-specific native
reasoning settings are deliberately kept separate from these policies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


VerifierMode = Literal["controller_rule", "rule", "independent_llm", "debate"]
VisionMode = Literal["disabled", "targeted", "exhaustive"]


@dataclass(frozen=True)
class EffortPolicy:
    name: str
    plan_rounds: int
    verify_escalations: int
    verifier_mode: VerifierMode
    vision_mode: VisionMode
    require_final_verifier: bool
    exhaustive_tools: bool
    meta_tool_limit: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


EFFORT_POLICIES: dict[str, EffortPolicy] = {
    "low": EffortPolicy(
        name="low",
        plan_rounds=1,
        verify_escalations=0,
        verifier_mode="controller_rule",
        vision_mode="disabled",
        require_final_verifier=False,
        exhaustive_tools=False,
        meta_tool_limit=1,
    ),
    "medium": EffortPolicy(
        name="medium",
        plan_rounds=2,
        verify_escalations=1,
        verifier_mode="rule",
        vision_mode="targeted",
        require_final_verifier=True,
        exhaustive_tools=False,
        meta_tool_limit=2,
    ),
    "high": EffortPolicy(
        name="high",
        plan_rounds=3,
        verify_escalations=1,
        verifier_mode="independent_llm",
        vision_mode="targeted",
        require_final_verifier=True,
        exhaustive_tools=False,
        meta_tool_limit=3,
    ),
    "max": EffortPolicy(
        name="max",
        plan_rounds=4,
        verify_escalations=2,
        verifier_mode="debate",
        vision_mode="targeted",
        require_final_verifier=True,
        exhaustive_tools=False,
        meta_tool_limit=4,
    ),
    "ultra": EffortPolicy(
        name="ultra",
        plan_rounds=5,
        verify_escalations=2,
        verifier_mode="debate",
        vision_mode="exhaustive",
        require_final_verifier=True,
        exhaustive_tools=True,
        meta_tool_limit=None,
    ),
}


def get_effort_policy(effort: str | None) -> EffortPolicy:
    """Return a validated policy, defaulting unknown legacy values to low."""
    return EFFORT_POLICIES.get(str(effort or "low").lower(), EFFORT_POLICIES["low"])

