"""Protocol-first OphAgent evaluation utilities.

This package provides task contracts, bounded runners, and dependency-free
metrics for reproducible evaluations built on the main ``OphSession`` runtime.
"""

from .protocols import (
    DR_SEVERITY_ICDR_SINGLE_IMAGE,
    EvidenceRequirement,
    EffortPolicy,
    TaskProtocol,
    get_effort_policy,
    get_protocol,
)
from .runner import EvaluationRunConfig, EvaluationRunResult, run_image

__all__ = [
    "DR_SEVERITY_ICDR_SINGLE_IMAGE",
    "EvidenceRequirement",
    "EffortPolicy",
    "TaskProtocol",
    "EvaluationRunConfig",
    "EvaluationRunResult",
    "get_effort_policy",
    "get_protocol",
    "run_image",
]
