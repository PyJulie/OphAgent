from __future__ import annotations

from ophagent.evaluation import (
    DR_SEVERITY_ICDR_SINGLE_IMAGE,
    EvaluationRunConfig,
    EvaluationRunResult,
)
from ophagent.evaluation.runner import architecture_ablation_flags, parse_final


def test_evaluation_package_exposes_stable_protocol_contract() -> None:
    config = EvaluationRunConfig()
    result = EvaluationRunResult(
        image_path="example.png",
        protocol_id=DR_SEVERITY_ICDR_SINGLE_IMAGE.task_id,
        backend=config.backend,
        model=config.model,
        effort=config.effort,
        elapsed_s=0.1,
    )

    assert DR_SEVERITY_ICDR_SINGLE_IMAGE.task_id == (
        "dr_severity_icdr_single_image_v2"
    )
    assert result.to_dict()["protocol_id"] == (
        "dr_severity_icdr_single_image_v2"
    )


def test_evaluation_runner_parses_final_block_and_architecture_arms() -> None:
    parsed = parse_final(
        'Brief evidence.\n===FINAL===\n{"grade": 2, "confidence": 0.8}'
    )

    assert parsed["grade"] == 2
    assert parsed["parse_method"] == "json_block"
    assert architecture_ablation_flags("full") == (False, False)
    assert architecture_ablation_flags("planner") == (False, True)
    assert architecture_ablation_flags("single") == (True, True)
