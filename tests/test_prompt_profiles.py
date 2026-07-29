from __future__ import annotations

import json

import pytest

from ophagent.chat.oph_session import OphSession, _effort_directive
from ophagent.chat.prompt_profiles import (
    COMPACT_MAC_SYSTEM_PROMPT,
    TASK_FOCUSED_DR_SYSTEM_PROMPT,
    compact_mac_tool_names,
    normalize_prompt_profile,
    task_focused_dr_tool_names,
    tool_result_for_profile,
    tool_schemas_for_profile,
)


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"exact description for {name}",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "exact path description",
                    }
                },
                "required": ["image_path"],
            },
        },
    }


def _complete_schema_set() -> list[dict]:
    names = set(compact_mac_tool_names()) | set(task_focused_dr_tool_names())
    schemas = [_schema(name) for name in sorted(names)]
    schemas += [_schema("oct_fmue_16class"), _schema("build_visual_report")]
    return schemas


def test_standard_remains_the_session_default() -> None:
    assert OphSession.new().prompt_profile == "standard"


def test_prompt_profile_persistence_is_backward_compatible(tmp_path) -> None:
    compact_path = tmp_path / "compact.json"
    compact = OphSession.new(prompt_profile="compact-mac")
    compact.save(compact_path)
    assert OphSession.load(compact_path).prompt_profile == "compact-mac"

    legacy_path = tmp_path / "legacy.json"
    legacy_payload = json.loads(compact_path.read_text(encoding="utf-8"))
    legacy_payload.pop("prompt_profile")
    legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
    assert OphSession.load(legacy_path).prompt_profile == "standard"


def test_profile_alias_and_unknown_profile() -> None:
    assert normalize_prompt_profile("compact") == "compact-mac"
    assert normalize_prompt_profile("compact_mac") == "compact-mac"
    assert normalize_prompt_profile("dr-grading") == "task-focused-dr"
    with pytest.raises(ValueError, match="unknown prompt profile"):
        normalize_prompt_profile("experimental")


def test_compact_mac_keeps_selected_schemas_byte_equivalent() -> None:
    schemas = _complete_schema_set()
    selected = tool_schemas_for_profile(
        "compact-mac",
        schemas,
        attached_modalities=["CFP"],
        multimodal=False,
    )
    expected = [
        schema
        for schema in schemas
        if schema["function"]["name"] in compact_mac_tool_names()
    ]
    assert selected == expected
    assert json.dumps(selected, sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert "oct_fmue_16class" not in {
        schema["function"]["name"] for schema in selected
    }
    assert "build_visual_report" not in {
        schema["function"]["name"] for schema in selected
    }


@pytest.mark.parametrize(
    ("modalities", "multimodal"),
    [(["OCT"], False), (["CFP", "OCT"], True), ([], False)],
)
def test_compact_mac_fails_closed_outside_single_cfp(
    modalities: list[str], multimodal: bool
) -> None:
    with pytest.raises(ValueError, match="requires exactly one attached modality"):
        tool_schemas_for_profile(
            "compact-mac",
            _complete_schema_set(),
            attached_modalities=modalities,
            multimodal=multimodal,
        )


def test_compact_prompt_preserves_safety_contract() -> None:
    prompt = COMPACT_MAC_SYSTEM_PROMPT.lower()
    for required_text in (
        "never invent",
        "undetermined",
        "quality result alone is not a diagnosis",
        "verify_findings",
        "preserve uncertainty",
        "machine-readable output contract",
    ):
        assert required_text in prompt


def test_compact_ultra_directive_matches_exposed_tool_contract() -> None:
    directive = _effort_directive(
        "ultra",
        vision_available=True,
        attached_modalities=["CFP"],
        prompt_profile="compact-mac",
    )
    for tool_name in compact_mac_tool_names():
        assert f"`{tool_name}`" in directive
    for unavailable_tool in (
        "cfp_pdr_cascade",
        "cfp_retizero",
        "cfp_flair",
        "cfp_glaucoma",
    ):
        assert f"`{unavailable_tool}`" not in directive
    assert "exactly once" in directive
    assert "do not compensate by repeating" in directive.lower()


def test_task_focused_dr_profile_keeps_only_icdr_tools() -> None:
    schemas = _complete_schema_set()
    selected = tool_schemas_for_profile(
        "task-focused-dr",
        schemas,
        attached_modalities=["CFP"],
        multimodal=False,
    )
    assert {
        schema["function"]["name"] for schema in selected
    } == set(task_focused_dr_tool_names())
    prompt = TASK_FOCUSED_DR_SYSTEM_PROMPT.lower()
    for required_text in (
        "icdr 0--4",
        "never invent",
        "4-2-1",
        "pdr",
        "verify_findings",
    ):
        assert required_text in prompt


def test_standard_tool_result_is_not_projected() -> None:
    result = {"predictions": {"x": 1}, "figure_urls": {"mask": "/large/path"}}
    assert tool_result_for_profile("standard", "cfp_efiqa", result) is result


def test_compact_retsam_keeps_clinical_headline_and_drops_assets() -> None:
    headline = {
        "optic_disc": {"vCDR": 0.63},
        "diabetic_retinopathy_signs": {"dr_signal_confidence": "high"},
        "amd_signs": {"drusen_count": 12},
        "other_findings": {"epiretinal_membrane_count": 1},
    }
    result = {
        "success": True,
        "undetermined": False,
        "llm_headline": headline,
        "predictions": {"mask_files": {"dr": "/large/mask/path"}},
        "figures": {"dr": "large-array"},
        "figure_urls": {"dr": "/large/overlay/path"},
        "metadata": {"output_dir": "/large/output/path"},
    }
    projected = tool_result_for_profile(
        "compact-mac", "cfp_retsam_segmentation", result
    )
    assert projected == {
        "success": True,
        "undetermined": False,
        "llm_headline": headline,
    }


def test_compact_verifier_drops_transcript_but_keeps_verdict() -> None:
    result = {
        "verify_passed": True,
        "next_actions": [],
        "recommendation": "finalise",
        "debate_review": {
            "final_diagnosis": "AMD",
            "resolved": True,
            "reason": "Concordant evidence",
            "request_tool": "",
            "_transcript": [{"large": "payload"}],
        },
    }
    projected = tool_result_for_profile(
        "compact-mac", "verify_findings", result
    )
    assert projected["verify_passed"] is True
    assert projected["debate_review"]["final_diagnosis"] == "AMD"
    assert "_transcript" not in projected["debate_review"]
