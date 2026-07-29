import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from ophagent.adapters.base import AdapterResult
from ophagent.adapters.cfp.biomarkers import CFPDRWorkupAdapter
from ophagent.adapters.cfp import biomarkers
from ophagent.chat.oph_session import OPH_SYSTEM_PROMPT, OphSession
from ophagent.chat.oph_tools import (
    OphToolKit,
    _make_retsam_headline,
    _verifier_top_candidates,
)


def _save_mask(path, array):
    Image.fromarray(array.astype(np.uint8)).save(path)
    return str(path)


def test_retsam_overlapping_dr_amd_hemorrhage_is_etiologically_ambiguous(
        tmp_path):
    dr_mask = np.zeros((20, 20), dtype=np.uint8)
    dr_mask[2:12, 2:12] = 255
    amd_mask = np.zeros((20, 20), dtype=np.uint8)
    amd_mask[3:12, 2:12] = 255
    fundus_mask = np.ones((20, 20), dtype=np.uint8) * 255

    result = {
        "predictions": {
            "mask_files": {
                "fundus_mask": _save_mask(
                    tmp_path / "fundus.png", fundus_mask
                ),
                "lesion_dr_hemorrhage": _save_mask(
                    tmp_path / "dr_hem.png", dr_mask
                ),
                "lesion_amd_patch_hemorrhage": _save_mask(
                    tmp_path / "amd_hem.png", amd_mask
                ),
            },
            "quantitative": {
                "top_line": {
                    "AMD_patch_hemorrhage_present": True,
                },
                "lesions": {
                    "groups": {
                        "lesion_dr": {
                            "classes": {
                                "hemorrhage": {
                                    "count": 13,
                                    "area": {"px": 100},
                                    "spatial": {
                                        "macula_zone_counts": {},
                                    },
                                },
                                "exudate": {
                                    "count": 2,
                                    "area": {"px": 20},
                                },
                            },
                        },
                        "lesion_amd": {
                            "classes": {
                                "drusen": {"count": 0},
                                "patch_hemorrhage": {
                                    "count": 1,
                                    "area": {"px": 90},
                                },
                            },
                        },
                    },
                },
            },
            "meta": {
                "modules_succeeded": ["lesions"],
                "module_errors": {},
            },
        },
    }

    headline = _make_retsam_headline(result)

    etiology = headline["hemorrhage_etiology"]
    assert etiology["status"] == "ambiguous"
    assert etiology["overlap_fraction_of_dr_head"] == 0.9
    assert (
        headline["diabetic_retinopathy_signs"]["dr_signal_confidence"]
        == "ambiguous"
    )
    assert (
        headline["diabetic_retinopathy_signs"][
            "morphology_signal_confidence_before_etiology_guard"
        ]
        == "high"
    )
    summary = " ".join(headline["natural_language_summary"])
    assert "unresolved etiology" in summary
    assert "independent proof of DR" in summary
    assert "strong DR sign" not in summary


def test_verifier_flags_reject_quality_low_clip_agreement_and_overlap():
    toolkit = OphToolKit()
    findings = {
        "tools_run": [
            "cfp_efiqa",
            "cfp_clip_ensemble",
            "cfp_retsam_segmentation",
        ],
        "results": [
            {
                "tool": "cfp_efiqa",
                "predictions": {
                    "quality": "Reject",
                    "usable_area_ratio": 0.46,
                },
            },
            {
                "tool": "cfp_clip_ensemble",
                "predictions": {
                    "agreement_level": "low",
                    "fused_top1": "Pathological myopia",
                    "fused_top1_probability": 0.296,
                },
            },
            {
                "tool": "cfp_retsam_segmentation",
                "predictions": {
                    "llm_headline": {
                        "hemorrhage_etiology": {
                            "status": "ambiguous",
                            "overlap_fraction_of_dr_head": 0.705,
                        },
                        "diabetic_retinopathy_signs": {
                            "dr_signal_confidence": "ambiguous",
                        },
                        "amd_signs": {
                            "patch_hemorrhage_present": True,
                        },
                    },
                },
            },
        ],
    }

    result = toolkit.execute(
        "verify_findings",
        findings_json=json.dumps(findings),
    )

    warnings = " ".join(result["warnings"])
    assert "rejected the CFP image" in warnings
    assert "low inter-model agreement" in warnings
    assert "substantially overlap" in warnings
    assert "active hemorrhagic macular lesion" in warnings
    assert "mutually consistent" not in result["recommendation"]


def test_dr_workup_suppresses_highly_confounded_pdr(monkeypatch):
    outputs = {
        "cfp_efiqa": AdapterResult(
            success=True,
            tool="cfp_efiqa",
            modality="CFP",
            task="quality",
            predictions={"quality": "Good"},
            confidence=0.90,
        ),
        "cfp_eyeq": AdapterResult(
            success=True,
            tool="cfp_eyeq",
            modality="CFP",
            task="quality",
            predictions={"quality": "Good", "is_rejected": False},
            confidence=0.90,
        ),
        "cfp_pdr_cascade": AdapterResult(
            success=True,
            tool="cfp_pdr_cascade",
            modality="CFP",
            task="classification",
            predictions={
                "category": "Active PDR",
                "predicted_reasons": ["neovascularisation"],
            },
            confidence=0.97,
        ),
        "cfp_clip_multi_disease": AdapterResult(
            success=True,
            tool="cfp_clip_multi_disease",
            modality="CFP",
            task="classification",
            predictions={
                "top_3": [
                    {
                        "label_en": "Pathological myopia",
                        "probability": 0.31,
                    },
                    {"label_en": "Glaucoma", "probability": 0.20},
                    {"label_en": "AMD", "probability": 0.18},
                ],
            },
            confidence=0.31,
        ),
    }

    monkeypatch.setattr(
        biomarkers.GLOBAL_REGISTRY,
        "predict",
        lambda name, image_path, **kwargs: outputs[name],
    )

    result = CFPDRWorkupAdapter(device="cpu").predict("example.png")

    assert result.success
    assert result.undetermined
    assert result.confidence == 0.0
    assert result.predictions["do_not_report_as_pdr"] is True
    assert result.predictions["pdr_eligible_for_reporting"] is False
    assert (
        result.predictions["raw_pdr_category_before_confound_guard"]
        == "Active PDR"
    )
    assert (
        result.predictions["pdr_category"]
        == "indeterminate_non_dr_confound"
    )


def test_verifier_blocks_composite_dr_bypasses_and_prioritises_mnv():
    toolkit = OphToolKit()
    findings = {
        "tools_run": ["cfp_dr_421_assessment", "cfp_dr_workup"],
        "results": [
            {
                "tool": "cfp_dr_421_assessment",
                "predictions": {
                    "eligible_for_dr_grading": False,
                    "etiology_guard": {"status": "ambiguous"},
                    "severity_label": "indeterminate_hemorrhagic_macular_lesion",
                },
            },
            {
                "tool": "cfp_dr_workup",
                "predictions": {
                    "pdr_confound_severity": "high",
                    "pdr_eligible_for_reporting": False,
                    "do_not_report_as_pdr": True,
                    "raw_pdr_category_before_confound_guard": "Active PDR",
                },
            },
        ],
    }

    candidates = _verifier_top_candidates(findings)
    result = toolkit.execute(
        "verify_findings",
        findings_json=json.dumps(findings),
    )
    warnings = " ".join(result["warnings"])

    assert candidates[:2] == ["Neovascular AMD/PCV", "Myopic CNV"]
    assert "suppressed its DR severity proxy" in warnings
    assert "audit-only" in warnings


def test_system_prompt_separates_active_macular_lesion_from_background_myopia():
    assert "Primary impression must name that active lesion separately" in (
        OPH_SYSTEM_PROMPT
    )
    assert "suspected myopic CNV versus nAMD/PCV" in OPH_SYSTEM_PROMPT


def test_single_image_core_evidence_requires_fresh_verifier():
    session = OphSession.new(effort="medium")
    session.context.current_image = "example.png"
    session.context.current_modality = "CFP"

    assert session._single_image_verifier_needed(
        {"cfp_clip_ensemble": "ok"}
    )
    assert not session._single_image_verifier_needed(
        {
            "cfp_clip_ensemble": "ok",
            "verify_findings": "ok",
        }
    )

    session.effort = "low"
    assert not session._single_image_verifier_needed(
        {"cfp_clip_ensemble": "ok"}
    )


def test_verifier_result_must_be_completed_and_machine_readable():
    assert OphSession._verifier_result_valid({
        "status": "ok",
        "verify_passed": True,
        "next_actions": [],
    })
    assert not OphSession._verifier_result_valid({
        "status": "warning",
        "verify_passed": False,
        "error": "findings_json was not valid JSON",
    })
    assert not OphSession._verifier_result_valid({
        "status": "ok",
        "next_actions": [],
    })
    assert not OphSession._verifier_result_valid({
        "status": "ok",
        "verify_passed": True,
        "next_actions": [],
        "n_tools_run": 0,
    })


def test_verifier_treats_empty_json_object_as_session_cache_request():
    image_path = "example.png"
    session = OphSession.new(effort="medium")
    session.context.current_image = image_path
    session.context.current_modality = "CFP"
    session.context.analyses = {
        image_path: {
            "cfp_clip_ensemble": {
                "success": True,
                "predictions": {
                    "agreement_level": "low",
                    "fused_top1": "Age-related macular degeneration",
                    "fused_top1_probability": 0.39,
                },
                "confidence": 0.39,
            },
        },
    }
    toolkit = OphToolKit(session=session)

    result = toolkit.execute(
        "verify_findings",
        findings_json="{}",
        image_path=image_path,
    )

    assert result["input_source"] == "session_cache"
    assert result["n_tools_run"] == 1
    assert (
        result["evidence"]["clip_ensemble"]["fused_top1"]
        == "Age-related macular degeneration"
    )
    assert "low inter-model agreement" in " ".join(result["warnings"])


def test_verifier_cannot_pass_without_any_evidence():
    result = OphToolKit().execute(
        "verify_findings",
        findings_json="{}",
    )

    assert result["n_tools_run"] == 0
    assert result["verify_passed"] is False
    assert "No structured tool results" in " ".join(result["issues"])


def test_chat_reverifies_after_verifier_requested_action():
    image_path = "example.png"
    verify_calls = 0
    tool_choices = []

    class FakeToolkit:
        tools = {}

        @staticmethod
        def get_all_schemas():
            return []

        def execute(self, name, **kwargs):
            nonlocal verify_calls
            if name == "cfp_clip_ensemble":
                return {
                    "success": True,
                    "predictions": {
                        "fused_top1": "Pathological myopia",
                        "fused_top1_probability": 0.30,
                    },
                    "confidence": 0.30,
                }
            if name == "vision_impression":
                return {
                    "status": "ok",
                    "impression_markdown": "Macular hemorrhagic lesion.",
                }
            if name == "verify_findings":
                verify_calls += 1
                if verify_calls == 1:
                    return {
                        "status": "ok",
                        "verify_passed": False,
                        "warnings": ["ambiguous macular hemorrhage"],
                        "next_actions": [{"tool": "vision_impression"}],
                        "recommendation": "Run targeted vision review.",
                    }
                return {
                    "status": "ok",
                    "verify_passed": True,
                    "warnings": ["etiology remains limited-confidence"],
                    "next_actions": [],
                    "recommendation": "Finalise with explicit differential.",
                }
            raise AssertionError(f"unexpected tool: {name}")

    def tool_call(name, arguments, call_id):
        return SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(
                name=name,
                arguments=json.dumps(arguments),
            ),
        )

    responses = iter([
        SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        tool_call(
                            "cfp_clip_ensemble",
                            {"image_path": image_path},
                            "call-clip",
                        ),
                    ],
                ),
            )],
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="Premature diagnostic draft.",
                    tool_calls=[],
                ),
            )],
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        tool_call(
                            "verify_findings",
                            {"findings_json": "", "image_path": image_path},
                            "call-verify-1",
                        ),
                    ],
                ),
            )],
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        tool_call(
                            "vision_impression",
                            {"image_path": image_path},
                            "call-vision",
                        ),
                    ],
                ),
            )],
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="Draft after the requested action.",
                    tool_calls=[],
                ),
            )],
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        tool_call(
                            "verify_findings",
                            {"findings_json": "", "image_path": image_path},
                            "call-verify-2",
                        ),
                    ],
                ),
            )],
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="Final verified diagnostic answer.",
                    tool_calls=[],
                ),
            )],
        ),
    ])

    session = OphSession.new(
        backend="aigcbest",
        model="gpt-5.5",
        effort="medium",
    )
    session.context.current_image = image_path
    session.context.current_modality = "CFP"
    session._toolkit = FakeToolkit()
    session._ensure_client = lambda: None

    def fake_completion(messages, tools, emit, tool_choice="auto"):
        tool_choices.append(tool_choice)
        return next(responses)

    session._safe_completion = fake_completion
    reply = session.chat("Interpret the image.", max_tool_steps=10)

    assert verify_calls == 2
    assert tool_choices[2] == {
        "type": "function",
        "function": {"name": "verify_findings"},
    }
    assert tool_choices[5] == {
        "type": "function",
        "function": {"name": "verify_findings"},
    }
    assert "Final verified diagnostic answer." in reply
