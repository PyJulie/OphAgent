"""
Tool definitions for the OCT Agent.

Each tool wraps a model inference call with a standardized interface
that the LLM agent can discover and invoke.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ...inference.predictor import OphPredictor
from ...models.caption.caption_model import OphCaptionModel
from ...utils.paths import OUTPUT_DIR


@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: list[ToolParameter]
    function: Callable | None = None

    def to_schema(self) -> dict:
        properties = {}
        required = []
        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            if p.default is not None:
                prop["default"] = p.default
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class OphAgentToolKit:
    """Collection of all OCT analysis tools available to the agent."""

    def __init__(
        self,
        predictor: OphPredictor,
        caption_model: OphCaptionModel | None = None,
        report_output_root: str | None = None,
    ):
        self.predictor = predictor
        self.caption_model = caption_model
        self.report_output_root = report_output_root or str(OUTPUT_DIR)
        self.last_report_dir: str | None = None
        self.tools: dict[str, Tool] = {}
        self._register_all_tools()

    def _register_all_tools(self) -> None:
        self.tools["assess_quality"] = Tool(
            name="assess_quality",
            description=(
                "Assess the quality of an OCT image. Returns quality level "
                "(high/medium/low) and confidence score. Should be called first "
                "to determine if the image is suitable for analysis."
            ),
            parameters=[
                ToolParameter("image_path", "string", "Path to the OCT image file"),
            ],
            function=self._assess_quality,
        )

        self.tools["classify_disease"] = Tool(
            name="classify_disease",
            description=(
                "Classify an OCT image into disease categories. "
                "Detects conditions like AMD, DME, CNV, Drusen, Glaucoma, etc. "
                "Returns predicted disease, confidence, and per-class probabilities."
            ),
            parameters=[
                ToolParameter("image_path", "string", "Path to the OCT image file"),
                ToolParameter(
                    "model_variant", "string",
                    "Which classifier to use: 'basic' (4-class Kermany), "
                    "'octdl' (7-class OCTDL), or 'broad' (8-class OCT-C8)",
                    required=False, default="broad",
                    enum=["basic", "octdl", "broad"],
                ),
            ],
            function=self._classify_disease,
        )

        self.tools["segment_fluid"] = Tool(
            name="segment_fluid",
            description=(
                "Segment retinal fluid regions in an OCT image. "
                "Detects and delineates IRF (intraretinal fluid), "
                "SRF (subretinal fluid), and PED (pigment epithelial detachment). "
                "Returns a segmentation mask and area measurements."
            ),
            parameters=[
                ToolParameter("image_path", "string", "Path to the OCT image file"),
            ],
            function=self._segment_fluid,
        )

        self.tools["segment_layers"] = Tool(
            name="segment_layers",
            description=(
                "Segment retinal layer boundaries in an OCT image. "
                "Identifies 9 retinal layers (ILM, NFL-GCL, IPL, INL, OPL, ONL, "
                "ELM, IS-OS, RPE). Returns layer thickness measurements."
            ),
            parameters=[
                ToolParameter("image_path", "string", "Path to the OCT image file"),
            ],
            function=self._segment_layers,
        )

        self.tools["denoise_image"] = Tool(
            name="denoise_image",
            description=(
                "Remove speckle noise from an OCT image. "
                "Improves image quality for better visualization and downstream analysis. "
                "Returns the denoised image."
            ),
            parameters=[
                ToolParameter("image_path", "string", "Path to the OCT image file"),
                ToolParameter(
                    "output_path", "string",
                    "Path to save the denoised image",
                    required=False,
                ),
            ],
            function=self._denoise_image,
        )

        self.tools["super_resolve"] = Tool(
            name="super_resolve",
            description=(
                "Enhance resolution of a low-quality OCT image by 2x. "
                "Useful for images from portable or older devices. "
                "Returns the super-resolved image."
            ),
            parameters=[
                ToolParameter("image_path", "string", "Path to the OCT image file"),
                ToolParameter(
                    "output_path", "string",
                    "Path to save the super-resolved image",
                    required=False,
                ),
            ],
            function=self._super_resolve,
        )

        self.tools["build_visual_report"] = Tool(
            name="build_visual_report",
            description=(
                "Generate a styled HTML clinical report with embedded figures: "
                "Grad-CAM heatmap + detection boxes from the classifier, "
                "fluid segmentation overlay + per-class bounding boxes, "
                "layer segmentation overlay, and the LLM clinical text. "
                "Call this AFTER all other tools and AFTER you have written the "
                "clinical report markdown. Pass the clinical report as "
                "report_markdown and the prior findings as findings_json."
            ),
            parameters=[
                ToolParameter("image_path", "string", "Path to the OCT image file"),
                ToolParameter(
                    "report_markdown", "string",
                    "Markdown-formatted clinical report text to embed in the HTML",
                ),
                ToolParameter(
                    "findings_json", "string",
                    "JSON string of prior tool findings (classification/quality/etc.)",
                    required=False, default="{}",
                ),
                ToolParameter(
                    "classifier", "string",
                    "Which classifier to use for Grad-CAM "
                    "(oct_classifier_octdl / oct_classifier_kermany / oct_classifier_broad)",
                    required=False, default="oct_classifier_octdl",
                    enum=["oct_classifier_octdl", "oct_classifier_kermany", "oct_classifier_broad"],
                ),
            ],
            function=self._build_visual_report,
        )

        self.tools["caption_image"] = Tool(
            name="caption_image",
            description=(
                "Generate a clinical-style free-text description of an OCT image "
                "using a vision-language model. The caption comments on image "
                "quality, layer integrity, lesions (fluid, drusen, PED, atrophy), "
                "and the appearance of the fovea/choroid. Useful for documenting "
                "visual findings the discriminative models may not name explicitly. "
                "Pass extra_context (JSON of earlier findings) to ground the caption."
            ),
            parameters=[
                ToolParameter("image_path", "string", "Path to the OCT image file"),
                ToolParameter(
                    "extra_context", "string",
                    "Optional JSON string of earlier findings to ground the caption",
                    required=False, default="",
                ),
            ],
            function=self._caption_image,
        )

        self.tools["generate_report"] = Tool(
            name="generate_report",
            description=(
                "Generate a structured clinical report from analysis results. "
                "Combines findings from multiple tools into a coherent summary."
            ),
            parameters=[
                ToolParameter(
                    "findings", "string",
                    "JSON string of analysis findings from other tools",
                ),
                ToolParameter(
                    "patient_info", "string",
                    "Optional patient context (age, history, etc.)",
                    required=False,
                ),
            ],
            function=self._generate_report,
        )

    def get_tool(self, name: str) -> Tool:
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found. Available: {list(self.tools.keys())}")
        return self.tools[name]

    def get_all_schemas(self) -> list[dict]:
        return [tool.to_schema() for tool in self.tools.values()]

    def execute(self, tool_name: str, **kwargs) -> dict[str, Any]:
        tool = self.get_tool(tool_name)
        if tool.function is None:
            raise RuntimeError(f"Tool '{tool_name}' has no implementation")
        return tool.function(**kwargs)

    # ── Tool implementations ──────────────────────────────────────────────

    def _assess_quality(self, image_path: str) -> dict[str, Any]:
        try:
            return self.predictor.predict("oct_quality_assessor", image_path)
        except Exception as e:
            return {"task": "quality_assessment", "error": str(e), "quality": "unknown"}

    def _classify_disease(
        self, image_path: str, model_variant: str = "broad"
    ) -> dict[str, Any]:
        variant_map = {
            "basic": "oct_classifier_kermany",
            "octdl": "oct_classifier_octdl",
            "broad": "oct_classifier_broad",
        }
        model_name = variant_map.get(model_variant, "oct_classifier_broad")
        return self.predictor.predict(model_name, image_path)

    def _segment_fluid(self, image_path: str) -> dict[str, Any]:
        result = self.predictor.predict("oct_fluid_segmentor", image_path)
        mask = result.pop("mask", None)
        if mask is not None:
            result["has_fluid"] = bool((mask > 0).any())
            result["fluid_types_detected"] = [
                name for name, area in result.get("class_areas", {}).items()
                if name != "Background" and area > 0
            ]
        return result

    def _segment_layers(self, image_path: str) -> dict[str, Any]:
        result = self.predictor.predict("oct_layer_segmentor", image_path)
        result.pop("mask", None)
        return result

    def _denoise_image(
        self, image_path: str, output_path: str | None = None
    ) -> dict[str, Any]:
        result = self.predictor.predict("oct_denoiser", image_path)
        denoised = result.pop("denoised_image", None)
        if denoised is not None and output_path:
            import cv2
            cv2.imwrite(output_path, denoised)
            result["saved_to"] = output_path
        result["status"] = "success"
        return result

    def _super_resolve(
        self, image_path: str, output_path: str | None = None
    ) -> dict[str, Any]:
        result = self.predictor.predict("oct_super_resolver", image_path)
        sr_image = result.pop("super_resolved_image", None)
        if sr_image is not None and output_path:
            import cv2
            cv2.imwrite(output_path, sr_image)
            result["saved_to"] = output_path
        result["status"] = "success"
        return result

    def _build_visual_report(
        self,
        image_path: str,
        report_markdown: str,
        findings_json: str = "{}",
        classifier: str = "oct_classifier_octdl",
    ) -> dict[str, Any]:
        from ..report_builder import build_visual_report
        import json as _json
        import time

        try:
            findings = _json.loads(findings_json) if findings_json else {}
        except _json.JSONDecodeError:
            findings = {"raw": findings_json}

        stem = Path(image_path).stem
        run_dir = Path(self.report_output_root) / f"{stem}_{int(time.time())}"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            patient_ctx = findings.get("patient_context") or findings.get("patient_info", "")
            result = build_visual_report(
                image_path=image_path,
                registry=self.predictor.registry,
                predictor=self.predictor,
                findings=findings,
                clinical_report_md=report_markdown,
                output_dir=run_dir,
                classifier_model_name=classifier,
                patient_context=patient_ctx,
            )
            self.last_report_dir = str(run_dir)
            return {
                "task": "visual_report",
                "status": "success",
                "report_html": result["report_html"],
                "report_pdf": result.get("report_pdf"),
                "figures_dir": result["figures_dir"],
                "grad_cam_summary": {
                    "predicted_class": result["grad_cam"].get("predicted_class"),
                    "confidence": result["grad_cam"].get("confidence"),
                    "num_boxes": len(result["grad_cam"].get("boxes", [])),
                } if result.get("grad_cam") and "error" not in result["grad_cam"] else None,
                "fluid_num_boxes": len(result["fluid"].get("boxes", []))
                    if result.get("fluid") and "error" not in result["fluid"] else None,
            }
        except Exception as e:
            return {"task": "visual_report", "status": "error", "error": str(e)}

    def _caption_image(
        self, image_path: str, extra_context: str = ""
    ) -> dict[str, Any]:
        if self.caption_model is None:
            return {
                "task": "caption",
                "error": "caption model not configured. Pass OphCaptionModel to OphAgentToolKit.",
                "caption": "",
            }
        try:
            text = self.caption_model.caption(image_path, extra_context=extra_context)
            return {
                "task": "caption",
                "caption": text,
                "model": self.caption_model.model,
                "backend": self.caption_model.backend,
            }
        except Exception as e:
            return {"task": "caption", "error": str(e), "caption": ""}

    def _generate_report(
        self, findings: str, patient_info: str | None = None
    ) -> dict[str, Any]:
        try:
            findings_data = json.loads(findings)
        except json.JSONDecodeError:
            findings_data = {"raw": findings}

        report_sections = []

        if "quality_assessment" in str(findings_data):
            quality = findings_data.get("quality", "unknown")
            report_sections.append(f"Image Quality: {quality}")

        if "classification" in str(findings_data):
            disease = findings_data.get("predicted_class", "Unknown")
            conf = findings_data.get("confidence", 0)
            report_sections.append(
                f"Diagnosis: {disease} (confidence: {conf:.1%})"
            )

        if "segmentation" in str(findings_data):
            areas = findings_data.get("class_areas", {})
            report_sections.append(f"Segmentation findings: {areas}")

        return {
            "task": "report",
            "report": "\n".join(report_sections) if report_sections else "No findings to report.",
            "findings_summary": findings_data,
        }
