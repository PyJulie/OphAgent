"""Lightweight heuristic fallbacks that keep OphAgent runnable offline.

These routines are not intended to replace trained clinical models. They
provide deterministic, structured outputs when a service, weight file, or
dependency is unavailable so the end-to-end agent loop can still execute.
"""
from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ophagent.utils.image_utils import load_image_pil


ImageSource = Union[str, Path, Image.Image]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _to_pil(image_or_path: ImageSource) -> Image.Image:
    if isinstance(image_or_path, Image.Image):
        return image_or_path.convert("RGB")
    return load_image_pil(image_or_path)


def _with_meta(
    payload: Dict[str, object],
    backend: str = "heuristic-fallback",
    error: Optional[Exception] = None,
) -> Dict[str, object]:
    result = dict(payload)
    result.setdefault("backend", backend)
    if error is not None:
        result.setdefault("fallback_reason", str(error))
    return result


def _normalise_scores(scores: Dict[str, float]) -> Dict[str, float]:
    clipped = {k: max(0.001, float(v)) for k, v in scores.items()}
    total = sum(clipped.values()) or 1.0
    return {k: float(v / total) for k, v in clipped.items()}


def _image_stats(image_or_path: ImageSource) -> Dict[str, float]:
    img = _to_pil(image_or_path)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
    sharpness = _clamp((gx + gy) * 2.5)
    return {
        "width": float(img.width),
        "height": float(img.height),
        "brightness": float(arr.mean()),
        "contrast": float(arr.std()),
        "sharpness": sharpness,
        "red": float(arr[:, :, 0].mean()),
        "green": float(arr[:, :, 1].mean()),
        "blue": float(arr[:, :, 2].mean()),
    }


def quality_assessment(
    image_or_path: ImageSource,
    error: Optional[Exception] = None,
) -> Dict[str, object]:
    stats = _image_stats(image_or_path)
    exposure = 1.0 - min(abs(stats["brightness"] - 0.5) / 0.5, 1.0)
    score = _clamp(
        0.2
        + 0.35 * exposure
        + 0.25 * _clamp(stats["contrast"] * 2.2)
        + 0.20 * stats["sharpness"],
        0.05,
        0.98,
    )
    if score >= 0.75:
        label = "Good"
    elif score >= 0.45:
        label = "Moderate"
    else:
        label = "Bad"
    reasoning = (
        f"Heuristic quality estimate based on brightness={stats['brightness']:.2f}, "
        f"contrast={stats['contrast']:.2f}, sharpness={stats['sharpness']:.2f}."
    )
    return _with_meta(
        {
            "quality_score": score,
            "quality_label": label,
            "reasoning": reasoning,
        },
        error=error,
    )


def cfp_disease_prediction(
    image_or_path: ImageSource,
    error: Optional[Exception] = None,
) -> Dict[str, object]:
    stats = _image_stats(image_or_path)
    scores = _normalise_scores(
        {
            "diabetic_retinopathy": 0.25 + stats["red"] * 0.9 + stats["contrast"] * 0.4,
            "amd": 0.22 + stats["blue"] * 0.6 + stats["sharpness"] * 0.3,
            "glaucoma": 0.18 + stats["green"] * 0.5 + (1.0 - stats["contrast"]) * 0.2,
            "normal": 0.3 + (1.0 - stats["sharpness"]) * 0.1 + (1.0 - stats["contrast"]) * 0.1,
        }
    )
    labels = [name for name, prob in scores.items() if name != "normal" and prob >= 0.24]
    if not labels:
        labels = ["normal"]
    top_label = max(scores, key=scores.get)
    return _with_meta(
        {
            "labels": labels,
            "top_label": top_label,
            "probabilities": scores,
        },
        error=error,
    )


def cfp_ffa_multimodal_prediction(
    cfp_image: ImageSource,
    ffa_image: ImageSource,
    error: Optional[Exception] = None,
) -> Dict[str, object]:
    cfp_scores = cfp_disease_prediction(cfp_image)["probabilities"]
    ffa_stats = _image_stats(ffa_image)
    scores = _normalise_scores(
        {
            "diabetic_retinopathy": cfp_scores["diabetic_retinopathy"] + ffa_stats["contrast"] * 0.2,
            "retinal_vascular_leakage": 0.2 + ffa_stats["brightness"] * 0.4 + ffa_stats["sharpness"] * 0.3,
            "choroidal_neovascularisation": 0.2 + cfp_scores["amd"] * 0.7 + ffa_stats["blue"] * 0.2,
            "normal": 0.2 + cfp_scores["normal"] * 0.8,
        }
    )
    labels = [name for name, prob in scores.items() if name != "normal" and prob >= 0.24]
    if not labels:
        labels = ["normal"]
    return _with_meta(
        {
            "labels": labels,
            "probabilities": scores,
        },
        error=error,
    )


def uwf_quality_disease_prediction(
    image_or_path: ImageSource,
    error: Optional[Exception] = None,
) -> Dict[str, object]:
    quality = quality_assessment(image_or_path)
    stats = _image_stats(image_or_path)
    scores = _normalise_scores(
        {
            "peripheral_degeneration": 0.2 + stats["contrast"] * 0.5 + stats["sharpness"] * 0.2,
            "diabetic_retinopathy": 0.25 + stats["red"] * 0.7,
            "retinal_break": 0.15 + (1.0 - stats["brightness"]) * 0.3 + stats["sharpness"] * 0.2,
            "normal": 0.3 + (1.0 - stats["contrast"]) * 0.2,
        }
    )
    labels = [name for name, prob in scores.items() if name != "normal" and prob >= 0.24]
    if not labels:
        labels = ["normal"]
    return _with_meta(
        {
            "quality_score": quality["quality_score"],
            "quality_label": quality["quality_label"],
            "disease_labels": labels,
            "probabilities": scores,
        },
        error=error,
    )


def glaucoma_prediction(
    image_or_path: ImageSource,
    error: Optional[Exception] = None,
) -> Dict[str, object]:
    stats = _image_stats(image_or_path)
    probability = _clamp(0.25 + stats["green"] * 0.45 + (1.0 - stats["contrast"]) * 0.25, 0.05, 0.97)
    cup_disc_ratio = _clamp(0.35 + probability * 0.35, 0.2, 0.9)
    return _with_meta(
        {
            "referable_glaucoma": probability >= 0.55,
            "probability": probability,
            "cup_disc_ratio": cup_disc_ratio,
            "structural_features": {
                "rim_thinning": probability >= 0.6,
                "disc_pallor": stats["brightness"] < 0.42,
            },
        },
        error=error,
    )


def pdr_prediction(
    image_or_path: ImageSource,
    error: Optional[Exception] = None,
) -> Dict[str, object]:
    disease = cfp_disease_prediction(image_or_path)
    dr_prob = float(disease["probabilities"]["diabetic_retinopathy"])
    if dr_prob >= 0.42:
        grade = "active"
    elif dr_prob >= 0.28:
        grade = "quiescent"
    else:
        grade = "unlikely"
    return _with_meta(
        {
            "grade": grade,
            "grade_index": {"unlikely": 0, "quiescent": 1, "active": 2}[grade],
            "probability": dr_prob,
            "findings": list(disease["labels"]),
        },
        error=error,
    )


def ffa_lesion_detection(
    image_or_path: ImageSource,
    confidence_threshold: float = 0.5,
    error: Optional[Exception] = None,
) -> Dict[str, object]:
    stats = _image_stats(image_or_path)
    width = int(stats["width"])
    height = int(stats["height"])
    detections = []
    lesion_score = _clamp(0.25 + stats["contrast"] * 0.8 + stats["sharpness"] * 0.2)
    if lesion_score >= confidence_threshold:
        detections.append(
            {
                "label": "hyperfluorescent_focus",
                "bbox": [int(width * 0.3), int(height * 0.35), int(width * 0.18), int(height * 0.18)],
                "confidence": lesion_score,
            }
        )
    if stats["brightness"] >= 0.45 and lesion_score >= max(0.35, confidence_threshold - 0.1):
        detections.append(
            {
                "label": "vascular_leakage",
                "bbox": [int(width * 0.55), int(height * 0.48), int(width * 0.16), int(height * 0.16)],
                "confidence": _clamp(lesion_score - 0.05),
            }
        )
    return _with_meta(
        {
            "detections": detections,
            "lesion_types": [d["label"] for d in detections],
        },
        error=error,
    )


def disc_fovea_localisation(
    image_or_path: ImageSource,
    error: Optional[Exception] = None,
) -> Dict[str, object]:
    stats = _image_stats(image_or_path)
    width = int(stats["width"])
    height = int(stats["height"])
    disc_center = [int(width * 0.32), int(height * 0.46)]
    fovea_center = [int(width * 0.67), int(height * 0.55)]
    return _with_meta(
        {
            "disc_center": disc_center,
            "disc_radius": float(min(width, height) * 0.08),
            "fovea_center": fovea_center,
        },
        error=error,
    )


def vqa_response(
    image_or_path: ImageSource,
    question: str,
    error: Optional[Exception] = None,
) -> Dict[str, object]:
    quality = quality_assessment(image_or_path)
    disease = cfp_disease_prediction(image_or_path)
    labels = ", ".join(disease["labels"])
    question_l = question.lower()
    if "quality" in question_l:
        answer = (
            f"The image quality appears {quality['quality_label'].lower()} "
            f"(score {quality['quality_score']:.2f})."
        )
    elif "glaucoma" in question_l:
        glaucoma = glaucoma_prediction(image_or_path)
        answer = (
            f"Referable glaucoma is {'suggested' if glaucoma['referable_glaucoma'] else 'not strongly suggested'} "
            f"with estimated probability {glaucoma['probability']:.2f}."
        )
    else:
        answer = (
            f"Heuristic review suggests {labels} with image quality "
            f"{quality['quality_label'].lower()}."
        )
    confidence = _clamp(max(disease["probabilities"].values()), 0.3, 0.9)
    return _with_meta(
        {
            "answer": answer,
            "confidence": confidence,
        },
        error=error,
    )


def clip_zero_shot_prediction(
    image_or_path: ImageSource,
    candidate_labels: Iterable[str],
    error: Optional[Exception] = None,
) -> Dict[str, object]:
    stats = _image_stats(image_or_path)
    scores: Dict[str, float] = {}
    for label in candidate_labels:
        label_l = label.lower()
        if "dr" in label_l or "diabetic" in label_l:
            score = 0.25 + stats["red"] * 0.7 + stats["contrast"] * 0.2
        elif "amd" in label_l or "drusen" in label_l:
            score = 0.22 + stats["blue"] * 0.5 + stats["sharpness"] * 0.25
        elif "glau" in label_l:
            score = 0.2 + stats["green"] * 0.4 + (1.0 - stats["contrast"]) * 0.2
        elif "normal" in label_l:
            score = 0.2 + (1.0 - stats["contrast"]) * 0.3
        else:
            score = 0.18 + stats["contrast"] * 0.2 + stats["sharpness"] * 0.2
        scores[label] = score
    probs = _normalise_scores(scores)
    top_label = max(probs, key=probs.get)
    return _with_meta(
        {
            "label": top_label,
            "probabilities": probs,
        },
        error=error,
    )


def fmue_prediction(
    image_or_path: ImageSource,
    error: Optional[Exception] = None,
) -> Dict[str, object]:
    stats = _image_stats(image_or_path)
    probs = _normalise_scores(
        {
            "Normal": 0.25 + (1.0 - stats["contrast"]) * 0.2,
            "AMD": 0.2 + stats["blue"] * 0.4,
            "DME": 0.22 + stats["red"] * 0.45,
            "DRUSEN": 0.18 + stats["sharpness"] * 0.3,
            "CNV": 0.18 + stats["contrast"] * 0.3 + stats["red"] * 0.2,
        }
    )
    label = max(probs, key=probs.get)
    return _with_meta({"label": label, "probabilities": probs}, error=error)


def uwf_mdd_prediction(
    image_or_path: ImageSource,
    error: Optional[Exception] = None,
) -> Dict[str, object]:
    stats = _image_stats(image_or_path)
    probs = _normalise_scores(
        {
            "normal": 0.25 + (1.0 - stats["contrast"]) * 0.25,
            "diabetic_retinopathy": 0.25 + stats["red"] * 0.6,
            "retinal_detachment": 0.16 + (1.0 - stats["brightness"]) * 0.4,
            "uveitis": 0.15 + stats["blue"] * 0.3 + stats["contrast"] * 0.2,
        }
    )
    labels = [name for name, prob in probs.items() if name != "normal" and prob >= 0.24]
    if not labels:
        labels = ["normal"]
    return _with_meta({"labels": labels, "probabilities": probs}, error=error)


def uwf_multi_abnormality_prediction(
    image_or_path: ImageSource,
    error: Optional[Exception] = None,
) -> Dict[str, object]:
    stats = _image_stats(image_or_path)
    probs = {
        "hemorrhage": _clamp(0.2 + stats["red"] * 0.6),
        "exudate": _clamp(0.15 + stats["blue"] * 0.35 + stats["sharpness"] * 0.2),
        "neovascularisation": _clamp(0.15 + stats["contrast"] * 0.4 + stats["green"] * 0.15),
        "scar": _clamp(0.1 + (1.0 - stats["brightness"]) * 0.25),
    }
    labels = [name for name, prob in probs.items() if prob >= 0.35]
    if not labels:
        labels = ["no_significant_abnormality"]
    return _with_meta({"labels": labels, "probabilities": probs}, error=error)


def _save_mask(mask: np.ndarray, image_or_path: ImageSource, suffix: str) -> str:
    stem = "image"
    if isinstance(image_or_path, (str, Path)):
        stem = Path(image_or_path).stem
    out_dir = Path(tempfile.gettempdir()) / "ophagent_fallbacks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_{suffix}.png"
    Image.fromarray(mask).save(out_path)
    return str(out_path)


def segmentation_prediction(
    image_or_path: ImageSource,
    error: Optional[Exception] = None,
) -> Dict[str, object]:
    img = _to_pil(image_or_path)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    threshold = float(np.percentile(gray, 58))
    mask = (gray >= threshold).astype(np.uint8) * 255
    mask_path = _save_mask(mask, image_or_path, "mask")
    with io.BytesIO() as buf:
        Image.fromarray(mask).save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return _with_meta(
        {
            "mask_path": mask_path,
            "mask_array_b64": mask_b64,
            "area_fraction": float(mask.mean() / 255.0),
        },
        error=error,
    )


def automorph_prediction(
    image_or_path: ImageSource,
    error: Optional[Exception] = None,
) -> Dict[str, object]:
    seg = segmentation_prediction(image_or_path)
    area_fraction = float(seg["area_fraction"])
    metrics = {
        "fractal_dim": round(1.1 + area_fraction * 0.5, 3),
        "vessel_density": round(area_fraction, 3),
        "av_ratio": round(0.6 + area_fraction * 0.25, 3),
        "cup_disc_ratio": round(0.35 + area_fraction * 0.2, 3),
    }
    return _with_meta(
        {
            "vessel_mask_path": seg["mask_path"],
            "metrics": metrics,
        },
        error=error,
    )


def gradcam_prediction(
    image_or_path: ImageSource,
    error: Optional[Exception] = None,
) -> Dict[str, object]:
    img = _to_pil(image_or_path)
    arr = np.asarray(img, dtype=np.uint8)
    gray = arr.mean(axis=2).astype(np.uint8)
    heatmap = np.clip(gray * 1.2, 0, 255).astype(np.uint8)
    overlay = arr.copy()
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], heatmap)

    def _encode(array: np.ndarray) -> str:
        with io.BytesIO() as buf:
            Image.fromarray(array).save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    return _with_meta(
        {
            "heatmap_b64": _encode(heatmap),
            "overlay_b64": _encode(overlay),
            "processed_heatmap_b64": _encode(heatmap),
            "overlay_processed_b64": _encode(overlay),
        },
        error=error,
    )


def ocr_prediction(
    image_or_path: ImageSource,
    error: Optional[Exception] = None,
) -> Dict[str, object]:
    return _with_meta(
        {
            "text": "",
            "blocks": [],
        },
        error=error,
    )


def empty_search_results(
    query: str,
    error: Optional[Exception] = None,
) -> Dict[str, object]:
    return _with_meta(
        {
            "results": [],
            "query": query,
            "count": 0,
        },
        backend="offline-fallback",
        error=error,
    )
