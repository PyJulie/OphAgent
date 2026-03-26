"""
FastAPI inference service factory for OphAgent model microservices.

Each service is selected by the ``MODEL_ID`` environment variable. When an
actual trained model is not available, the service falls back to lightweight
heuristics so the rest of the agent pipeline can still execute.
"""
from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from ophagent.utils.fallback_inference import (
    cfp_disease_prediction,
    cfp_ffa_multimodal_prediction,
    disc_fovea_localisation,
    ffa_lesion_detection,
    glaucoma_prediction,
    pdr_prediction,
    quality_assessment,
    uwf_quality_disease_prediction,
    vqa_response,
)
from ophagent.utils.logger import get_logger

logger = get_logger("inference.service")


class ImageRequest(BaseModel):
    image_b64: str
    params: Dict[str, Any] = Field(default_factory=dict)


class DualImageRequest(BaseModel):
    cfp_b64: str
    ffa_b64: str
    params: Dict[str, Any] = Field(default_factory=dict)


class InferenceResponse(BaseModel):
    success: bool
    result: Dict[str, Any]
    latency_ms: float
    model_id: str


@dataclass
class ServiceSpec:
    model_id: str
    loader: Callable[[], Any]
    inference: Callable[[Any, Dict[str, Any]], Dict[str, Any]]
    dual_image: bool = False


def _runtime_mode() -> str:
    try:
        from config.settings import get_settings

        return get_settings().runtime.mode.lower()
    except Exception:
        return os.environ.get("OPHAGENT_RUNTIME__MODE", "graceful").lower()


def _strict_runtime() -> bool:
    return _runtime_mode() == "strict"


def _is_heuristic_backend(bundle: Any) -> bool:
    return isinstance(bundle, dict) and bundle.get("backend") == "heuristic-fallback"


def _assert_real_backend(bundle: Any, model_id: str) -> None:
    if _strict_runtime() and _is_heuristic_backend(bundle):
        raise RuntimeError(
            f"Service {model_id} is running with heuristic fallback, which is not allowed in strict mode."
        )


def create_app(
    model_id: str,
    model_loader: Callable[[], Any],
    inference_fn: Callable[[Any, Dict[str, Any]], Dict[str, Any]],
    dual_image: bool = False,
) -> FastAPI:
    app = FastAPI(title=f"OphAgent:{model_id}", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.model = None
    app.state.model_id = model_id

    @app.on_event("startup")
    def _load_model():
        logger.info(f"Loading service model: {model_id}")
        app.state.model = model_loader()
        _assert_real_backend(app.state.model, model_id)
        logger.info(f"Service model ready: {model_id}")

    @app.get("/health")
    def health():
        degraded = _is_heuristic_backend(app.state.model)
        return {
            "status": "degraded" if degraded else "ok",
            "model_id": model_id,
            "backend": getattr(app.state.model, "get", lambda *_: "unknown")("backend"),
            "strict_mode": _strict_runtime(),
        }

    def _decode_image(b64_str: str) -> Image.Image:
        data = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(data)).convert("RGB")

    if dual_image:
        @app.post("/run", response_model=InferenceResponse)
        def run_dual(req: DualImageRequest):
            t0 = time.monotonic()
            try:
                inputs = {
                    "cfp_image": _decode_image(req.cfp_b64),
                    "ffa_image": _decode_image(req.ffa_b64),
                    **req.params,
                }
                result = inference_fn(app.state.model, inputs)
            except Exception as exc:
                logger.error(f"Inference error in {model_id}: {exc}")
                raise HTTPException(status_code=500, detail=str(exc))
            return InferenceResponse(
                success=True,
                result=result,
                latency_ms=(time.monotonic() - t0) * 1000,
                model_id=model_id,
            )
    else:
        @app.post("/run", response_model=InferenceResponse)
        def run_single(req: ImageRequest):
            t0 = time.monotonic()
            try:
                inputs = {"image": _decode_image(req.image_b64), **req.params}
                result = inference_fn(app.state.model, inputs)
            except Exception as exc:
                logger.error(f"Inference error in {model_id}: {exc}")
                raise HTTPException(status_code=500, detail=str(exc))
            return InferenceResponse(
                success=True,
                result=result,
                latency_ms=(time.monotonic() - t0) * 1000,
                model_id=model_id,
            )

    @app.get("/info")
    def info():
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        except Exception:
            device = "cpu"
            gpu = "N/A"
        backend = getattr(app.state.model, "get", lambda *_: "unknown")("backend")
        return {
            "model_id": model_id,
            "backend": backend,
            "device": device,
            "gpu": gpu,
        }

    return app


def _heuristic_loader(model_id: str) -> Dict[str, Any]:
    logger.warning(f"Using heuristic fallback backend for service {model_id}.")
    return {"model_id": model_id, "backend": "heuristic-fallback"}


def _load_cfp_quality() -> Dict[str, Any]:
    model_id = os.environ.get("MODEL_ID", "cfp_quality")
    weight_path = os.environ.get("MODEL_WEIGHT", "models_weights/cfp_quality/best.pth")
    if not os.path.exists(weight_path):
        return _heuristic_loader(model_id)
    try:
        import timm
        import torch

        model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=3)
        ckpt = torch.load(weight_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        load_info = model.load_state_dict(state, strict=False)
        if _strict_runtime() and (load_info.missing_keys or load_info.unexpected_keys):
            raise RuntimeError(
                "CFP quality checkpoint is incompatible with the service model: "
                f"missing={load_info.missing_keys}, unexpected={load_info.unexpected_keys}"
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        return {"model_id": model_id, "backend": "timm", "model": model}
    except Exception as exc:
        if _strict_runtime():
            raise RuntimeError(f"Failed to load real CFP quality model in strict mode: {exc}") from exc
        logger.warning(f"Falling back to heuristic CFP quality service: {exc}")
        return _heuristic_loader(model_id)


def _infer_cfp_quality(bundle: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    if bundle.get("backend") != "timm":
        return quality_assessment(inputs["image"])

    from torchvision import transforms
    import torch.nn.functional as F

    model = bundle["model"]
    device = next(model.parameters()).device
    transform = transforms.Compose(
        [
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(inputs["image"]).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=-1).cpu()[0].tolist()

    labels = ["Bad", "Moderate", "Good"]
    top_idx = int(torch.tensor(probs).argmax())
    return {
        "quality_label": labels[top_idx],
        "quality_score": probs[top_idx],
        "probabilities": dict(zip(labels, probs)),
        "backend": "timm",
    }


def _infer_cfp_disease(bundle: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    return cfp_disease_prediction(inputs["image"])


def _infer_multimodal(bundle: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    return cfp_ffa_multimodal_prediction(inputs["cfp_image"], inputs["ffa_image"])


def _infer_uwf_quality_disease(bundle: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    return uwf_quality_disease_prediction(inputs["image"])


def _infer_glaucoma(bundle: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    return glaucoma_prediction(inputs["image"])


def _infer_pdr(bundle: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    return pdr_prediction(inputs["image"])


def _infer_ffa_lesion(bundle: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    return ffa_lesion_detection(
        inputs["image"],
        confidence_threshold=float(inputs.get("confidence_threshold", 0.5)),
    )


def _infer_disc_fovea(bundle: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    return disc_fovea_localisation(inputs["image"])


def _infer_vqa(bundle: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    return vqa_response(inputs["image"], str(inputs.get("question", "Describe the findings.")))


def _service_spec(model_id: str) -> ServiceSpec:
    if model_id == "cfp_quality":
        return ServiceSpec(model_id=model_id, loader=_load_cfp_quality, inference=_infer_cfp_quality)
    if model_id == "cfp_disease":
        return ServiceSpec(model_id=model_id, loader=lambda: _heuristic_loader(model_id), inference=_infer_cfp_disease)
    if model_id == "cfp_ffa_multimodal":
        return ServiceSpec(
            model_id=model_id,
            loader=lambda: _heuristic_loader(model_id),
            inference=_infer_multimodal,
            dual_image=True,
        )
    if model_id == "uwf_quality_disease":
        return ServiceSpec(model_id=model_id, loader=lambda: _heuristic_loader(model_id), inference=_infer_uwf_quality_disease)
    if model_id == "cfp_glaucoma":
        return ServiceSpec(model_id=model_id, loader=lambda: _heuristic_loader(model_id), inference=_infer_glaucoma)
    if model_id == "cfp_pdr":
        return ServiceSpec(model_id=model_id, loader=lambda: _heuristic_loader(model_id), inference=_infer_pdr)
    if model_id == "ffa_lesion":
        return ServiceSpec(model_id=model_id, loader=lambda: _heuristic_loader(model_id), inference=_infer_ffa_lesion)
    if model_id == "disc_fovea":
        return ServiceSpec(model_id=model_id, loader=lambda: _heuristic_loader(model_id), inference=_infer_disc_fovea)
    if model_id in {"fundus_expert", "vision_unite"}:
        return ServiceSpec(model_id=model_id, loader=lambda: _heuristic_loader(model_id), inference=_infer_vqa)
    raise ValueError(f"Unsupported MODEL_ID for service factory: {model_id}")


def _build_default_app() -> FastAPI:
    model_id = os.environ.get("MODEL_ID", "cfp_quality")
    spec = _service_spec(model_id)
    return create_app(
        model_id=spec.model_id,
        model_loader=spec.loader,
        inference_fn=spec.inference,
        dual_image=spec.dual_image,
    )


app = _build_default_app()
