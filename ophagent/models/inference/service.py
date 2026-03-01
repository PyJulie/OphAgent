"""
FastAPI Inference Service Template for OphAgent newly-developed models.

Each newly-developed model (cfp_quality, cfp_disease, etc.) runs as an
independent FastAPI microservice inside a Docker container.  This module
provides the base application factory used by all services.

Usage::
    # In each service's main.py:
    from ophagent.models.inference.service import create_app
    app = create_app(model_loader=..., inference_fn=...)

    # Or run directly:
    uvicorn ophagent.models.inference.service:app --host 0.0.0.0 --port 8110
"""
from __future__ import annotations

import base64
import io
import os
import time
from typing import Any, Callable, Dict, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from ophagent.utils.logger import get_logger

logger = get_logger("inference.service")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ImageRequest(BaseModel):
    image_b64: str
    params: Dict[str, Any] = {}


class DualImageRequest(BaseModel):
    cfp_b64: str
    ffa_b64: str
    params: Dict[str, Any] = {}


class InferenceResponse(BaseModel):
    success: bool
    result: Dict[str, Any]
    latency_ms: float
    model_id: str


# ---------------------------------------------------------------------------
# Base FastAPI app factory
# ---------------------------------------------------------------------------

def create_app(
    model_id: str,
    model_loader: Callable[[], Any],
    inference_fn: Callable[[Any, Dict[str, Any]], Dict[str, Any]],
    dual_image: bool = False,
) -> FastAPI:
    """
    Factory to create a FastAPI application for a single model.

    Args:
        model_id:      Identifier string for the model (used in responses).
        model_loader:  Callable that returns the loaded model.
        inference_fn:  Callable(model, inputs_dict) -> result_dict.
        dual_image:    If True, the /run endpoint accepts DualImageRequest.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title=f"OphAgent:{model_id}", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # State
    app.state.model = None
    app.state.model_id = model_id

    @app.on_event("startup")
    def _load_model():
        logger.info(f"Loading model: {model_id}")
        app.state.model = model_loader()
        logger.info(f"Model {model_id} ready.")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_id": model_id}

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
            except Exception as e:
                logger.error(f"Inference error in {model_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
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
            except Exception as e:
                logger.error(f"Inference error in {model_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            return InferenceResponse(
                success=True,
                result=result,
                latency_ms=(time.monotonic() - t0) * 1000,
                model_id=model_id,
            )

    @app.get("/info")
    def info():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return {
            "model_id": model_id,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        }

    return app


# ---------------------------------------------------------------------------
# Example: CFP Quality service (used as template for all services)
# ---------------------------------------------------------------------------

def _cfp_quality_loader():
    import timm
    import torch
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=3)
    weight_path = os.environ.get("MODEL_WEIGHT", "models_weights/cfp_quality/best.pth")
    if os.path.exists(weight_path):
        ckpt = torch.load(weight_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model


def _cfp_quality_inference(model, inputs: dict) -> dict:
    from torchvision import transforms
    import torch.nn.functional as F

    device = next(model.parameters()).device
    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
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
    }


# Default app for uvicorn (cfp_quality as example)
app = create_app(
    model_id=os.environ.get("MODEL_ID", "cfp_quality"),
    model_loader=_cfp_quality_loader,
    inference_fn=_cfp_quality_inference,
)
