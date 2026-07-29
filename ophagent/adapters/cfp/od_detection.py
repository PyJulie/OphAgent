"""
Adapter: Optic Disc + Macula joint detector (YOLOv8m).

Wraps an external OD/fovea detector. Configure the project with
``OPHAGENT_OD_SRC`` or the checkpoint with ``OPHAGENT_OD_WEIGHTS``.
Outputs bounding boxes for OD and macula (fovea) regions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import AdapterBase, ToolMetadata, AdapterResult, register
from ...utils.paths import checkpoint_file, external_dir


OD_SRC = external_dir("OPHAGENT_OD_SRC", "od_detection")
DEFAULT_WEIGHTS = checkpoint_file("OPHAGENT_OD_WEIGHTS", "cfp", "od_fovea.pt")


@register
class ODDetectionAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="cfp_od_detection",
        modality="CFP",
        task="detection",
        description=(
            "Joint optic-disc (OD) and macula/fovea bounding-box detection on a "
            "colour fundus photograph. Returns (x, y, w, h) for each detected "
            "structure. Useful as a preprocessor for any task that needs an "
            "OD-centred or fovea-centred crop (glaucoma grading, macula lesions)."
        ),
        input_size=(1024, 1024),
        labels=["OD", "Fovea"],   # YOLO class names — verified at load
        confidence_threshold=0.25,
        limitations=[
            "Designed for ~3 MP fundus images; small or poorly centered crops "
            "may yield false negatives",
        ],
        cost_class="fast",
        source_dir=str(OD_SRC),
    )

    def _load_impl(self) -> None:
        from ultralytics import YOLO
        weights = DEFAULT_WEIGHTS
        if not weights.exists():
            # fallback to other available best.pt
            candidates = sorted((OD_SRC / "runs").rglob("weights/best.pt"))
            if not candidates:
                raise FileNotFoundError(
                    "OD/fovea detector weights not found. Set OPHAGENT_OD_WEIGHTS "
                    f"or OPHAGENT_OD_SRC. Expected: {DEFAULT_WEIGHTS}"
                )
            weights = candidates[-1]
        self._impl = YOLO(str(weights))
        self._labels = list(self._impl.names.values())

    # Confidence below which a direct fovea detection is considered unreliable
    # and we fall back to deriving it from the joint bbox + OD position.
    FOVEA_CONF_RELIABLE = 0.40

    def _predict_impl(self, image_path: str, conf: float = 0.25, **_) -> AdapterResult:
        results = self._impl(image_path, conf=conf, verbose=False)
        if not results:
            return AdapterResult(
                success=True, tool=self.metadata.name, modality="CFP",
                task="detection", predictions={"detections": []}, confidence=0.0,
            )
        r0 = results[0]
        boxes = r0.boxes  # (N, 4) xyxy + cls + conf
        detections: list[dict[str, Any]] = []
        max_conf = 0.0
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            cls_idx = int(boxes.cls[i].cpu().item())
            cls_conf = float(boxes.conf[i].cpu().item())
            label = self._labels[cls_idx] if cls_idx < len(self._labels) else str(cls_idx)
            x1, y1, x2, y2 = xyxy
            detections.append({
                "label": label,
                "confidence": cls_conf,
                "xyxy": [x1, y1, x2, y2],
                "xywh": [x1, y1, x2 - x1, y2 - y1],
                "center": [(x1 + x2) / 2, (y1 + y2) / 2],
            })
            max_conf = max(max_conf, cls_conf)

        # Convenience: pick the top detection per *canonical* class
        # (label case in the YOLO weights file is unpredictable: "OD" / "od",
        # "Fovea" / "fovea", "joint" / "Joint").
        def _canon(label: str) -> str:
            s = label.lower()
            if s in {"od", "disc", "optic_disc", "opticdisc"}: return "OD"
            if s in {"fovea", "macula"}:                       return "Fovea"
            if s == "joint":                                   return "joint"
            return label   # unknown — keep verbatim

        per_class: dict[str, dict] = {}
        for d in detections:
            canon = _canon(d["label"])
            d["canonical_label"] = canon
            if canon not in per_class or d["confidence"] > per_class[canon]["confidence"]:
                per_class[canon] = d

        # ── Fovea fallback: when direct fovea detection is missing or
        # low-confidence, derive its position from the joint bbox.
        # The joint box encloses BOTH the OD and the fovea, so the fovea
        # centre is the mirror of the OD centre about the joint centre:
        #     fovea_xy ≈ 2 * joint_centre - OD_centre
        # This works regardless of laterality (the geometry handles OD/OS).
        fovea_inferred = False
        fovea_det = per_class.get("Fovea")
        if (fovea_det is None or fovea_det.get("confidence", 0) < self.FOVEA_CONF_RELIABLE) \
                and "OD" in per_class and "joint" in per_class:
            od = per_class["OD"]
            jt = per_class["joint"]
            od_cx, od_cy = od["center"]
            jt_cx, jt_cy = jt["center"]
            fcx = 2 * jt_cx - od_cx
            fcy = 2 * jt_cy - od_cy
            # estimate side from OD bbox (fovea region usually ~OD size)
            od_w = od["xywh"][2]
            od_h = od["xywh"][3]
            half_w, half_h = od_w / 2, od_h / 2
            inferred = {
                "label": "Fovea",
                "canonical_label": "Fovea",
                "confidence": (fovea_det["confidence"] if fovea_det else 0.0),
                "xyxy": [fcx - half_w, fcy - half_h, fcx + half_w, fcy + half_h],
                "xywh": [fcx - half_w, fcy - half_h, od_w, od_h],
                "center": [fcx, fcy],
                "inferred_from": "joint - OD",
                "direct_detection_confidence": (fovea_det["confidence"] if fovea_det else None),
            }
            # Replace per_class[Fovea] with the inferred one (keep direct conf
            # for transparency)
            per_class["Fovea"] = inferred
            fovea_inferred = True

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="CFP",
            task="detection",
            predictions={
                "detections": detections,
                "best": per_class,
                "has_od": "OD" in per_class,
                "has_fovea": "Fovea" in per_class,
                "fovea_inferred": fovea_inferred,
            },
            confidence=max_conf,
            raw_output={"n_detections": len(detections)},
        )

    # Public helper for other adapters (e.g. glaucoma needs the OD crop)
    def crop_disc(self, image_path: str, padding_ratio: float = 1.25) -> "tuple[Any, dict] | None":
        """Return a PIL.Image cropped to the OD region with `padding_ratio` × side margin, or None."""
        result = self.predict(image_path)
        if not result.success or "OD" not in result.predictions.get("best", {}):
            return None
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        od = result.predictions["best"]["OD"]
        x1, y1, x2, y2 = od["xyxy"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1) * padding_ratio
        left = max(0, int(cx - side / 2))
        top = max(0, int(cy - side / 2))
        right = min(W, int(cx + side / 2))
        bottom = min(H, int(cy + side / 2))
        return img.crop((left, top, right, bottom)), {
            "crop_box": [left, top, right, bottom],
            "od_box": od["xyxy"],
            "od_confidence": od["confidence"],
        }
