"""
Adapter: FFA lesion detection (YOLO11x, 47 raw classes → 7 clinical groups).

Wraps an external FFA lesion detector. Configure the local project with
``OPHAGENT_FFA_PROJECT`` or the checkpoint with
``OPHAGENT_FFA_DETECTOR_WEIGHTS``.

Outputs:
  - Per-lesion bounding boxes (47 raw codes from FFAIR / Tongren lesion taxonomy)
  - Aggregated detection counts per merged clinical category (DR / RVO / AMD /
    Macular Disorders / Uveitis / etc.) using merge_map_ids.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..base import AdapterBase, ToolMetadata, AdapterResult, register
from ...utils.paths import checkpoint_file, external_dir


FFA_PROJECT = external_dir("OPHAGENT_FFA_PROJECT", "ffa")
FFA_WEIGHTS = checkpoint_file("OPHAGENT_FFA_DETECTOR_WEIGHTS", "ffa", "lesion_yolo11x.pt")
MERGE_CSV = checkpoint_file("OPHAGENT_FFA_MERGE_MAP", "ffa", "merge_map_ids.csv")


def _load_merge_map() -> dict[str, str]:
    """label_id (e.g. '3.1') → merged_label ('DR')."""
    if not MERGE_CSV.exists():
        return {}
    out: dict[str, str] = {}
    with open(MERGE_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[str(row["label_id"]).strip()] = row["merged_label"].strip()
    return out


@register
class FFALesionDetectionAdapter(AdapterBase):
    metadata = ToolMetadata(
        name="ffa_lesion_detection",
        modality="FFA",
        task="detection",
        description=(
            "Lesion detection on a fluorescein angiography (FFA) image "
            "(YOLO11x, 47-class FFAIR taxonomy). Returns per-lesion bounding "
            "boxes plus aggregated counts per merged clinical group "
            "(DR / RVO / AMD / Uveitis / Macular Disorders / etc.), AND draws "
            "the boxes on the image — `figures.lesion_overlay` is an annotated "
            "PNG the chat can display. THIS is the tool to call when the user "
            "asks to mark / show / 标出 / 圈出 lesions on an FFA image; do NOT "
            "hand-draw boxes in the `compute` sandbox. Also use it for leakage "
            "patterns, vascular anomalies, or specific lesion locations."
        ),
        input_size=(640, 640),
        labels=[],   # populated at load
        confidence_threshold=0.25,
        limitations=[
            "Trained on FFAIR + Tongren cohort — out-of-distribution FFA "
            "(different angiograph, time point) may underperform",
            "47 raw class IDs are clinical sub-codes; the merged 7-group view "
            "is more interpretable for routine reporting",
        ],
        cost_class="fast",
        source_dir=str(FFA_PROJECT),
    )

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._merge_map = _load_merge_map()

    def _load_impl(self) -> None:
        from ultralytics import YOLO
        if not FFA_WEIGHTS.exists():
            raise FileNotFoundError(
                "FFA detector weights not found. Set OPHAGENT_FFA_DETECTOR_WEIGHTS "
                f"or OPHAGENT_FFA_PROJECT. Expected: {FFA_WEIGHTS}"
            )
        self._impl = YOLO(str(FFA_WEIGHTS))
        # YOLO model.names is {idx: 'class_3.1', ...}
        self._raw_names = self._impl.names

    def _predict_impl(self, image_path: str, conf: float = 0.25,
                      output_dir: str | None = None, **_) -> AdapterResult:
        results = self._impl(image_path, conf=conf, verbose=False)
        if not results:
            return AdapterResult(
                success=True, tool=self.metadata.name, modality="FFA",
                task="detection", predictions={"detections": [], "merged_counts": {}},
                confidence=0.0,
            )
        r0 = results[0]
        boxes = r0.boxes

        detections: list[dict[str, Any]] = []
        merged_counts: dict[str, int] = {}
        merged_confidence_max: dict[str, float] = {}
        max_conf_overall = 0.0

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            cls_idx = int(boxes.cls[i].cpu().item())
            c = float(boxes.conf[i].cpu().item())
            raw_name = self._raw_names.get(cls_idx, str(cls_idx))
            # Strip "class_" prefix to get the raw label id (e.g. "3.1")
            raw_id = raw_name.replace("class_", "")
            merged = self._merge_map.get(raw_id, "Other")
            detections.append({
                "raw_label": raw_name, "raw_id": raw_id,
                "merged_group": merged,
                "confidence": c, "xyxy": xyxy,
                "xywh": [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]],
            })
            merged_counts[merged] = merged_counts.get(merged, 0) + 1
            merged_confidence_max[merged] = max(merged_confidence_max.get(merged, 0), c)
            max_conf_overall = max(max_conf_overall, c)

        # Draw the detected boxes onto the image so the chat UI can SHOW the
        # lesions, not just list coordinates. Failure here must never break the
        # tool — the numeric detections are the source of truth.
        figures: dict[str, str] = {}
        if detections and output_dir:
            try:
                ov = self._render_overlay(image_path, detections, output_dir)
                if ov:
                    figures["lesion_overlay"] = ov
            except Exception as _e:
                figures["_overlay_error"] = f"{type(_e).__name__}: {_e}"

        return AdapterResult(
            success=True,
            tool=self.metadata.name,
            modality="FFA",
            task="detection",
            predictions={
                "n_detections": len(detections),
                "detections": detections,
                "merged_counts": merged_counts,
                "merged_confidence_max": merged_confidence_max,
                "top_merged_group": (
                    max(merged_counts.items(), key=lambda kv: kv[1])[0]
                    if merged_counts else None
                ),
            },
            confidence=max_conf_overall,
            figures={k: v for k, v in figures.items() if not k.startswith("_")},
            metadata={"weights": str(FFA_WEIGHTS), "n_raw_classes": len(self._raw_names)},
        )

    @staticmethod
    def _render_overlay(image_path: str, detections: list[dict],
                        output_dir: str) -> str | None:
        """Draw each detection's box + merged-group label on the image and save
        a PNG. Unicode-safe read/write (cv2.imread/imwrite break on CJK paths).
        Returns the saved path, or None on failure."""
        import cv2
        import numpy as np
        arr = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)   # BGR, HxWx3
        if img is None:
            return None
        H, W = img.shape[:2]
        thick = max(2, int(round(min(H, W) * 0.004)))
        # Stable colour per merged group (BGR).
        palette = [(0, 0, 255), (0, 200, 255), (0, 220, 0), (255, 128, 0),
                   (255, 0, 200), (200, 0, 255), (255, 255, 0)]
        group_color: dict[str, tuple] = {}
        for d in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
            x1, x2 = max(0, min(W - 1, x1)), max(0, min(W - 1, x2))
            y1, y2 = max(0, min(H - 1, y1)), max(0, min(H - 1, y2))
            grp = d.get("merged_group") or d.get("raw_label") or "?"
            col = group_color.setdefault(grp, palette[len(group_color) % len(palette)])
            cv2.rectangle(img, (x1, y1), (x2, y2), col, thick)
            label = f"{grp} {d.get('confidence', 0):.2f}"
            fs = max(0.4, min(H, W) / 1400.0)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            ly = max(th + 4, y1)
            cv2.rectangle(img, (x1, ly - th - 4), (x1 + tw + 4, ly), col, -1)
            cv2.putText(img, label, (x1 + 2, ly - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (255, 255, 255), 1, cv2.LINE_AA)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dst = out / f"{Path(image_path).stem}_lesions.png"
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            return None
        buf.tofile(str(dst))
        return str(dst)
