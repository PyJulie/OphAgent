"""OCR Text Detector auxiliary tool."""
from __future__ import annotations
from typing import Any, Dict, List
from ophagent.tools.base import BaseTool, ToolMetadata


class OCRDetectorTool(BaseTool):
    """
    Extracts text from images using EasyOCR or Tesseract.
    Input:  {"image_path": str, "languages": list[str] (default ["en"])}
    Output: {"text": str, "blocks": list[{"text": str, "bbox": list, "confidence": float}]}
    """
    def __init__(self, metadata: ToolMetadata):
        super().__init__(metadata)
        self._reader = None

    def load_model(self) -> None:
        import easyocr
        from config.settings import get_settings
        backend = get_settings().tools.ocr_backend
        try:
            if backend == "easyocr":
                self._reader = easyocr.Reader(["en"], gpu=True)
            self._model_loaded = True
        except Exception as exc:
            self._reader = None
            self._model_loaded = False
            raise RuntimeError(f"OCR model failed to load: {exc}") from exc

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._reader is None:
            raise RuntimeError(
                "OCR reader is not initialised; model loading may have failed."
            )
        langs = inputs.get("languages", ["en"])
        results = self._reader.readtext(inputs["image_path"])
        blocks = [
            {"text": r[1], "bbox": r[0], "confidence": float(r[2])}
            for r in results
        ]
        full_text = " ".join(b["text"] for b in blocks)
        return {"text": full_text, "blocks": blocks}

    def fallback_run(self, inputs: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        from ophagent.utils.fallback_inference import ocr_prediction
        return ocr_prediction(inputs["image_path"], error=error)
