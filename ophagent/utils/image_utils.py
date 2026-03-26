"""Image utility functions for OphAgent."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - fallback for lean environments
    cv2 = None


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image as BGR numpy array via OpenCV."""
    if cv2 is not None:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return img
    return np.array(load_image_pil(path))[:, :, ::-1]


def load_image_pil(path: Union[str, Path]) -> Image.Image:
    """Load an image as PIL Image (RGB)."""
    return Image.open(str(path)).convert("RGB")


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert PIL Image (RGB) to BGR numpy array."""
    arr = np.array(img)
    if cv2 is None:
        return arr[:, :, ::-1]
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to PIL Image (RGB)."""
    if cv2 is None:
        return Image.fromarray(img[:, :, ::-1])
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def image_to_base64(path: Union[str, Path, Image.Image]) -> str:
    """Encode an image to base64 string (PNG)."""
    if isinstance(path, (str, Path)):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    # PIL Image
    buf = io.BytesIO()
    path.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_pil(b64_str: str) -> Image.Image:
    """Decode a base64 string to PIL Image."""
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data)).convert("RGB")


def resize_image(
    img: Union[np.ndarray, Image.Image],
    size: Tuple[int, int],
    keep_aspect: bool = True,
) -> Union[np.ndarray, Image.Image]:
    """Resize image to *size* (W, H)."""
    is_pil = isinstance(img, Image.Image)
    if is_pil:
        if keep_aspect:
            img.thumbnail(size, Image.LANCZOS)
            return img
        return img.resize(size, Image.LANCZOS)
    # numpy
    h, w = size[1], size[0]
    if keep_aspect:
        orig_h, orig_w = img.shape[:2]
        scale = min(w / orig_w, h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        if cv2 is not None:
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
    if cv2 is not None:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def crop_region(
    img: Union[np.ndarray, Image.Image],
    x: int, y: int, w: int, h: int,
) -> Union[np.ndarray, Image.Image]:
    """Crop a rectangular region from *img*."""
    if isinstance(img, Image.Image):
        return img.crop((x, y, x + w, y + h))
    return img[y:y + h, x:x + w]


def normalize_fundus(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE normalisation commonly used for fundus images."""
    if cv2 is None:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def overlay_heatmap(
    img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: Optional[int] = None,
) -> np.ndarray:
    """Overlay a single-channel heatmap onto *img*."""
    if heatmap.dtype != np.uint8:
        heatmap = (heatmap * 255).clip(0, 255).astype(np.uint8)
    if cv2 is None:
        heatmap_rgb = np.stack([heatmap, np.zeros_like(heatmap), 255 - heatmap], axis=-1)
        heatmap_resized = np.array(
            Image.fromarray(heatmap_rgb).resize((img.shape[1], img.shape[0]), Image.BILINEAR)
        )
        return np.clip((1 - alpha) * img + alpha * heatmap_resized, 0, 255).astype(np.uint8)
    if colormap is None:
        colormap = cv2.COLORMAP_JET
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    heatmap_resized = cv2.resize(heatmap_colored, (img.shape[1], img.shape[0]))
    return cv2.addWeighted(img, 1 - alpha, heatmap_resized, alpha, 0)


def get_image_info(path: Union[str, Path]) -> dict:
    """Return basic metadata dict for an image file."""
    img = load_image_pil(path)
    return {
        "path": str(path),
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "size_kb": Path(path).stat().st_size / 1024,
    }
