"""
OCT volume loaders.

Supports:
  - Heidelberg Spectralis DICOM (.dcm) — OPT modality, frames = B-scans
  - generic multi-frame DICOM
  - .nii / .nii.gz (NIfTI) — converted to (N, H, W)
  - .npy / .npz — raw arrays
  - folder of 2D images (each file is a B-scan, sorted by filename)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class OCTVolume:
    """A B-scan stack with metadata."""
    volume: np.ndarray   # shape (N_slices, H, W), uint8
    spacing: tuple[float, float, float] | None  # (z, y, x) in mm if available
    source: str          # path
    modality: str = "OPT"
    metadata: dict[str, Any] = None

    @property
    def n_slices(self) -> int:
        return self.volume.shape[0]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.volume.shape

    def slice(self, i: int) -> np.ndarray:
        return self.volume[i]


def load_volume(path: str | Path) -> OCTVolume:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    suffix = p.suffix.lower()
    if p.is_dir():
        return _load_image_folder(p)
    if suffix == ".dcm":
        return _load_dicom(p)
    if suffix in (".npy", ".npz"):
        return _load_npy(p)
    if suffix == ".gz" and p.stem.endswith(".nii"):
        return _load_nifti(p)
    if suffix in (".nii",):
        return _load_nifti(p)
    if suffix in (".tif", ".tiff"):
        return _load_tiff(p)
    if suffix in (".mhd", ".mha"):
        return _load_metaimage(p)
    raise ValueError(f"Unsupported volume format: {p}")


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr.astype(np.float32) - lo) / (hi - lo) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _load_dicom(path: Path) -> OCTVolume:
    import pydicom
    # Some Spectralis / 3rd-party exports strip the 128-byte preamble + "DICM"
    # magic but still contain a valid pixel data stream. Try the strict read
    # first (cheap, gets us PatientID/Manufacturer cleanly), then fall back to
    # force-read on failure.
    try:
        ds = pydicom.dcmread(str(path))
    except pydicom.errors.InvalidDicomError:
        ds = pydicom.dcmread(str(path), force=True)
        # Force-read without a known transfer syntax — set Implicit VR Little
        # Endian as a permissive default so pixel_array can decode.
        if not hasattr(ds.file_meta, "TransferSyntaxUID") \
                or not ds.file_meta.TransferSyntaxUID:
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    arr = ds.pixel_array  # likely (N_frames, H, W) for multi-frame OPT
    if arr.ndim == 2:
        arr = arr[None, ...]  # promote single B-scan to a 1-frame volume

    arr = _to_uint8(arr)

    # try to grab voxel spacing (Spectralis uses Pixel Measures sequence)
    spacing = None
    try:
        # standard pixel spacing (row, col) in mm
        ps = ds.get("PixelSpacing", None)
        slice_gap = ds.get("SpacingBetweenSlices", None) or ds.get("SliceThickness", None)
        if ps is not None and slice_gap is not None:
            spacing = (float(slice_gap), float(ps[0]), float(ps[1]))
    except Exception:
        pass

    meta = {
        "Modality": str(ds.get("Modality", "")),
        "Manufacturer": str(ds.get("Manufacturer", "")),
        "Model": str(ds.get("ManufacturerModelName", "")),
        "StudyDate": str(ds.get("StudyDate", "")),
        "PatientID": str(ds.get("PatientID", "")) if ds.get("PatientID", None) else "",
        "PhotometricInterpretation": str(ds.get("PhotometricInterpretation", "")),
        "NumberOfFrames": int(ds.get("NumberOfFrames", arr.shape[0])),
    }
    return OCTVolume(
        volume=arr, spacing=spacing, source=str(path),
        modality=str(ds.get("Modality", "OPT")), metadata=meta,
    )


def _load_nifti(path: Path) -> OCTVolume:
    import nibabel as nib
    img = nib.load(str(path))
    arr = img.get_fdata()
    if arr.ndim == 3:
        # Heuristic: smallest dim is slice axis (assuming H,W >> N for cube OCT)
        slice_axis = int(np.argmin(arr.shape))
        arr = np.moveaxis(arr, slice_axis, 0)
    arr = _to_uint8(arr)
    sp = img.header.get_zooms()
    spacing = tuple(float(x) for x in sp[:3]) if len(sp) >= 3 else None
    return OCTVolume(volume=arr, spacing=spacing, source=str(path), modality="OPT",
                     metadata={"NumberOfFrames": arr.shape[0]})


def _load_npy(path: Path) -> OCTVolume:
    if path.suffix == ".npz":
        data = np.load(path)
        key = list(data.keys())[0]
        arr = data[key]
    else:
        arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[None, ...]
    arr = _to_uint8(arr)
    return OCTVolume(volume=arr, spacing=None, source=str(path), modality="OPT",
                     metadata={"NumberOfFrames": arr.shape[0]})


def _load_tiff(path: Path) -> OCTVolume:
    """Multi-page TIFF holding a full B-scan stack (one page = one B-scan).

    Used by the TowardPi SS-OCT AMD/DME 3D dataset (each volume is a single
    512x512x512 multi-page .tif). tifffile reads all pages as (N, H, W);
    falls back to OpenCV's imreadmulti if tifffile is unavailable.
    """
    arr = None
    try:
        import tifffile
        arr = tifffile.imread(str(path))
    except Exception:
        import cv2
        ok, pages = cv2.imreadmulti(str(path), flags=cv2.IMREAD_UNCHANGED)
        if ok and pages:
            arr = np.stack([p if p.ndim == 2 else p[..., 0] for p in pages], axis=0)
    if arr is None:
        raise ValueError(f"could not read TIFF stack: {path}")
    arr = np.asarray(arr)
    if arr.ndim == 2:               # single B-scan
        arr = arr[None, ...]
    if arr.ndim == 4:               # (N, H, W, C) → drop channels to grey
        arr = arr[..., 0]
    arr = _to_uint8(arr)
    return OCTVolume(volume=arr, spacing=None, source=str(path), modality="OPT",
                     metadata={"NumberOfFrames": arr.shape[0]})


def _load_metaimage(path: Path) -> OCTVolume:
    """ITK MetaImage (.mhd header + .raw data) — the RETOUCH OCT-fluid format
    (oct.mhd/oct.raw, masks reference.mhd). Read via SimpleITK; the array comes
    back (z, y, x) = (B-scans, H, W) and spacing as (x, y, z) → reorder to (z,y,x).
    """
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)        # (z, y, x)
    if arr.ndim == 2:
        arr = arr[None, ...]
    arr = _to_uint8(arr)
    sp = img.GetSpacing()                     # (x, y, z)
    spacing = (float(sp[2]), float(sp[1]), float(sp[0])) if len(sp) >= 3 else None
    return OCTVolume(volume=arr, spacing=spacing, source=str(path), modality="OPT",
                     metadata={"NumberOfFrames": arr.shape[0], "format": "metaimage"})


def _load_image_folder(folder: Path) -> OCTVolume:
    import cv2
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise ValueError(f"No image files found in {folder}")
    slices = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
    arr = np.stack(slices, axis=0)
    return OCTVolume(volume=arr, spacing=None, source=str(folder),
                     modality="OPT", metadata={"NumberOfFrames": arr.shape[0]})
