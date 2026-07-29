from __future__ import annotations

from pathlib import Path


PREFERRED_COLUMNS = (
    "path",
    "filepath",
    "file_path",
    "file",
    "filename",
    "source_path",
    "src",
    "id",
    "patient_id",
    "case_id",
    "vid",
)

HEADER_HINTS = {
    "path",
    "filepath",
    "file_path",
    "file",
    "filename",
    "source_path",
    "src",
    "id",
    "patient_id",
    "case_id",
    "vid",
}


def _normalize_cell(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"nan", "none", "null"}:
        return None
    return text


def _select_named_column(columns: list[str]) -> str | None:
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for preferred in PREFERRED_COLUMNS:
        if preferred in normalized:
            return normalized[preferred]
    return None


def _drop_header_like_first_value(values: list[str]) -> list[str]:
    if not values:
        return values
    first = values[0].strip().lower()
    if first in HEADER_HINTS:
        return values[1:]
    return values


def _read_tabular(manifest_path: Path, *, header: int | None):
    import pandas as pd

    suffix = manifest_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(manifest_path, header=header, dtype=str)
        except ImportError as exc:
            raise RuntimeError(
                f"Reading {manifest_path.suffix} manifests requires openpyxl in the execution environment."
            ) from exc
    if suffix == ".tsv":
        return pd.read_csv(manifest_path, header=header, dtype=str, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(manifest_path, header=header, dtype=str)
    if suffix == ".txt":
        with manifest_path.open("r", encoding="utf-8") as handle:
            lines = [line.rstrip("\n\r") for line in handle]
        if header is None:
            rows = lines
        else:
            rows = lines[1:]
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def read_manifest_entries(manifest_path: Path) -> list[str]:
    suffix = manifest_path.suffix.lower()
    if suffix not in {".csv", ".tsv", ".txt", ".xlsx", ".xls"}:
        raise ValueError(f"Unsupported manifest format: {manifest_path}")

    with_headers = _read_tabular(manifest_path, header=0)
    if not with_headers.empty:
        column = _select_named_column([str(column) for column in with_headers.columns])
        if column is not None:
            values = [
                normalized
                for normalized in (_normalize_cell(value) for value in with_headers[column].tolist())
                if normalized is not None
            ]
            if values:
                return values

    without_headers = _read_tabular(manifest_path, header=None)
    if without_headers.empty or without_headers.shape[1] == 0:
        raise ValueError(f"No entries found in manifest: {manifest_path}")

    values = [
        normalized
        for normalized in (_normalize_cell(value) for value in without_headers.iloc[:, 0].tolist())
        if normalized is not None
    ]
    values = _drop_header_like_first_value(values)
    if not values:
        raise ValueError(f"No usable entries found in manifest: {manifest_path}")
    return values
