"""
Patient-level data splitting utilities.

Critical for OCT: random B-scan splitting causes data leakage because
multiple scans from the same patient share retinal features.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def extract_patient_id(filename: str) -> str:
    """Heuristic extraction of patient ID from filename.

    Tries common naming patterns:
      - patient_001_scan_003.png -> patient_001
      - P001_S003.png -> P001
      - Subject-01-slice-05.png -> Subject-01
    Falls back to the full stem if no pattern matches.
    """
    patterns = [
        r"(patient[_-]?\d+)",
        r"(subject[_-]?\d+)",
        r"(P\d+)[_-]",
        r"(S\d+)[_-]",
        r"(\d+)[_-]\d+$",
    ]
    stem = Path(filename).stem
    for pattern in patterns:
        match = re.search(pattern, stem, re.IGNORECASE)
        if match:
            return match.group(1)
    return stem


def patient_level_split(
    samples: list[tuple[Path, int]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    patient_id_fn=extract_patient_id,
) -> dict[str, list[tuple[Path, int]]]:
    """Split samples ensuring no patient appears in multiple splits."""

    patient_to_samples: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for path, label in samples:
        pid = patient_id_fn(path.name)
        patient_to_samples[pid].append((path, label))

    patient_ids = list(patient_to_samples.keys())

    rng = np.random.RandomState(seed)
    indices = np.arange(len(patient_ids))
    rng.shuffle(indices)

    n = len(patient_ids)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))

    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    splits: dict[str, list[tuple[Path, int]]] = {"train": [], "val": [], "test": []}
    for idx in train_idx:
        splits["train"].extend(patient_to_samples[patient_ids[idx]])
    for idx in val_idx:
        splits["val"].extend(patient_to_samples[patient_ids[idx]])
    for idx in test_idx:
        splits["test"].extend(patient_to_samples[patient_ids[idx]])

    print(f"Patient-level split: "
          f"{len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test patients, "
          f"{len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])} samples")

    return splits


def save_split(splits: dict[str, list[tuple[Path, int]]], output_path: Path) -> None:
    serializable = {}
    for split_name, samples in splits.items():
        serializable[split_name] = [
            {"path": str(p), "label": l} for p, l in samples
        ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved splits to {output_path}")
