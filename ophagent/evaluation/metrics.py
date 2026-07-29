"""Small, dependency-free metrics for protocol-based grading evaluations."""

from __future__ import annotations

from collections import Counter
from typing import Iterable


def valid_pairs(rows: Iterable[dict], gt_key: str = "gt_grade",
                pred_key: str = "grade") -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for row in rows:
        gt = row.get(gt_key)
        pred = row.get(pred_key)
        if isinstance(gt, str) and gt.isdigit():
            gt = int(gt)
        if isinstance(pred, str) and pred.isdigit():
            pred = int(pred)
        if isinstance(gt, int) and isinstance(pred, int) and 0 <= gt <= 4 and 0 <= pred <= 4:
            out.append((gt, pred))
    return out


def accuracy(pairs: Iterable[tuple[int, int]]) -> float | None:
    data = list(pairs)
    if not data:
        return None
    return sum(1 for gt, pred in data if gt == pred) / len(data)


def binary_accuracy(pairs: Iterable[tuple[int, int]], threshold: int) -> float | None:
    data = list(pairs)
    if not data:
        return None
    return sum(1 for gt, pred in data if (gt >= threshold) == (pred >= threshold)) / len(data)


def confusion_matrix(pairs: Iterable[tuple[int, int]], n_classes: int = 5) -> list[list[int]]:
    matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for gt, pred in pairs:
        if 0 <= gt < n_classes and 0 <= pred < n_classes:
            matrix[gt][pred] += 1
    return matrix


def per_class_f1(pairs: Iterable[tuple[int, int]], n_classes: int = 5) -> list[float | None]:
    data = list(pairs)
    scores: list[float | None] = []
    for cls in range(n_classes):
        tp = sum(1 for gt, pred in data if gt == cls and pred == cls)
        fp = sum(1 for gt, pred in data if gt != cls and pred == cls)
        fn = sum(1 for gt, pred in data if gt == cls and pred != cls)
        if tp + fp == 0 or tp + fn == 0:
            scores.append(None)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        scores.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return scores


def quadratic_weighted_kappa(pairs: Iterable[tuple[int, int]], n_classes: int = 5) -> float | None:
    data = list(pairs)
    if not data:
        return None
    matrix = confusion_matrix(data, n_classes)
    gt_counts = [sum(row) for row in matrix]
    pred_counts = [sum(matrix[i][j] for i in range(n_classes)) for j in range(n_classes)]
    total = sum(gt_counts)
    if total == 0:
        return None

    observed = 0.0
    expected = 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            weight = (i - j) ** 2
            observed += weight * matrix[i][j]
            expected += weight * gt_counts[i] * pred_counts[j] / total
    return 1.0 - observed / expected if expected else None


def summarize_grading(rows: list[dict], gt_key: str = "gt_grade",
                      pred_key: str = "grade") -> dict:
    pairs = valid_pairs(rows, gt_key=gt_key, pred_key=pred_key)
    gt_dist = Counter(gt for gt, _ in pairs)
    pred_dist = Counter(pred for _, pred in pairs)
    return {
        "n_total": len(rows),
        "n_valid": len(pairs),
        "gt_distribution": {str(k): gt_dist.get(k, 0) for k in range(5)},
        "pred_distribution": {str(k): pred_dist.get(k, 0) for k in range(5)},
        "accuracy_5class": accuracy(pairs),
        "accuracy_any_dr_ge1": binary_accuracy(pairs, threshold=1),
        "accuracy_referable_dr_ge2": binary_accuracy(pairs, threshold=2),
        "accuracy_pdr_eq4": (
            sum(1 for gt, pred in pairs if (gt == 4) == (pred == 4)) / len(pairs)
            if pairs else None
        ),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(pairs),
        "per_class_f1": per_class_f1(pairs),
        "confusion_matrix": confusion_matrix(pairs),
    }
