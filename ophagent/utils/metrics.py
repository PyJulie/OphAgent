"""Evaluation metrics for OCT tasks."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)


# ── Classification ──────────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }
    if y_prob is not None and y_prob.shape[-1] > 1:
        try:
            metrics["auc_macro"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
        except ValueError:
            pass
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    metrics["per_class"] = {
        k: v for k, v in report.items()
        if k not in ("accuracy", "macro avg", "weighted avg")
    }
    return metrics


# ── Segmentation ────────────────────────────────────────────────────────────

def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred_flat = pred.flatten(1).float()
    target_flat = target.flatten(1).float()
    intersection = (pred_flat * target_flat).sum(1)
    return (2.0 * intersection + eps) / (pred_flat.sum(1) + target_flat.sum(1) + eps)


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred_flat = pred.flatten(1).float()
    target_flat = target.flatten(1).float()
    intersection = (pred_flat * target_flat).sum(1)
    union = pred_flat.sum(1) + target_flat.sum(1) - intersection
    return (intersection + eps) / (union + eps)


def hausdorff_distance_95(
    pred: np.ndarray, target: np.ndarray
) -> float:
    from scipy.ndimage import distance_transform_edt

    if pred.sum() == 0 or target.sum() == 0:
        return float("inf")

    pred_boundary = pred.astype(bool)
    target_boundary = target.astype(bool)

    dt_pred = distance_transform_edt(~pred_boundary)
    dt_target = distance_transform_edt(~target_boundary)

    d_pred_to_target = dt_target[pred_boundary]
    d_target_to_pred = dt_pred[target_boundary]

    all_distances = np.concatenate([d_pred_to_target, d_target_to_pred])
    return float(np.percentile(all_distances, 95))


def compute_segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> dict:
    metrics = {}
    for c in range(num_classes):
        pred_c = (pred == c).long()
        target_c = (target == c).long()
        metrics[f"dice_class_{c}"] = dice_coefficient(
            pred_c.unsqueeze(0), target_c.unsqueeze(0)
        ).item()
        metrics[f"iou_class_{c}"] = iou_score(
            pred_c.unsqueeze(0), target_c.unsqueeze(0)
        ).item()
    metrics["dice_mean"] = np.mean(
        [v for k, v in metrics.items() if k.startswith("dice_")]
    )
    metrics["iou_mean"] = np.mean(
        [v for k, v in metrics.items() if k.startswith("iou_")]
    )
    return metrics
