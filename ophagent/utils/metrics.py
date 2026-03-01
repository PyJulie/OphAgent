"""Evaluation metrics used across OphAgent model training and verification."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def auc_roc(y_true: List[int], y_score: List[float]) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def f1_score(y_true: List[int], y_pred: List[int], average: str = "macro") -> float:
    from sklearn.metrics import f1_score as _f1
    return float(_f1(y_true, y_pred, average=average, zero_division=0))


def sensitivity_specificity(
    y_true: List[int], y_pred: List[int]
) -> Dict[str, float]:
    """Compute binary sensitivity and specificity."""
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    return {"sensitivity": float(sensitivity), "specificity": float(specificity)}


def dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Binary Dice coefficient for segmentation masks."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = (pred & gt).sum()
    return float(2 * intersection / (pred.sum() + gt.sum() + 1e-8))


def iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Intersection-over-Union for binary masks."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(intersection / (union + 1e-8))


def classification_report(
    y_true: List[int],
    y_pred: List[int],
    labels: Optional[List[str]] = None,
) -> str:
    from sklearn.metrics import classification_report as _cr
    return _cr(y_true, y_pred, target_names=labels, zero_division=0)


def kappa_score(y_true: List[int], y_pred: List[int]) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y_true, y_pred))
