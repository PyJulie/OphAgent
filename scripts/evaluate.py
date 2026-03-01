"""
Evaluate OphAgent on a benchmark dataset.

Usage::
    python scripts/evaluate.py \
        --tool cfp_disease \
        --test-file data/cfp_disease/test_annotations.json \
        --output results/cfp_disease_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Evaluate OphAgent Tool")
    parser.add_argument("--tool", "-t", required=True, help="Tool ID to evaluate")
    parser.add_argument("--test-file", required=True, help="Test annotation JSON")
    parser.add_argument("--output", "-o", default="results/eval.json")
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()

    from ophagent.tools.scheduler import ToolScheduler
    from ophagent.utils.metrics import (
        accuracy, f1_score, auc_roc, dice_coefficient, iou
    )

    scheduler = ToolScheduler()

    with open(args.test_file, encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Evaluating {args.tool} on {len(test_data)} samples...")

    y_true, y_pred, y_score = [], [], []
    errors = 0

    for item in test_data:
        image_path = item["image_path"]
        gt_label = item.get("label", item.get("labels"))
        inputs = {"image_path": image_path}

        try:
            result = scheduler.run(args.tool, inputs)
        except Exception as e:
            print(f"  Error on {image_path}: {e}")
            errors += 1
            continue

        pred = result.get("label", result.get("labels"))
        probs = result.get("probabilities", {})

        if isinstance(gt_label, int) and isinstance(pred, str):
            # Map string label back to int if needed
            pred_int = list(probs.keys()).index(pred) if probs else 0
            y_true.append(gt_label)
            y_pred.append(pred_int)
        elif isinstance(gt_label, int):
            y_true.append(gt_label)
            y_pred.append(pred if isinstance(pred, int) else 0)

        if probs:
            y_score.append(list(probs.values()))

    print(f"\nResults ({len(y_true)} samples, {errors} errors):")
    metrics = {}

    if y_true and y_pred:
        metrics["accuracy"] = accuracy(y_true, y_pred)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Macro: {metrics['f1_macro']:.4f}")

    if y_score and len(y_score[0]) == 2:
        scores_flat = [s[1] for s in y_score]
        try:
            metrics["auc"] = auc_roc(y_true, scores_flat)
            print(f"  AUC-ROC:  {metrics['auc']:.4f}")
        except Exception:
            pass

    metrics["total"] = len(y_true)
    metrics["errors"] = errors

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
