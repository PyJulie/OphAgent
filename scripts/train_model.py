"""
Train a newly-developed OphAgent model.

Usage::
    python scripts/train_model.py --model cfp_quality --data-root data/cfp_quality
    python scripts/train_model.py --model cfp_disease --annotation-file data/cfp_disease/ann.json
    python scripts/train_model.py --model ffa_lesion --epochs 100 --batch-size 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TRAINER_MAP = {
    "cfp_quality":        "ophagent.models.training.cfp_quality_trainer:CFPQualityTrainer",
    "cfp_disease":        "ophagent.models.training.cfp_disease_trainer:CFPDiseaseTrainer",
    "cfp_ffa_multimodal": "ophagent.models.training.cfp_ffa_multimodal_trainer:CFPFFAMultimodalTrainer",
    "uwf_quality_disease":"ophagent.models.training.uwf_quality_disease_trainer:UWFQualityDiseaseTrainer",
    "cfp_glaucoma":       "ophagent.models.training.cfp_glaucoma_trainer:CFPGlaucomaTrainer",
    "cfp_pdr":            "ophagent.models.training.cfp_pdr_trainer:CFPPDRTrainer",
    "ffa_lesion":         "ophagent.models.training.ffa_lesion_trainer:FFALesionTrainer",
    "disc_fovea":         "ophagent.models.training.disc_fovea_trainer:DiscFoveaTrainer",
}


def get_trainer_cls(model_name: str):
    import importlib
    spec = TRAINER_MAP[model_name]
    module_path, cls_name = spec.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def main():
    parser = argparse.ArgumentParser(description="Train OphAgent Model")
    parser.add_argument("--model", "-m", required=True, choices=list(TRAINER_MAP.keys()))
    parser.add_argument("--data-root", type=str, help="Dataset root directory")
    parser.add_argument("--annotation-file", type=str, help="Annotation JSON file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="models_weights/checkpoints")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--num-classes", type=int, default=None)
    args = parser.parse_args()

    config = {
        "model_name": args.model,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "checkpoint_dir": args.checkpoint_dir,
    }
    if args.data_root:
        config["data_root"] = args.data_root
    if args.annotation_file:
        config["annotation_file"] = args.annotation_file
    if args.num_classes:
        config["num_classes"] = args.num_classes

    print(f"Training model: {args.model}")
    TrainerCls = get_trainer_cls(args.model)
    trainer = TrainerCls(config=config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    history = trainer.train()

    import json
    out_path = Path(args.checkpoint_dir) / f"{args.model}_history.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {out_path}")


if __name__ == "__main__":
    main()
