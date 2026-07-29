"""
Unified training framework for all OCT model types.

Supports: classification, segmentation, generation, SSL pretraining.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.metrics import (
    compute_classification_metrics,
    compute_segmentation_metrics,
    dice_coefficient,
)
from ..utils.paths import CKPT_DIR


class Trainer:
    """Base trainer with common training loop logic."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        criterion: nn.Module | None = None,
        device: str | torch.device = "cuda",
        output_dir: str | Path | None = None,
        max_epochs: int = 100,
        patience: int = 15,
        grad_clip: float = 1.0,
        mixed_precision: bool = True,
        log_interval: int = 50,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.output_dir = Path(output_dir) if output_dir else CKPT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.log_interval = log_interval

        self.optimizer = optimizer or AdamW(
            model.parameters(), lr=1e-4, weight_decay=0.01
        )
        self.scheduler = scheduler or CosineAnnealingLR(
            self.optimizer, T_max=max_epochs
        )
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.scaler = torch.amp.GradScaler("cuda") if mixed_precision else None
        self.mixed_precision = mixed_precision

        self.best_metric = -float("inf")
        self.epochs_no_improve = 0
        self.history: list[dict] = []

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            loss = self._train_step(batch)
            total_loss += loss
            num_batches += 1

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix(loss=f"{loss:.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        if self.scheduler:
            self.scheduler.step()
        return {"train_loss": avg_loss}

    def _train_step(self, batch: dict) -> float:
        self.optimizer.zero_grad()

        if self.mixed_precision and self.scaler:
            with torch.amp.autocast("cuda"):
                loss = self._compute_loss(batch)
            self.scaler.scale(loss).backward()
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self._compute_loss(batch)
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        return loss.item()

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        raise NotImplementedError

    def fit(self) -> dict[str, Any]:
        print(f"Training for up to {self.max_epochs} epochs (patience={self.patience})")
        print(f"Output dir: {self.output_dir}")

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate() if self.val_loader else {}
            elapsed = time.time() - t0

            metrics = {**train_metrics, **val_metrics, "epoch": epoch, "time": elapsed}
            self.history.append(metrics)

            print(
                f"[Epoch {epoch}] "
                + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, float))
            )

            monitor_key = next(
                (k for k in ("val_auc", "val_dice", "val_f1", "val_acc") if k in val_metrics),
                None,
            )
            if monitor_key:
                val_score = val_metrics[monitor_key]
                if val_score > self.best_metric:
                    self.best_metric = val_score
                    self.epochs_no_improve = 0
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.epochs_no_improve += 1

                if self.epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                self._save_checkpoint(epoch, is_best=(epoch == 1))

        self._save_checkpoint(epoch, filename="last.pt")
        return {"history": self.history, "best_metric": self.best_metric}

    def _save_checkpoint(
        self, epoch: int, is_best: bool = False, filename: str | None = None
    ) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
        }
        if filename:
            path = self.output_dir / filename
        elif is_best:
            path = self.output_dir / "best.pt"
        else:
            path = self.output_dir / f"epoch_{epoch}.pt"
        torch.save(state, path)


class ClassificationTrainer(Trainer):
    """Trainer for OCT classification models."""

    def __init__(self, class_names: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_names = class_names

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)
        output = self.model(images)
        return self.criterion(output["logits"], labels)

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"]
            output = self.model(images)
            probs = output["probabilities"].cpu()
            preds = probs.argmax(dim=-1)
            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)

        import numpy as np
        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        probs = torch.cat(all_probs).numpy()

        metrics = compute_classification_metrics(
            labels, preds, probs, self.class_names
        )
        return {
            "val_acc": metrics["accuracy"],
            "val_f1": metrics["f1_macro"],
            "val_kappa": metrics["kappa"],
            **({f"val_auc": metrics["auc_macro"]} if "auc_macro" in metrics else {}),
        }


class SegmentationTrainer(Trainer):
    """Trainer for OCT segmentation models."""

    def __init__(self, num_classes: int = 9, **kwargs):
        if "criterion" not in kwargs:
            kwargs["criterion"] = DiceCELoss(num_classes=num_classes)
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        images = batch["image"].to(self.device)
        masks = batch["mask"].to(self.device)
        output = self.model(images)
        return self.criterion(output["logits"], masks)

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        total_dice = 0.0
        count = 0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            output = self.model(images)
            pred = output["logits"].argmax(dim=1)

            for c in range(1, self.num_classes):
                d = dice_coefficient(
                    (pred == c).unsqueeze(1).float(),
                    (masks == c).unsqueeze(1).float(),
                )
                total_dice += d.mean().item()
                count += 1

        avg_dice = total_dice / max(count, 1)
        return {"val_dice": avg_dice}


class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss for segmentation."""

    def __init__(self, num_classes: int, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)

        probs = torch.softmax(logits, dim=1)
        targets_onehot = nn.functional.one_hot(
            targets.long(), self.num_classes
        ).permute(0, 3, 1, 2).float()

        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))
        dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
        dice_loss = 1.0 - dice.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


class SSLPretrainer(Trainer):
    """Trainer for MAE self-supervised pretraining."""

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        images = batch["image"].to(self.device)
        output = self.model(images)
        pred = output["pred"]
        mask = output["mask"]

        B = images.shape[0]
        p = self.model.patch_size
        target = images.unfold(2, p, p).unfold(3, p, p)
        target = target.contiguous().view(B, -1, p * p * images.shape[1])

        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        count = 0
        for batch in self.val_loader:
            loss = self._compute_loss(batch)
            total_loss += loss.item()
            count += 1
        return {"val_loss": total_loss / max(count, 1)}


class GenerationTrainer(Trainer):
    """Trainer for diffusion-based generation models."""

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        images = batch["image"].to(self.device)
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        return self.model.training_loss(images)

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        count = 0
        for batch in self.val_loader:
            loss = self._compute_loss(batch)
            total_loss += loss.item()
            count += 1
        return {"val_loss": total_loss / max(count, 1)}
