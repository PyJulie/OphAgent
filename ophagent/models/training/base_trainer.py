"""
Base Trainer for OphAgent newly-developed models.

All model-specific trainers inherit from BaseTrainer, which provides:
  - Training loop with AMP (automatic mixed precision)
  - Checkpoint saving/loading
  - Metric logging via loguru
  - Early stopping
  - LR scheduling
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from ophagent.utils.logger import get_logger
from ophagent.utils.metrics import accuracy, auc_roc, f1_score

logger = get_logger("models.base_trainer")


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class BaseTrainer(ABC):
    """
    Abstract base trainer.  Subclasses implement:
      - build_model()  -> nn.Module
      - build_datasets() -> (train_dataset, val_dataset)
      - compute_loss(outputs, targets) -> Tensor
      - compute_metrics(outputs, targets) -> dict
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        from config.settings import get_settings
        cfg = get_settings().training
        self.config = config or {}

        self.batch_size: int = self.config.get("batch_size", cfg.batch_size)
        self.lr: float = self.config.get("learning_rate", cfg.learning_rate)
        self.num_epochs: int = self.config.get("num_epochs", cfg.num_epochs)
        self.warmup_epochs: int = self.config.get("warmup_epochs", cfg.warmup_epochs)
        self.weight_decay: float = self.config.get("weight_decay", cfg.weight_decay)
        self.mixed_precision: bool = self.config.get("mixed_precision", cfg.mixed_precision)
        self.save_every: int = self.config.get("save_every_n_epochs", cfg.save_every_n_epochs)
        self.checkpoint_dir: Path = Path(self.config.get("checkpoint_dir", str(cfg.checkpoint_root)))
        self.model_name: str = self.config.get("model_name", "model")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.scaler = GradScaler() if self.mixed_precision else None

        self.early_stopping = EarlyStopping(
            patience=self.config.get("patience", 10)
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build and return the model."""

    @abstractmethod
    def build_datasets(self) -> Tuple[Any, Any]:
        """Return (train_dataset, val_dataset)."""

    @abstractmethod
    def compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        """Compute the training loss."""

    def compute_metrics(self, outputs: Any, targets: Any) -> Dict[str, float]:
        """Compute validation metrics.  Override for task-specific metrics."""
        if hasattr(outputs, "argmax"):
            preds = outputs.argmax(dim=-1).cpu().tolist()
            gt = targets.cpu().tolist() if hasattr(targets, "cpu") else targets
            return {"accuracy": accuracy(gt, preds)}
        return {}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, List[float]]:
        """Run the full training loop. Returns history dict."""
        # Build model
        self.model = self.build_model().to(self.device)
        logger.info(
            f"Training {self.model_name}: "
            f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} params"
        )

        # Build datasets
        train_ds, val_ds = self.build_datasets()
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size * 2, shuffle=False,
            num_workers=4, pin_memory=True,
        )

        # Optimizer + scheduler (§2.2.2: 5-epoch linear warmup → cosine annealing)
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        warmup_ep = max(1, self.warmup_epochs)
        cosine_ep = max(1, self.num_epochs - warmup_ep)
        _warmup_sched  = LinearLR(self.optimizer,
                                   start_factor=1e-3, end_factor=1.0,
                                   total_iters=warmup_ep)
        _cosine_sched  = CosineAnnealingLR(self.optimizer, T_max=cosine_ep)
        self.scheduler = SequentialLR(self.optimizer,
                                       schedulers=[_warmup_sched, _cosine_sched],
                                       milestones=[warmup_ep])

        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "val_metric": []
        }
        best_val_metric = float("-inf")

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.monotonic()
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self._val_epoch(val_loader)
            self.scheduler.step()

            val_metric = val_metrics.get("accuracy", val_metrics.get("auc", 0.0))
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_metric"].append(val_metric)

            elapsed = time.monotonic() - t0
            logger.info(
                f"Epoch {epoch}/{self.num_epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_metric={val_metric:.4f} | {elapsed:.1f}s"
            )

            # Save checkpoint
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, val_metric)

            # Best model
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                self._save_checkpoint(epoch, val_metric, suffix="best")

            # Early stopping
            if self.early_stopping(val_metric):
                logger.info(f"Early stopping at epoch {epoch}.")
                break

        logger.info(f"Training complete. Best val metric: {best_val_metric:.4f}")
        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            inputs, targets = self._unpack_batch(batch)
            self.optimizer.zero_grad()
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.compute_loss(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, targets)
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    def _val_epoch(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_outputs, all_targets = [], []
        with torch.no_grad():
            for batch in loader:
                inputs, targets = self._unpack_batch(batch)
                if self.mixed_precision:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.compute_loss(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.compute_loss(outputs, targets)
                total_loss += loss.item()
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu() if hasattr(targets, "cpu") else targets)

        all_out = torch.cat(all_outputs, dim=0)
        all_tgt = torch.cat(all_targets, dim=0) if isinstance(all_targets[0], torch.Tensor) else sum(all_targets, [])
        metrics = self.compute_metrics(all_out, all_tgt)
        return total_loss / max(len(loader), 1), metrics

    def _unpack_batch(self, batch) -> Tuple[Any, Any]:
        """Override if batch format differs from (inputs, targets)."""
        inputs, targets = batch
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        if isinstance(targets, torch.Tensor):
            targets = targets.to(self.device)
        return inputs, targets

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, metric: float, suffix: str = "") -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{self.model_name}_{suffix or f'ep{epoch}'}_metric{metric:.4f}.pth"
        path = self.checkpoint_dir / fname
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metric": metric,
            "config": self.config,
        }, str(path))
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load a checkpoint.  Returns the epoch number."""
        ckpt = torch.load(path, map_location=self.device)
        if self.model is None:
            self.model = self.build_model().to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")
        return ckpt.get("epoch", 0)
