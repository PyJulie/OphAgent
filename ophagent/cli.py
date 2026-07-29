"""
CLI entry point for OphAgent.

Usage:
    ophagent download [--dataset NAME] [--all]
    ophagent train --config CONFIG_PATH
    ophagent predict --model MODEL_NAME --image IMAGE_PATH
    ophagent analyze --image IMAGE_PATH [--query QUERY] [--backend anthropic|openai|local]
    ophagent list-models
    ophagent list-datasets
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="ophagent", help="OphAgent: multimodal ophthalmology agent")
console = Console()


@app.command()
def download(
    dataset: str = typer.Option(None, help="Dataset name to download"),
    all_datasets: bool = typer.Option(False, "--all", help="Download all available datasets"),
    root: str = typer.Option("datasets", help="Root directory for datasets"),
):
    """Download public OCT datasets."""
    from ophagent.data.download.downloader import download_dataset, download_all

    if all_datasets:
        results = download_all(root)
        console.print(f"[green]Downloaded {len(results)} datasets[/green]")
    elif dataset:
        path = download_dataset(dataset, root)
        console.print(f"[green]Downloaded {dataset} to {path}[/green]")
    else:
        console.print("[yellow]Specify --dataset NAME or --all[/yellow]")


@app.command()
def train(
    config: str = typer.Argument(..., help="Path to training config YAML"),
):
    """Train an OCT model."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config)
    console.print(f"[blue]Training with config: {config}[/blue]")
    console.print(OmegaConf.to_yaml(cfg))

    task = cfg.task

    if task == "classification":
        _train_classifier(cfg)
    elif task == "segmentation":
        _train_segmentor(cfg)
    elif task == "generation":
        _train_generator(cfg)
    elif task == "ssl_pretrain":
        _train_ssl(cfg)
    else:
        console.print(f"[red]Unknown task: {task}[/red]")


def _train_classifier(cfg):
    import torch
    from torch.utils.data import DataLoader
    from ophagent.data.preprocessing.datasets import OCTClassificationDataset
    from ophagent.data.preprocessing.transforms import get_classification_transforms
    from ophagent.models.classification.classifier import OCTClassifier
    from ophagent.training.trainer import ClassificationTrainer

    train_transform = get_classification_transforms(cfg.data.image_size, is_training=True)
    val_transform = get_classification_transforms(cfg.data.image_size, is_training=False)

    train_ds = OCTClassificationDataset(
        root=Path(cfg.data.root) / "train", transform=train_transform
    )
    val_ds = OCTClassificationDataset(
        root=Path(cfg.data.root) / "val", transform=val_transform
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size,
        shuffle=True, num_workers=cfg.data.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.data.batch_size,
        shuffle=False, num_workers=cfg.data.num_workers, pin_memory=True,
    )

    model = OCTClassifier(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
    )

    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=cfg.training.max_epochs,
        patience=cfg.training.patience,
        output_dir=cfg.output.dir,
        class_names=train_ds.class_names,
    )

    result = trainer.fit()
    console.print(f"[green]Training complete. Best metric: {result['best_metric']:.4f}[/green]")


def _train_segmentor(cfg):
    import torch
    from torch.utils.data import DataLoader, Subset
    from ophagent.data.preprocessing.datasets import OCTSegmentationDataset
    from ophagent.data.preprocessing.transforms import get_segmentation_transforms
    from ophagent.models.segmentation.unet import OCTUNet
    from ophagent.training.trainer import SegmentationTrainer

    train_transform = get_segmentation_transforms(tuple(cfg.data.image_size), is_training=True)
    val_transform = get_segmentation_transforms(tuple(cfg.data.image_size), is_training=False)

    train_full = OCTSegmentationDataset(
        images_dir=cfg.data.images_dir,
        masks_dir=cfg.data.masks_dir,
        transform=train_transform,
        num_classes=cfg.model.num_classes,
    )

    val_images_dir = cfg.data.get("val_images_dir")
    val_masks_dir = cfg.data.get("val_masks_dir")
    if bool(val_images_dir) != bool(val_masks_dir):
        raise ValueError("Set both data.val_images_dir and data.val_masks_dir, or neither")

    if val_images_dir:
        train_ds = train_full
        val_ds = OCTSegmentationDataset(
            images_dir=val_images_dir,
            masks_dir=val_masks_dir,
            transform=val_transform,
            num_classes=cfg.model.num_classes,
        )
    else:
        # Build two dataset views so validation never receives random training
        # augmentation, then apply the same deterministic index split to both.
        sample_count = len(train_full)
        if sample_count < 2:
            raise ValueError("Segmentation training requires at least two image-mask pairs")
        val_ratio = float(cfg.data.get("val_ratio", 0.15))
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("data.val_ratio must be between 0 and 1")
        val_count = min(sample_count - 1, max(1, round(sample_count * val_ratio)))
        split_seed = int(cfg.data.get("split_seed", 42))
        indices = torch.randperm(
            sample_count, generator=torch.Generator().manual_seed(split_seed)
        ).tolist()
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]
        val_full = OCTSegmentationDataset(
            images_dir=cfg.data.images_dir,
            masks_dir=cfg.data.masks_dir,
            transform=val_transform,
            num_classes=cfg.model.num_classes,
        )
        train_ds = Subset(train_full, train_indices)
        val_ds = Subset(val_full, val_indices)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size,
        shuffle=True, num_workers=cfg.data.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.data.batch_size,
        shuffle=False, num_workers=cfg.data.num_workers,
    )

    model = OCTUNet(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        use_attention=cfg.model.use_attention,
    )

    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=cfg.training.max_epochs,
        patience=cfg.training.patience,
        output_dir=cfg.output.dir,
        num_classes=cfg.model.num_classes,
    )

    result = trainer.fit()
    console.print(f"[green]Training complete. Best Dice: {result['best_metric']:.4f}[/green]")


def _train_generator(cfg):
    from torch.utils.data import DataLoader
    from ophagent.data.preprocessing.datasets import OCTClassificationDataset
    from ophagent.data.preprocessing.transforms import get_generation_transforms
    from ophagent.models.generation.diffusion import OCTDiffusionModel
    from ophagent.training.trainer import GenerationTrainer

    transform = get_generation_transforms(cfg.data.image_size, is_training=True)
    dataset = OCTClassificationDataset(root=cfg.data.root, transform=transform)
    loader = DataLoader(
        dataset, batch_size=cfg.data.batch_size,
        shuffle=True, num_workers=cfg.data.num_workers,
    )

    model = OCTDiffusionModel(
        image_channels=cfg.model.image_channels,
        timesteps=cfg.model.timesteps,
    )

    trainer = GenerationTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        max_epochs=cfg.training.max_epochs,
        patience=cfg.training.patience,
        output_dir=cfg.output.dir,
    )

    result = trainer.fit()
    console.print(
        f"[green]Generation training complete. Best metric: "
        f"{result['best_metric']:.4f}[/green]"
    )


def _train_ssl(cfg):
    console.print("[yellow]SSL pretraining — use configs/training/ssl_pretrain.yaml[/yellow]")


@app.command()
def predict(
    model_name: str = typer.Option(..., "--model", help="Registered model name"),
    image: str = typer.Option(..., "--image", help="Path to OCT image"),
    output: str = typer.Option(None, "--output", help="Output path for results"),
):
    """Run inference with a trained model."""
    from ophagent.inference.model_registry import create_default_registry
    from ophagent.inference.predictor import OphPredictor

    registry = create_default_registry()
    predictor = OphPredictor(registry)

    result = predictor.predict(model_name, image)

    display_result = {
        k: v for k, v in result.items()
        if not isinstance(v, __import__("numpy").ndarray)
    }
    console.print_json(json.dumps(display_result, indent=2, default=str))

    if output:
        with open(output, "w") as f:
            json.dump(display_result, f, indent=2, default=str)


@app.command()
def analyze(
    image: str = typer.Option(..., "--image", help="Path to OCT image"),
    query: str = typer.Option(
        "Analyze this OCT image for retinal pathologies",
        "--query", help="Clinical question",
    ),
    backend: str = typer.Option("local", "--backend", help="LLM backend: anthropic, openai, local"),
    patient_info: str = typer.Option("", "--patient-info", help="Patient context"),
):
    """Run agent-based OCT analysis."""
    from ophagent.inference.model_registry import create_default_registry
    from ophagent.inference.predictor import OphPredictor
    from ophagent.agent.tools.oct_tools import OphAgentToolKit
    from ophagent.agent.engine import OphAgent

    registry = create_default_registry()
    predictor = OphPredictor(registry)
    toolkit = OphAgentToolKit(predictor)
    agent = OphAgent(toolkit, backend=backend)

    console.print(f"[blue]Analyzing: {image}[/blue]")
    console.print(f"[blue]Query: {query}[/blue]")
    console.print(f"[blue]Backend: {backend}[/blue]")

    result = agent.analyze(
        query=query,
        image_paths=[image],
        patient_context=patient_info,
    )

    console.print("\n[bold green]═══ Analysis Report ═══[/bold green]")
    console.print(result.final_report)
    console.print(f"\n[dim]Steps taken: {len(result.steps)}[/dim]")
    for step in result.steps:
        console.print(f"  Step {step['step']}: {step['tool']}")


@app.command(name="list-models")
def list_models():
    """List all registered models."""
    from ophagent.inference.model_registry import create_default_registry

    registry = create_default_registry()
    cards = registry.list_models()

    table = Table(title="Registered OCT Models")
    table.add_column("Name", style="cyan")
    table.add_column("Task", style="green")
    table.add_column("Description")
    table.add_column("Checkpoint", style="dim")

    for card in cards:
        exists = "✓" if Path(card.checkpoint_path).exists() else "✗"
        table.add_row(card.name, card.task, card.description[:60], exists)

    console.print(table)


@app.command(name="list-datasets")
def list_datasets():
    """List all available OCT datasets."""
    from ophagent.data.download.registry import DATASET_REGISTRY

    table = Table(title="Available OCT Datasets")
    table.add_column("Key", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Size")
    table.add_column("Tasks")
    table.add_column("Source", style="dim")

    for key, info in DATASET_REGISTRY.items():
        tasks = ", ".join(t.value for t in info.tasks)
        table.add_row(key, info.name, info.size, tasks, info.source.value)

    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()
