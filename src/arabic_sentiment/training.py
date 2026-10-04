"""CPU-capable AraBERT fine-tuning and one-time final test evaluation."""

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Adafactor, AutoModelForSequenceClassification, AutoTokenizer

from .modeling import ID_TO_LABEL, LABEL_TO_ID, MODEL_CHECKPOINT


@dataclass(frozen=True)
class TrainingConfig:
    model_checkpoint: str = MODEL_CHECKPOINT
    seed: int = 42
    learning_rate: float = 2e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    epochs: int = 1
    max_sequence_length: int = 96
    max_train_samples: int | None = None
    weight_decay: float = 0.01
    max_gradient_norm: float = 1.0
    warmup_ratio: float = 0.06
    evaluation_strategy: str = "epoch"
    best_model_metric: str = "validation_f1"
    checkpoint_strategy: str = "save best validation F1 only"
    train_csv: str = "data/processed/train.csv"
    validation_csv: str = "data/processed/validation.csv"
    test_csv: str = "data/processed/test.csv"
    artifact_dir: str = "models/baseline-arabert"
    num_workers: int = 0

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        with Path(path).open(encoding="utf-8") as config_file:
            return cls(**json.load(config_file))


class ReviewDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        missing = {"review", "sentiment"} - set(frame.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
        if frame["review"].isna().any() or frame["sentiment"].isna().any():
            raise ValueError("Review and sentiment columns cannot contain missing values")
        unexpected = set(frame["sentiment"].unique()) - LABEL_TO_ID.keys()
        if unexpected:
            raise ValueError(f"Unexpected sentiment labels: {sorted(unexpected)}")
        self.texts = frame["review"].astype(str).tolist()
        self.labels = frame["sentiment"].map(LABEL_TO_ID).astype("int64").tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], self.labels[index]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_config(path: str | Path) -> TrainingConfig:
    config = TrainingConfig.from_json(path)
    if config.model_checkpoint != MODEL_CHECKPOINT:
        raise ValueError(
            f"This Track A baseline is fixed to {MODEL_CHECKPOINT}; refusing unrelated checkpoint "
            f"{config.model_checkpoint!r}"
        )
    if config.epochs < 1 or config.batch_size < 1 or config.max_sequence_length < 2:
        raise ValueError("epochs, batch_size, and max_sequence_length must be positive")
    return config


def _collate(tokenizer, max_length: int):
    def collate(batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts, labels = zip(*batch)
        encoded = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    return collate


def _metrics(targets: list[int], predictions: list[int], loss_sum: float) -> dict[str, float]:
    tp = sum(y == 1 and p == 1 for y, p in zip(targets, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(targets, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(targets, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(targets, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "loss": loss_sum / max(len(targets), 1),
        "accuracy": (tp + tn) / max(len(targets), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@torch.inference_mode()
def evaluate(model, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    loss_sum = 0.0
    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        count = int(batch["labels"].shape[0])
        loss_sum += float(outputs.loss.item()) * count
        predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
        targets.extend(batch["labels"].cpu().tolist())
    return _metrics(targets, predictions, loss_sum)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sample_training_rows(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    """Take a deterministic proportional sample from training data only."""
    if limit is None or len(frame) <= limit:
        return frame
    if limit < frame["sentiment"].nunique():
        raise ValueError("max_train_samples must provide at least one row per class")

    counts = frame["sentiment"].value_counts().sort_index()
    exact = counts / len(frame) * limit
    allocations = exact.map(np.floor).astype(int)
    remainder = limit - int(allocations.sum())
    remainders = (exact - allocations).sort_values(ascending=False, kind="stable")
    for label in remainders.index[:remainder]:
        allocations.loc[label] += 1

    sampled = [
        group.sample(n=int(allocations.loc[label]), random_state=seed)
        for label, group in frame.groupby("sentiment", sort=True)
        if allocations.loc[label]
    ]
    return pd.concat(sampled).sample(frac=1, random_state=seed).reset_index(drop=True)


def train_and_evaluate(config: TrainingConfig) -> dict[str, Any]:
    """Fine-tune one CPU baseline, select on validation F1, and test once."""
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

    train_frame = pd.read_csv(config.train_csv, encoding="utf-8", usecols=["review", "sentiment"])
    validation_frame = pd.read_csv(
        config.validation_csv, encoding="utf-8", usecols=["review", "sentiment"]
    )
    test_frame = pd.read_csv(config.test_csv, encoding="utf-8", usecols=["review", "sentiment"])
    available_train_rows = len(train_frame)
    train_frame = _sample_training_rows(train_frame, config.max_train_samples, config.seed)
    train_dataset = ReviewDataset(train_frame)
    validation_dataset = ReviewDataset(validation_frame)
    test_dataset = ReviewDataset(test_frame)

    tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint, use_fast=True)
    collate = _collate(tokenizer, config.max_sequence_length)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_checkpoint,
        num_labels=2,
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
    ).to(device)
    optimizer = Adafactor(
        model.parameters(),
        lr=config.learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=config.weight_decay,
    )
    optimizer_steps = math.ceil(len(train_loader) / config.gradient_accumulation_steps) * config.epochs
    warmup_steps = int(optimizer_steps * config.warmup_ratio)

    def lr_factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (optimizer_steps - step) / max(optimizer_steps - warmup_steps, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    artifact_dir = Path(config.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    validation_history = []
    completed_steps = 0

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        rows_seen = 0
        for batch_index, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            batch_size = int(batch["labels"].shape[0])
            loss = model(**batch).loss
            epoch_loss_sum += float(loss.detach().item()) * batch_size
            rows_seen += batch_size
            (loss / config.gradient_accumulation_steps).backward()

            update_now = (batch_index + 1) % config.gradient_accumulation_steps == 0
            final_batch = batch_index + 1 == len(train_loader)
            if update_now or final_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1
                if completed_steps % 100 == 0:
                    print(
                        f"epoch={epoch + 1}/{config.epochs} optimizer_step={completed_steps}/"
                        f"{optimizer_steps} train_loss={epoch_loss_sum / rows_seen:.4f}",
                        flush=True,
                    )

        validation_metrics = evaluate(model, validation_loader, device)
        validation_metrics.update(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss_sum / max(rows_seen, 1),
            }
        )
        validation_history.append(validation_metrics)
        print(f"validation_epoch_{epoch + 1}: {json.dumps(validation_metrics, sort_keys=True)}", flush=True)

        if validation_metrics["f1"] > best_f1:
            best_f1 = validation_metrics["f1"]
            model.save_pretrained(artifact_dir, safe_serialization=True)
            tokenizer.save_pretrained(artifact_dir)
            _write_json(
                artifact_dir / "training_config.json",
                {
                    **asdict(config),
                    "device": str(device),
                    "optimizer": "transformers.Adafactor",
                    "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
                    "optimizer_steps": optimizer_steps,
                    "available_train_rows": available_train_rows,
                    "actual_train_rows": len(train_dataset),
                    "actual_validation_rows": len(validation_dataset),
                    "label_to_id": LABEL_TO_ID,
                },
            )

    best_model = AutoModelForSequenceClassification.from_pretrained(artifact_dir, local_files_only=True)
    best_model.to(device)
    test_metrics = evaluate(best_model, test_loader, device)
    report = {
        "model_checkpoint": config.model_checkpoint,
        "device": str(device),
        "seed": config.seed,
        "training_config": asdict(config),
        "optimizer": "transformers.Adafactor",
        "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
        "actual_rows": {
            "available_train": available_train_rows,
            "train": len(train_dataset),
            "validation": len(validation_dataset),
            "test": len(test_dataset),
        },
        "label_to_id": LABEL_TO_ID,
        "validation_history": validation_history,
        "best_validation_f1": best_f1,
        "test_metrics": test_metrics,
        "test_evaluations": 1,
    }
    _write_json(artifact_dir / "training_results.json", report)
    return report
