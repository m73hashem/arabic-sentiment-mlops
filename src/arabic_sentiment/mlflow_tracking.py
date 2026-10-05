"""Local MLflow tracking, evaluation, and registry helpers for AraBERT."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow import MlflowClient
from mlflow.models import ModelSignature
from mlflow.pyfunc import PythonModel
from mlflow.types import ColSpec, Schema

from .dvc_lineage import (
    DEFAULT_DVC_POINTER,
    current_git_revision,
    mlflow_lineage_tags,
)
from .inference import SentimentPredictor
from .modeling import LABEL_TO_ID, MODEL_CHECKPOINT

EXPERIMENT_NAME = "arabic-sentiment"
REGISTERED_MODEL_NAME = "arabic-sentiment"
MODEL_ARTIFACT_PATH = "model"
DEFAULT_TRACKING_DB = Path("mlflow.db").resolve()
DEFAULT_ARTIFACT_ROOT = Path("mlartifacts").resolve()
OFFICIAL_ARTIFACT = Path("models/full-gpu-arabert-inference")
OFFICIAL_BASELINE_LEARNING_RATE = 2e-5
OFFICIAL_BASELINE_METRICS = {
    "validation_loss": 0.122829,
    "validation_accuracy": 0.963765,
    "validation_precision": 0.955407,
    "validation_recall": 0.972942,
    "validation_f1": 0.964095,
    "test_loss": 0.12363797426223755,
    "test_accuracy": 0.9628192999053926,
    "test_precision": 0.9549851190476191,
    "test_recall": 0.9714285714285714,
    "test_f1": 0.9631366663540005,
}


class TransformerSentimentPyfunc(PythonModel):
    """MLflow pyfunc adapter that delegates prediction to the existing predictor."""

    def load_context(self, context):
        self.predictor = SentimentPredictor(context.artifacts["model_dir"], device="cpu")

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        predictions = self.predictor.predict_many(model_input["review"].astype(str).tolist())
        return pd.DataFrame(predictions, columns=["label", "confidence"])


def configure_tracking(
    tracking_uri: str | None = None, artifact_root: str | Path | None = None
) -> str:
    """Configure the local SQLite tracking/registry store and local artifact root."""
    uri = tracking_uri or os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{DEFAULT_TRACKING_DB.as_posix()}"
    )
    mlflow.set_tracking_uri(uri)
    # Phase 5 uses one local store for tracking and registry metadata.
    mlflow.set_registry_uri(uri)
    selected_artifact_root = Path(artifact_root or DEFAULT_ARTIFACT_ROOT).resolve()
    selected_artifact_root.mkdir(parents=True, exist_ok=True)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=str((selected_artifact_root / EXPERIMENT_NAME).resolve()),
        )
        client.set_experiment_tag(experiment_id, "project", "Arabic Sentiment Analysis MLOps")
    mlflow.set_experiment(EXPERIMENT_NAME)
    return uri


def _git_commit() -> str:
    return current_git_revision()


def start_project_run(
    run_name: str, *, pointer_path: str | Path = DEFAULT_DVC_POINTER
):
    """Start an MLflow run with the current DVC dataset and Git lineage tags."""
    tags = mlflow_lineage_tags(pointer_path)
    run = mlflow.start_run(run_name=run_name)
    mlflow.set_tags(tags)
    return run


def _log_model(artifact_dir: str | Path):
    source_root = Path(__file__).resolve().parents[1]
    return mlflow.pyfunc.log_model(
        name=MODEL_ARTIFACT_PATH,
        python_model=TransformerSentimentPyfunc(),
        artifacts={"model_dir": str(Path(artifact_dir).resolve())},
        code_paths=[str(source_root)],
        pip_requirements=[
            "mlflow==3.16.1",
            "numpy==2.5.3",
            "pandas==3.0.6",
            "torch==2.14.1+cpu",
            "transformers==5.18.0",
            "sentencepiece==0.2.2",
        ],
        signature=ModelSignature(
            inputs=Schema([ColSpec("string", "review")]),
            outputs=Schema([ColSpec("string", "label"), ColSpec("double", "confidence")]),
        ),
    )


def _register_run(run_id: str, tags: dict[str, Any]) -> str:
    uri = f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"
    version = mlflow.register_model(uri, REGISTERED_MODEL_NAME, await_registration_for=120)
    client = MlflowClient()
    for key, value in tags.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, key, str(value))
    return str(version.version)


def log_official_gpu_baseline(
    artifact_dir: str | Path = OFFICIAL_ARTIFACT,
) -> dict[str, Any]:
    """Track and register the existing official Colab model; this function never trains."""
    configure_tracking()
    artifact_path = Path(artifact_dir).resolve()
    required = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (artifact_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Official inference artifact is incomplete: {missing}")

    client = MlflowClient()
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    for previous in client.search_runs(
        [experiment_id], filter_string="tags.run_kind = 'official_gpu_baseline'", max_results=100
    ):
        version = previous.data.tags.get("model_version")
        if previous.info.status == "FINISHED" and version:
            return {"run_id": previous.info.run_id, "model_version": version, "skipped": True}

    params = {
        "model_name": "AraBERTv0.2-base",
        "model_checkpoint": MODEL_CHECKPOINT,
        "seed": 42,
        "epochs": 1,
        "max_length": 96,
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 16,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "optimizer": "AdamW",
        "fp16": True,
        "warmup_steps": 317,
        "train_rows": 84558,
        "validation_rows": 10570,
        "test_rows": 10570,
        "training_device": "Tesla T4",
        "training_environment": "Google Colab; trained before local MLflow tracking",
        "baseline_type": "official full-dataset GPU baseline",
        "source_artifact_path": str(artifact_path),
        "training_time_minutes_approx": 11.24,
    }
    with start_project_run("official-full-dataset-gpu-baseline") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(OFFICIAL_BASELINE_METRICS)
        mlflow.set_tags(
            {
                "run_kind": "official_gpu_baseline",
                "track": "A",
                "task": "arabic_sentiment",
                "model_family": "AraBERT",
                "baseline": "true",
                "source": "colab_gpu",
                "gpu": "Tesla T4",
                "seed": "42",
                "training_performed_by_mlflow": "false",
                "git_commit": _git_commit(),
            }
        )
        mlflow.log_dict(
            {
                "training_location": "Google Colab",
                "tracked_locally": True,
                "training_performed_by_mlflow": False,
                "dataset_split_sizes": {"train": 84558, "validation": 10570, "test": 10570},
                "metrics_source": "completed official GPU baseline results provided for this project",
                "artifact_source": str(artifact_path),
                "metrics": OFFICIAL_BASELINE_METRICS,
            },
            "metadata/official_baseline.json",
        )
        model_info = _log_model(artifact_path)
        run_id = run.info.run_id

    version = _register_run(
        run_id,
        {
            "experiment_type": "official_gpu_baseline",
            "model_checkpoint": MODEL_CHECKPOINT,
            "source": "colab_gpu",
            "training_device": "Tesla T4",
            "training_sample_count": 84558,
            "validation_f1": OFFICIAL_BASELINE_METRICS["validation_f1"],
            "test_f1": OFFICIAL_BASELINE_METRICS["test_f1"],
            "cpu_constrained": False,
        },
    )
    client.set_tag(run_id, "model_version", version)
    client.set_tag(run_id, "logged_model_id", str(model_info.model_id))
    return {"run_id": run_id, "model_version": version, "skipped": False}


def _stratified_subset(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count < 2 or frame["sentiment"].nunique() < 2:
        raise ValueError("Evaluation requires at least two rows and both sentiment classes")
    fraction = count / len(frame)
    pieces = []
    remaining = count
    groups = list(frame.groupby("sentiment", sort=True))
    for index, (_, group) in enumerate(groups):
        allocated = remaining if index == len(groups) - 1 else min(
            len(group), round(len(group) * fraction)
        )
        pieces.append(group.sample(n=allocated, random_state=seed + index))
        remaining -= allocated
    sample = pd.concat(pieces).sample(frac=1, random_state=seed).reset_index(drop=True)
    if len(sample) != count:
        raise ValueError(f"Unable to create requested evaluation sample of {count} rows")
    return sample


def evaluate_artifact(
    *,
    split: str,
    sample_count: int,
    max_sequence_length: int,
    batch_size: int,
    seed: int = 42,
    artifact_dir: str | Path = OFFICIAL_ARTIFACT,
    official_run_id: str | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Run an actual deterministic, no-training evaluation of the saved model."""
    configure_tracking()
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    csv_path = Path(f"data/processed/{split}.csv")
    frame = pd.read_csv(csv_path, encoding="utf-8", usecols=["review", "sentiment"])
    sampled = _stratified_subset(frame, sample_count, seed)
    predictor = SentimentPredictor(
        artifact_dir, device="cpu", max_sequence_length=max_sequence_length
    )
    targets = sampled["sentiment"].map(LABEL_TO_ID).astype(int).to_numpy()
    reviews = sampled["review"].astype(str).tolist()

    predictions: list[int] = []
    loss_sum = 0.0
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    start = time.perf_counter()
    for offset in range(0, len(reviews), batch_size):
        batch_reviews = reviews[offset : offset + batch_size]
        batch_targets = torch.tensor(targets[offset : offset + batch_size], dtype=torch.long)
        encoded = predictor.tokenizer(
            batch_reviews,
            truncation=True,
            max_length=max_sequence_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(predictor.device) for key, value in encoded.items()}
        batch_targets = batch_targets.to(predictor.device)
        with torch.inference_mode():
            output = predictor.model(**encoded, labels=batch_targets)
        loss_sum += float(output.loss.item()) * len(batch_reviews)
        predictions.extend(output.logits.argmax(dim=-1).cpu().tolist())
    elapsed_seconds = time.perf_counter() - start

    predicted = np.asarray(predictions, dtype=np.int64)
    tp = int(np.sum((targets == 1) & (predicted == 1)))
    tn = int(np.sum((targets == 0) & (predicted == 0)))
    fp = int(np.sum((targets == 0) & (predicted == 1)))
    fn = int(np.sum((targets == 1) & (predicted == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        f"{split}_subset_loss": loss_sum / len(targets),
        f"{split}_subset_accuracy": float(np.mean(targets == predicted)),
        f"{split}_subset_precision": precision,
        f"{split}_subset_recall": recall,
        f"{split}_subset_f1": f1,
        "accuracy": float(np.mean(targets == predicted)),
        "f1": f1,
        "f1_macro": (
            f1
            + (2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0)
        )
        / 2,
        "evaluation_seconds": elapsed_seconds,
        "samples_per_second": len(targets) / elapsed_seconds if elapsed_seconds else 0.0,
    }
    sample_digest = hashlib.sha256(
        "\n".join(f"{text}\t{label}" for text, label in zip(reviews, targets)).encode("utf-8")
    ).hexdigest()
    return {
        "split": split,
        "sample_count": len(sampled),
        "available_split_rows": len(frame),
        "seed": seed,
        "max_sequence_length": max_sequence_length,
        "batch_size": batch_size,
        "training_performed": False,
        "sample_sha256": sample_digest,
        "metrics": metrics,
        "source_model_run_id": official_run_id,
        "registered_model_version": model_version,
    }


def log_evaluation_run(
    *,
    run_name: str,
    run_kind: str,
    split: str,
    sample_count: int,
    max_sequence_length: int,
    batch_size: int,
    official_run_id: str,
    model_version: str,
) -> dict[str, Any]:
    result = evaluate_artifact(
        split=split,
        sample_count=sample_count,
        max_sequence_length=max_sequence_length,
        batch_size=batch_size,
        official_run_id=official_run_id,
        model_version=model_version,
    )
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    previous = next(
        (
            run
            for run in client.search_runs([experiment.experiment_id], max_results=1000)
            if run.info.run_name == run_name
            and run.info.status == "FINISHED"
            and run.data.tags.get("run_kind") == run_kind
        ),
        None,
    )
    if previous is not None:
        # Augment an earlier run's record with the macro-F1 and model-source fields
        # from a fresh deterministic re-evaluation; never create a duplicate run.
        for key, value in result["metrics"].items():
            client.log_metric(previous.info.run_id, key, float(value))
        additions = {
            "model_name": "AraBERTv0.2-base",
            "learning_rate": str(OFFICIAL_BASELINE_LEARNING_RATE),
            "learning_rate_scope": "source model training configuration; no training in this run",
            "batch_size": str(batch_size),
        }
        for key, value in additions.items():
            if key not in previous.data.params:
                client.log_param(previous.info.run_id, key, value)
        client.log_dict(previous.info.run_id, result, "metadata/evaluation_revalidation.json")
        return {"run_id": previous.info.run_id, **result, "updated_existing_run": True}

    with start_project_run(run_name) as run:
        mlflow.log_params(
            {
                "model_name": "AraBERTv0.2-base",
                "model_checkpoint": MODEL_CHECKPOINT,
                "learning_rate": OFFICIAL_BASELINE_LEARNING_RATE,
                "learning_rate_scope": "source model training configuration; no training in this run",
                "seed": result["seed"],
                "evaluation_split": split,
                "available_split_rows": result["available_split_rows"],
                "evaluation_sample_count": result["sample_count"],
                "max_sequence_length": max_sequence_length,
                "inference_batch_size": batch_size,
                "batch_size": batch_size,
                "model_version": model_version,
                "model_run_id": official_run_id,
                "training_performed": False,
            }
        )
        mlflow.log_metrics(result["metrics"])
        mlflow.set_tags(
            {
                "run_kind": run_kind,
                "experiment_type": "deterministic_local_evaluation",
                "track": "A",
                "model_family": "AraBERT",
                "source": "local_full_gpu_inference_artifact",
                "cpu_constrained": "true",
                "training_performed": "false",
                "sample_sha256": result["sample_sha256"],
                "git_commit": _git_commit(),
            }
        )
        mlflow.log_dict(result, "metadata/evaluation_result.json")
        mlflow.log_dict(
            {
                "model_uri": f"models:/{REGISTERED_MODEL_NAME}/{model_version}",
                "source_run_id": official_run_id,
                "source_artifact_dir": str(OFFICIAL_ARTIFACT.resolve()),
            },
            "metadata/model_reference.json",
        )
        run_id = run.info.run_id
    return {"run_id": run_id, **result}


def promote_official_baseline_after_quality_gate(
    *, model_version: str, model_load_prediction: dict[str, Any]
) -> dict[str, Any]:
    """Promote the official version only after measured baseline checks pass."""
    configure_tracking()
    client = MlflowClient()
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    official = next(
        run for run in client.search_runs(
            [experiment_id], filter_string="tags.run_kind = 'official_gpu_baseline'", max_results=100
        )
        if run.info.status == "FINISHED" and run.data.tags.get("model_version") == str(model_version)
    )
    historical = next(
        run for run in client.search_runs(
            [experiment_id], filter_string="tags.run_kind = 'phase2_baseline_reference'", max_results=100
        )
        if run.info.status == "FINISHED"
    )
    registered = client.get_model_version(REGISTERED_MODEL_NAME, model_version)
    candidate = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "candidate")
    checks = {
        "registered_version_ready": registered.status == "READY",
        "registered_run_matches_official": registered.run_id == official.info.run_id,
        "candidate_alias_matches_version": str(candidate.version) == str(model_version),
        "validation_f1_at_least_historical_cpu_baseline": (
            official.data.metrics["validation_f1"]
            >= historical.data.metrics["validation_f1"]
        ),
        "test_f1_at_least_historical_cpu_baseline": (
            official.data.metrics["test_f1"] >= historical.data.metrics["test_f1"]
        ),
        "model_uri_load_prediction_valid": (
            model_load_prediction.get("label") in {"negative", "positive"}
            and 0.0 <= float(model_load_prediction.get("confidence", -1.0)) <= 1.0
        ),
    }
    passed = all(checks.values())
    if not passed:
        raise RuntimeError(f"Production quality gate failed: {checks}")

    existing_gate = next(
        (
            run
            for run in client.search_runs([experiment_id], max_results=1000)
            if run.info.status == "FINISHED"
            and run.data.tags.get("run_kind") == "production_promotion_quality_gate"
            and run.data.params.get("model_version") == str(model_version)
            and run.data.tags.get("promotion_decision") == "passed"
        ),
        None,
    )
    result = {
        "decision": "passed",
        "model_version": str(model_version),
        "official_run_id": official.info.run_id,
        "historical_reference_run_id": historical.info.run_id,
        "checks": checks,
        "official_validation_f1": official.data.metrics["validation_f1"],
        "historical_validation_f1": historical.data.metrics["validation_f1"],
        "official_test_f1": official.data.metrics["test_f1"],
        "historical_test_f1": historical.data.metrics["test_f1"],
        "model_load_prediction": model_load_prediction,
    }
    if existing_gate is None:
        with start_project_run("official-baseline-production-quality-gate") as run:
            mlflow.log_params(
                {
                    "model_name": "AraBERTv0.2-base",
                    "model_version": model_version,
                    "official_model_run_id": official.info.run_id,
                    "historical_reference_run_id": historical.info.run_id,
                    "validation_f1_threshold": historical.data.metrics["validation_f1"],
                    "test_f1_threshold": historical.data.metrics["test_f1"],
                    "training_performed": False,
                }
            )
            mlflow.log_metrics(
                {
                    "official_validation_f1": result["official_validation_f1"],
                    "historical_validation_f1_threshold": result["historical_validation_f1"],
                    "validation_f1_margin": (
                        result["official_validation_f1"] - result["historical_validation_f1"]
                    ),
                    "official_test_f1": result["official_test_f1"],
                    "historical_test_f1_threshold": result["historical_test_f1"],
                    "test_f1_margin": result["official_test_f1"] - result["historical_test_f1"],
                }
            )
            mlflow.set_tags(
                {
                    "run_kind": "production_promotion_quality_gate",
                    "promotion_decision": "passed",
                    "training_performed": "false",
                    "registered_model_name": REGISTERED_MODEL_NAME,
                    "model_version": str(model_version),
                    "git_commit": _git_commit(),
                }
            )
            mlflow.log_dict(result, "metadata/quality_gate.json")
            gate_run_id = run.info.run_id
    else:
        gate_run_id = existing_gate.info.run_id

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", str(model_version))
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, model_version, "promotion_quality_gate", "passed"
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, model_version, "promotion_gate_run_id", gate_run_id
    )
    result["quality_gate_run_id"] = gate_run_id
    result["production_alias_version"] = str(
        client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "production").version
    )
    return result


def assign_candidate_alias(model_version: str) -> str:
    """Point the candidate alias at a verified registered version; do not promote production."""
    configure_tracking()
    MlflowClient().set_registered_model_alias(
        REGISTERED_MODEL_NAME, "candidate", str(model_version)
    )
    return registered_alias_version("candidate")


def registered_alias_version(alias: str) -> str:
    configure_tracking()
    version = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
    return str(version.version)


def load_selected_pyfunc(model_uri: str):
    """Load a logged model by run, registered version, or registered-model alias URI."""
    configure_tracking()
    return mlflow.pyfunc.load_model(model_uri)


def verify_project_history(minimum_runs: int = 5) -> dict[str, Any]:
    """Audit completed runs, required official metrics, artifacts, and model traceability."""
    configure_tracking()
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs([experiment.experiment_id], max_results=1000)
    valid = [
        run for run in runs
        if run.info.status == "FINISHED"
        and run.data.params
        and run.data.metrics
        and client.list_artifacts(run.info.run_id)
    ]
    if len(valid) < minimum_runs:
        raise RuntimeError(f"Expected at least {minimum_runs} completed valid runs; found {len(valid)}")
    official = next(
        (run for run in valid if run.data.tags.get("run_kind") == "official_gpu_baseline"), None
    )
    if official is None:
        raise RuntimeError("The official full-GPU baseline run is missing")
    required_metrics = set(OFFICIAL_BASELINE_METRICS)
    missing_metrics = required_metrics - official.data.metrics.keys()
    if missing_metrics:
        raise RuntimeError(f"Official run missing metrics: {sorted(missing_metrics)}")
    version = official.data.tags.get("model_version")
    if not version:
        raise RuntimeError("Official baseline is not registered")
    registered = client.get_model_version(REGISTERED_MODEL_NAME, version)
    if registered.run_id != official.info.run_id:
        raise RuntimeError("Registry version does not resolve to the official baseline run")
    production = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "production")
    if str(production.version) != str(version):
        raise RuntimeError("Production alias does not resolve to the quality-gated official version")
    gate = next(
        (
            run for run in valid
            if run.data.tags.get("run_kind") == "production_promotion_quality_gate"
            and run.data.tags.get("promotion_decision") == "passed"
            and run.data.tags.get("model_version") == str(version)
        ),
        None,
    )
    if gate is None:
        raise RuntimeError("A passing quality-gate run for the production version is missing")
    return {
        "experiment_name": experiment.name,
        "valid_run_count": len(valid),
        "valid_run_ids": [run.info.run_id for run in valid],
        "official_run_id": official.info.run_id,
        "official_model_version": version,
        "candidate_alias_version": str(
            client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "candidate").version
        ),
        "production_alias_set": True,
        "production_alias_version": str(production.version),
        "quality_gate_run_id": gate.info.run_id,
    }
