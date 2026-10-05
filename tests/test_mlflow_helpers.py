import json

import pandas as pd
import pytest
from mlflow import MlflowClient

from arabic_sentiment import mlflow_experiments
from arabic_sentiment.mlflow_tracking import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TRACKING_DB,
    EXPERIMENT_NAME,
    _stratified_subset,
    configure_tracking,
    log_official_gpu_baseline,
    verify_project_history,
)


def test_tracking_configuration_reuses_named_experiment(tmp_path):
    tracking_uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    artifact_root = tmp_path / "artifacts"
    assert configure_tracking(tracking_uri, artifact_root) == tracking_uri
    first = MlflowClient(tracking_uri).get_experiment_by_name(EXPERIMENT_NAME)
    configure_tracking(tracking_uri, artifact_root)
    second = MlflowClient(tracking_uri).get_experiment_by_name(EXPERIMENT_NAME)
    assert first.experiment_id == second.experiment_id
    assert artifact_root.is_dir()
    configure_tracking(
        f"sqlite:///{DEFAULT_TRACKING_DB.as_posix()}", DEFAULT_ARTIFACT_ROOT
    )


def test_official_baseline_requires_complete_existing_artifact(tmp_path, monkeypatch):
    tracking_uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    with pytest.raises(FileNotFoundError, match="inference artifact is incomplete"):
        log_official_gpu_baseline(tmp_path / "missing-model")
    assert (
        MlflowClient(tracking_uri).get_experiment_by_name(EXPERIMENT_NAME) is not None
    )


def test_stratified_subset_is_reproducible_balanced_and_rejects_invalid_counts():
    frame = pd.DataFrame(
        {
            "review": [f"نص {index}" for index in range(20)],
            "sentiment": ["negative"] * 10 + ["positive"] * 10,
        }
    )
    first = _stratified_subset(frame, 8, 42)
    second = _stratified_subset(frame, 8, 42)
    assert first.to_dict("records") == second.to_dict("records")
    assert first.sentiment.value_counts().to_dict() == {"negative": 4, "positive": 4}
    with pytest.raises(ValueError, match="at least two rows"):
        _stratified_subset(frame, 1, 42)
    with pytest.raises(ValueError, match="both sentiment classes"):
        _stratified_subset(frame.assign(sentiment="positive"), 5, 42)


def test_history_audit_rejects_empty_experiment(tmp_path):
    tracking_uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    configure_tracking(tracking_uri, tmp_path / "artifacts")
    with pytest.raises(RuntimeError, match="at least 5 completed valid runs"):
        verify_project_history()
    configure_tracking(
        f"sqlite:///{DEFAULT_TRACKING_DB.as_posix()}", DEFAULT_ARTIFACT_ROOT
    )


def test_mlflow_cli_verify_only_skips_experiment_creation_and_evaluation(
    monkeypatch, capsys
):
    monkeypatch.setattr("sys.argv", ["mlflow-experiments", "--verify-only"])
    monkeypatch.setattr(
        mlflow_experiments, "verify_project_history", lambda: {"valid_run_count": 5}
    )
    monkeypatch.setattr(
        mlflow_experiments,
        "log_official_gpu_baseline",
        lambda *_: pytest.fail("verify-only must not register a baseline"),
    )

    mlflow_experiments.main()

    assert json.loads(capsys.readouterr().out) == {"valid_run_count": 5}


def test_mlflow_cli_orchestrates_configured_tracking_and_quality_gate(
    tmp_path, monkeypatch, capsys
):
    config_path = tmp_path / "experiments.json"
    config_path.write_text(
        json.dumps({"official_artifact_dir": "local-artifact", "evaluations": []}),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        "sys.argv", ["mlflow-experiments", "--config", str(config_path)]
    )
    monkeypatch.setattr(
        mlflow_experiments,
        "log_official_gpu_baseline",
        lambda path: calls.append(("baseline", path))
        or {"run_id": "run-a", "model_version": "3"},
    )
    monkeypatch.setattr(
        mlflow_experiments,
        "assign_candidate_alias",
        lambda version: calls.append(("candidate", version)) or version,
    )

    class Service:
        def __init__(self, uri):
            calls.append(("service", uri))

        def predict(self, review):
            calls.append(("predict", review))
            return {"label": "positive", "confidence": 0.9}

    monkeypatch.setattr(mlflow_experiments, "MlflowModelService", Service)
    monkeypatch.setattr(
        mlflow_experiments,
        "promote_official_baseline_after_quality_gate",
        lambda **kwargs: calls.append(("promotion", kwargs["model_version"]))
        or {"decision": "passed"},
    )
    monkeypatch.setattr(
        mlflow_experiments,
        "verify_project_history",
        lambda: {"valid_run_count": 5},
    )

    mlflow_experiments.main()

    assert calls == [
        ("baseline", "local-artifact"),
        ("candidate", "3"),
        ("service", "models:/arabic-sentiment@candidate"),
        ("predict", "الخدمة ممتازة والغرفة نظيفة ومريحة"),
        ("promotion", "3"),
    ]
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result["candidate_alias_version"] == "3"
    assert result["production_promotion"] == {"decision": "passed"}
