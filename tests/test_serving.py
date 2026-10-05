import json

import pandas as pd
import pytest

from arabic_sentiment import serving
from arabic_sentiment.modeling import MODEL_CHECKPOINT


def test_local_service_reads_finetuned_version_and_loads_predictor_lazily(tmp_path, monkeypatch):
    (tmp_path / "training_config.json").write_text(
        json.dumps({"model_checkpoint": MODEL_CHECKPOINT, "seed": 42}), encoding="utf-8"
    )
    calls = []

    class Predictor:
        def predict(self, review):
            calls.append(review)
            return {"label": "positive", "confidence": 0.9}

    constructions = []

    def load_predictor(path, device):
        constructions.append((path, device))
        return Predictor()

    monkeypatch.setattr(serving, "SentimentPredictor", load_predictor)
    service = serving.LocalAraBERTService(tmp_path)
    assert service.model_version == f"{MODEL_CHECKPOINT}-finetuned-seed-42"
    assert constructions == []
    assert service.predict("جيد") == {"label": "positive", "confidence": 0.9}
    assert service.predict("ممتاز") == {"label": "positive", "confidence": 0.9}
    assert constructions == [(tmp_path, "cpu")]
    assert calls == ["جيد", "ممتاز"]


def test_local_service_identifies_official_inference_export_and_rejects_unknown_artifact(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    service = serving.LocalAraBERTService(tmp_path)
    assert service.model_version == f"{MODEL_CHECKPOINT}-official-full-gpu-seed-42"

    with pytest.raises(FileNotFoundError, match="Model artifact metadata not found"):
        serving.LocalAraBERTService(tmp_path / "missing")


def test_mlflow_service_resolves_version_and_alias_without_loading_until_predict(monkeypatch):
    exact = serving.MlflowModelService("models:/arabic-sentiment/7")
    assert exact.model_version == "7"
    assert exact._model is None

    monkeypatch.setattr("arabic_sentiment.mlflow_tracking.registered_alias_version", lambda alias: "11")
    alias = serving.MlflowModelService("models:/arabic-sentiment@candidate")
    assert alias.model_version == "11"
    with pytest.raises(ValueError, match="must not be empty"):
        serving.MlflowModelService("")


def test_mlflow_service_loads_selected_model_and_converts_prediction(monkeypatch):
    class Model:
        def predict(self, frame):
            assert frame.to_dict("records") == [{"review": "نص عربي"}]
            return pd.DataFrame([{"label": "negative", "confidence": 0.75}])

    loaded = []
    monkeypatch.setattr(
        "arabic_sentiment.mlflow_tracking.load_selected_pyfunc",
        lambda uri: loaded.append(uri) or Model(),
    )
    service = serving.MlflowModelService("runs:/example/model")
    assert service.predict("نص عربي") == {"label": "negative", "confidence": 0.75}
    assert service.predict("نص عربي") == {"label": "negative", "confidence": 0.75}
    assert loaded == ["runs:/example/model"]


def test_factory_uses_configured_registry_uri_or_local_fallback(monkeypatch):
    monkeypatch.setenv("SENTIMENT_MODEL_URI", "models:/arabic-sentiment/3")
    selected = serving.create_prediction_service()
    assert isinstance(selected, serving.MlflowModelService)
    assert selected.model_uri == "models:/arabic-sentiment/3"

    monkeypatch.delenv("SENTIMENT_MODEL_URI")
    local = object()
    monkeypatch.setattr(serving, "LocalAraBERTService", lambda: local)
    assert serving.create_prediction_service() is local
