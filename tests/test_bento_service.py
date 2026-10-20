import asyncio
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import httpx
import pytest
from bentoml.exceptions import (
    InternalServerError,
    ServiceUnavailable,
    UnprocessableEntity,
)

from arabic_sentiment import serving as shared_serving

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
bento_service = import_module("serving.service")


class PredictorBoundary:
    model_version = "test-arabert-version"

    def __init__(self):
        self.single_reviews = []
        self.batch_reviews = []

    def predict(self, review):
        self.single_reviews.append(review)
        return {"label": "positive", "confidence": 0.875}

    def predict_many(self, reviews):
        self.batch_reviews.append(list(reviews))
        return [
            {"label": label, "confidence": confidence}
            for label, confidence in zip(
                ("positive", "negative", "positive"), (0.875, 0.625, 0.75)
            )
        ][: len(reviews)]


def test_service_import_is_safe_without_model_artifact(tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        str(path) for path in (PROJECT_ROOT / "src", PROJECT_ROOT)
    )
    environment["SENTIMENT_MODEL_DIR"] = str(tmp_path / "missing-model")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import serving.service; print('bento service imported')",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "bento service imported" in result.stdout


def test_mounted_api_health_and_prediction_preserve_contract(monkeypatch):
    async def exercise_routes():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=bento_service.http_app),
            base_url="http://testserver",
        ) as client:
            health = await client.get("/health")
            response = await client.post("/predict", json={"review": "نص عربي"})
            invalid = await client.post("/predict", json={"review": "  "})
        return health, response, invalid

    predictor = PredictorBoundary()
    monkeypatch.setattr(bento_service.prediction_service, "_delegate", predictor)
    health, response, invalid = asyncio.run(exercise_routes())

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert response.status_code == 200
    assert response.json() == {
        "label": "positive",
        "confidence": 0.875,
        "model_version": "test-arabert-version",
    }
    assert invalid.status_code == 422
    assert predictor.single_reviews == ["نص عربي"]


def test_batch_api_uses_shared_batch_predictor_and_preserves_order(monkeypatch):
    predictor = PredictorBoundary()
    monkeypatch.setattr(bento_service.prediction_service, "_delegate", predictor)
    service = bento_service.SentimentService()

    results = service.predict_batch.local(["first", "second", "third"])

    assert predictor.batch_reviews == [["first", "second", "third"]]
    assert [result.label for result in results] == [
        "positive",
        "negative",
        "positive",
    ]
    assert [result.confidence for result in results] == [0.875, 0.625, 0.75]
    assert [result.model_version for result in results] == [
        "test-arabert-version",
    ] * 3
    assert len(results) == 3


def test_local_serving_batch_delegates_to_the_shared_predictor(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    constructions = []

    class Predictor:
        def predict_many(self, reviews):
            return [{"label": "positive", "confidence": 0.9} for _ in reviews]

    def make_predictor(path, device):
        constructions.append((path, device))
        return Predictor()

    monkeypatch.setattr(shared_serving, "SentimentPredictor", make_predictor)
    service = shared_serving.LocalAraBERTService(tmp_path)

    results = service.predict_many(["first", "second"])

    assert len(results) == 2
    assert constructions == [(tmp_path, "cpu")]


def test_batch_api_validates_nonblank_reviews(monkeypatch):
    service = bento_service.SentimentService()
    monkeypatch.setattr(
        bento_service.prediction_service,
        "_delegate",
        PredictorBoundary(),
    )

    with pytest.raises(UnprocessableEntity) as error:
        service.predict_batch.local(["valid", "  "])

    assert error.value.error_code == 422


def test_batch_api_maps_missing_model_to_service_unavailable(monkeypatch):
    class MissingArtifact:
        @property
        def model_version(self):
            raise FileNotFoundError("missing model")

        def predict_many(self, reviews):
            raise FileNotFoundError("missing model")

    monkeypatch.setattr(
        bento_service.prediction_service, "_delegate", MissingArtifact()
    )
    service = bento_service.SentimentService()

    with pytest.raises(ServiceUnavailable) as error:
        service.predict_batch.local(["review"])

    assert error.value.error_code == 503


def test_batch_api_rejects_unexpected_prediction_count(monkeypatch):
    class WrongCount:
        model_version = "test-arabert-version"

        def predict_many(self, reviews):
            return []

    monkeypatch.setattr(bento_service.prediction_service, "_delegate", WrongCount())
    service = bento_service.SentimentService()

    with pytest.raises(InternalServerError):
        service.predict_batch.local(["review"])


def test_batch_api_declares_conservative_adaptive_batching():
    api = bento_service.SentimentService.apis["predict_batch"]
    assert api.batchable is True
    assert api.max_batch_size == bento_service.MAX_BATCH_SIZE == 8
    assert api.max_latency_ms == bento_service.MAX_BATCH_LATENCY_MS == 50
    assert api.route == "/predict/batch"
    bento_app = bento_service.SentimentService.to_asgi()
    registered_routes = {
        route.path: getattr(route, "methods", None)
        for route in bento_app.routes
        if getattr(route, "path", None) is not None
    }
    assert registered_routes["/predict/batch"] == {"POST"}
    mounted_paths = {
        (route.path, method)
        for app, path, _ in bento_service.SentimentService.mount_apps
        if path == "/"
        for route in app.routes
        for method in getattr(route, "methods", set())
    }
    assert ("/health", "GET") in mounted_paths
    assert ("/predict", "POST") in mounted_paths
