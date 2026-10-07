import os
import subprocess
import sys
import unittest
from pathlib import Path

import httpx

from arabic_sentiment.api import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_GPU_ARTIFACT = PROJECT_ROOT / "models/full-gpu-arabert-inference"
BASELINE_ARTIFACT = PROJECT_ROOT / "models/baseline-arabert"


def _available_default_artifact() -> Path | None:
    configured_path = os.environ.get("SENTIMENT_MODEL_DIR")
    candidates = (
        (Path(configured_path),)
        if configured_path
        else (FULL_GPU_ARTIFACT, BASELINE_ARTIFACT)
    )
    for path in candidates:
        if (path / "config.json").is_file() and (path / "model.safetensors").is_file():
            return path
    return None


class FakePredictionService:
    model_version = "test-arabert-seed-42"

    def predict(self, review: str) -> dict[str, str | float]:
        self.last_review = review
        return {"label": "positive", "confidence": 0.875}


class UnavailablePredictionService:
    model_version = "unavailable-model"

    def predict(self, review: str) -> dict[str, str | float]:
        raise FileNotFoundError("local model artifact is missing")


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = FakePredictionService()
        transport = httpx.ASGITransport(app=create_app(self.service))
        self.client = httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_health_success(self):
        response = await self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    async def test_predict_success_and_response_fields(self):
        review = "الخدمة ممتازة والغرفة نظيفة"
        response = await self.client.post("/predict", json={"review": review})
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["label"], "positive")
        self.assertIn(body["label"], {"negative", "positive"})
        self.assertGreaterEqual(body["confidence"], 0.0)
        self.assertLessEqual(body["confidence"], 1.0)
        self.assertEqual(body["model_version"], self.service.model_version)
        self.assertEqual(self.service.last_review, review)

    async def test_predict_rejects_missing_empty_and_whitespace_review(self):
        for payload in ({}, {"review": ""}, {"review": "  \n\t"}):
            with self.subTest(payload=payload):
                response = await self.client.post("/predict", json=payload)
                self.assertEqual(response.status_code, 422)

    async def test_predict_returns_service_unavailable_when_model_artifact_is_missing(
        self,
    ):
        transport = httpx.ASGITransport(app=create_app(UnavailablePredictionService()))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            response = await client.post("/predict", json={"review": "نص عربي"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "Sentiment model is unavailable"})


def test_api_import_and_health_work_without_local_model_artifact(tmp_path):
    source_path = str(PROJECT_ROOT / "src")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (source_path, environment.get("PYTHONPATH", "")) if path
    )
    environment.pop("SENTIMENT_MODEL_DIR", None)
    environment.pop("SENTIMENT_MODEL_URI", None)
    script = """
import asyncio
import httpx
from arabic_sentiment.api import app

async def main():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://testserver'
    ) as client:
        health = await client.get('/health')
        prediction = await client.post('/predict', json={'review': 'نص عربي'})
    assert health.status_code == 200, health.text
    assert health.json() == {'status': 'ok'}
    assert prediction.status_code == 503, prediction.text

asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@unittest.skipIf(
    _available_default_artifact() is None,
    "local AraBERT inference artifact is unavailable",
)
class LocalArtifactApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_predict_route_uses_available_local_arabert_artifact(self):
        app = create_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/predict", json={"review": "الخدمة ممتازة والغرفة نظيفة ومريحة"}
            )

        body = response.json()
        self.assertEqual(response.status_code, 200, body)
        self.assertIn(body["label"], {"negative", "positive"})
        self.assertGreaterEqual(body["confidence"], 0.0)
        self.assertLessEqual(body["confidence"], 1.0)
        self.assertTrue(body["model_version"])


if __name__ == "__main__":
    unittest.main()
