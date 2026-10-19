"""Equivalent single-review workload for FastAPI and BentoML ``/predict``."""

from __future__ import annotations

import itertools
import math
import os

from locust import HttpUser, constant, task

from benchmarks.reviews import MODEL_VERSION, REVIEWS

EXPECTED_MODEL_VERSION = os.environ.get("LOCUST_EXPECTED_MODEL_VERSION", MODEL_VERSION)
_user_slots = itertools.count()


class SentimentPredictionUser(HttpUser):
    """Issue a closed-loop stream of the same fixed review corpus per user."""

    wait_time = constant(0)

    def on_start(self) -> None:
        self._review_index = next(_user_slots) % len(REVIEWS)

    @task
    def predict_review(self) -> None:
        review = REVIEWS[self._review_index % len(REVIEWS)]
        self._review_index += 1
        with self.client.post(
            "/predict",
            json={"review": review},
            name="/predict",
            timeout=120,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"unexpected HTTP {response.status_code}")
                return
            try:
                payload = response.json()
                label = payload["label"]
                confidence = float(payload["confidence"])
                model_version = payload["model_version"]
            except (ValueError, TypeError, KeyError):
                response.failure("invalid prediction response schema")
                return
            if label not in {"negative", "positive"}:
                response.failure("invalid sentiment label")
            elif not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                response.failure("invalid confidence value")
            elif model_version != EXPECTED_MODEL_VERSION:
                response.failure("unexpected model version")
