"""BentoML wrapper for the shared Arabic sentiment prediction service."""

from __future__ import annotations

from typing import Annotated, Any

import bentoml
from bentoml.exceptions import (
    InternalServerError,
    ServiceUnavailable,
    UnprocessableEntity,
)
from fastapi import FastAPI
from pydantic import Field, ValidationError

from arabic_sentiment.api import (
    PredictionRequest,
    PredictionResponse,
    create_app,
)
from arabic_sentiment.serving import LocalAraBERTService

MAX_BATCH_SIZE = 8
MAX_BATCH_LATENCY_MS = 50
# Initial conservative limits only; tune them using Locust measurements later.


class LazyAraBERTService:
    """Resolve the configured local model only when prediction is requested."""

    def __init__(self) -> None:
        self._delegate: LocalAraBERTService | None = None

    def _get_delegate(self) -> LocalAraBERTService:
        if self._delegate is None:
            self._delegate = LocalAraBERTService()
        return self._delegate

    @property
    def model_version(self) -> str:
        return self._get_delegate().model_version

    def predict(self, review: str) -> dict[str, Any]:
        return self._get_delegate().predict(review)

    def predict_many(self, reviews: list[str]) -> list[dict[str, Any]]:
        return self._get_delegate().predict_many(reviews)


prediction_service = LazyAraBERTService()

# Mount the existing FastAPI contract unchanged. Its health route stays available
# without model files because the shared prediction provider is lazy.
http_app: FastAPI = create_app(prediction_service)


@bentoml.service(name="arabic-sentiment")
@bentoml.asgi_app(http_app, path="/")
class SentimentService:
    """Serve the existing HTTP API and an adaptive-batched inference endpoint."""

    @bentoml.api(
        route="/predict/batch",
        batchable=True,
        max_batch_size=MAX_BATCH_SIZE,
        max_latency_ms=MAX_BATCH_LATENCY_MS,
    )
    def predict_batch(
        self, reviews: Annotated[list[str], Field(min_length=1)], /
    ) -> list[PredictionResponse]:
        """Return predictions in input order for the submitted Arabic reviews."""
        try:
            validated_reviews = [
                PredictionRequest(review=review).review for review in reviews
            ]
        except ValidationError as exc:
            raise UnprocessableEntity(
                "Each review must contain non-whitespace text"
            ) from exc

        try:
            predictions = prediction_service.predict_many(validated_reviews)
            model_version = prediction_service.model_version
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            raise ServiceUnavailable("Sentiment model is unavailable") from exc

        if len(predictions) != len(validated_reviews):
            raise InternalServerError(
                "The prediction service returned an unexpected result count"
            )

        return [
            PredictionResponse(
                label=prediction["label"],
                confidence=prediction["confidence"],
                model_version=model_version,
            )
            for prediction in predictions
        ]
