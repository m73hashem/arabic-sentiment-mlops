"""FastAPI application for local Arabic sentiment prediction."""

from time import perf_counter
from typing import Literal, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, field_validator

from .monitoring import (
    PREDICTION_CONFIDENCE,
    PREDICTION_DURATION,
    PREDICTION_REQUESTS,
    prometheus_payload,
)
from .serving import create_prediction_service


class PredictionService(Protocol):
    """Model service contract consumed by the HTTP layer."""

    model_version: str

    def predict(self, review: str) -> dict[str, str | float]: ...


class PredictionRequest(BaseModel):
    review: str = Field(min_length=1, description="Arabic review text to classify")

    @field_validator("review")
    @classmethod
    def review_must_contain_non_whitespace_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("review must contain non-whitespace text")
        return value


class PredictionResponse(BaseModel):
    label: Literal["negative", "positive"]
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str = Field(min_length=1)


class HealthResponse(BaseModel):
    status: Literal["ok"]


def create_app(service: PredictionService | None = None) -> FastAPI:
    """Create the API, resolving its model service only when prediction is requested."""
    prediction_service = service
    application = FastAPI(
        title="Arabic Sentiment Analysis API",
        version="1.0.0",
        description="Local AraBERT sentiment prediction for Arabic reviews.",
    )

    @application.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        payload, content_type = prometheus_payload()
        return Response(content=payload, media_type=content_type)

    @application.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest) -> PredictionResponse:
        nonlocal prediction_service
        started_at = perf_counter()
        try:
            if prediction_service is None:
                prediction_service = create_prediction_service()
            result = prediction_service.predict(request.review)
            model_version = prediction_service.model_version
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            PREDICTION_REQUESTS.labels(status="unavailable").inc()
            PREDICTION_DURATION.labels(status="unavailable").observe(
                perf_counter() - started_at
            )
            raise HTTPException(
                status_code=503, detail="Sentiment model is unavailable"
            ) from exc
        response = PredictionResponse(
            label=result["label"],
            confidence=result["confidence"],
            model_version=model_version,
        )
        PREDICTION_REQUESTS.labels(status="success").inc()
        PREDICTION_CONFIDENCE.observe(response.confidence)
        PREDICTION_DURATION.labels(status="success").observe(
            perf_counter() - started_at
        )
        return response

    return application


app = create_app()


def main() -> None:
    uvicorn.run("arabic_sentiment.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
