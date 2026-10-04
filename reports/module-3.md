# Module 3 — FastAPI API

## Implementation

The Phase 3 API is implemented in `src/arabic_sentiment/api.py`. It defines typed Pydantic request and response models and exposes two routes. `src/arabic_sentiment/serving.py` adapts the HTTP layer to the existing `SentimentPredictor` in `inference.py`; the endpoint depends on a prediction-service interface, so another model provider can be injected without changing route behavior. The local adapter loads the saved artifact on its first prediction, uses CPU, and the predictor loads tokenizer/model files locally.

The saved artifact path is `models/baseline-arabert/`. No training or model-artifact changes were made in Phase 3.

## Endpoints and schema

| Method and path | Request | Response |
|---|---|---|
| `GET /health` | No body | `{"status":"ok"}` |
| `POST /predict` | `{"review":"Arabic review text"}` | `{"label":"negative" or "positive","confidence":0..1,"model_version":"..."}` |

`review` must be a non-empty string containing non-whitespace text. Missing, empty, or whitespace-only review input receives HTTP 422. The response schema restricts labels to `negative` and `positive`, confidence to the inclusive range 0–1, and model version to a non-empty string.

## Model version

The adapter reads `model_checkpoint` and `seed` from the artifact's `training_config.json`. It reports the version as `{model_checkpoint}-finetuned-seed-{seed}`. For the existing local artifact this is `aubmindlab/bert-base-arabertv02-finetuned-seed-42`. This identifies the baseline configuration; it is not a content hash of the weights.

## Verification

The complete test suite was run with:

```text
PYTHONPATH=src HF_HUB_OFFLINE=1 .venv/bin/python -m unittest discover -s tests -v
```

Result: **14 tests passed** (3 API tests, 6 Phase 1 data tests, and 5 Phase 2 model tests).

The running Uvicorn application was also queried over localhost. Its actual responses were:

```json
{"status":"ok"}
```

```json
{"label":"positive","confidence":0.9891707301139832,"model_version":"aubmindlab/bert-base-arabertv02-finetuned-seed-42"}
```

The prediction caused the local model weights to load (201/201 tensors reported by Transformers), and the response came from the saved model. Inference was forced to CPU. No retraining occurred, and Phase 2 metrics were not changed.

## Limitations and phase boundary

- The API serves a local CPU artifact and has not been containerized or deployed.
- The model version describes the checkpoint and training seed in metadata; repeated artifacts trained with the same values share that version string.
- CPU inference executes inline in the async prediction handler, so a prediction occupies the server event loop until that inference finishes. This is adequate for the local Phase 3 check but will need a serving/concurrency design for higher request volume.
- Docker, MLflow, DVC, BentoML, Locust, monitoring, optimization/distillation/ONNX, canary deployment, and rollback are not part of Phase 3 and were not implemented.
