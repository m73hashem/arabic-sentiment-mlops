# Module 4 — Docker

## Dockerfile design

`Dockerfile` uses the official `python:3.13.16-slim-bookworm` base, matching the project’s Python 3.13 environment. It installs the pinned API and inference dependencies from `requirements-api.txt`, including `torch==2.14.1+cpu` from the PyTorch CPU wheel index. The image copies the application source, runs Uvicorn on `0.0.0.0:8000`, exposes port 8000, and runs as the non-root `app` user. Build arguments `APP_UID` and `APP_GID` let the local image user match the owner of the mounted model files.

The separate runtime dependency file omits training and data packages unused by the API, including pandas, PyArrow, SciPy, and scikit-learn. The final image measured 1,712,570,581 bytes (about 1.71 GB decimal), largely due to the CPU PyTorch inference runtime.

## Model delivery

The 542 MB baseline artifact is not copied into the image and is not committed to Git. `.dockerignore` excludes both `models/` and `data/`, as well as `.git`, virtual environments, tests, reports, notebooks, configs, logs, temporary files, and secrets. The final Docker build context was 2.23 kB.

For local validation, the existing artifact directory was mounted read-only at `/models/baseline-arabert`, and `SENTIMENT_MODEL_DIR` pointed the `LocalAraBERTService` adapter to that path. `serving.py` now reads that environment variable and retains the existing `models/baseline-arabert` default outside Docker. The image itself contains no model files. The local weight file is owner-readable, so `APP_UID`/`APP_GID` were set to the host user and group during the build; the process remained non-root. The prediction adapter and `PredictionService` contract keep the HTTP routes independent of artifact delivery, allowing a later registry-backed service to replace the local adapter without changing route logic.

## Build and container validation

Build command:

```bash
docker build \
  --build-arg APP_UID="$(id -u)" \
  --build-arg APP_GID="$(id -g)" \
  -t arabic-sentiment-mlops:phase4 .
```

Result: **PASS**. The image built successfully as `arabic-sentiment-mlops:phase4` using the Python 3.13.16 slim Bookworm base. Its build context excluded the model and datasets.

Run command:

```bash
docker run -d --name arabic-sentiment-api-phase4 \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/models/baseline-arabert:/models/baseline-arabert:ro" \
  -e SENTIMENT_MODEL_DIR=/models/baseline-arabert \
  arabic-sentiment-mlops:phase4
```

The container reported `running`, used the non-root `app` user, and served on port 8000. Docker inspection confirmed the host artifact mount was read-only. After endpoint verification, the temporary validation container was stopped and removed; the built image remains local.

Actual endpoint responses from the container:

```json
{"status":"ok"}
```

```json
{"label":"positive","confidence":0.9891707301139832,"model_version":"aubmindlab/bert-base-arabertv02-finetuned-seed-42"}
```

The Arabic prediction request was `الخدمة ممتازة والغرفة نظيفة ومريحة`. Container logs showed the request completed with HTTP 200 and AraBERT weights loading completed for all 201 tensors. A container-side check confirmed `cuda_available=False`, that the mounted weights were readable, and that pandas and PyArrow were absent. This verifies the real local AraBERT artifact produced the prediction.

## Tests and phase boundary

The complete test suite command was:

```text
PYTHONPATH=src HF_HUB_OFFLINE=1 .venv/bin/pytest -q
```

Result: **14 passed, 3 subtests passed** in 18.82 seconds. No model training or baseline metric changes occurred.

Docker is the only phase component implemented here. MLflow, DVC, CI/CD, BentoML, Locust, canary deployment, rollback, optimization/distillation, ONNX/INT8, Prometheus, Grafana, drift detection, and retraining remain out of scope.

## Limitations

- The image remains about 1.71 GB because CPU PyTorch and transformer inference libraries are large, even though unneeded data and training packages were excluded.
- A host artifact mounted with restrictive permissions must be readable by the container user. The documented local build aligns the non-root UID/GID with the host file owner.
- The validated delivery path is a local read-only artifact mount. Registry download, model promotion, production orchestration, and serving concurrency are later-phase work.
