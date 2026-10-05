# Project Status

Requirements remain unchecked until implemented and tested.

Overall checklist progress: 31 of 76 items complete (40.8%).

## Phase 0 — Project Foundation

- [x] Git repository
- [x] Project structure
- [x] README
- [x] Project status tracking

## Phase 1 — Data

- [x] Raw data management
- [x] Data parsing
- [x] Preprocessing
- [x] Deterministic train/validation/test split
- [x] Data tests

## Phase 2 — Deep Learning

- [x] Transformer baseline
- [x] Training
- [x] Evaluation
- [x] Reproducibility
- [x] Model artifact

## Phase 3 — API

- [x] FastAPI `/predict`
- [x] FastAPI `/health`
- [x] Confidence
- [x] Label
- [x] Model version
- [x] API tests

## Phase 4 — Docker

- [x] Dockerfile
- [x] Image build
- [x] Container execution
- [x] Endpoint validation

## Phase 5 — Experiment Tracking

- [x] Local MLflow tracking and `arabic-sentiment` experiment
- [x] At least 5 completed, meaningful runs with actual parameters, metrics, and artifacts
- [x] Official full-GPU baseline logged and registered with run-to-version traceability
- [x] Deterministic CPU-only evaluation runs with reproducible sample hashes
- [x] Model Registry candidate alias and serving selection abstraction

The `candidate` and `production` aliases point to the official baseline after the logged quality gate passed against the historical CPU baseline metrics and the registry model load/prediction check. This registry selection does not claim a live production deployment.

## Phase 6 — Data Versioning and Reproducibility

- [x] DVC
- [x] Reproducible data pipeline

Phase 6 implementation steps 6.1–6.7 are complete (7/7). The raw dataset is tracked by DVC and its MinIO remote object is present. The prepare/train/evaluate pipeline is up to date locally; the final remote-cache check showed its generated processed splits, lightweight model, and metrics are not uploaded to MinIO.

## Phase 7 — CI/CD

- [ ] GitHub Actions
- [ ] Tests
- [ ] Linting
- [ ] Model quality gate

## Phase 8 — Production Serving

- [ ] BentoML
- [ ] Production serving validation

## Phase 9 — Load Testing

- [ ] Locust
- [ ] p50
- [ ] p95
- [ ] p99
- [ ] RPS
- [ ] Failure rate
- [ ] Saturation point

## Phase 9 — Deployment Safety

- [ ] Canary deployment
- [ ] Rollback

## Phase 10 — Optimization

- [ ] Baseline model
- [ ] 6-layer student/distillation
- [ ] ONNX
- [ ] INT8 quantization
- [ ] Benchmark comparison

## Phase 11 — Monitoring

- [ ] Prometheus
- [ ] Grafana
- [ ] Latency
- [ ] Errors
- [ ] Model version
- [ ] Prediction distribution
- [ ] Vocabulary drift / PSI

## Phase 12 — Retraining

- [ ] Retraining pipeline
- [ ] Evaluation
- [ ] Quality gate
- [ ] Model promotion
- [ ] Human approval
- [ ] Rollback

## Phase 13 — Final Submission

- [ ] README
- [ ] Architecture diagram
- [ ] `reports/module-1.md`
- [ ] `reports/module-2.md`
- [ ] `reports/module-3.md`
- [ ] `reports/module-4.md`
- [ ] `reports/module-5.md`
- [ ] Tests
- [ ] Docker
- [ ] CI/CD
- [ ] Peer review
- [ ] Final audit
