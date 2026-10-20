# Module 7 — CI model quality gate

## Metric source and model scope

GitHub Actions reads the tracked candidate snapshot at `metrics/dvc-reproducibility.json`. The DVC `evaluate` stage writes generated metrics to the ignored `metrics/generated/dvc-reproducibility.json`; after evaluation, the tracked candidate snapshot is refreshed from that output when preparing a candidate. The evaluator reports `f1_macro` for both `negative` and `positive` labels, with `zero_division=0`. CI does not reproduce the pipeline or train AraBERT: DVC raw data and model weights are not available in a clean GitHub checkout, and the DVC pipeline is only the lightweight TF-IDF + Logistic Regression reproducibility demonstration.

The full-dataset official AraBERT baseline does not have a tracked full-split `f1_macro` result. Its report and MLflow metadata contain full-split positive-class F1, while macro-F1 in MLflow is available only for deterministic subsets; the local MLflow database is not part of a clean checkout. Those values are not silently treated as equivalent. Instead, `configs/quality-gate.json` records the measured Version 1 DVC reproducibility test macro-F1 as a frozen regression baseline.

The frozen Version 1 baseline is **0.9392593984962405** on the DVC test split. Its measured JSON is preserved in `metrics/dvc-reproducibility-v1-baseline.json`, and `configs/quality-gate.json` continues to use that value without adjustment. After the approved Version 2 curation, the DVC-generated candidate test macro-F1 was **0.9385747787247837**. The quality-gate command therefore exits 1 with a below-baseline result; equality still passes. This regression remains visible and Docker publishing is blocked. Tests use separate temporary fixture values and do not represent project metrics.

The DVC stage owns only the ignored generated metrics path, declared with `cache: false`. The separate tracked candidate snapshot remains available to clean-checkout CI without conflicting with DVC output ownership.

## Gate and publishing dependency

Run the same fail-closed comparison locally:

```bash
PYTHONPATH=src python -m arabic_sentiment.quality_gate \
  --candidate metrics/dvc-reproducibility.json \
  --baseline configs/quality-gate.json
```

The gate rejects missing files/fields, malformed JSON, unsupported schemas, non-numeric or non-finite scores, and values outside `[0, 1]`. GitHub Actions runs lint, tests/coverage, and the Docker build as before; the quality-gate job runs after tests. Docker Hub publishing requires both the build and gate jobs to succeed. Pull requests never run the login/push job.

This is explicitly a regression gate for the DVC reproducibility model's tracked evaluation result. It is not a claim that CI evaluates the production AraBERT weights or that the lightweight DVC model is the deployed model. A later model-aware gate would need a clean, versioned source of candidate and full-baseline macro-F1 results.
