# Module 7 — CI model quality gate

## Metric source and model scope

GitHub Actions reads the candidate evaluation from `metrics/dvc-reproducibility.json`, the output of the DVC `evaluate` stage. The DVC evaluator now reports `f1_macro` for both `negative` and `positive` labels, with `zero_division=0`. CI does not reproduce the pipeline or train AraBERT: DVC raw data and model weights are not available in a clean GitHub checkout, and the DVC pipeline is only the lightweight TF-IDF + Logistic Regression reproducibility demonstration.

The full-dataset official AraBERT baseline does not have a tracked full-split `f1_macro` result. Its report and MLflow metadata contain full-split positive-class F1, while macro-F1 in MLflow is available only for deterministic subsets; the local MLflow database is not part of a clean checkout. Those values are not silently treated as equivalent. Instead, `configs/quality-gate.json` records the real test macro-F1 of the current DVC reproducibility model as a frozen regression baseline.

The measured baseline and current candidate are both **0.9392593984962405** on the DVC test split. The baseline value came from executing `src/arabic_sentiment/dvc_pipeline/evaluate.py` with the existing local TF-IDF + Logistic Regression artifact and test split. Equality passes; lower candidate values fail. Tests use separate temporary fixture values and do not represent project metrics.

The DVC evaluation stage declares this Git-tracked metric output with `cache: false`. This lets DVC regenerate and diff the small metrics JSON without treating the Git-tracked file as a DVC-cached model/data artifact.

## Gate and publishing dependency

Run the same fail-closed comparison locally:

```bash
PYTHONPATH=src python -m arabic_sentiment.quality_gate \
  --candidate metrics/dvc-reproducibility.json \
  --baseline configs/quality-gate.json
```

The gate rejects missing files/fields, malformed JSON, unsupported schemas, non-numeric or non-finite scores, and values outside `[0, 1]`. GitHub Actions runs lint, tests/coverage, and the Docker build as before; the quality-gate job runs after tests. Docker Hub publishing requires both the build and gate jobs to succeed. Pull requests never run the login/push job.

This is explicitly a regression gate for the DVC reproducibility model's tracked evaluation result. It is not a claim that CI evaluates the production AraBERT weights or that the lightweight DVC model is the deployed model. A later model-aware gate would need a clean, versioned source of candidate and full-baseline macro-F1 results.
