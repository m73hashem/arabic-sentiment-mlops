# Module 5 — MLflow Experiment Tracking and Model Registry

## Handbook alignment and local storage

The Track A handbook requires at least five MLflow runs with parameters, metrics, artifacts, and the best model in the Registry promoted to Production. This phase provides ten completed meaningful runs, logs the official model and evaluations, and promotes the official model only after an explicit logged quality gate passes.

The experiment is `arabic-sentiment`. Tracking and Registry metadata use installed MLflow **3.16.1** with local SQLite at `sqlite:///.../mlflow.db`; run artifacts and logged-model artifacts are under local `mlartifacts/`. `mlflow.db`, its SQLite sidecars, `mlartifacts/`, and `mlruns/` are ignored by Git. No remote service or object store is configured.

Run the baseline registration, configured evaluations, alias assignment, quality gate, and history audit with:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 .venv/bin/python -m arabic_sentiment.mlflow_experiments
```

Audit the current history without creating runs with:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 .venv/bin/python -m arabic_sentiment.mlflow_experiments --verify-only
```

## Official full-GPU baseline

The official full-dataset model was trained **before MLflow, in Google Colab on a Tesla T4**. Local MLflow tracked and registered the existing inference artifact at `models/full-gpu-arabert-inference/`; it did not perform training. The inference export contains `config.json`, `model.safetensors`, `tokenizer.json`, and `tokenizer_config.json`; no optimizer, scheduler, scaler, RNG, or trainer-state files are needed.

Logged training parameters: `aubmindlab/bert-base-arabertv02`, seed 42, 1 epoch, max length 96, batch size 16, gradient accumulation 1 (effective 16), learning rate 2e-5, weight decay 0.01, AdamW, fp16 true, 317 warmup steps, 84,558/10,570/10,570 train/validation/test rows, Tesla T4, Google Colab, approximately 11.24 minutes. The following full-split metrics are the supplied results from that completed training and were logged unchanged:

| Split (rows) | Loss | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Validation (10,570) | 0.122829 | 0.963765 | 0.955407 | 0.972942 | 0.964095 |
| Test (10,570) | 0.12363797426223755 | 0.9628192999053926 | 0.9549851190476191 | 0.9714285714285714 | 0.9631366663540005 |

## Completed valid MLflow runs

The local history audit found **10 FINISHED valid runs**. Each has parameters, real measured or supplied baseline metrics, and artifacts. A valid run is completed, meaningful, and has logged artifacts; failed/interrupted records remain in the store but are excluded. Runs 1–3 are real completed work retained from the earlier Phase 2/Phase 5 attempt. No training occurred during the resumed workflow; runs 4–10 are baseline tracking, CPU evaluation, or a promotion gate.

| # | Run name — run ID | Purpose; training? | Key configuration | Actual metrics |
|---:|---|---|---|---|
| 1 | `phase2-baseline-reference` — `ec85ea9074aa43a99f711bcbfc23482b` | Preserve historical CPU 5K model; yes, Phase 2. Registry v1. | 5,000 train, seed 42, 1 epoch, max 96, batch 8 × accumulation 2, LR 2e-5, Adafactor, validation/test 10,570. | Val loss .1862863229493837; accuracy .9460737937559129; precision .9234776360696965; recall .9727530747398297; F1 .9474751197935866. Test loss .18257551909205852; accuracy .9459791863765373; precision .920442383160899; recall .9763481551561022; F1 .9475713892204572. |
| 2 | `cpu-lr-2e-5-n128-seq96-accum2` — `f2988ee96a434e4fba055fe5f9ac052b` | Bounded CPU training experiment; yes, prior Phase 5 attempt. Registry v2. | 128 train, seed 42, 1 epoch, max 96, batch 8 × accumulation 2, LR 2e-5, Adafactor; validation subset 256; test held out. | Val loss .6368963159620762; accuracy .71484375; precision .7272727272727273; recall .6875; F1 .7068273092369478; train loss .705858189612627. |
| 3 | `cpu-lr-1e-5-n32-seq96-effective16` — `7d95c9f23f254037bd600fef8f22b4df` | Distinct bounded CPU hyperparameter experiment; yes, prior Phase 5 attempt. Registry v3. | 32 train, seed 42, 1 epoch, max 96, batch 16 × accumulation 1, LR 1e-5, Adafactor; validation subset 64; test held out. | Val loss .6970818787813187; accuracy .5; precision .5; recall .65625; F1 .5675675675675675; train loss .6985616981983185. |
| 4 | `official-full-dataset-gpu-baseline` — `9929168869d44bd4aaa7c1243c8fe7b6` | Track the existing official Colab model and register its artifact; no training by MLflow. Registry v4. | Full GPU configuration and source artifact described above. | Full validation/test metrics in the preceding table. |
| 5 | `official-artifact-validation-subset-seq96` — `6492c71f10c74e8085f296f44f7f3084` | Official artifact evaluation on deterministic stratified validation subset; no training. | 64 rows, seed 42, max 96, batch 8; source run 4/version 4. SHA256 `1cf92c642c45fe66d51447eab4f4c0fc64f1cb5d02b0dc6ab7a6b0771c967a16`. | Loss .10642046545399353; accuracy .96875; macro-F1 .9687194525904204; positive F1 .967741935483871; precision 1.0; recall .9375; 9.290458905001287 s; 6.888787804179102 rows/s. |
| 6 | `official-artifact-validation-truncation-seq64` — `0fe2a4ecd11249d495399e3136851d91` | Same sample, sequence truncation comparison; no training. | 64 rows, seed 42, max 64, batch 8; same SHA256 as run 5; source run 4/version 4. | Loss .10608380287885666; accuracy .96875; macro-F1 .9687194525904204; positive F1 .967741935483871; precision 1.0; recall .9375; 6.282583162002993 s; 10.186892612432324 rows/s. |
| 7 | `official-artifact-validation-batch16-sample128` — `5f4df96273fb41979ef5d8b5b9bd20e6` | Larger validation sample and batch comparison; no training. | 128 rows, seed 42, max 96, batch 16; SHA256 `63b38e86bb129dcec083cc5a7f31316b9d5041ab29f86b74a4e3ae604ce213ee`; source run 4/version 4. | Loss .15267841378226876; accuracy .9609375; macro-F1 .960935115668681; positive F1 .9606299212598425; precision .9682539682539683; recall .953125; 17.52854612900046 s; 7.302374027942214 rows/s. |
| 8 | `official-artifact-test-subset-seq96` — `4c3bef13d2744728824065d82336b3fe` | Deterministic held-out test subset check; no training. | 64 rows, seed 42, max 96, batch 8; SHA256 `71083665e839fdc82af8893e9260cbc51ae58467c0947169242383b32cc68f45`; source run 4/version 4. | Loss .29484882013639435; accuracy .921875; macro-F1 .9218559218559218; positive F1 .923076923076923; precision .9090909090909091; recall .9375; 7.4481175809996785 s; 8.592775195072845 rows/s. |
| 9 | `official-artifact-test-truncation-seq64` — `96f9bb09653141979ef5d8b5b9bd20a6` | Same test sample, sequence truncation comparison; no training. | 64 rows, seed 42, max 64, batch 8; same SHA256 as run 8; source run 4/version 4. | Loss .2950311424792744; accuracy .921875; macro-F1 .9218559218559218; positive F1 .923076923076923; precision .9090909090909091; recall .9375; 6.158134115001303 s; 10.392758391554851 rows/s. |
| 10 | `official-baseline-production-quality-gate` — `6107fe5128144fb9b920dc9ea247507e` | Record the actual registry promotion checks; no training. | Candidate/registered version 4; F1 thresholds are the recorded Phase 2 CPU reference validation/test F1; actual pyfunc alias load and prediction required. | Passed all six checks. Validation F1 .964095 vs .9474751197935866 (margin .01661988020641335); test F1 .9631366663540005 vs .9475713892204572 (margin .0155652771335433). Alias model prediction: `positive`, confidence .9883586168289185. |

Runs 5–9 also log `model_name`, the source model's learning rate (explicitly tagged as inherited configuration; none of these runs trained), batch size, accuracy, and macro-F1 to support the handbook comparison fields. All record model version/source run ID, split/count, seed, max length, training-performed=false, sample hash, elapsed time, throughput, and a JSON artifact. The two truncation comparisons share the same split-specific sample to isolate max length. These small subset metrics are not full-split quality estimates.

## Historical CPU comparison

The former 5,000-row CPU-constrained Phase 2 experiment remains a historical reference, not the official baseline. Its test F1 was **0.9475713892204572**. The official full-dataset GPU test F1 is **0.9631366663540005**, a measured difference of **0.0155652771335433** (about **1.56 percentage points**). This comparison also changes training-data volume and optimizer/configuration, so it is not a causal estimate of hardware alone. Runs 2 and 3 above are additional real, very small prior CPU experiments and are not used to select the official model.

## Registry and promotion

The registered model is **`arabic-sentiment`**. Versions 1–3 are traceable to the historical CPU run IDs above. Version **4** is READY and linked to official baseline run `9929168869d44bd4aaa7c1243c8fe7b6`; its MLflow logged-model source is packaged from `models/full-gpu-arabert-inference/`. The run records the local source artifact and the original Colab/Tesla T4 training metadata.

Both `candidate` and `production` aliases resolve to version 4. The recorded production gate compared the official full-split validation/test F1 against the actual Phase 2 CPU reference run's corresponding values, required the registered version to be READY and traceable to the official run, required the candidate alias to select that version, and required a valid label/confidence from loading the MLflow alias model and predicting. All checks passed. This is a Registry promotion; it is **not** a claim of live production deployment.

## Serving/model selection

`src/arabic_sentiment/serving.py` selects the model provider outside the API routes. `SENTIMENT_MODEL_URI` supports an MLflow URI such as `models:/arabic-sentiment@candidate`, `models:/arabic-sentiment@production`, or a registered version such as `models:/arabic-sentiment/4`. The actual `candidate` URI was loaded through `MlflowModelService` and produced a CPU prediction: label `positive`, confidence `0.9883586168289185`, model version `4`.

Without `SENTIMENT_MODEL_URI`, the adapter uses the local filesystem artifact, preferring the official inference artifact and falling back to the older CPU artifact. The FastAPI route contract was not changed. `requirements-api.txt` intentionally excludes MLflow; the existing Docker image therefore uses filesystem fallback unless its runtime later adds the optional MLflow dependency and tracking configuration.

## Verification and limits

- Tracking: `sqlite:///.../mlflow.db`; artifacts: local `mlartifacts/`.
- Final history audit found 10 valid completed runs, official baseline metrics, Registry version/run traceability, candidate and production aliases both at version 4, and quality-gate run `6107fe5128144fb9b920dc9ea247507e`.
- Test command: `PYTHONPATH=src HF_HUB_OFFLINE=1 MLFLOW_DISABLE_AGENT_HINT=1 TMPDIR="$PWD/tmp/mlflow" .venv/bin/pytest -q`; result: **17 passed, 3 subtests passed**, with one MLflow/SQLAlchemy deprecation warning.
- Docker 29.8.0 build of `arabic-sentiment-mlops:phase5-compat-check` succeeded. A temporary container with the official model mount returned HTTP 200 from `/health` (`{"status":"ok"}`) and `/predict` (`{"label":"positive","confidence":0.9883586168289185,"model_version":"aubmindlab/bert-base-arabertv02-official-full-gpu-seed-42"`). The temporary container was removed; the image remains local.
- No AraBERT training occurred during the resumed workflow. No DVC, CI/CD, BentoML, load testing, canary, rollback, optimization, monitoring, or retraining pipeline is implemented in this phase.
- Evaluation timing/throughput values are single-run CPU subset measurements, not a serving load benchmark. The official full-split scores are the completed Colab results supplied for this project; they were not recomputed locally.
- The local inference artifact is 542,581,371 bytes across four files (model weights: 540,803,072 bytes). It is left available and untracked, not staged. MLflow stores a separate local logged-model copy under ignored `mlartifacts/`.

## DVC dataset lineage added in Phase 6.6

Project-owned MLflow run creation now reads the raw dataset hash from `data/raw/balanced-reviews.txt.dvc` and adds it as the `dvc.dataset_hash` tag. The helper also records `git_commit`, following the existing revision tag convention. The current DVC pointer records MD5 `8b3523338727a05a3f8bac9522073b75` for the 43,480,662-byte raw dataset. A run can therefore be associated with the dataset content version by reading its `dvc.dataset_hash` tag and matching that digest to the DVC pointer/object. The raw-data hash identifies the DVC-tracked source; the TF-IDF DVC model remains a lightweight reproducibility demonstration and is not the official AraBERT registered model.

The historical official GPU baseline and earlier Phase 5 runs were not rewritten. Their stored parameters and metadata identify the Colab training source, artifact, and split sizes, but do not contain the DVC content hash needed to establish raw-dataset provenance at run time. New project-created runs record both tags automatically. The historical official AraBERT model and Phase 6.5 DVC outputs were not changed for this lineage update.
