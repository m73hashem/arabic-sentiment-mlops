# Module 2 — Baseline Deep Learning Model

## Objective and checkpoint

Fine-tune a reusable Arabic transformer classifier for the Phase 1 hotel-review sentiment task. The checkpoint is **AraBERTv0.2-base**, `aubmindlab/bert-base-arabertv02`, from the AUB MIND Lab's official AraBERT model family. The [official AraBERT model card](https://huggingface.co/aubmindlab/bert-base-arabertv02) identifies the v0.2 base variant and its tokenizer/model files.

AraBERT is published under `aubmindlab`; CAMeL-Lab's CAMeLBERT is a distinct model family. The implementation uses the actual AraBERT checkpoint rather than substituting CAMeLBERT or a multilingual model. AraBERTv0.2 is used without adding Farasa segmentation or other normalization, so the Phase 1 cleaned review text is passed directly to its tokenizer.

## Dataset and labels

- Source files: `data/processed/train.csv`, `validation.csv`, and `test.csv`; the Phase 1 files and split membership were not changed.
- Available split sizes: train 84,558; validation 10,570; test 10,570.
- Training used a deterministic, class-proportional sample of 5,000 rows from the existing training split: 2,500 negative and 2,500 positive. The successful run records seed 42. No rows were moved between splits.
- Validation used all 10,570 validation rows. Final test evaluation used all 10,570 test rows exactly once.
- Label mapping: `negative` → 0, `positive` → 1. This mapping is stored in the model config and training results.

The training sample limit was introduced after a full-split CPU run showed a very slow measured step pace before reaching its first 100-step log. That attempt was interrupted before validation and test evaluation and produced no metrics or artifact. The successful run used the 5,000-row sample; validation and test were not reduced.

## Tokenization and model

The AraBERT fast tokenizer tokenizes the Phase 1 `review` text with truncation at 96 tokens and dynamic padding to the longest item in each batch. Original processed CSVs remain unchanged. No stemming, translation, stop-word removal, Farasa segmentation, or additional Arabic normalization was applied. The pretrained masked-language-model head was replaced with a two-class sequence-classification head and fine-tuned on the training sample.

## Training configuration and hardware

| Setting | Value |
|---|---|
| Checkpoint | `aubmindlab/bert-base-arabertv02` (AraBERTv0.2-base) |
| Device | CPU; CUDA unavailable |
| Training rows | 5,000 of 84,558, sampled proportionally by label with seed 42 |
| Validation / test rows | 10,570 / 10,570 |
| Seed | 42 (Python, NumPy, PyTorch, and DataLoader generator) |
| Epochs | 1 |
| Learning rate | 0.00002 |
| Optimizer | Transformers Adafactor |
| Weight decay | 0.01 |
| Micro-batch size | 8 |
| Gradient accumulation | 2 steps; effective batch size 16 |
| Maximum sequence length | 96 tokens |
| Learning-rate warmup | 6% of optimizer steps; linear decay afterward |
| Gradient clipping | 1.0 |
| Evaluation strategy | Full validation pass after each epoch |
| Best-model selection | Highest validation F1; save best only |
| Optimizer steps | 313 for the successful epoch |

The environment had 4 available CPU threads and no CUDA. Training used PyTorch 2.14.1+cpu and Transformers 5.18.0; no packages were installed for this phase.

## Actual results

Metrics are positive-class precision, recall, and F1. Validation and test losses are mean cross-entropy loss per example.

| Split | Rows evaluated | Loss | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Validation (epoch 1, selected best) | 10,570 | 0.186286 | 0.946074 | 0.923478 | 0.972753 | 0.947475 |
| Test (one final evaluation) | 10,570 | 0.182576 | 0.945979 | 0.920442 | 0.976348 | 0.947571 |

The observed training loss at the epoch boundary was 0.234076. Validation F1 selected the epoch-1 checkpoint. These are results from the actual CPU run and a 5,000-row training sample; they are not full-train-split metrics.

## Artifact and inference

The reusable model and tokenizer are saved at `models/baseline-arabert/`. The directory contains `model.safetensors`, the model config, tokenizer files, `training_config.json`, and `training_results.json`. The weights file is 540,803,072 bytes; all artifact files total 542,584,016 bytes. The model loads locally with `SentimentPredictor` and does not need to contact Hugging Face after the artifact is saved.

Example run using the saved artifact:

```python
from arabic_sentiment.inference import SentimentPredictor

predictor = SentimentPredictor("models/baseline-arabert")
predictor.predict("الخدمة ممتازة والغرفة نظيفة ومريحة")
# {'label': 'positive', 'confidence': 0.9891707301139832}
```

The example above was executed against the saved model. It is one local prediction, not a dataset metric.

## Verification and limitations

- All 11 tests passed, including all 6 Phase 1 tests, deterministic label/sample tests, local artifact loading, valid prediction labels/confidence, and repeatable inference.
- The training and final test evaluation completed on CPU. The saved results record `test_evaluations: 1`.
- Training was limited to 5,000 examples because full-split CPU fine-tuning was impractically slow. This baseline should be treated as a resource-constrained first model; it is not evidence of full-data training performance.
- The eight conflicting-label duplicate text groups documented in Module 1 remain in their original splits and can contribute ambiguous supervision.
- No API, Docker, tracking, data-versioning, serving, optimization, or monitoring component is implemented in this phase.
