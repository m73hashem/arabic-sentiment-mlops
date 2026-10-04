"""Shared AraBERT model configuration and sentiment label mapping."""

from transformers import AutoModelForSequenceClassification

MODEL_CHECKPOINT = "aubmindlab/bert-base-arabertv02"
LABEL_TO_ID = {"negative": 0, "positive": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def build_classifier(checkpoint: str = MODEL_CHECKPOINT):
    """Load the AraBERT checkpoint with a fresh binary classification head."""
    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=2,
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
    )
