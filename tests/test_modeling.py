from arabic_sentiment import modeling


def test_build_classifier_requests_binary_sentiment_label_metadata(monkeypatch):
    calls = []
    expected_model = object()

    def load(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs))
        return expected_model

    monkeypatch.setattr(modeling.AutoModelForSequenceClassification, "from_pretrained", load)

    assert modeling.build_classifier("local-checkpoint") is expected_model
    assert calls == [
        (
            "local-checkpoint",
            {
                "num_labels": 2,
                "label2id": {"negative": 0, "positive": 1},
                "id2label": {0: "negative", 1: "positive"},
            },
        )
    ]
