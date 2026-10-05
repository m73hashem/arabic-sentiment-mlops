import pandas as pd
import pytest

from arabic_sentiment.split import split_reviews, verify_no_review_leakage


@pytest.mark.parametrize(
    ("frame", "ratios", "message"),
    [
        (pd.DataFrame(columns=["review", "sentiment"]), (0.8, 0.1, 0.1), "empty dataset"),
        (pd.DataFrame({"review": [""], "sentiment": ["negative"]}), (0.8, 0.1, 0.1), "nonempty"),
        (pd.DataFrame({"review": ["text"], "sentiment": ["neutral"]}), (0.8, 0.1, 0.1), "only"),
        (pd.DataFrame({"review": ["text"], "sentiment": ["negative"]}), (0.8, 0.3, 0.1), "sum to 1"),
    ],
)
def test_split_rejects_empty_invalid_data_and_ratios(frame, ratios, message):
    with pytest.raises(ValueError, match=message):
        split_reviews(frame, ratios=ratios)


def test_leakage_verifier_detects_duplicate_text_across_split_frames():
    splits = {
        "train": pd.DataFrame({"review": ["مكرر"]}),
        "validation": pd.DataFrame({"review": ["مكرر"]}),
        "test": pd.DataFrame({"review": ["مختلف"]}),
    }
    assert not verify_no_review_leakage(splits)
