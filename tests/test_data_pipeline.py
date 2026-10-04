import pandas as pd
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from arabic_sentiment.data import RAW_COLUMNS, load_raw_data, validate_raw_schema
from arabic_sentiment.preprocessing import clean_review_text, prepare_reviews
from arabic_sentiment.split import split_reviews, verify_no_review_leakage


def synthetic_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "no": ["1", "2", "3", "4", "5", "6", "7", "8"],
            "Hotel name": ["Hotel"] * 8,
            "rating": ["1", "2", "4", "5", "3", "invalid", pd.NA, "5"],
            "user type": ["Guest"] * 8,
            "room type": ["Room"] * 8,
            "nights": ["1"] * 8,
            "review": [
                "  سيء   جدا ",
                "سيء جدا",
                "جيد!",
                "ممتاز",
                "محايد",
                "غير معروف",
                "نص بلا تقييم",
                "   \t  ",
            ],
        }
    )


class DataPipelineTests(unittest.TestCase):
    def test_raw_schema_validation_and_utf16_tsv_loading(self):
        raw = synthetic_raw()
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw.txt"
            raw.to_csv(path, sep="\t", index=False, encoding="utf-16", lineterminator="\r\n")

            loaded = load_raw_data(path)
            self.assertEqual(loaded.columns.tolist(), RAW_COLUMNS)
            self.assertEqual(len(loaded), len(raw))

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                validate_raw_schema(loaded.drop(columns=["review"]))

    def test_rating_mapping_invalid_values_and_missing_values_are_filtered(self):
        result, stats = prepare_reviews(synthetic_raw())

        self.assertEqual(
            result[["rating", "sentiment"]].values.tolist(),
            [[1, "negative"], [2, "negative"], [4, "positive"], [5, "positive"]],
        )
        self.assertEqual(stats["neutral_rating_rows"], 1)
        self.assertEqual(stats["invalid_ratings"], 1)
        self.assertEqual(stats["missing_ratings"], 1)
        self.assertEqual(stats["empty_reviews"], 1)
        self.assertEqual(stats["excluded_rows"], 4)

    def test_review_cleaning_preserves_arabic_and_punctuation(self):
        values = pd.Series(["  مَرْحَبًا!!!\n   بالعالم  ", None, "  "])
        self.assertEqual(clean_review_text(values).tolist(), ["مَرْحَبًا!!! بالعالم", "", ""])

    def test_empty_reviews_are_not_in_prepared_data(self):
        result, stats = prepare_reviews(synthetic_raw())
        self.assertTrue(result["review"].ne("").all())
        self.assertNotIn("   \t  ", result["review"].tolist())
        self.assertEqual(stats["whitespace_only_reviews"], 1)

    def test_grouped_split_is_deterministic_and_keeps_class_balance(self):
        data = pd.DataFrame(
            {
                "review": [f"نص {i}" for i in range(100)] + ["مكرر", "مكرر", "مكرر"],
                "rating": [1] * 50 + [4] * 53,
                "sentiment": ["negative"] * 50 + ["positive"] * 53,
            }
        )
        first = split_reviews(data, seed=42)
        second = split_reviews(data, seed=42)

        self.assertEqual(
            {name: frame.to_dict("records") for name, frame in first.items()},
            {name: frame.to_dict("records") for name, frame in second.items()},
        )
        self.assertEqual([len(first[name]) for name in ("train", "validation", "test")], [83, 10, 10])
        self.assertTrue(verify_no_review_leakage(first))
        self.assertEqual(first["validation"]["sentiment"].nunique(), 2)
        self.assertEqual(first["test"]["sentiment"].nunique(), 2)

    def test_duplicate_review_text_never_crosses_splits_including_conflicting_labels(self):
        data = pd.DataFrame(
            {
                "review": ["duplicate"] * 4 + [f"other {i}" for i in range(16)],
                "rating": [1, 1, 4, 4] + [1] * 8 + [4] * 8,
                "sentiment": ["negative", "negative", "positive", "positive"]
                + ["negative"] * 8
                + ["positive"] * 8,
            }
        )
        splits = split_reviews(data, seed=42)

        self.assertTrue(verify_no_review_leakage(splits))
        occurrences = [name for name, frame in splits.items() if frame["review"].eq("duplicate").any()]
        self.assertEqual(len(occurrences), 1)


if __name__ == "__main__":
    unittest.main()
