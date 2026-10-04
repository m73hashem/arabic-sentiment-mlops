# Module 1 — Data Ingestion and Preprocessing

## Source inspection

- Source: `data/raw/balanced-reviews.txt`; preserved unchanged. Its SHA-256 remained `86c1ae1b41267262d806275df4c0385b7948b18a685c93212ca858aa8df2ab02` after pipeline execution.
- Encoding and line endings: UTF-16 little-endian with CRLF, verified from the file and its BOM.
- Delimiter: tab (`\t`), read with pandas.
- Shape: 105,698 rows × 7 columns.
- Columns: `no`, `Hotel name`, `rating`, `user type`, `room type`, `nights`, `review`.
- Pandas dtypes: all seven columns were read as `string` to preserve source values during inspection.
- Missing values by column: 0 in every column. Missing hotel names: 0. Missing ratings: 0. Missing reviews: 0.
- Exact duplicate full rows: 0.

## Ratings and filtering

| Rating | Rows | Sentiment treatment |
|---:|---:|---|
| 1 | 14,382 | negative |
| 2 | 38,467 | negative |
| 3 | 0 | excluded as neutral if present |
| 4 | 26,450 | positive |
| 5 | 26,399 | positive |

There were 0 invalid ratings, 0 missing ratings, and 0 rating-3 rows to exclude. There were 0 empty reviews and 0 whitespace-only reviews after cleaning. No rows were excluded; 105,698 usable rows remain. The final sentiment distribution is 52,849 negative and 52,849 positive.

## Text cleaning and duplicates

Review text is converted to a controlled empty string when missing, trimmed at both ends, and runs of whitespace are collapsed to one space. Arabic characters, diacritics, and punctuation are preserved. Empty reviews are removed after cleaning. No stemming, translation, stop-word removal, or aggressive Arabic normalization is performed.

There are 0 exact duplicate rows. After review cleaning, 1,671 rows repeat text already present, across 575 repeated-text groups. These rows were retained. Eight repeated-text groups (41 rows) contain both sentiment labels; the labels were not altered. Review text is the split grouping key, so every copy of a cleaned review, including conflicting-label copies, remains in one split.

## Deterministic split

The split uses seed 42 and a deterministic largest-group-first allocator. It assigns complete cleaned-review groups while minimizing deviation from the target class counts and split proportions.

| Split | Rows | Proportion | Negative | Positive |
|---|---:|---:|---:|---:|
| Train | 84,558 | 80.00% | 42,279 | 42,279 |
| Validation | 10,570 | 10.00% | 5,285 | 5,285 |
| Test | 10,570 | 10.00% | 5,285 | 5,285 |

The leakage check found no cleaned review text shared between splits. The CSV schema is `review`, `rating`, `sentiment`, encoded as UTF-8. The rating is retained as metadata, and sentiment labels are the strings `negative` and `positive`.

## Generated files and validation

| File | Rows | Size (bytes) | SHA-256 |
|---|---:|---:|---|
| `data/processed/train.csv` | 84,558 | 21,988,703 | `d41e1ff96a12447821ecdfdcdc4e0900afcac007ea7e0f15ff8436f5d29b78f7` |
| `data/processed/validation.csv` | 10,570 | 2,728,256 | `bbfdfa22bdd2824d1fd368f5ae595cc7559c98b2e124f9735495d70f0b297f82` |
| `data/processed/test.csv` | 10,570 | 2,745,964 | `981acfe2f56823fe4a310d70fe7b09cce6a304ec8e3de2978f6d48a0e79a27cf` |

All 6 tests passed using Python's built-in `unittest` runner. The complete pipeline was run twice; all three CSV hashes, row counts, labels, and split membership were identical on the second run.

## Limitations and observations

The dataset contains no rating-3, missing, or invalid-rating examples, so those exclusion paths are covered by synthetic tests rather than observed source rows. Duplicate review text with conflicting sentiment is present and retained; grouping prevents leakage but does not resolve those source-label conflicts. The generated CSVs are local outputs and have not been staged in Git.
