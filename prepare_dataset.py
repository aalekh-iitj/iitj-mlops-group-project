"""Standalone data preparation script — run once locally (or in a Kaggle CPU notebook) to
produce the clean train/test CSVs + id2label.json/label2id.json + data_summary.json.

Default task: AG News (4-class topic classification) text dataset, kept under 50k samples
per the assignment brief. Swap out `DATASET` / load_fn if your group picked a different one.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.utils import DATA_DIR, PROCESSED_DIR, save_id2label, save_label2id

SAMPLE_SIZE = 40000  # total; 80/20 split
TEST_FRACTION = 0.2
TEXT_COL = "text"
LABEL_COL = "label"
RANDOM_STATE = 42


def main() -> None:
    print("[1/4] Downloading AG News (≈30s, ~30k train + 1.9k test in HF cache)…")
    ds = load_dataset("fancyzhx/ag_news", split="train")
    # Shuffle, take a slice — well under 50k as required
    ds = ds.shuffle(seed=RANDOM_STATE).select(range(min(SAMPLE_SIZE, len(ds))))
    df = ds.to_pandas()
    print(f"      loaded shape: {df.shape}, columns: {list(df.columns)}")

    print("[2/4] Cleaning…")
    before = len(df)
    # Text cleaning
    df[TEXT_COL] = (
        df[TEXT_COL].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    )
    # Drop empties, duplicates, nulls
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df[df[TEXT_COL].str.len() > 0]
    df = df.drop_duplicates(subset=[TEXT_COL])
    # Label encoding — keep stable ordering from HF
    unique_labels = sorted(df[LABEL_COL].unique().tolist())
    label2id = {str(name): i for i, name in enumerate(unique_labels)}
    id2label = {i: str(name) for name, i in label2id.items()}
    df[LABEL_COL] = df[LABEL_COL].astype(str).map(label2id)
    print(f"      dropped {before - len(df)} rows; remaining: {len(df)}")

    print("[3/4] Train/test split + writing CSVs…")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df = df.sample(frac=1 - TEST_FRACTION, random_state=RANDOM_STATE)
    test_df = df.drop(train_df.index)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    save_id2label(id2label, DATA_DIR / "id2label.json")
    save_label2id(label2id, DATA_DIR / "label2id.json")

    summary = {
        "dataset": "fancyzhx/ag_news",
        "total_samples": int(len(df)),
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "num_labels": len(unique_labels),
        "labels": [str(x) for x in unique_labels],
        "class_distribution_train": dict(Counter(train_df[LABEL_COL].tolist())),
        "class_distribution_test": dict(Counter(test_df[LABEL_COL].tolist())),
        "cleaning_notes": [
            "stripped whitespace and collapsed internal whitespace",
            "dropped null / empty / duplicate texts",
            "shuffled with fixed seed 42",
            f"sampled {SAMPLE_SIZE} from full train split to stay under 50k",
        ],
    }
    with open(DATA_DIR / "data_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[4/4] Done. Artefacts:")
    print(f"      - {PROCESSED_DIR / 'train.csv'}")
    print(f"      - {PROCESSED_DIR / 'test.csv'}")
    print(f"      - {DATA_DIR / 'id2label.json'}  (committed)")
    print(f"      - {DATA_DIR / 'label2id.json'}  (committed)")
    print(f"      - {DATA_DIR / 'data_summary.json'}  (committed)")


if __name__ == "__main__":
    main()
