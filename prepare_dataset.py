"""
Task 2: Data Preparation & Normalisation
Dataset: SST-2 (Stanford Sentiment Treebank) — binary sentiment
Covers: inspection, cleaning, class distribution, label encoding, id2label.json
"""

import json
import logging
import os
import re
from collections import Counter

import pandas as pd
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def inspect_dataset(dataset) -> None:
    """Print structure, size, class distribution, and quality checks."""
    log.info("=== Dataset Inspection ===")
    for split, ds in dataset.items():
        log.info(
            f"  Split '{split}': {len(ds):,} examples | columns: {ds.column_names}"
        )

    # Class distribution in training split
    train_df = dataset["train"].to_pandas()
    dist = Counter(train_df["label"].tolist())
    log.info(f"  Train class distribution: {dict(dist)}")
    log.info(f"  Null sentences: {train_df['sentence'].isnull().sum()}")
    log.info(f"  Duplicate sentences: {train_df.duplicated(subset='sentence').sum()}")
    log.info(
        f"  Avg sentence length (chars): {train_df['sentence'].str.len().mean():.1f}"
    )


def clean_text(text: str) -> str:
    """
    Text normalisation steps (justified):
    1. Strip leading/trailing whitespace  — removes accidental padding
    2. Collapse multiple spaces → single  — noisy SST-2 entries have extra spaces
    3. Remove HTML entities               — a few SST-2 sentences have & etc.
    4. Lowercase                          — DistilBERT uncased; casing adds no signal
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"&", "&", text)
    text = re.sub(r"<", "<", text)
    text = re.sub(r">", ">", text)
    text = text.lower()
    return text


def clean_split(dataset_split):
    """Apply cleaning to a HF dataset split, drop nulls and duplicates."""
    df = dataset_split.to_pandas()

    before = len(df)
    df["sentence"] = df["sentence"].apply(
        lambda x: clean_text(str(x)) if pd.notnull(x) else None
    )
    df.dropna(subset=["sentence"], inplace=True)
    df.drop_duplicates(subset="sentence", inplace=True)
    after = len(df)

    log.info(f"  Cleaned: {before:,} → {after:,} rows ({before - after} removed)")
    return df


def save_id2label(label_names: list, out_path: str = "id2label.json") -> None:
    """Save {int_id: label_name} mapping as required by rubric."""
    mapping = {i: name for i, name in enumerate(label_names)}
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    log.info(f"  Saved id2label → {out_path}: {mapping}")


def main():
    log.info("Loading SST-2 from Hugging Face datasets...")
    dataset = load_dataset("nyu-mll/glue", "sst2")

    # Step 1: Inspect
    inspect_dataset(dataset)

    # Step 2: Clean each split
    log.info("Cleaning splits...")
    train_df = clean_split(dataset["train"])
    val_df = clean_split(dataset["validation"])

    # Step 3: Label encoding — SST-2 already uses 0/1 integers
    # 0 = negative, 1 = positive
    label_names = dataset["train"].features["label"].names
    log.info(f"Label names from HF: {label_names}")

    # Step 4: Save id2label.json (required by rubric)
    save_id2label(label_names, out_path="id2label.json")

    # Step 5: Report final dataset summary (to stdout / logs, NOT committed to git)
    summary = {
        "dataset": "glue/sst2",
        "task": "binary sentiment classification",
        "train_size": len(train_df),
        "val_size": len(val_df),
        "num_labels": len(label_names),
        "label_names": label_names,
        "cleaning_steps": [
            "Strip whitespace",
            "Collapse multiple spaces",
            "Decode HTML entities",
            "Lowercase",
            "Drop null sentences",
            "Drop duplicate sentences",
        ],
    }
    log.info(f"Summary: {json.dumps(summary, indent=2)}")
    log.info("Data preparation complete. id2label.json is ready.")
    log.info(
        "NOTE: Dataset itself is loaded from HF Hub at training time — not saved locally."
    )


if __name__ == "__main__":
    main()
