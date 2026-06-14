"""Dataset loading + tokenisation used by both training and inference.


The encoder is tolerant to two representations of the `label` column:

  - string label names (e.g. "Business") — looked up in label2id.json

  - integer label ids    (e.g. 0, 1, 2, 3) — passed through if within range


This is important because the CSV written by `prepare_dataset.py` already

contains integer labels, and `Dataset.from_csv` reads them back as int.

"""

from __future__ import annotations

from typing import Dict, Tuple

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from .utils import load_id2label, load_label2id

DEFAULT_MODEL = "distilbert-base-uncased"

MAX_LENGTH = 128


def load_text_dataset(csv_path: str, text_col: str, label_col: str) -> Dataset:
    """Load a CSV with two columns: text and label (str OR int)."""

    ds = Dataset.from_csv(csv_path)

    cols_to_drop = [c for c in ds.column_names if c not in (text_col, label_col)]

    if cols_to_drop:
        ds = ds.remove_columns(cols_to_drop)

    return ds


def encode_labels(
    example: Dict, label2id: Dict[str, int], id2label: Dict[int, str]
) -> Dict:
    """Idempotently turn the label column into an int class id.


    Handles all three reasonable inputs:

      1. label is an int  → cast, pass through

      2. label is a str  → look up in label2id (string → int)

      3. label is a str  that already looks like an int → cast, pass through

    """

    val = example["label"]

    # Case 1: already an int (the common path after prepare_dataset.py)

    if isinstance(val, int):
        if val not in id2label:
            raise KeyError(
                f"Label id {val} not in id2label mapping (have {sorted(id2label)}). "
                f"Check that id2label.json was generated from the same dataset."
            )

        return example  # no change needed

    # Case 2 / 3: string form

    s = str(val).strip()

    if s in label2id:
        example["label"] = label2id[s]

        return example

    # Maybe the CSV got written with str(int) — try parsing

    try:
        as_int = int(s)

        if as_int in id2label:
            example["label"] = as_int

            return example

    except (ValueError, TypeError):
        pass

    raise KeyError(
        f"Label value {val!r} (string={s!r}) not in label2id ({list(label2id)}). "
        f"Re-run prepare_dataset.py to regenerate the mapping."
    )


def prepare_dataset(
    train_csv: str,
    test_csv: str,
    text_col: str = "text",
    label_col: str = "label",
    model_name: str = DEFAULT_MODEL,
    max_length: int = MAX_LENGTH,
) -> Tuple[DatasetDict, str]:
    """Tokenise + align labels. Returns (tokenised DatasetDict, model_name)."""

    label2id = load_label2id()

    id2label = load_id2label()

    assert len(label2id) == len(id2label), (
        f"id2label.json and label2id.json disagree on #labels "
        f"(id2label={len(id2label)}, label2id={len(label2id)})"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    raw = DatasetDict(
        {
            "train": load_text_dataset(train_csv, text_col, label_col),
            "test": load_text_dataset(test_csv, text_col, label_col),
        }
    )

    def _enc(ex):

        return encode_labels(ex, label2id, id2label)

    raw = raw.map(_enc)

    def tokenise(batch):

        return tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenised = raw.map(tokenise, batched=True, remove_columns=[text_col])

    tokenised.set_format("torch")

    return tokenised, model_name
