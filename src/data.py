"""Dataset loading + tokenisation used by both training and inference."""
from __future__ import annotations

from typing import Dict, Tuple

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from .utils import load_id2label, load_label2id

DEFAULT_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 128


def load_text_dataset(csv_path: str, text_col: str, label_col: str) -> Dataset:
    """Load a CSV with two columns: text and label (string)."""
    ds = Dataset.from_csv(csv_path)
    cols_to_drop = [c for c in ds.column_names if c not in (text_col, label_col)]
    if cols_to_drop:
        ds = ds.remove_columns(cols_to_drop)
    return ds


def encode_labels(example: Dict, label2id: Dict[str, int]) -> Dict:
    example["label"] = label2id[example["label"]]
    return example


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
    # sanity check: id2label must match
    id2label = load_id2label()
    assert len(label2id) == len(id2label), "id2label.json and label2id.json disagree on #labels"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    raw = DatasetDict(
        {
            "train": load_text_dataset(train_csv, text_col, label_col),
            "test": load_text_dataset(test_csv, text_col, label_col),
        }
    )
    raw = raw.map(lambda ex: encode_labels(ex, label2id))

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
