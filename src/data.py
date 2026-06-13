"""
Loads and tokenises the SST-2 dataset.
Used by both train.py (on Kaggle) and can be imported for local testing.
"""

import logging

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

log = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128


def get_tokenizer(model_name: str = MODEL_NAME):
    """Load and return the tokenizer for the chosen model."""
    log.info(f"Loading tokenizer: {model_name}")
    return AutoTokenizer.from_pretrained(model_name)


def load_and_tokenize(
    model_name: str = MODEL_NAME,
    max_length: int = MAX_LENGTH,
    sample_size: int = None,
) -> DatasetDict:
    """
    Load SST-2, optionally subsample, tokenise with padding+truncation.

    Args:
        model_name:  HF model identifier (must match tokenizer).
        max_length:  Token sequence length (128 fits SST-2 well).
        sample_size: If set, use only this many train+val examples
                     (useful for fast debug runs on CPU).

    Returns:
        DatasetDict with 'train' and 'validation' keys, tokenised columns.
    """
    dataset = load_dataset("glue", "sst2")
    tokenizer = get_tokenizer(model_name)

    if sample_size:
        dataset["train"] = dataset["train"].select(
            range(min(sample_size, len(dataset["train"])))
        )
        dataset["validation"] = dataset["validation"].select(
            range(min(sample_size // 10, len(dataset["validation"])))
        )
        log.info(
            f"Subsampled: train={len(dataset['train'])}, val={len(dataset['validation'])}"
        )

    def tokenize_batch(batch):
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_batch, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    log.info(
        f"Tokenised — train: {len(tokenized['train'])}, val: {len(tokenized['validation'])}"
    )
    return tokenized
