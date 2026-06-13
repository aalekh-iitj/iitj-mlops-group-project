"""
Model fine-tuning script.
Run this inside a Kaggle Notebook — NOT in GitHub Actions.
Supports two versions via VERSION env var: 'v1' or 'v2'.

Usage on Kaggle:
    VERSION=v1 python src/train.py
    VERSION=v2 python src/train.py
"""

import json
import logging
import os

import wandb
from huggingface_hub import login
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data import MODEL_NAME, get_tokenizer, load_and_tokenize
from src.utils import compute_metrics, load_id2label

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Hyperparameter configurations ─────────────────────────────────────────────
CONFIGS = {
    "v1": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 32,
        "learning_rate": 3e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
    },
    "v2": {
        "num_train_epochs": 4,
        "per_device_train_batch_size": 16,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.06,
        "weight_decay": 0.05,
    },
}
# v1 vs v2 differ in: batch size (32 vs 16), lr (3e-5 vs 2e-5), epochs (3 vs 4), weight_decay


def train(version: str = "v1", hf_model_repo: str = None):
    version = version.lower()
    cfg = CONFIGS[version]
    id2label = load_id2label("id2label.json")
    label2id = {v: k for k, v in id2label.items()}

    # ── W&B init ─────────────────────────────────────────────────────────────────
    run = wandb.init(
        project="iitj-mlops-group-project",
        name=f"distilbert-sst2-{version}",
        config={
            "model": MODEL_NAME,
            "version": version,
            "platform": "Kaggle",
            "dataset": "glue/sst2",
            **cfg,
        },
    )

    # ── Load tokenised data ───────────────────────────────────────────────────────
    tokenized = load_and_tokenize(model_name=MODEL_NAME)
    train_ds = tokenized["train"]
    val_ds = tokenized["validation"]

    # ── Load model ────────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    # ── Training arguments ───────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=f"./results/{version}",
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=64,
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="wandb",
        run_name=f"distilbert-sst2-{version}",
        fp16=True,  # Enable on Kaggle GPU T4
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    log.info(f"Starting training: {version} | config: {cfg}")
    trainer.train()

    # ── Evaluate ──────────────────────────────────────────────────────────────────
    results = trainer.evaluate()
    log.info(f"Final eval results: {results}")
    wandb.log(results)

    # ── Push to Hugging Face Hub ──────────────────────────────────────────────────
    if hf_model_repo:
        tokenizer = get_tokenizer(MODEL_NAME)
        model.push_to_hub(hf_model_repo)
        tokenizer.push_to_hub(hf_model_repo)
        hf_url = f"https://huggingface.co/{hf_model_repo}"
        wandb.run.summary["huggingface_model_url"] = hf_url
        log.info(f"Model pushed to HF Hub: {hf_url}")

    wandb.finish()
    return results


if __name__ == "__main__":
    # On Kaggle: load secrets, login, then run
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    HF_REPO = os.environ.get(
        "HF_MODEL_REPO", ""
    )  # e.g. "your-username/distilbert-sst2"
    VERSION = os.environ.get("VERSION", "v1")

    if HF_TOKEN:
        login(token=HF_TOKEN)

    train(version=VERSION, hf_model_repo=HF_REPO or None)
