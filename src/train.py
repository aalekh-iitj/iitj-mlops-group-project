"""Training entrypoint — designed to run on Kaggle Notebooks (GPU T4)."""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import wandb
from huggingface_hub import login as hf_login
from kaggle_secrets import UserSecretsClient  # type: ignore
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .data import prepare_dataset
from .utils import load_id2label, load_label2id, get_env


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def load_secrets_kaggle() -> None:
    """Pull WANDB_API_KEY and HF_TOKEN from Kaggle Secrets — never hardcode."""
    secrets = UserSecretsClient()
    os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
    os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True, help="Path to processed train.csv")
    parser.add_argument("--test_csv", required=True, help="Path to processed test.csv")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--run_name", default="run-v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push_to_hub", action="store_true", help="Push best model to HF Hub")
    parser.add_argument("--hub_repo_id", default=None, help="e.g. your-username/mlops-a3-model")
    args = parser.parse_args()

    # 1) Secrets (Kaggle path)
    try:
        load_secrets_kaggle()
    except Exception as e:  # local fallback
        print(f"[warn] Kaggle secrets unavailable ({e}); assuming env vars are set.")

    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY not set in environment.")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN not set in environment.")

    hf_login(token=os.environ["HF_TOKEN"])
    wandb.login(key=os.environ["WANDB_API_KEY"])

    # 2) Repro
    set_seed(args.seed)

    # 3) Data
    tokenised, model_name = prepare_dataset(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        model_name=args.model_name,
    )
    id2label = load_id2label()
    label2id = load_label2id()

    # 4) Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 5) W&B run
    wandb.init(
        project="mlops-assignment3",
        name=args.run_name,
        config={
            "model": model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "platform": "Kaggle",
        },
    )

    # 6) Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=args.learning_rate,
        report_to="wandb",
        run_name=args.run_name,
        logging_steps=20,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()
    print(f"[eval] {eval_metrics}")

    # 7) Push to HF Hub (Task 5)
    if args.push_to_hub and args.hub_repo_id:
        trainer.push_to_hub(args.hub_repo_id)
        wandb.run.summary["huggingface_model"] = f"https://huggingface.co/{args.hub_repo_id}"
    elif args.push_to_hub and not args.hub_repo_id:
        print("[warn] --push_to_hub set but --hub_repo_id missing; skipping upload.")

    wandb.finish()


if __name__ == "__main__":
    main()
