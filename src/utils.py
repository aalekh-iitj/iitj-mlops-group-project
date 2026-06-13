"""
Shared utilities: compute_metrics for Trainer, W&B helpers.
"""

import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


def compute_metrics(eval_pred):
    """
    Called by HuggingFace Trainer at each evaluation step.
    Logs accuracy + weighted F1 to W&B automatically via report_to='wandb'.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}


def load_id2label(path: str = "id2label.json") -> dict:
    """Load id2label mapping; keys are coerced to int."""
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def print_classification_report(labels, predictions, id2label: dict):
    """Pretty-print sklearn classification report with label names."""
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    print(classification_report(labels, predictions, target_names=target_names))


def get_num_labels(id2label_path: str = "id2label.json") -> int:
    return len(load_id2label(id2label_path))
