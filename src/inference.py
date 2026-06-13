"""Inference entrypoint — used by Docker container and by GitHub Actions workflow."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .utils import ID2LABEL_PATH, load_id2label, get_env


def main():
    # 1) Where is the model?
    #    Inside the Docker image: baked in at build time via $HF_MODEL_NAME.
    #    In CI (GitHub Actions): pulled at runtime from HF_TOKEN-secured repo.
    model_name = os.environ.get("HF_MODEL_NAME", "distilbert-base-uncased")

    # 2) Auth (for private models). Public models don't strictly need it.
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    # 3) Load
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token)
    model.eval()

    # 4) Input — Docker passes via env, CI via workflow input
    text = os.environ.get("INPUT_TEXT", "I love this assignment!")
    # Allow multi-line via stdin passthrough
    if not os.environ.get("INPUT_TEXT") and not sys_stdin_isatty():
        text = sys_stdin_read().strip() or text

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = int(logits.argmax(-1).item())
    id2label = load_id2label()
    label = id2label.get(pred_id, str(pred_id))

    result = {"input": text, "predicted_label_id": pred_id, "predicted_label": label}
    print(json.dumps(result, indent=2))
    # Machine-readable single line for CI logs
    print(f"PRED={label}")


def sys_stdin_isatty() -> bool:
    import sys
    return sys.stdin.isatty()


def sys_stdin_read() -> str:
    import sys
    return sys.stdin.read()


if __name__ == "__main__":
    main()
