"""
Inference script — loads best model from HF Hub, classifies INPUT_TEXT.
Called by .github/workflows/inference.yml via:
    env:
      HF_TOKEN: ${{ secrets.HF_TOKEN }}
      INPUT_TEXT: ${{ github.event.inputs.input_text }}
    run: python src/inference.py

Also works locally:
    HF_TOKEN=hf_xxx HF_MODEL_NAME=your-user/distilbert-sst2 \
    INPUT_TEXT="This movie was absolutely brilliant!" python src/inference.py
"""

import logging
import os
import sys

from huggingface_hub import login
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Config from environment ────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL = os.environ.get("HF_MODEL_NAME", "your-username/distilbert-sst2")
INPUT_TEXT = os.environ.get("INPUT_TEXT", "")


def run_inference(text: str, model_name: str, hf_token: str = "") -> dict:
    """Load model from HF Hub and classify the given text."""
    if not text:
        log.error("INPUT_TEXT is empty. Set via environment variable.")
        sys.exit(1)

    if hf_token:
        login(token=hf_token)
        log.info("Logged in to Hugging Face Hub.")

    log.info(f"Loading model: {model_name}")
    classifier = pipeline(
        task="text-classification",
        model=model_name,
        tokenizer=model_name,
        token=hf_token or None,
    )

    log.info(f"Running inference on: '{text}'")
    result = classifier(text)[0]

    label = result["label"]
    score = result["score"]

    # GitHub Actions will capture this output in the workflow log
    print(f"\n{'=' * 50}")
    print(f"Input:      {text}")
    print(f"Prediction: {label}")
    print(f"Confidence: {score:.4f} ({score * 100:.1f}%)")
    print(f"{'=' * 50}\n")

    return {"label": label, "score": score}


if __name__ == "__main__":
    run_inference(
        text=INPUT_TEXT,
        model_name=HF_MODEL,
        hf_token=HF_TOKEN,
    )
