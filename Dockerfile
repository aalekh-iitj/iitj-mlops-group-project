# syntax=docker/dockerfile:1
# ---------- builder stage: keep image small for inference-only use ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

WORKDIR /app

# Slim, inference-only deps. We deliberately do NOT install torch+transformers
# from a pinned range here — the slim base plus an explicit torch CPU wheel
# (or matching CUDA wheel) keeps the image predictable. We install from a
# curated subset of requirements.txt to avoid pulling wandb/datasets etc.
COPY requirements.txt /app/requirements.txt

# Install just what inference needs. We install torch CPU (small) and
# transformers + tokenizer + hf hub. If you want a GPU image, change the
# torch install line and use a cuda base like nvidia/cuda:12.2.0-runtime-ubuntu22.04.
RUN pip install --no-cache-dir \
        "torch>=2.2,<3.0" \
        "transformers>=4.41,<5.0" \
        "tokenizers>=0.15,<1.0" \
        "huggingface-hub>=0.24,<1.0" \
        "numpy>=1.26,<3.0"

# Copy only what the inference script needs
COPY src/ /app/src/
COPY data/id2label.json /app/data/id2label.json
COPY data/label2id.json /app/data/label2id.json

# Model is pulled at build time (public) or runtime (private). Default is
# a public distilbert so the image works out of the box; override at build
# with --build-arg HF_MODEL_NAME=your-username/your-model.
ARG HF_MODEL_NAME=distilbert-base-uncased
ENV HF_MODEL_NAME=${HF_MODEL_NAME}

# Pre-download model weights at build time so the image is self-contained.
# (Skip if model is private; pass HF_TOKEN as a build secret or runtime env.)
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    tok = AutoTokenizer.from_pretrained('${HF_MODEL_NAME}'); \
    m = AutoModelForSequenceClassification.from_pretrained('${HF_MODEL_NAME}'); \
    print('preloaded', '${HF_MODEL_NAME}')" || true

# Default input — can be overridden with -e INPUT_TEXT='…' at run time
ENV INPUT_TEXT="I love this assignment!"

# Run inference as a module so the package imports resolve cleanly
CMD ["python", "-m", "src.inference"]
