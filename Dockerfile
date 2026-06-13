# ── Base: slim Python 3.11 ────────────────────────────────────────────────────
FROM python:3.11-slim

# Build argument: override at build time with --build-arg
ARG HF_MODEL_NAME="your-username/distilbert-sst2"
ENV HF_MODEL_NAME=${HF_MODEL_NAME}

# Runtime env vars (supply at docker run via -e)
ENV HF_TOKEN=""
ENV INPUT_TEXT="This movie was great!"

WORKDIR /app

# Install only inference dependencies (no training libs like accelerate)
RUN pip install --no-cache-dir \
    transformers==4.40.2 \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    huggingface-hub==0.23.0 \
    numpy==1.26.4

# Copy source
COPY src/inference.py ./src/inference.py

# Default command
CMD ["python", "src/inference.py"]
