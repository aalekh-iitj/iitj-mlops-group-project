FROM python:3.11-slim

ARG HF_MODEL_NAME=rnema4/distilbert-agnews-topics
ENV HF_MODEL_NAME=$HF_MODEL_NAME

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir transformers torch huggingface_hub

COPY src/inference.py .

CMD ["python", "inference.py"]