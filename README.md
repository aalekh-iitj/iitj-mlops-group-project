# MLOps Group Project — IIT Jodhpur (PGD AI, Assignment 3)

End-to-end MLOps pipeline: HF model → Kaggle training → W&B tracking → HF Hub → Docker → GitHub Actions.

> **Live links for the report** (fill these in once your team has them):
> - GitHub repo: `https://github.com/<org>/iitj-mlops-group-project`
> - Kaggle notebook v1: `https://www.kaggle.com/<user>/<notebook-slug>`
> - Kaggle notebook v2: `https://www.kaggle.com/<user>/<notebook-slug>`
> - Hugging Face model: `https://huggingface.co/<user>/<repo>`
> - Docker image: `https://hub.docker.com/r/<user>/mlops-a3-inference`
> - W&B project: `https://wandb.ai/<user>/mlops-assignment3`

## Quick start

```bash
# 1. (One time, locally) Prepare data — produces data/processed/*.csv + id2label/label2id
python prepare_dataset.py

# 2. (Kaggle GPU) Train two versions — see notebooks/README.md for the Kaggle path
python -m src.train --train_csv data/processed/train.csv --test_csv data/processed/test.csv \
    --run_name run-v1 --epochs 3 --learning_rate 3e-5 --batch_size 16 \
    --push_to_hub --hub_repo_id <your-hf-user>/<your-model-repo>

# 3. (Docker) Run inference locally
docker build --build-arg HF_MODEL_NAME=<your-hf-user>/<your-model-repo> -t mlops-a3-inference .
docker run --rm -e HF_TOKEN=$HF_TOKEN -e INPUT_TEXT="Stocks rallied today." mlops-a3-inference
```

See `RUNBOOK.md` for the full step-by-step process.
