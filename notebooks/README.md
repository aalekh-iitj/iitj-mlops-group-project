# notebooks/

This folder is intentionally empty in the repo. The actual training notebooks live on **Kaggle** (one per experiment version) — see `RUNBOOK.md` STEP 5.

When you create a Kaggle notebook:

1. Settings → Accelerator → **GPU T4 x2**.
2. **Add secrets** in the Kaggle sidebar: `HF_TOKEN`, `WANDB_API_KEY`.
3. Title it `mlops-a3-train-v1` and a second one `mlops-a3-train-v2`.
4. Set the visibility of both notebooks to **Public**.
5. Paste the public URLs into the top of the project `README.md` and into the report.
