# DistilBERT Goodreads Genre Classifier

Fine-tuned DistilBERT on the UCSD Goodreads dataset to classify book reviews into 8 genres.
Built for MLOps Assignment 2 | PGD AI Programme | IIT Jodhpur.

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python train.py
python eval.py
```

## Results
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.5156 |
| F1 Score  | 0.5121 |
| Eval Loss | 1.3423 |

- Hugging Face: https://huggingface.co/gauti0001/distilbert-goodreads-genres
- W&B Dashboard: https://wandb.ai/g25ait2034-indian-institute-of-technology/mlops-assignment2
