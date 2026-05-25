import os, wandb
from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
from huggingface_hub import login
from data import load_and_prepare
from utils import id2label, label2id, compute_metrics

MODEL_NAME     = "distilbert-base-cased"
MAX_LENGTH     = 512
HF_USERNAME    = "gauti0001"
HF_REPO        = f"{HF_USERNAME}/distilbert-goodreads-genres"
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
WANDB_PROJECT  = "mlops-assignment2"
WANDB_RUN_NAME = "distilbert-run-1"

wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config={
    "model": MODEL_NAME, "epochs": 3, "batch_size": 16,
    "learning_rate": 3e-5, "max_length": MAX_LENGTH, "dataset": "UCSD Goodreads",
})

print("=== Loading data ===")
(train_dataset, test_dataset, tokenizer), _ = load_and_prepare(
    model_name=MODEL_NAME, max_length=MAX_LENGTH,
    reviews_per_genre=200,   # <-- reduced from 1000 to 200 for speed
    use_cache=False
)

print("=== Loading model ===")
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(id2label), id2label=id2label, label2id=label2id)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
    run_name=WANDB_RUN_NAME,
)

trainer = Trainer(model=model, args=training_args,
                  train_dataset=train_dataset, eval_dataset=test_dataset,
                  compute_metrics=compute_metrics)

print("=== Training ===")
trainer.train()
trainer.save_model("./best_model")
tokenizer.save_pretrained("./best_model")

print("=== Pushing to HuggingFace ===")
login(token=HF_TOKEN)
model.push_to_hub(HF_REPO)
tokenizer.push_to_hub(HF_REPO)
wandb.run.summary["huggingface_model"] = f"https://huggingface.co/{HF_REPO}"
wandb.finish()
print("=== Done ===")
