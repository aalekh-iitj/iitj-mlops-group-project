import json, wandb
from transformers import DistilBertForSequenceClassification, Trainer
from sklearn.metrics import classification_report
from data import load_and_prepare
from utils import id2label, compute_metrics

wandb.init(project="mlops-assignment2", name="eval-run-1")

print("=== Loading data ===")
(_, test_dataset, _), _ = load_and_prepare(use_cache=False, reviews_per_genre=200)

print("=== Loading model ===")
model = DistilBertForSequenceClassification.from_pretrained("./best_model")
trainer = Trainer(model=model, compute_metrics=compute_metrics)

eval_results = trainer.evaluate(test_dataset)
print(eval_results)

wandb.log({
    "final/loss":     eval_results["eval_loss"],
    "final/accuracy": eval_results["eval_accuracy"],
    "final/f1":       eval_results["eval_f1"],
})

preds      = trainer.predict(test_dataset).predictions.argmax(-1)
labels_int = [item["labels"].item() for item in test_dataset]
labels_str = [id2label[l] for l in labels_int]
preds_str  = [id2label[p] for p in preds]

print(classification_report(labels_str, preds_str, target_names=list(id2label.values())))
report = classification_report(labels_str, preds_str, target_names=list(id2label.values()), output_dict=True)

with open("eval_report.json", "w") as f:
    json.dump(report, f, indent=2)

artifact = wandb.Artifact("eval-report", type="evaluation")
artifact.add_file("eval_report.json")
wandb.log_artifact(artifact)
wandb.finish()
print("=== Evaluation complete ===")
