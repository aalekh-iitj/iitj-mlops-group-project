import torch
from sklearn.metrics import accuracy_score, f1_score

GENRES = [
    "poetry", "children", "comics_graphic", "fantasy_paranormal",
    "history_biography", "mystery_thriller_crime", "romance", "young_adult",
]

label2id = {genre: idx for idx, genre in enumerate(GENRES)}
id2label  = {idx: genre for genre, idx in label2id.items()}

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted"),
    }
