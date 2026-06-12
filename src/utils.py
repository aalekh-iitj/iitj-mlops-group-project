"""
Utility functions and shared helpers
"""
import torch
from torch.utils.data import Dataset
import json

class TextClassificationDataset(Dataset):
    """Custom dataset for text classification"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    """Save data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def get_device():
    """Get available device (GPU or CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def print_model_info(model):
    """Print model information"""
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Device: {next(model.parameters()).device}")

if __name__ == "__main__":
    print("Utils module loaded successfully")