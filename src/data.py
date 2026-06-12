"""
Data loading and preprocessing module
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os

def load_data(file_path):
    """Load dataset from CSV or JSON file"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format")

def preprocess_data(df):
    """Clean and preprocess the data"""
    # Remove nulls
    df = df.dropna()
    # Remove duplicates
    df = df.drop_duplicates()
    return df

def create_train_test_split(df, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )
    return train_df, test_df

def encode_labels(df, label_column):
    """Encode labels to integers"""
    unique_labels = sorted(df[label_column].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    df['label_id'] = df[label_column].map(label2id)
    return df, label2id, id2label

def save_mappings(id2label, output_path='id2label.json'):
    """Save label mappings to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(id2label, f, indent=2)
    print(f'Saved label mappings to {output_path}')

if __name__ == "__main__":
    # Example usage
    print("Data preprocessing module loaded successfully")
