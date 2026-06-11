"""
Dataset Preparation Script
Downloads, inspects, cleans, and prepares the IMDB sentiment dataset for training
"""
import os
import pandas as pd
import numpy as np
from src.data import (
    load_data,
    preprocess_data,
    create_train_test_split,
    encode_labels,
    save_mappings
)
from datasets import load_dataset
import json

def download_and_prepare_imdb():
    """
    Download IMDB dataset and prepare it for training
    Dataset: 25,000 movie reviews for sentiment classification (positive/negative)
    """
    print("=" * 60)
    print("STEP 1: Downloading IMDB Dataset")
    print("=" * 60)
    
    # Load IMDB dataset from Hugging Face
    dataset = load_dataset('imdb', split='train')
    
    # Convert to pandas DataFrame
    df = pd.DataFrame({
        'text': dataset['text'],
        'label': ['positive' if label == 1 else 'negative' for label in dataset['label']]
    })
    
    print(f"Dataset loaded: {len(df)} samples")
    
    # STEP 2: Inspect Raw Data
    print("\n" + "=" * 60)
    print("STEP 2: Inspecting Raw Data")
    print("=" * 60)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    
    print(f"\nClass Distribution:")
    print(df['label'].value_counts())
    print(f"\nClass Distribution (%):")
    print(df['label'].value_counts(normalize=True) * 100)
    
    print(f"\nSample Review (first 200 chars):")
    print(df['text'].iloc[0][:200] + "...")
    
    print(f"\nText Length Statistics:")
    df['text_length'] = df['text'].str.len()
    print(f"Mean: {df['text_length'].mean():.2f}")
    print(f"Median: {df['text_length'].median():.2f}")
    print(f"Min: {df['text_length'].min()}")
    print(f"Max: {df['text_length'].max()}")
    
    # STEP 3: Clean and Preprocess Data
    print("\n" + "=" * 60)
    print("STEP 3: Cleaning and Preprocessing")
    print("=" * 60)
    
    print("\nCleaning steps:")
    print("- Removing null values")
    print("- Removing duplicate reviews")
    print("- Converting text to lowercase")
    print("- Removing extra whitespace")
    
    # Use the preprocess_data function from data.py
    df_clean = preprocess_data(df)
    
    # Additional text-specific cleaning
    df_clean['text'] = df_clean['text'].str.lower()
    df_clean['text'] = df_clean['text'].str.strip()
    df_clean['text'] = df_clean['text'].str.replace(r'\s+', ' ', regex=True)
    
    print(f"\nRows before cleaning: {len(df)}")
    print(f"Rows after cleaning: {len(df_clean)}")
    print(f"Rows removed: {len(df) - len(df_clean)}")
    
    # STEP 4: Encode Labels
    print("\n" + "=" * 60)
    print("STEP 4: Encoding Labels")
    print("=" * 60)
    
    df_clean, label2id, id2label = encode_labels(df_clean, 'label')
    
    print(f"\nLabel Encoding:")
    print(f"label2id: {label2id}")
    print(f"id2label: {id2label}")
    
    # STEP 5: Save Mappings
    print("\n" + "=" * 60)
    print("STEP 5: Saving Label Mappings")
    print("=" * 60)
    
    save_mappings(id2label, 'id2label.json')
    
    # Also save label2id for reference
    with open('label2id.json', 'w') as f:
        json.dump(label2id, f, indent=2)
    print(f"Saved label2id mappings to label2id.json")
    
    # STEP 6: Create Train/Test Split
    print("\n" + "=" * 60)
    print("STEP 6: Creating Train/Test Split")
    print("=" * 60)
    
    train_df, test_df = create_train_test_split(df_clean, test_size=0.2, random_state=42)
    
    print(f"\nTrain set: {len(train_df)} samples ({len(train_df)/len(df_clean)*100:.1f}%)")
    print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df_clean)*100:.1f}%)")
    
    print(f"\nTrain set class distribution:")
    print(train_df['label'].value_counts())
    
    print(f"\nTest set class distribution:")
    print(test_df['label'].value_counts())
    
    # STEP 7: Save Prepared Data
    print("\n" + "=" * 60)
    print("STEP 7: Saving Prepared Dataset")
    print("=" * 60)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print(f"Saved train.csv: {len(train_df)} samples")
    print(f"Saved test.csv: {len(test_df)} samples")
    
    # Generate summary statistics
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    summary = {
        "dataset_name": "IMDB Movie Reviews",
        "task": "Sentiment Classification (Binary)",
        "total_samples": len(df_clean),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "num_classes": len(id2label),
        "classes": list(label2id.keys()),
        "avg_text_length": float(df_clean['text_length'].mean()),
        "cleaning_steps": [
            "Removed null values",
            "Removed duplicate reviews",
            "Converted text to lowercase",
            "Removed extra whitespace"
        ]
    }
    
    with open('data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDataset: {summary['dataset_name']}")
    print(f"Task: {summary['task']}")
    print(f"Total Samples: {summary['total_samples']}")
    print(f"Train/Test Split: {summary['train_samples']}/{summary['test_samples']}")
    print(f"Number of Classes: {summary['num_classes']}")
    print(f"Classes: {summary['classes']}")
    print(f"Average Text Length: {summary['avg_text_length']:.2f} characters")
    
    print("\n✅ Data preparation complete!")
    print("\nGenerated files:")
    print("  - id2label.json (label mappings - commit this)")
    print("  - label2id.json (reverse mappings - commit this)")
    print("  - data_summary.json (dataset statistics - commit this)")
    print("  - data/train.csv (training data - DO NOT commit)")
    print("  - data/test.csv (test data - DO NOT commit)")
    
    return df_clean, train_df, test_df, id2label

if __name__ == "__main__":
    try:
        df_clean, train_df, test_df, id2label = download_and_prepare_imdb()
        print("\n🎉 Success! Dataset is ready for training.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
