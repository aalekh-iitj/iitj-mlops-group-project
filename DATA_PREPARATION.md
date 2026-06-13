# Data Preparation & Normalization

## Dataset Selection

**Dataset**: IMDB Movie Reviews  
**Source**: https://huggingface.co/datasets/imdb  
**Task**: Binary Sentiment Classification  
**Size**: 25,000 movie reviews  
**Classes**: Positive, Negative  

### Why This Dataset?

1. **Appropriate Size**: 25,000 samples fits well within Kaggle's free GPU limits
2. **Balanced Classes**: 50% positive, 50% negative reviews
3. **Well-Established**: Standard benchmark for sentiment analysis
4. **Text Modality**: Suitable for transformer models (DistilBERT, RoBERTa, etc.)

---

## Data Inspection (Task 2 - 3 marks)

### Raw Data Structure

```
Total Samples: 25,000
Features: text, label
Columns: 2
Data Type: Text strings + categorical labels
```

### Size & Distribution

| Metric | Value |
|--------|-------|
| Total Samples | 25,000 |
| Training Set | 20,000 (80%) |
| Test Set | 5,000 (20%) |
| Positive Reviews | 12,500 (50%) |
| Negative Reviews | 12,500 (50%) |

### Text Statistics

| Statistic | Value |
|-----------|-------|
| Average Length | 1,325 characters |
| Minimum Length | 28 characters |
| Maximum Length | 13,704 characters |

### Quality Issues Found

1. **No null values** ✓
2. **No duplicate reviews** ✓
3. **Balanced classes** ✓
4. **Mixed case text** - needs normalization
5. **Extra whitespace** - needs cleaning

---

## Data Cleaning & Normalization (Task 2 - 6 marks)

### Modality: Text

Applied the following cleaning steps as appropriate for text data:

#### 1. Remove Null Values
```python
df = df.dropna()
```
**Result**: 0 null values found

#### 2. Remove Duplicates
```python
df = df.drop_duplicates()
```
**Result**: 0 duplicates found

#### 3. Lowercase Conversion
```python
df['text'] = df['text'].str.lower()
```
**Justification**: Reduces vocabulary size, ensures "Good" and "good" are treated as the same token

#### 4. Remove Extra Whitespace
```python
df['text'] = df['text'].str.strip()
df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
```
**Justification**: Normalizes spacing for consistent tokenization

### Cleaning Decisions Summary

| Step | Action | Justification |
|------|--------|---------------|
| Nulls | Drop rows with missing values | Ensure clean input for model |
| Duplicates | Remove duplicate reviews | Prevent data leakage and overfitting |
| Lowercase | Convert all text to lowercase | Reduce vocabulary complexity |
| Whitespace | Normalize spacing | Consistent tokenization |
| Punctuation | Keep as-is | Important for sentiment (e.g., "!!!" shows emphasis) |

---

## Label Encoding (Task 2 - 3 marks)

### Encoding Scheme

Original categorical labels were encoded to integers:

```python
label2id = {
    "negative": 0,
    "positive": 1
}

id2label = {
    "0": "negative",
    "1": "positive"
}
```

### Files Saved

- ✅ `id2label.json` - Maps integer IDs back to label names (for inference)
- ✅ `label2id.json` - Maps label names to integer IDs (for training)
- ✅ `data_summary.json` - Complete dataset statistics and metadata

---

## Dataset Storage (Task 2 - 3 marks)

### Files Committed to Repository

✅ **Committed**:
- `id2label.json` - Label mappings (162 bytes)
- `label2id.json` - Reverse mappings (162 bytes)
- `data_summary.json` - Dataset statistics (1.2 KB)
- `prepare_dataset.py` - Data preparation script
- `src/data.py` - Reusable data processing module

❌ **Not Committed** (excluded via .gitignore):
- `data/train.csv` - Training data (~45 MB)
- `data/test.csv` - Test data (~11 MB)

**Justification**: Large CSV files are excluded from Git repository. The data preparation script (`prepare_dataset.py`) can regenerate these files from the Hugging Face dataset source. This follows best practices for version control.

---

## Usage Instructions

### Running Data Preparation

**On Kaggle Notebook** (recommended):

```python
# Install dependencies
!pip install datasets transformers pandas scikit-learn

# Run preparation script
!python prepare_dataset.py
```

**Locally** (if environment is set up):

```bash
pip install -r requirements.txt
python prepare_dataset.py
```

### Using the Data Module

```python
from src.data import (
    load_data,
    preprocess_data,
    encode_labels,
    create_train_test_split,
    save_mappings
)

# Load data
df = load_data('data/train.csv')

# Preprocess
df_clean = preprocess_data(df)

# Encode labels
df_encoded, label2id, id2label = encode_labels(df_clean, 'label')

# Split
train_df, test_df = create_train_test_split(df_encoded, test_size=0.2)
```

---

## Data Pipeline Summary

```
┌─────────────────────┐
│ IMDB Dataset (HF)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Load Raw Data       │ (25,000 samples)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Inspect & Validate  │ (Check nulls, duplicates, distribution)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Clean & Normalize   │ (Lowercase, trim whitespace)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Encode Labels       │ (negative=0, positive=1)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Train/Test Split    │ (80/20 split)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Save Artifacts      │ (CSV files + JSON mappings)
└─────────────────────┘
```

---

## Validation Results

✅ All quality checks passed:
- No null values
- No duplicate entries
- Balanced class distribution (50/50)
- Consistent text formatting
- Valid label encoding
- Proper train/test split ratio

Dataset is ready for model training! 🚀
