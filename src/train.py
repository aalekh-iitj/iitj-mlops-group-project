""" 
Model training module with W&B integration 
""" 
import wandb 
import torch 
from transformers import ( 
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification, 
    TrainingArguments, 
    Trainer 
) 
from sklearn.metrics import accuracy_score, f1_score 
import json 
import os 
  
def load_model_and_tokenizer(model_name, num_labels): 
    """Load pre-trained model and tokenizer""" 
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name) 
    model = DistilBertForSequenceClassification.from_pretrained( 
        model_name, 
        num_labels=num_labels 
    ) 
    return model, tokenizer 
  
def compute_metrics(pred): 
    """Compute evaluation metrics""" 
    labels = pred.label_ids 
    preds = pred.predictions.argmax(-1) 
     
    accuracy = accuracy_score(labels, preds) 
    f1 = f1_score(labels, preds, average='weighted') 
     
    return { 
        'accuracy': accuracy, 
        'f1_score': f1 
    } 
  
def initialize_wandb(project_name, run_name, config): 
    """Initialize Weights & Biases tracking""" 
    wandb.init( 
        project=project_name, 
        name=run_name, 
        config=config 
    ) 
    return wandb.run 
  
def create_training_args(output_dir, **kwargs): 
    """Create training arguments""" 
    default_args = { 
        'output_dir': output_dir, 
        'num_train_epochs': 3, 
        'per_device_train_batch_size': 16, 
        'per_device_eval_batch_size': 32, 
        'warmup_steps': 100, 
  'weight_decay': 0.01, 
        'logging_steps': 50, 
        'eval_strategy': 'epoch', 
        'save_strategy': 'epoch', 
        'load_best_model_at_end': True, 
        'report_to': 'wandb', 
        'run_name': 'distilbert-run-1' 
    } 
     
    # Update with any custom arguments 
    default_args.update(kwargs) 
     
    return TrainingArguments(**default_args) 
  
def train_model(model, tokenizer, train_dataset, eval_dataset, 
               training_args): 
    """Train the model""" 
    trainer = Trainer( 
        model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        compute_metrics=compute_metrics, 
    ) 
     
    trainer.train() 
    return trainer 
  
if __name__ == "__main__": 
    # Example configuration 
    config = { 
        'model_name': 'distilbert-base-cased', 
        'max_length': 512, 
        'num_labels': 8, 
        'epochs': 3, 
        'batch_size': 16, 
        'learning_rate': 3e-5 
    } 
     
    print("Training module loaded successfully") 
    print(f'Configuration: {config}')
