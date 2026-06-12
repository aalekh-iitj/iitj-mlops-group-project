"""
Inference script - loads model from Hugging Face and classifies input text
"""
import os
from transformers import pipeline

HF_MODEL = os.environ.get("HF_MODEL_NAME", "rnema4/distilbert-agnews-topics")
INPUT_TEXT = os.environ.get("INPUT_TEXT", "NASA launches new satellite into orbit")

def run_inference(text, model_name):
    """Run inference on input text"""
    classifier = pipeline("text-classification", model=model_name)
    result = classifier(text)
    return result

if __name__ == "__main__":
    print(f"Model: {HF_MODEL}")
    print(f"Input: {INPUT_TEXT}")
    result = run_inference(INPUT_TEXT, HF_MODEL)
    print(f"Prediction: {result}")