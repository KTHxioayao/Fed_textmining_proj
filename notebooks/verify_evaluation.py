import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from transformers import pipeline
import sys

# 1. Load Data
try:
    df_gold = pd.read_csv('../data/gold_standard_to_label_150_labeled.csv')
    df_gold['label'] = df_gold['label'].str.lower().str.strip()
    print(f"Loaded {len(df_gold)} records.")
except FileNotFoundError:
    print("Error: '../data/gold_standard_labeled.csv' not found. Cannot run verification.")
    sys.exit(1)

# 2. Setup Models
POS_WORDS = set(['strong', 'growth', 'improvement', 'gain', 'solid', 'recovery', 'stable', 'progress', 'positive', 'confident', 'robust'])
NEG_WORDS = set(['weak', 'recession', 'decline', 'loss', 'difficult', 'negative', 'risk', 'inflation', 'tight', 'slow', 'uncertainty', 'deterioration'])

def baseline_predict(text):
    text = str(text).lower()
    words = text.split()
    pos_count = sum(1 for w in words if w in POS_WORDS)
    neg_count = sum(1 for w in words if w in NEG_WORDS)
    if pos_count > neg_count: return 'positive'
    elif neg_count > pos_count: return 'negative'
    else: return 'neutral'

print("Loading FinBERT...")
model_name = "ProsusAI/finbert"
nlp = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, framework="pt")

def finbert_predict(text):
    try: 
        res = nlp(text[:512])[0]
        return res['label'].lower()
    except:
        return 'neutral'

print("Loading RoBERTa (Large)...")
roberta_model = "j-hartmann/sentiment-roberta-large-english-3-classes"
nlp_roberta = pipeline("sentiment-analysis", model=roberta_model, tokenizer=roberta_model, framework="pt")

def roberta_predict(text):
    try:
        # Truncate to 512 tokens as RoBERTa has similar limits
        res = nlp_roberta(text[:512])[0]
        return res['label'].lower()
    except Exception as e:
        print(f"Error: {e}")
        return 'neutral'

print("Running Predictions...")
df_gold['baseline_pred'] = df_gold['text'].apply(baseline_predict)
df_gold['finbert_pred'] = df_gold['text'].apply(finbert_predict)
df_gold['roberta_pred'] = df_gold['text'].apply(roberta_predict)

# 3. Stratified Analysis
sources = ['Speech', 'Minutes', 'Press Conf']
print("\n" + "="*50)
print("STRATIFIED EVALUATION RESULTS")
print("="*50)

for src in sources:
    subset = df_gold[df_gold['source'] == src]
    if len(subset) == 0:
        print(f"\n[!] No data for {src}")
        continue
        
    acc_fin = accuracy_score(subset['label'], subset['finbert_pred'])
    acc_rob = accuracy_score(subset['label'], subset['roberta_pred'])
    
    print(f"\n>>> Source: {src.upper()} (n={len(subset)})")
    print(f"FinBERT Accuracy: {acc_fin:.2%}")
    print(f"RoBERTa Accuracy: {acc_rob:.2%}")
    
    print("Classification Report (FinBERT):")
    print(classification_report(subset['label'], subset['finbert_pred'], zero_division=0))
    print("Classification Report (RoBERTa):")
    print(classification_report(subset['label'], subset['roberta_pred'], zero_division=0))
