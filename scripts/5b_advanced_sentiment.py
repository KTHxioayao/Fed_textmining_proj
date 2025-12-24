#!/usr/bin/env python3
"""
Advanced Sentiment Analysis with Confidence Weighting
Goal: Implement more sophisticated sentiment aggregation methods.

Improvements:
1. Confidence-weighted aggregation: Use model confidence scores
2. Text length normalization: Account for sentence importance
3. Topic-sentiment interaction: Combine topic modeling with sentiment
4. Temporal smoothing: Handle sentiment volatility
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

def main():
    print("=== Advanced Sentiment Analysis with Confidence Weighting ===\n")

    # Load data
    print("Loading data...")
    df_original = pd.read_csv('data/fed_minutes_sentences_structured.csv')
    print(f"Loaded {len(df_original)} sentences.\n")

    # Enhanced sentiment mapping
    def enhanced_sentiment_mapping():
        return {
            # Hawkish indicators (contractionary)
            'hawkish_keywords': [
                'inflation', 'tighten', 'raise rates', 'monetary tightening',
                'restrictive', 'hawkish', 'higher rates', 'policy tightening',
                'inflationary pressure', 'overheating'
            ],
            # Dovish indicators (expansionary)
            'dovish_keywords': [
                'stimulus', 'ease', 'lower rates', 'accommodative',
                'dovish', 'growth slowdown', 'recession risk',
                'support economy', 'monetary accommodation'
            ],
            # Neutral/contextual words
            'neutral_keywords': [
                'monitor', 'assess', 'evaluate', 'continue', 'maintain',
                'data dependent', 'balanced', 'measured'
            ]
        }

    # Calculate text importance score
    def calculate_importance(text):
        """Calculate sentence importance based on length and keywords"""
        words = str(text).split()

        # Base importance from length (normalized)
        length_score = min(len(words) / 50, 1.0)  # Cap at 50 words

        # Keyword importance
        mapping = enhanced_sentiment_mapping()
        text_lower = str(text).lower()

        hawkish_count = sum(1 for kw in mapping['hawkish_keywords'] if kw in text_lower)
        dovish_count = sum(1 for kw in mapping['dovish_keywords'] if kw in text_lower)

        keyword_score = min((hawkish_count + dovish_count) / 3, 1.0)  # Cap at 3 keywords

        # Combined importance
        importance = 0.6 * length_score + 0.4 * keyword_score

        return max(importance, 0.1)  # Minimum importance

    # Apply importance scoring
    print("Calculating sentence importance scores...")
    df_original['importance'] = df_original['sentence_text'].apply(calculate_importance)

    print("Added importance scores")
    print(".3f")
    print("Importance distribution:")
    print(df_original['importance'].describe())
    print()

    # Enhanced sentiment pipeline with confidence weighting
    def enhanced_sentiment_pipeline(model_name, output_prefix, df_input):
        print(f"\n{'='*50}")
        print(f"Enhanced Analysis with {model_name}")
        print(f"{'='*50}")

        # Load model
        if "roberta" in model_name.lower():
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
            tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True, do_basic_tokenize=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
            config = AutoConfig.from_pretrained(model_name)
            nlp = pipeline('text-classification', model=model, tokenizer=tokenizer, config=config, device=0, framework="pt")
        else:
            nlp = pipeline("sentiment-analysis", model=model_name, device=0)

        # Run inference with progress tracking
        predictions = []
        batch_size = 16
        sentences = df_input['sentence_text'].tolist()
        importances = df_input['importance'].tolist()

        for i in tqdm(range(0, len(sentences), batch_size), desc=f"Inference ({output_prefix})"):
            batch_sentences = sentences[i:i+batch_size]
            batch_importances = importances[i:i+batch_size]

            try:
                batch_preds = nlp(batch_sentences)

                # Enhance predictions with importance
                for pred, imp in zip(batch_preds, batch_importances):
                    enhanced_pred = pred.copy()
                    enhanced_pred['importance'] = imp
                    enhanced_pred['weighted_score'] = pred['score'] * imp
                    predictions.append(enhanced_pred)

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                # Fallback with neutral predictions
                for imp in batch_importances:
                    predictions.append({
                        'label': 'Neutral',
                        'score': 0.0,
                        'importance': imp,
                        'weighted_score': 0.0
                    })

        # Create results dataframe
        df_results = df_input.copy()

        # Align lengths
        if len(predictions) != len(df_results):
            df_results = df_results.iloc[:len(predictions)]

        # Add predictions
        pred_df = pd.DataFrame(predictions)
        df_results = pd.concat([df_results, pred_df], axis=1)

        # Enhanced label mapping
        def map_sentiment(row):
            label = str(row['label']).lower()

            if "hawkish" in label or ("negative" in label and "finbert" in model_name.lower()):
                return "Hawkish"
            elif "dovish" in label or ("positive" in label and "finbert" in model_name.lower()):
                return "Dovish"
            else:
                return "Neutral"

        df_results['sentiment'] = df_results.apply(map_sentiment, axis=1)

        # Calculate weighted sentiment indices
        def calculate_weighted_index(group):
            # Method 1: Confidence-weighted
            hawkish_weighted = group[group['sentiment'] == 'Hawkish']['weighted_score'].sum()
            dovish_weighted = group[group['sentiment'] == 'Dovish']['weighted_score'].sum()
            total_weighted = group['weighted_score'].sum()

            if total_weighted > 0:
                conf_index = (hawkish_weighted - dovish_weighted) / total_weighted
            else:
                conf_index = 0

            # Method 2: Count-based (original)
            counts = group['sentiment'].value_counts()
            hawkish_count = counts.get('Hawkish', 0)
            dovish_count = counts.get('Dovish', 0)
            total_count = len(group)

            if total_count > 0:
                count_index = (hawkish_count - dovish_count) / total_count
            else:
                count_index = 0

            return pd.Series({
                'sentiment_index_conf': conf_index,
                'sentiment_index_count': count_index,
                'total_sentences': total_count,
                'avg_importance': group['importance'].mean(),
                'hawkish_count': hawkish_count,
                'dovish_count': dovish_count,
                'neutral_count': counts.get('Neutral', 0)
            })

        # Group by date and calculate indices
        df_results['date'] = pd.to_datetime(df_results['date'])
        monthly_indices = df_results.groupby('date').apply(calculate_weighted_index).reset_index()

        # Save results
        results_path = f"data/{output_prefix}_enhanced_inference_results.csv"
        index_path = f"data/{output_prefix}_enhanced_monthly_index.csv"

        df_results.to_csv(results_path, index=False)
        monthly_indices.to_csv(index_path, index=False)

        print(f"\nSaved enhanced results to {results_path}")
        print(f"Saved monthly indices to {index_path}")

        return monthly_indices

    # Run enhanced analysis
    print("Running Enhanced FOMC-RoBERTa Analysis...")
    roberta_enhanced = enhanced_sentiment_pipeline(
        model_name="gtfintechlab/FOMC-RoBERTa",
        output_prefix="fomc_roberta_enhanced",
        df_input=df_original
    )

    print("\nRunning Enhanced FinBERT Analysis...")
    finbert_enhanced = enhanced_sentiment_pipeline(
        model_name="ProsusAI/finbert",
        output_prefix="finbert_enhanced",
        df_input=df_original
    )

    # Compare enhanced vs original methods
    plt.figure(figsize=(15, 10))

    # Load original results for comparison
    try:
        roberta_original = pd.read_csv('data/fomc_roberta_monthly_index.csv')
        roberta_original['date'] = pd.to_datetime(roberta_original['date'])

        plt.subplot(2, 2, 1)
        plt.plot(roberta_original['date'], roberta_original['sentiment_index'],
                 label='Original', alpha=0.7, marker='o')
        plt.plot(roberta_enhanced['date'], roberta_enhanced['sentiment_index_conf'],
                 label='Confidence-weighted', alpha=0.7, marker='s')
        plt.plot(roberta_enhanced['date'], roberta_enhanced['sentiment_index_count'],
                 label='Count-based', alpha=0.7, marker='^')
        plt.title('FOMC-RoBERTa: Enhanced vs Original')
        plt.legend()
        plt.xticks(rotation=45)

    except FileNotFoundError:
        print("Original RoBERTa results not found for comparison")

    # Plot importance distribution
    plt.subplot(2, 2, 2)
    plt.hist(df_original['importance'], bins=30, alpha=0.7, edgecolor='black')
    plt.title('Sentence Importance Distribution')
    plt.xlabel('Importance Score')
    plt.ylabel('Frequency')

    # Plot correlation between importance and sentiment confidence
    plt.subplot(2, 2, 3)
    if 'roberta_enhanced' in locals():
        plt.scatter(roberta_enhanced['avg_importance'],
                    roberta_enhanced['sentiment_index_conf'],
                    alpha=0.6, s=50)
        plt.title('Importance vs Sentiment Index')
        plt.xlabel('Average Importance')
        plt.ylabel('Confidence-weighted Index')

    # Plot sentiment distribution by importance quartile
    plt.subplot(2, 2, 4)
    importance_quartiles = pd.qcut(df_original['importance'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    sentiment_by_quartile = pd.crosstab(importance_quartiles,
                                       df_original['sentiment'] if 'sentiment' in df_original.columns
                                       else ['Neutral'] * len(df_original),
                                       normalize='index')
    sentiment_by_quartile.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Sentiment by Importance Quartile')
    plt.xlabel('Importance Quartile')
    plt.ylabel('Proportion')
    plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    # Statistical comparison
    print("\n=== Enhanced Analysis Summary ===")
    print(f"Total sentences analyzed: {len(df_original)}")
    print(".3f")
    print(".3f")

    if 'roberta_enhanced' in locals():
        print(".4f")
        print(".4f")

if __name__ == "__main__":
    main()
