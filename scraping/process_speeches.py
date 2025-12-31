import pandas as pd
import os
import re
import sys
from datetime import datetime

# Path setup to import utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # e.g. e:\Textming
sys.path.append(PROJECT_ROOT)

from utils.utilities import (
    split_into_sentences_robust, 
    clean_common_noise
)

# Configuration
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "fed_speeches.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "fed_speeches_sentences.csv")

def extract_date_from_url(url):
    """
    Extracts date from URL.
    Expected format: .../powell20181206a.htm -> 2018-12-06
    """
    if not isinstance(url, str):
        return None
    
    # Look for 8 digits in the URL (YYYYMMDD)
    match = re.search(r'(\d{4})(\d{2})(\d{2})', url)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"
    return None

def process_speeches():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return
    else:
        df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} raw speech segments.")

    processed_rows = []

    # Check if 'text' column exists, if not try 'text_segment'
    text_col = 'text' if 'text' in df.columns else 'text_segment'
    if text_col not in df.columns:
        print(f"Error: Neither 'text' nor 'text_segment' column found in {INPUT_FILE}")
        return

    for idx, row in df.iterrows():
        text = row.get(text_col, '') 
        if not isinstance(text, str):
            continue

        # Extract date from URL if possible
        url = row.get('url', '')
        extracted_date = extract_date_from_url(url)
        
        # Use extracted date if available, otherwise fallback to existing date column
        # If existing date is 'Unknown' or missing, extracted_date takes precedence
        current_date = row.get('date')
        final_date = extracted_date if extracted_date else current_date

        # 1. Pre-cleaning (remove headers, references) using shared utility
        # We can perform additional specific cleaning here if needed
        cleaned_text = clean_common_noise(text)
        
        # Additional speech specific cleaning: remove lines starting with citation style numbers if not handled by utils
        # (Though clean_common_noise handles headers, let's keep the citation check if it's specific)
        if re.match(r'^\d+\.\s+[A-Z].*\(\d{4}\)', cleaned_text):
            continue

        if not cleaned_text:
            continue

        # 2. Robust Sentence Segmentation
        # This utility takes care of splitting, noise filtering (short sentences), and recursive splitting
        sentences = split_into_sentences_robust(cleaned_text)

        for sent in sentences:
            processed_rows.append({
                'date': final_date,
                'title': row.get('title', 'Unknown'),
                'text': sent,
                'source_type': 'Speech',
                'url': url # Keeping URL for reference
            })

    # Save
    df_out = pd.DataFrame(processed_rows)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"Processing complete. Saved {len(df_out)} sentences to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_speeches()