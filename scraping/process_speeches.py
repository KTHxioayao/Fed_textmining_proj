import pandas as pd
import nltk
import os
import re
from datetime import datetime

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('punkt_tab')

# Configuration
# We assume this script is running from e:\Textming\Fed_Project\scraping or root
# Adjust paths relative to this script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # Go up one level to Fed_Project

# CHANGED: Read input from 'data' folder instead of 'scraping' folder
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "fed_speeches.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "fed_speeches_sentences.csv")

def clean_text(text):
    """
    Basic text cleaning before segmentation.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove "Return to text" references specifically
    # Pattern: Number. Name (Year), "Title," ... Return to text
    # We look for lines ending with "Return to text" or starting with a number and looking like a citation
    if "Return to text" in text:
        return ""
    
    # Remove lines that look purely like citations (e.g., "1. Author (Year)...")
    if re.match(r'^\d+\.\s+[A-Z].*\(\d{4}\)', text):
        return ""

    return text.strip()

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

        # 1. Pre-cleaning (remove references)
        cleaned_text = clean_text(text)
        if not cleaned_text:
            continue

        # 2. Sentence Segmentation
        sentences = nltk.sent_tokenize(cleaned_text)

        for sent in sentences:
            sent = sent.strip()
            
            # 3. Filter short sentences or noise
            if len(sent.split()) < 5:
                continue
            
            # Double check for "Return to text" artifacts inside sentences
            if "Return to text" in sent:
                continue

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