import pandas as pd
import re
import nltk
import os

# Download nltk tokenizer data if first run
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_data(file_path):
    """Load raw CSV data"""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    # Convert date format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Ensure text column is string
    df['text'] = df['text'].astype(str)
    
    print(f"Successfully loaded {len(df)} documents.")
    return df

def extract_sections(text):
    """
    Core Logic: Parse specific sections of FOMC Minutes
    Returns a list: [{'section_name': 'Staff Review', 'text': '...'}, ...]
    """
    
    # 1. Define "High Value" section headers (Standardized Naming)
    # Keys are standardized column names, values are possible header keywords in text
    # Note: Arranged by approximate order of appearance, but code sorts by actual index
    section_patterns = {
        "Developments in Financial Markets": [
            "Developments in Financial Markets and Open Market Operations"
        ],
        "Inflation Analysis": [
            "Inflation Analysis and Forecasting"
        ],
        "Staff Review of Economic Situation": [
            "Staff Review of the Economic Situation", 
            "The information reviewed for the" # Sometimes this section has no header, starts directly with this phrase
        ],
        "Staff Review of Financial Situation": [
            "Staff Review of the Financial Situation"
        ],
        "Staff Economic Outlook": [
            "Staff Economic Outlook"
        ],
        "Participants' Views": [
            "Participants' Views on Current Conditions and the Economic Outlook", # Full title
            "Participants' Views on Current Conditions",
            "Participants’ Views on Current Conditions", # Smart quotes
            "Discussion of Monetary Policy" # Common in older minutes
        ],
        "Committee Policy Action": [
            "Committee Policy Action"
        ]
    }
    
    # 2. Find positions of all headers in the text
    matches = []
    text_lower = text.lower()
    
    for section_name, keywords in section_patterns.items():
        for keyword in keywords:
            idx = text_lower.find(keyword.lower())
            if idx != -1:
                matches.append({
                    "section_name": section_name,
                    "start_index": idx,
                    "header_length": len(keyword), # Record length to skip header
                    "priority": idx 
                })
                # Once a keyword is found, the section is located, skip other aliases
                break 
    
    # 3. Sort by appearance order in text
    if not matches:
        return []
    
    matches.sort(key=lambda x: x['priority'])
    
    # 4. Split text
    extracted_data = []
    
    for i in range(len(matches)):
        current_match = matches[i]
        start = current_match['start_index']
        # Skip the header itself
        content_start = start + current_match['header_length']
        
        # End position is the start of the next section, or end of text
        if i < len(matches) - 1:
            end = matches[i+1]['start_index']
        else:
            end = len(text)
            
        # Extract section text (remove header)
        section_text = text[content_start:end].strip()
        
        extracted_data.append({
            "section_name": current_match['section_name'],
            "section_text": section_text
        })
        
    return extracted_data

def segment_sentences(df):
    """
    Convert document-level DataFrame to sentence-level DataFrame, adding 'section' column
    """
    processed_rows = []
    
    print("Segmenting text into sentences with section labels...")
    
    for idx, row in df.iterrows():
        doc_text = row['text']
        date = row['date']
        doc_id = idx
        
        # 1. Extract sections (This automatically filters out administrative content not in the list)
        sections = extract_sections(doc_text)
        
        # If no sections extracted (possibly format too old or too new), skip or log for safety
        if not sections:
            # Can add fallback logic here, e.g., keep full text
            continue
            
        for section in sections:
            section_name = section['section_name']
            section_content = section['section_text']
            
            # Clean up extra whitespace in text
            section_content = re.sub(r'\s+', ' ', section_content).strip()
            
            # 2. Split sentences
            sentences = nltk.sent_tokenize(section_content)
            
            for sent in sentences:
                sent = sent.strip()
                
                # 3. Sentence-level filtering
                if len(sent.split()) < 5: 
                    continue
                
                # Filter administrative noise (may appear even within sections)
                lower_sent = sent.lower()
                noise_phrases = [
                    "meeting adjourned", 
                    "vote against", 
                    "voting for this action",
                    "voting against this action"
                ]
                
                if any(phrase in lower_sent for phrase in noise_phrases):
                    continue
                
                processed_rows.append({
                    'original_doc_id': doc_id,
                    'date': date,
                    'section': section_name, # New column!
                    'sentence_text': sent,
                    'source_type': 'Minutes'
                })
            
    return pd.DataFrame(processed_rows)

if __name__ == "__main__":
    # Configure paths
    INPUT_PATH = r"e:\Textming\data\raw\fed_minutes.csv"
    OUTPUT_PATH = r"e:\Textming\data\processed\fed_minutes_sentences_structured.csv"
    
    df_raw = load_data(INPUT_PATH)
    
    if df_raw is not None:
        df_sentences = segment_sentences(df_raw)
        
        print("\n--- Processing Complete ---")
        print(f"Original Documents: {len(df_raw)}")
        print(f"Generated Sentences: {len(df_sentences)}")
        
        print("\nDistribution of Sections:")
        if 'section' in df_sentences.columns:
            print(df_sentences['section'].value_counts())
        
        print("\nSample Data:")
        display_cols = ['date', 'section', 'sentence_text']
        print(df_sentences[display_cols].head(10))
        
        df_sentences.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
        print(f"\nSaved structured data to: {OUTPUT_PATH}")