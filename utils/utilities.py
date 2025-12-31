import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK data (punkt) is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


# --- Text Processing Utilities ---

def is_noise_content(text):
    """
    Check if a sentence is conversational noise or irrelevant short text.
    Target: "Thank you", "You're on mute", "[No response]", etc.
    """
    noise_phrases = [
        "Thank you", "Thanks", "You're on mute", "Can you hear me", 
        "[No response]", "(No response)", "hearing no objection",
        "so moved", "second", "all in favor", "aye",
        "Return to text" # Common artifact in speeches
    ]
    text_lower = text.lower().strip("., ")
    
    # 1. Too short to be meaningful
    if len(text) < 5: 
        return True 
        
    # 2. Check for noise phrases
    for phrase in noise_phrases:
        if text_lower == phrase.lower(): 
            return True
        # If phrase is inside a short-ish sentence (e.g. "Thank you very much.")
        if len(text) < 30 and phrase.lower() in text_lower: 
            return True
            
    return False

def split_long_sentence(sentence):
    """
    Split overly long sentences (e.g. > 30 words) by punctuation.
    """
    # Split by full stops, question marks, exclamation marks, semicolons
    split_patterns = r'(?<=[.!?;])\s+'
    parts = re.split(split_patterns, sentence)
    valid_parts = []

    for part in parts:
        part = part.strip()
        # Ensure part is substantial
        if len(part.split()) >= 5 and len(part) >= 30 and not is_noise_content(part):
            # If still huge, try splitting by comma
            if len(part.split()) > 40:
                comma_parts = re.split(r'(?<=[,])\s+', part)
                for comma_part in comma_parts:
                    comma_part = comma_part.strip()
                    if len(comma_part.split()) >= 5 and len(comma_part) >= 20:
                        valid_parts.append(comma_part)
            else:
                valid_parts.append(part)

    return valid_parts if valid_parts else [sentence]

def force_split_long_segment(text):
    """
    Force split a text block if it's way too huge (e.g. > 100 words) and has no punctuation.
    """
    words = text.split()
    segments = []
    # Split every 50 words
    for i in range(0, len(words), 50):
        segment_words = words[i:i+50]
        segment = ' '.join(segment_words)
        if len(segment_words) >= 3:
            segments.append(segment)
    return segments

def preprocess_content(text):
    """
    Pre-process raw text block to get smaller, manageable paragraphs/segments.
    """
    # Split by double newlines or similar paragraph breaks
    paragraphs = re.split(r'\n\s*\n', text)
    processed_segments = []
    
    for para in paragraphs:
        para = para.strip()
        if not para: continue
        
        # If paragraph is long, try splitting by sentence terminators
        if len(para.split()) > 100:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent = sent.strip()
                if len(sent.split()) > 15:
                    sub_parts = re.split(r'(?<=[;:,])\s+', sent)
                    processed_segments.extend([p.strip() for p in sub_parts if p.strip() and len(p.split()) >= 3])
                else:
                    processed_segments.append(sent)
        else:
            processed_segments.append(para)
            
    # Final cleanup: limit size
    filtered_segments = []
    for seg in processed_segments:
        word_count = len(seg.split())
        if 3 <= word_count <= 100:
            filtered_segments.append(seg)
        elif word_count > 100:
            filtered_segments.extend(force_split_long_segment(seg))
            
    return filtered_segments

def split_into_sentences_robust(text):
    """
    Robust sentence splitting pipeline.
    1. Check if empty or short.
    2. NLTK sent_tokenize.
    3. Filter noise/short sentences.
    4. Recursively split long sentences.
    """
    if not text or len(text.strip()) < 10:
        return []

    sentences = sent_tokenize(text)
    valid_sentences = []

    for sent in sentences:
        sent = sent.strip()
        
        # Basic filtering
        if len(sent.split()) < 5 or len(sent) < 30:
            continue
        if is_noise_content(sent):
            continue
            
        # Long sentence handling
        if len(sent.split()) > 30:
            sub_sentences = split_long_sentence(sent)
            valid_sentences.extend(sub_sentences)
        else:
            valid_sentences.append(sent)
            
    return valid_sentences

def clean_common_noise(text):
    """
    Remove common PDF/Scraping artifacts (headers, page numbers).
    """
    lines = text.split('\n')
    cleaned = []
    patterns = [
        r"Page \d+ of \d+",   
        r"^FINAL$",           
        r"Transcript of .* Press Conference", 
        r"^[A-Z][a-z]+ \d{1,2}, 20\d{2}$",
        r"^SECTION HEADER", # Example placeholder
        r"Return to text"
    ]
    
    for line in lines:
        line = line.strip()
        if not line or line.isdigit(): continue
        
        is_noise = False
        for pat in patterns:
            if re.search(pat, line, re.IGNORECASE):
                is_noise = True
                break
        if not is_noise:
            cleaned.append(line)
            
    return " ".join(cleaned)

def calculate_net_sentiment_counts(group):
    """
    Calculate Net Sentiment Index using Hard Counts.
    Formula: (Hawkish-Dovish) / Total Counts
    Result: Positive = Hawkish, Negative = Dovish
    """
    # Determine column name
    col = 'sentiment' if 'sentiment' in group.columns else 'sentiment_label'
    
    if col not in group.columns:
        return 0
        
    counts = group[col].value_counts()
    
    # Get counts (defaults to 0 if missing)
    h = counts.get('Hawkish', 0)
    d = counts.get('Dovish', 0)
    n = counts.get('Neutral', 0)
    
    total = h + d + n
    
    if total == 0:
        return 0

    # 修正：(鹰派-鸽派) / 总数
    # 这样鹰派多的时候，结果才是正数
    return (h - d) / total


def calculate_net_sentiment_scores(group):
    # Sum scores instead of counts
    h_score = group.loc[group['sentiment'] == 'Hawkish', 'sentiment_score'].sum()
    d_score = group.loc[group['sentiment'] == 'Dovish', 'sentiment_score'].sum()

    # Count only Hawkish + Dovish
    h_count = (group['sentiment'] == 'Hawkish').sum()
    d_count = (group['sentiment'] == 'Dovish').sum()

    total = h_count + d_count
    if total == 0:
        return 0

    # 修正：(鹰派分数 - 鸽派分数)
    return (h_score-d_score ) / total

def get_sentiment_label_FinBERT_FOMC(raw_label):
    """
    Map FinBERT-FOMC labels to Hawkish/Dovish/Neutral.
    Correct Logic: 
    - Negative/Label_2 (Bad Economy) -> Dovish (Fed cuts rates)
    - Positive/Label_1 (Good Economy) -> Hawkish (Fed hikes rates)
    """
    label_clean = str(raw_label).upper().replace("LABEL_", "")
    
    # --- 1. 优先匹配文本 ---
    # 修正点：Negative (经济差) 意味着央行要放水 -> 鸽派 (Dovish)
    if 'NEGATIVE' in label_clean: 
        return 'Dovish'   
    # 修正点：Positive (经济好) 意味着央行要收紧 -> 鹰派 (Hawkish)
    if 'POSITIVE' in label_clean: 
        return 'Hawkish'  
    if 'NEUTRAL' in label_clean:
        return 'Neutral'

    # --- 2. 匹配数字标签 (基于你提供的 Paper: 0=Neutral, 1=Positive, 2=Negative) ---
    if label_clean == '0': 
        return 'Neutral'
    # 1对应Positive -> 经济好 -> 鹰派
    elif label_clean == '1': 
        return 'Hawkish'
    # 2对应Negative -> 经济差 -> 鸽派
    elif label_clean == '2': 
        return 'Dovish'
    
    # --- 3. 匹配关键词 ---
    if 'HAWK' in label_clean: return 'Hawkish'
    if 'DOVE' in label_clean: return 'Dovish'
    
    return 'Neutral'


def get_sentiment_label_RoBERTa(raw_label):
    """
    Map RoBERTa-large labels to Hawkish/Dovish/Neutral.
    Standard RoBERTa Sentiment Output:
    - LABEL_0 = Negative
    - LABEL_1 = Neutral
    - LABEL_2 = Positive
    
    Fed Policy Logic (Inverted Sentiment):
    - Negative (Bad Economy) -> Dovish (Cut Rates)
    - Positive (Good Economy) -> Hawkish (Hike Rates)
    """
    label_clean = str(raw_label).upper().replace("LABEL_", "")
    
    # --- 1. 优先匹配文本标签 ---
    if 'NEGATIVE' in label_clean:
        return 'Dovish'   # <--- 改正：坏消息 = 鸽派 (救市)
    if 'POSITIVE' in label_clean:
        return 'Hawkish'  # <--- 改正：好消息 = 鹰派 (加息)
    if 'NEUTRAL' in label_clean:
        return 'Neutral'
    
    # --- 2. 匹配数字标签 (Standard RoBERTa) ---
    if label_clean == '0':    # 0 is Negative
        return 'Dovish'       # <--- 改正：Negative -> Dovish
    elif label_clean == '1':  # 1 is Neutral
        return 'Neutral'
    elif label_clean == '2':  # 2 is Positive
        return 'Hawkish'      # <--- 改正：Positive -> Hawkish
    
    # --- 3. 默认返回 ---
    return 'Neutral'
