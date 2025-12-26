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

# --- 1. 定义关键词列表 ---
FED_KEYWORDS = [
    'inflation expectation', 'interest rate', 'bank rate', 'fund rate',
    'price', 'economic activity', 'inflation', 'employment',
    'unemployment', 'growth', 'exchange rate', 'productivity', 'deficit', 'demand', 'job',
    'market', 'monetary policy'
]

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) 

INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "fed_speeches.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "fed_speeches_sentences.csv")

def clean_text(text):
    """
    Basic text cleaning before segmentation.
    """
    if not isinstance(text, str):
        return ""
    
    if "Return to text" in text:
        return ""
    
    if re.match(r'^\d+\.\s+[A-Z].*\(\d{4}\)', text):
        return ""

    return text.strip()

def extract_date_from_url(url):
    """
    Extracts date from URL.
    """
    if not isinstance(url, str):
        return None
    
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

    text_col = 'text' if 'text' in df.columns else 'text_segment'
    if text_col not in df.columns:
        print(f"Error: Neither 'text' nor 'text_segment' column found in {INPUT_FILE}")
        return

    # 统计跳过了多少行（可选，用于调试）
    skipped_count = 0

    for idx, row in df.iterrows():
        # --- 2. 新增：基于 Title 的关键词过滤逻辑 ---
        title = str(row.get('title', '')) # 获取标题并转为字符串
        title_lower = title.lower()       # 转为小写以便不区分大小写匹配
        
        # 检查 title 是否包含任何一个关键词
        # 如果列表中没有任何一个词出现在 title 中，则跳过该行
        if not any(keyword in title_lower for keyword in FED_KEYWORDS):
            skipped_count += 1
            continue 
        # ------------------------------------------

        text = row.get(text_col, '') 
        if not isinstance(text, str):
            continue

        url = row.get('url', '')
        extracted_date = extract_date_from_url(url)
        
        current_date = row.get('date')
        final_date = extracted_date if extracted_date else current_date

        cleaned_text = clean_text(text)
        if not cleaned_text:
            continue

        sentences = nltk.sent_tokenize(cleaned_text)

        for sent in sentences:
            sent = sent.strip()
            
            if len(sent.split()) < 5:
                continue
            
            if "Return to text" in sent:
                continue

            processed_rows.append({
                'date': final_date,
                'title': title, # 使用原始标题
                'text': sent,
                'source_type': 'Speech',
                'url': url 
            })

    df_out = pd.DataFrame(processed_rows)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"Processing complete.")
    print(f"Skipped {skipped_count} documents due to irrelevant titles.")
    print(f"Saved {len(df_out)} sentences to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_speeches()