import pandas as pd
import re
import nltk
import os

# 首次运行需要下载 nltk 的分词数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_data(file_path):
    """加载原始 CSV 数据"""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    # 转换日期格式
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # 确保 text 列是字符串
    df['text'] = df['text'].astype(str)
    
    print(f"Successfully loaded {len(df)} documents.")
    return df

def extract_sections(text):
    """
    核心逻辑：解析 FOMC Minutes 的特定章节
    返回一个列表：[{'section_name': 'Staff Review', 'text': '...'}, ...]
    """
    
    # 1. 定义我们关心的“高价值”章节标题 (标准化命名)
    # 键是标准化的列名，值是可能出现在文本中的标题关键词列表
    # 注意：按大概的出现顺序排列，但代码会根据实际 index 排序
    section_patterns = {
        "Developments in Financial Markets": [
            "Developments in Financial Markets and Open Market Operations"
        ],
        "Inflation Analysis": [
            "Inflation Analysis and Forecasting"
        ],
        "Staff Review of Economic Situation": [
            "Staff Review of the Economic Situation", 
            "The information reviewed for the" # 有时这部分没有标题，直接以这句话开头
        ],
        "Staff Review of Financial Situation": [
            "Staff Review of the Financial Situation"
        ],
        "Staff Economic Outlook": [
            "Staff Economic Outlook"
        ],
        "Participants' Views": [
            "Participants' Views on Current Conditions and the Economic Outlook", # 完整标题
            "Participants' Views on Current Conditions",
            "Participants’ Views on Current Conditions", # 智能引号
            "Discussion of Monetary Policy" # 旧版纪要常用
        ],
        "Committee Policy Action": [
            "Committee Policy Action"
        ]
    }
    
    # 2. 找到所有标题在文中的位置
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
                # 找到一个关键词后，该章节就定位了，跳过该章节的其他别名
                break 
    
    # 3. 按在文中出现的顺序排序
    if not matches:
        return []
    
    matches.sort(key=lambda x: x['priority'])
    
    # 4. 切分文本
    extracted_data = []
    
    for i in range(len(matches)):
        current_match = matches[i]
        start = current_match['start_index']
        # Skip the header itself
        content_start = start + current_match['header_length']
        
        # 结束位置是下一个章节的开始，或者是文末
        if i < len(matches) - 1:
            end = matches[i+1]['start_index']
        else:
            end = len(text)
            
        # 提取该段文本 (去除标题)
        section_text = text[content_start:end].strip()
        
        extracted_data.append({
            "section_name": current_match['section_name'],
            "section_text": section_text
        })
        
    return extracted_data

def segment_sentences(df):
    """
    将文档级的 DataFrame 转换为句子级的 DataFrame，并增加 'section' 列
    """
    processed_rows = []
    
    print("Segmenting text into sentences with section labels...")
    
    for idx, row in df.iterrows():
        doc_text = row['text']
        date = row['date']
        doc_id = idx
        
        # 1. 提取章节 (这会自动过滤掉不在列表中的 administrative 内容)
        sections = extract_sections(doc_text)
        
        # 如果没有提取到任何章节 (可能是格式太旧或太新)，为了安全起见，暂时跳过或记录
        if not sections:
            # 可以在这里添加 fallback 逻辑，比如保留全文
            continue
            
        for section in sections:
            section_name = section['section_name']
            section_content = section['section_text']
            
            # 清理一下文本中的多余空白
            section_content = re.sub(r'\s+', ' ', section_content).strip()
            
            # 2. 分句
            sentences = nltk.sent_tokenize(section_content)
            
            for sent in sentences:
                sent = sent.strip()
                
                # 3. 句子级过滤
                if len(sent.split()) < 5: 
                    continue
                
                # 过滤行政废话 (即使在章节内也可能出现)
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
                    'section': section_name, # 新增的列！
                    'sentence_text': sent,
                    'source_type': 'Minutes'
                })
            
    return pd.DataFrame(processed_rows)

if __name__ == "__main__":
    # 配置路径
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