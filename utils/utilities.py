def calculate_net_sentiment_counts(group):
    """
    Calculate Net Sentiment Index using Hard Counts.
    Formula: (Dovish - Hawkish) / Total Counts
    Result: Positive = Dovish, Negative = Hawkish
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

    # 修正：(鸽派 - 鹰派) / 总数
    # 这样鹰派多的时候，结果才是负数
    return (d - h) / total


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

    # 修正：(鸽派分数 - 鹰派分数)
    return (d_score - h_score) / total


def get_sentiment_label_FinBERT_FOMC(raw_label):
    """
    Map FinBERT-FOMC labels to Hawkish/Dovish/Neutral.
    Compatible with both LABEL_X format and text output (Negative/Positive).
    """
    label_clean = str(raw_label).upper().replace("LABEL_", "")
    
    # --- 1. 优先匹配文本 (防止 pipeline 输出 Negative/Positive) ---
    if 'NEGATIVE' in label_clean: 
        return 'Hawkish'
    if 'POSITIVE' in label_clean: 
        return 'Dovish'
    if 'NEUTRAL' in label_clean:
        return 'Neutral'

    # --- 2. 匹配数字标签 (ZiweiChen 标准: 0=Neutral, 1=Hawkish, 2=Dovish) ---
    if label_clean == '0': return 'Neutral'
    elif label_clean == '1': return 'Hawkish'
    elif label_clean == '2': return 'Dovish'
    
    # --- 3. 匹配关键词 ---
    if 'HAWK' in label_clean: return 'Hawkish'
    if 'DOVE' in label_clean: return 'Dovish'
    
    return 'Neutral'


def get_sentiment_label_RoBERTa(raw_label):
    """
    Map RoBERTa-large labels to Hawkish/Dovish/Neutral.
    RoBERTa model outputs: LABEL_0 (Negative), LABEL_1 (Neutral), LABEL_2 (Positive)
    Fed context mapping: Negative=Hawkish, Positive=Dovish, Neutral=Neutral
    """
    label_clean = str(raw_label).upper().replace("LABEL_", "")
    
    # --- 1. 优先匹配文本标签 (如果pipeline输出text) ---
    if 'NEGATIVE' in label_clean:
        return 'Hawkish'  # 在美联储语境下，负面=鹰派
    if 'POSITIVE' in label_clean:
        return 'Dovish'   # 在美联储语境下，正面=鸽派
    if 'NEUTRAL' in label_clean:
        return 'Neutral'
    
    # --- 2. 匹配数字标签 (j-hartmann RoBERTa标准) ---
    # LABEL_0 = Negative → Hawkish
    # LABEL_1 = Neutral → Neutral  
    # LABEL_2 = Positive → Dovish
    if label_clean == '0':
        return 'Hawkish'
    elif label_clean == '1':
        return 'Neutral'
    elif label_clean == '2':
        return 'Dovish'
    
    # --- 3. 默认返回 ---
    return 'Neutral'