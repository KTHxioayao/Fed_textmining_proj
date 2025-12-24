import pandas as pd
import os

DATA_DIR = '../data'
FILES = {
    'minutes': 'fed_minutes_sentences_structured.csv',
    'press': 'fed_press_conf_structured.csv',
    'speech': 'fed_speeches_sentences.csv' 
}

def load_and_standardize(file_name, source_type):
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[Error] Failed to read {file_name}: {e}")
        return pd.DataFrame()
    
    # Standardize columns
    if 'sentence_text' in df.columns:
        df = df.rename(columns={'sentence_text': 'text'})
    elif 'section_text' in df.columns:
        df = df.rename(columns={'section_text': 'text'})
    
    if 'section' not in df.columns:
        df['section'] = 'General'
        
    df['source'] = source_type
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    if 'speaker' not in df.columns:
        df['speaker'] = 'N/A'
        
    cols = ['date', 'text', 'section', 'source', 'speaker']
    return df[cols]

# Main execution
print("--- Exporting Data ---")
df_minutes = load_and_standardize(FILES['minutes'], 'Minutes')
df_press = load_and_standardize(FILES['press'], 'Press Conf')
df_speech = load_and_standardize(FILES['speech'], 'Speech')

frames = [df for df in [df_minutes, df_press, df_speech] if not df.empty]

if frames:
    df_master = pd.concat(frames, ignore_index=True)
    df_master = df_master.sort_values('date').reset_index(drop=True)
    
    # Cleaning
    df_master = df_master.dropna(subset=['text'])
    df_master['word_count'] = df_master['text'].apply(lambda x: len(str(x).split()))
    df_clean = df_master[df_master['word_count'] >= 5].copy()
    
    output_path = os.path.join(DATA_DIR, 'fed_master_corpus.csv')
    df_clean.to_csv(output_path, index=False)
    print(f"SUCCESS: Saved {len(df_clean)} sentences to {output_path}")
else:
    print("ERROR: No data found.")
