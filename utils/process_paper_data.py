import pandas as pd
import os
import glob

# Paths
raw_data_dir = r"e:\Textming\data\from_paper"
output_path = r"e:\Textming\data\gold_standard\paper_test_set.csv"

# Load files and track source
files = sorted(glob.glob(os.path.join(raw_data_dir, "*.xlsx")))
dfs = []

# Map filenames to source labels
source_map = {
    'lab-manual-mm-split-test-944601.xlsx': 'Minutes',
    'lab-manual-pc-split-test-5768.xlsx': 'Press',
    'lab-manual-pc-test-78516.xlsx': 'Press',
    'lab-manual-pc-test-944601.xlsx': 'Press',
    'lab-manual-sp-split-test-5768.xlsx': 'Speech'
}

for f in files:
    try:
        temp_df = pd.read_excel(f)
        filename = os.path.basename(f)
        
        # Assign source based on filename
        temp_df['source'] = source_map.get(filename, filename)
        dfs.append(temp_df)
        print(f"Loaded {filename}: {len(temp_df)} samples → {temp_df['source'].iloc[0]}")
    except Exception as e:
        print(f"Error reading {f}: {e}")

if not dfs:
    raise ValueError("No files found or loaded.")

# Combine
combined_df = pd.concat(dfs, ignore_index=True)

# DO NOT DROP DUPLICATES - User requested maximum samples
# combined_df.drop_duplicates(subset=['sentence'], inplace=True)

# Correct Label Map based on user rule
# 0: Dovish -> positive
# 1: Hawkish -> negative
# 2: Neutral -> neutral
label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}

combined_df['label'] = combined_df['label'].map(label_map)

# Rename columns
combined_df.rename(columns={'sentence': 'text'}, inplace=True)

# Add date column
combined_df['date'] = combined_df['year'].apply(lambda x: f"{x}-01-01")

# Select columns
final_df = combined_df[['date', 'source', 'text', 'label']]

# Save
final_df.to_csv(output_path, index=False)
print(f"\n{'='*60}")
print(f"Saved processed test set to {output_path}")
print(f"Total Samples: {len(final_df)}")
print(f"\nSource Distribution:")
print(final_df['source'].value_counts())
print(f"\nLabel Counts:")
print(final_df['label'].value_counts())
print('='*60)
