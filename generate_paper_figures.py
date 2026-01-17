#!/usr/bin/env python3
"""
Generate high-resolution figures for ACL paper: FinBERT-FOMC vs Economic Indicators.
Optimized for ACL \textwidth (approx 6.3 inches) using Golden Ratio aspect.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ==========================================
# 1. ACL Paper Layout Configuration
# ==========================================
# ACL standard text width is approx 6.3 inches (16cm)
ACL_WIDTH = 6.3  

# Golden Ratio Calculation (approx 1.618)
# Height = Width / 1.618
ACL_HEIGHT = ACL_WIDTH / 1.618  # Approx 3.89 inches

# Set high-quality plotting style for ACL paper
plt.rcParams.update({
    'font.family': 'serif', # Matches LaTeX font
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,   # Slightly smaller legend font
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'text.usetex': False 
})

sns.set_style("whitegrid")

# ==========================================
# 2. Data Loading
# ==========================================
def load_data():
    """Load FinBERT-FOMC sentiment data and economic indicators"""
    
    # Define paths
    sentiment_path = "data/result/FinBERT-FOMC/monthly_index_FinBERT_FOMC.csv"
    econ_path = "data/raw/econ_indicators.csv"
    
    # Check if files exist
    if not os.path.exists(sentiment_path) or not os.path.exists(econ_path):
        print(f"Error: Data files not found.\nCheck: {sentiment_path}\nCheck: {econ_path}")
        return None, None

    # Load FinBERT sentiment data
    df_sentiment = pd.read_csv(sentiment_path)
    df_sentiment['month'] = pd.to_datetime(df_sentiment['month'])
    df_sentiment.set_index('month', inplace=True)

    # Load economic indicators
    df_econ = pd.read_csv(econ_path)
    df_econ['DATE'] = pd.to_datetime(df_econ['DATE'])
    df_econ.set_index('DATE', inplace=True)

    # Filter to matching timeframe
    min_date = df_sentiment.index.min()
    max_date = df_sentiment.index.max()
    df_econ = df_econ[(df_econ.index >= min_date) & (df_econ.index <= max_date)]

    return df_sentiment, df_econ

# ==========================================
# 3. Main Plotting Function
# ==========================================
def create_individual_economic_plots(df_sentiment, df_econ, use_ma=False):
    """Create individual plots with legend at 1/3 position"""
    model_name = "sentiment_index"

    # Handle Moving Average logic
    if use_ma:
        df_sentiment_plot = df_sentiment.rolling(window=3, center=False).mean()
        title_suffix = "3M Moving Average"
        filename_suffix = "3m_ma"
    else:
        df_sentiment_plot = df_sentiment
        title_suffix = "Raw"
        filename_suffix = "raw"

    # Indicators configuration: (Column Name, Label, Color)
    indicators = [
        ("CPI_YoY", "CPI YoY (%)", "tab:red"),
        ("PPI_YoY", "PPI YoY (%)", "tab:orange"),
        ("Fed_Funds_Rate", "Fed Funds Rate (%)", "tab:green")
    ]
    
    # Ensure output directory exists
    os.makedirs("reports/figures", exist_ok=True)

    for indicator, ylabel, color in indicators:
        # Create figure with ACL dimensions (Golden Ratio)
        fig, ax1 = plt.subplots(1, 1, figsize=(ACL_WIDTH, ACL_HEIGHT), dpi=300)

        # --- Plot 1: Economic Indicator (Left Axis) ---
        line1 = ax1.plot(df_econ.index, df_econ[indicator], color=color, 
                         linewidth=1.5, marker="o", markersize=3, alpha=0.9, 
                         label=ylabel) # Label for legend
        
        ax1.set_ylabel(ylabel, color=color, fontsize=10, fontweight='bold')
        ax1.tick_params(axis="y", labelcolor=color, labelsize=9)
        ax1.tick_params(axis="x", labelsize=9)
        
        # --- Plot 2: Sentiment (Right Axis) ---
        ax2 = ax1.twinx()
        line2 = ax2.plot(df_sentiment_plot.index, df_sentiment_plot[model_name],
                         color="tab:blue", linewidth=1.5, marker="s", markersize=3, alpha=0.9, 
                         label="FinBERT Sentiment") # Label for legend
        
        ax2.set_ylabel("Sentiment Score", color="tab:blue", fontsize=10, fontweight='bold')
        ax2.tick_params(axis="y", labelcolor="tab:blue", labelsize=9)
        
        # Zero line for sentiment
        ax2.axhline(0, color="gray", linestyle=":", alpha=0.6, linewidth=1)
        
        # --- Unified Legend at 1/3 Position (Vertical / 2 Rows) ---
        # Combine handles and labels from both axes
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        
        # Position: x=0.33 (1/3 width), y=0.98 (top), aligned by 'upper center'
        # ncol=1 forces 1 column (vertical stack = 2 rows)
        ax1.legend(lines, labels, 
                   loc='upper center', 
                   bbox_to_anchor=(0.33, 0.98), 
                   ncol=1,              # 1 column = 2 rows (Vertical)
                   frameon=True,        # Show box
                   framealpha=0.90,     # Semi-opaque background
                   edgecolor='gray')

        # Title and Grid
        indicator_name = indicator.replace('_', ' ')
        plt.title(f"FinBERT-FOMC {title_suffix} vs {indicator_name}", 
                  fontsize=11, fontweight='bold', pad=10)
        
        ax1.grid(True, alpha=0.2, linestyle='--')
        ax2.grid(False) # Disable grid for second axis to avoid clutter

        # Layout optimization
        plt.tight_layout()

        # Save
        filename = f"finbert_{indicator.lower()}_{filename_suffix}.png"
        output_path = f"reports/figures/{filename}"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f"Saved: {output_path}")

        plt.close()

# ==========================================
# 4. Correlation Heatmap
# ==========================================
def create_correlation_plot(df_sentiment, df_market):
    """Create a compact correlation heatmap suitable for single-column"""
    
    # Prepare data
    df_sentiment_daily = df_sentiment.resample('D').ffill()
    df_analysis_market = df_sentiment_daily.join(df_market, how='inner')

    # Compact size for single column (approx 3.5 inches)
    # Heatmap doesn't necessarily need Golden Ratio as it's square-ish usually
    plt.figure(figsize=(3.5, 3.2), dpi=300) 
    
    corr_matrix = df_analysis_market.corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                fmt=".2f", square=True, linewidths=0.5, 
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
                
    plt.title("Correlation Matrix", fontsize=10, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    output_path = "reports/figures/finbert_market_correlation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved: {output_path}")
    plt.close()

# ==========================================
# 5. Execution
# ==========================================
if __name__ == "__main__":
    print("Generating ACL paper-quality figures (Golden Ratio)...")
    
    # Load
    df_sentiment, df_econ = load_data()
    
    if df_sentiment is not None and df_econ is not None:
        print(f"Data Loaded: {len(df_sentiment)} sentiment records, {len(df_econ)} economic records.")

        # 1. Raw Plots
        print("\nGenerating Raw sentiment plots...")
        create_individual_economic_plots(df_sentiment, df_econ, use_ma=False)

        # 2. Moving Average Plots
        print("\nGenerating 3M Moving Average plots...")
        create_individual_economic_plots(df_sentiment, df_econ, use_ma=True)

        # 3. Correlation
        print("\nGenerating Correlation heatmap...")
        create_correlation_plot(df_sentiment, df_econ)

        print("\nDone! Check 'reports/figures/' for output.")
    else:
        print("Skipping generation due to missing data.")