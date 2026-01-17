#!/usr/bin/env python3
"""
Generate ACL paper-quality model comparison figures.
Creates publication-ready visualizations comparing FinBERT-FOMC, RoBERTa-Base, and RoBERTa-Large models.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import numpy as np

# ==========================================
# 1. ACL Paper Layout Configuration
# ==========================================
# ACL standard text width is approx 6.3 inches (16cm) for double column
ACL_WIDTH = 6.3
ACL_HEIGHT = ACL_WIDTH / 1.618  # Golden Ratio

# Set high-quality plotting style for ACL paper
# Using Times New Roman to match LaTeX/PDF standard of ACL proceedings
plt.rcParams.update({
    'font.family': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.linewidth': 0.8,    # Thinner, more precise axes lines
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,     # Readable legend
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'text.usetex': False      # kept False to avoid system dependencies, but styled to look like TeX
})

# customized seaborn style to remove clutter
sns.set_style("ticks", {'axes.grid': True, 'grid.linestyle': ':'})

def load_index(path, model_name):
    """Load sentiment index data for a specific model"""
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Ensure date format
        col_date = 'month' if 'month' in df.columns else 'date'
        df[col_date] = pd.to_datetime(df[col_date])

        # Rename score column to model name
        col_score = 'sentiment_score' if 'sentiment_score' in df.columns else 'sentiment_index'
        df = df[[col_date, col_score]].rename(columns={col_score: model_name, col_date: 'date'})

        # Set index for easy joining
        df.set_index('date', inplace=True)
        return df
    else:
        print(f"Warning: File not found for {model_name} at {path}")
        return pd.DataFrame()

def create_model_comparison_plot(df_sentiment, df_econ):
    """Create individual plots for each model vs economic indicators (ACL format)"""

    models = df_sentiment.columns
    # Professional, high-contrast colors suitable for academic papers
    model_colors = {
        'FinBERT_FOMC': ('#005A9C', '#8FBCE6'),   # Academic Blue
        'RoBERTa_Base': ('#D35400', '#F5B7B1'),   # Burnt Orange
        'RoBERTa_Large': ('#27AE60', '#A9DFBF')   # Emerald Green
    }

    # Economic indicators with distinct, darker colors for visibility
    econ_indicators = [
        ('CPI_YoY', 'CPI YoY (%)', '#C0392B'),        # Dark Red
        ('PPI_YoY', 'PPI YoY (%)', '#E74C3C'),        # Lighter Red
        ('Fed_Funds_Rate', 'Fed Funds Rate (%)', '#34495E') # Dark Slate
    ]

    for model_name in models:
        if model_name not in df_sentiment.columns:
            continue

        # Create figure with ACL dimensions
        fig, ax1 = plt.subplots(1, 1, figsize=(ACL_WIDTH, ACL_HEIGHT), dpi=300)

        # Calculate 3-Month Moving Average
        raw_sentiment = df_sentiment[model_name]
        ma_3m = raw_sentiment.rolling(window=3, center=True).mean()

        primary_color, secondary_color = model_colors.get(model_name, ('#1f77b4', '#aec7e8'))

        # --- Plot Sentiment (Left Axis) ---
        # Raw sentiment: Thin line, faint alpha
        ax1.plot(df_sentiment.index, raw_sentiment,
                 color=secondary_color, linewidth=1.0, alpha=0.6,
                 label=f'{model_name} (Raw)')

        # 3M MA: Thicker line, solid color
        ax1.plot(df_sentiment.index, ma_3m,
                 color=primary_color, linewidth=2.0, alpha=1.0,
                 label=f'{model_name} (3M MA)')

        ax1.set_ylabel("Sentiment Score", fontsize=10, fontweight='bold')
        ax1.tick_params(axis='y', labelsize=9)
        
        # Professional Date Formatting
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        plt.setp(ax1.get_xticklabels(), rotation=0, ha='center')

        # Zero line for sentiment
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

        # --- Plot Economic Indicators (Right Axis) ---
        ax2 = ax1.twinx()

        legend_elements = []
        for econ_col, econ_label, econ_color in econ_indicators:
            if econ_col in df_econ.columns:
                line = ax2.plot(df_econ.index, df_econ[econ_col],
                              color=econ_color, linewidth=1.2, alpha=0.8,
                              linestyle='--', dashes=(3, 2), # Distinct dash pattern
                              label=econ_label)
                legend_elements.extend(line)

        ax2.set_ylabel("Economic Indicators (%)", fontsize=10, fontweight='bold', rotation=270, labelpad=15)
        ax2.tick_params(axis='y', labelsize=9)
        
        # Ensure Top Line is visible (ACL Standard Frame)
        ax1.spines['top'].set_visible(True)
        ax2.spines['top'].set_visible(True)

        # Compact Title
        model_display_name = model_name.replace('_', '-')
        plt.title(f"{model_display_name} vs. Economic Indicators",
                 fontsize=11, fontweight='bold', pad=10, fontname='Times New Roman')

        # Grid settings
        ax1.grid(True, which='major', axis='y', alpha=0.3, linestyle=':')
        ax2.grid(False) 

        # Combined Legend with semi-transparent background
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc='upper center',
                   bbox_to_anchor=(0.5, -0.15), # Below plot for ACL papers is often cleaner
                   ncol=3,
                   frameon=False, # Minimalist legend
                   fontsize=9,
                   borderaxespad=0.5)

        plt.tight_layout()

        # Save high-resolution version
        safe_model_name = model_name.lower().replace('-', '_')
        output_path = f"reports/figures/model_comparison_{safe_model_name}.png"
        os.makedirs("reports/figures", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f"Saved: {output_path}")

        plt.close()

def create_correlation_heatmap(df_sentiment, df_econ):
    """Create correlation heatmap suitable for ACL paper"""

    # Merge data for correlation analysis
    df_analysis = df_sentiment.join(df_econ[['CPI_YoY', 'PPI_YoY', 'Fed_Funds_Rate']], how='inner')

    # Compact size for ACL paper - single column width usually fits heatmaps better
    plt.figure(figsize=(ACL_WIDTH * 0.8, ACL_WIDTH * 0.7), dpi=300)

    # Create correlation matrix
    corr_matrix = df_analysis.corr()

    # Create heatmap using a diverging colormap appropriate for academic papers (Red-Blue)
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, fmt=".2f", square=True,
                linewidths=0.5, linecolor='white',
                annot_kws={"size": 9, "family": "serif"},
                cbar_kws={"shrink": 0.7})

    plt.title("Correlation Matrix",
             fontsize=11, fontweight='bold', pad=10, fontname='Times New Roman')
    
    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=9, fontname='Times New Roman')
    plt.yticks(rotation=0, fontsize=9, fontname='Times New Roman')

    plt.tight_layout()

    # Save
    output_path = "reports/figures/model_comparison_correlation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved: {output_path}")
    plt.close()

def create_summary_statistics_plot(df_sentiment):
    """Create summary statistics comparison plot"""

    models = df_sentiment.columns
    stats_data = []

    for model in models:
        if model in df_sentiment.columns:
            series = df_sentiment[model].dropna()
            stats_data.append({
                'Model': model.replace('_', '-'),
                'Mean': series.mean(),
                'Std': series.std(),
                'Min': series.min(),
                'Max': series.max(),
                'Volatility': series.std() / abs(series.mean()) if series.mean() != 0 else 0
            })

    df_stats = pd.DataFrame(stats_data)
    df_stats.set_index('Model', inplace=True)

    # Create figure - more compact
    fig, axes = plt.subplots(2, 2, figsize=(ACL_WIDTH, ACL_WIDTH * 0.8), dpi=300)
    fig.suptitle("Model Performance Statistics", fontsize=11, fontweight='bold', y=0.98, fontname='Times New Roman')

    # Subplots
    metrics = ['Mean', 'Std', 'Volatility', 'Max']
    titles = ['Mean Sentiment', 'Standard Deviation', 'Volatility Ratio', 'Maximum Value']
    colors = ['#4A7ebb', '#D35400', '#27AE60'] # Consistent with line plots

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i//2, i%2]
        
        # Plot bars with zorder to put grid behind
        bars = ax.bar(df_stats.index, df_stats[metric],
                      color=colors[:len(df_stats)],
                      alpha=0.85, width=0.6, zorder=3)
        
        ax.set_title(title, fontsize=10, fontweight='bold', pad=5, fontname='Times New Roman')
        
        # Clean x-axis
        ax.tick_params(axis='x', labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontname='Times New Roman')
        
        ax.tick_params(axis='y', labelsize=8)
        
        # Minimalist grid behind bars
        ax.grid(True, axis='y', color='#D3D3D3', linestyle=':', zorder=0)
        
        # Enable top spine (black line) as requested
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(False)

        # Add value labels on bars (Corrected logic)
        for bar in bars:
            height = bar.get_height()
            # Determine vertical offset based on sign
            offset = abs(height) * 0.05 if height != 0 else 0.01
            va = 'bottom' if height >= 0 else 'top'
            y_pos = height + offset if height >= 0 else height - offset
            
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{height:.2f}', # Corrected formatting
                   ha='center', va=va, fontsize=8, color='black', fontweight='bold')

    plt.tight_layout(pad=1.0, h_pad=1.5, w_pad=1.0)

    # Save
    output_path = "reports/figures/model_comparison_statistics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved: {output_path}")
    plt.close()

def main():
    print("=== Generating ACL Paper-Quality Model Comparison Figures ===")

    # Define paths
    BASE_DIR = "data/result"
    PATH_FINBERT_FOMC = os.path.join(BASE_DIR, "FinBERT-FOMC", "monthly_index_FinBERT_FOMC.csv")
    PATH_ROBERTA_BASE = os.path.join(BASE_DIR, "RoBERTa_Base", "monthly_index_Twitter_RoBERTa.csv")
    PATH_ROBERTA_LARGE = os.path.join(BASE_DIR, "RoBERTa_Large", "monthly_index_RoBERTa_large.csv")
    PATH_ECON = "data/raw/econ_indicators.csv"

    # Load sentiment data
    print("\nLoading sentiment data...")
    df_finbert = load_index(PATH_FINBERT_FOMC, "FinBERT_FOMC")
    df_roberta_b = load_index(PATH_ROBERTA_BASE, "RoBERTa_Base")
    df_roberta_l = load_index(PATH_ROBERTA_LARGE, "RoBERTa_Large")

    # Merge sentiment data
    dfs = [df_finbert, df_roberta_b, df_roberta_l]
    dfs = [d for d in dfs if not d.empty]

    if not dfs:
        print("No sentiment data loaded!")
        return

    df_merged = pd.concat(dfs, axis=1).sort_index()
    df_merged.dropna(how='all', inplace=True)

    print(f"Loaded {len(df_merged)} months of sentiment data")
    print(f"Models: {list(df_merged.columns)}")

    # Load economic data
    print("\nLoading economic data...")
    if os.path.exists(PATH_ECON):
        df_econ = pd.read_csv(PATH_ECON)
        df_econ['DATE'] = pd.to_datetime(df_econ['DATE'])
        df_econ.rename(columns={'DATE': 'date'}, inplace=True)
        df_econ.set_index('date', inplace=True)

        # Filter to matching timeframe
        min_date = df_merged.index.min()
        max_date = df_merged.index.max()
        df_econ = df_econ[(df_econ.index >= min_date) & (df_econ.index <= max_date)]

        print(f"Loaded {len(df_econ)} months of economic data")
    else:
        print("Economic data not found!")
        df_econ = pd.DataFrame()

    # Generate plots
    if not df_econ.empty:
        print("\nGenerating individual model comparison plots...")
        create_model_comparison_plot(df_merged, df_econ)

        print("\nGenerating correlation heatmap...")
        create_correlation_heatmap(df_merged, df_econ)

        print("\nGenerating summary statistics...")
        create_summary_statistics_plot(df_merged)

        print("\n=== All figures generated successfully! ===")
        print("Check 'reports/figures/' for ACL paper-quality PNG files:")
        print("- model_comparison_finbert_fomc.png")
        print("- model_comparison_roberta_base.png")
        print("- model_comparison_roberta_large.png")
        print("- model_comparison_correlation.png")
        print("- model_comparison_statistics.png")
        print("\nFigures are optimized for ACL paper submission (300 DPI, proper sizing)")
    else:
        print("Cannot generate plots without economic data!")

if __name__ == "__main__":
    main()