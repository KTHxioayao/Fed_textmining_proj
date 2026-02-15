# Analysis of Federal Open Market Committee Communication: Topic Evolution and Sentiment Analysis (2018-2024)
> **Quantifying the Impact of Central Bank Sentiment on Monetary Policy Signals**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NLP](https://img.shields.io/badge/NLP-FinBERT%20%7C%20RoBERTa-green.svg)](https://huggingface.co/ProsusAI/finbert)

## 📌 Overview
This repository provides a comprehensive NLP framework to analyze Federal Open Market Committee (FOMC) communications from 2018 to 2024. The project employs **BERTopic** for topic modeling to identify three distinct phases of discourse (Monetary Policy, Inflation Shock, and Disinflation) and evaluates five sentiment analysis models to decode policy stance from FOMC documents (Minutes, Press Conferences, Speeches).

Our key findings include: (1) **FinBERT-FOMC** outperforms other models (F1-score: 0.429) in capturing domain-specific monetary policy language; (2) identification of the **"Inflation Paradox"**—a regime-dependent inversion where negative sentiment during high inflation periods signals hawkish policy rather than dovish stance; and (3) temporal evolution of topics that reflects real-time economic realities.

## 🚀 Key Features
- **Comprehensive Data Pipeline**: Web-scraping and processing of Federal Reserve Board documents (Speeches, Minutes, Press Conferences) from January 2018 to December 2024, resulting in over 8,202 sentence-level segments.
- **Advanced Topic Modeling**: Utilizing **BERTopic** with UMAP dimensionality reduction and HDBSCAN clustering to identify and track the evolution of five dominant topics over time, revealing three distinct phases: Monetary Policy, Inflation Shock, and Disinflation.
- **Multi-Model Sentiment Analysis**: Comprehensive benchmarking of five models on a labeled dataset (~600 sentences):
    - **FinBERT-FOMC** (Best performer: F1-score 0.429, 53.0% non-neutral detection)
    - **FinBERT** (ProsusAI, finance-general)
    - **RoBERTa-Large** (General domain, highest accuracy but biased toward neutral)
    - **Cardiff-RoBERTa** (Social media optimized)
    - **Rule-based Baseline** (Dictionary-based)
- **Economic Validation**: Correlation analysis of sentiment indices with macroeconomic indicators (CPI YoY, PPI YoY, Federal Funds Rate) from FRED database.
- **Academic Output**: Full LaTeX pipeline for generating ACL-format research papers with comprehensive methodology and results.

## 🛠 Project Workflow (Notebooks)
The analysis is structured into a sequential pipeline located in the `notebooks/` directory:

| Step | Notebook | Description |
|------|----------|-------------|
| 01 | `1_Data_Collection_and_Processing.ipynb` | Scrapes raw text from Federal Reserve sources, performs geometric PDF cropping, speaker diarization, and keyword-based filtering. |
| 02 | `2_Data_Integration.ipynb` | Merges text data with timestamps and sources into a master corpus with metadata (date, section, document type). |
| 03 | `3_Exploratory_Analysis.ipynb` | Basic statistics, word distributions, and linguistic feature analysis across data sources (Minutes, Press Conferences, Speeches). |
| 04a| `4_Topic_Modeling_BERTopic.ipynb` | Extracts latent topics using BERTopic (all-MiniLM-L6-v2 embeddings, UMAP, HDBSCAN) to visualize thematic shifts and identify three distinct phases. |
| 04b| `4_Model_Evaluation.ipynb` | Evaluates five sentiment models against a Gold Standard labeled dataset (~600 sentences, balanced classes) with stratified evaluation. |
| 05 | `5_Sentiment_Pipeline_*.ipynb` | Fine-tuned pipelines for **FinBERT**, **FinBERT-FOMC**, **RoBERTa-Base**, and **RoBERTa-Large** variants, generating sentiment indices. |
| 06 | `6_Model_Comparison_Visualization.ipynb` | Final comparative visualization of model signals vs. Economic indicators (CPI, PPI, Fed Funds Rate) and identification of the Inflation Paradox. |

## 📂 Repository Structure
- **`data/`**: 
  - `raw/`: Original PDF and HTML files from Federal Reserve sources
  - `processed/`: Cleaned and filtered sentence-level corpus
  - `master/`: Integrated master corpus (`fed_master_corpus.csv`) with metadata
  - `gold_standard/`: Labeled evaluation dataset (`paper_test_set.csv`)
  - `result/`: Model output files organized by model type (FinBERT, FinBERT-FOMC, RoBERTa variants)
  - `from_paper/`: Additional reference data from related research
- **`notebooks/`**: Interactive Jupyter notebooks covering the full analysis pipeline (Data Collection, Integration, EDA, Topic Modeling, Model Evaluation, Sentiment Analysis, Visualization).
- **`scraping/`**: Python scripts for data collection:
  - `scrape_*.py`: Web scrapers for Minutes, Press Conferences, and Speeches
  - `process_*.py`: Processing scripts with PDF cropping and speaker diarization
  - `fetch_econ_data.py`: Economic indicator retrieval from FRED
  - `fetch_market_data.py`: Market data collection
- **`utils/`**: Shared helper modules:
  - `utilities.py`: Text processing, sentiment scoring, and plotting functions
  - `generate_paper_figures.py`: Figure generation for LaTeX paper
  - `generate_model_comparison_figures.py`: Model comparison visualizations
  - `process_paper_data.py`: Data processing utilities
- **`scripts/`**: (Ignored by Git) Runtime artifacts, model caches, and temporary outputs.
- **`reports/`**: (Ignored by Git) Generated outputs:
  - `figures/`: All visualization outputs (topic evolution, model comparisons, economic correlations)
  - `Report/acl/latex/`: LaTeX source code for the ACL-format academic paper (`paper.tex`)
  - `topics_over_time_line.html`: Interactive topic evolution visualization

## 🔬 Key Findings

### Model Performance
- **FinBERT-FOMC** achieves the best balanced performance (F1-score: 0.429, Macro F1: 0.38) with 53.0% non-neutral detection, demonstrating the importance of domain adaptation.
- **RoBERTa-Large** shows highest raw accuracy (47.6%) but severe class imbalance (only 4.5% non-neutral predictions), effectively acting as a majority-class classifier.
- Domain-specific models significantly outperform general-domain models, highlighting the technical nuances of "Fedspeak."

### The Inflation Paradox
Our analysis reveals a critical regime-dependent failure in traditional sentiment analysis. During the 2021-2023 inflation shock:
- Negative sentiment (driven by risk-related terminology) correctly detected economic distress
- However, this "negativity" triggered **hawkish** policy (tightening), contradicting the traditional "Negative = Dovish" mapping
- This inversion demonstrates that static sentiment models are prone to misinterpretation during economic turning points

### Topic Evolution
BERTopic analysis identifies three distinct phases:
1. **Pre-2021**: Alternating focus on Monetary Policy and Inflation
2. **2021-2022**: Surge in "Inflation & Down" topic during tightening cycle
3. **2023-Present**: Shift to "Percent & Inflation" technical disinflation discourse

## 📖 References
- **Loughran, T., & McDonald, B. (2011).** When is a Liability not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *The Journal of Finance*.
- **Hansen, S., & McMahon, M. (2016).** Shocking language: Understanding the effects of central bank communication. *Journal of International Economics*.
- **Shah, A., et al. (2023).** Trillion Dollar Words: A New Dataset for FOMC Communication. *ACM Conference on Knowledge Discovery and Data Mining*.
- **Grootendorst, M. (2022).** BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint*.
- **Taylor, J. B. (1993).** Discretion versus policy rules in practice. *Carnegie-Rochester Conference Series on Public Policy*.

## 👤 Author
**Xiaochen Liu**  
Linköping University, Sweden  
Email: xiali125@student.liu.se

## 📄 Citation
If you use this code or findings in your research, please cite:
```
Analysis of Federal Open Market Committee Communication: 
Topic Evolution and Sentiment Analysis (2018-2024)
Xiaochen Liu, Linköping University
```