# The Fedspeak Effect
> **Quantifying the Causality between Central Bank Sentiment and Bond Market Volatility**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NLP](https://img.shields.io/badge/NLP-FinBERT%20%7C%20RoBERTa-green.svg)](https://huggingface.co/ProsusAI/finbert)

## 📌 Overview
This repository provides a quantitative framework to measure the impact of Federal Reserve communications on financial markets. By leveraging advanced NLP techniques, the project decodes sentiment from FOMC documents (Minutes, Press Conferences, Speeches) and analyzes its relationship with economic indicators like the **10-year Treasury Yield (TNX)** and **Fed Funds Rate**.

We compare multiple state-of-the-art transformer models to determine which best captures the nuance of central bank communication ("Fedspeak").

## 🚀 Key Features
- **Comprehensive Data Pipeline**: Web-scraping of Federal Reserve Board documents (Speeches, Minutes, Press Conferences) from 2018 to Present.
- **Advanced Topic Modeling**: Utilizing **BERTopic** to track the evolution of economic themes over time.
- **Multi-Model Sentiment Analysis**: 
    - **FinBERT** (ProsusAI)
    - **FinBERT-FOMC** (specialized for Fed language)
    - **RoBERTa** (Base & Large)
- **Econometric Analysis**: Correlation and visualization of sentiment trends against Macroeconomic data (CPI, PPI, Unemployment, Fed Funds Rate).
- **Academic Output**: Full LaTeX pipeline for generating ACL-format research papers.

## 🛠 Project Workflow (Notebooks)
The analysis is structured into a sequential pipeline located in the `notebooks/` directory:

| Step | Notebook | Description |
|------|----------|-------------|
| 01 | `1_Data_Collection_and_Processing.ipynb` | Scrapes raw text from Federal Reserve sources and performs initial cleaning. |
| 02 | `2_Data_Integration.ipynb` | Merges text data with timestamps and sources into a master corpus. |
| 03 | `3_Exploratory_Analysis.ipynb` | Basic statistics, word distributions, and linguistic feature analysis. |
| 04a| `4_Topic_Modeling_BERTopic.ipynb` | Extracts latent topics (e.g., Inflation, Housing) to visualize thematic shifts. |
| 04b| `4_Model_Evaluation.ipynb` | Evaluates model performance against a Gold Standard dataset. |
| 05 | `6_Sentiment_Pipeline_*.ipynb` | Fine-tuned pipelines for **FinBERT**, **FinBERT-FOMC**, and **RoBERTa** variants. |
| 06 | `7b_Model_Comparison_Visualization.ipynb` | Final comparative visualization of model signals vs. Market Data. |

## 📂 Repository Structure
- **`notebooks/`**: Core analysis and experiment notebooks (see table above).
- **`scripts/`**: Utility scripts (e.g., `fetch_econ_data.py` for retrieving FRED data).
- **`reports/`**: Documentation and academic output, including the LaTeX source for the project paper (`acl/latex/paper.tex`).
- **`utils/`**: Helper modules for common text processing and plotting functions.
- **`data/`**: (Ignored by Git) Local storage for datasets and intermediate model outputs.

## 📖 References
- **Loughran, T., & McDonald, B. (2011).** When is a Liability not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *The Journal of Finance*.
- **Hansen, S., & McMahon, M. (2016).** Shocking language: Understanding the effects of central bank communication. *Journal of International Economics*.
- **Ardia, D., et al. (2019).** Questioning the News: Sentiment Analysis of Financial Reporting and the Term Structure of Interest Rates.