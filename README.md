# The Fedspeak Effect
> **Quantifying the Causality between Central Bank Sentiment and Bond Market Volatility**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NLP](https://img.shields.io/badge/NLP-FinBERT-green.svg)](https://huggingface.co/ProsusAI/finbert)

## 📌 Overview
This repository provides a quantitative framework to measure the impact of Federal Reserve communications on financial markets. By leveraging advanced NLP techniques, the project decodes sentiment from FOMC speeches (Jerome Powell era) and analyzes its causal relationship with the **10-year Treasury Yield (TNX)** and **Bond Market Volatility (MOVE Index)**.

## 🚀 Key Features
- **Automated Fed Scraper**: Custom pipeline to collect and structure FOMC speeches, minutes, and press conference transcripts (2018–Present).
- **Dual-Layer Sentiment Analysis**: 
    - **Lexicon-based**: Using the Loughran-McDonald financial dictionary.
    - **Transformer-based**: Utilizing **FinBERT** for context-aware sentiment extraction.
- **Econometric Modeling**: Implementation of Granger Causality tests and correlation matrices to identify lead-lag relationships between "Fedspeak" and market movements.

## 🛠 Technical Implementation
### 1. Data Pipeline
- **NLP Sources**: Web-scraping of the Federal Reserve Board's official site.
- **Market Data**: Real-time integration with `yfinance` for TNX and MOVE index tracking.
- **Processing**: Sentence-level tokenization and cleaning for high-granularity analysis.

### 2. Analytics Engine
- **Sentiment Scoring**: Mapping qualitative text to quantitative sentiment vectors.
- **Causality Testing**: Statistical validation of whether central bank rhetoric significantly "grangers-causes" yield curve shifts.
- **Error Analysis**: Diagnostic tools to investigate outliers and policy "shocks."

## 📂 Project Structure
- `notebooks/`: Exploratory Data Analysis and data integration.
- `scraping/`: Python scripts for automated data collection.
- `data/`: (Local) Structured CSVs for processed corpus and market indicators.

## 📖 References
- **Loughran, T., & McDonald, B. (2011).** When is a Liability not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *The Journal of Finance*.
- **Hansen, S., & McMahon, M. (2016).** Shocking language: Understanding the effects of central bank communication. *Journal of International Economics*.
- **Ardia, D., et al. (2019).** Questioning the News: Sentiment Analysis of Financial Reporting and the Term Structure of Interest Rates.