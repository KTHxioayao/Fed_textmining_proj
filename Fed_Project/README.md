# The Fedspeak Effect: Quantifying the Causality between Central Bank Sentiment and Bond Market Volatility

## Project Goal
Quantify how Federal Reserve Chair Jerome Powell's speech sentiment drives market pricing, specifically targeting the 10-year Treasury Yield (TNX) and Bond Market Volatility (MOVE Index).

## Target Grade: A (LiU 732A81)

## Workflow
1.  **Data Collection**: 
    - Scrape Federal Reserve speeches (2018-2024).
    - Fetch Market Data (TNX, MOVE) via `yfinance`.
2.  **Modeling**:
    - Baseline: Loughran-McDonald Dictionary.
    - Advanced: FinBERT (Zero-shot/Fine-tuned).
3.  **Analysis**:
    - Correlation Matrix.
    - Granger Causality Test.
    - Error Analysis on outliers.

## References
- Loughran & McDonald (2011)
- Hansen & McMahon (2016)
- Ardia et al. (2019)
