import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
OUTPUT_FILE = os.path.join(DATA_DIR, 'market_data_vix_tnx.csv')

# Tickers to fetch
# ^VIX: CBOE Volatility Index
# ^TNX: CBOE Interest Rate 10 Year T Note
# DX-Y.NYB: US Dollar Index
# ^GSPC: S&P 500
# ^MOVE: ICE BofAML MOVE Index (Note: reliable public data for MOVE is hard to find on Yahoo, sticking to core ones)
TICKERS = ['^VIX', '^TNX', 'DX-Y.NYB', '^GSPC']

def fetch_market_data(start_date='2000-01-01', end_date=None):
    """
    Fetches market data from Yahoo Finance.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Fetching data for {TICKERS} from {start_date} to {end_date}...")
    
    try:
        # Download data
        data = yf.download(TICKERS, start=start_date, end=end_date, group_by='ticker')
        
        # Extract 'Close' prices for each ticker
        df_list = []
        for ticker in TICKERS:
            try:
                # yfinance returns a MultiIndex if multiple tickers, or just the dataframe if one.
                # When group_by='ticker', it's usually Ticker -> OHLCV
                # We need to handle potential failures or empty data
                if ticker in data.columns.levels[0]:
                    ticker_data = data[ticker]['Close'].copy()
                    ticker_data.name = ticker
                    df_list.append(ticker_data)
                else:
                    print(f"Warning: No data found for {ticker}")
            except Exception as e:
                print(f"Error processing {ticker}: {e}")

        if not df_list:
            print("No data could be fetched.")
            return None

        # Combine into a single DataFrame
        market_df = pd.concat(df_list, axis=1)
        
        # Reset index to make Date a column
        market_df.reset_index(inplace=True)
        
        # Rename columns to be friendlier
        column_mapping = {
            '^VIX': 'VIX',
            '^TNX': 'TNX_10Y',
            'DX-Y.NYB': 'Dollar_Index',
            '^GSPC': 'SP500',
            'Date': 'date'
        }
        market_df.rename(columns=column_mapping, inplace=True)
        
        return market_df

    except Exception as e:
        print(f"An error occurred during download: {e}")
        return None

def save_data(df):
    """
    Saves the data to CSV.
    """
    if df is not None:
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Data saved successfully to {OUTPUT_FILE}")
        print(df.head())
    else:
        print("No data to save.")

if __name__ == "__main__":
    print("Starting market data fetch...")
    df = fetch_market_data(start_date='1990-01-01')
    save_data(df)
    print("Done.")
