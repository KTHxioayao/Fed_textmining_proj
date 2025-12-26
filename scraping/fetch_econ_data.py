import pandas_datareader.data as web
import pandas as pd
import os
from datetime import datetime

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUTPUT_FILE = os.path.join(DATA_DIR, 'econ_indicators.csv')

# FRED Series IDs
# Following the methodology from "Trillion Dollar Words":
# 1. CPI: Consumer Price Index for All Urban Consumers: All Items (CPIAUCSL)
# 2. PPI: Producer Price Index for All Commodities (PPIACO) - Used for long-term historical consistency
INDICATORS = {
    'CPIAUCSL': 'CPI',
    'PPIACO': 'PPI',
    'FEDFUNDS': 'Fed_Funds_Rate' 
}

def fetch_econ_data(start_date='2018-01-01', end_date=None):
    """
    Fetches economic data from FRED.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Fetching economic indicators from FRED ({start_date} to {end_date})...")
    
    try:
        # Fetch data
        df = web.DataReader(list(INDICATORS.keys()), 'fred', start_date, end_date)
        
        # Rename columns
        df.rename(columns=INDICATORS, inplace=True)
        
        # Calculate Year-over-Year (YoY) change
        # Methodology: "percentage change from last year"
        # Formula: (Current - Year_Ago) / Year_Ago * 100
        df['CPI_YoY'] = df['CPI'].pct_change(12) * 100
        df['PPI_YoY'] = df['PPI'].pct_change(12) * 100
        df['Fed_Funds_Rate_YoY'] = df['Fed_Funds_Rate'].pct_change(12) * 100
        
        # Reset index to make DATE a column
        df.reset_index(inplace=True)
        
        return df

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
        print(df.tail())
    else:
        print("No data to save.")

if __name__ == "__main__":
    # Fetch starting from 2017 to ensure we have data for 2018 YoY calculations
    # (Need 12 months of prior data for the first pct_change calculation)
    df = fetch_econ_data(start_date='2017-01-01')
    save_data(df)