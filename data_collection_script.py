import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    return data

def prepare_data(data):
    # Calculate additional features
    data['Returns'] = data['Close'].pct_change()
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['Volatility'] = data['Returns'].rolling(window=20).std() * (252 ** 0.5)
    
    # Drop rows with NaN values
    data.dropna(inplace=True)
    
    return data

def main():
    # Set parameters
    symbol = 'AAPL'  # Example: Apple Inc.
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5 years of data
    
    # Fetch and prepare data
    raw_data = fetch_stock_data(symbol, start_date, end_date)
    prepared_data = prepare_data(raw_data)
    
    # Save to CSV
    prepared_data.to_csv(f'{symbol}_stock_data.csv')
    print(f"Data for {symbol} has been saved to {symbol}_stock_data.csv")

if __name__ == "__main__":
    main()
