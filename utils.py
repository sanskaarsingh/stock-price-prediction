import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv


load_dotenv()

def get_marketstack_data(symbol, start_date, end_date):
    """
    Fetch historical stock data from MarketStack API
    """
    api_key = os.getenv('MARKETSTACK_API_KEY')
    if not api_key:
        raise ValueError("MarketStack API key not found in environment variables")
    
    base_url = "http://api.marketstack.com/v1/eod"
    params = {
        'access_key': api_key,
        'symbols': symbol,
        'date_from': start_date,
        'date_to': end_date
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        if 'data' not in data or not data['data']:
            raise ValueError("No data returned from API")
        
        df = pd.DataFrame(data['data'])
        
       
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        
        cols_to_drop = ['symbol', 'exchange', 'adj_high', 'adj_low', 'adj_open', 
                       'split_factor', 'dividend']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
        
        
        numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        
        df = df.select_dtypes(include=[np.number])
        
        return df
    
    except Exception as e:
        raise ValueError(f"Error fetching data: {str(e)}")

def calculate_technical_indicators(df):
    """
    Calculate technical indicators for the stock data
    """
    try:
        if df.empty:
            return df
            
       
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        
        
        df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        
        df['Middle_Band'] = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['Upper_Band'] = df['Middle_Band'] + (std * 2)
        df['Lower_Band'] = df['Middle_Band'] - (std * 2)
        
       
        df.dropna(inplace=True)
        
        return df
    
    except Exception as e:
        raise ValueError(f"Error calculating indicators: {str(e)}")