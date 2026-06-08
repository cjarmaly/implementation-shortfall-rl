import yfinance as yf
import numpy as np
import pandas as pd

def get_data(ticker, period, interval):
    """
    Download data from Yahoo Finance
    ticker: str
    period: str
    interval: str
    returns: dataframe with OHLCV data
    """
    data = yf.download(ticker, period=period, interval=interval)

    return data



if __name__ == "__main__":
    df = get_data("RKLB", period="60d", interval="2m")
    print(df.shape)
    
