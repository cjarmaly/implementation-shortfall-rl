import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from constants import DATA_PARAMS


def fetch_data(ticker, period, interval):
    """
    Download data from Yahoo Finance
    ticker: str
    period: str
    interval: str
    returns: dataframe with OHLCV data
    """

    # multi_level_index=False is used to avoid future squeeze
    data = yf.download(ticker, period=period, interval=interval, multi_level_index=False)

    return data

def compute_realized_vol(df, window=10):
    """
    Compute realized volatility
    df: dataframe with OHLCV data
    returns: realized volatility
    """
    # infer bar duration in minutes from the index
    bar_in_minutes = (df.index[1] - df.index[0]).total_seconds() / 60
    bars_per_day = 390 / bar_in_minutes # 390 minutes in a trading day
    annualized_factor = np.sqrt(bars_per_day * 252) # 252 trading days in a year

    # compute log returns
    log_returns = np.log(df["Close"] / df["Close"].shift(1))
    rolling_std = log_returns.rolling(window=window).std()

    return rolling_std * annualized_factor

def compute_intraday_vol_profile(df, window=10):
    """
    Compute intraday volatility
    df: dataframe with OHLCV data
    returns: intraday volatility
    """
    vol = compute_realized_vol(df, window=window)
    vol = vol.dropna()

    vol_df = vol.to_frame(name="vol")

    vol_df["time"] = vol_df.index.time
    vol_df["date"] = vol_df.index.date

    # drop first `window` bars of each day to prevent cross-day data contamination
    vol_df["bar_num"] = vol_df.groupby("date").cumcount()
    vol_df = vol_df[vol_df["bar_num"] >= window]

    profile = vol_df.groupby("time")["vol"].mean()

    return profile


def compute_ou_paramters(profile):
    """
    Fit OU parameters (theta, mu, sigma) to the intraday volatility profile using MLE.
    profile: series of intraday volatility
    returns: (theta, mu, sigma)
    """

    v = profile.dropna().values
    n = len(v)
    
    # analytical OU parameter estimation via linear regression
    # X[t+1] = alpha + beta * X[t] + epsilon
    # where alpha = mu*(1 - e^{-theta}), beta = e^{-theta}
    
    x = v[:-1]
    y = v[1:]
    
    beta  = (n * np.sum(x*y) - np.sum(x)*np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    alpha = (np.sum(y) - beta * np.sum(x)) / n
    
    residuals = y - alpha - beta * x
    sigma_sq  = np.var(residuals)
    
    theta = -np.log(beta)
    mu    = alpha / (1 - beta)
    sigma = np.sqrt(sigma_sq * 2 * theta / (1 - beta**2))
    
    return {"theta": theta, "mu": mu, "sigma": sigma}




    




if __name__ == "__main__":
    df = fetch_data(DATA_PARAMS["ticker"], period=DATA_PARAMS["period"], interval=DATA_PARAMS["interval"])
    vol = compute_realized_vol(df, window=30)
    profile = compute_intraday_vol_profile(df)
    parameters = compute_ou_paramters(profile)

    # Print OU parameters
    print(f"Theta: {parameters['theta']}, Mu: {parameters['mu']}, Sigma: {parameters['sigma']}")

    # Plot realized volatility
    plt.figure(figsize=(12, 4))
    plt.plot(vol.index, vol.values)
    plt.title("Realized Vol — SPY 2-minute bars")
    plt.xlabel("Date")
    plt.ylabel("Annualized vol")
    plt.savefig("realized_vol.png")
    plt.xticks(rotation=45)
    plt.show()
    plt.close()

    # Plot intraday volatility profile
    plt.figure(figsize=(12, 4))
    plt.plot(profile.values)
    plt.title("Average Intraday Vol Profile — SPY")
    plt.xlabel("Bar (9:30am → 4:00pm)")
    plt.ylabel("Avg annualized vol")
    plt.tight_layout()
    plt.show()
    
