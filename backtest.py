import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
import os
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import argparse
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# === Ticker Fetching Functions ===
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    tickers = [row.find_all('td')[0].text.strip().replace('.', '-') for row in table.find_all('tr')[1:]]
    return tickers

def get_ftse100_tickers():
    url = 'https://en.wikipedia.org/wiki/FTSE_100_Index'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = [row.find_all('td')[1].text.strip().replace('.', '-') for row in table.find_all('tr')[1:] if len(row.find_all('td')) > 1]
    return tickers

# === Technical Indicator Functions ===
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df):
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(df, window=20):
    sma = df['Close'].rolling(window).mean()
    std = df['Close'].rolling(window).std()
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std
    return upper_band, lower_band

def calculate_stochastic(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# === Signal Scoring ===
def score_signals(df):
    latest = df.iloc[-1]
    score = 0
    if latest['RSI'] < 30:
        score += 1
    if latest['MACD'] > latest['Signal_Line']:
        score += 1
    if latest['MACD_Crossover'] == 1:
        score += 1
    if latest['Close'] < latest['Lower_BB']:
        score += 1
    if latest['%K'] < 20 and latest['%D'] < 20:
        score += 1
    if latest['ATR'] < df['ATR'].mean():
        score += 1
    return score

# === Generate Final Signal ===
def generate_signal(df):
    recent = df.iloc[-1]
    if recent['Signal'] == 'Buy':
        return 'Buy'
    elif recent['Signal'] == 'Sell':
        return 'Sell'
    else:
        return 'Hold'

# === Backtesting Metrics ===
def calculate_metrics(df):
    df = df.copy()
    df['Daily_Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)

    start_value = df['Close'].iloc[0]
    end_value = df['Close'].iloc[-1]
    total_days = (df.index[-1] - df.index[0]).days
    cagr = ((end_value / start_value) ** (365.0 / total_days)) - 1

    mean_daily_return = df['Daily_Return'].mean()
    std_daily_return = df['Daily_Return'].std()
    sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252)

    return sharpe_ratio, cagr

# === Backtest Strategy ===
def backtest_strategy(df):
    balance = 1000
    position = 0
    entry_price = 0
    df['Signal'] = ''

    for i in range(20, len(df)):
        window_df = df.iloc[:i].copy()
        score = score_signals(window_df)
        if score >= 5 and position == 0:
            position = balance / df.iloc[i]['Close']
            entry_price = df.iloc[i]['Close']
            df.at[df.index[i], 'Signal'] = 'Buy'
            balance = 0
        elif score <= -2 and position > 0:
            balance = position * df.iloc[i]['Close']
            position = 0
            df.at[df.index[i], 'Signal'] = 'Sell'
        else:
            df.at[df.index[i], 'Signal'] = 'Hold'

    if position > 0:
        balance = position * df.iloc[-1]['Close']

    df['Final_Equity'] = balance + position * df['Close']
    return balance - 1000, df

# === Chart Plotting ===
def plot_chart(df, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['Final_Equity'], label='Equity Curve')
    plt.title(f"{ticker} Backtest")
    plt.legend()
    plt.savefig(f"charts/{ticker}.png")
    plt.close()

# === Main Backtest Run ===
def run_backtest(index="both", limit=50, export_charts=True):
    if index == "sp500":
        tickers = get_sp500_tickers()
    elif index == "ftse":
        tickers = get_ftse100_tickers()
    else:
        tickers = get_sp500_tickers() + get_ftse100_tickers()
    tickers = tickers[:limit]

    os.makedirs("charts", exist_ok=True)
    results = []

    for ticker in tqdm(tickers, desc="Processing"):
        try:
            df = yf.download(ticker, period="12mo", interval="1d", auto_adjust=True)
            df['RSI'] = calculate_rsi(df)
            df['MACD'], df['Signal_Line'] = calculate_macd(df)
            df['MACD_Crossover'] = np.where(
                (df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1)), 1,
                np.where((df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1)), -1, 0)
            )
            df['Upper_BB'], df['Lower_BB'] = calculate_bollinger_bands(df)
            df['%K'], df['%D'] = calculate_stochastic(df)
            df['ATR'] = calculate_atr(df)
            df.dropna(inplace=True)

            if df.empty:
                raise ValueError("Dataframe is empty after indicator calculations.")

            profit, updated_df = backtest_strategy(df)
            signal = generate_signal(updated_df)
            sharpe, cagr = calculate_metrics(updated_df)

            if export_charts:
                plot_chart(updated_df, ticker)

            results.append({
                "Ticker": ticker,
                "Final Profit": profit,
                "Signal": signal,
                "Sharpe Ratio": round(sharpe, 2),
                "CAGR": round(cagr * 100, 2)
            })
        except Exception as e:
            print(f"Failed {ticker}: {e}")

    summary = pd.DataFrame(results)
    if not summary.empty:
        summary.to_csv("backtest_summary.csv", index=False)
        print(summary.sort_values(by="Final Profit", ascending=False).head(10))
    else:
        print("No valid results to display.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', choices=['sp500', 'ftse', 'both'], default='both')
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--no-charts', action='store_true')
    args = parser.parse_args()

    run_backtest(index=args.index, limit=args.limit, export_charts=not args.no_charts)
