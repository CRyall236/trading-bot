import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# === Technical Indicators ===
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(df, span):
    return df['Close'].ewm(span=span, adjust=False).mean()

def calculate_macd(df):
    ema12 = calculate_ema(df, 12)
    ema26 = calculate_ema(df, 26)
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(df, period=20):
    sma = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    return upper, lower

def calculate_stochastic(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# === Ticker Utilities ===
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    return tables[0]['Symbol'].tolist()

def get_ftse100_tickers():
    url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
    tables = pd.read_html(url, header=0)
    for table in tables:
        for col in table.columns:
            if 'Ticker' in col or 'EPIC' in col or 'Symbol' in col:
                tickers = table[col].dropna().tolist()
                return [ticker.replace('.', '-') + '.L' for ticker in tickers]
    return []

# === Visualisation ===
def visualize_ticker(ticker, df):
    df = df.copy()
    df['RSI'] = calculate_rsi(df)
    df['EMA12'] = calculate_ema(df, 12)
    df['MACD'], df['Signal_Line'] = calculate_macd(df)
    df['Upper_BB'], df['Lower_BB'] = calculate_bollinger_bands(df)
    df['%K'], df['%D'] = calculate_stochastic(df)
    df['ATR'] = calculate_atr(df)

    df['Crossover_Bullish'] = (df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1))
    df['Crossover_Bearish'] = (df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1))

    df['Score'] = 0
    df.loc[(df['RSI'] < 30), 'Score'] += 1
    df.loc[(df['MACD'] > df['Signal_Line']), 'Score'] += 1
    df.loc[(df['Close'] > df['EMA12']), 'Score'] += 1
    df.loc[(df['%K'] > df['%D']), 'Score'] += 1
    df.loc[(df['Close'] < df['Lower_BB']), 'Score'] += 1

    df['Signal'] = 'Hold'
    df.loc[df['Score'] >= 4, 'Signal'] = 'Buy'
    df.loc[df['Score'] <= 1, 'Signal'] = 'Sell'

    features = ['RSI', 'EMA12', 'MACD', 'ATR', '%K', '%D']
    df = df.dropna(subset=features + ['Close'])

    X = df[features]
    y = df['Close']
    if X.shape[1] != len(features):
        raise ValueError(f"Expected {len(features)} features but got {X.shape[1]}")

    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X, y)

    forecast_inputs = []
    forecast_dates = []
    forecast_signals = []
    current_features = df.iloc[-1][features].copy()
    current_date = df.index[-1]
    us_bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())

    for i in range(1, 6):
        predicted_price = model.predict(pd.DataFrame([current_features], columns=features))[0]
        current_date += us_bday
        forecast_inputs.append(predicted_price)
        forecast_dates.append(current_date)

        score = 0
        if current_features['RSI'] < 30: score += 1
        if current_features['MACD'] > df['Signal_Line'].iloc[-1]: score += 1
        if predicted_price > current_features['EMA12']: score += 1
        if current_features['%K'] > current_features['%D']: score += 1
        if predicted_price < df['Lower_BB'].iloc[-1]: score += 1

        signal = 'Hold'
        if score >= 4: signal = 'Buy'
        elif score <= 1: signal = 'Sell'
        forecast_signals.append(signal)

        current_features['EMA12'] = (current_features['EMA12'] * 11 + predicted_price) / 12

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])

    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close', line=dict(color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_inputs, mode='lines+markers', name='Forecast (+5d)', line=dict(dash='dot', color='orange')), row=1, col=1)

    buy_signals = df[df['Signal'] == 'Buy']
    sell_signals = df[df['Signal'] == 'Sell']

    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers',
                             name='Buy', marker=dict(color='green', symbol='circle', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers',
                             name='Sell', marker=dict(color='red', symbol='x', size=10)), row=1, col=1)

    for date, price, signal in zip(forecast_dates, forecast_inputs, forecast_signals):
        if signal == 'Buy':
            fig.add_trace(go.Scatter(x=[date], y=[price], mode='markers',
                                     name='Forecast Buy', marker=dict(color='green', symbol='triangle-up', size=12)), row=1, col=1)
        elif signal == 'Sell':
            fig.add_trace(go.Scatter(x=[date], y=[price], mode='markers',
                                     name='Forecast Sell', marker=dict(color='red', symbol='triangle-down', size=12)), row=1, col=1)

    # Add MACD and Signal Line to second subplot
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='orange')), row=2, col=1)

    # Mark MACD Bullish and Bearish Crossovers
    bullish_crossovers = df[df['Crossover_Bullish']]
    bearish_crossovers = df[df['Crossover_Bearish']]

    fig.add_trace(go.Scatter(x=bullish_crossovers.index, y=bullish_crossovers['MACD'], mode='markers',
                             name='MACD Bullish', marker=dict(color='lime', symbol='triangle-up', size=8)), row=2, col=1)
    fig.add_trace(go.Scatter(x=bearish_crossovers.index, y=bearish_crossovers['MACD'], mode='markers',
                             name='MACD Bearish', marker=dict(color='red', symbol='triangle-down', size=8)), row=2, col=1)

    fig.update_layout(title=f"{ticker} | Score: {df['Score'].iloc[-1]} | MACD, Forecast & Signals",
                      xaxis_title='Date', yaxis_title='Price', height=800)

    output_dir = 'charts'
    os.makedirs(output_dir, exist_ok=True)
    fig.write_image(f"{output_dir}/{ticker}.png")
    fig.write_html(f"{output_dir}/{ticker}.html")

# === Run Analysis ===
def analyze():
    tickers = get_sp500_tickers() + get_ftse100_tickers()
    data = yf.download(tickers, period="6mo", interval="1d", group_by='ticker', auto_adjust=True, threads=True)

    for ticker in tickers:
        try:
            if ticker not in data:
                continue
            df = data[ticker]
            if not isinstance(df, pd.DataFrame):
                continue
            df = df.dropna().copy()
            if len(df) < 30:
                continue
            visualize_ticker(ticker, df)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

if __name__ == '__main__':
    analyze()
