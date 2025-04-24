import yfinance as yf
import pandas as pd
import numpy as np

# === Technical Indicator Functions === 
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(df, span):
    return df['Close'].ewm(span=span, adjust=False).mean()

def calculate_macd(df):
    ema12 = calculate_ema(df, 12)
    ema26 = calculate_ema(df, 26)
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_vma(df, period=20):
    return df['Volume'].rolling(window=period).mean()

def calculate_bollinger_bands(df, period=20):
    sma = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    return upper_band, lower_band

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
    atr = tr.rolling(window=period).mean()
    return atr

# === Scoring Signal Logic with Window Averaging and Weights ===
def score_signals(row, window=3):
    score = 0

    # Make sure we are accessing individual row values (not series)
    rsi_value = row['RSI']
    macd_value = row['MACD']
    signal_line_value = row['Signal_Line']
    close_value = row['Close']
    lower_bb_value = row['Lower_BB']
    upper_bb_value = row['Upper_BB']
    k_value = row['%K']
    d_value = row['%D']

    # RSI conditions
    if isinstance(rsi_value, (int, float)):
        if rsi_value < 40:
            score += 1
        elif rsi_value > 60:
            score -= 1
    else:
        print(f"Non-scalar RSI value: {rsi_value}")

    # MACD conditions
    if isinstance(macd_value, (int, float)):
        if macd_value > signal_line_value:
            score += 1
        else:
            score -= 1
    else:
        print(f"Non-scalar MACD value: {macd_value}")

    # Bollinger Bands conditions
    if isinstance(close_value, (int, float)):
        if close_value < lower_bb_value:
            score += 1
        elif close_value > upper_bb_value:
            score -= 1
    else:
        print(f"Non-scalar Close value: {close_value}")

    # Stochastic conditions
    if isinstance(k_value, (int, float)) and isinstance(d_value, (int, float)):
        if k_value < 30 and d_value < 30:
            score += 1
        elif k_value > 70 and d_value > 70:
            score -= 1
    else:
        print(f"Non-scalar Stochastic value: K={k_value}, D={d_value}")

    # Weighted total score
    score = (score * 1.2)
    return score

def generate_signal(row):
    score = score_signals(row)
    if score >= 4:
        return "Strong Buy"
    elif score >= 2:
        return "Buy"
    elif score <= -4:
        return "Strong Sell"
    elif score <= -2:
        return "Sell"
    else:
        return "Hold"

# === Backtesting Logic ===
def backtest(tickers, start_date, end_date):
    results = []

    for ticker in tickers:
        print(f"Processing {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Calculate Technical Indicators
        data['RSI'] = calculate_rsi(data)
        data['EMA50'] = calculate_ema(data, 50)
        data['MACD'], data['Signal_Line'] = calculate_macd(data)
        data['Upper_BB'], data['Lower_BB'] = calculate_bollinger_bands(data)
        data['%K'], data['%D'] = calculate_stochastic(data)
        
        # Generate Signals
        data['Signal'] = data.apply(generate_signal, axis=1)
        
        # Calculate Profit/Loss based on the generated signals
        data['Buy_Signal'] = (data['Signal'] == 'Strong Buy') | (data['Signal'] == 'Buy')
        data['Sell_Signal'] = (data['Signal'] == 'Strong Sell') | (data['Signal'] == 'Sell')
        
        # Calculate Profit/Loss for each signal
        data['Position'] = np.nan
        data['Position'] = np.where(data['Buy_Signal'], 1, data['Position'])
        data['Position'] = np.where(data['Sell_Signal'], -1, data['Position'])
        
        # Forward fill the position (buy/hold/sell)
        data['Position'].fillna(method='ffill', inplace=True)

        # Calculate profit/loss
        data['Daily_Return'] = data['Close'].pct_change()
        data['Strategy_Return'] = data['Daily_Return'] * data['Position']
        
        data['Cumulative_Return'] = (1 + data['Strategy_Return']).cumprod()

        results.append({
            'Ticker': ticker,
            'Cumulative_Return': data['Cumulative_Return'].iloc[-1],
            'Total_Profit': data['Cumulative_Return'].iloc[-1] - 1
        })

        # Save results to CSV
        data.to_csv(f"{ticker}_backtest_results.csv")

    # Create a summary DataFrame
    summary = pd.DataFrame(results)
    summary.to_csv('backtest_summary.csv', index=False)
    print("Backtest complete! Results saved to CSV.")

# === Main Program ===
if __name__ == "__main__":
    # Example tickers (S&P 500 and FTSE 100)
    sp500_tickers = ['AAPL', 'MSFT', 'GOOG']  # Add more tickers as needed
    ftse100_tickers = ['HSBC', 'VOD', 'BP']  # Add more tickers as needed
    all_tickers = sp500_tickers + ftse100_tickers

    # Run backtest
    backtest(all_tickers, start_date='2023-01-01', end_date='2024-01-01')
