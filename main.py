import yfinance as yf
import pandas as pd
import schedule
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

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
def score_signals(df, window=3):
    score = 0
    recent = df.iloc[-window:]

    rsi_score = 0
    macd_score = 0
    bb_score = 0
    stoch_score = 0

    for _, row in recent.iterrows():
        if row['RSI'] < 40:
            rsi_score += 1
        elif row['RSI'] > 60:
            rsi_score -= 1

        if row['MACD'] > row['Signal_Line']:
            macd_score += 1
        else:
            macd_score -= 1

        # Add crossover scoring
        if row.get('MACD_Crossover', 0) == 1:
            macd_score += 1.5  # bullish crossover
        elif row.get('MACD_Crossover', 0) == -1:
            macd_score -= 1.5  # bearish crossover

        if row['Close'] < row['Lower_BB']:
            bb_score += 1
        elif row['Close'] > row['Upper_BB']:
            bb_score -= 1

        if row['%K'] < 30 and row['%D'] < 30:
            stoch_score += 1
        elif row['%K'] > 70 and row['%D'] > 70:
            stoch_score -= 1

#weighted scroring
    score = (rsi_score * 1.2) + (macd_score * 1.5) + (bb_score * 1.2) + (stoch_score * 1.1)
    return score

def generate_signal(df):
    if df.empty or df[['RSI', 'MACD', 'Signal_Line']].isnull().all().any():
        return "No Signal"
    score = score_signals(df)
    if score >= 5:
        return "Strong Buy"
    elif score >= 4:
        return "Buy"
    elif score <= -4:
        return "Strong Sell"
    elif score <= -2:
        return "Sell"
    else:
        return "Hold"

def backtest_signals(df, days_forward=5):
    results = []
    for i in range(len(df) - days_forward):
        window = df.iloc[i - 2:i + 1] if i >= 2 else df.iloc[0:i + 1]
        score = score_signals(window)

        if score >= 2:  # Buy or Strong Buy
            entry_date = df.index[i]
            entry_price = df.iloc[i]['Close']
            exit_price = df.iloc[i + days_forward]['Close']
            return_pct = ((exit_price - entry_price) / entry_price) * 100

            results.append({
                'Date': entry_date.strftime('%Y-%m-%d'),
                'Entry_Price': round(entry_price, 2),
                'Exit_Price': round(exit_price, 2),
                'Return_%': round(return_pct, 2)
            })

    return results

# === Ticker Processing ===
def process_ticker(ticker, df):
    ticker_start = time.time()
    print(f"Processing {ticker}...")
    df = df.copy()
    df['RSI'] = calculate_rsi(df)
    df['EMA12'] = calculate_ema(df, span=12)
    df['MACD'], df['Signal_Line'] = calculate_macd(df)
    df['VolumeMA'] = calculate_vma(df)
    df['Upper_BB'], df['Lower_BB'] = calculate_bollinger_bands(df)
    df['%K'], df['%D'] = calculate_stochastic(df)
    df['ATR'] = calculate_atr(df)

    # === MACD Crossover Detection ===
    df['MACD_Crossover'] = 0
    df.loc[
        (df['MACD'].shift(1) < df['Signal_Line'].shift(1)) & (df['MACD'] > df['Signal_Line']),
        'MACD_Crossover'
    ] = 1  # Bullish crossover
    df.loc[
        (df['MACD'].shift(1) > df['Signal_Line'].shift(1)) & (df['MACD'] < df['Signal_Line']),
        'MACD_Crossover'
    ] = -1  # Bearish crossover

    signal = generate_signal(df)

    latest = df.iloc[-1]
    metrics = {
        'RSI': round(latest['RSI'], 2),
        'MACD': round(latest['MACD'], 2),
        'Signal_Line': round(latest['Signal_Line'], 2),
        'EMA12': round(latest['EMA12'], 2),
        'VolumeMA': round(latest['VolumeMA'], 2),
        'Upper_BB': round(latest['Upper_BB'], 2),
        'Lower_BB': round(latest['Lower_BB'], 2),
        '%K': round(latest['%K'], 2),
        '%D': round(latest['%D'], 2),
        'ATR': round(latest['ATR'], 2),
        'MACD_Crossover': int(latest['MACD_Crossover']),
        'Signal': signal,
        'Date': latest.name.strftime('%Y-%m-%d')
    }

    print(f"{ticker} processed in {time.time() - ticker_start:.2f} seconds")
    return ticker, metrics

# === Fetch Tickers ===
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    return df['Symbol'].tolist()

def get_ftse100_tickers():
    url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
    tables = pd.read_html(url, header=0)
    for table in tables:
        for col in table.columns:
            if 'Ticker' in col or 'EPIC' in col or 'Symbol' in col:
                tickers = table[col].dropna().tolist()
                return [ticker.replace('.', '-') + '.L' for ticker in tickers]
    raise ValueError("Couldn't find a ticker column in any of the tables.")

# === Main Bot Function ===
def run_trading_bot():
    start_time = time.time()

    sp500_tickers = get_sp500_tickers()
    ftse100_tickers = get_ftse100_tickers()
    all_tickers = sp500_tickers + ftse100_tickers

    print("Downloading data...")
    download_start = time.time()
    data = yf.download(all_tickers, period="6mo", interval="1d", group_by='ticker', auto_adjust=True, threads=True)
    print(f"Download completed in {time.time() - download_start:.2f} seconds")

    signals = {}
    print("Processing tickers...")
    process_start = time.time()
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_ticker, ticker, data[ticker])
            for ticker in all_tickers if ticker in data
        ]
        for future in futures:
            ticker, metrics = future.result()
            signals[ticker] = metrics
    print(f"Processing completed in {time.time() - process_start:.2f} seconds")

    signals_df = pd.DataFrame.from_dict(signals, orient='index')
    signals_df.reset_index(inplace=True)
    signals_df.rename(columns={'index': 'Ticker'}, inplace=True)

    # Optional: Backtesting mode
    backtest_mode = True
    all_backtest_results = []

    if backtest_mode:
        print("Running backtests...")
        for ticker in all_tickers:
            if ticker in data:
                df = data[ticker].copy()
                df['RSI'] = calculate_rsi(df)
                df['EMA12'] = calculate_ema(df, span=12)
                df['MACD'], df['Signal_Line'] = calculate_macd(df)
                df['VolumeMA'] = calculate_vma(df)
                df['Upper_BB'], df['Lower_BB'] = calculate_bollinger_bands(df)
                df['%K'], df['%D'] = calculate_stochastic(df)
                df['ATR'] = calculate_atr(df)

                backtest_results = backtest_signals(df)
                for result in backtest_results:
                    result['Ticker'] = ticker
                    all_backtest_results.append(result)

        backtest_df = pd.DataFrame(all_backtest_results)
        backtest_df.to_csv("backtest_results.csv", index=False)
        print(f"Backtest results saved to backtest_results.csv")

        if not backtest_df.empty:
            avg_return = backtest_df['Return_%'].mean()
            win_rate = (backtest_df['Return_%'] > 0).mean() * 100
            total_trades = len(backtest_df)
            print(f"Total Trades: {total_trades}")
            print(f"Average Return: {avg_return:.2f}%")
            print(f"Win Rate: {win_rate:.2f}%")

    signals_df.to_csv("enhanced_signals_with_scores.csv", index=False)
    print("Signals and metrics exported to enhanced_signals_with_scores.csv")
    print("Signals:", signals)
    print(f"Total run time: {time.time() - start_time:.2f} seconds")

# === Run Immediately ===
run_trading_bot()
