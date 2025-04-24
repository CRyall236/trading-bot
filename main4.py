import yfinance as yf
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup

# === Technical Indicator Functions ===
def calculate_rsi(prices, period=14):
    gains = []
    losses = []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(max(change, 0))
        losses.append(abs(min(change, 0)))
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_ema(prices, span):
    return sum(prices[-span:]) / span

def calculate_macd(prices):
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    macd = ema12 - ema26
    signal = calculate_ema([macd] * 9, 9)
    return macd, signal

def calculate_bollinger_bands(prices, period=20):
    sma = sum(prices[-period:]) / period
    std = (sum([(price - sma)**2 for price in prices[-period:]]) / period) ** 0.5
    upper_band = sma + 2 * std
    lower_band = sma - 2 * std
    return upper_band, lower_band

def calculate_stochastic(highs, lows, closes, k_period=14, d_period=3):
    low_min = min(lows[-k_period:])
    high_max = max(highs[-k_period:])
    k = 100 * (closes[-1] - low_min) / (high_max - low_min)
    d = sum([k] * d_period) / d_period
    return k, d

def calculate_atr(highs, lows, closes, period=14):
    if len(highs) < 2 or len(lows) < 2 or len(closes) < 2:
        return 0  # Return 0 if there isn't enough data
    tr = max(highs[-1] - lows[-1], abs(highs[-1] - closes[-2]), abs(lows[-1] - closes[-2]))
    return tr

# === Scoring Signal Logic with Window Averaging and Weights ===
def score_signals(prices, highs, lows, closes, window=3):
    if len(prices) < window or len(highs) < window or len(lows) < window or len(closes) < window:
        return 0  # Skip if not enough data for the window size

    score = 0
    recent_prices = prices[-window:]
    recent_highs = highs[-window:]
    recent_lows = lows[-window:]
    recent_closes = closes[-window:]

    # Compute the indicators and calculate scores
    rsi_score = calculate_rsi(recent_prices)
    macd_score, signal_line = calculate_macd(recent_prices)
    upper_bb, lower_bb = calculate_bollinger_bands(recent_prices)
    k, d = calculate_stochastic(recent_highs, recent_lows, recent_closes)
    atr = calculate_atr(recent_highs, recent_lows, recent_closes)

    # Scoring based on the indicators
    if rsi_score < 40:
        score += 1
    elif rsi_score > 60:
        score -= 1

    if macd_score > signal_line:
        score += 1
    else:
        score -= 1

    if recent_prices[-1] < lower_bb:
        score += 1
    elif recent_prices[-1] > upper_bb:
        score -= 1

    if k < 30 and d < 30:
        score += 1
    elif k > 70 and d > 70:
        score -= 1

    return score

# === Backtest Function ===
def backtest_signals(prices, highs, lows, closes, days_forward=5):
    results = []
    for i in range(len(prices) - days_forward):
        score = score_signals(prices[:i+1], highs[:i+1], lows[:i+1], closes[:i+1])
        if score >= 2:  # Buy or Strong Buy
            entry_price = closes[i]
            exit_price = closes[i + days_forward]
            return_pct = ((exit_price - entry_price) / entry_price) * 100
            results.append({
                'Entry_Price': round(entry_price, 2),
                'Exit_Price': round(exit_price, 2),
                'Return_%': round(return_pct, 2)
            })
    return results

# === Ticker Processing ===
def process_ticker(ticker, data):
    ticker_start = time.time()
    print(f"Processing {ticker}...")

    # Calculate indicators for the given data
    df = data.copy()
    df['RSI'] = calculate_rsi(df)
    df['EMA12'] = calculate_ema(df, span=12)
    df['MACD'], df['Signal_Line'] = calculate_macd(df)
    df['VolumeMA'] = calculate_vma(df)
    df['Upper_BB'], df['Lower_BB'] = calculate_bollinger_bands(df)
    df['%K'], df['%D'] = calculate_stochastic(df)
    df['ATR'] = calculate_atr(df)
    
    # Generate signal for the given data
    signal = generate_signal(df)

    # Get the latest data point for metrics
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
        'Signal': signal,
        'Date': latest.name.strftime('%Y-%m-%d')  # Current candle's date
    }

    print(f"{ticker} processed in {time.time() - ticker_start:.2f} seconds")
    
    # Return the ticker and its metrics as a tuple
    return ticker, metrics

# === Fetch Tickers ===
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable'})
    tickers = []
    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        if len(cells) > 1:
            tickers.append(cells[0].get_text(strip=True))
    return tickers

def get_ftse100_tickers():
    url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable'})
    tickers = []
    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        if len(cells) > 1:
            ticker = cells[0].get_text(strip=True)
            tickers.append(ticker.replace('.', '-') + '.L')
    return tickers

# === Main Bot Function ===
def run_trading_bot():
    start_time = time.time()  # Ensure start_time is defined within the function

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

    # Now print the total run time inside the function
    print(f"Total run time: {time.time() - start_time:.2f} seconds")

# === Run Immediately ===
run_trading_bot()
