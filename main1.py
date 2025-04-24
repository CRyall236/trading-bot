import yfinance as yf
import pandas as pd
import schedule
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

# === Predictive Model Functions ===
def prepare_features(df):
    df = df.copy()
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_vs_EMA'] = df['Close'] - df['EMA12']
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    df.dropna(subset=['RSI', 'EMA12', 'MACD', 'VolumeMA', 'Price_Change', 'Price_vs_EMA', 'MACD_Hist'], inplace=True)
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df.dropna(inplace=True)
    features = df[['RSI', 'EMA12', 'MACD', 'VolumeMA', 'Price_Change', 'Price_vs_EMA', 'MACD_Hist']]
    target = df['Target']
    return features, target

def run_predictive_model(df, ticker):
    features, target = prepare_features(df)
    print(f"{ticker} class balance: {target.value_counts().to_dict()}")  # Debug info

    if len(features) < 30:
        return "N/A", 0.0  # Not enough data

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    prob = model.predict_proba(features.iloc[[-1]])[0][1]  # Probability of "Up"
    prediction = "Up" if prob > 0.55 else "Down"
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    return prediction, report['accuracy']

# === Signal Generation Logic ===
def generate_signal(df):
    if df.empty or 'RSI' not in df.columns or df['RSI'].isnull().all():
        return "No Signal"
    latest = df.iloc[-1]
    if latest['RSI'] < 40 and latest['MACD'] > latest['Signal_Line']:
        return "Buy"
    elif latest['RSI'] > 80 and latest['MACD'] < latest['Signal_Line']:
        return "Sell"
    else:
        return "No Signal"

# === Ticker Processing ===
def process_ticker(ticker, df):
    ticker_start = time.time()
    print(f"Processing {ticker}...")
    df = df.copy()
    df['RSI'] = calculate_rsi(df)
    df['EMA12'] = calculate_ema(df, span=12)
    df['MACD'], df['Signal_Line'] = calculate_macd(df)
    df['VolumeMA'] = calculate_vma(df)
    signal = generate_signal(df)
    prediction, accuracy = run_predictive_model(df, ticker)

    latest = df.iloc[-1]
    metrics = {
        'RSI': latest['RSI'],
        'MACD': latest['MACD'],
        'Signal_Line': latest['Signal_Line'],
        'EMA12': latest['EMA12'],
        'Signal': signal,
        'Prediction': prediction,
        'Model_Accuracy': round(accuracy, 3)
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

    signals_df.to_csv("sp500_ftse100_signals.csv", index=False)
    print("Signals and metrics exported to sp500_ftse100_signals.csv")

    print("Signals:", signals)
    print(f"Total run time: {time.time() - start_time:.2f} seconds")

# === Run Immediately ===
run_trading_bot()
