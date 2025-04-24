import yfinance as yf
import pandas as pd
import datetime
import os
import requests
from bs4 import BeautifulSoup
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands

# === CONFIG ===
START_DATE = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime('%Y-%m-%d')
END_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
EMA_PERIOD = 20
OUTPUT_FOLDER = "results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === TICKERS ===
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    soup = BeautifulSoup(requests.get(url).text, "lxml")
    table = soup.find("table", {"id": "constituents"})
    tickers = [row.find_all("td")[0].text.strip().replace(".", "-") for row in table.find_all("tr")[1:]]
    return tickers

def get_ftse100_tickers():
    url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
    soup = BeautifulSoup(requests.get(url).text, "lxml")
    table = soup.find("table", {"class": "wikitable sortable"})
    tickers = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) >= 2:
            text = cells[1].text.strip()
            if text and text.isalpha():
                tickers.append(text.replace(".", "-") + ".L")
    return tickers

TICKERS = get_sp500_tickers() + get_ftse100_tickers()

# === SCORING STRATEGY ===
def score_signals(df, window=3):
    recent = df.iloc[-window:]
    rsi_score = macd_score = bb_score = stoch_score = 0

    for _, row in recent.iterrows():
        if row['RSI'] < 40:
            rsi_score += 1
        elif row['RSI'] > 60:
            rsi_score -= 1

        if row['MACD'] > row['Signal_Line']:
            macd_score += 1
        else:
            macd_score -= 1

        if row['Close'] < row['Lower_BB']:
            bb_score += 1
        elif row['Close'] > row['Upper_BB']:
            bb_score -= 1

        if row['%K'] < 30 and row['%D'] < 30:
            stoch_score += 1
        elif row['%K'] > 70 and row['%D'] > 70:
            stoch_score -= 1

    # Weighted total
    score = (rsi_score * 1.2) + (macd_score * 1.5) + (bb_score * 1.2) + (stoch_score * 1.1)
    return score

def generate_signal(df):
    if df.empty or df[['RSI', 'MACD', 'Signal_Line']].isnull().all().any():
        return "No Signal"
    score = score_signals(df)
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

# === BACKTEST LOGIC ===
def backtest_simple(df):
    position = None
    buy_price = 0
    profit = 0
    trades = []

    for i in range(EMA_PERIOD, len(df)):
        row = df.iloc[:i+1].copy()
        signal = generate_signal(row)

        if signal in ["Buy", "Strong Buy"] and position is None:
            buy_price = row.iloc[-1]['Close']
            trades.append(('Buy', row.index[-1], buy_price))
            position = 'Long'
        elif signal in ["Sell", "Strong Sell"] and position == 'Long':
            sell_price = row.iloc[-1]['Close']
            trades.append(('Sell', row.index[-1], sell_price))
            profit += sell_price - buy_price
            position = None

    return profit, trades

# === PROCESS TICKERS ===
results = []

for ticker in TICKERS:
    print(f"Processing {ticker}...")
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

        if df.empty:
            print(f"No data for {ticker}")
            continue

        # Indicators
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        df['EMA'] = EMAIndicator(df['Close'], window=EMA_PERIOD).ema_indicator()
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['Signal_Line'] = macd.macd_signal()
        bb = BollingerBands(df['Close'])
        df['Upper_BB'] = bb.bollinger_hband()
        df['Lower_BB'] = bb.bollinger_lband()
        stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['%K'] = stoch.stoch()
        df['%D'] = stoch.stoch_signal()

        # Drop NAs
        df.dropna(inplace=True)

        # Run backtest
        profit, trades = backtest_simple(df)

        # Store results
        results.append({
            "Ticker": ticker,
            "Profit": round(profit, 2),
            "Trades": len(trades) // 2
        })

        # Export signals
        df.to_csv(f"{OUTPUT_FOLDER}/{ticker}_signals.csv")
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# === EXPORT SUMMARY ===
if results:
    summary_df = pd.DataFrame(results)
    summary_df['Total Return (%)'] = (summary_df['Profit'] / summary_df['Profit'].abs().max()) * 100
    summary_df.to_csv(f"{OUTPUT_FOLDER}/backtest_results.csv", index=False)
    print(f"\nBacktest complete. Results saved to {OUTPUT_FOLDER}/backtest_results.csv")
else:
    print("No valid results. All tickers failed.")
