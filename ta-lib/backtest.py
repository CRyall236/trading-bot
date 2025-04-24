import vectorbt as vbt
import numpy as np
import pandas as pd

# === CONFIG ===
rsi_oversold = 40
rsi_overbought = 60
ema_window = 20
window = 3  # Lookback window for scoring

# === Get tickers from Wikipedia ===
import yfinance as yf

def get_sp500_tickers():
    # Manually list or use a reliable source for the S&P 500 tickers
    sp500_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Add full list of tickers
    return sp500_tickers

def get_ftse100_tickers():
    # Manually list or use a reliable source for the FTSE 100 tickers
    ftse100_tickers = ['HSBC', 'AAPL.L', 'BHP', 'BP', 'RDSA']  # Add full list of tickers
    return ftse100_tickers

tickers = get_sp500_tickers() + get_ftse100_tickers()

# === Download Data ===
price = vbt.YFData.download(tickers, start='2023-10-01', end='2024-04-01').get('Close')

# === Indicators ===
rsi = vbt.RSI.run(price)
ema = vbt.EMA.run(price, window=ema_window)
macd = vbt.MACD.run(price)
bbands = vbt.BBANDS.run(price)
stoch = vbt.StochasticOscillator.run(price)

# === Scoring Function ===
def scoring_logic():
    scores = []
    for i in range(window, len(price)):
        recent_rsi = rsi.rsi.iloc[i - window:i]
        recent_macd = macd.macd.iloc[i - window:i]
        recent_signal = macd.signal.iloc[i - window:i]
        recent_close = price.iloc[i - window:i]
        recent_upper = bbands.upper.iloc[i - window:i]
        recent_lower = bbands.lower.iloc[i - window:i]
        recent_k = stoch.k.iloc[i - window:i]
        recent_d = stoch.d.iloc[i - window:i]

        rsi_score = np.where(recent_rsi < rsi_oversold, 1, np.where(recent_rsi > rsi_overbought, -1, 0)).sum()
        macd_score = np.where(recent_macd > recent_signal, 1, -1).sum()
        bb_score = np.where(recent_close < recent_lower, 1, np.where(recent_close > recent_upper, -1, 0)).sum()
        stoch_score = np.where((recent_k < 30) & (recent_d < 30), 1, np.where((recent_k > 70) & (recent_d > 70), -1, 0)).sum()

        score = (rsi_score * 1.2) + (macd_score * 1.5) + (bb_score * 1.2) + (stoch_score * 1.1)
        scores.append(score)

    scores = pd.DataFrame(scores, index=price.index[window:], columns=price.columns)
    return scores

# === Generate Signals ===
scores = scoring_logic()
entries = scores > 4
exits = scores < -4

# === Align entries and exits with original price DataFrame ===
entries = entries.reindex(price.index).fillna(False)
exits = exits.reindex(price.index).fillna(False)

# === Backtest ===
portfolio = vbt.Portfolio.from_signals(price, entries, exits, freq='1D')

# === Output ===
portfolio.stats().to_csv('vectorbt_backtest_stats.csv')
portfolio.plot().show()
