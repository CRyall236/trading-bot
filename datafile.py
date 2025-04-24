import yfinance as yf

# Define the stock ticker and the date range
ticker = "AAPL"  # You can replace this with any stock symbol (e.g., GOOGL, MSFT)
start_date = "2010-01-01"
end_date = "2023-01-01"

# Download the historical data
data = yf.download(ticker, start=start_date, end=end_date)

# Save the data to a CSV file
data.to_csv(f"{ticker}_historical_data.csv")

# Display the first few rows of the data
print(data.head())
