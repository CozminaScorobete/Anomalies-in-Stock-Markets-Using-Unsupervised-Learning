import yfinance as yf
import matplotlib.pyplot as plt
import os

# Define the ticker symbols for the companies
ticker_symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]  # Example: Apple, Microsoft, Google, Amazon, Tesla

# Create an empty dictionary to store data for each company
data_dict = {}

# Create a directory to save CSV files
output_dir = "financial_data"
os.makedirs(output_dir, exist_ok=True)

# Download data for each company
for ticker in ticker_symbols:
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
    data_dict[ticker] = data  # Store the DataFrame in the dictionary

    # Save the data to a CSV file
    output_file = os.path.join(output_dir, f"{ticker}_data.csv")
    data.to_csv(output_file)
    print(f"Data for {ticker} saved to {output_file}.")

# Display the first few rows for each company
for ticker, data in data_dict.items():
    print(f"\nData for {ticker}:")
    print(data.head())

# Plot the closing prices for all companies
plt.figure(figsize=(12, 6))
for ticker, data in data_dict.items():
    plt.plot(data['Close'], label=f"{ticker} Closing Price")

plt.title("Closing Prices of Selected Companies (2020-2023)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()
