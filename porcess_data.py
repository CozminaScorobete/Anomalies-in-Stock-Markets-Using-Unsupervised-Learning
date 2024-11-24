import os  # For handling file paths and directory operations
import pandas as pd  # For data manipulation and handling CSV files
from sklearn.preprocessing import MinMaxScaler  # For normalizing data
import numpy as np  # For numerical operations and handling sequences

def preprocess_financial_data(file_path, seq_length=30):
    """
    Preprocess the financial data (e.g., stock prices).
    - Reads the data from a CSV file.
    - Removes unnecessary rows and selects specific columns.
    - Converts data to numeric and handles missing values.
    - Normalizes the 'Close' prices.
    - Creates sequences of time-series data for use in machine learning models.
    """
    # Read the CSV file without headers (since the file contains extra rows initially)
    data = pd.read_csv(file_path, header=None)
    
    # Skip the first two rows of the file, which don't contain meaningful data
    data = data.iloc[2:]
    
    # Rename the columns for easier referencing, based on the structure of your data
    data.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    
    # Select only the 'Date' and 'Close' columns, as 'Close' represents the primary variable of interest
    data = data[['Date', 'Close']]
    
    # Convert the 'Close' column to numeric, coercing invalid entries (like text) to NaN
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    
    # Drop any rows where 'Close' has missing or invalid values
    data = data.dropna(subset=['Close'])
    
    # Normalize the 'Close' prices to a range of 0 to 1, which is helpful for machine learning models
    scaler = MinMaxScaler()  # Create a scaler object
    data['Close'] = scaler.fit_transform(data[['Close']])  # Normalize the 'Close' column
    
    # Create sequences of the specified length from the normalized 'Close' prices
    # Each sequence will consist of `seq_length` consecutive values
    sequences = []
    for i in range(len(data) - seq_length):  # Iterate over the data to create sequences
        seq = data['Close'].iloc[i:i + seq_length].values  # Extract a sequence of `seq_length` values
        sequences.append(seq)  # Append the sequence to the list
    
    # Convert the list of sequences to a NumPy array for efficient numerical operations
    sequences = np.array(sequences)
    
    # Return the processed sequences and the scaler for future transformations
    return sequences, scaler
def preprocess_synthetic_data(file_path, seq_length=10):
    """
    Preprocess the synthetic data (e.g., Lorenz attractor).
    - Normalize 'x', 'y', and 'z' variables.
    - Create sequences for time-series modeling.
    """
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Check that the dataset has 'x', 'y', and 'z' columns
    if not {'x', 'y', 'z'}.issubset(data.columns):
        raise ValueError(f"File {file_path} does not have 'x', 'y', 'z' columns.")
    
    # Normalize the 'x', 'y', 'z' variables using MinMaxScaler
    scaler = MinMaxScaler()
    data[['x', 'y', 'z']] = scaler.fit_transform(data[['x', 'y', 'z']])
    
    # Ensure that we have enough data for the specified sequence length
    if len(data) < seq_length:
        raise ValueError(f"Data has fewer than {seq_length} rows, cannot create sequences.")
    
    # Create sequences of length 'seq_length' (30 in this case)
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[['x', 'y', 'z']].iloc[i:i + seq_length].values
        sequences.append(seq)
    
    sequences = np.array(sequences)
    
    # If there are more than 30 sequences and you need exactly 30, you can truncate
    # This ensures the number of sequences is equal to the target length
   
    
    return sequences, scaler
def save_sequences(sequences, output_file):
    """
    Save the processed sequences to a CSV file for further use.
    - Flattens each sequence to a single row.
    - Saves all rows to the specified CSV file.
    """
    # Reshape the sequences into a 2D array (one sequence per row)
    flattened_sequences = sequences.reshape(sequences.shape[0], -1)
    
    # Convert the NumPy array to a Pandas DataFrame for easy saving
    df = pd.DataFrame(flattened_sequences)
    
    # Save the DataFrame to a CSV file without the index
    df.to_csv(output_file, index=False)
    
    # Print a confirmation message with the file path
    print(f"Processed data saved to {output_file}")

def main():
    """
    Main function to process financial data and save the sequences to a file.
    - Sets up paths for input and output files.
    - Preprocesses the data and saves the results.
    """
    # Path to the input financial data file (adjust the path as needed)
    financial_data_path = "financial_data/AMZN_data.csv"
    
    # Directory where the processed data will be saved
    output_dir = "synthetic_data/processed_data"
    
    # Create the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Preprocess the financial data to generate sequences
    financial_sequences, financial_scaler = preprocess_financial_data(financial_data_path)
    
    # Save the processed sequences to a CSV file
    save_sequences(financial_sequences, os.path.join(output_dir, "financial_data_sequences.csv"))

    synthetic_sequences, synthetic_scaler = preprocess_synthetic_data("synthetic_data/lorenz_data.csv")
    save_sequences(synthetic_sequences, os.path.join(output_dir, "synthetic_data_sequences.csv"))
# Execute the main function when the script is run
if __name__ == "__main__":
    main()
