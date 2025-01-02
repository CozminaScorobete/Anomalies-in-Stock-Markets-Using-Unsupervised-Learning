import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def van_der_pol(t, z, mu):
    """
    Van der Pol oscillator equations.
    :param t: Time variable (scalar)
    :param z: State vector [x, y] (array)
    :param mu: Parameter for nonlinearity and damping (scalar)
    :return: Derivatives [dx/dt, dy/dt] (array)
    """
    x, y = z
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]

def preprocess_synthetic_data(file_path, seq_length=15):
    """
    Preprocess the synthetic data (e.g., Van der Pol oscillator).
    - Normalizes 'x' and 'y' variables.
    - Creates sequences for time-series modeling.

    Parameters:
    - file_path: Path to the synthetic data CSV file.
    - seq_length: Length of each sequence for time-series data.

    Returns:
    - sequences: NumPy array of processed sequences.
    - scaler: MinMaxScaler object used for normalization.
    """
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Check that the dataset has 'x' and 'y' columns
    if not {'x', 'y'}.issubset(data.columns):
        raise ValueError(f"File {file_path} does not have 'x' and 'y' columns.")
    
    # Normalize the 'x' and 'y' variables using MinMaxScaler
    scaler = MinMaxScaler()
    data[['x', 'y']] = scaler.fit_transform(data[['x', 'y']])
    
    # Ensure that we have enough data for the specified sequence length
    if len(data) < seq_length:
        raise ValueError(f"Data has fewer than {seq_length} rows, cannot create sequences.")
    
    # Create sequences of length 'seq_length'
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[['x', 'y']].iloc[i:i + seq_length].values
        sequences.append(seq)
    
    # Convert to a NumPy array
    sequences = np.array(sequences)
    
    return sequences, scaler

def save_sequences(sequences, output_file):
    """
    Save the processed sequences to a CSV file for further use.
    - Flattens each sequence to a single row.
    - Saves all rows to the specified CSV file.

    Parameters:
    - sequences: NumPy array of sequences.
    - output_file: Path to the output CSV file.
    """
    # Reshape the sequences into a 2D array (one sequence per row)
    flattened_sequences = sequences.reshape(sequences.shape[0], -1)
    
    # Convert the NumPy array to a Pandas DataFrame for easy saving
    df = pd.DataFrame(flattened_sequences)
    
    # Save the DataFrame to a CSV file without the index
    df.to_csv(output_file, index=False)
    
    print(f"Processed data saved to {output_file}")

def main():
    """
    Main function to process synthetic data and save the sequences to a file.
    - Sets up paths for input and output files.
    - Preprocesses the data and saves the results.
    """
    # Path to the input synthetic data file
    synthetic_data_path = "synthetic_data/van_der_pol_data_with_noise.csv"
    
    # Directory where the processed data will be saved
    output_dir = "synthetic_data/processed_data"
    
    # Create the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Preprocess the synthetic data to generate sequences
    synthetic_sequences, synthetic_scaler = preprocess_synthetic_data(synthetic_data_path)
    
    # Save the processed sequences to a CSV file
    save_sequences(synthetic_sequences, os.path.join(output_dir, "van_der_pol_data_sequences.csv"))

# Execute the main function when the script is run
if __name__ == "__main__":
    main()