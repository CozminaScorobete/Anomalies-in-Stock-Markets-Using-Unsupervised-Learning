import numpy as np
import matplotlib.pyplot as plt

def load_synthetic_data(file_path):
    """
    Load synthetic data from a CSV file.
    """
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)  # Skip header row
    return data

def compute_reconstruction_error(autoencoder, test_data):
    """
    Compute reconstruction errors for the test dataset.
    """
    reconstructed = autoencoder.predict(test_data)
    errors = np.mean(np.square(test_data - reconstructed), axis=(1, 2))  # Mean squared error per sequence
    return errors

def plot_reconstruction_errors(errors, threshold=None):
    """
    Plot reconstruction error distribution.
    """
    plt.hist(errors, bins=50, alpha=0.7, label='Reconstruction Errors')
    if threshold is not None:
        plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
    plt.legend()
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.show()

def validate_on_synthetic_data(autoencoder, test_data):
    """
    Validate the autoencoder on synthetic data by analyzing reconstruction errors.
    """
    print("Validating model on synthetic data...")
    
    # Compute reconstruction errors
    errors = compute_reconstruction_error(autoencoder, test_data)
    
    # Define a threshold
    threshold = np.mean(errors) + 2 * np.std(errors)  # Example threshold
    
    # Identify potential anomalies
    anomalies = errors > threshold
    num_anomalies = np.sum(anomalies)
    print(f"Threshold: {threshold}")
    print(f"Number of potential anomalies: {num_anomalies}")
    
    ## Plot reconstruction errors
    ##plot_reconstruction_errors(errors, threshold)

    return errors, anomalies
