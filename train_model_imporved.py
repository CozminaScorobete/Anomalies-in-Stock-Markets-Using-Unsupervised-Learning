import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
from validate import validate_model_on_synthetic

# Helper Functions
def load_and_combine_data(financial_data_path, synthetic_data_path):
    """
    Load and combine financial and synthetic datasets.
    """
    financial_data = pd.read_csv(financial_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)
    
    # Combine and shuffle datasets
    combined_data = np.vstack([financial_data.values, synthetic_data.values])
    np.random.shuffle(combined_data)
    return combined_data

def split_data(data, test_size=0.2):
    """
    Split data into training and testing sets.
    """
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, test_data

def build_autoencoder(input_dim, latent_dim=32):
    """
    Build an improved LSTM-based autoencoder model.
    """
    inputs = Input(shape=(input_dim, 1))
    # Encoder with increased layers
    encoded = LSTM(128, activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(64, activation='relu', return_sequences=True)(encoded)
    encoded = LSTM(latent_dim, activation='relu', return_sequences=False)(encoded)
    
    # Latent Representation
    latent = RepeatVector(input_dim)(encoded)
    
    # Decoder with increased layers
    decoded = LSTM(latent_dim, activation='relu', return_sequences=True)(latent)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(128, activation='relu', return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(1))(decoded)
    
    # Autoencoder model
    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder

def plot_loss(history):
    """
    Plot training and validation loss.
    """
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

def apply_smote(train_data):
    """
    Apply SMOTE to oversample anomalies.
    """
    print("Applying SMOTE for oversampling anomalies...")
    labels = np.array([1 if np.random.rand() < 0.2 else 0 for _ in range(len(train_data))])
    train_data_flat = train_data.reshape(train_data.shape[0], -1)  # Flatten for SMOTE
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    smote_data, _ = smote.fit_resample(train_data_flat, labels)
    return smote_data.reshape(smote_data.shape[0], train_data.shape[1], 1)

def smooth_predictions(predictions, window_size=5):
    """
    Smooth predictions using a moving average.
    """
    smoothed = np.convolve(predictions, np.ones(window_size), 'same') / window_size
    return (smoothed > 0.5).astype(int)

# Main Training Script
def main():
    # Paths to data
    financial_data_path = "synthetic_data/processed_data/financial_data_sequences.csv"
    synthetic_data_path = "synthetic_data/processed_data/synthetic_data_sequences_with_anomalies.csv"

    # Load and preprocess data
    print("Loading and combining data...")
    combined_data = load_and_combine_data(financial_data_path, synthetic_data_path)
    combined_data = combined_data.reshape(combined_data.shape[0], combined_data.shape[1], 1)  # Reshape for LSTM
    
    # Apply SMOTE
    smote_data = apply_smote(combined_data)
    
    # Split data
    train_data, test_data = split_data(smote_data)
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data.reshape(-1, train_data.shape[-1])).reshape(train_data.shape)
    test_data = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)

    # Add noise augmentation to training data
    noisy_train_data = train_data + np.random.normal(0, 0.02, train_data.shape)

    # Build the autoencoder
    print("Building the autoencoder model...")
    input_dim = train_data.shape[1]
    autoencoder = build_autoencoder(input_dim)

    # Train the autoencoder
    print("Training the autoencoder...")
    history = autoencoder.fit(
        noisy_train_data, train_data,
        epochs=10,
        batch_size=16,
        validation_data=(test_data, test_data),
        verbose=1
    )

    # Plot training loss
    plot_loss(history)

    # Save the model
    print("Saving the trained model...")
    model_save_path = "trained_models/autoencoder_model.h5"
    os.makedirs("trained_models", exist_ok=True)
    autoencoder.save(model_save_path)
    print(f"Model saved at {model_save_path}")

    # Load synthetic test data
    print("Evaluating the model...")
    autoencoder = load_model(model_save_path)
    test_sequences_file = "synthetic_data/processed_data/synthetic_data_sequences_with_anomalies.csv"
    ground_truth_file = "synthetic_data/lorenz_data_with_anomalies.csv"

    # Validate the model
    predicted_anomalies, errors = validate_model_on_synthetic(
        autoencoder, test_sequences_file, ground_truth_file
    )
    # Smooth predictions
    smoothed_predictions = smooth_predictions(predicted_anomalies)
    print("Final smoothed predictions generated.")

if __name__ == "__main__":
    main()
