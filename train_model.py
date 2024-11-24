import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from validate import validate_on_synthetic_data

# Helper Functions
def load_and_combine_data(financial_data_path, synthetic_data_path):
    """
    Load and combine financial and synthetic datasets.
    """
    # Load datasets
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

def build_autoencoder(input_dim, latent_dim=16):
    """
    Build an LSTM-based autoencoder model.
    """
    # Encoder
    inputs = Input(shape=(input_dim, 1))
    encoded = LSTM(64, activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(latent_dim, activation='relu', return_sequences=False)(encoded)
    
    # Latent representation
    latent = RepeatVector(input_dim)(encoded)
    
    # Decoder
    decoded = LSTM(latent_dim, activation='relu', return_sequences=True)(latent)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
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

# Main Training Script
def main():
    # Paths to data
    financial_data_path = "synthetic_data/processed_data/financial_data_sequences.csv"
    synthetic_data_path = "synthetic_data/processed_data/synthetic_data_sequences.csv"
    
    # Load and preprocess data
    print("Loading and combining data...")
    combined_data = load_and_combine_data(financial_data_path, synthetic_data_path)
    combined_data = combined_data.reshape(combined_data.shape[0], combined_data.shape[1], 1)  # Reshape for LSTM
    
    # Split data into training and testing sets
    train_data, test_data = split_data(combined_data)
    
    # Build the autoencoder
    print("Building the autoencoder model...")
    input_dim = train_data.shape[1]  # Sequence length
    autoencoder = build_autoencoder(input_dim)
    
    # Train the autoencoder
    print("Training the autoencoder...")
    history = autoencoder.fit(
        train_data, train_data,
        epochs=5,
        batch_size=32,
        validation_data=(test_data, test_data),
        verbose=1
    )
    
    # Plot training loss
    print("Plotting training loss...")
    plot_loss(history)
    
    # Evaluate the model
    print("Evaluating the model...")
    reconstruction_loss = autoencoder.evaluate(test_data, test_data, verbose=0)
    print(f"Reconstruction Loss: {reconstruction_loss}")
    
    # Save the model
    print("Saving the trained model...")
    model_save_path = "trained_models/autoencoder_model.h5"
    os.makedirs("trained_models", exist_ok=True)
    autoencoder.save(model_save_path)
    print(f"Model saved at {model_save_path}")



# Validate the model
    validate_on_synthetic_data(autoencoder, test_data, test_labels)

if __name__ == "__main__":
    main()
