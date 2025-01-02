from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from tensorflow.keras.models import load_model
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

def smooth_predictions(predictions, window_size=5):
    """
    Smooth predictions using a moving average filter.
    """
    smoothed = np.convolve(predictions, np.ones(window_size), 'same') / window_size
    return (smoothed > 0.5).astype(int)  # Re-binarize predictions


def apply_smote(data, labels):
    """
    Apply SMOTE to oversample anomaly data.
    """
    smote = SMOTE(sampling_strategy=0.5, random_state=42)  # 50% anomalies
    flattened_data = data.reshape(data.shape[0], -1)  # Flatten for SMOTE
    smote_data, smote_labels = smote.fit_resample(flattened_data, labels)
    smote_data = smote_data.reshape(smote_data.shape[0], data.shape[1], 1)  # Reshape back
    return smote_data, smote_labels


# Helper Functions
def load_and_combine_data(financial_data_path, synthetic_data_path):
    financial_data = pd.read_csv(financial_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)
    combined_data = np.vstack([financial_data.values, synthetic_data.values])
    np.random.shuffle(combined_data)
    return combined_data

def split_data(data, test_size=0.2):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, test_data

def build_autoencoder(input_dim, latent_dim=32):
    inputs = Input(shape=(input_dim, 1))
    # Encoder
    encoded = LSTM(128, activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(64, activation='relu', return_sequences=True)(encoded)
    encoded = LSTM(latent_dim, activation='relu', return_sequences=False)(encoded)
    # Latent Space
    latent = RepeatVector(input_dim)(encoded)
    # Decoder
    decoded = LSTM(latent_dim, activation='relu', return_sequences=True)(latent)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(128, activation='relu', return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(1))(decoded)
    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder

def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

def ensemble_anomaly_detection(autoencoder, data, threshold=None):
    reconstructed_data = autoencoder.predict(data)
    reconstruction_errors = np.mean(np.square(data - reconstructed_data), axis=(1, 2))

    if threshold is None:
        threshold = np.percentile(reconstruction_errors, 80)  # Default 80th percentile
        threshold = threshold * 0.9  # Lower threshold further by 10%
        print(f"Lowered threshold further: {threshold}")

    autoencoder_predictions = (reconstruction_errors > threshold).astype(int)
    reconstruction_errors = reconstruction_errors.reshape(-1, 1)

    # Isolation Forest on reconstruction errors
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    isolation_predictions = iso_forest.fit_predict(reconstruction_errors)
    isolation_predictions = np.where(isolation_predictions == -1, 1, 0)

    # One-Class SVM on reconstruction errors
    one_class_svm = OneClassSVM(nu=0.05, kernel="rbf", gamma="auto")
    svm_predictions = one_class_svm.fit_predict(reconstruction_errors)
    svm_predictions = np.where(svm_predictions == -1, 1, 0)

    # Weighted Voting
    ensemble_predictions = (1 * autoencoder_predictions + 3 * isolation_predictions + 3 * svm_predictions) >= 4

    ensemble_predictions = ensemble_predictions.astype(int)

    return ensemble_predictions, reconstruction_errors, autoencoder_predictions, isolation_predictions, svm_predictions

# Main Script
def main():
    financial_data_path = "synthetic_data/processed_data/financial_data_sequences.csv"
    synthetic_data_path = "synthetic_data/processed_data/synthetic_data_sequences_with_anomalies.csv"

    print("Loading and combining data...")
    combined_data = load_and_combine_data(financial_data_path, synthetic_data_path)
    combined_data = combined_data.reshape(combined_data.shape[0], combined_data.shape[1], 1)
    # Generate labels (assume anomalies are 1, normal is 0 for SMOTE)
    labels = np.array([1 if np.random.rand() < 0.2 else 0 for _ in range(len(combined_data))])
    combined_data, labels = apply_smote(combined_data, labels)


    train_data, test_data = split_data(combined_data)

    print("Building and training the autoencoder...")
    input_dim = train_data.shape[1]
    autoencoder = build_autoencoder(input_dim)
    # Add diverse noise during training for robustness
    gaussian_noise = np.random.normal(0, 0.02, train_data.shape)  # Increased Gaussian noise
    salt_pepper_noise = np.random.choice([0, 1], size=train_data.shape, p=[0.98, 0.02])  # 2% salt-and-pepper noise
    combined_noise = gaussian_noise + salt_pepper_noise

    noisy_train_data = train_data + combined_noise  # Combine noise types
  # Add Gaussian noise

    history = autoencoder.fit(
        noisy_train_data, train_data,
        epochs=20,  # Increased epochs
        batch_size=16,  # Smaller batch size
        validation_data=(test_data, test_data),
        verbose=1
    )

    plot_loss(history)

    model_save_path = "trained_models/autoencoder_model_forest.h5"
    os.makedirs("trained_models", exist_ok=True)
    autoencoder.save(model_save_path)
    print(f"Model saved at {model_save_path}")

    print("Evaluating with ensemble methods...")
    autoencoder = load_model(model_save_path)
    test_sequences_file = "synthetic_data/processed_data/synthetic_data_sequences_with_anomalies.csv"
    ground_truth_file = "synthetic_data/lorenz_data_with_anomalies.csv"

    test_data = pd.read_csv(test_sequences_file).values
    test_data = test_data.reshape(test_data.shape[0], -1, 1)
    ground_truth = pd.read_csv(ground_truth_file)["anomaly"].values
    ground_truth = ground_truth[-test_data.shape[0]:]

    ensemble_preds, errors, ae_preds, iso_preds, svm_preds = ensemble_anomaly_detection(autoencoder, test_data)

    print("\n--- Ensemble Anomaly Detection Results ---")
    print("Autoencoder Predictions:", np.sum(ae_preds), "anomalies detected")
    print("Isolation Forest Predictions:", np.sum(iso_preds), "anomalies detected")
    print("One-Class SVM Predictions:", np.sum(svm_preds), "anomalies detected")
    # Apply smoothing to ensemble predictions
    ensemble_preds_smoothed = smooth_predictions(ensemble_preds, window_size=5)
    print("Smoothed Ensemble Predictions:", np.sum(ensemble_preds_smoothed), "anomalies detected")


    print("\nClassification Report:")
    print(classification_report(ground_truth, ensemble_preds))

    print("Confusion Matrix:")
    print(confusion_matrix(ground_truth, ensemble_preds))

    # ROC Curve to Find Optimal Threshold
    fpr, tpr, thresholds = roc_curve(ground_truth, errors)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold from ROC Curve: {optimal_threshold}")

if __name__ == "__main__":
    main()
