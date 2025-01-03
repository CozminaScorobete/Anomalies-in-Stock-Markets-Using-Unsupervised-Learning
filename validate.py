import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model


def validate_model_on_synthetic(autoencoder, test_sequences_file, ground_truth_file, threshold=None):
    """
    Validate the autoencoder model on synthetic sequences and compare predictions to ground truth labels.

    Parameters:
    - autoencoder: Trained autoencoder model.
    - test_sequences_file: Path to the CSV file with test sequences (preprocessed).
    - ground_truth_file: Path to the CSV file with ground truth labels.
    - threshold: Reconstruction error threshold; if None, dynamically calculated.

    Returns:
    - predicted_anomalies: Indices of detected anomalies.
    - errors: Reconstruction errors for each sequence.
    """
    # Load test sequences
    test_sequences = np.loadtxt(test_sequences_file, delimiter=",")
    test_sequences = test_sequences.reshape((test_sequences.shape[0], -1, 1))  # Reshape to (samples, timesteps, features)

    # Load ground truth labels
    ground_truth_data = pd.read_csv(ground_truth_file)
    labels = ground_truth_data["anomaly"].values

    # Ensure label alignment
    if len(test_sequences) != len(labels):
        labels = labels[-len(test_sequences):]  # Align labels with test sequences if needed

    # Reconstruct the sequences using the autoencoder
    reconstructed_data = autoencoder.predict(test_sequences)

    # Compute reconstruction errors
    errors = np.mean(np.square(test_sequences - reconstructed_data), axis=(1, 2))

    # Determine the threshold dynamically if not provided
    if threshold is None:
        threshold = np.percentile(errors, 90)
        print(f"Using dynamically calculated threshold: {threshold}")

    # Predict anomalies based on the threshold
    predicted_anomalies = (errors > threshold).astype(int)

    # Plot reconstruction errors
    plt.figure(figsize=(12, 6))
    plt.plot(errors, label="Reconstruction Errors", alpha=0.7)
    plt.scatter(np.where(labels == 1)[0], errors[labels == 1], color="red", label="True Anomalies")
    plt.axhline(threshold, color="green", linestyle="--", label=f"Threshold = {threshold}")
    plt.title("Reconstruction Errors with Ground Truth Anomalies")
    plt.xlabel("Data Point Index")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.show()

    # Classification report
    print("Classification Report:")
    print(classification_report(labels, predicted_anomalies, target_names=["Normal", "Anomaly"]))

    # Confusion matrix
    cm = confusion_matrix(labels, predicted_anomalies)
    print("Confusion Matrix:")
    print(cm)

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, errors)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guess")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    return predicted_anomalies, errors
