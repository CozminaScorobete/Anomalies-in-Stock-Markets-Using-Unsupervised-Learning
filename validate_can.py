import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model

def validate_model_on_synthetic_van(autoencoder, test_sequences_file, ground_truth_file, threshold=None, sequence_length=30):
    """
    Validate the autoencoder model on synthetic sequences and compare predictions to ground truth labels.

    Parameters:
    - autoencoder: Trained autoencoder model.
    - test_sequences_file: Path to the CSV file with test sequences (preprocessed).
    - ground_truth_file: Path to the CSV file with ground truth labels (1 for normal, -1 for anomaly).
    - threshold: Reconstruction error threshold; if None, dynamically calculated.
    - sequence_length: Length of sequences used during preprocessing.

    Returns:
    - predicted_anomalies: Indices of detected anomalies.
    - errors: Reconstruction errors for each sequence.
    """
    # Load test sequences
    test_sequences = np.loadtxt(test_sequences_file, delimiter=",")

    # Debugging: Log initial shape
    print(f"Loaded test sequences with shape: {test_sequences.shape}")

    # Ensure compatibility for reshaping
    total_elements = test_sequences.size
    num_sequences = total_elements // (sequence_length * 2)

    if num_sequences * sequence_length * 2 != total_elements:
        print(f"Trimming {total_elements % (sequence_length * 2)} excess elements to align data.")
        test_sequences = test_sequences[:num_sequences * sequence_length * 2]

    # Reshape to (samples, timesteps, features)
    test_sequences = test_sequences.reshape((num_sequences, sequence_length, 2))

    # Reduce features to a single dimension (e.g., combine x and y)
    test_sequences = np.mean(test_sequences, axis=2).reshape((num_sequences, sequence_length, 1))

    # Load ground truth labels
    ground_truth_data = pd.read_csv(ground_truth_file)
    labels = ground_truth_data["label"].values  # Column with ground truth labels

    # Align labels with test sequences
    if len(labels) > num_sequences:
        print(f"Trimming labels from {len(labels)} to {num_sequences} to match sequences.")
        labels = labels[:num_sequences]
    elif len(labels) < num_sequences:
        raise ValueError(f"Labels ({len(labels)}) are fewer than test sequences ({num_sequences}). Ensure alignment.")

    # Reconstruct the sequences using the autoencoder
    reconstructed_data = autoencoder.predict(test_sequences)

    # Compute reconstruction errors
    errors = np.mean(np.square(test_sequences - reconstructed_data), axis=(1, 2))

    # Determine the threshold dynamically if not provided
    if threshold is None:
        threshold = np.percentile(errors, 90)  # Dynamically set threshold at 90th percentile
        print(f"Using dynamically calculated threshold: {threshold}")

    # Predict anomalies based on the threshold
    predicted_anomalies = np.where(errors > threshold, -1, 1)  # -1 for anomaly, 1 for normal

    # Plot reconstruction errors
    plt.figure(figsize=(12, 6))
    plt.plot(errors, label="Reconstruction Errors", alpha=0.7)
    plt.scatter(np.where(labels == -1)[0], errors[labels == -1], color="red", label="True Anomalies")
    plt.axhline(threshold, color="green", linestyle="--", label=f"Threshold = {threshold}")
    plt.title("Reconstruction Errors with Ground Truth Anomalies")
    plt.xlabel("Data Point Index")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.show()

    # Classification report
    print("Classification Report:")
    print(classification_report(labels, predicted_anomalies, target_names=["Normal", "Anomaly"], zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(labels, predicted_anomalies)
    print("Confusion Matrix:")
    print(cm)

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, errors, pos_label=-1)
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
