import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from imblearn.over_sampling import SMOTE

# Main function to generate, balance, and save data
def save_lorenz_with_anomalies_to_csv(output_dir="synthetic_data", filename="lorenz_data_with_anomalies.csv", anomaly_percentage=20, clustered=True):
    """
    Generate Lorenz data, inject anomalies, apply SMOTE to balance, and save to CSV.
    """
    t, states = generate_lorenz_data()
    
    # Inject random and clustered anomalies
    print("Injecting anomalies...")
    states_with_anomalies, labels = inject_combined_anomalies(states, anomaly_percentage, scale=10, clustered=clustered)

    # Prepare the data for SMOTE (flattened features)
    data = pd.DataFrame(states_with_anomalies, columns=["x", "y", "z"])
    data["anomaly"] = labels

    print("\nOriginal Class Distribution:")
    print(data["anomaly"].value_counts())

    # SMOTE handling with a dynamic strategy
    X = data[["x", "y", "z"]].values  # Features
    y = data["anomaly"].values       # Labels
    
    majority_count = np.sum(y == 0)
    minority_count = np.sum(y == 1)
    
    if minority_count / majority_count > 0.4:  # Avoid error when ratio is already high
        print("Warning: Anomaly class already close to the desired ratio. Skipping SMOTE.")
        X_resampled, y_resampled = X, y
    else:
        target_ratio = min(0.4, 2 * (minority_count / majority_count))  # Adjust dynamically
        print(f"Using dynamic SMOTE ratio: {target_ratio:.2f}")
        smote = SMOTE(sampling_strategy=target_ratio, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

    # Combine the resampled data into a DataFrame
    resampled_data = pd.DataFrame(X_resampled, columns=["x", "y", "z"])
    resampled_data["anomaly"] = y_resampled
    resampled_data["time"] = np.linspace(0, len(y_resampled) * 0.01, len(y_resampled))

    print("\nBalanced Class Distribution:")
    print(resampled_data["anomaly"].value_counts())

    # Save the balanced data
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    resampled_data.to_csv(file_path, index=False)
    print(f"\nSynthetic Lorenz data with balanced anomalies saved to {file_path}")

# Lorenz Attractor Parameters
def lorenz(t, state, sigma, beta, rho):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Generate synthetic chaotic data
def generate_lorenz_data(sigma=10, beta=8/3, rho=28, t_span=(0, 50), dt=0.01, initial_state=[1.0, 1.0, 1.0]):
    t = np.arange(t_span[0], t_span[1], dt)
    solution = solve_ivp(lorenz, t_span, initial_state, t_eval=t, args=(sigma, beta, rho))
    return t, solution.y.T

# Inject combined anomalies
def inject_combined_anomalies(states, anomaly_percentage=20, scale=10, clustered=True):
    """
    Inject a combination of random, clustered, and varied anomalies into the data.
    """
    num_points = states.shape[0]
    num_anomalies = int(num_points * (anomaly_percentage / 100))

    states_with_anomalies = np.copy(states)
    labels = np.zeros(num_points, dtype=int)

    # Step 1: Clustered anomalies
    if clustered:
        cluster_size = max(1, num_anomalies // 10)
        start_indices = np.random.choice(np.arange(num_points - cluster_size), size=10, replace=False)
        for start in start_indices:
            for i in range(cluster_size):
                idx = start + i
                states_with_anomalies[idx] += scale * np.random.randn(*states[idx].shape) * np.random.uniform(1, 2)
                labels[idx] = 1

    # Step 2: Random anomalies
    remaining_anomalies = num_anomalies - (10 * cluster_size if clustered else 0)
    anomaly_indices = np.random.choice(np.arange(num_points), size=remaining_anomalies, replace=False)
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(["spike", "offset", "flatline"])
        if anomaly_type == "spike":
            states_with_anomalies[idx] += scale * np.random.randn(*states[idx].shape)
        elif anomaly_type == "offset":
            states_with_anomalies[idx] += scale * np.random.uniform(0.5, 2.0)
        elif anomaly_type == "flatline":
            states_with_anomalies[idx] = np.zeros_like(states[idx])
        labels[idx] = 1

    # Step 3: Duplicate anomalies with slight variations
    anomaly_indices = np.where(labels == 1)[0]
    for idx in anomaly_indices:
        duplicate_idx = np.random.randint(0, num_points)
        states_with_anomalies[duplicate_idx] = states_with_anomalies[idx] * np.random.uniform(0.9, 1.1)
        labels[duplicate_idx] = 1

    return states_with_anomalies, labels

# Run the code with enhanced anomalies
save_lorenz_with_anomalies_to_csv(anomaly_percentage=5, clustered=True)
