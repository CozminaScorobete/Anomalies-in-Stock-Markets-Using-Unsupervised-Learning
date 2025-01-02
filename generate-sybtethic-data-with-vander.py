import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

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

# Parameters
mu = 1.0  # Nonlinearity parameter
time_span = (0, 100)  # Time range for integration
initial_conditions = [2.0, 0.0]  # Initial conditions [x0, y0]
t_eval = np.linspace(time_span[0], time_span[1], 5000)  # Time points to evaluate

# Solve the system
solution = solve_ivp(van_der_pol, time_span, initial_conditions, t_eval=t_eval, args=(mu,))

# Extract the solution
x, y = solution.y

# Add noise to the data
noise_level = 0.1  # Adjust noise level
x_noisy = x + noise_level * np.random.randn(len(x))
y_noisy = y + noise_level * np.random.randn(len(y))

# Create labels for anomalies (e.g., 1 for normal, -1 for anomalies)
anomalies = np.random.choice([1, -1], size=len(x), p=[0.95, 0.05])  # 5% anomalies

# Combine data into a DataFrame
data = pd.DataFrame({
    "time": solution.t,
    "x": x_noisy,
    "y": y_noisy,
    "label": anomalies
})

# Plot the results with noise
plt.figure(figsize=(10, 5))
plt.plot(solution.t, x_noisy, label="x (position with noise)")
plt.plot(solution.t, y_noisy, label="y (velocity with noise)")
plt.title("Van der Pol Oscillator with Noise")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Phase space plot with noise
plt.figure(figsize=(6, 6))
plt.plot(x_noisy, y_noisy, label="Noisy Phase Space Trajectory")
plt.title("Phase Space of Van der Pol Oscillator (Noisy)")
plt.xlabel("x (position)")
plt.ylabel("y (velocity)")
plt.legend()
plt.grid()
plt.show()

# Save the data to a CSV file
output_dir = "synthetic_data"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "van_der_pol_data_with_noise.csv")
data.to_csv(output_path, index=False)
print(f"Data with noise and labels saved to '{output_path}'")
