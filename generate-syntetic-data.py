import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os

# Lorenz Attractor Parameters
def lorenz(t, state, sigma, beta, rho):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Generate synthetic chaotic data
def generate_lorenz_data(sigma=10, beta=8/3, rho=28, t_span=(0, 50), dt=0.01, initial_state=[1.0, 1.0, 1.0]):
    # Time points
    t = np.arange(t_span[0], t_span[1], dt)
    # Solve the Lorenz system
    solution = solve_ivp(
        lorenz, t_span, initial_state, t_eval=t, args=(sigma, beta, rho)
    )
    return t, solution.y.T  # Time array and state variables

# Main function to generate and save data
def save_lorenz_to_csv(output_dir="synthetic_data", filename="lorenz_data.csv"):
    # Generate Lorenz data
    t, states = generate_lorenz_data()
    # Create a DataFrame
    data = pd.DataFrame(states, columns=["x", "y", "z"])
    data["time"] = t  # Add time column
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Save to CSV
    file_path = os.path.join(output_dir, filename)
    data.to_csv(file_path, index=False)
    print(f"Synthetic Lorenz data saved to {file_path}")

# Run the code
save_lorenz_to_csv()
