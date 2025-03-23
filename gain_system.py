import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the ODE system
def system(t, X):
    x1, x2 = X
    dx1_dt = -x1 + 10 * x2
    dx2_dt = x2 * (10 * np.exp(-x1**2 / 100) - x2) * (x2 - 1)
    return [dx1_dt, dx2_dt]

# Function to compute the gain G for given d and T
def compute_gain(d, T, theta):
    # Convert polar coordinates to Cartesian for initial conditions
    x0 = [d * np.cos(np.pi * theta), d * np.sin(np.pi * theta)]
    # Solve the system of ODEs
    sol = solve_ivp(system, [0, T], x0, t_eval=[T], method='DOP853')
    # Compute the norm of the solution at t = T
    xT = sol.y[:, -1]
    norm_xT = np.linalg.norm(xT)
    return norm_xT**2 / d**2

# Parameters
T = 2  # Time horizon
d_values = [1e-4, 0.9]  # Values of d
#d_values = [1.0001, 1.2]
theta_values = np.linspace(0, 2, 2000)  # Range of theta

# Compute gains
gains = {}
for d in d_values:
    gains[d] = [compute_gain(d, T, theta) for theta in theta_values]

# Plot the results
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for i, d in enumerate(d_values):
    axs[i].plot(theta_values, gains[d])
    axs[i].set_title(f"d = {d}")
    axs[i].set_xlabel(r"$\theta$")
    axs[i].set_ylabel("Gain (G)")
    axs[i].grid()

#plt.tight_layout()
plt.show()
