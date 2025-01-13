import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the ODE function
def ode_system(t, X):
    X1 = X[0] 
    X2 = X[1]  # Assuming X is a 2D system [X1, X2]
    nu1 = X[2]
    nu2 = X[3]

    # Define the right-hand side of the equation dX/dt
    dX1_dt = (-X1 + 10 * X2) 
    dX2_dt = X2 * (10 * np.exp(-X1**2 / 100) - X2) * (X2 - 1)
    dnu1_dt = nu1 + (1/5) * X1 * nu2 * (X2**2 - X2) * np.exp(-X1**2 / 100)
    dnu2_dt = -10 * nu1 + nu2 * (3 * (X2**2) - 2 * X2) - 10 * nu2 * (2 * X2 - 1) * np.exp(-X1**2 / 100)


    return [dX1_dt, dX2_dt, dnu1_dt, dnu2_dt]  

# Initial conditions
X0 = [5, 6, 3, 4]  # Initial conditions for X1, X2, X3 and X4
t_span = (0, 10)  # Time span for the solution
t_eval = np.linspace(0, 10, 300)  # Time points at which to store the computed values

sol = solve_ivp(ode_system, t_span, X0, method='RK45', t_eval=t_eval)

# Plot the results
#plt.plot(sol.t, sol.y[0], label='X1(t)')
#plt.plot(sol.t, sol.y[1], label='X2(t)', linestyle='--') 

print(sol.y[0])
print(sol.y[1])
print(sol.y[2])
print(sol.y[3])

plt.plot(sol.y[0], sol.y[1])
plt.plot(sol.y[2], sol.y[3])

plt.title("ODE System Solution")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid()
plt.show()