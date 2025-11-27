import numpy as np

print("Checking data validity...")

try:
    rho = np.load('drone_rho.npy')
    u = np.load('drone_u.npy')
    states = np.load('drone_states.npy')
    forces = np.load('drone_forces.npy')

    print(f"Rho shape: {rho.shape}")
    print(f"Rho min: {np.min(rho)}, Max: {np.max(rho)}")
    print(f"Rho NaNs: {np.isnan(rho).sum()}")
    
    print(f"U shape: {u.shape}")
    print(f"U min: {np.min(u)}, Max: {np.max(u)}")
    print(f"U NaNs: {np.isnan(u).sum()}")

    print(f"States shape: {states.shape}")
    print(f"States min: {np.min(states, axis=0)}")
    print(f"States max: {np.max(states, axis=0)}")
    
    print(f"Forces shape: {forces.shape}")
    print(f"Forces min: {np.min(forces, axis=0)}")
    print(f"Forces max: {np.max(forces, axis=0)}")

except Exception as e:
    print(f"Error loading data: {e}")
