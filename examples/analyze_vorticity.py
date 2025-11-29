import numpy as np
from pathlib import Path

# Load the last frame
frames_dir = Path("examples/npy_files/frames")
last_frame_idx = 150

print(f"Loading frame {last_frame_idx}...")
rho = np.load(frames_dir / f'rho_{last_frame_idx:05d}.npy')
u = np.load(frames_dir / f'u_{last_frame_idx:05d}.npy')

print(f"Shape - rho: {rho.shape}, u: {u.shape}")

# Compute vorticity (curl of velocity field)
# omega_x = dw/dy - dv/dz
# omega_y = du/dz - dw/dx  
# omega_z = dv/dx - du/dy

ux = u[0]  # (NX, NY, NZ)
uy = u[1]
uz = u[2]

# Use central differences for derivatives (interior points)
omega_x = np.zeros_like(ux)
omega_y = np.zeros_like(ux)
omega_z = np.zeros_like(ux)

# omega_x = duz/dy - duy/dz
omega_x[:, 1:-1, 1:-1] = (uz[:, 2:, 1:-1] - uz[:, :-2, 1:-1]) / 2.0 - \
                          (uy[:, 1:-1, 2:] - uy[:, 1:-1, :-2]) / 2.0

# omega_y = dux/dz - duz/dx
omega_y[1:-1, :, 1:-1] = (ux[1:-1, :, 2:] - ux[1:-1, :, :-2]) / 2.0 - \
                          (uz[2:, :, 1:-1] - uz[:-2, :, 1:-1]) / 2.0

# omega_z = duy/dx - dux/dy
omega_z[1:-1, 1:-1, :] = (uy[2:, 1:-1, :] - uy[:-2, 1:-1, :]) / 2.0 - \
                          (ux[1:-1, 2:, :] - ux[1:-1, :-2, :]) / 2.0

# Sphere location and radius (from simulation)
NX, NY, NZ = 400, 200, 100
sphere_loc = (NX//4, NY//2, NZ//2)  # (100, 100, 50)
sphere_radius = 10

print(f"\nSphere location: {sphere_loc}, radius: {sphere_radius}")

# Define regions
# Front of sphere: x from sphere_loc[0] - 3*radius to sphere_loc[0] - radius
# Behind sphere: x from sphere_loc[0] + radius to sphere_loc[0] + 3*radius

front_x_start = max(0, sphere_loc[0] - 3*sphere_radius)
front_x_end = max(0, sphere_loc[0] - sphere_radius)
behind_x_start = min(NX, sphere_loc[0] + sphere_radius)
behind_x_end = min(NX, sphere_loc[0] + 3*sphere_radius)

# Use middle slice region around sphere (y and z centered)
y_start = max(0, sphere_loc[1] - 2*sphere_radius)
y_end = min(NY, sphere_loc[1] + 2*sphere_radius)
z_start = max(0, sphere_loc[2] - 2*sphere_radius)
z_end = min(NZ, sphere_loc[2] + 2*sphere_radius)

print(f"\nRegions analyzed:")
print(f"Front: x=[{front_x_start}:{front_x_end}], y=[{y_start}:{y_end}], z=[{z_start}:{z_end}]")
print(f"Behind: x=[{behind_x_start}:{behind_x_end}], y=[{y_start}:{y_end}], z=[{z_start}:{z_end}]")

# Extract vorticity in these regions
omega_x_front = omega_x[front_x_start:front_x_end, y_start:y_end, z_start:z_end]
omega_x_behind = omega_x[behind_x_start:behind_x_end, y_start:y_end, z_start:z_end]

# Compute statistics
print(f"\n{'='*70}")
print(f"VORTICITY (omega_x) ANALYSIS - Time Step {last_frame_idx}")
print(f"{'='*70}")

print(f"\n--- FRONT OF SPHERE (upstream) ---")
print(f"  Mean (absolute): {np.mean(np.abs(omega_x_front)):.6f}")
print(f"  Max (absolute):  {np.max(np.abs(omega_x_front)):.6f}")
print(f"  Std deviation:   {np.std(omega_x_front):.6f}")
print(f"  Min value:       {np.min(omega_x_front):.6f}")
print(f"  Max value:       {np.max(omega_x_front):.6f}")

print(f"\n--- BEHIND SPHERE (wake) ---")
print(f"  Mean (absolute): {np.mean(np.abs(omega_x_behind)):.6f}")
print(f"  Max (absolute):  {np.max(np.abs(omega_x_behind)):.6f}")
print(f"  Std deviation:   {np.std(omega_x_behind):.6f}")
print(f"  Min value:       {np.min(omega_x_behind):.6f}")
print(f"  Max value:       {np.max(omega_x_behind):.6f}")

# Calculate ratio
ratio_mean = np.mean(np.abs(omega_x_behind)) / (np.mean(np.abs(omega_x_front)) + 1e-10)
ratio_max = np.max(np.abs(omega_x_behind)) / (np.max(np.abs(omega_x_front)) + 1e-10)

print(f"\n--- COMPARISON ---")
print(f"  Wake/Front ratio (mean): {ratio_mean:.2f}x")
print(f"  Wake/Front ratio (max):  {ratio_max:.2f}x")

if ratio_mean > 10:
    print(f"\n✓ Wake vorticity is {ratio_mean:.1f}x stronger than frontal values")
    print(f"  → Frontal vortices are likely VISUALIZATION ARTIFACTS from color scaling")
elif ratio_mean > 3:
    print(f"\n⚠ Wake vorticity is only {ratio_mean:.1f}x stronger than frontal values")
    print(f"  → Frontal vortices may be weak numerical artifacts or real upstream disturbances")
else:
    print(f"\n✗ Wake and frontal vorticity are comparable ({ratio_mean:.1f}x)")
    print(f"  → This suggests numerical instability or strong upstream influence")

# Additional analysis: vorticity magnitude
omega_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
omega_mag_front = omega_mag[front_x_start:front_x_end, y_start:y_end, z_start:z_end]
omega_mag_behind = omega_mag[behind_x_start:behind_x_end, y_start:y_end, z_start:z_end]

print(f"\n--- TOTAL VORTICITY MAGNITUDE ---")
print(f"  Front mean:  {np.mean(omega_mag_front):.6f}")
print(f"  Behind mean: {np.mean(omega_mag_behind):.6f}")
print(f"  Ratio:       {np.mean(omega_mag_behind) / (np.mean(omega_mag_front) + 1e-10):.2f}x")

print(f"\n{'='*70}")

# Check global statistics
print(f"\n--- GLOBAL VORTICITY (omega_x) STATISTICS ---")
print(f"  Global min:  {np.min(omega_x):.6f}")
print(f"  Global max:  {np.max(omega_x):.6f}")
print(f"  Global mean (abs): {np.mean(np.abs(omega_x)):.6f}")
print(f"  95th percentile (abs): {np.percentile(np.abs(omega_x), 95):.6f}")
print(f"  99th percentile (abs): {np.percentile(np.abs(omega_x), 99):.6f}")
