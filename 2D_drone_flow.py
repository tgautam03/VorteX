import jax
import jax.numpy as jnp
import numpy as np
import xlb
from xlb import ComputeBackend, PrecisionPolicy
from xlb.velocity_set.d2q9 import D2Q9
from xlb.grid import grid_factory
from xlb.operator.collision.kbc import KBC
from xlb.operator.equilibrium.quadratic_equilibrium import QuadraticEquilibrium
from xlb.operator.boundary_condition import (
    ZouHeBC,
    FullwayBounceBackBC,
)
from xlb.operator.stream import Stream
from xlb.operator.macroscopic import Macroscopic
from xlb.helper.nse_solver import create_nse_fields

# --- Configuration ---
# Physical parameters
Re_target = 1000.0  
u_prop = 0.1 
L_char = 40 

# --- LBM Stability Constraint ---
# LBM requires tau > 0.5. For safety, we enforce a minimum tau.
tau_min_stable = 0.7 # Ensures stable omega = 1/0.7 = 1.4286

# 1. Calculate the required viscosity (nu) based on the target Reynolds number (Re_target)
nu_required = (u_prop * L_char) / Re_target

# 2. Check the relaxation time (tau) resulting from this viscosity
tau_result = 3.0 * nu_required + 0.5

# 3. Apply stability override if needed
if tau_result < tau_min_stable:
    print(f"WARNING: Calculated tau ({tau_result:.4f}) is unstable or too marginal. Setting tau to {tau_min_stable:.4f}.")
    tau = tau_min_stable
    
    # Calculate the new, stable nu
    nu = (tau - 0.5) / 3.0
    
    # Recalculate the effective Reynolds number (Re_effective) that the simulation must run at
    Re_effective = (u_prop * L_char) / nu
else:
    tau = tau_result
    nu = nu_required
    Re_effective = Re_target

omega = 1.0 / tau

print("--- LBM Parameters ---")
print(f"Target Re: {Re_target}, Effective Re: {Re_effective:.2f}")
print(f"u_prop: {u_prop}, L_char: {L_char}")
print(f"nu: {nu:.6f}, tau: {tau:.4f}, omega: {omega:.4f}")

# Grid parameters
nx, ny = 600, 600

# Backend and Precision
compute_backend = ComputeBackend.JAX
precision_policy = PrecisionPolicy.FP32FP32

# Initialize Velocity Set
velocity_set = D2Q9(precision_policy, compute_backend)

# Initialize XLB
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# --- Setup ---
grid = grid_factory((nx, ny))
grid, f_0, f_1, missing_mask, bc_mask = create_nse_fields(grid=grid)

# --- Geometry Definition ---
x_coords = jnp.arange(nx)
y_coords = jnp.arange(ny)
X, Y = jnp.meshgrid(x_coords, y_coords, indexing="ij")

drone_cx, drone_cy = nx // 2, ny - 150 

# 1. Drone Body (Solid)
body_radius_x, body_radius_y = 30, 20
mask_body = ((X - drone_cx)**2 / body_radius_x**2) + ((Y - drone_cy)**2 / body_radius_y**2) <= 1.0
mask_arms = (jnp.abs(X - drone_cx) < 180 // 2) & (jnp.abs(Y - drone_cy) < 10 // 2)

motor_offset = 90
mask_motors = (jnp.abs(X - (drone_cx - motor_offset)) < 15) | (jnp.abs(X - (drone_cx + motor_offset)) < 15)
mask_motors = mask_motors & (jnp.abs(Y - (drone_cy + 5)) < 15)

drone_solid_mask = mask_body | mask_arms | mask_motors

# 2. Propellers (Actuator Disks) - Internal forcing regions
# We make them slightly thicker (3 pixels) to ensure they catch the fluid effectively
prop_y = drone_cy - 10
mask_prop_left = (jnp.abs(X - (drone_cx - motor_offset)) < 25) & (jnp.abs(Y - prop_y) <= 1.5)
mask_prop_right = (jnp.abs(X - (drone_cx + motor_offset)) < 25) & (jnp.abs(Y - prop_y) <= 1.5)
propeller_mask = mask_prop_left | mask_prop_right


# --- Boundary Conditions ---
# We ONLY define external boundaries here. 
# Propellers are handled manually in the step function.

rho_ambient = 1.0

bc_ground = FullwayBounceBackBC()
bc_open = ZouHeBC(bc_type="pressure", prescribed_values=rho_ambient)
bc_drone_body = FullwayBounceBackBC()

ID_GROUND = bc_ground.id
ID_OPEN = bc_open.id
ID_BODY = bc_drone_body.id

print(f"BC IDs: Ground={ID_GROUND}, Open={ID_OPEN}, Body={ID_BODY}")

# --- Apply Masks ---
# 1. Ground (Bottom)
bc_mask = bc_mask.at[0, :, 0].set(ID_GROUND)

# 2. Open Air (Sides/Top)
bc_mask = bc_mask.at[0, :, -1].set(ID_OPEN) 
bc_mask = bc_mask.at[0, 0, :].set(ID_OPEN) 
bc_mask = bc_mask.at[0, -1, :].set(ID_OPEN)

# 3. Drone Body
bc_mask = bc_mask.at[0, drone_solid_mask].set(ID_BODY)
missing_mask = missing_mask.at[0, drone_solid_mask].set(True)

# Note: We do NOT set bc_mask for propellers. They remain "Fluid" nodes (ID 0).

bcs = [bc_ground, bc_open, bc_drone_body]

# --- Operators ---
eq_op = QuadraticEquilibrium()
collision_op = KBC() 
stream_op = Stream()
macroscopic_op = Macroscopic()

# --- Simulation Loop ---
rho_init = jnp.ones((1, nx, ny))
u_init = jnp.zeros((2, nx, ny))

f_0 = eq_op(rho_init, u_init)
f_1 = jnp.zeros_like(f_0)

# We define the target propeller velocity vector
prop_vel_x = 0.0
prop_vel_y = -u_prop # Downward thrust

# --- FIX START ---
# 1. Get the coordinates of the True values for the propeller mask 
#    using standard NumPy's where. This must be done OUTSIDE the JIT function.
prop_coords = np.where(propeller_mask) # Returns (row_indices, col_indices)

# 2. Convert the coordinates into JAX arrays once before the loop
prop_coords_x = jnp.array(prop_coords[0])
prop_coords_y = jnp.array(prop_coords[1])

# 3. JIT compile the step function WITHOUT static_argnums 
#    and with coordinates as dynamic arguments.
@jax.jit
def step(f_pre, f_post, bc_mask, missing_mask, prop_coords_x, prop_coords_y):
    # 1. Calculate Macroscopic moments
    rho, u = macroscopic_op(f_pre)
    
    # --- ACTUATOR DISK IMPLEMENTATION ---
    # Overwrite velocity at propeller locations using coordinates
    # prop_coords_x and prop_coords_y are dynamic JAX arrays
    u = u.at[0, prop_coords_x, prop_coords_y].set(prop_vel_x)
    u = u.at[1, prop_coords_x, prop_coords_y].set(prop_vel_y)
    # ------------------------------------

    # 2. Equilibrium (using the FORCED velocity)
    feq = eq_op(rho, u)
    
    # 3. Collision
    f_out = collision_op(f_pre, feq, rho, u, omega)
    
    # 4. Stream
    f_streamed = stream_op(f_out)
    
    # 5. Boundary Conditions
    f_curr = f_streamed
    for bc in bcs:
        f_curr = bc(f_streamed, f_curr, bc_mask, missing_mask)
    
    return f_curr

# Run
num_steps = 100000 
save_interval = 50
print("Starting simulation...")
import time
start_time = time.time()

# Storage for visualization
saved_steps = []
saved_rho = []
saved_u = []
saved_vorticity = []
# Visualization storage for max speed to ensure it's running
max_speeds = []

current_f = f_0
next_f = f_1

for i in range(num_steps):
    next_f = step(current_f, next_f, bc_mask, missing_mask, prop_coords_x, prop_coords_y) # <--- UPDATED CALL
    current_f, next_f = next_f, current_f
    
    if i % save_interval == 0:
        rho, u = macroscopic_op(current_f)
        
        # Calculate Vorticity
        du_y_dx = jnp.gradient(u[1], axis=0)
        du_x_dy = jnp.gradient(u[0], axis=1)
        vorticity = du_y_dx - du_x_dy
        
        saved_rho.append(np.array(rho[0]))
        saved_u.append(np.array(u))
        saved_vorticity.append(np.array(vorticity))
        saved_steps.append(i)
    
    if i % 100 == 0:
        rho, u = macroscopic_op(current_f)
        u_mag = jnp.sqrt(u[0]**2 + u[1]**2)
        curr_max = jnp.max(u_mag)
        max_speeds.append(curr_max)
        print(f"Step {i}: Max Speed = {curr_max:.4f}")

end_time = time.time()
print(f"Simulation finished in {end_time - start_time:.2f} seconds.")

if len(max_speeds) > 0 and max_speeds[-1] > 0.0:
    print("SUCCESS: Flow detected.")
else:
    print("WARNING: Flow is still zero.")

# Save
np.save('drone_rho.npy', np.array(saved_rho))
np.save('drone_u.npy', np.array(saved_u))
np.save('drone_vorticity.npy', np.array(saved_vorticity))
np.save('drone_steps.npy', np.array(saved_steps))
print("Data saved.")