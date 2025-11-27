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
from jax.tree_util import register_pytree_node_class

# --- Configuration ---
# Thermal LBM Configuration
Re_target = 1000.0  # High turbulence but achievable
u_prop = 0.15 
L_char = 80 

# --- LBM Stability Constraint ---
tau_min_stable = 0.5 # Reduced slightly as we want some instability/turbulence
nu_required = (u_prop * L_char) / Re_target
tau_result = 3.0 * nu_required + 0.5

if tau_result < tau_min_stable:
    print(f"WARNING: Calculated tau ({tau_result:.4f}) is unstable. Setting tau to {tau_min_stable:.4f}.")
    tau = tau_min_stable
    nu = (tau - 0.5) / 3.0
    Re_effective = (u_prop * L_char) / nu
else:
    tau = tau_result
    nu = nu_required
    Re_effective = Re_target

omega = 1.0 / tau

print("--- LBM Parameters ---")
print(f"Target Re: {Re_target}, Effective Re: {Re_effective:.2f}")
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

# --- DRONE PHYSICS & GEOMETRY ---

@register_pytree_node_class
class DroneState:
    def __init__(self, pos, vel, angle, angular_vel):
        self.pos = pos       # [x, y]
        self.vel = vel       # [vx, vy]
        self.angle = angle   # radians
        self.angular_vel = angular_vel # radians/s

    def tree_flatten(self):
        return ((self.pos, self.vel, self.angle, self.angular_vel), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

# Drone Geometry Constants
BODY_RADIUS_X = 30.0
BODY_RADIUS_Y = 20.0
ARM_LENGTH = 180.0
ARM_THICKNESS = 10.0
MOTOR_OFFSET = 90.0
MOTOR_SIZE = 30.0
PROP_WIDTH = 50.0
PROP_OFFSET_Y = 10.0

# Mass Properties
DRONE_MASS = 2000.0 # 2 kg drone
DRONE_INERTIA = 1_000_000.0  # High agility
GRAVITY = -0.00005 

def get_drone_markers(state):
    """Generates Lagrangian markers for the drone surface based on state."""
    cx, cy = state.pos[0], state.pos[1]
    theta = state.angle
    
    # Rotation matrix
    c, s = jnp.cos(theta), jnp.sin(theta)
    rot = jnp.array([[c, -s], [s, c]])
    
    # 1. Body (Ellipse) - Discretized
    num_body_pts = 100
    t = jnp.linspace(0, 2*jnp.pi, num_body_pts, endpoint=False)
    body_x = BODY_RADIUS_X * jnp.cos(t)
    body_y = BODY_RADIUS_Y * jnp.sin(t)
    body_pts = jnp.stack([body_x, body_y], axis=1)
    
    # 2. Arms (Rectangle)
    # Simplified as a line of points for IBM
    num_arm_pts = 60
    arm_x = jnp.linspace(-ARM_LENGTH/2, ARM_LENGTH/2, num_arm_pts)
    arm_y = jnp.zeros_like(arm_x)
    arm_pts = jnp.stack([arm_x, arm_y], axis=1)
    
    # 3. Motors (Boxes)
    # Left Motor
    motor_pts_list = []
    for offset in [-MOTOR_OFFSET, MOTOR_OFFSET]:
        # Box around (offset, 5) size 30x30
        mx = jnp.array([-15, 15, 15, -15, -15]) + offset
        my = jnp.array([-15, -15, 15, 15, -15]) + 5 # +5 y offset
        # Interpolate points along edges
        for i in range(4):
            p1 = jnp.array([mx[i], my[i]])
            p2 = jnp.array([mx[i+1], my[i+1]])
            num_edge = 10
            alphas = jnp.linspace(0, 1, num_edge, endpoint=False)
            edge_pts = p1[None, :] * (1 - alphas[:, None]) + p2[None, :] * alphas[:, None]
            motor_pts_list.append(edge_pts)
            
    motor_pts = jnp.concatenate(motor_pts_list, axis=0)

    # Combine all local points
    all_local_pts = jnp.concatenate([body_pts, arm_pts, motor_pts], axis=0)
    
    # Rotate and Translate
    # (N, 2) @ (2, 2) -> (N, 2)
    rotated_pts = jnp.dot(all_local_pts, rot.T)
    global_pts = rotated_pts + jnp.array([cx, cy])
    
    return global_pts

# --- Control Options ---
USE_SMART_CONTROLLER = False # Set to False for "Dumb" drone with noise

def get_propeller_force_field(state, grid_shape, key):
    """Computes the actuator disk force field for the propellers."""
    cx, cy = state.pos[0], state.pos[1]
    theta = state.angle
    
    # Propeller positions relative to center
    # Left: (-90, 10), Right: (90, 10)
    
    # Base Thrust (reduced to enable descent)
    base_thrust = 0.00008
    
    if USE_SMART_CONTROLLER:
        # === SIMPLE PD CONTROLLER ===
        # Keep it simple: Proportional + Derivative control
        
        # Roll Stabilization
        target_angle = 0.0
        kp = 0.5    # Proportional gain
        kd = 5.0    # Derivative (damping) gain
        
        angle_error = target_angle - state.angle
        omega_error = 0.0 - state.angular_vel
        
        roll_correction = kp * angle_error + kd * omega_error
        
        # Altitude Control
        target_vy = -0.005
        kp_y = 0.001
        
        vy_error = target_vy - state.vel[1]
        thrust_correction = kp_y * vy_error
        
        # Motor mixing
        thrust_left = base_thrust + thrust_correction - roll_correction
        thrust_right = base_thrust + thrust_correction + roll_correction
        
        # Clamp thrust
        max_thrust = 0.001
        thrust_left = jnp.clip(thrust_left, 0.0, max_thrust)
        thrust_right = jnp.clip(thrust_right, 0.0, max_thrust)
        
    else:
        # --- DUMB DRONE (Random Noise) ---
        k1, k2 = jax.random.split(key)
        noise_left = jax.random.uniform(k1, minval=-0.1, maxval=0.1) 
        noise_right = jax.random.uniform(k2, minval=-0.1, maxval=0.1) 
        
        thrust_left = base_thrust * (1.0 + noise_left)
        thrust_right = base_thrust * (1.0 + noise_right)
    
    # Direction of thrust: Downward in body frame is (0, -1)
    # Rotated: (-sin(theta), -cos(theta))
    thrust_dir = jnp.array([-jnp.sin(theta), -jnp.cos(theta)])
    
    # Propeller Centers
    c, s = jnp.cos(theta), jnp.sin(theta)
    rot = jnp.array([[c, -s], [s, c]])
    
    p_left_local = jnp.array([-MOTOR_OFFSET, PROP_OFFSET_Y])
    p_right_local = jnp.array([MOTOR_OFFSET, PROP_OFFSET_Y])
    
    p_left = jnp.dot(rot, p_left_local) + jnp.array([cx, cy])
    p_right = jnp.dot(rot, p_right_local) + jnp.array([cx, cy])
    
    # Create a mask/kernel for the propellers on the grid
    # Simple Gaussian blobs for force distribution
    X, Y = jnp.meshgrid(jnp.arange(grid_shape[0]), jnp.arange(grid_shape[1]), indexing="ij")
    
    def gaussian_blob(x0, y0, sigma=10.0):
        return jnp.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    
    mask_left = gaussian_blob(p_left[0], p_left[1])
    mask_right = gaussian_blob(p_right[0], p_right[1])
    
    fx = (mask_left * thrust_left + mask_right * thrust_right) * thrust_dir[0]
    fy = (mask_left * thrust_left + mask_right * thrust_right) * thrust_dir[1]
    
    # Return total thrust forces for physics calc
    # Sum of mask is approx 2*pi*sigma^2 ? No, sum of gaussian on grid.
    # We can approximate the integral or just sum the grid.
    # For physics, we need the total force vector.
    
    return fx, fy


# --- IBM FUNCTIONS ---

def kernel_2pt(r):
    """2-point hat kernel function."""
    r_abs = jnp.abs(r)
    return jnp.where(r_abs <= 1.0, 1.0 - r_abs, 0.0)

def interpolate(u_grid, markers):
    """Interpolate grid velocity to marker positions."""
    # u_grid: (2, nx, ny)
    # markers: (N, 2)
    
    x_m, y_m = markers[:, 0], markers[:, 1]
    
    # Indices of bottom-left grid point
    i = jnp.floor(x_m).astype(int)
    j = jnp.floor(y_m).astype(int)
    
    u_interp = jnp.zeros((markers.shape[0], 2))
    
    # 2x2 stencil
    for di in range(2):
        for dj in range(2):
            w_x = kernel_2pt(x_m - (i + di))
            w_y = kernel_2pt(y_m - (j + dj))
            weight = w_x * w_y
            
            # Safe indexing (clamp to grid)
            idx_x = jnp.clip(i + di, 0, nx - 1)
            idx_y = jnp.clip(j + dj, 0, ny - 1)
            
            u_node = u_grid[:, idx_x, idx_y].T # (N, 2)
            u_interp += u_node * weight[:, None]
            
    return u_interp

def spread(forces, markers, grid_shape):
    """Spread marker forces to the grid."""
    # forces: (N, 2)
    # markers: (N, 2)
    
    fx_grid = jnp.zeros(grid_shape)
    fy_grid = jnp.zeros(grid_shape)
    
    x_m, y_m = markers[:, 0], markers[:, 1]
    i = jnp.floor(x_m).astype(int)
    j = jnp.floor(y_m).astype(int)
    
    for di in range(2):
        for dj in range(2):
            w_x = kernel_2pt(x_m - (i + di))
            w_y = kernel_2pt(y_m - (j + dj))
            weight = w_x * w_y # (N,)
            
            idx_x = jnp.clip(i + di, 0, nx - 1)
            idx_y = jnp.clip(j + dj, 0, ny - 1)
            
            # Scatter add
            fx_grid = fx_grid.at[idx_x, idx_y].add(forces[:, 0] * weight)
            fy_grid = fy_grid.at[idx_x, idx_y].add(forces[:, 1] * weight)
            
    return fx_grid, fy_grid

# --- Boundary Conditions ---
rho_ambient = 1.0
bc_ground = FullwayBounceBackBC()
bc_open = ZouHeBC(bc_type="pressure", prescribed_values=rho_ambient)
ID_GROUND = bc_ground.id
ID_OPEN = bc_open.id

# Apply Masks
bc_mask = bc_mask.at[0, :, 0].set(ID_GROUND)
bc_mask = bc_mask.at[0, :, -1].set(ID_OPEN) 
bc_mask = bc_mask.at[0, 0, :].set(ID_OPEN) 
bc_mask = bc_mask.at[0, -1, :].set(ID_OPEN)

bcs = [bc_ground, bc_open]

# --- Operators ---
eq_op = QuadraticEquilibrium()
collision_op = KBC() 
stream_op = Stream()
macroscopic_op = Macroscopic()

# --- Simulation Step ---

@jax.jit
def step(f_pre, f_post, bc_mask, missing_mask, drone_state, key):
    # 1. Macroscopic moments
    rho, u = macroscopic_op(f_pre)
    
    # --- IBM & PHYSICS START ---
    
    # A. Get Markers
    markers = get_drone_markers(drone_state)
    
    # B. Calculate Target Velocity at Markers (Rigid Body Kinematics)
    # v_target = v_cm + omega x r
    r_vec = markers - jnp.array(drone_state.pos) # (N, 2)
    # Cross product in 2D: omega * r_perp
    # r = (rx, ry) -> r_perp = (-ry, rx)
    v_rot_x = -drone_state.angular_vel * r_vec[:, 1]
    v_rot_y = drone_state.angular_vel * r_vec[:, 0]
    
    u_target_x = drone_state.vel[0] + v_rot_x
    u_target_y = drone_state.vel[1] + v_rot_y
    u_target = jnp.stack([u_target_x, u_target_y], axis=1)
    
    # C. Interpolate Fluid Velocity to Markers
    u_interp = interpolate(u, markers)
    
    # D. Compute IBM Force (Direct Forcing)
    # Force needed to bring fluid velocity to target velocity
    # F = (u_target - u_interp) / dt (dt=1)
    # Stiffer IBM: alpha = 0.5
    ibm_force_markers = (u_target - u_interp) * 0.5
    
    # E. Spread IBM Force to Grid
    fx_ibm, fy_ibm = spread(ibm_force_markers, markers, (nx, ny))
    
    # F. Add Propeller Thrust (Actuator Disk) with Noise
    fx_prop, fy_prop = get_propeller_force_field(drone_state, (nx, ny), key)
    
    # Total Fluid Force Field
    fx_total = fx_ibm + fx_prop
    fy_total = fy_ibm + fy_prop
    
    # G. Apply Force to Fluid (Source Term)
    # Simple velocity shift approach: u_new = u + F * tau
    # Or standard Guo forcing. For simplicity/stability in KBC, we modify velocity entering EQ.
    u_forced = u.at[0].add(fx_total)
    u_forced = u_forced.at[1].add(fy_total)
    
    # CLAMP VELOCITY FOR STABILITY
    u_forced = jnp.clip(u_forced, -0.3, 0.3)
    
    # H. Physics Update (Reaction Forces)
    # Force on Drone = - Force on Fluid (Newton's 3rd Law)
    # Sum forces from markers
    # Note: The force we calculated `ibm_force_markers` is Force density * Volume? 
    # In 2D IBM, it's often treated as force per unit length.
    # We sum the markers' reaction forces.
    
    # Reaction force from IBM (Drag/Lift from body)
    # F_reaction = - sum(ibm_force_markers)
    f_reaction = -jnp.sum(ibm_force_markers, axis=0) # [Fx, Fy]
    
    # Torque from IBM
    # Torque = r x F
    # r is r_vec
    # F is -ibm_force_markers
    # Cross product 2D: rx*Fy - ry*Fx
    f_markers = -ibm_force_markers
    torque_markers = r_vec[:, 0] * f_markers[:, 1] - r_vec[:, 1] * f_markers[:, 0]
    total_torque_fluid = jnp.sum(torque_markers)
    
    # Propeller Thrust Reaction (Pushing the drone UP)
    # The fluid gets pushed DOWN (fy_prop is negative), so drone gets pushed UP.
    # We calculated fx_prop, fy_prop on the grid.
    # Total thrust force = - sum(grid_prop_forces)
    thrust_x = -jnp.sum(fx_prop)
    thrust_y = -jnp.sum(fy_prop)
    
    # Gravity
    f_gravity = jnp.array([0.0, DRONE_MASS * GRAVITY])
    
    # Total Force on Drone
    total_force = f_reaction + jnp.array([thrust_x, thrust_y]) + f_gravity
    
    # Total Torque
    # Propeller torque? If props are balanced, net torque is 0 from thrust itself (unless differential).
    # For now, assume balanced thrust, so only fluid drag creates torque.
    total_torque = total_torque_fluid
    
    # CLAMP FORCES AND TORQUE
    total_force = jnp.clip(total_force, -0.05, 0.05)
    total_torque = jnp.clip(total_torque, -5000.0, 5000.0)
    
    # Update State (Euler Integration)
    # a = F/m
    accel = total_force / DRONE_MASS
    alpha = total_torque / DRONE_INERTIA
    
    new_vel = drone_state.vel + accel # dt=1
    new_pos = drone_state.pos + new_vel
    new_omega = drone_state.angular_vel + alpha
    new_angle = drone_state.angle + new_omega
    
    # Note: Removed angle saturation - not needed with high inertia
    
    # Ground Collision (Simple bounce)
    # If y < radius, bounce
    # Radius approx 20
    ground_y = 20.0
    
    # Soft constraint to keep it in bounds
    new_pos_y = jnp.maximum(new_pos[1], ground_y)
    
    # If hit ground, kill y velocity (inelastic)
    new_vel_y = jnp.where(new_pos[1] < ground_y, 0.0, new_vel[1])
    
    # Reconstruct vectors with clamped values
    new_pos = jnp.array([new_pos[0], new_pos_y])
    new_vel = jnp.array([new_vel[0], new_vel_y])
    
    # Clamp Drone Velocity to prevent explosions
    new_vel = jnp.clip(new_vel, -0.5, 0.5)
    
    new_drone_state = DroneState(new_pos, new_vel, new_angle, new_omega)
    
    # --- IBM & PHYSICS END ---

    # 2. Equilibrium (using the FORCED velocity)
    feq = eq_op(rho, u_forced)
    
    # 3. Collision
    f_out = collision_op(f_pre, feq, rho, u_forced, omega)
    
    # 4. Stream
    f_streamed = stream_op(f_out)
    
    # 5. Boundary Conditions
    f_curr = f_streamed
    for bc in bcs:
        f_curr = bc(f_streamed, f_curr, bc_mask, missing_mask)
    
    return f_curr, new_drone_state, total_force, total_torque

# --- Main Loop ---

# Initial State
rho_init = jnp.ones((1, nx, ny))
u_init = jnp.zeros((2, nx, ny))
f_0 = eq_op(rho_init, u_init)
f_1 = jnp.zeros_like(f_0)

# Initial Drone State
# Start high up
start_pos = jnp.array([nx/2, ny - 150.0])
start_vel = jnp.array([0.0, 0.0])
drone_state = DroneState(start_pos, start_vel, 0.0, 0.0)

num_steps = 50000 
save_interval = 50 # Save less frequently to save space

print("Starting simulation...")
import time
start_time = time.time()

saved_steps = []
saved_rho = []
saved_u = []
saved_vorticity = []
saved_drone_states = [] # [x, y, angle]
saved_forces = [] # [Fx, Fy, Torque]

current_f = f_0
next_f = f_1

# RNG Key
key = jax.random.PRNGKey(0)

for i in range(num_steps):
    key, subkey = jax.random.split(key)
    next_f, drone_state, force, torque = step(current_f, next_f, bc_mask, missing_mask, drone_state, subkey)
    current_f, next_f = next_f, current_f
    
    if i % save_interval == 0:
        rho, u = macroscopic_op(current_f)
        
        du_y_dx = jnp.gradient(u[1], axis=0)
        du_x_dy = jnp.gradient(u[0], axis=1)
        vorticity = du_y_dx - du_x_dy
        
        saved_rho.append(np.array(rho[0]))
        saved_u.append(np.array(u))
        saved_vorticity.append(np.array(vorticity))
        saved_steps.append(i)
        
        # Save drone state
        d_state = [float(drone_state.pos[0]), float(drone_state.pos[1]), float(drone_state.angle)]
        saved_drone_states.append(d_state)
        
        # Save forces
        f_data = [float(force[0]), float(force[1]), float(torque)]
        saved_forces.append(f_data)
    
    if i % 100 == 0:
        print(f"Step {i}: Drone X={drone_state.pos[0]:.2f}, Drone Y={drone_state.pos[1]:.2f}, Vy={drone_state.vel[1]:.4f}, Angle={drone_state.angle:.2f}")
        
        # Check for NaN
        if jnp.isnan(drone_state.pos[1]) or jnp.isnan(drone_state.vel[1]):
            print("ERROR: NaN detected in drone state. Stopping.")
            break

end_time = time.time()
print(f"Simulation finished in {end_time - start_time:.2f} seconds.")

# Save
np.save('drone_rho.npy', np.array(saved_rho))
np.save('drone_u.npy', np.array(saved_u))
np.save('drone_vorticity.npy', np.array(saved_vorticity))
np.save('drone_steps.npy', np.array(saved_steps))
np.save('drone_states.npy', np.array(saved_drone_states))
np.save('drone_forces.npy', np.array(saved_forces))
print("Data saved.")