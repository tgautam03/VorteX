import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import xlb
import jax
import jax.numpy as jnp
import numpy as np
from xlb.velocity_set.d2q9 import D2Q9 
from xlb.grid import grid_factory
from xlb.helper.nse_solver import create_nse_fields
from xlb.operator.boundary_condition import EquilibriumBC, FullwayBounceBackBC, ZouHeBC
from xlb.operator.equilibrium.quadratic_equilibrium import QuadraticEquilibrium
from xlb.operator.collision.kbc import KBC
from xlb.operator.collision.bgk import BGK 
from xlb.operator.stream import Stream
from xlb.operator.macroscopic import Macroscopic

from vortex.obstacles import cylinder
from vortex.boundary import OpenBoundary, ConvectiveOutflowBC
from vortex.utils import create_sponge_omega

def main():
    #########################################
    # Folder where results should always go #
    #########################################
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "npy_files"
    results_dir.mkdir(exist_ok=True)

    ############################################
    # Setting Simulation Environment Variables #
    ############################################
    NX, NY = 800, 200                           # number of points in x and y (units: lattice_units)
    DS = 0.01                                    # Equal grid spacing in x and y (units: meters)
    u_lattice = 0.1                            # Lattice velocity for LBM stability (units: lattice_units/s)
    u_real = 5                                  # Means: u_lattice cooresponds to u_real (units: m/s)
    dt = (u_lattice / u_real) * DS              # Time spacing (units: seconds)
    t = 10                                       # How long should the simulation run (units: seconds)
    NT = int(t / dt)                            # Number of time steps to run simulation for t seconds
    cylinder_radius = 10                        # Radius of the cylinder obstacle (units: lattice_units)
    loc = (NX//8, NY//2)                        # Location of cylinder

    print(f"(NX, NY): ({NX},{NY}), DS: {DS}, dt: {dt}, NT: {NT}")

    Re = 200000                                   # Turbulence: Low (Re < 100), Medium (100 < Re < 1000), High (Re > 1000)
    nu = u_lattice * (2*cylinder_radius) / Re   # Fluid Kinematic Viscosity (units: m^2/s) 
    tau = 3 * nu + 0.5                          # Relaxation tau
    assert tau > 0.5, f"tau: {tau}"             # For stability (pushing it at 0.5, ideally would want atleast 0.7)
    omega = 1 / tau                             # Relaxation omega (for collision step)

    print(f"Re: {Re}, nu: {nu}, tau: {tau}, omega: {omega}")

    #############################################
    ####### Setting Simulation Environment ######
    #############################################
    # Backend and Precision
    compute_backend = xlb.ComputeBackend.JAX
    precision_policy = xlb.PrecisionPolicy.FP32FP32

    # Initialize velocity set
    velocity_set = D2Q9(precision_policy, compute_backend)

    # Initialize XLB
    xlb.init(velocity_set, compute_backend, precision_policy)

    # Create grid
    grid = grid_factory((NX, NY))

    # Create fields
    grid, f_0, f_1, obstacle_mask, bc_mask = create_nse_fields(grid=grid)

    # Get obstacle mask
    cylinder_mask = cylinder(NX, NY, cylinder_radius, loc)
    obstacle_mask = obstacle_mask.at[:, cylinder_mask].set(True) # Assign cylinder mask to obstacle mask

    # Define boundary conditions
    # left_inlet_bc = EquilibriumBC(rho=1, u=(u_lattice, 0))                      # Inflow from left
    # LEFT_BC_ID = left_inlet_bc.id

    right_indices = [(0, NX - 1, y) for y in range(NY)]
    # right_outlet_bc = OpenBoundary(direction=(-1, 0), indices=right_indices)    # Outflow from right
    right_outlet_bc = ConvectiveOutflowBC(direction=(-1, 0), u_conv=u_lattice, indices=right_indices)    # Outflow from right
    RIGHT_BC_ID = right_outlet_bc.id

    top_indices = [(0, x, NY-1) for x in range(NX)]
    # top_outlet_bc = OpenBoundary(direction=(0, -1), indices=top_indices)        # Outflow from top
    top_outlet_bc = ConvectiveOutflowBC(direction=(0, -1), u_conv=u_lattice, indices=top_indices)
    TOP_BC_ID = top_outlet_bc.id

    bottom_indices = [(0, x, 0) for x in range(NX)]
    # bottom_outlet_bc = OpenBoundary(direction=(0, 1), indices=bottom_indices)   # Outflow from bottom
    bottom_outlet_bc = ConvectiveOutflowBC(direction=(0, 1), u_conv=u_lattice, indices=bottom_indices)
    BOTTOM_BC_ID = bottom_outlet_bc.id

    cylinder_bc = FullwayBounceBackBC()                                         # Cylinder is fixed obstacle
    CYLINDER_BC_ID = cylinder_bc.id

    # Apply boundary conditions to the bc_mask
    # bc_mask = bc_mask.at[0,0,:].set(LEFT_BC_ID)
    bc_mask = bc_mask.at[0,-1,:].set(RIGHT_BC_ID)
    bc_mask = bc_mask.at[0,:,-1].set(TOP_BC_ID)
    bc_mask = bc_mask.at[0,:,0].set(BOTTOM_BC_ID)
    bc_mask = bc_mask.at[0, cylinder_mask].set(CYLINDER_BC_ID)

    # List of all the boundary conditions
    # bcs = [left_inlet_bc, right_outlet_bc, top_outlet_bc, bottom_outlet_bc, cylinder_bc]
    bcs = [right_outlet_bc, top_outlet_bc, bottom_outlet_bc, cylinder_bc]

    #############################################
    ############### Let's Simulate ##############
    #############################################
    eq_op = QuadraticEquilibrium()  # Equilibrium operator
    col_op = KBC()                  # Collision operator
    stream_op = Stream()            # Streaming operator
    macroscopic_op = Macroscopic()  # Opertor to get macroscopic fields

    # Initializing fields
    rho_init = jnp.ones((1, NX, NY))            # Initial Density = 1
    u_init = jnp.zeros((2, NX, NY))             
    u_init = u_init.at[0,:,:].set(u_lattice)    # X component of initial velocity is same as inlet velocity (flow is from left to right) 
    u_init = u_init.at[1,:,:].set(0)

    # Equilibrium initialization
    f_0 = eq_op(rho_init, u_init)
    f_1 = jnp.zeros_like(f_0)

        # One simulation step
    @jax.jit
    def step(f_prev, bc_mask, obstacle_mask):
        # 0. INLET FIX (Pre-Streaming Stress Correction)
        # -----------------------------------------------------------
        # Calculate Inlet Equilibrium 
        # eq_op(1.0, ...) returns shape (q,) i.e., (9,)
        f_eq_vec = eq_op(1.0, jnp.array([u_lattice, 0.0])) 
        
        # Reshape to (q, 1) and tile to (q, NY) -> (9, 800)
        f_eq_inlet_col = jnp.tile(f_eq_vec.reshape(-1, 1), (1, NY))
        
        # Calculate Neighbor Non-Equilibrium (at x=1)
        # f_prev[:, 1:2, :] is (q, 1, NY) -> e.g., (9, 1, 800)
        rho_n, u_n = macroscopic_op(f_prev[:, 1:2, :])
        f_eq_n = eq_op(rho_n, u_n) # (q, 1, NY)
        f_neq_n = f_prev[:, 1:2, :] - f_eq_n # (q, 1, NY)
        
        # Apply Fix: Inlet = Eq_Inlet + Neq_Neighbor
        # f_neq_n[:, 0, :] squeezes (9, 1, 800) to (9, 800)
        f_prev = f_prev.at[:, 0, :].set(f_eq_inlet_col + f_neq_n[:, 0, :])
        # -----------------------------------------------------------


        # Get macroscopic fields
        rho, u = macroscopic_op(f_prev)


        # Compute equilibrium distribution
        f_eq = eq_op(rho, u)


        # =========================================================
        # ===           REGULARIZATION STEP (FILTERING)         ===
        # =========================================================
        # Purpose: Remove high-frequency numerical noise ("ghost modes")
        # from f_prev before collision to prevent FP32 explosion.
        
        # A. Prepare geometry constants (safe for JAX jit)
        c_arr = jnp.array(velocity_set.c) # Shape (q, d) or (d, q)
        w_arr = jnp.array(velocity_set.w) # Shape (q,)
        
        # Handle Shape variants ((9,2) vs (2,9))
        if c_arr.shape[0] == velocity_set.d: # Case (2, 9)
            cx = c_arr[0, :].reshape((-1, 1, 1))
            cy = c_arr[1, :].reshape((-1, 1, 1))
        else: # Case (9, 2) standard
            cx = c_arr[:, 0].reshape((-1, 1, 1))
            cy = c_arr[:, 1].reshape((-1, 1, 1))
            
        w = w_arr.reshape((-1, 1, 1))
        cs2 = 1.0 / 3.0
        
        # B. Compute Non-Equilibrium
        f_neq = f_prev - f_eq
        
        # C. Project f_neq onto 2nd-order Stress Tensor (Pi)
        # This extracts the physical stress and discards higher-order noise
        # Summing over q index (axis 0)
        pi_xx = jnp.sum(cx * cx * f_neq, axis=0)
        pi_yy = jnp.sum(cy * cy * f_neq, axis=0)
        pi_xy = jnp.sum(cx * cy * f_neq, axis=0)
        
        # D. Reconstruct Clean f_neq from Stress Tensor only
        # Formula: f_neq_new = (w / (2*cs^4)) * Q : Pi
        # 1/(2*cs^4) = 1/(2 * 1/9) = 4.5
        
        # Q_xx term: (c_x^2 - cs^2) * pi_xx
        term_xx = (cx * cx - cs2) * pi_xx
        # Q_yy term: (c_y^2 - cs^2) * pi_yy
        term_yy = (cy * cy - cs2) * pi_yy
        # Q_xy term: 2 * c_x * c_y * pi_xy
        term_xy = (2.0 * cx * cy) * pi_xy
        
        f_neq_regularized = (w * 4.5) * (term_xx + term_yy + term_xy)
        
        # E. Create the Cleaned Pre-Collision Distribution
        f_prev_clean = f_eq + f_neq_regularized
        
        # =========================================================
        # ===           END REGULARIZATION                      ===
        # =========================================================


        # Perform collision
        f_post_col = col_op(f_prev_clean, f_eq, rho, u, omega)


        # Streaming
        f_streamed = stream_op(f_post_col)


        # Apply boundary conditions (one by one)
        f_next = f_streamed
        for bc in bcs:
            f_next = bc(f_prev, f_next, bc_mask, obstacle_mask)
        
        return f_next

    
    print("Starting simulation...")
    desired_fps = 30 # Visualization FPS
    save_interval = max(1, int(1/(desired_fps * dt))) 
    # Storage for visualization
    saved_steps = []
    saved_rho = []
    saved_u = []
    saved_vorticity = []

    f_prev = f_0
    f_next = f_1

    for i in range(NT):
        f_next = step(f_prev, bc_mask, obstacle_mask)
        f_prev = f_next

        if i % save_interval == 0:
            rho, u = macroscopic_op(f_prev)
            # Compute vorticity (curl of velocity in 2D)
            # vorticity = du_y/dx - du_x/dy
            du_y_dx = jnp.gradient(u[1], axis=0)
            du_x_dy = jnp.gradient(u[0], axis=1)
            vorticity = du_y_dx - du_x_dy
            
            saved_steps.append(i)
            saved_rho.append(np.array(rho[0]))
            saved_u.append(np.array(u))
            saved_vorticity.append(np.array(vorticity))
        
        if i % 100 == 0:
            rho, u = macroscopic_op(f_prev)
            u_mag = jnp.sqrt(u[0]**2 + u[1]**2)
            print(f"Step {i}: Max U = {jnp.max(u_mag):.4f}, Min Rho = {jnp.min(rho):.4f}")

    # Convert to numpy arrays for easier handling
    saved_rho = np.array(saved_rho)
    saved_u = np.array(saved_u)
    saved_vorticity = np.array(saved_vorticity)

    print(f"\nSaved {len(saved_steps)} frames for visualization")

    # Save data to files for visualization
    np.save(results_dir / 'cylinder_saved_steps.npy', np.array(saved_steps))
    np.save(results_dir / 'cylinder_saved_rho.npy', saved_rho)
    np.save(results_dir / 'cylinder_saved_u.npy', saved_u)
    np.save(results_dir / 'cylinder_saved_vorticity.npy', saved_vorticity)
    print("Data saved to .npy files")

if __name__ == "__main__":
    main()