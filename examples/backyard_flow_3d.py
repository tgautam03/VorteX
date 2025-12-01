import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from tqdm import tqdm
import xlb
import jax
import jax.numpy as jnp
import numpy as np
from xlb.velocity_set.d3q19 import D3Q19
from xlb.grid import grid_factory
from xlb.helper.nse_solver import create_nse_fields
from xlb.operator.boundary_condition import FullwayBounceBackBC
from xlb.operator.equilibrium.quadratic_equilibrium import QuadraticEquilibrium
from xlb.operator.collision.bgk import BGK
from xlb.operator.stream import Stream
from xlb.operator.macroscopic import Macroscopic

from vortex.obstacles import backyard_scene
from vortex.boundary import ConvectiveOutflowBC

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
    NX, NY, NZ = 500, 250, 250                   # number of points in x, y and z (units: lattice_units)
    DS = 0.01                                    # Equal grid spacing in x, y and z (units: meters)
    u_lattice = 0.1                             # Lattice velocity for LBM stability (units: lattice_units/s)
    u_real = 5                                   # Means: u_lattice cooresponds to u_real (units: m/s)
    dt = (u_lattice / u_real) * DS               # Time spacing (units: seconds)
    t = 10                                       # How long should the simulation run (units: seconds)
    NT = int(t / dt)                             # Number of time steps to run simulation for t seconds

    print(f"(NX, NY, NZ): ({NX},{NY},{NZ}), DS: {DS}, dt: {dt}, NT: {NT}")

    Re = 100                                    # Turbulence: Low (Re < 100), Medium (100 < Re < 1000), High (Re > 1000)
    L_c = 0.1*NX                                 # Characteristic length (units: lattice_units)     
    nu = u_lattice * (L_c) / Re                  # Fluid Kinematic Viscosity (units: m^2/s) 
    tau = 3 * nu + 0.5                           # Relaxation tau
    assert tau > 0.5, f"tau: {tau}"              # For stability (pushing it at 0.5, ideally would want atleast 0.7)
    omega = 1 / tau                              # Relaxation omega (for collision step)

    print(f"Re: {Re}, nu: {nu}, tau: {tau}, omega: {omega}")

    #############################################
    ####### Setting Simulation Environment ######
    #############################################
    # Backend and Precision
    compute_backend = xlb.ComputeBackend.JAX
    precision_policy = xlb.PrecisionPolicy.FP32FP32

    # Initialize velocity set
    velocity_set = D3Q19(precision_policy, compute_backend)

    # Initialize XLB
    xlb.init(velocity_set, compute_backend, precision_policy)

    # Create grid
    grid = grid_factory((NX, NY, NZ))

    print(f"Velocity set dimension: {velocity_set.d}")

    # Create fields
    grid, f_0, f_1, obstacle_mask, bc_mask = create_nse_fields(grid=grid)

    # Get obstacle mask
    backyard_mask = backyard_scene(NX, NY, NZ)
    obstacle_mask = obstacle_mask.at[:, backyard_mask].set(True) # Assign backyard mask to obstacle mask

    ########################################
    # Define and apply boundary conditions #
    ########################################
    # Flow enters from x=0 and leaves from everywhere else.
    # Inlet is at x=0.
    # Outlets are at x=NX-1, y=0, y=NY-1, z=0, z=NZ-1.

    # Outflow from right (x=NX-1)
    right_indices = [(NX - 1, y, z) for y in range(NY) for z in range(NZ)]
    right_outlet_bc = ConvectiveOutflowBC(direction=(-1, 0, 0), u_conv=u_lattice, indices=right_indices)       
    RIGHT_BC_ID = right_outlet_bc.id

    # Outflow from y=0 (Replacing old left outlet)
    y0_indices = [(x, 0, z) for x in range(NX) for z in range(NZ)]
    y0_outlet_bc = ConvectiveOutflowBC(direction=(0, 1, 0), u_conv=u_lattice, indices=y0_indices)
    Y0_BC_ID = y0_outlet_bc.id

    # Outflow from top (y=NY-1)
    top_indices = [(x, NY-1, z) for x in range(NX) for z in range(NZ)]
    top_outlet_bc = ConvectiveOutflowBC(direction=(0, -1, 0), u_conv=u_lattice, indices=top_indices)           
    TOP_BC_ID = top_outlet_bc.id

    # Outflow from bottom (z=0)
    # bottom_indices = [(x, y, 0) for x in range(NX) for y in range(NY)]
    # bottom_outlet_bc = ConvectiveOutflowBC(direction=(0, 0, 1), u_conv=u_lattice, indices=bottom_indices)     
    bottom_outlet_bc = FullwayBounceBackBC()
    BOTTOM_BC_ID = bottom_outlet_bc.id

    # Outflow from front (z=NZ-1)
    front_indices = [(x, y, NZ-1) for x in range(NX) for y in range(NY)]
    front_outlet_bc = ConvectiveOutflowBC(direction=(0, 0, -1), u_conv=u_lattice, indices=front_indices)
    FRONT_BC_ID = front_outlet_bc.id

    # Backyard as fixed obstacle
    backyard_bc = FullwayBounceBackBC()                                         
    BACKYARD_BC_ID = backyard_bc.id

    # Apply boundary conditions to the bc_mask
    # Note: create_nse_fields returns bc_mask with shape (1, NX, NY, NZ) or similar?
    # Let's check D2Q9 code: bc_mask.at[0,-1,:].set(RIGHT_BC_ID) -> (1, NX, NY)
    # So for 3D it should be (1, NX, NY, NZ)
    
    # x=NX-1
    bc_mask = bc_mask.at[0, -1, :, :].set(RIGHT_BC_ID)
    # x=0 (Inlet - No BC applied here, handled by manual fix)
    
    # y=0
    bc_mask = bc_mask.at[0, :, 0, :].set(Y0_BC_ID)
    # y=NY-1
    bc_mask = bc_mask.at[0, :, -1, :].set(TOP_BC_ID)
    # z=0
    bc_mask = bc_mask.at[0, :, :, 0].set(BOTTOM_BC_ID)
    # z=NZ-1
    bc_mask = bc_mask.at[0, :, :, -1].set(FRONT_BC_ID)
    
    # Backyard obstacle
    bc_mask = bc_mask.at[0, backyard_mask].set(BACKYARD_BC_ID)

    # List of all the boundary conditions (convert to tuple for JIT)
    bcs = tuple([right_outlet_bc, y0_outlet_bc, top_outlet_bc, bottom_outlet_bc, front_outlet_bc, backyard_bc])

    #############################################
    ############### Let's Simulate ##############
    #############################################
    eq_op = QuadraticEquilibrium()  # Equilibrium operator
    col_op = BGK()                  # Collision operator
    stream_op = Stream()            # Streaming operator
    macroscopic_op = Macroscopic()  # Opertor to get macroscopic fields

    # Initializing fields
    rho_init = jnp.ones((1, NX, NY, NZ))            # Initial Density = 1
    u_init = jnp.zeros((3, NX, NY, NZ))             
    # Flow enters from x=0 -> Flowing in +X direction
    u_init = u_init.at[0,:,:,:].set(u_lattice)    # X component of initial velocity
    u_init = u_init.at[1,:,:,:].set(0)
    u_init = u_init.at[2,:,:,:].set(0)

    # Equilibrium initialization
    f_0 = eq_op(rho_init, u_init)
    f_1 = jnp.zeros_like(f_0)
    
    print(f"f_0 shape: {f_0.shape}")

    #############################################
    # PRE-COMPUTE CONSTANTS FOR PERFORMANCE
    #############################################
    print("Pre-computing constants for JIT optimization...")
    
    # Pre-compute velocity set arrays (avoid recomputation in JIT)
    c_arr = jnp.array(velocity_set.c)
    w_arr = jnp.array(velocity_set.w)
    
    # Pre-compute regularization constants
    if c_arr.shape[0] == velocity_set.d:  # Case (3, 19)
        cx = c_arr[0, :].reshape((-1, 1, 1, 1))
        cy = c_arr[1, :].reshape((-1, 1, 1, 1))
        cz = c_arr[2, :].reshape((-1, 1, 1, 1))
    else:  # Case (19, 3) standard
        cx = c_arr[:, 0].reshape((-1, 1, 1, 1))
        cy = c_arr[:, 1].reshape((-1, 1, 1, 1))
        cz = c_arr[:, 2].reshape((-1, 1, 1, 1))
    
    w = w_arr.reshape((-1, 1, 1, 1))
    cs2 = 1.0 / 3.0
    
    # Pre-compute inlet equilibrium
    inlet_velocity = jnp.array([u_lattice, 0.0, 0.0])
    f_eq_inlet_base = eq_op(1.0, inlet_velocity)
    f_eq_inlet_tiled = jnp.tile(f_eq_inlet_base.reshape(-1, 1, 1), (1, NY, NZ))

    # One simulation step - OPTIMIZED
    @jax.jit
    def step(f_prev, bc_mask, obstacle_mask, cx, cy, cz, w, cs2, f_eq_inlet_tiled, omega):
        """
        Optimized LBM step with pre-computed constants.
        
        Performance improvements:
        - All constants pre-computed outside JIT
        - Single macroscopic calculation per step
        - Optimal memory reuse
        """
        
        # 0. INLET FIX (Pre-Streaming Stress Correction)
        # -----------------------------------------------------------
        # Calculate Neighbor Non-Equilibrium (at x=1)
        rho_n, u_n = macroscopic_op(f_prev[:, 1:2, :, :])
        f_eq_n = eq_op(rho_n, u_n)
        f_neq_n = f_prev[:, 1:2, :, :] - f_eq_n
        
        # Apply Fix: Inlet = Eq_Inlet + Neq_Neighbor (using pre-computed inlet equilibrium)
        f_prev = f_prev.at[:, 0, :, :].set(f_eq_inlet_tiled + f_neq_n[:, 0, :, :])
        # -----------------------------------------------------------

        # Get macroscopic fields (SINGLE CALCULATION - reused throughout)
        rho, u = macroscopic_op(f_prev)

        # Compute equilibrium distribution
        f_eq = eq_op(rho, u)

        # =========================================================
        # ===           REGULARIZATION STEP (FILTERING)         ===
        # =========================================================
        # Purpose: Remove high-frequency numerical noise ("ghost modes")
        # from f_prev before collision to prevent FP32 explosion.
        # NOW USING PRE-COMPUTED CONSTANTS (cx, cy, cz, w, cs2)
        
        # Compute Non-Equilibrium
        f_neq = f_prev - f_eq
        
        # Project f_neq onto 2nd-order Stress Tensor (Pi)
        pi_xx = jnp.sum(cx * cx * f_neq, axis=0)
        pi_yy = jnp.sum(cy * cy * f_neq, axis=0)
        pi_zz = jnp.sum(cz * cz * f_neq, axis=0)
        pi_xy = jnp.sum(cx * cy * f_neq, axis=0)
        pi_xz = jnp.sum(cx * cz * f_neq, axis=0)
        pi_yz = jnp.sum(cy * cz * f_neq, axis=0)
        
        # Reconstruct Clean f_neq from Stress Tensor
        # Pre-compute tensor products once
        cx_cx = cx * cx
        cy_cy = cy * cy
        cz_cz = cz * cz
        cx_cy_2 = 2.0 * cx * cy
        cx_cz_2 = 2.0 * cx * cz
        cy_cz_2 = 2.0 * cy * cz
        
        term_xx = (cx_cx - cs2) * pi_xx
        term_yy = (cy_cy - cs2) * pi_yy
        term_zz = (cz_cz - cs2) * pi_zz
        term_xy = cx_cy_2 * pi_xy
        term_xz = cx_cz_2 * pi_xz
        term_yz = cy_cz_2 * pi_yz
        
        f_neq_regularized = (w * 4.5) * (term_xx + term_yy + term_zz + term_xy + term_xz + term_yz)
        
        # Create the Cleaned Pre-Collision Distribution
        f_prev_clean = f_eq + f_neq_regularized
        # =========================================================

        # Perform collision
        f_post_col = col_op(f_prev_clean, f_eq, rho, u, omega)

        # Streaming
        f_streamed = stream_op(f_post_col)

        # Apply boundary conditions (loop unrolled by JIT)
        f_next = f_streamed
        for bc in bcs:
            f_next = bc(f_prev, f_next, bc_mask, obstacle_mask)
        
        return f_next, rho, u  # Return macroscopic fields to avoid recomputation

    
    print("Starting simulation...")
    print("Compiling JIT kernels (first iteration will be slow)...")
    
    desired_fps = 30 # Visualization FPS
    save_interval = max(1, int(1/(desired_fps * dt))) 
    
    # Create frames directory for saving data
    frames_dir = results_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Clear all existing .npy files in frames directory
    for npy_file in frames_dir.glob("*.npy"):
        npy_file.unlink()
    print(f"Cleared existing frames in {frames_dir}")

    f_prev = f_0
    
    frame_count = 0
    
    # Performance tracking
    import time
    compile_time = 0
    simulation_start = time.time()

    pbar = tqdm(range(NT), desc="Simulating", unit="step")
    for i in pbar:
        iter_start = time.time()
        
        # Run step and get macroscopic fields
        f_next, rho, u = step(f_prev, bc_mask, obstacle_mask, cx, cy, cz, w, cs2, f_eq_inlet_tiled, omega)
        
        # Block to ensure computation completes (for accurate timing)
        f_next = jax.block_until_ready(f_next)
        
        iter_time = time.time() - iter_start
        
        # Track compilation time (first iteration is slower due to JIT)
        if i == 0:
            compile_time = iter_time
            pbar.write(f"JIT compilation completed in {compile_time:.3f}s")
        
        f_prev = f_next

        if i % save_interval == 0:
            # Use already-computed macroscopic fields
            # Save frame directly to disk
            np.save(frames_dir / f'rho_{frame_count:05d}.npy', np.array(rho[0]))
            np.save(frames_dir / f'u_{frame_count:05d}.npy', np.array(u))
            np.save(frames_dir / f'step_{frame_count:05d}.npy', i)
            frame_count += 1
        
        # Update progress bar with metrics every 100 steps
        if i % 100 == 0:
            u_mag = jnp.sqrt(u[0]**2 + u[1]**2 + u[2]**2)
            mlups = (NX * NY * NZ) / (iter_time * 1e6)
            pbar.set_postfix({
                'Max_U': f'{jnp.max(u_mag):.4f}',
                'Min_Rho': f'{jnp.min(rho):.4f}',
                'Time_ms': f'{iter_time*1000:.1f}',
                'MLUPS': f'{mlups:.1f}'
            })

    total_time = time.time() - simulation_start
    avg_time_per_step = (total_time - compile_time) / (NT - 1) if NT > 1 else 0
    avg_mlups = (NX * NY * NZ) / (avg_time_per_step * 1e6) if avg_time_per_step > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total simulation time: {total_time:.2f}s")
    print(f"JIT compilation time: {compile_time:.3f}s")
    print(f"Average time per step: {avg_time_per_step*1000:.2f}ms")
    print(f"Average MLUPS: {avg_mlups:.2f}")
    print(f"Grid size: {NX}x{NY}x{NZ} = {NX*NY*NZ:,} lattice points")
    print(f"{'='*60}\n")
    
    print(f"Saved {frame_count} frames to {frames_dir}")
    print("To load frames: rho = np.load('frames/rho_00000.npy')")
    print("Data saved to disk")

if __name__ == "__main__":
    main()