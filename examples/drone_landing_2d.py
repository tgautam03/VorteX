"""
2D Drone Landing Simulation

A clean executable to run drone landing simulations using the vortex framework.
"""

import jax.numpy as jnp
from vortex.units import UnitConverter
from vortex.environment2d import Environment2D
from vortex.drone2d import Drone2D, DroneState2D
from vortex.simulation import DroneSimulation2D


def main():
    """Run the 2D drone landing simulation."""
    
    # ===== 1. SETUP UNITS =====
    dx = 0.01  # meters per lattice unit
    u_real = 5.0  # m/s (expected downwash velocity)
    u_lattice = 0.15  # lattice units per timestep (for LBM stability)
    
    units = UnitConverter(dx=dx, u_real=u_real, u_lattice=u_lattice)
    print(units)
    print()
    
    # ===== 2. CREATE ENVIRONMENT =====
    nx,ny = 600, 800
    env = Environment2D(nx=nx, ny=ny, precision_policy="FP32FP32")
    
    # Setup boundary conditions
    env.setup_boundaries()
    
    # Calculate LBM parameters
    Re_target = 1000.0  # High turbulence
    prop_width_lattice = 50.0  # Characteristic length
    env.calculate_lbm_parameters(Re_target=Re_target, L_char=prop_width_lattice, 
                                   u_char=u_lattice, tau_min_stable=0.5)
    
    # Initialize fluid
    env.initialize_fluid()
    print()
    
    # ===== 3. CREATE DRONE =====
    g_SI = -9.81  # m/s²
    gravity_lattice = units.gravity_to_lattice(g_SI)
    
    drone = Drone2D(gravity=gravity_lattice)
    
    # Initial state: start high up
    start_pos = jnp.array([nx/2, ny - 150.0])
    start_vel = jnp.array([0.0, 0.0])
    initial_state = DroneState2D(start_pos, start_vel, 0.0, 0.0)
    
    # ===== 4. RUN SIMULATION =====
    sim = DroneSimulation2D(
        environment=env,
        drone=drone,
        initial_state=initial_state,
        units=units,
        save_interval=50
    )
    
    num_steps = 10000
    sim.run(num_steps=num_steps)
    
    # ===== 5. SAVE DATA =====
    sim.save_data(prefix='drone')


if __name__ == "__main__":
    main()
