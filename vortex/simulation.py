"""
DroneSimulation2D - Main simulation runner for drone landing simulations.

This module orchestrates the LBM fluid solver, IBM coupling, and drone physics.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np

from vortex.ibm import interpolate, spread


class DroneSimulation2D:
    """
    Main simulation runner for 2D drone landing simulations.
    
    Combines LBM fluid solver, Immersed Boundary Method, and drone physics.
    """
    
    def __init__(self, environment, drone, initial_state, units, save_interval=50, hover=True):
        """
        Initialize the simulation.
        
        Args:
            environment: Environment2D instance
            drone: Drone2D instance
            initial_state: DroneState2D instance
            units: UnitConverter instance
            save_interval: Save data every N steps
            hover: Whether to hover the drone at a fixed height or let it land
        """
        self.env = environment
        self.drone = drone
        self.state = initial_state
        self.units = units
        self.save_interval = save_interval
        self.hover = hover
        if hover:
            self.target_height = initial_state.pos[1]
        else:
            self.target_height = None
        
        # Storage for results
        self.saved_steps = []
        self.saved_rho = []
        self.saved_u = []
        self.saved_vorticity = []
        self.saved_drone_states = []
        self.saved_forces = []
        
        # Current distribution functions
        self.current_f = environment.f_0
        self.next_f = environment.f_1
        
        # RNG key for random noise
        self.key = jax.random.PRNGKey(0)
    
    @staticmethod
    def _step_jit(env, drone, f_pre, f_post, bc_mask, missing_mask, drone_state, key, nx, ny, hover, target_height):
        """
        Single simulation step (JIT compiled).
        
        This is separated as a static method to allow JAX JIT compilation.
        
        Args:
            hover: Whether to use hovering mode
            target_height: Target altitude for hovering (if hover=True)
        """
        # 1. Macroscopic moments
        rho, u = env.macroscopic_op(f_pre)
        
        # === IBM & PHYSICS START ===
        
        # A. Get markers
        markers = drone.get_markers(drone_state)
        
        # B. Calculate target velocity at markers (rigid body kinematics)
        # v_target = v_cm + omega x r
        r_vec = markers - jnp.array(drone_state.pos)  # (N, 2)
        # Cross product in 2D: omega * r_perp
        v_rot_x = -drone_state.angular_vel * r_vec[:, 1]
        v_rot_y = drone_state.angular_vel * r_vec[:, 0]
        
        u_target_x = drone_state.vel[0] + v_rot_x
        u_target_y = drone_state.vel[1] + v_rot_y
        u_target = jnp.stack([u_target_x, u_target_y], axis=1)
        
        # C. Interpolate fluid velocity to markers
        u_interp = interpolate(u, markers)
        
        # D. Compute IBM force (direct forcing)
        ibm_force_markers = (u_target - u_interp) * 0.5
        
        # E. Spread IBM force to grid
        fx_ibm, fy_ibm = spread(ibm_force_markers, markers, (nx, ny))
        
        # F. Add propeller thrust (actuator disk)
        fx_prop, fy_prop = drone.get_propeller_force_field(
            drone_state, (nx, ny), key, hover=hover, target_height=target_height
        )
        
        # Total fluid force field
        fx_total = fx_ibm + fx_prop
        fy_total = fy_ibm + fy_prop
        
        # G. Apply force to fluid
        u_forced = u.at[0].add(fx_total)
        u_forced = u_forced.at[1].add(fy_total)
        
        # Clamp velocity for stability
        u_forced = jnp.clip(u_forced, -0.3, 0.3)
        
        # H. Physics update (reaction forces)
        # Force on drone = - force on fluid (Newton's 3rd law)
        f_reaction = -jnp.sum(ibm_force_markers, axis=0)
        
        # Torque from IBM
        f_markers = -ibm_force_markers
        torque_markers = r_vec[:, 0] * f_markers[:, 1] - r_vec[:, 1] * f_markers[:, 0]
        total_torque_fluid = jnp.sum(torque_markers)
        
        # Propeller thrust reaction
        thrust_x = -jnp.sum(fx_prop)
        thrust_y = -jnp.sum(fy_prop)
        
        # Gravity
        f_gravity = jnp.array([0.0, drone.DRONE_MASS * drone.gravity])
        
        # Total force on drone
        total_force = f_reaction + jnp.array([thrust_x, thrust_y]) + f_gravity
        
        # Total torque
        total_torque = total_torque_fluid
        
        # Clamp forces and torque
        total_force = jnp.clip(total_force, -0.05, 0.05)
        total_torque = jnp.clip(total_torque, -5000.0, 5000.0)
        
        # Update state (Euler integration)
        accel = total_force / drone.DRONE_MASS
        alpha = total_torque / drone.DRONE_INERTIA
        
        new_vel = drone_state.vel + accel
        new_pos = drone_state.pos + new_vel
        new_omega = drone_state.angular_vel + alpha
        new_angle = drone_state.angle + new_omega
        
        # Ground collision (simple bounce)
        ground_y = 20.0
        new_pos_y = jnp.maximum(new_pos[1], ground_y)
        new_vel_y = jnp.where(new_pos[1] < ground_y, 0.0, new_vel[1])
        
        # Reconstruct vectors
        new_pos = jnp.array([new_pos[0], new_pos_y])
        new_vel = jnp.array([new_vel[0], new_vel_y])
        
        # Clamp drone velocity
        new_vel = jnp.clip(new_vel, -0.5, 0.5)
        
        from vortex.drone2d import DroneState2D
        new_drone_state = DroneState2D(new_pos, new_vel, new_angle, new_omega)
        
        # === IBM & PHYSICS END ===
        
        # 2. Equilibrium (using forced velocity)
        feq = env.eq_op(rho, u_forced)
        
        # 3. Collision
        f_out = env.collision_op(f_pre, feq, rho, u_forced, env.omega)
        
        # 4. Stream
        f_streamed = env.stream_op(f_out)
        
        # 5. Boundary conditions
        f_curr = f_streamed
        for bc in env.bcs:
            f_curr = bc(f_streamed, f_curr, bc_mask, missing_mask)
        
        return f_curr, new_drone_state, total_force, total_torque
    
    def run(self, num_steps: int):
        """
        Run the simulation for a specified number of steps.
        
        Args:
            num_steps: Number of timesteps to simulate
        """
        print("Starting simulation...")
        start_time = time.time()
        
        # JIT compile the step function
        hover = self.hover
        target_height = self.target_height if self.hover else None
        
        step = jax.jit(lambda f_pre, f_post, bc_mask, missing_mask, drone_state, key: 
                      self._step_jit(self.env, self.drone, f_pre, f_post, bc_mask, missing_mask,
                                     drone_state, key, self.env.nx, self.env.ny, hover, target_height))
        
        for i in range(num_steps):
            self.key, subkey = jax.random.split(self.key)
            self.next_f, self.state, force, torque = step(
                self.current_f, self.next_f, self.env.bc_mask, 
                self.env.missing_mask, self.state, subkey
            )
            self.current_f, self.next_f = self.next_f, self.current_f
            
            if i % self.save_interval == 0:
                rho, u = self.env.macroscopic_op(self.current_f)
                
                du_y_dx = jnp.gradient(u[1], axis=0)
                du_x_dy = jnp.gradient(u[0], axis=1)
                vorticity = du_y_dx - du_x_dy
                
                self.saved_rho.append(np.array(rho[0]))
                self.saved_u.append(np.array(u))
                self.saved_vorticity.append(np.array(vorticity))
                self.saved_steps.append(i)
                
                # Save drone state
                d_state = [float(self.state.pos[0]), float(self.state.pos[1]), float(self.state.angle)]
                self.saved_drone_states.append(d_state)
                
                # Save forces
                f_data = [float(force[0]), float(force[1]), float(torque)]
                self.saved_forces.append(f_data)
            
            if i % 100 == 0:
                print(f"Step {i}: Drone X={self.state.pos[0]:.2f}, Drone Y={self.state.pos[1]:.2f}, "
                      f"Vy={self.state.vel[1]:.4f}, Angle={self.state.angle:.2f}")
                
                # Check for NaN
                if jnp.isnan(self.state.pos[1]) or jnp.isnan(self.state.vel[1]):
                    print("ERROR: NaN detected in drone state. Stopping.")
                    break
        
        end_time = time.time()
        print(f"Simulation finished in {end_time - start_time:.2f} seconds.")
    
    def save_data(self, prefix='drone'):
        """
        Save simulation data to numpy files.
        
        Args:
            prefix: Prefix for output files
        """
        np.save(f'{prefix}_rho.npy', np.array(self.saved_rho))
        np.save(f'{prefix}_u.npy', np.array(self.saved_u))
        np.save(f'{prefix}_vorticity.npy', np.array(self.saved_vorticity))
        np.save(f'{prefix}_steps.npy', np.array(self.saved_steps))
        np.save(f'{prefix}_states.npy', np.array(self.saved_drone_states))
        np.save(f'{prefix}_forces.npy', np.array(self.saved_forces))
        print("Data saved.")
