"""
Environment2D - LBM fluid environment for drone simulations.

This module manages the LBM grid, boundary conditions, and fluid properties.
"""

import jax.numpy as jnp
import xlb
from xlb import ComputeBackend, PrecisionPolicy
from xlb.velocity_set.d2q9 import D2Q9
from xlb.grid import grid_factory
from xlb.operator.collision.kbc import KBC
from xlb.operator.equilibrium.quadratic_equilibrium import QuadraticEquilibrium
from xlb.operator.boundary_condition import ZouHeBC, FullwayBounceBackBC
from xlb.operator.stream import Stream
from xlb.operator.macroscopic import Macroscopic
from xlb.helper.nse_solver import create_nse_fields


class Environment2D:
    """
    Represents the LBM fluid environment for drone simulations.
    
    Manages grid, boundary conditions, operators, and fluid properties.
    """
    
    def __init__(self, nx: int, ny: int, precision_policy: str = "FP32FP32"):
        """
        Initialize the LBM environment.
        
        Args:
            nx: Number of grid points in x-direction
            ny: Number of grid points in y-direction
            precision_policy: Precision policy ("FP32FP32", "FP64FP32", etc.)
        """
        self.nx = nx
        self.ny = ny
        
        # Convert precision policy string to XLB enum
        precision_map = {
            "FP64FP64": PrecisionPolicy.FP64FP64,
            "FP32FP32": PrecisionPolicy.FP32FP32,
            "FP64FP32": PrecisionPolicy.FP64FP32,
            "FP64FP16": PrecisionPolicy.FP64FP16,
            "FP32FP16": PrecisionPolicy.FP32FP16,
        }
        if precision_policy not in precision_map:
            raise ValueError(f"Unsupported precision policy: {precision_policy}")
        
        self.precision_policy = precision_map[precision_policy]
        self.compute_backend = ComputeBackend.JAX
        
        # Initialize velocity set
        self.velocity_set = D2Q9(self.precision_policy, self.compute_backend)
        
        # Initialize XLB
        xlb.init(
            velocity_set=self.velocity_set,
            default_backend=self.compute_backend,
            default_precision_policy=self.precision_policy,
        )
        
        # Create grid
        grid = grid_factory((nx, ny))
        grid, f_0, f_1, missing_mask, bc_mask = create_nse_fields(grid=grid)
        
        self.grid = grid
        self.f_0 = f_0
        self.f_1 = f_1
        self.missing_mask = missing_mask
        self.bc_mask = bc_mask
        
        # LBM operators (initialized later)
        self.eq_op = QuadraticEquilibrium()
        self.collision_op = KBC()
        self.stream_op = Stream()
        self.macroscopic_op = Macroscopic()
        
        # Boundary conditions (set up in setup_boundaries)
        self.bc_ground = None
        self.bc_open = None
        self.bcs = []
        
        # Fluid properties (set in calculate_lbm_parameters)
        self.nu = None
        self.tau = None
        self.omega = None
        self.rho_ambient = 1.0
    
    def setup_boundaries(self):
        """
        Set up boundary conditions: ground (bottom) and open (top, left, right).
        """
        self.bc_ground = FullwayBounceBackBC()
        self.bc_open = ZouHeBC(bc_type="pressure", prescribed_values=self.rho_ambient)
        
        ID_GROUND = self.bc_ground.id
        ID_OPEN = self.bc_open.id
        
        # Apply masks
        self.bc_mask = self.bc_mask.at[0, :, 0].set(ID_GROUND)  # Bottom
        self.bc_mask = self.bc_mask.at[0, :, -1].set(ID_OPEN)  # Top
        self.bc_mask = self.bc_mask.at[0, 0, :].set(ID_OPEN)   # Left
        self.bc_mask = self.bc_mask.at[0, -1, :].set(ID_OPEN)  # Right
        
        self.bcs = [self.bc_ground, self.bc_open]
        
        print(f"registered bc {self.bc_ground.__class__.__name__}_{id(self.bc_ground)} with id {ID_GROUND}")
        print(f"registered bc {self.bc_open.__class__.__name__}_{id(self.bc_open)} with id {ID_OPEN}")
    
    def calculate_lbm_parameters(self, Re_target: float, L_char: float, u_char: float, 
                                   tau_min_stable: float = 0.5):
        """
        Calculate LBM parameters (nu, tau, omega) from target Reynolds number.
        
        Args:
            Re_target: Target Reynolds number
            L_char: Characteristic length (lattice units)
            u_char: Characteristic velocity (lattice units)
            tau_min_stable: Minimum stable tau value
        """
        # Calculate required viscosity
        nu_required = (u_char * L_char) / Re_target
        tau_result = 3.0 * nu_required + 0.5
        
        if tau_result < tau_min_stable:
            print(f"WARNING: Calculated tau ({tau_result:.4f}) is unstable. "
                  f"Setting tau to {tau_min_stable:.4f}.")
            tau = tau_min_stable
            nu = (tau - 0.5) / 3.0
            Re_effective = (u_char * L_char) / nu
        else:
            tau = tau_result
            nu = nu_required
            Re_effective = Re_target
        
        self.nu = nu
        self.tau = tau
        self.omega = 1.0 / tau
        
        print("--- LBM Parameters ---")
        print(f"Target Re: {Re_target}, Effective Re: {Re_effective:.2f}")
        print(f"nu: {nu:.6f}, tau: {tau:.4f}, omega: {self.omega:.4f}")
        
        return Re_effective
    
    def initialize_fluid(self):
        """
        Initialize fluid with equilibrium distribution.
        """
        rho_init = jnp.ones((1, self.nx, self.ny))
        u_init = jnp.zeros((2, self.nx, self.ny))
        self.f_0 = self.eq_op(rho_init, u_init)
        self.f_1 = jnp.zeros_like(self.f_0)