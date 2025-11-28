import jax.numpy as jnp
from jax import jit
import jax.lax as lax
from functools import partial
from typing import Tuple, Union

from xlb.operator import Operator
from xlb.operator.boundary_condition.boundary_condition import (
    BoundaryCondition,
    ImplementationStep,
)
from xlb.compute_backend import ComputeBackend
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy

class ConvectiveOutflowBC(BoundaryCondition):
    """
    Convective Outflow Boundary Condition.
    Allows vortices to exit without reflection by solving the convection equation:
    df/dt + U * df/dx = 0
    
    Approximated as:
    f_out(t+1) = (f_out(t) + U * f_neighbor(t+1)) / (1 + U)
    """

    def __init__(
        self,
        direction: Union[Tuple[int, int], Tuple[int, int, int]],
        u_conv: float,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_vertices=None,
    ):
        """
        Args:
            u_conv: Convective velocity (usually u_lattice).
            direction: Normal vector pointing INTO the domain.
                       2D Example: Right boundary -> (-1, 0)
                       3D Example: Right boundary -> (-1, 0, 0)
        """
        super().__init__(
            ImplementationStep.STREAMING,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
            mesh_vertices,
        )
        self.direction = direction
        self.u_conv = u_conv

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        # 1. Identify boundary nodes
        boundary = bc_mask == self.id
        
        # Broadcast boundary mask to (q, nx, ny) for 2D or (q, nx, ny, nz) for 3D
        new_shape = (self.velocity_set.q,) + boundary.shape[1:]
        boundary = lax.broadcast_in_dim(boundary, new_shape, tuple(range(self.velocity_set.d + 1)))
        
        # 2. Get Neighbor values (f_neighbor at t+1, which is f_post in this context)
        # Shift to grab interior neighbor.
        # 2D: For Right BC (-1, 0), neighbor is at x-1
        # 3D: For Right BC (-1, 0, 0), neighbor is at x-1
        shift = tuple(-d for d in self.direction)
        # Determine axes to roll: (1, 2) for 2D, (1, 2, 3) for 3D
        # velocity_set.d gives dimension. Axes are 1 to d.
        axes = tuple(range(1, self.velocity_set.d + 1))
        f_neighbor = jnp.roll(f_post, shift, axis=axes)
        
        # 3. Get Previous Boundary values (f_pre at t)
        # f_pre contains the distribution functions from the START of the step
        # At the boundary, this is f_out(t)
        f_boundary_prev = f_pre
        
        # 4. Apply Convective Formula
        # f_out_new = (f_prev_boundary + u_conv * f_neighbor) / (1 + u_conv)
        # Note: This simple form assumes dx=1, dt=1 in lattice units
        f_convective = (f_boundary_prev + self.u_conv * f_neighbor) / (1.0 + self.u_conv)
        
        # 5. Select: Apply CBC at boundary, keep original f_post elsewhere
        return jnp.where(boundary, f_convective, f_post)
