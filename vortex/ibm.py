"""
Immersed Boundary Method (IBM) utilities for LBM simulations.

This module provides functions for interpolating velocities to markers
and spreading forces back to the grid.
"""

import jax.numpy as jnp


def kernel_2pt(r):
    """
    2-point hat kernel function for IBM interpolation/spreading.
    
    Args:
        r: Distance from grid point (can be array)
        
    Returns:
        Kernel weight (1 - |r| for |r| <= 1, else 0)
    """
    r_abs = jnp.abs(r)
    return jnp.where(r_abs <= 1.0, 1.0 - r_abs, 0.0)


def interpolate(u_grid, markers):
    """
    Interpolate grid velocity to marker positions using 2-point kernel.
    
    Args:
        u_grid: Velocity field on grid, shape (2, nx, ny)
        markers: Marker positions, shape (N, 2)
        
    Returns:
        Interpolated velocities at markers, shape (N, 2)
    """
    nx, ny = u_grid.shape[1], u_grid.shape[2]
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
            
            u_node = u_grid[:, idx_x, idx_y].T  # (N, 2)
            u_interp += u_node * weight[:, None]
            
    return u_interp


def spread(forces, markers, grid_shape):
    """
    Spread marker forces to the grid using 2-point kernel.
    
    Args:
        forces: Forces at markers, shape (N, 2)
        markers: Marker positions, shape (N, 2)
        grid_shape: Tuple (nx, ny)
        
    Returns:
        Force fields on grid: (fx_grid, fy_grid), each shape (nx, ny)
    """
    nx, ny = grid_shape
    fx_grid = jnp.zeros(grid_shape)
    fy_grid = jnp.zeros(grid_shape)
    
    x_m, y_m = markers[:, 0], markers[:, 1]
    i = jnp.floor(x_m).astype(int)
    j = jnp.floor(y_m).astype(int)
    
    for di in range(2):
        for dj in range(2):
            w_x = kernel_2pt(x_m - (i + di))
            w_y = kernel_2pt(y_m - (j + dj))
            weight = w_x * w_y  # (N,)
            
            idx_x = jnp.clip(i + di, 0, nx - 1)
            idx_y = jnp.clip(j + dj, 0, ny - 1)
            
            # Scatter add
            fx_grid = fx_grid.at[idx_x, idx_y].add(forces[:, 0] * weight)
            fy_grid = fy_grid.at[idx_x, idx_y].add(forces[:, 1] * weight)
            
    return fx_grid, fy_grid
