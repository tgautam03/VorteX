import jax.numpy as jnp

def circle(NX, NY, cylinder_radius, loc):
    # Coordinate arrays
    x_coords = jnp.arange(NX)
    y_coords = jnp.arange(NY)
    X, Y = jnp.meshgrid(x_coords, y_coords, indexing="ij") # Shape (NX, NY)

    return (X - loc[0])**2 + (Y - loc[1])**2 <= cylinder_radius**2

def sphere(NX, NY, NZ, radius, loc):
    # Coordinate arrays
    x_coords = jnp.arange(NX)
    y_coords = jnp.arange(NY)
    z_coords = jnp.arange(NZ)
    X, Y, Z = jnp.meshgrid(x_coords, y_coords, z_coords, indexing="ij") # Shape (NX, NY, NZ)

    return (X - loc[0])**2 + (Y - loc[1])**2 + (Z - loc[2])**2 <= radius**2