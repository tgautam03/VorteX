import jax.numpy as jnp

def cylinder(NX, NY, cylinder_radius, loc):
    # Coordinate arrays
    x_coords = jnp.arange(NX)
    y_coords = jnp.arange(NY)
    X, Y = jnp.meshgrid(x_coords, y_coords, indexing="ij") # Shape (NX, NY)

    return (X - loc[0])**2 + (Y - loc[1])**2 <= cylinder_radius**2