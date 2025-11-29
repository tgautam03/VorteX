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

def backyard_scene_scalable(NX, NY, NZ):
    """
    Create a 3D boolean mask for a backyard scene where objects scale to fit the grid.
    Ground is x-z plane at y = 0.
    
    Args:
        NX, NY, NZ: Grid dimensions (integers)
        
    Returns:
        Boolean array of shape (NX, NY, NZ)
    """
    import jnp
    
    # Coordinate arrays
    x_coords = jnp.arange(NX)
    y_coords = jnp.arange(NY)
    z_coords = jnp.arange(NZ)
    X, Y, Z = jnp.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    
    mask = jnp.zeros((NX, NY, NZ), dtype=bool)

    # --- Helper Functions ---
    def box(x_min, x_max, y_min, y_max, z_min, z_max):
        return (X >= x_min) & (X <= x_max) & \
               (Y >= y_min) & (Y <= y_max) & \
               (Z >= z_min) & (Z <= z_max)

    def cylinder(x_c, z_c, radius, y_min, y_max):
        return ((X - x_c)**2 + (Z - z_c)**2 <= radius**2) & \
               (Y >= y_min) & (Y <= y_max)

    def ellipsoid(x_c, y_c, z_c, rx, ry, rz):
        return ((X - x_c)**2 / rx**2 + (Y - y_c)**2 / ry**2 + \
                (Z - z_c)**2 / rz**2) <= 1

    # --- SCALING FACTORS ---
    # We define a "unit" relative to the smallest ground dimension to keep proportions
    # consistent (circular objects stay circular).
    scale = min(NX, NZ) / 100.0 
    
    # --- 1. CENTRAL FURNITURE ---
    # Table in the center
    cx, cz = 0.3 * NX, 0.3 * NZ  # Center of patio area
    table_h = 12 * scale
    table_r = 6 * scale
    
    # Tabletop
    mask |= cylinder(cx, cz, table_r, table_h, table_h + (2*scale))
    # Table leg (central pillar style)
    mask |= cylinder(cx, cz, 1.5 * scale, 0, table_h)
    
    # 4 Chairs around table
    chair_dist = 10 * scale
    seat_h = 6 * scale
    seat_size = 4 * scale
    
    offsets = [(chair_dist, 0), (-chair_dist, 0), (0, chair_dist), (0, -chair_dist)]
    for dx, dz in offsets:
        chair_x, chair_z = cx + dx, cz + dz
        # Seat
        mask |= box(chair_x - seat_size, chair_x + seat_size, 
                    seat_h - (1*scale), seat_h, 
                    chair_z - seat_size, chair_z + seat_size)
        # Backrest (simple box)
        mask |= box(chair_x - seat_size, chair_x + seat_size, 
                    seat_h, seat_h + (8*scale), 
                    chair_z + seat_size - (1*scale), chair_z + seat_size)
        # Leg (central)
        mask |= cylinder(chair_x, chair_z, 1*scale, 0, seat_h)

    # --- 2. VEGETATION ---
    # Large Tree (Back Right Corner)
    tx, tz = 0.8 * NX, 0.8 * NZ
    mask |= cylinder(tx, tz, 4 * scale, 0, 20 * scale) # Trunk
    mask |= ellipsoid(tx, 25 * scale, tz, 12 * scale, 10 * scale, 12 * scale) # Canopy
    
    # Small Tree (Front Right)
    tx2, tz2 = 0.8 * NX, 0.2 * NZ
    mask |= cylinder(tx2, tz2, 2.5 * scale, 0, 12 * scale)
    mask |= ellipsoid(tx2, 15 * scale, tz2, 8 * scale, 6 * scale, 8 * scale)

    # Hedges/Bushes along the left wall
    for i in range(5):
        bx, bz = 0.1 * NX, (0.1 + i * 0.15) * NZ
        mask |= ellipsoid(bx, 4 * scale, bz, 5 * scale, 4 * scale, 5 * scale)

    # --- 3. STRUCTURES ---
    # Fence (Back and Right edges)
    post_thick = 2 * scale
    rail_thick = 1 * scale
    fence_h = 15 * scale
    
    # Back fence (along X axis at max Z)
    z_fence = NZ - (5 * scale)
    mask |= box(0, NX, 0, NX, 5 * scale, 5 * scale + rail_thick, z_fence, z_fence + post_thick) # Bottom rail
    mask |= box(0, NX, 0, NX, 12 * scale, 12 * scale + rail_thick, z_fence, z_fence + post_thick) # Top rail
    
    # Vertical slats for fence
    num_slats = 20
    for i in range(num_slats):
        slat_x = i * (NX / num_slats)
        mask |= box(slat_x, slat_x + post_thick, 0, fence_h, z_fence, z_fence + post_thick)

    return mask
