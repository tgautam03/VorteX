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

def backyard_scene(NX, NY, NZ):
    """
    Create a 3D boolean mask for a backyard scene.
    - Most objects (furniture, fences) are constrained to bottom 10% of NY.
    - Tall trees extend to 50% of NY.
    - Objects are horizontally smaller (s_xz reduced).
    
    Args:
        NX, NY, NZ: Grid dimensions (e.g., 400, 200, 200)
    
    Returns:
        Boolean array of shape (NX, NY, NZ)
    """
    # Coordinate arrays
    x_coords = jnp.arange(NX)
    y_coords = jnp.arange(NY)
    z_coords = jnp.arange(NZ)
    X, Y, Z = jnp.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    
    # Initialize empty mask
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
    
    # 1. Horizontal Scale (Reduced size)
    # Previously 4.0. Now 2.5 to make objects smaller in X/Z.
    s_xz = 2.5  
    
    # 2. Vertical Scale (Low Objects)
    # Constrain "low" objects (chairs, fence) to bottom 10%
    # Base reference height is 40 units, so we scale it to fit 0.1 * NY
    s_y_low = (0.1 * NY) / 40.0
    
    # 3. Vertical Scale (High Objects)
    # Constrain "tall" trees to reach 50% of domain height
    # Base reference height is 100 units, so we scale it to fit 0.5 * NY
    s_y_tall = (0.5 * NY) / 100.0

    
    # --- SECTION 1: TALL TREES (Up to 0.5 NY) ---
    
    # Tall Tree 1 (Back Left Corner)
    tx1, tz1 = 0.15 * NX, 0.15 * NZ
    h_tall1 = 80 * s_y_tall  # ~0.4 NY
    mask |= cylinder(tx1, tz1, 3 * s_xz, 0, h_tall1) # Trunk
    # Large canopy high up
    mask |= ellipsoid(tx1, h_tall1, tz1, 12 * s_xz, 15 * s_y_tall, 12 * s_xz)
    
    # Tall Tree 2 (Front Right Corner)
    tx2, tz2 = 0.85 * NX, 0.80 * NZ
    h_tall2 = 90 * s_y_tall # ~0.45 NY
    mask |= cylinder(tx2, tz2, 4 * s_xz, 0, h_tall2)
    mask |= ellipsoid(tx2, h_tall2, tz2, 15 * s_xz, 20 * s_y_tall, 15 * s_xz)

    
    # --- SECTION 2: LOW FURNITURE (Bottom 10% NY) ---
    cx, cz = 0.5 * NX, 0.5 * NZ # Center of domain
    
    # Table (Center)
    t_h = 12 * s_y_low
    t_r = 6 * s_xz
    mask |= cylinder(cx, cz, t_r, t_h, t_h + 2*s_y_low) # Top
    mask |= cylinder(cx, cz, 1.5 * s_xz, 0, t_h) # Leg
    
    # Chairs (Smaller footprint now)
    cd = 10 * s_xz # Closer to table
    seat_h = 8 * s_y_low
    seat_w = 2.5 * s_xz
    # Ensure leg radius is at least 2.0 to avoid sub-pixel LBM issues
    leg_r = max(0.8 * s_xz, 2.0) 
    
    chair_offsets = [(cd, 0), (-cd, 0), (0, cd), (0, -cd)]
    for dx, dz in chair_offsets:
        chx, chz = cx + dx, cz + dz
        # Seat
        mask |= box(chx-seat_w, chx+seat_w, seat_h-s_y_low, seat_h, chz-seat_w, chz+seat_w)
        # Backrest
        mask |= box(chx-seat_w, chx+seat_w, seat_h, seat_h+8*s_y_low, chz+seat_w-s_xz, chz+seat_w)
        # Legs
        mask |= cylinder(chx-seat_w+s_xz, chz-seat_w+s_xz, leg_r, 0, seat_h)
        mask |= cylinder(chx+seat_w-s_xz, chz-seat_w+s_xz, leg_r, 0, seat_h)
        mask |= cylinder(chx-seat_w+s_xz, chz+seat_w-s_xz, leg_r, 0, seat_h)
        mask |= cylinder(chx+seat_w-s_xz, chz+seat_w-s_xz, leg_r, 0, seat_h)


    # --- SECTION 3: LOW VEGETATION & STRUCTURES ---
    
    # Small Decorative Trees (Low)
    mask |= ellipsoid(0.3 * NX, 15*s_y_low, 0.7 * NZ, 6*s_xz, 6*s_y_low, 6*s_xz)
    mask |= cylinder(0.3 * NX, 0.7 * NZ, 1.5*s_xz, 0, 15*s_y_low)
    
    # Fence (Low barrier)
    fence_x = NX - (5 * s_xz)
    fence_h = 20 * s_y_low
    # Posts spaced out
    for z in range(int(0.05*NZ), int(0.95*NZ), int(10 * s_xz)):
        mask |= box(fence_x, fence_x + 2*s_xz, 0, fence_h, z, z + 2*s_xz)
    # Rails
    mask |= box(fence_x, fence_x + 1*s_xz, 8 * s_y_low, 10 * s_y_low, 0, NZ)
    mask |= box(fence_x, fence_x + 1*s_xz, 16 * s_y_low, 18 * s_y_low, 0, NZ)

    # Grill
    gx, gz = 0.2 * NX, 0.3 * NZ
    mask |= box(gx, gx+15*s_xz, 0, 12 * s_y_low, gz, gz+8*s_xz)

    return mask

