import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


class Drone2D:
    """
    Represents a 2D drone model with defined physical and geometric properties.
    This class is intended to encapsulate the drone's structure, mass, and inertia
    for simulation or control purposes.
    """
    def __init__(self, gravity, body_width=30.0, body_height=20.0, 
                arm_length=180.0, arm_thickness=10.0, 
                motor_offset=90.0, motor_size=30.0, 
                prop_width=50.0, prop_height_above_body_center=10.0,
                mass=2000.0, inertia=1_000_000.0):
        """
        Drone geometry parameters:
        gravity: gravity acceleration (negative)
        body_width: width of the body (x-direction)
        body_height: height of the body (y-direction)
        arm_length: length of the arms
        arm_thickness: thickness of the arms
        motor_offset: offset of the motors from the center of the body
        motor_size: size of the motors
        prop_width: width of the propellers
        prop_height_above_body_center: height of the propellers above the body center
        mass: mass of the drone
        inertia: inertia of the drone
        """
        self.BODY_RADIUS_X = body_width
        self.BODY_RADIUS_Y = body_height
        self.ARM_LENGTH = arm_length
        self.ARM_THICKNESS = arm_thickness
        self.MOTOR_OFFSET = motor_offset
        self.MOTOR_SIZE = motor_size
        self.PROP_WIDTH = prop_width
        self.PROP_OFFSET_Y = prop_height_above_body_center
        self.DRONE_MASS = mass
        self.DRONE_INERTIA = inertia
        self.gravity = gravity  # Store gravity for use in simulation
        
        # Calculate Base Thrust to hover
        # Total Force = Mass * |Gravity|
        # Force per Propeller = (Mass * |Gravity|) / 2
        # Force Field Integral = Thrust * 2 * pi * sigma^2
        # So: Thrust = (Mass * |Gravity| / 2) / (2 * pi * sigma^2)
        # Thrust = Mass * |Gravity| / (4 * pi * sigma^2)
        self.PROP_SIGMA = 0.2 * prop_width
        self.BASE_THRUST = (mass * abs(gravity)) / (4 * jnp.pi * self.PROP_SIGMA**2)

    def get_markers(self, state):
        """Generates Lagrangian markers for the drone surface based on state."""
        cx, cy = state.pos[0], state.pos[1]
        theta = state.angle
        
        # Rotation matrix
        c, s = jnp.cos(theta), jnp.sin(theta)
        rot = jnp.array([[c, -s], [s, c]])
        
        # 1. Body (Ellipse) - Discretized
        num_body_pts = 100
        t = jnp.linspace(0, 2*jnp.pi, num_body_pts, endpoint=False)
        body_x = self.BODY_RADIUS_X * jnp.cos(t)
        body_y = self.BODY_RADIUS_Y * jnp.sin(t)
        body_pts = jnp.stack([body_x, body_y], axis=1)
        
        # 2. Arms (Rectangle)
        # Simplified as a line of points for IBM
        num_arm_pts = 60
        arm_x = jnp.linspace(-self.ARM_LENGTH/2, self.ARM_LENGTH/2, num_arm_pts)
        arm_y = jnp.zeros_like(arm_x)
        arm_pts = jnp.stack([arm_x, arm_y], axis=1)
        
        # 3. Motors (Boxes)
        # Left Motor
        motor_pts_list = []
        for offset in [-self.MOTOR_OFFSET, self.MOTOR_OFFSET]:
            # Box around (offset, 5) size 30x30
            mx = jnp.array([-15, 15, 15, -15, -15]) + offset
            my = jnp.array([-15, -15, 15, 15, -15]) + 5 # +5 y offset
            # Interpolate points along edges
            for i in range(4):
                p1 = jnp.array([mx[i], my[i]])
                p2 = jnp.array([mx[i+1], my[i+1]])
                num_edge = 10
                alphas = jnp.linspace(0, 1, num_edge, endpoint=False)
                edge_pts = p1[None, :] * (1 - alphas[:, None]) + p2[None, :] * alphas[:, None]
                motor_pts_list.append(edge_pts)
                
        motor_pts = jnp.concatenate(motor_pts_list, axis=0)

        # Combine all local points
        all_local_pts = jnp.concatenate([body_pts, arm_pts, motor_pts], axis=0)
        
        # Rotate and Translate
        # (N, 2) @ (2, 2) -> (N, 2)
        rotated_pts = jnp.dot(all_local_pts, rot.T)
        global_pts = rotated_pts + jnp.array([cx, cy])
        
        return global_pts

    def get_propeller_force_field(self, state, grid_shape, key, hover=False, target_height=None):
        """
        Computes the actuator disk force field for the propellers.
        
        Args:
            state: DroneState2D instance
            grid_shape: Tuple (nx, ny)
            key: JAX random key
            hover: If True, use clean thrust + altitude control. If False, use noisy thrust.
            target_height: Target altitude for hovering (only used if hover=True)
        """
        cx, cy = state.pos[0], state.pos[1]
        theta = state.angle
        angular_vel = state.angular_vel
        
        # Base Thrust (calculated to hover)
        base_thrust = self.BASE_THRUST

        # Random noise for hover mode (very small to prevent wobbles)
        k1, k2 = jax.random.split(key)
        noise_left = jax.random.uniform(k1, minval=-0.01, maxval=0.01)*0  # Reduced from ±0.1 to ±0.01
        noise_right = jax.random.uniform(k2, minval=-0.01, maxval=0.01)*0 
        
        if hover and target_height is not None:
            # --- HOVERING MODE (PD Control) ---
            
            # 1. Altitude Control (P Controller)
            altitude_error = target_height - cy
            alt_gain = 1.0
            thrust_adj_vertical = alt_gain * altitude_error

            # 2. Horizontal Position Control (Cascade to Angle)
            target_x = grid_shape[0] / 2.0
            
            x_error = target_x - cx
            vx = state.vel[0]
            
            # PID gains for position → angle (STRENGTHENED)
            kp_pos = 0.15  # Was 0.02 (7.5x stronger position hold)
            kd_pos = 0.5   # Was 0.1 (5x stronger damping)
            
            # Calculate target angle to correct position
            target_angle = -(kp_pos * x_error - kd_pos * vx)
            
            # Clamp target angle (REDUCED to prevent saturation)
            target_angle = jnp.clip(target_angle, -jnp.pi/12, jnp.pi/12) # +/- ~15 degrees
            
            # 3. Angle Stabilization (PD Controller)
            angle_error = target_angle - theta
            
            # --- STRENGTHENED PID FOR HOVER STABILITY ---
            # Increased from (kp=0.1, kd=0.2) to suppress wobbles
            kp = 0.8    # Proportional gain (strong for hover)
            kd = 3.0    # Derivative gain (damps oscillations)
            
            # PD Control equation
            thrust_adj_angle = (kp * angle_error) - (kd * angular_vel)

            # Mix controls:
            thrust_left = base_thrust * (1.0 + noise_left) + (thrust_adj_vertical / 2.0) + thrust_adj_angle
            thrust_right = base_thrust * (1.0 + noise_right) + (thrust_adj_vertical / 2.0) - thrust_adj_angle
        else:
            # --- LANDING MODE (Random Noise) ---
            thrust_left = base_thrust * (1.0 + noise_left)
            thrust_right = base_thrust * (1.0 + noise_right)
        
        # Direction of thrust: Downward in body frame is (0, -1)
        # Rotated: (-sin(theta), -cos(theta))
        thrust_dir = jnp.array([-jnp.sin(theta), -jnp.cos(theta)])
        
        # Propeller Centers
        c, s = jnp.cos(theta), jnp.sin(theta)
        rot = jnp.array([[c, -s], [s, c]])
        
        p_left_local = jnp.array([-self.MOTOR_OFFSET, self.PROP_OFFSET_Y])
        p_right_local = jnp.array([self.MOTOR_OFFSET, self.PROP_OFFSET_Y])
        
        p_left = jnp.dot(rot, p_left_local) + jnp.array([cx, cy])
        p_right = jnp.dot(rot, p_right_local) + jnp.array([cx, cy])
        
        # Create a mask/kernel for the propellers on the grid
        # Simple Gaussian blobs for force distribution
        X, Y = jnp.meshgrid(jnp.arange(grid_shape[0]), jnp.arange(grid_shape[1]), indexing="ij")
        
        def gaussian_blob(x0, y0, sigma=self.PROP_SIGMA):
            return jnp.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
        
        mask_left = gaussian_blob(p_left[0], p_left[1])
        mask_right = gaussian_blob(p_right[0], p_right[1])
        
        fx = (mask_left * thrust_left + mask_right * thrust_right) * thrust_dir[0]
        fy = (mask_left * thrust_left + mask_right * thrust_right) * thrust_dir[1]
        
        return fx, fy

@register_pytree_node_class
class DroneState2D:
    """
    Represents the state of the drone in 2D space.
    """
    def __init__(self, pos, vel, angle, angular_vel):
        """
        pos: [x, y] - position in meters
        vel: [vx, vy] - velocity in meters per second
        angle: radians - orientation in radians
        angular_vel: radians/s - angular velocity in radians per second
        """
        self.pos = pos       # [x, y]
        self.vel = vel       # [vx, vy]
        self.angle = angle   # radians
        self.angular_vel = angular_vel # radians/s

    def tree_flatten(self):
        return ((self.pos, self.vel, self.angle, self.angular_vel), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)