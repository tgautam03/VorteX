"""
Interactive visualization for 2D DRONE FLOW simulation.
Optimized for smooth, real-time slider interaction.

Usage: python plot_drone.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse # Import Ellipse for the body
from matplotlib.widgets import Slider

# Load the saved data
print("Loading simulation data...")
try:
    saved_steps = np.load('drone_steps.npy')
    saved_rho = np.load('drone_rho.npy')
    saved_u = np.load('drone_u.npy')
    saved_vorticity = np.load('drone_vorticity.npy')
    print(f"Loaded {len(saved_steps)} frames")
except FileNotFoundError:
    print("Error: Data files not found. Please run 2D_drone_flow.py first.")
    exit(1)

# --- 1. Simulation & Geometry Parameters ---
nx, ny = 600, 600

# Geometry parameters (must match the simulation script)
drone_cx, drone_cy = nx // 2, ny - 150 

# Drone Body (Ellipse)
body_radius_x, body_radius_y = 30, 20

# Drone Arms (Rectangles)
arm_half_length = 180 // 2
arm_half_width = 10 // 2
# Note: The original mask_arms was (jnp.abs(X - drone_cx) < 180 // 2) & (jnp.abs(Y - drone_cy) < 10 // 2)
# This creates a single horizontal rectangle.
arm_x_start = drone_cx - arm_half_length
arm_y_start = drone_cy - arm_half_width
arm_width = 2 * arm_half_length
arm_height = 2 * arm_half_width


# Motor Housings (Rectangles)
motor_offset = 90
motor_half_size = 15 # Original mask was < 15, so total size 30
motor_y_offset = 5 # Original mask was (Y - (drone_cy + 5))
motor_x_size = 2 * motor_half_size
motor_y_size = 2 * motor_half_size
motor_housing_y_center = drone_cy + motor_y_offset


# Propellers (Actuator Disks) - for drawing
prop_y = drone_cy - 10
prop_width = 50 


# --- 2. Color Scale Clipping & Pre-computation ---
print("Pre-computing velocity magnitudes and setting visualization scales...")
saved_u_mag = np.sqrt(saved_u[:, 0]**2 + saved_u[:, 1]**2)

# Global Max/Min for Velocity and X-Velocity
u_mag_max = np.max(saved_u_mag)
ux_min, ux_max = np.min(saved_u[:, 0]), np.max(saved_u[:, 0])

# FIX A: CLIP DENSITY COLOR SCALE
RHO_CENTER = 1.0
RHO_CLIP = 0.01 
rho_min, rho_max = RHO_CENTER - RHO_CLIP, RHO_CENTER + RHO_CLIP

# FIX B: CLIP VORTICITY COLOR SCALE (STATIC CLIPPING)
VORT_MAX = 0.005 
vort_min, vort_max = -VORT_MAX, VORT_MAX


print("Setting up visualization...")

# Create figure with subplots
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.05], hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax_slider = fig.add_subplot(gs[2, :])

# Helper function to draw the drone footprint
def draw_drone_geometry(ax): # Renamed function
    """Draws the drone body, arms, motors, and propellers."""
    
    # 1. Drone Body (Ellipse)
    body_ellipse = Ellipse(
        (drone_cx, drone_cy), 
        width=2 * body_radius_x, 
        height=2 * body_radius_y, 
        color='grey', alpha=0.8, zorder=11
    )
    ax.add_patch(body_ellipse)

    # 2. Drone Arms (Single horizontal rectangle)
    arm_rect = Rectangle(
        (arm_x_start, arm_y_start), 
        arm_width, arm_height, 
        color='darkgrey', alpha=0.8, zorder=11
    )
    ax.add_patch(arm_rect)

    # 3. Motor Housings (Rectangles)
    # Left Motor Housing
    motor_left = Rectangle(
        (drone_cx - motor_offset - motor_half_size, motor_housing_y_center - motor_half_size), 
        motor_x_size, motor_y_size, 
        color='black', alpha=0.9, zorder=12
    )
    ax.add_patch(motor_left)
    # Right Motor Housing
    motor_right = Rectangle(
        (drone_cx + motor_offset - motor_half_size, motor_housing_y_center - motor_half_size), 
        motor_x_size, motor_y_size, 
        color='black', alpha=0.9, zorder=12
    )
    ax.add_patch(motor_right)

    # 4. Propellers (Actuator Disks - Red Rectangles)
    # Left Propeller location
    rect_left_prop = Rectangle(
        (drone_cx - motor_offset - prop_width / 2, prop_y - 1.5), 
        prop_width, 3, color='red', alpha=0.6, zorder=13
    )
    ax.add_patch(rect_left_prop)
    # Right Propeller location
    rect_right_prop = Rectangle(
        (drone_cx + motor_offset - prop_width / 2, prop_y - 1.5), 
        prop_width, 3, color='red', alpha=0.6, zorder=13
    )
    ax.add_patch(rect_right_prop)

# Initialize with first frame
frame_idx = 0

# --- Velocity Magnitude ---
im1 = ax1.imshow(
    saved_u_mag[frame_idx].T,
    origin='lower',
    cmap='jet',
    extent=[0, nx, 0, ny],
    vmin=0,
    vmax=u_mag_max,
    aspect='equal'
)
draw_drone_geometry(ax1) # Call the new function
ax1.set_title(r'Velocity Magnitude $|u|$')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
cbar1 = plt.colorbar(im1, ax=ax1, label=r'$|u|$')

# --- Vorticity ---
im2 = ax2.imshow(
    saved_vorticity[frame_idx].T,
    origin='lower',
    cmap='RdBu_r',
    extent=[0, nx, 0, ny],
    vmin=vort_min, 
    vmax=vort_max, 
    aspect='equal'
)
draw_drone_geometry(ax2) # Call the new function
ax2.set_title(r'Vorticity ($\omega$)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
cbar2 = plt.colorbar(im2, ax=ax2, label=r'$\omega$')

# --- Density ---
im3 = ax3.imshow(
    saved_rho[frame_idx].T,
    origin='lower',
    cmap='viridis',
    extent=[0, nx, 0, ny],
    vmin=rho_min, 
    vmax=rho_max,
    aspect='equal'
)
draw_drone_geometry(ax3) # Call the new function
ax3.set_title(r'Density ($\rho$)')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
cbar3 = plt.colorbar(im3, ax=ax3, label=r'$\rho$')

# --- Y-velocity Component ---
im4 = ax4.imshow(
    saved_u[frame_idx, 1].T,
    origin='lower',
    cmap='coolwarm',
    extent=[0, nx, 0, ny],
    vmin=ux_min,
    vmax=ux_max,
    aspect='equal'
)
draw_drone_geometry(ax4) # Call the new function
ax4.set_title(r'Y-Velocity Component ($u_y$)')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
cbar4 = plt.colorbar(im4, ax=ax4, label=r'$u_y$')

# Set viewport to focus on the drone wake
VIEW_HEIGHT = 350
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_ylim(drone_cy - VIEW_HEIGHT, ny)
    ax.set_xlim(drone_cx - 250, drone_cx + 250)


title = fig.suptitle(
    f'2D Drone Flow (Actuator Disk) - Step {saved_steps[frame_idx]}',
    fontsize=16,
    fontweight='bold'
)

# Create slider
slider = Slider(
    ax=ax_slider,
    label='Frame',
    valmin=0,
    valmax=len(saved_steps) - 1,
    valinit=0,
    valstep=1
)

def update(val):
    """Update plots with dynamic vorticity scaling."""
    frame_idx = int(slider.val)

    im1.set_data(saved_u_mag[frame_idx].T)
    im3.set_data(saved_rho[frame_idx].T)
    im4.set_data(saved_u[frame_idx, 1].T) 
    
    vort = saved_vorticity[frame_idx]
    vmin = np.percentile(vort, 2)
    vmax = np.percentile(vort, 98)
    im2.set_clim(vmin, vmax) 
    im2.set_data(vort.T)
    cbar2.update_normal(im2)   

    title.set_text(f'2D Drone Flow (Actuator Disk) - Step {saved_steps[frame_idx]}')

    fig.canvas.draw_idle()

slider.on_changed(update)

print("\nVisualization ready!")
print("Use the slider to scroll through timesteps.")
print("Close the window to exit.")

plt.show()