"""
Interactive visualization for 2D DRONE FLOW simulation.
Optimized for smooth, real-time slider interaction.

Usage: python plot_drone.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.widgets import Slider
import matplotlib.transforms as transforms

# Load the saved data
print("Loading simulation data...")
try:
    saved_steps = np.load('drone_steps.npy')
    saved_rho = np.load('drone_rho.npy')
    saved_u = np.load('drone_u.npy')
    saved_vorticity = np.load('drone_vorticity.npy')
    saved_drone_states = np.load('drone_states.npy') # [x, y, angle]
    saved_forces = np.load('drone_forces.npy') # [Fx, Fy, Torque]
    print(f"Loaded {len(saved_steps)} frames")
except FileNotFoundError:
    print("Error: Data files not found. Please run 2D_drone_flow.py first.")
    exit(1)

# --- 1. Simulation & Geometry Parameters ---
nx, ny = 600, 800

# Create default drone to get geometry constants
from vortex.drone2d import Drone2D
dx = 0.01  # meters per lattice unit
u_real = 5.0  # m/s
g_SI = 9.81  # m/s^2


# u_lattice is the "reference velocity" for the physics. It defines the fluid's behavior (viscosity).
u_lattice = 0.15  # lattice units per timestep (Fixed for LBM stability) 

# Time scale
dt = (u_lattice / u_real) * dx  # = 0.15/5 * 0.01 = 0.0003 s

# Gravity conversion
GRAVITY = g_SI * dt**2 / dx
_drone_ref = Drone2D(gravity=GRAVITY)
BODY_RADIUS_X = _drone_ref.BODY_RADIUS_X
BODY_RADIUS_Y = _drone_ref.BODY_RADIUS_Y
ARM_LENGTH = _drone_ref.ARM_LENGTH
ARM_THICKNESS = _drone_ref.ARM_THICKNESS
MOTOR_OFFSET = _drone_ref.MOTOR_OFFSET
MOTOR_SIZE = _drone_ref.MOTOR_SIZE
PROP_WIDTH = _drone_ref.PROP_WIDTH
PROP_OFFSET_Y = _drone_ref.PROP_OFFSET_Y

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
# Layout: 3 rows. Top 2 for fields, Bottom 1 for Forces/Torque
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 0.6, 0.05], hspace=0.4, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Force Plots
ax_force = fig.add_subplot(gs[2, 0])
ax_torque = fig.add_subplot(gs[2, 1])

ax_slider = fig.add_subplot(gs[3, :])

# Storage for patches to update them later
drone_patches = {
    'ax1': [], 'ax2': [], 'ax3': [], 'ax4': []
}

def create_drone_patches(ax, key):
    """Creates the drone patches and adds them to the axes."""
    patches = []
    
    # 1. Body (Ellipse)
    body = Ellipse((0, 0), width=2*BODY_RADIUS_X, height=2*BODY_RADIUS_Y, color='grey', alpha=0.8, zorder=11)
    ax.add_patch(body)
    patches.append(body)
    
    # 2. Arms (Rectangle)
    # Centered at (0,0) initially
    arm = Rectangle((-ARM_LENGTH/2, -ARM_THICKNESS/2), ARM_LENGTH, ARM_THICKNESS, color='darkgrey', alpha=0.8, zorder=11)
    ax.add_patch(arm)
    patches.append(arm)
    
    # 3. Motors (Rectangles)
    # Left
    m_left = Rectangle((-MOTOR_OFFSET - 15, -15 + 5), 30, 30, color='black', alpha=0.9, zorder=12)
    ax.add_patch(m_left)
    patches.append(m_left)
    
    # Right
    m_right = Rectangle((MOTOR_OFFSET - 15, -15 + 5), 30, 30, color='black', alpha=0.9, zorder=12)
    ax.add_patch(m_right)
    patches.append(m_right)
    
    # 4. Propellers (Red Rectangles)
    # Left
    p_left = Rectangle((-MOTOR_OFFSET - PROP_WIDTH/2, PROP_OFFSET_Y - 1.5), PROP_WIDTH, 3, color='red', alpha=0.6, zorder=13)
    ax.add_patch(p_left)
    patches.append(p_left)
    
    # Right
    p_right = Rectangle((MOTOR_OFFSET - PROP_WIDTH/2, PROP_OFFSET_Y - 1.5), PROP_WIDTH, 3, color='red', alpha=0.6, zorder=13)
    ax.add_patch(p_right)
    patches.append(p_right)
    
    drone_patches[key] = patches

def update_drone_patches(key, x, y, angle_rad):
    """Updates the position and rotation of the drone patches."""
    patches = drone_patches[key]
    
    # Create transform: Rotate then Translate
    tr = transforms.Affine2D().rotate(angle_rad).translate(x, y)
    
    for p in patches:
        # We need to apply the data transform so it maps to plot coordinates
        p.set_transform(tr + p.axes.transData)

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
create_drone_patches(ax1, 'ax1')
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
create_drone_patches(ax2, 'ax2')
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
create_drone_patches(ax3, 'ax3')
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
create_drone_patches(ax4, 'ax4')
ax4.set_title(r'Y-Velocity Component ($u_y$)')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
cbar4 = plt.colorbar(im4, ax=ax4, label=r'$u_y$')

# --- Force Plots ---
steps = np.arange(len(saved_steps))
# Forces
ax_force.plot(steps, saved_forces[:, 0], label='Fx (Drag)', color='blue', alpha=0.7)
ax_force.plot(steps, saved_forces[:, 1], label='Fy (Lift)', color='green', alpha=0.7)
ax_force.set_title('Aerodynamic Forces')
ax_force.set_xlabel('Frame')
ax_force.set_ylabel('Force')
ax_force.legend()
ax_force.grid(True, alpha=0.3)
line_force = ax_force.axvline(x=0, color='red', linestyle='--')

# Torque
ax_torque.plot(steps, saved_forces[:, 2], label='Torque', color='orange', alpha=0.7)
ax_torque.set_title('Aerodynamic Torque')
ax_torque.set_xlabel('Frame')
ax_torque.set_ylabel('Torque')
ax_torque.legend()
ax_torque.grid(True, alpha=0.3)
line_torque = ax_torque.axvline(x=0, color='red', linestyle='--')


# Initial update
state = saved_drone_states[0]
for key, ax in zip(['ax1', 'ax2', 'ax3', 'ax4'], [ax1, ax2, ax3, ax4]):
    update_drone_patches(key, state[0], state[1], state[2])

title = fig.suptitle(
    f'2D Drone Flow (IBM + Physics) - Step {saved_steps[frame_idx]}',
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

    title.set_text(f'2D Drone Flow (IBM + Physics) - Step {saved_steps[frame_idx]}')
    
    # Update Drone Position
    state = saved_drone_states[frame_idx]
    for key in ['ax1', 'ax2', 'ax3', 'ax4']:
        update_drone_patches(key, state[0], state[1], state[2])
        
    # Update Force Lines
    line_force.set_xdata([frame_idx])
    line_torque.set_xdata([frame_idx])

    fig.canvas.draw_idle()

slider.on_changed(update)

print("\nVisualization ready!")
print("Use the slider to scroll through timesteps.")
print("Close the window to exit.")

plt.show()