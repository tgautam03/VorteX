"""
Render 3D backyard flow simulation as a video with:
- Rotating camera view from above
- Animated slices moving through the domain
- Configurable FPS, duration, and resolution
"""

import pyvista as pv
import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path
import sys

# Add path to import vortex module
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from vortex.obstacles import backyard_scene

# ============================================================================
# CONFIGURATION
# ============================================================================

# Video Settings
VIDEO_FPS = 10              # Frames per second for output video
                            # Duration will be auto-calculated: num_simulation_frames / VIDEO_FPS
RENDER_WIDTH = 1920         # Video width (e.g., 1920 for Full HD, 3840 for 4K)
RENDER_HEIGHT = 1080        # Video height (e.g., 1080 for Full HD, 2160 for 4K)
FRAME_SKIP = 0              # Render every Nth simulation frame (higher = faster but choppier)
VIDEO_QUALITY = 10           # Video quality 1-10 (lower = faster encoding but larger file)

# Animation Settings
NUM_SLICE_CYCLES = 2        # How many times slices traverse the domain during animation
CAMERA_ROTATIONS = 0.75        # Number of full rotations around domain

# Paths
FRAMES_DIR = Path("examples/npy_files/frames")
OUTPUT_VIDEO = Path("examples/backyard_flow_animation.mp4")

# ============================================================================


def load_simulation_frame(frames_dir, frame_idx):
    """Load a single simulation frame."""
    rho_files = sorted(list(frames_dir.glob("rho_*.npy")))
    u_files = sorted(list(frames_dir.glob("u_*.npy")))
    
    if frame_idx >= len(rho_files):
        return None, None, None, None, None
    
    rho = np.load(rho_files[frame_idx])
    u = np.load(u_files[frame_idx])
    
    nx, ny, nz = rho.shape
    
    # Compute vorticity
    du_dx = np.gradient(u[0], axis=0)
    du_dy = np.gradient(u[0], axis=1)
    du_dz = np.gradient(u[0], axis=2)
    
    dv_dx = np.gradient(u[1], axis=0)
    dv_dy = np.gradient(u[1], axis=1)
    dv_dz = np.gradient(u[1], axis=2)
    
    dw_dx = np.gradient(u[2], axis=0)
    dw_dy = np.gradient(u[2], axis=1)
    dw_dz = np.gradient(u[2], axis=2)
    
    omega_x = dw_dy - dv_dz
    omega_y = du_dz - dw_dx
    omega_z = dv_dx - du_dy
    
    vorticity_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
    velocity_mag = np.sqrt(u[0]**2 + u[1]**2 + u[2]**2)
    
    # Apply smoothing to vorticity
    omega_x_smooth = gaussian_filter(omega_x, sigma=1.5)
    
    # Create grid
    grid = pv.ImageData(dimensions=(nx, ny, nz))
    grid.point_data["Density"] = rho.flatten(order='F')
    grid.point_data["Velocity Magnitude"] = velocity_mag.flatten(order='F')
    grid.point_data["Omega_X"] = omega_x_smooth.flatten(order='F')
    
    # Add vector data
    u_flat = np.zeros((nx * ny * nz, 3))
    u_flat[:, 0] = u[0].flatten(order='F')
    u_flat[:, 1] = u[1].flatten(order='F')
    u_flat[:, 2] = u[2].flatten(order='F')
    grid.point_data["Velocity"] = u_flat
    
    return grid, nx, ny, nz, omega_x_smooth


def create_obstacle_mesh(nx, ny, nz):
    """Create obstacle mesh from backyard scene."""
    backyard_mask = np.array(backyard_scene(nx, ny, nz))
    obstacle_grid = pv.ImageData(dimensions=(nx, ny, nz))
    obstacle_grid.point_data["Obstacle"] = backyard_mask.flatten(order='F').astype(float)
    obstacles = obstacle_grid.contour([0.5], scalars="Obstacle")
    return obstacles


def get_camera_position(angle_degrees, nx, ny, nz):
    """Calculate camera position for given rotation angle."""
    angle_rad = np.radians(angle_degrees)
    
    # Camera orbits around the center at a distance - Balanced zoom-out
    radius_x = nx * 3.5
    radius_z = nz * 3.5
    
    camera_x = nx / 2 + radius_x * np.cos(angle_rad)
    camera_y = ny * 4.5
    camera_z = nz / 2 + radius_z * np.sin(angle_rad)
    
    focal_point = (nx / 2, ny / 4, nz / 2)
    
    return [(camera_x, camera_y, camera_z), focal_point, (0, 1, 0)]


def get_slice_positions(progress, ny, nz, num_cycles):
    """
    Calculate slice positions based on animation progress.
    Slices move back and forth across the domain.
    
    Args:
        progress: 0.0 to 1.0 (animation progress)
        ny, nz: domain dimensions
        num_cycles: number of times to traverse the domain
    
    Returns:
        (y_slice, z_slice) positions
    """
    # Y slice (horizontal): oscillates from 0 to ny
    y_phase = (progress * num_cycles) % 1.0
    # Use sine wave for smooth back-and-forth
    y_slice = int(ny * 0.5 * (1 + np.sin(2 * np.pi * y_phase)))
    y_slice = max(0, min(ny - 1, y_slice))
    
    # Z slice (vertical): oscillates with a phase offset
    z_phase = ((progress * num_cycles) + 0.5) % 1.0  # 180° phase shift
    z_slice = int(nz * 0.5 * (1 + np.sin(2 * np.pi * z_phase)))
    z_slice = max(0, min(nz - 1, z_slice))
    
    return y_slice, z_slice


def render_frame(plotter, grid, obstacles, nx, ny, nz, omega_data, y_slice, z_slice, camera_pos):
    """Render a single frame to the plotter."""
    plotter.clear()
    
    titles = ["Velocity Magnitude", "Vorticity (Omega_X)", "Density"]
    scalars = ["Velocity Magnitude", "Omega_X", "Density"]
    cmaps = ["viridis", "RdBu_r", "coolwarm"]
    
    for i in range(3):
        plotter.subplot(0, i)
        plotter.add_text(titles[i], font_size=12, position='upper_edge')
        plotter.show_bounds(grid='front', location='outer', all_edges=True)
        
        # Add obstacles
        if obstacles.n_points > 0:
            plotter.add_mesh(obstacles, color='saddlebrown', opacity=0.7)
        
        # Create slices
        slice_y = grid.slice(normal='y', origin=(0, y_slice, 0))
        slice_z = grid.slice(normal='z', origin=(0, 0, z_slice))
        
        # Determine color limits
        if scalars[i] == "Omega_X":
            vort_p5 = np.percentile(omega_data, 5)
            vort_p95 = np.percentile(omega_data, 95)
            vort_max = max(abs(vort_p5), abs(vort_p95))
            clim = [-vort_max, vort_max]
        else:
            clim = None
        
        # Add slices
        if slice_y.n_points > 0:
            plotter.add_mesh(slice_y, scalars=scalars[i], cmap=cmaps[i], 
                           show_scalar_bar=True, clim=clim)
        if slice_z.n_points > 0:
            plotter.add_mesh(slice_z, scalars=scalars[i], cmap=cmaps[i], 
                           show_scalar_bar=False, clim=clim)
        
        # Set camera AFTER all meshes are added (critical for clipping range calculation)
        plotter.camera_position = camera_pos
        
        # Set wider field of view to prevent clipping at edges
        plotter.camera.view_angle = 50  # Default is ~30, increase to see more
        
        # FIX: Reset clipping range AFTER meshes are added
        plotter.reset_camera_clipping_range()


def main():
    print("="*70)
    print("3D Backyard Flow Video Renderer")
    print("="*70)
    
    # Check if simulation data exists
    if not FRAMES_DIR.exists():
        print(f"ERROR: Frames directory not found: {FRAMES_DIR}")
        print("Please run the simulation first (backyard_flow_3d.py)")
        return
    
    # Count available frames
    rho_files = sorted(list(FRAMES_DIR.glob("rho_*.npy")))
    num_sim_frames = len(rho_files)
    
    if num_sim_frames == 0:
        print(f"ERROR: No simulation frames found in {FRAMES_DIR}")
        return
    
    print(f"Found {num_sim_frames} simulation frames")
    
    # Calculate duration from available frames and FPS (1:1 mapping)
    VIDEO_DURATION = num_sim_frames / VIDEO_FPS
    total_video_frames = num_sim_frames
    
    print(f"\nVideo Settings:")
    print(f"  FPS: {VIDEO_FPS}")
    print(f"  Duration: {VIDEO_DURATION:.2f}s (auto-calculated from {num_sim_frames} frames)")
    print(f"  Resolution: {RENDER_WIDTH}x{RENDER_HEIGHT}")
    print(f"  Output: {OUTPUT_VIDEO}")
    print(f"\nAnimation Settings:")
    print(f"  Slice cycles: {NUM_SLICE_CYCLES}")
    print(f"  Camera rotations: {CAMERA_ROTATIONS}")
    print(f"\nGenerating {total_video_frames} video frames...")
    print("="*70)
    
    # Setup plotter (off-screen for rendering)
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, 
                        window_size=[RENDER_WIDTH, RENDER_HEIGHT])
    plotter.set_background("white")
    
    # Load first frame to get dimensions
    print("\nLoading first simulation frame...")
    grid, nx, ny, nz, omega_data = load_simulation_frame(FRAMES_DIR, 0)
    
    if grid is None:
        print("ERROR: Could not load first frame")
        return
    
    print(f"Domain size: {nx} x {ny} x {nz}")
    
    # Create obstacle mesh (static)
    print("Creating obstacle mesh...")
    obstacles = create_obstacle_mesh(nx, ny, nz)
    
    # Open video writer
    plotter.open_movie(str(OUTPUT_VIDEO), framerate=VIDEO_FPS, quality=VIDEO_QUALITY)
    
    print("\nRendering frames...")
    # Apply frame skipping to reduce total frames to render
    frames_to_render = list(range(0, total_video_frames, FRAME_SKIP + 1))
    print(f"Skipping frames: rendering {len(frames_to_render)} of {total_video_frames} frames (every {FRAME_SKIP + 1}th frame)")
    
    for idx, video_frame in enumerate(frames_to_render):
        # Calculate progress (0.0 to 1.0)
        progress = video_frame / total_video_frames
        
        # Map to simulation frame (loop if necessary)
        sim_frame = int((progress * num_sim_frames) % num_sim_frames)
        
        # Load simulation data
        grid, nx, ny, nz, omega_data = load_simulation_frame(FRAMES_DIR, sim_frame)
        
        # Calculate camera angle (rotates over the duration)
        camera_angle = (progress * 360 * CAMERA_ROTATIONS) % 360
        camera_pos = get_camera_position(camera_angle, nx, ny, nz)
        
        # Calculate slice positions
        y_slice, z_slice = get_slice_positions(progress, ny, nz, NUM_SLICE_CYCLES)
        
        # Render frame
        render_frame(plotter, grid, obstacles, nx, ny, nz, omega_data, 
                    y_slice, z_slice, camera_pos)
        
        # Write frame to video
        plotter.write_frame()
        
        # Progress indicator
        if (idx + 1) % 10 == 0 or idx == len(frames_to_render) - 1:
            percent = 100 * (idx + 1) / len(frames_to_render)
            print(f"  Progress: {percent:.1f}% ({idx + 1}/{len(frames_to_render)}) "
                  f"[Sim frame: {sim_frame}, Camera: {camera_angle:.1f}°, "
                  f"Slices: Y={y_slice}, Z={z_slice}]")
    
    # Finalize video
    plotter.close()
    
    print("\n" + "="*70)
    print(f"✓ Video saved to: {OUTPUT_VIDEO}")
    print(f"  Duration: {VIDEO_DURATION:.2f}s")
    print(f"  Rendered frames: {len(frames_to_render)} (skipped every {FRAME_SKIP}th)")
    print(f"  Resolution: {RENDER_WIDTH}x{RENDER_HEIGHT}")
    print(f"  Quality: {VIDEO_QUALITY}/10")
    print("="*70)


if __name__ == "__main__":
    main()
