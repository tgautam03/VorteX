import pyvista as pv
import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path
import sys

class SimulationVisualizer:
    def __init__(self, frames_dir):
        self.frames_dir = Path(frames_dir)
        self.rho_files = sorted(list(self.frames_dir.glob("rho_*.npy")))
        self.u_files = sorted(list(self.frames_dir.glob("u_*.npy")))
        
        if not self.rho_files or not self.u_files:
            print(f"No data found in {frames_dir}")
            sys.exit(1)
            
        self.num_frames = len(self.rho_files)
        print(f"Found {self.num_frames} frames.")
        
        # Load first frame to get dimensions
        rho_0 = np.load(self.rho_files[0])
        # rho shape is (NX, NY, NZ)
        self.nx, self.ny, self.nz = rho_0.shape
        
        # Setup Plotter
        self.plotter = pv.Plotter(shape=(1, 3))
        self.plotter.set_background("white")
        
        # Initial State
        self.current_frame = 0
        self.slice_y = self.ny // 2
        self.slice_z = self.nz // 2
        
        # Sphere Parameters
        self.sphere_radius = 10
        self.sphere_center = (self.nx // 4, self.ny // 2, self.nz // 2)
        
        # Placeholders for meshes
        self.mesh = None
        self.slices = {} # Store slice actors for each subplot
        self.streamlines_actor = None
        
        # Initialize
        self.load_frame(0)
        self.setup_scene()
        self.add_widgets()
        
    def load_frame(self, frame_idx):
        """Lazy load data for a specific frame."""
        if frame_idx < 0 or frame_idx >= self.num_frames:
            return

        print(f"Loading frame {frame_idx}...")
        rho = np.load(self.rho_files[frame_idx])    # (NX, NY, NZ)
        u = np.load(self.u_files[frame_idx])        # (3, NX, NY, NZ)
        
        # Compute Vorticity (Curl of U)
        # u is (3, NX, NY, NZ). PyVista expects (NX, NY, NZ, 3) for vectors usually, 
        # or we can compute gradients using numpy.
        # Curl = (dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy)
        # u = [u, v, w]
        
        # Transpose u to (NX, NY, NZ, 3) for easier handling if needed, but numpy gradient works on axes.
        # u[0] = u_x, u[1] = u_y, u[2] = u_z
        
        # Gradients
        # np.gradient returns [d/dx, d/dy, d/dz] for each component
        # We need to be careful with axis ordering. 
        # Array is (3, NX, NY, NZ). 
        # axis 1 is X, axis 2 is Y, axis 3 is Z.
        
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
        
        # Create/Update Mesh
        # We use a StructuredGrid
        x = np.arange(self.nx)
        y = np.arange(self.ny)
        z = np.arange(self.nz)
        grid = pv.StructuredGrid(*np.meshgrid(x, y, z, indexing='ij'))
        
        grid.point_data["Density"] = rho.flatten(order='F') # PyVista uses Fortran order for flattening by default? No, usually C.
        # Wait, meshgrid indexing='ij' means x varies fastest? 
        # Let's check PyVista StructuredGrid convention.
        # If we pass meshgrid(x,y,z, indexing='ij'), grid.points will be ordered such that x changes, then y, then z?
        # Actually, let's just use grid.dimensions = (NX, NY, NZ) and flatten accordingly.
        # If we assign data, it must match the point ordering.
        
        # Let's trust PyVista's wrap or UniformGrid for simplicity if grid is uniform.
        # The simulation uses uniform grid.
        grid = pv.ImageData(dimensions=(self.nx, self.ny, self.nz))
        
        # Assign data
        # ImageData points are ordered x, then y, then z.
        # Our arrays are (NX, NY, NZ). Flattening 'F' (Fortran) means first index changes fastest.
        # In numpy (NX, NY, NZ), 'F' means NX changes fastest. This matches ImageData x-fastest.
        
        grid.point_data["Density"] = rho.flatten(order='F')
        grid.point_data["Velocity Magnitude"] = velocity_mag.flatten(order='F')
        grid.point_data["Vorticity Magnitude"] = vorticity_mag.flatten(order='F')
        
        # Add individual vorticity components for better visualization
        # For flow in +X direction, vortex shedding rotates around X-axis (omega_x)
        # Apply Gaussian smoothing to reduce numerical noise
        omega_x_smooth = gaussian_filter(omega_x, sigma=1.5)
        grid.point_data["Omega_X"] = omega_x_smooth.flatten(order='F')  # For visualizing vortex shedding
        
        # Add vector data for streamlines
        # PyVista expects vectors to be (N, 3)
        # u is (3, NX, NY, NZ). Flatten each component.
        u_flat = np.zeros((self.nx * self.ny * self.nz, 3))
        u_flat[:, 0] = u[0].flatten(order='F')
        u_flat[:, 1] = u[1].flatten(order='F')
        u_flat[:, 2] = u[2].flatten(order='F')
        grid.point_data["Velocity"] = u_flat
        
        self.mesh = grid
        self.current_frame = frame_idx

    def setup_scene(self):
        titles = ["Velocity Magnitude", "Vorticity (Omega_X)", "Density"]
        scalars = ["Velocity Magnitude", "Omega_X", "Density"]
        cmaps = ["viridis", "RdBu_r", "coolwarm"]
        
        for i in range(3):
            self.plotter.subplot(0, i)
            self.plotter.add_text(titles[i], font_size=10)
            self.plotter.show_grid()
            self.plotter.show_bounds(grid='front', location='outer', all_edges=True)
            
            # Add Sphere Obstacle
            sphere = pv.Sphere(radius=self.sphere_radius, center=self.sphere_center)
            self.plotter.add_mesh(sphere, color='gray', opacity=0.5)
            
            # Initial Slices
            self.update_subplot(i, scalars[i], cmaps[i])
            
        # Add Streamlines to Vorticity Plot (Index 1)
        self.update_streamlines()
            
    def update_subplot(self, index, scalar, cmap):
        self.plotter.subplot(0, index)
        # Clear previous actors if any (except bounds/text)
        # It's easier to just remove specific actors if we track them.
        # But PyVista slices are actors.
        
        # We will keep track of slice actors in a list/dict
        if index in self.slices:
            for actor in self.slices[index]:
                self.plotter.remove_actor(actor)
        
        self.slices[index] = []
        
        # Create Slices
        # Vertical (Y-plane) - Longitudinal
        slice_y = self.mesh.slice(normal='y', origin=(0, self.slice_y, 0))
        # Horizontal (Z-plane)
        slice_z = self.mesh.slice(normal='z', origin=(0, 0, self.slice_z))
        
        # For Omega_X (vorticity), use symmetric color limits centered at 0
        # Use percentile clipping for better contrast (ignore extreme outliers)
        if scalar == "Omega_X":
            vort_data = self.mesh.point_data["Omega_X"]
            # Use 5th and 95th percentile for robust limits
            vort_p5 = np.percentile(vort_data, 5)
            vort_p95 = np.percentile(vort_data, 95)
            vort_max = max(abs(vort_p5), abs(vort_p95))
            clim = [-vort_max, vort_max]
        else:
            clim = None
        
        # Add to plot
        if slice_y.n_points > 0:
            actor_y = self.plotter.add_mesh(slice_y, scalars=scalar, cmap=cmap, show_scalar_bar=True, clim=clim)
            self.slices[index].append(actor_y)
            
        if slice_z.n_points > 0:
            actor_z = self.plotter.add_mesh(slice_z, scalars=scalar, cmap=cmap, show_scalar_bar=False, clim=clim)
            self.slices[index].append(actor_z)

    def update_streamlines(self):
        # Update streamlines in Vorticity subplot (Index 1)
        self.plotter.subplot(0, 1)
        
        if self.streamlines_actor:
            self.plotter.remove_actor(self.streamlines_actor)
            
        # Generate Streamlines
        # Seed from a plane near the inlet or around the sphere
        # Let's seed from a plane at x=0 (Inlet)
        # stream = self.mesh.streamlines(vectors='Velocity', source_center=(0, self.ny/2, self.nz/2), source_radius=self.ny/2, n_points=100)
        
        # Or better, seed from a line or plane upstream of the sphere
        # Sphere is at NX/4. Let's seed at NX/8.
        seed_x = self.nx // 8
        stream = self.mesh.streamlines(
            vectors='Velocity', 
            source_center=(seed_x, self.ny/2, self.nz/2),
            source_radius=min(self.ny, self.nz)/2,
            n_points=50,
            integration_direction='forward',
            max_time=1000
        )
        
        if stream.n_points > 0:
            self.streamlines_actor = self.plotter.add_mesh(stream, color='white', opacity=0.3, line_width=1)

    def update_all_plots(self):
        scalars = ["Velocity Magnitude", "Omega_X", "Density"]
        cmaps = ["viridis", "RdBu_r", "coolwarm"]
        for i in range(3):
            self.update_subplot(i, scalars[i], cmaps[i])
        self.update_streamlines()

    def add_widgets(self):
        # Time Slider
        self.plotter.subplot(0, 0) # Add global widgets to first subplot or global?
        # PyVista widgets are usually per renderer, but we can try adding to one.
        
        def time_callback(value):
            frame = int(value)
            if frame != self.current_frame:
                self.load_frame(frame)
                self.update_all_plots()
                
        self.plotter.add_slider_widget(
            time_callback,
            [0, self.num_frames - 1],
            value=0,
            title="Time Step",
            pointa=(0.1, 0.9),
            pointb=(0.4, 0.9),
            style='modern'
        )
        
        # Y-Slice Slider (Vertical)
        def y_slice_callback(value):
            self.slice_y = int(value)
            self.update_all_plots()
            
        self.plotter.add_slider_widget(
            y_slice_callback,
            [0, self.ny - 1],
            value=self.ny // 2,
            title="Y Slice (Vertical)",
            pointa=(0.5, 0.9),
            pointb=(0.8, 0.9),
            style='modern'
        )
        
        # Z-Slice Slider (Horizontal)
        # Add to another subplot to avoid overlap or adjust position
        self.plotter.subplot(0, 1)
        def z_slice_callback(value):
            self.slice_z = int(value)
            self.update_all_plots()
            
        self.plotter.add_slider_widget(
            z_slice_callback,
            [0, self.nz - 1],
            value=self.nz // 2,
            title="Z Slice (Horizontal)",
            pointa=(0.1, 0.9),
            pointb=(0.4, 0.9),
            style='modern'
        )

    def show(self):
        self.plotter.show()

if __name__ == "__main__":
    frames_dir = Path("examples/npy_files/frames")
    if not frames_dir.exists():
        # Fallback for running from root
        frames_dir = Path("examples/npy_files/frames")
        if not frames_dir.exists():
             print("Could not find frames directory.")
             sys.exit(1)
             
    viz = SimulationVisualizer(frames_dir)
    viz.show()
