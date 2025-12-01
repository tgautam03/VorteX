# VorteX - 2D Drone Landing Simulation

Accelerated drone landing simulation using lattice Boltzmann method (LBM) with Immersed Boundary Method (IBM), powered by XLB and JAX.

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Run drone landing simulation
PYTHONPATH=/home/rvn/pet-projects/VorteX python examples/drone_landing_2d.py

# Visualize results
python plot_drone.py
```

## Architecture

VorteX uses a modular architecture with clear separation of concerns:

```
vortex/
├── units.py           # Unit conversions (SI ↔ lattice)
├── environment2d.py   # LBM grid, BCs, fluid properties
├── drone2d.py         # Drone geometry, state, forces
├── ibm.py             # Immersed Boundary Method
└── simulation.py      # Main simulation runner

examples/
└── drone_landing_2d.py  # Clean executable script
```

## Features

- **GPU-accelerated**: JAX backend with CUDA support (~50 MLUPS)
- **Modular design**: Reusable components for different simulations
- **Physically accurate**: Proper unit conversions and realistic drone physics
- **IBM coupling**: Two-way fluid-structure interaction
- **Interactive visualization**: Matplotlib with timestep slider

## Example Usage

```python
from vortex.units import UnitConverter
from vortex.environment2d import Environment2D
from vortex.drone2d import Drone2D, DroneState2D
from vortex.simulation import DroneSimulation2D

# Setup units
units = UnitConverter(dx=0.01, u_real=5.0, u_lattice=0.15)

# Create environment
env = Environment2D(nx=600, ny=800)
env.setup_boundaries()
env.calculate_lbm_parameters(Re_target=1000, L_char=50, u_char=0.15)

# Create drone
drone = Drone2D(gravity=units.gravity_to_lattice(-9.81))
initial_state = DroneState2D(...)

# Run simulation
sim = DroneSimulation2D(env, drone, initial_state, units)
sim.run(num_steps=10000)
sim.save_data('drone')
```

## Simulation Parameters

- **Grid**: 600 × 800 lattice points
- **Reynolds Number**: 1000 (turbulent flow)
- **Drone**: 2 kg mass, 0.5 m propellers
- **Boundary conditions**: Ground bounce-back, open top/sides
- **Physics**: Gravity, thrust, aerodynamic forces

## Requirements

```bash
pip install jax xlb matplotlib numpy
```

## Documentation

- `drone_geometry_explained.md` - Drone geometry parameters
- `physics_explanation.md` - Flow physics details
- `walkthrough.md` - Architecture overview
