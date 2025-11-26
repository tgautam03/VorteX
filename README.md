# VorteX - 2D Cylinder Flow Simulation

Accelerated computational fluid dynamics using lattice Boltzmann method with XLB and JAX.

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# 1. Run simulation (saves data to .npy files)
python 2D_cylinder_flow.py

# 2. Visualize results with interactive slider
python plot_flow.py
```

## Files

- **`2D_cylinder_flow.py`** - Main LBM simulation (saves 200 frames)
- **`plot_flow.py`** - Interactive visualization with slider

## Features

- **GPU-accelerated**: JAX backend with CUDA support (~50 MLUPS)
- **Interactive visualization**: Matplotlib slider to scroll through 200 timesteps
- **4 field views**: Velocity magnitude + streamlines, vorticity, density, x-velocity
- **No Jupyter required**: Standalone matplotlib visualization

## Simulation Parameters

- Grid: 800 × 200
- Reynolds Number: 100
- Cylinder: radius=10, center=(100,100)
- Boundary conditions: Zou-He inlet/outlet, bounce-back walls/cylinder

## Requirements

```bash
pip install jax xlb matplotlib numpy
```
