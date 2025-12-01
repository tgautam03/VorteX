# Comprehensive Step-by-Step Implementation Guide
**Project:** Enterprise LBM + AI Design Suite (ThermAI, Sense & Control, AeroOpt)
**Timeline:** Accelerated (Thermal First)
**Goal:** Complete Thermal & Control platforms in Month 1.

---

## **Phase 1: The Thermal Sprint (Weeks 1-4)**

### Step 1.1: Thermal LBM Engine (Week 1)
**Goal:** Simulate heat transfer with Conjugate Heat Transfer (CHT).

1.  **Extend `vortex/lbm/`:**
    *   **Temperature Field:** Add `g` distribution (D2Q5 or D2Q9).
    *   **Collision:** Implement BGK for Temperature: $g_i^{out} = g_i^{in} - \frac{1}{\tau_g} (g_i^{in} - g_i^{eq})$.
    *   **Equilibrium:** $g_i^{eq} = T w_i (1 + \frac{c_i \cdot u}{c_s^2})$.
    *   **Coupling:** Fluid velocity `u` (from NSE) drives Temperature advection.
2.  **Conjugate Heat Transfer (Solids):**
    *   In Solid nodes, set velocity $u=0$.
    *   Use a different diffusivity $\alpha_{solid}$ (via $\tau_{g, solid}$).
    *   Ensure harmonic mean handling of $\tau$ at the interface.

### Step 1.2: DeepONet Surrogate & Optimization (Week 2)
**Goal:** Generative Design of Cold Plates.

1.  **Data Factory:**
    *   Script: `ai/data/generate_thermal_dataset.py`.
    *   Generate 2,000+ random channel masks (Perlin Noise).
    *   Simulate to steady state. Save `(Mask, T_field)`.
2.  **DeepONet Model:**
    *   **Branch:** CNN to encode the Mask $(1, NX, NY)$.
    *   **Trunk:** MLP to encode $(x, y)$.
    *   **Loss:** MSE of Temperature field.
3.  **Topology Optimization:**
    *   Freeze DeepONet.
    *   Optimize Input Mask to minimize $T_{max}$ + Pressure Drop.

### Step 1.3: Virtual Sensing with PINNs (Week 3)
**Goal:** Reconstruct full T-field from 5 sensors.

1.  **PINN Setup:**
    *   Use `deepxde`.
    *   **PDE:** Navier-Stokes + Advection-Diffusion.
    *   **BCs:** Known Temperature at Inlet/Walls.
    *   **Data:** 5 random points $(x_k, y_k)$ with known $T$.
2.  **Training:**
    *   Train on a single complex simulation snapshot.
    *   Verify reconstruction error away from sensors.

### Step 1.4: Predictive Control (Week 4)
**Goal:** Pre-cool for Pulsed Heat Loads.

1.  **Environment:**
    *   Heat Source $Q(t)$ pulses every 500 steps.
    *   Action: Pump Velocity $u_{in} \in [0, U_{max}]$.
2.  **RL Agent:**
    *   **Input:** Current $T$ (from PINN) + "Next Pulse Time" (Countdown).
    *   **Reward:** $- (T_{max} - T_{safe})^2 - \lambda \cdot Power$.
    *   **Result:** Agent should ramp up flow *before* the pulse.

---

## **Phase 2: Aerodynamics (Month 2)**

### Step 2.1: Airfoil Data Factory
*   Generate NACA airfoil dataset (Velocity/Pressure fields).

### Step 2.2: FNO Surrogate
*   Train FNO to predict flow over airfoils.

### Step 2.3: Active Gust Control
**Goal:** Train RL agent with FNO-Augmented Inference.

1.  **Strategy A: The "All-Seeing" Pilot (Virtual Sensing)**
    *   **Problem:** RL usually only sees local pressure sensors (sparse data).
    *   **Solution:** Run FNO in real-time to reconstruct the *entire* velocity field from sparse sensors.
    *   **Flow:** Sensors $\rightarrow$ FNO $\rightarrow$ Full Field Image $\rightarrow$ RL Agent (CNN Policy) $\rightarrow$ Action.
    *   **Benefit:** RL makes better decisions because it "sees" the gust structure before it hits the wing.

2.  **Strategy B: The "Safety Shield" (Predictive Safety)**
    *   **Problem:** RL might try a risky maneuver that leads to stall.
    *   **Solution:** Before executing RL action $A_t$, use FNO to predict $S_{t+1}$.
    *   **Logic:**
        ```python
        action = rl_agent.predict(state)
        future_state = fno.predict(state, action)
        if is_stall(future_state):
            action = safe_recovery_maneuver() # Override
        execute(action)
        ```
    *   **Benefit:** Prevents catastrophic failures (crashes) while allowing aggressive control.

---

## **Directory Structure Updates**
```bash
VorteX/
├── vortex/
│   ├── thermal/          # <--- START HERE
│   │   ├── solver.py     # Thermal LBM Solver
│   │   └── boundary.py   # Thermal BCs
│   ├── lbm/              # Existing NSE Solver
│   └── ...
├── ai/
│   ├── thermal_data/     # <--- Store Week 1 Data here
│   ├── models/
│   │   ├── deeponet.py   # Week 2
│   │   └── pinn.py       # Week 3
│   └── rl/               # Week 4
```
