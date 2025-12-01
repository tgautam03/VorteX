# Accelerated Roadmap: Thermal & Intelligent Systems First

**Objective:** Complete **Project 2 (Generative Thermal)** and **Project 3 (Intelligent Control)** in **1 Month**.
**Follow-up:** Project 1 (Aerodynamics) in Month 2.

> [!WARNING]
> **Intensity Alert:** This is a "Sprint" schedule. It requires implementing Thermal LBM, DeepONet, PINNs, and RL in 4 weeks. Parallel execution of Data Generation and Model Training is essential.

---

## **Month 1: The "Thermal Intelligence" Sprint**
**Goal:** Build the "ThermAI" and "Sense & Control" platforms for Parker-Hannifin.

### **Week 1: The Physics Engine (Thermal LBM)**
*   **Focus:** Core Solver & Data Factory.
*   **Deliverables:**
    *   [ ] **Thermal LBM:** Add Temperature field ($g$) to your JAX solver.
    *   [ ] **Conjugate Heat Transfer:** Implement Solid-Fluid thermal interaction.
    *   [ ] **Data Factory:** Script to generate 2,000+ random "Cold Plate" geometries and simulate them overnight.
    *   *Milestone:* A video of heat flowing through a random channel.

### **Week 2: Generative Design (DeepONet)**
*   **Focus:** Real-time Surrogate & Optimization.
*   **Deliverables:**
    *   [ ] **Train DeepONet:** Map Geometry $\rightarrow$ Temperature Field.
    *   [ ] **Inverse Design Loop:** Use the differentiable DeepONet to optimize channel shapes for min($T_{max}$).
    *   *Milestone:* "Bionic" Cold Plate design that is 30% more efficient than a straight channel.

### **Week 3: Virtual Sensing (PINNs)**
*   **Focus:** Reconstructing reality from sparse sensors.
*   **Deliverables:**
    *   [ ] **PINN Setup:** Define Navier-Stokes + Energy Eq residuals in DeepXDE.
    *   [ ] **Training:** Train PINN on sparse data (5 points) from a Week 1 simulation.
    *   *Milestone:* Dashboard showing Full Field reconstruction from just 5 sensor dots.

### **Week 4: Predictive Control (RL)**
*   **Focus:** Intelligent Active Cooling.
*   **Deliverables:**
    *   [ ] **Pulsed Environment:** Modify LBM to have a time-varying heat source (Laser pulse).
    *   [ ] **RL Agent:** Train PPO to control pump speed based on PINN state + Future Heat indicators.
    *   *Milestone:* Demo of "Pre-cooling" preventing a thermal spike.

---

## **Month 2: Aerodynamics (Project 1)**
**Goal:** Build "AeroOpt" for Zipline (Gust Control).

*   **Week 5:** Airfoil Data Factory & FNO Surrogate.
*   **Week 6:** Active Gust Control (RL) with FNO-Augmented Inference.
*   **Week 7:** Integration & Portfolio Polish.
*   **Week 8:** Final Presentations.

---

## **Technical Dependencies (Execute in Order)**
1.  **Thermal LBM** is the blocker for everything. **Start this immediately.**
2.  **Data Generation** takes time. Start generating data *while* you code the DeepONet.
3.  **PINNs** and **RL** can be developed in parallel if you have a second GPU/machine, otherwise do PINNs first as they are easier to debug.
