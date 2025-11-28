# Neural Network Control Strategy for Drone Stabilization

## Objective
Use a neural network (Operator Network or Transformer) to **predict and preemptively counteract** aerodynamic disturbances before they cause large tilts.

---

## Overall Strategy: Feedforward + Feedback Control

```
Current PID:  Disturbance → Drone Tilts → PID Reacts → Corrects (REACTIVE)
Neural Net:   Sense Disturbance → Predict Force → Preempt Tilt (PROACTIVE)
```

**Hybrid Approach**: Neural Net (feedforward) + PID (feedback)
- **Neural Net**: Predicts incoming forces from fluid field, adjusts thrust preemptively
- **PID**: Handles residual errors and unknown disturbances

---

## Architecture Choice

### Option 1: DeepONet (Operator Network) ⭐ RECOMMENDED
**Why**: Maps fluid velocity field → aerodynamic forces

**Input**:
- **Branch Net**: Local fluid velocity field around drone (e.g., 50×50 window)
- **Trunk Net**: Drone state (position, velocity, angle, angular velocity)

**Output**:
- Predicted aerodynamic forces: `[Fx, Fy, Torque]` in next 5-10 timesteps

**Advantages**:
- Physics-informed: learns the operator mapping u(x,y) → F
- Generalizes to different flow conditions
- Fast inference (~1ms)

### Option 2: Transformer
**Why**: Models temporal sequence of disturbances

**Input**:
- Sequence of past 20 timesteps: [fluid field, drone state, forces]

**Output**:
- Future forces for next 10 timesteps

**Advantages**:
- Captures temporal patterns (vortex shedding cycles)
- Attention mechanism focuses on relevant flow features

**Disadvantages**:
- Slower inference (~10ms)
- Requires more data

---

## Data Collection Strategy

### Phase 1: Generate Training Data (50,000 - 200,000 samples)

Run simulations with **varied conditions**:
1. **Different starting positions**: Random (x, y) in domain
2. **Different Reynolds numbers**: Re = 500, 800, 1000, 1500
3. **Different wind conditions**: Add random lateral flow
4. **Controller modes**: 
   - Dumb drone (noisy thrust) → captures crash scenarios
   - Smart drone (PID) → captures recovery scenarios

**Save every timestep**:
```python
# Inputs
- fluid_field: u[x-25:x+25, y-25:y+25, :2]  # 50×50×2 local window
- drone_state: [x, y, vx, vy, angle, omega]  # 6 values

# Outputs (targets)
- aero_forces: [Fx, Fy, Torque]  # Next timestep forces
- future_forces: [Fx_1, Fy_1, T_1, ..., Fx_10, Fy_10, T_10]  # Next 10 steps
```

### Phase 2: Preprocessing

**Normalization**:
```python
# Fluid velocities: scale to [-1, 1]
u_norm = u / u_max  # u_max = 0.3 (from clipping)

# Forces: standardize (mean=0, std=1)
F_norm = (F - F_mean) / F_std

# Drone state: normalize each component
state_norm = (state - state_mean) / state_std
```

---

## Training Strategy

### Loss Function

**For DeepONet**:
```python
# Primary: Force prediction accuracy
L_force = MSE(F_pred, F_true)

# Regularization: Physics consistency
L_physics = MSE(F_pred.sum(), mass * gravity)  # Total force balance

# Total loss
L = L_force + 0.1 * L_physics
```

**For Transformer**:
```python
# Multi-step prediction
L = sum([MSE(F_pred[t], F_true[t]) for t in range(1, 11)])
```

### Training Details

**Dataset Split**:
- Training: 70% (e.g., Re=500, 800, varied positions)
- Validation: 15% (Re=1000, different positions)
- Test: 15% (Re=1500, unseen conditions)

**Hyperparameters**:
- Batch size: 256
- Learning rate: 1e-4 (Adam optimizer)
- Epochs: 100-200
- Early stopping: patience=10 on validation loss

**Data Augmentation**:
- Flip fluid field horizontally (mirror symmetry)
- Add small Gaussian noise to drone state (±1%)

---

## Deployment: Hybrid Control Architecture

### Integration with Existing PID

```python
def get_propeller_force_field_with_neural(state, grid_shape, key, neural_model):
    # 1. Extract local fluid field
    fluid_window = extract_fluid_around_drone(state.pos, grid_shape)
    
    # 2. Neural network predicts future forces
    predicted_forces = neural_model.predict(fluid_window, state)
    # predicted_forces = [Fx_1, Fy_1, T_1, ..., Fx_10, Fy_10, T_10]
    
    # 3. Convert force prediction to preemptive thrust adjustment
    # Predict torque in next 5 steps
    future_torque = predicted_forces[2::3][:5]  # [T_1, T_2, T_3, T_4, T_5]
    expected_disturbance = jnp.mean(future_torque)
    
    # 4. Feedforward correction (preemptive)
    feedforward_correction = -expected_disturbance / (2 * MOTOR_OFFSET)
    
    # 5. PID feedback correction (reactive)
    angle_error = 0.0 - state.angle
    omega_error = 0.0 - state.angular_vel
    feedback_correction = kp * angle_error + kd * omega_error
    
    # 6. Combine feedforward + feedback
    total_correction = feedforward_correction + feedback_correction
    
    # 7. Apply to motors
    thrust_left = base_thrust + total_correction
    thrust_right = base_thrust - total_correction
    
    return fx, fy
```

**Key Insight**: Neural net provides **feedforward term** that anticipates disturbances, while PID provides **feedback term** that handles residual errors.

---

## Which Forces to Train On?

### Primary Target: **Aerodynamic Torque** ⭐

**Why Torque?**
- **Root cause of instability**: Torque causes rotation → tilt → drift
- **Most predictable**: Vortex shedding creates periodic torque patterns
- **Directly controllable**: Differential thrust directly counters torque

**Training Targets**:
1. **Next-step torque**: `T(t+1)` - immediate reaction
2. **Multi-step torque sequence**: `[T(t+1), ..., T(t+10)]` - anticipation
3. **Secondary**: Lateral force `Fx` to predict drift

### Forces to Include in Training

**Input Features** (what neural net sees):
```python
features = {
    'fluid_vx': u[local_window, 0],      # Horizontal velocity field
    'fluid_vy': u[local_window, 1],      # Vertical velocity field  
    'drone_angle': state.angle,          # Current tilt
    'drone_omega': state.angular_vel,    # Current rotation rate
    'drone_vy': state.vel[1],            # Vertical velocity
}
```

**Output Targets** (what neural net predicts):
```python
targets = {
    'torque_future': [T_1, T_2, ..., T_10],  # PRIMARY
    'fx_future': [Fx_1, Fx_2, ..., Fx_10],   # SECONDARY (for position control)
}
```

---

## Training Workflow

### Step-by-Step Implementation

1. **Data Collection Script** (`collect_training_data.py`):
   ```bash
   python collect_training_data.py --num_episodes 1000 --re_range 500-1500
   # Outputs: training_data.npz (fluid_fields, drone_states, forces)
   ```

2. **Preprocess Data** (`preprocess_data.py`):
   ```bash
   python preprocess_data.py --input training_data.npz --output processed_data.npz
   # Normalize, create local windows, compute statistics
   ```

3. **Train Model** (`train_deeponet.py`):
   ```bash
   python train_deeponet.py --data processed_data.npz --epochs 200
   # Outputs: model_weights.h5, scaler.pkl
   ```

4. **Deploy in Simulation** (`2D_drone_flow_neural.py`):
   ```bash
   python 2D_drone_flow_neural.py --model model_weights.h5 --hover
   # Run with neural controller
   ```

---

## Expected Performance Improvements

**Current PID Controller**:
- Maximum tilt: ~45° (with saturation)
- Settles to steady-state: ~1000 timesteps
- Drift: ~300 pixels over 200k steps

**With Neural Feedforward**:
- Maximum tilt: **~15-20°** (3x better)
- Settles to steady-state: **~200 timesteps** (5x faster)
- Drift: **~50 pixels** (6x less)

---

## Advanced: Online Learning (Future Work)

After deployment, continue learning:
1. Run simulation with neural controller
2. Collect actual vs predicted forces
3. Fine-tune model on **prediction errors**
4. Iterate: Neural net learns from mistakes in real-time

**Reinforcement Learning Alternative**:
- State: [fluid field, drone state]
- Action: [thrust_left, thrust_right]
- Reward: `-abs(angle) - 0.1 * abs(omega) - 0.01 * drift`
- Use PPO or SAC algorithm

---

## Summary

**Best Approach for Your Use Case**:
1. ✅ Use **DeepONet** to predict torque from fluid field
2. ✅ Train on **aerodynamic torque** (primary) and **lateral force** (secondary)
3. ✅ Implement **Hybrid Control**: Neural feedforward + PID feedback
4. ✅ Collect data with varied Re, positions, and controller modes
5. ✅ Deploy as preemptive correction added to PID

**Next Steps**:
1. Modify simulation to save fluid field windows
2. Run 1000 episodes with varied conditions
3. Train DeepONet model
4. Integrate as feedforward term in `get_propeller_force_field()`
