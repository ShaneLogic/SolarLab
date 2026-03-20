---
name: ionmonger-performance-optimization
description: Optimize IonMonger simulation performance using analytic Jacobian calculation. Compare IonMonger 1.0 (numerical approximation) with IonMonger 2.0 (analytic Jacobian) for faster simulations. Use when running current-voltage sweeps or impedance spectrum simulations.
---

# IonMonger Performance Optimization

## When to Use
- Running simulations in IonMonger 2.0
- Performing current-voltage sweeps
- Calculating impedance spectra
- Experiencing slow convergence or long simulation times
- Comparing performance across IonMonger versions

## Jacobian Calculation Methods

### IonMonger 1.0 Method

**Approach:** Numerical approximation
- Jacobian (J) of DAE system approximated numerically by ode15s
- Default MATLAB solver behavior
- Performance limited by numerical derivative calculations

### IonMonger 2.0 Method

**Approach:** Analytic calculation
- Jacobian (J) calculated analytically in function `AnJac.m`
- Exact derivatives computed symbolically
- Direct implementation of DAE system derivatives

## Performance Impact

### Speed Improvements
- **Current-Voltage Sweeps:** Significant speed increase
- **Impedance Spectrum Simulation (60 sample frequencies):**
  - IonMonger 1.0: 61 seconds
  - IonMonger 2.0: 31 seconds
  - **Speedup: ~2x (nearly 50% faster)**

### Stability Improvements
- **Convergence:** Improved stability in response to different parameter sets
- **ode15s Behavior:** Changes to Jacobian structure aid solver convergence
- **Robustness:** Better handling of stiff systems across parameter variations

## Implementation Details

**Function Name:** `AnJac.m`

**Purpose:** Provides exact Jacobian matrix for DAE system

**Integration:** Automatically used by IonMonger 2.0 solver

**Variables:**
- J: Jacobian matrix of DAE system
- ode15s: MATLAB solver for stiff differential equations

## When Optimization Applies

The analytic Jacobian optimization is **automatically applied** when:
- Using IonMonger 2.0 or later
- Running standard simulation protocols
- Using the default ode15s solver

No user configuration required—the optimization is built into the software.