---
name: impedance-spectroscopy-simulation
description: Simulates impedance spectroscopy for planar perovskite solar cells to study frequency response, degradation mechanisms, and ionic/electronic process interplay. Use when you need to analyze device dynamics, separate ionic from electronic contributions, or investigate degradation pathways through frequency-domain analysis.
---

# Impedance Spectroscopy Simulation

## When to Use This Skill

Use this skill when:
- Analyzing frequency response of perovskite solar cells
- Investigating degradation mechanisms
- Disentangling ionic and electronic processes
- Characterizing dynamic device behavior

## Prerequisites

- Converged steady-state solution must exist
- Device at steady-state operating point

## Constraints

- Perturbation amplitude must be small (< 20 mV) to ensure linearity

## Core Workflow

### 1. Apply Voltage Perturbation

Perturb the device from steady state by applying a voltage oscillation:

```
V(t) = V_DC + V_p * sin(ω * t)
```

Where:
- `V_DC`: Steady-state voltage
- `V_p`: Perturbation amplitude
- `ω`: Angular frequency

### 2. Ensure Linearity

Verify that `V_p` is sufficiently small (< 20 mV) so that the current response is an approximately linear function of applied voltage.

### 3. Measure Current Response

Capture the current response:

```
J(t) = J_DC + J_p * sin(ω * t + θ)
```

Where:
- `J_DC`: Steady-state current
- `J_p`: Amplitude of sinusoidal component
- `θ`: Phase shift relative to voltage

### 4. Calculate Complex Impedance

Compute impedance as:

```
Z(ω) = (Complex Voltage Perturbation) / (Complex Current Response)
```

### 5. Visualize Results

Generate the following visualizations:

- **Nyquist Plot**: Projection onto Real (R) - Imaginary (X) plane showing Z = R + iX
- **Frequency Plots**: Real and imaginary components vs frequency (R-f and X-f)

## Expected Output

Complex impedance spectrum Z(ω) that disentangles ionic and electronic processes, revealing:
- Frequency-dependent behavior
- Ionic vs electronic contributions
- Degradation-related changes in device properties