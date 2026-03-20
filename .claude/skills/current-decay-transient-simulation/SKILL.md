---
name: current-decay-transient-simulation
description: Simulates and analyzes rapid current decay transients using a tanh voltage profile protocol. Use when modeling current decay scenarios or time-of-flight measurements where capturing fast timescale transients (initial current rise at very short times) is critical for accurate analysis.
---

# Current Decay Transient Simulation

This skill implements a protocol for simulating current decay transients that captures both the long-term asymptotic behavior and the fast timescale dynamics that occur at very short times.

## When to Use

Apply this protocol when:
- Simulating rapid current decay transients in semiconductor or electrochemical cells
- Performing time-of-flight measurements where voltage drops occur
- You need to capture initial current rise phenomena at very short times
- Comparing numerical solutions to asymptotic analytical results

## Prerequisites

- Cell must be in a preconditioned state (transients eliminated from initial bias)
- Simulation environment capable of handling steep voltage profiles

## Procedure

### 1. Preconditioning

Hold the cell at applied bias φ = φ_bi = 40 for a sufficiently long time to eliminate all initial transients before starting the decay protocol.

### 2. Voltage Protocol Configuration

Set up the voltage profile for the decay transient:

- **Initial bias:** φ_bi = 40
- **Final bias:** φ = 0
- **End time:** t_end = 1
- **Steepness parameter:** β = 10^6

Apply the voltage profile:
```
φ(t) = φ_bi × [1 - tanh(β × t) / tanh(β × t_end)]
```

This creates a smooth but rapid decrease in applied bias starting at t=0.

### 3. Simulation Settings

Configure numerical parameters:
- **Number of subintervals:** N = 400
- **Relative tolerance:** RelTol = 10^-6
- **Absolute tolerance:** AbsTol = 10^-8

### 4. Execution

Run the simulation with the configured parameters to compute photocurrent as a function of time.

## Analysis

Compare the numerical photocurrent results to the asymptotic solution:

- **Long times (t → t_end):** Numerical results should agree with asymptotic predictions
- **Short times (t → 0):** Numerical method will capture fast timescale transients (initial rise in current density) that are absent in the asymptotic solution

The key advantage of this protocol is its ability to resolve the fast transient dynamics at very short time scales while maintaining accuracy at longer times.