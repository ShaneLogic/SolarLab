---
name: dae-solver-configuration-matlab
description: Configure MATLAB's ode15s solver for time evolution simulation of discretized DAE systems. Use when you have a discretized system of equations from finite element or finite difference methods and need to perform time-dependent simulations with adaptive step size control and error tolerance specification.
---

# DAE Solver Configuration in MATLAB

This skill configures MATLAB's ode15s solver for time evolution simulations of discretized semiconductor device models.

## When to Use
- You have a discretized system of DAE equations
- You need to perform time evolution simulation
- You're working in the MATLAB environment
- You need adaptive step size and order control
- Error tolerances must be specified for accuracy control

## Prerequisites
- Discretized system of equations available
- State variables identified (e.g., P, φ, n, p)

## Solver Configuration Procedure

### 1. Solver Selection

- **Tool**: MATLAB's `ode15s`
- **Method**: Based on Numerical Differentiation Formulae (NDF) of orders 1-5, which are related to Backward Differentiation Formulae (BDF)
- **Key Features**:
  - Variable step size to meet error tolerances
  - Variable order (1-5) for efficiency
  - Suitable for stiff DAE systems

### 2. System Assembly

**Goal**: Minimize computational cost by eliminating superfluous variables.

Eliminate the following variables from the state vector:
- FP (flux-related variable)
- jn (electron current density)
- jp (hole current density)
- E (electric field)

### 3. State Vector Construction

The state vector **u(t)** is a column vector of length **4N + 4**:

```
u(t) = [ P(t)^T, φ(t)^T, n(t)^T, p(t)^T ]^T
```

Where:
- **P(t)**: Column vector of anion vacancy concentrations at N+1 grid points
- **φ(t)**: Column vector of electric potential at N+1 grid points
- **n(t)**: Column vector of electron densities at N+1 grid points
- **p(t)**: Column vector of hole densities at N+1 grid points

### 4. Problem Formulation

Write the system in the standard DAE form:

```
M * du/dt = f(t, u)
```

or:

```
F(t, u, du/dt) = 0
```

Ensure the form matches what `ode15s` expects for your specific problem type.

## Required Parameters

- **Relative tolerance**: Specify for accuracy control (e.g., `RelTol`)
- **Absolute tolerance**: Specify per variable or as a scalar (e.g., `AbsTol`)
- **Time span**: [t0, tf] for the simulation

## Output

A configured MATLAB solver setup ready for time integration via `ode15s`.