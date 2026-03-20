---
name: driftfusion-solver-execution
description: Configure and execute the Driftfusion solver, generate time meshes and voltage/light functions, and apply convergence strategies for difficult simulations. Use this skill when running simulations, defining protocols, or troubleshooting convergence issues.
---

# Driftfusion Solver Execution

Configure the core solver, generate time-dependent functions, and execute simulation protocols with convergence strategies.

## When to Use
- Running any Driftfusion simulation protocol
- Defining voltage and light protocols
- Troubleshooting convergence problems
- Executing multi-step experimental protocols

## Core Solver Configuration

Execute master function `df` with device parameters:

1. **Master Function Call**:
   ```matlab
   sol = df(par, voltage_conditions, light_conditions)
   ```

2. **Subfunction `dfpde`** (continuity equations):
   - Uses MATLAB `pdepe` solver form: $c(x,t,u,ux)u_t = x^{-m}(x^mf(x,t,u,ux))_x + s(x,t,u,ux)$
   - Variables in vector `u`:
     1. Electrostatic potential (V)
     2. Electron density (n)
     3. Hole density (p)
     4. Cation density (c)
     5. Anion density (a)
   - Coefficients: `C` (time derivative), `F` (flux), `S` (source/sink)
   - Equations editable via Equation Editor

3. **Subfunction `dfic`** (initial conditions):
   - `'equilibrate'` or `df` with empty input: Use Section 3.8 conditions
   - Otherwise: Use final time point of input solution

4. **Subfunction `dfbc`** (boundary conditions):
   - Coefficients `P` and `Q` passed to `pdepe`
   - Dirichlet: `P` non-zero (defines variable values)
   - Neumann: `Q` non-zero (defines variable flux)
   - Default expressions in Listing 2

## Time Mesh and Function Generation

1. **Time Mesh** (`meshgen_t`):
   - Called at start of code
   - Solver uses adaptive timestep, interpolates to user-defined mesh

2. **Convergence Control**:
   - `tmax` and `MaxStepFactor` strongly influence convergence
   - If problematic: Reduce `tmax` or `MaxStepFactor` and obtain solution in stages

3. **Function Generator** (`fun_gen`):
   - Generates time-dependent algebraic functions for voltage and light
   - Supports two light intensity functions (bias light + pump pulse)
   - Requires coefficients array (elements depend on function type)
   - Examples: Sine wave, square wave

## Protocol Execution Strategy

1. **Define Protocol Function**:
   - Takes input solution (initial conditions)
   - Produces output solution

2. **Create Temporary Parameters**:
   - Duplicate input solution parameters object
   - Write new voltage and light parameters for function generator

3. **Split Complex Protocols**:
   - Divide into intermediate steps to facilitate convergence

## Convergence Strategy for Mobility Differences

**Trigger**: Ionic mobilities separated by many orders from electronic mobilities

1. **Identify Problem**: Check mobility separation

2. **Adjust Ionic Mobilities**:
   - Temporarily increase using `K_a` and `K_c` properties
   - Make similar to electronic carrier mobilities

3. **Run Simulation**:
   - Use appropriate timestep
   - Confirm steady-state reached

4. **Restore Parameters**:
   - Return to original mobility values for final solution

## Output
- Solution structure `sol` containing:
  - `u`: 3D matrix [time, space, variable]
  - `x`: Spatial mesh
  - `t`: Time mesh
  - `par`: Parameters object