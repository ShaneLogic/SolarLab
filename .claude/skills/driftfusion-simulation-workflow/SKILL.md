---
name: driftfusion-simulation-workflow
description: Execute core Driftfusion simulation workflows including device structure rebuilding, equilibrium solution calculation, and parallel parameter exploration for mixed ionic-electronic conducting devices.
---

# Driftfusion Simulation Workflow

Use this skill when you need to:
- Rebuild device structures after modifying layer properties
- Calculate equilibrium solutions for devices
- Run multiple simulations with varying parameters in parallel
- Set up parameter sweeps and optimization studies

## Device Structure and Mesh Rebuilding

**When to use:** After modifying device properties (e.g., layer widths) in a user-defined script

**Procedure:**
1. Identify that device properties have been modified
2. Execute the `refresh_device` function:
   ```matlab
   [dev, dev_sub, x, x_sub] = refresh_device(dev, dev_sub, x, x_sub, params, params_sub);
   ```
3. Store the returned updated structures

**Note:** This is not performed automatically to maintain performance when non-device-related parameters change.

**Variables:**
- `dev`: Main device structure
- `dev_sub`: Sub-device structure
- `x`: Main spatial mesh
- `x_sub`: Sub spatial mesh
- `params`: Main parameter set
- `params_sub`: Sub parameter set

## Equilibrium Solution Calculation

**When to use:** When device parameters object exists and equilibrium state is required

**Procedure:**
1. Execute the `equilibrate` function with device parameters object:
   ```matlab
   soleq = equilibrate(par);
   ```
2. The function uses initial conditions described in Section 3.8
3. Results stored in output structure `soleq` with two solutions:
   - `soleq.el`: Only electronic carriers mobile at equilibrium
   - `soleq.ion`: Electronic and ionic carriers mobile at equilibrium

**Purpose:** Allows comparison between devices with and without mobile ionic charge.

## Parallel Parameter Exploration

**When to use:** Running multiple simulations with varying parameters

**Prerequisites:**
- MATLAB Parallel Computing Toolbox installed
- Driftfusion `explore` class loaded
- Parallel pool available

**Procedure:**
1. Initialize the `explore` class
2. Enable parallel pool calculations via Parallel Computing Toolbox
3. Configure parameter space (e.g., active layer thickness vs light intensity)
4. Execute simulation runs in parallel
5. Use embedded plotting tools to visualize outputs

**Example:** See `explore_script` for active layer thickness vs light intensity exploration.