---
name: driftfusion-analysis-plotting
description: Analyze simulation solutions, calculate physical quantities, and generate plots. Use this skill when processing completed simulations, extracting currents/densities, or visualizing results.
---

# Driftfusion Analysis and Plotting

Access solution structures, calculate physical quantities, and generate visualization plots.

## When to Use
- Analyzing completed simulation results
- Calculating currents, quasi-Fermi levels, or recombination rates
- Visualizing spatial or temporal profiles
- Following standard simulation workflows

## Solution Structure Access

Access solution structure `sol` with components:

1. **`u`**: 3D matrix [time, space, variable]
2. **`x`**: Spatial mesh
3. **`t`**: Time mesh
4. **`par`**: Parameters object

**Variable Order in `u`**:
1. Electrostatic potential
2. Electron density
3. Hole density
4. Cation density (if 1 mobile ionic carrier)
5. Anion density (if 2 mobile ionic carriers)

## Output Analysis

Use `dfana` class to calculate physical quantities:

```matlab
result = dfana.my_calculation(sol)
```

Common calculations:
- Currents (electron, hole, total)
- Quasi-Fermi levels
- Recombination rates
- Carrier densities

**CRITICAL WARNING**: The physical model in Equation Editor is NOT coupled to analysis functions. Users must manually ensure models in `dfana` and `df` are consistent.

## Plotting

Use `dfplot` class for visualization:

```matlab
dfplot.my_plot(sol, optional_arguments)
```

### Variable vs Position
```matlab
dfplot.n(sol, [t1, t2, ... tm])  % Plot at specific time points
```
- If time vector omitted, plots final time point

### Variable vs Time
```matlab
dfplot.J(sol, x_position)  % Plot at specific position
```

### Integrated Variables
```matlab
dfplot.Q(sol, [x1, x2])  % Integrate over spatial range
```

### Generic 2D Plots
```matlab
dfplot.x2d(sol, 'variable_name')
```

## Standard Simulation Workflow

Example: Cyclic Voltammogram

1. **Initialize system**:
   ```matlab
   init_driftfusion
   ```

2. **Create parameters object**:
   ```matlab
   par = pci('path/to/Spiro-OMeTAD_perovskite_TiO2.csv')
   ```

3. **Find equilibrium**:
   ```matlab
   soleq = equilibrate(par)
   ```

4. **Run protocol**:
   ```matlab
   sol = doCV(soleq.ion, 0, 1.2, -0.2, 0, 50e-3)
   % Parameters: input, V_start, V_max, V_min, V_end, scan_rate
   ```

5. **Plot results**:
   ```matlab
   dfplot.JVapp(sol, d_midactive)
   ```

## Common Plot Variables

- `d_midactive`: Position at midpoint of active layer
- `n`, `p`: Electron and hole densities
- `V`: Electrostatic potential
- `J`: Current density
- `Efn`, `Efp`: Quasi-Fermi levels

## Output
- Calculated physical quantities (currents, densities, recombination rates)
- Visual plots of simulation data
- Analysis of device performance metrics