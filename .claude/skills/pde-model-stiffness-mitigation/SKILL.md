---
name: pde-model-stiffness-mitigation
description: Identify and mitigate stiffness in PDE drift-diffusion models for perovskite solar cells. Use when solving drift-diffusion PDE models with realistic parameters, analyzing stiffness sources, or selecting numerical strategies for ion vacancy and charge carrier transport.
---

# PDE Model Stiffness Mitigation

Use this skill when solving the drift-diffusion PDE model with realistic parameters, or when analyzing stiffness in perovskite solar cell simulations.

## Identify Sources of Stiffness

### 1. Small Debye Lengths
- The ratio λ = Debye length / layer width is typically ≈ 10^(-3)
- Causes rapid changes in solution across narrow Debye layers

### 2. Large Potential Differences
- Potential drops across the device cause large gradients
- Typical potential drop of 0.5V across a Debye layer ≈ 20 dimensionless units
- Results in exponential changes in carrier concentrations

### 3. Vastly Different Timescales
- Ratio ν = carrier motion timescale / ion vacancy motion timescale ≈ 10^(-8)
- Electronic carriers move much faster than ion vacancies
- Creates a wide disparity in temporal dynamics

### 4. Concentration Disparities
- Ratio δ = carrier concentration / vacancy concentration ≈ 10^(-2)

## Assess Stiffness Severity

The problem is characterized as "extremely stiff" because:
- Concentrations in Debye layers vary exponentially with potential
- Example: exp(20) ≈ O(10^9) change over width O(10^-3)
- Results in large condition numbers and round-off errors

## Apply Mitigation Strategies

### 1. Non-uniform Meshing
- **Purpose**: Concentrate grid points selectively in the Debye layers
- **Benefit**: Maintains accuracy without prohibitive computational cost
- **Implementation**: Use finer mesh near x=0 and x=b interfaces

### 2. Adaptive Timestep
- **Purpose**: Handle vastly different timescales
- **Benefit**: Allows efficient computation across fast (electronic) and slow (ionic) dynamics
- **Implementation**: Reduce timestep during rapid changes, increase during slow evolution

## Known Limitations

- **Chefun**: Current version fails with realistic parameters
- Direct computation in the relevant regime requires specialized stiff solvers

## Key Variables

- **λ** (lambda): Debye length ratio
- **ν** (nu): Timescale disparity ratio
- **δ** (delta): Concentration ratio
- **Debye_length**: Characteristic length scale of electric screening
- **timescales**: Transport times for ions vs electrons

## Output

Selection of appropriate non-uniform meshing and adaptive timestepping strategies for stable numerical solution.