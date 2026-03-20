---
name: grid-point-concentration
description: Configure grid point distribution for numerical simulation grids when high accuracy is required in layer regions. Use when setting up Tanh grids or similar numerical grids for simulations with boundary layers, especially when the dimensionless Debyelength is known and potential difference affects grid stretching.
---

# Grid Point Concentration Strategy

Apply this strategy when configuring numerical simulation grids (particularly Tanh grids) that require high accuracy in layer regions due to rapid solution variations such as boundary layers.

## When to Use
- Setting up numerical tests requiring high accuracy in layer regions
- Configuring grids for simulations with known dimensionless Debyelength
- Dealing with boundary layers where solution varies rapidly

## Procedure

### 1. Distribute Grid Points
Allocate grid points according to the concentration strategy:
- **Layers**: Concentrate 20% of total grid points within each layer
- **Bulk**: Allocate the remaining 60% of grid points to span the bulk region

### 2. Set Grid Stretching Parameters
Configure the stretching parameter σ based on physical properties:
- For a dimensionless Debyelength of λ = 2.4 × 10^-3, use σ = 5
- Adjust σ based on your specific Debyelength value

### 3. Optimize for Simulation Conditions
Fine-tune parameters accounting for dependencies:
- σ values depend strongly on potential difference
- σ values depend strongly on Debyelength
- Determine optimal σ values through testing for your specific simulation conditions

## Variables
- **X**: Fraction of gridpoints in layers (default: 0.2)
- **σ**: Grid stretching parameter (determined by λ and potential difference)
- **λ**: Dimensionless Debyelength