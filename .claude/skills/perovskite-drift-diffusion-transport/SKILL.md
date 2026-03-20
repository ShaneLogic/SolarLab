---
name: perovskite-drift-diffusion-transport
description: Model charge carrier transport in perovskite solar cells using drift-diffusion equations. Use when calculating time evolution of electron and hole densities, modeling ion vacancy migration, or solving coupled transport problems in perovskite layers. Covers both dimensional and non-dimensional formulations.
---

# Perovskite Drift-Diffusion Transport

## When to Use
Apply this skill when:
- Calculating the time evolution of electron and hole densities in perovskite absorber layers
- Modeling charge transport in perovskite solar cells
- Simulating ion vacancy migration (anions and cations)
- Solving coupled partial differential equations for carrier transport
- Working with drift-diffusion models in semiconductor physics

## Prerequisites
- Understanding of partial differential equations
- Knowledge of semiconductor statistics
- Familiarity with dimensionless variables (for non-dimensional formulation)

## Core Workflow

### Dimensional Formulation

**Step 1: Write the hole conservation equation**
```
∂p/∂t + (1/q) * (∂j_p/∂x) = G(x) - R(n, p)
```
Where hole current density:
```
j_p = -q * D_p * (∂p/∂x) + V_T * p * (∂φ/∂x)
```

**Step 2: Write the electron conservation equation**
```
∂n/∂t - (1/q) * (∂j_n/∂x) = G(x) - R(n, p)
```
Where electron current density:
```
j_n = q * D_n * (∂n/∂x) + V_T * n * (∂φ/∂x)
```

**Step 3: Identify key parameters**
- q: Elementary charge
- D_p, D_n: Hole and electron diffusivities
- V_T: Thermal voltage (k_B * T / q)
- G(x): Photo-generation rate
- R(n, p): Bulk recombination and thermal generation rate

### Non-Dimensional Formulation

**Step 1: Write anion vacancy conservation**
```
∂P/∂t + ∂FP/∂x = 0
```
Where flux:
```
FP = -∂P/∂x - P*E
```

**Step 2: Write electron conservation**
```
∂n/∂t + ∂jn/∂x = G(x) - R(n, p)
```
Where current:
```
jn = -∂n/∂x + n*E
```

**Step 3: Write hole conservation**
```
∂p/∂t + ∂jp/∂x = G(x) - R(n, p)
```
Where current:
```
jp = -∂p/∂x - p*E
```

**Step 4: Write Poisson's equation**
```
∂E/∂x = (λ^2 / 2) * (P^(-1) + δ(p - n))
```
Where E = -∂φ/∂x

**Step 5: Define charge density components**
The term (P^(-1) + δ(p - n)) represents:
- P^(-1): Contribution from anion vacancies
- -1: Contribution from stationary cation vacancies
- δ*p: Contribution from holes
- -δ*n: Contribution from electrons

## Key Variables

| Variable | Type | Description |
|----------|------|-------------|
| n | Density | Free-electron density |
| p | Density | Hole density |
| P | Concentration | Anion vacancy concentration |
| j_n, jn | Current Density | Electron current density |
| j_p, jp | Current Density | Hole current density |
| φ | Potential | Electric potential |
| E | Field | Electric field (E = -∂φ/∂x) |
| G(x) | Function | Photo-generation rate |
| R(n,p) | Function | Recombination rate |
| λ | Parameter | Ratio of Debye length to layer width |
| δ | Parameter | Ratio of carrier to vacancy concentration |

## Constraints
- Valid within the perovskite absorber layer
- Non-dimensional formulation valid for 0 < x < 1
- Assumes drift-diffusion transport mechanism

## Output
A system of coupled partial differential equations describing:
- Time evolution of charge carrier densities
- Current flow via drift and diffusion
- Electric potential distribution
- Ion vacancy migration (non-dimensional form)

## Next Steps
For detailed derivations, boundary conditions, and numerical solution methods, refer to the references.