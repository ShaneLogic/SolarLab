---
name: minority-carrier-diffusion-currents
description: Calculate minority carrier diffusion and generation-recombination (GR) currents in PN junctions, bulk semiconductors, and diodes. Use when analyzing minority carrier transport, calculating diffusion currents from density gradients, or determining thermal GR-currents at boundaries where carrier density deviates from equilibrium.
---

# Minority Carrier Diffusion and GR Currents

## When to Use
- Calculating minority carrier currents in bulk semiconductors or at boundaries
- Analyzing diffusion currents from minority carrier density gradients
- Determining generation-recombination currents when carrier density deviates from equilibrium
- Evaluating thermal excitation effects in PN junctions or diodes

## Core Formulas

### Hole Diffusion Current
```
j_p = -e * D_p * (dp/dx)
```
Where:
- `j_p`: Hole diffusion current density
- `e`: Elementary charge
- `D_p`: Hole diffusion coefficient
- `dp/dx`: Hole density gradient

### Electron Diffusion Current
```
j_n = -e * D_n * (dn/dx)
```
Where:
- `j_n`: Electron diffusion current density
- `D_n`: Electron diffusion coefficient
- `dn/dx`: Electron density gradient

## Generation-Recombination Current Analysis

### Net Generation Rate
When hole density at boundary is lowered below equilibrium (p < p10):
```
U = -(p10 - p) / tau_p
```

### Thermal Generation Rate
```
g_th = p10 / tau_p
```

### Recombination Rate
```
r = p / tau_p
```

## Decision Logic

1. **Calculate diffusion currents** from density gradients using Eq. 28.1 and 28.2
2. **Determine net generation/recombination**:
   - IF p < p10: Net generation (U < 0)
   - IF p > p10: Net recombination (U > 0)
   - IF p = p10: Equilibrium (U = 0)
3. **Account for concurrent changes** in majority carrier current due to GR processes

## Important Notes
- Minority currents are often negligible compared to majority currents in homogeneous bulk unless gradients are large
- Diffusion currents are related to concurrent changes in electron current through generation/recombination
- Minority carrier lifetime (tau_p) is a critical parameter for GR current calculations