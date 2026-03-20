---
name: schottky-approximation-method
description: Calculate electric field and potential distributions in Schottky barrier devices using the Schottky approximation. Apply this method when analyzing metal-semiconductor junctions where the electron density in the space-charge region is much smaller than the donor density (large bulk-to-surface carrier density ratio), enabling analytical solutions for field and potential profiles.
---

# Schottky Approximation Method

## When to Use
- Analyzing Schottky barrier devices with depletion regions
- Electron density in space-charge region << donor density
- Large ratio of bulk-to-surface carrier densities (n₁₀/nc)
- Transition distance xDt is less than barrier width xD

## Core Assumption
The space charge becomes independent of electron density n(x) in substantial fraction of the junction region:

```
ρ(x) = e[Nd - n(x)] ≈ eNd (constant)
```

## Execution Procedure

### 1. Verify Applicability
Confirm the trigger condition:
- Electron density in space-charge region rapidly decreases to values very small compared to donor density
- xDt < xD (transition distance less than barrier width)

### 2. Use Decoupled Governing Equations
The Poisson equation decouples from the transport equation, allowing analytical solutions:

- **Transport equation**: jn = en(x)μnF(x) + μnkT(dn/dx)
- **Poisson equation**: dF/dx = eNd/(εε₀)
- **Potential relation**: F(x) = -dψn/dx

### 3. Calculate Field Distribution
```
F(x) = Fc + (eNd/εε₀)x
```
Characteristics:
- Field decreases linearly with increasing distance from interface
- Fc = maximum field at x = 0 (integration constant)
- For n-type semiconductor: Fc is negative

### 4. Calculate Potential Distribution
```
ψn(x) = ψn,D - Fc·x - (eNd/2εε₀)x²
```
Alternative form using Debye length:
```
ψn(x) = ψn,D[1 - (x/LD)²]
```
Characteristics:
- Decreases parabolically with increasing x
- ψn,D = electron diffusion potential (positive for n-type)

### 5. Determine Barrier Layer Thickness
- Define xD from linear extrapolation where F(xD) = 0
- See references for detailed computation method

### 6. Account for Finite Current Behavior
- Solutions F(x) and ψn(x) have same form
- Integration constants become current-dependent
- Results in parallel shift of curves with changing jn

## Constraints
- Approximation is satisfactory near electrode but deviates at larger distances
- Physically meaningless for x > xD
- Assumes xDt ≈ xD