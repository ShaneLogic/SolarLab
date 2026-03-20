---
name: pn-junction-physics-analysis
description: Analyze pn-junction electrostatics, extract device parameters from C-V measurements, and assess model applicability for abrupt, linearly graded, and complex homojunctions. Use when characterizing pn-junctions, estimating field distributions, or determining doping profiles.
---

# PN-Junction Physics Analysis

## Overview
Use this skill to perform electrostatic analysis of pn-junctions, extract doping parameters from capacitance measurements, and determine which physical models apply to specific device configurations.

## When to Use
- Calculating electric field and potential distributions in pn-junctions
- Extracting doping densities and diffusion voltages from C-V data
- Analyzing linearly graded vs abrupt junctions
- Evaluating limitations of simplified depletion models
- Assessing Si homojunction characteristics (wide bandgap effects)
- Considering complex factors like doping gradients, series resistance, high injection

## Equilibrium Electrostatics (Abrupt Junctions)

### 1. Define Doping Profiles
- Acceptor: Na(x) = Na for x < 0, 0 for x ≥ 0
- Donor: Nd(x) = 0 for x < 0, Nd for x ≥ 0

### 2. Depletion Approximation
Assume complete depletion in junction region with widths:
- lp: depletion width in p-region
- ln: depletion width in n-region

### 3. Apply Charge Neutrality
```
Na * lp = Nd * ln
```

### 4. Calculate Electric Field
Maximum field at junction (x=0):
```
F_max = e * Nd * ln / (ε * ε0) = e * Na * lp / (ε * ε0)
```
Field distribution is triangular.

### 5. Calculate Diffusion Potential
```
ψ_D = (kT/e) * ln(Na * Nd / ni²)
```
Potential distribution is parabolic.

### 6. Calculate Depletion Widths
Solve for lp and ln using neutrality condition and diffusion potential.

## Linearly Graded Junctions

For junctions with substantial doping gradient:

### Field Distribution
```
F(x) = (q * a / ε) * (x * W/2 - x²/2)
F_max = (q * a * W²) / (8 * ε)
```

### Barrier Width
```
W = [(12 * ε * (Vd - V)) / (q * a)]^(1/3)
```

### Diffusion Potential
```
Vd = (kT/q) * ln((a * W) / (2 * n_i))
```

## Junction Capacitance and Parameter Extraction

### 1. Capacitance per Unit Area
```
C = dQ/dV = (ε * ε0 * e * Na * Nd) / [(Na + Nd) * (ψn,D - V)]
```

### 2. Linear Plotting for Extraction
```
1/C² = [2 * (Na + Nd) / (ε * ε0 * e * Na * Nd)] * (ψn,D - V)
```

### 3. Extract Diffusion Voltage
- Plot 1/C² versus bias voltage V
- Intercept on voltage axis (where 1/C² = 0) gives ψn,D

### 4. Extract Doping Density (Asymmetric Junctions)
For Na >> Nd:
```
Slope ≈ 2 / (ε * ε0 * e * Nd)
Nd = 2 / (ε * ε0 * e * slope)
```

## Model Applicability Guidelines

### Simplified Model Use Cases
Use simplified box-like space charge model for:
- First estimates of junction fields, barrier heights, capacitance
- Understanding importance of junction recombination
- Qualitative I-V characteristics

### Simplified Model Limitations
DO NOT use for:
- Quantitative agreement with actual devices
- Space charge distribution analysis from capacitance measurements
- Critical evaluation of reverse saturation currents
- Detailed diode curve shape analysis

### Wide Bandgap (Si) Considerations

#### Frozen-in Equilibrium
- Si band gap: 1.16 eV
- When quasi-Fermi level in reverse bias > 1 eV from band edge, consider frozen-in equilibrium
- Minimum minority carrier density ~10² cm⁻³
- Quasi-Fermi approach may overestimate density if ignored

#### Highly Doped Thin Layers
- Thin layer (e.g., 500 Å) near contact with high doping (10¹⁸ cm⁻³) requires surface recombination consideration
- Minority carrier density ~200 cm⁻³ (near frozen-in equilibrium)

#### Lack of Saturation
- Total current may lack saturation due to expanding GR region with bias
- Current increases until junction fills entire bulk
- Field-enhanced diffusion shifts saturation onset to higher bias

## Complex Junction Factors

Consider these factors for non-ideal/complex devices:

### Primary Effects
- Doping gradients
- Inhomogeneous recombination center distribution
- Series resistance contribution
- High injection effects

### Position-Dependent Parameters
Caused by inhomogeneous heavy doping, graded composition, surface treatments:
- Ec(x), Ev(x): band edge distributions
- Eg(x): band gap distribution
- gc(x), gv(x): density-of-states distributions
- μn(x), μp(x): carrier mobilities
- ε(x), κ(x), nk(x): optical constants

## Output
- Electric field profile (triangular or rounded)
- Potential distribution (parabolic)
- Depletion widths (lp, ln or W)
- Diffusion voltage (ψD) from C-V intercept
- Doping density from C-V slope
- Assessment of model applicability
