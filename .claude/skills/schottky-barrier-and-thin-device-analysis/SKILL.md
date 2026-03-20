---
name: schottky-barrier-and-thin-device-analysis
description: Solve two-carrier semiconductor models, analyze current transport in Schottky barriers, and evaluate quasi-Fermi level behavior in thin/thick devices with surface recombination. Use when performing numerical integration, analyzing device characteristics, or assessing ambipolar approximation validity.
---

# Schottky Barrier and Thin Device Analysis

## Overview
Use this skill to solve the governing equations for two-carrier semiconductor models, analyze current transport mechanisms, and understand quasi-Fermi level behavior in Schottky barriers and thin devices with surface recombination.

## When to Use
- Performing numerical integration of semiconductor devices
- Analyzing current flow in Schottky barriers
- Evaluating quasi-Fermi level behavior in thin vs thick devices
- Assessing validity of ambipolar diffusion approximation
- Calculating GR currents and their contribution to total current
- Analyzing devices with strong surface recombination

## Two-Carrier Governing Equations

Solve this system of six first-order differential equations:

### 1. Electron Transport
```
jn = q * μn * n * F + q * Dn * (dn/dx)
```
(Drift current + Diffusion current)

### 2. Hole Transport
```
jp = q * μp * p * F - q * Dp * (dp/dx)
```
(Drift current - Diffusion current)

### 3. Electron Continuity
```
(1/q) * (djn/dx) = U
```

### 4. Hole Continuity
```
(1/q) * (djp/dx) = -U
```

### 5. Poisson Equation (Field)
```
dF/dx = -ρ / ε
```

### 6. Charge Density
```
ρ = q * (p - n + Nd - Na)
```

**Note**: Requires six boundary conditions (nb, Fb, ψb, pb, jnb, jpb). Use mixed condition approach and iteration.

## Current Transport in Schottky Barriers

### Main Current Calculation
```
jni = j = nc * e * v*
```
- nc: electron density at interface
- v*: thermal velocity parameter

Example (Ge Schottky barrier):
- nc = 4.48 × 10¹³ cm⁻³
- v* = 5.7 × 10⁶ cm/s
- j(s) = -40.6 A/cm²

### Supporting Field in Bulk
```
F10 = reverse saturation field
```
Example: F10 = 6.55 V/cm

### Divergence-Free Hole Current
```
jpi = -μp * p10 * e * F10
```
Typically > 5 orders smaller than jni.

### Generation-Recombination Current
Calculate by numerical integration of net rate U:
```
Δjgr = ∫U dx
```
- Near electron reverse saturation: Δjgr ≈ 20 μA/cm²
- > 6 orders smaller than jni
- Increases with reverse bias (does not saturate)
- Electron and hole GR currents are complementary

## Quasi-Fermi Level Behavior

### Thin Devices (Strong Surface Recombination)

#### Quasi-Fermi Level Collapse
- Collapse at both boundaries: x=0 and x=d1

#### Near Neutral Contact
- EFn: constant (n is constant)
- EFp(x): decreases in reverse bias, joins EFn at x=d1

#### Impact on Carriers
**Minority Carriers**:
- Substantial changes: p(x), EFp(x), jp(x)

**Majority Carriers**:
- Essentially unchanged

**Exception - DRO Range**:
- Condition: minority carrier density > majority dopant density
- Result: slight reduction of DRO-range width
- Consequence: "slight steepening of characteristics" before saturation

### Thick Devices

#### Spatial Distribution
- Quasi-Fermi levels spread beyond junction region
- Join gradually as electrodes approached

#### Region Distinction (Higher Reverse Bias)
**DRO-region (Depletion Region)**:
- Quasi-Fermi levels and band edges slope parallel
- Characterized by rapid changes

**DO-region (Diffusion Only)**:
- Band edges essentially horizontal
- More gradual quasi-Fermi level changes

## Carrier Density Distribution (Strong Surface Recombination)

### Condition
Applies when n >> p (electron density much greater than hole density)

### Hole Density p(x) Behavior
- Increases in bulk region
- Minimum point shifts further into bulk
- Depth of minimum is reduced (not as deep)
- Reason: increased diffusion current toward right surface

### Electron Density n(x) Behavior
- Remains essentially unchanged
- Reason: stability due to n >> p condition

## Step-like GR Rate Approximation

Use when simplifying GR current calculations:

### 1. Bulk Region (x > xD)
```
U10 = (ni² / (τp * Nd)) - (n10 / τn)
```
For wider slabs, U(x) vanishes for x > Lp.

### 2. Barrier Region (0 < x < xD)
```
Uj = (ni² / (τp * n*)) - (pj* / τn)
```
Where:
```
pj* = ni² / pjD
```
pjD: density at barrier-to-bulk interface

### 3. Bias Influence
- Bias influences boundary density pjD (or p*)
- Changes step-height of U
- Changes changeover width: reverse bias → wider, forward bias → narrower

### 4. Solution Method
- Use standard diffusion equation within each step region
- Approximate integral j_gr by line segments (rectangles) for U10 and Uj

## Ambipolar Diffusion Approximation Limitation

### Condition to Check
Analyzing device wider than barrier width with regions beyond diffusion lengths.

### Validation Procedure
1. Analyze electron density approach: n(x) → n10
2. Deviation: δn = n10 - n(x) = n10 * exp(-x/LD)
3. Slope dn/dx decreases exponentially for x > LD
4. Compare with hole density slope dp/dx (remains nearly constant, ~10¹² cm⁻⁴)
5. Verify van Roosbroek assumption: valid only when dn/dx ~ dp/dx

### Conclusion
Ambipolar approximation is INVALID when:
- dn/dx decreases exponentially
- dp/dx remains nearly constant
- Large difference precludes ambipolar coefficients D* and μ*

## Output
- Solution profiles: n(x), p(x), F(x), ψ(x), jn(x), jp(x)
- Current magnitudes: jni, jpi, Δjgr
- Quasi-Fermi level spatial profiles
- Validity assessment for ambipolar approximation
- Simplified GR rate values (U10, Uj)
