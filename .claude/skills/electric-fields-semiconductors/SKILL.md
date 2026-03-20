---
name: Electric Fields in Semiconductors
description: Calculate and analyze electric field profiles in homogeneous and graded band-gap semiconductors. Use when determining field distributions, analyzing built-in fields from composition gradients, or understanding band bending and carrier drift in semiconductor devices.
---

# Electric Fields in Semiconductors

## When to Use
- Calculating electric field in homogeneous semiconductors
- Analyzing built-in fields in graded composition structures
- Understanding band bending under bias
- Designing graded band-gap devices
- Analyzing drag effects and acousto-electric phenomena

## Prerequisites
- Applied voltage V (if external bias)
- Electrode distance d
- Band structure information
- Distance d >> interatomic spacing

## Case 1: Homogeneous Semiconductors

### Conditions
- Steady-state operation
- No space charge
- Uniform material composition

### Field Calculation
```
E = V / d
```

### Band Representation
```
E = (1/e) × dEc/dx = -dΨ/dx
```

Where:
- Ec = Conduction band energy
- Ψ = Electrostatic potential

### Key Observation
- Conduction band (Ec) and valence band (Ev) slopes are parallel
- Field causes uniform band tilting

## Case 2: Graded Band-Gap Semiconductors

### Conditions
- Varying composition (e.g., ZnSe_ξS_{1-ξ})
- Position-dependent band gap Eg(x)

### Built-in Field Origin
Composition variation causes band gap changes, creating built-in fields even without external bias.

### Asymmetry Factor (AE)
Measures fraction of band gap change occurring in conduction band:
```
AE = ΔEc / ΔEg
```

### Built-in Field Calculations

**Electron Field:**
```
En = -(1/e) × dEc/dx = -(AE/e) × dEg/dx
```

**Hole Field:**
```
Ep = -(1/e) × dEv/dx = -((1-AE)/e) × dEg/dx
```

### Equilibrium Condition
In thermodynamic equilibrium:
- Net current = 0
- Drift and diffusion currents exactly compensate

## Procedure

### For Homogeneous Material
1. Determine applied voltage V
2. Measure electrode separation d
3. Calculate E = V/d
4. Verify no space charge present

### For Graded Material
1. Determine composition profile Eg(x)
2. Find asymmetry factor AE for material system
3. Calculate dEg/dx (band gap gradient)
4. Compute En and Ep using formulas above
5. Verify equilibrium conditions

## Drag Effects

### Electron Drag
- Electrons drifting in field push phonons
- Creates temperature gradient
- Superimposed on Joule heating

### Phonon Drag
- Temperature gradient pushes electrons
- From warm to cold end
- Enhances thermoelectric effects

### Acousto-electric Effects
When drift velocity > sound velocity:
- Coherent phonon waves generated
- Can cause current oscillations

## Example: ZnSe_ξS_{1-ξ} Graded Layer

Given:
- ξ varies from 0 to 1 over 1 μm
- ΔEg = 0.3 eV
- AE ≈ 0.6

Calculate:
```
dEg/dx = 0.3 eV / 10^-4 cm = 3000 eV/cm

En = -(0.6/e) × 3000 = -1800 V/cm
Ep = -(0.4/e) × 3000 = -1200 V/cm
```