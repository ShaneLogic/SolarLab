---
name: quantized-hall-effect-2d-systems
description: Analyze and calculate the quantized Hall effect in two-dimensional electron systems including quantum wells and 2D semiconductor structures. Use when working with systems where thickness < 100Å under high magnetic fields and low temperatures, or when observing Hall conductance plateaus and Landau level quantization.
---

# Quantized Hall Effect in 2D Electron Systems

## When to Use This Skill
Apply this skill when:
- Working with two-dimensional electron gases (thickness < 100Å)
- Analyzing quantum well structures under high magnetic fields
- Investigating Hall conductance quantization and plateaus
- Studying low-temperature quantum transport phenomena
- Determining Landau level filling factors

## Verify System Requirements

Confirm the following conditions before analysis:

1. **Dimensionality check**: Third dimension must be < 100Å
   - Ensures true 2D electron behavior
   - Quantum confinement in z-direction

2. **Environmental conditions**:
   - High magnetic field applied (B field direction orthogonal to 2D plane)
   - Low temperature (typically cryogenic)
   - Satisfies ħω > kT for significant Landau-level splitting

3. **Structure type**: Quantum well or 2D semiconductor system forming 2D electron gas

## Core Analysis Procedure

### Step 1: Analyze Electron Motion

**Without scattering** (ideal conditions):
- Electrons move in circular orbits due to Lorentz force (ev × B)
- Cyclotron motion determines radius, frequency, and energy

**With orthogonal electric field** (Fx in z-direction):
- Electrons drift in y-direction perpendicular to both Bz and Fx
- Motion forms trochoids (flat spirals)
- Constant velocity of orbit centers

### Step 2: Calculate Landau Level Properties

Determine the quantized energy levels:

- **Landau level energy**: E = (nq + 1/2)ħωc
- **Degeneracy per level**: nL = eB/h electrons per unit area
- **Quantization condition**: ħω > kT (thermal energy)

### Step 3: Determine Filling Factor

Calculate how electrons distribute among Landau levels:

1. First nL electrons fill level 0 (nq = 0)
2. Next nL electrons fill level 1 (nq = 1)
3. Continue until all n electrons are distributed

**Filling factor**: ν = n / nL = nh / eB

At T = 0, highest level is partially filled unless n is an integer multiple of nL

### Step 4: Analyze Conductivity Tensor

Evaluate the two key components:

- **σxy**: Hall conductivity (responsible for Hall voltage)
- **σxx**: Longitudinal conductivity (responsible for magneto-resistance)

**With substantial scattering**:
- σxy decreases
- σxx increases

### Step 5: Apply Scattering Constraints

Verify electron scattering behavior:

- Scattering can only occur for electrons in the highest Landau level
- Requires the highest level to be incompletely filled OR kT ≮ ħω
- When constraints are met, electron ensemble follows unperturbed trochoids

## Interpret Results

**Quantized Hall conductance plateaus occur when**:
- Landau levels are completely filled
- Filling factor ν is an integer (ν = 1, 2, 3, ...)
- Longitudinal resistance approaches zero
- Hall resistance is quantized as Rxy = h/νe²

**Key output parameters**:
- Filling factor ν = n / nL
- Landau level spacing ħωc
- Hall conductance quantization value

## Common Scenarios

1. **Electron injection method**: Increasing bias at constant magnetic field
2. **Magnetic field sweep**: Constant bias with increasing B (fewer Landau levels below EF)
3. **Carrier rearrangement**: When Landau level passes over Fermi level (tends to increase EF)

## Validation Checklist

- [ ] System thickness < 100Å confirmed
- [ ] Magnetic field direction verified (orthogonal to 2D plane)
- [ ] Temperature satisfies ħω > kT
- [ ] Landau level degeneracy calculated correctly (nL = eB/h)
- [ ] Filling factor determined from electron density
- [ ] Conductivity tensor components evaluated