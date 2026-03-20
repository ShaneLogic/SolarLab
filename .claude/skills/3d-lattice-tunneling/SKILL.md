---
name: 3D Lattice Tunneling
description: Calculate electron transmission probability through planar barriers in 3D crystals by applying momentum conservation principles. Use when analyzing tunneling in three-dimensional lattices, when perpendicular momentum conservation must be accounted for, or when converting 1D tunneling results to 3D scenarios involving crystal structures.
---

# 3D Lattice Tunneling

## When to Use This Skill

Apply this skill when:
- Analyzing electron tunneling through planar barriers in 3D crystal structures
- Converting 1D tunneling transmission probabilities to 3D lattice scenarios
- Accounting for momentum conservation effects on tunneling rates
- Calculating reduction factors due to perpendicular momentum distribution

## Prerequisites

- Understanding of momentum conservation in crystals
- Foundation in 1D tunneling theory
- Planar barrier geometry

## Constraints

- Assumes planar barrier geometry only
- Valid for electrons in 3D crystalline materials

## Procedure

### Step 1: Apply Momentum Conservation Principle

Recognize that in a 3D lattice, momentum components perpendicular to the tunneling direction are conserved. Only the momentum component in the tunneling direction decreases exponentially through the barrier.

### Step 2: Calculate Perpendicular Momentum Energy

Compute the energy associated with momentum perpendicular to tunneling:

```
E_perp = (ℏ² × k_perp²) / (2m)
```

where:
- ℏ = reduced Planck constant
- k_perp = wavevector component perpendicular to tunneling
- m = electron mass

### Step 3: Calculate Mean Tunneling Energy

Determine the mean energy of the tunneling electron:

```
E_bar = e × F × a
```

where:
- e = electron charge
- F = electric field strength
- a = lattice constant

### Step 4: Compute Reduction Factor

Calculate the reduction factor η representing the fraction of electrons with favorable momentum distribution:

```
η = 1 / (1 + (e × F × a) / (4 × E))
```

This integrated result accounts for the distribution of perpendicular momenta across the electron population.

### Step 5: Calculate 3D Transmission Probability

Combine the reduction factor with the 1D transmission probability:

```
Te_3D = η × Te_1D
```

For a flat plate parabolic barrier, the complete expression becomes:

```
Te_3D = [1 / (1 + (e × F × a) / (4 × E))] × exp(-π × (ΔE)² / (2 × ℏ × e × F))
```

where ΔE is the energy difference across the barrier.

## Key Variables

| Variable | Type | Description |
|----------|------|-------------|
| η | Factor | Reduction factor due to momentum distribution |
| E_perp | Energy | Energy of perpendicular momentum component |
| E_bar | Energy | Mean energy of tunneling electron |
| k_perp | Wavenumber | Wavevector perpendicular to tunneling direction |
| Te_3D | Probability | 3D transmission probability |
| Te_1D | Probability | 1D transmission probability |

## Output

Returns the reduced transmission probability Te_3D that accounts for 3D momentum distribution effects in crystal lattice tunneling.

## Reference

Based on calculations by Moll (1964).