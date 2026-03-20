---
name: tunneling-current-calculation
description: Calculate tunneling current density across tunnel junctions, n+-p+ junctions, and metal-semiconductor contacts using Fowler-Nordheim field dependence. Use when computing quantum tunneling current in semiconductor devices where carriers tunnel through a potential barrier under high electric field conditions.
---

# Tunneling Current Calculation

## When to Use
Use this skill when:
- Calculating current flow across tunnel junctions
- Analyzing n+-p+ junction devices
- Working with metal-semiconductor contacts exhibiting tunneling
- Computing quantum mechanical tunneling current under high electric fields
- Determining critical field for substantial tunneling (> 10^-3 A/cm²)

## Core Formula

The tunneling current density follows the Fowler-Nordheim dependence:

```
j ~ F² × exp(-F₀ / F)
```

Where:
- `j` = tunneling current density (A/cm²)
- `F` = applied electric field (V/cm)
- `F₀` = characteristic field constant

## Calculation Procedure

1. **Define net current as difference between right-going and left-going currents:**
   - j = e × integral[Tₑ × (Nᵥ(E) × fₙ - N꜀(E) × fₚ) dE]
   - Nᵥ(E), N꜀(E) = density of states in valence and conduction bands
   - fₙ, fₚ = Fermi distributions

2. **Account for charge and velocity in k-space:**
   - j = (4πem/h³) × integral[Tₑ × (fₙ - fₚ) dE]

3. **Apply approximations for Va >> kT/e and Va >> E/e:**
   - Use the simplified Fowler-Nordheim form

4. **Calculate characteristic field constant F₀:**
   - F₀ = (4√(2m) × ΔE^(3/2)) / (3ħe)

5. **Compute final current density:**
   - j = A × F² × exp(-F₀ / F)
   - A = e³ / (8πhΔE)

## Critical Field Calculation

Calculate the critical field for substantial tunneling:

```
F_crit = (4√(2m) × ΔE^(3/2)) / (3ħe × ln(A/j))
```

**Note:** Reduced effective mass (mₙ = 0.1m₀) lowers the critical field by approximately a factor of 3.

## Prerequisites
- Transmission probability (Tₑ)
- Density of states in relevant bands
- Fermi distribution functions
- Same isotropic effective mass on both sides of the junction (assumed)

## Key Assumptions
- Isotropic effective mass on both sides of junction
- High voltage conditions: Va >> kT/e and Va >> E/e
- Tunneling barrier height ΔE is well-defined

## Common Applications
- Tunnel diodes
- Zener breakdown in heavily doped p-n junctions
- Schottky barrier tunneling
- Metal-insulator-metal junctions