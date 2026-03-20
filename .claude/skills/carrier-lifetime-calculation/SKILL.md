---
name: carrier-lifetime-calculation
description: Calculate minority carrier lifetimes using SRH recombination model and understand their spatial distribution in semiconductor devices with pn-junctions. Use when analyzing recombination rates, carrier lifetime in bulk materials, or lifetime variation across device junctions.
---

# Carrier Lifetime Calculation (SRH Model)

## When to Use
- Calculating electron or hole lifetimes in semiconductors
- Analyzing recombination center effects on carrier lifetime
- Understanding lifetime spatial variation in pn-junction devices
- Computing recombination rate U from carrier concentrations

## Prerequisites
- Capture coefficients (cₙ for electrons, cₚ for holes)
- Density of recombination centers (Nᵣ)
- Carrier densities (n, p, n₀, p₀)
- Intrinsic carrier concentration (nᵢ)
- Trap energy level (Eₜ) relative to intrinsic level (Eᵢ)

## Simplified Minority Carrier Lifetimes (Bulk Regions)

**For p-type bulk material:**
```
τₙ₀ = 1 / (cₙ × Nᵣ)
```

**For n-type bulk material:**
```
τₚ₀ = 1 / (cₚ × Nᵣ)
```

These assume p >> n (for electron lifetime) or n >> p (for hole lifetime).

## General Lifetime Expressions

### Polynomial Form for Electrons:
```
τₙ = τₙ₀ × (p + p₁) / (n + p + 2nᵢ × cosh((Eₜ - Eᵢ)/kT))
```

### General Two-Carrier Semiconductor Lifetime:
```
τₙ = (p - p₀) / U
τₚ = (n - n₀) / U
```

### Recombination Rate Relation:
```
U = Δp / τₚ = Δn / τₙ
```

## Spatial Distribution Rules

1. **Thermal equilibrium (U = 0):** Carrier lifetime is infinity

2. **Non-equilibrium:** Lifetime is finite only when deviating from equilibrium

3. **Bulk p/n materials:** Lifetimes are nearly constant (τₙ₀, τₚ₀)

4. **At pn-junction interface:** Lifetime reaches maximum when Fermi level (E_F) coincides with intrinsic level (Eᵢ)

5. **Key principle:** Carrier lifetimes are never the same throughout a semiconducting device including a junction

## Variables
| Symbol | Description | Type |
|--------|-------------|------|
| τₙ₀ | Electron lifetime in p-type bulk | Time |
| τₚ₀ | Hole lifetime in n-type bulk | Time |
| cₙ | Electron capture coefficient | Coefficient |
| cₚ | Hole capture coefficient | Coefficient |
| Nᵣ | Density of recombination centers | Density |

See `references/lifetime-spatial-distribution.md` for detailed derivation and edge cases.