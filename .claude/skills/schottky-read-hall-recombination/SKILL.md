---
name: schottky-read-hall-recombination
description: Calculate the net recombination rate (U) through deep-level centers using the Schottky-Read-Hall (SRH) model. Use this for analyzing recombination via deep centers in semiconductors, computing carrier lifetimes, and modeling recombination traffic in devices like solar cells, Schottky barriers, and p-n junctions.
---

# Schottky-Read-Hall Net Recombination Traffic

## When to Use
- Analyzing recombination via deep centers in semiconductors
- Computing net recombination/generation rates
- Modeling recombination in Schottky barriers and p-n junctions
- Determining carrier lifetimes from defect parameters
- Analyzing solar cell efficiency losses

## Execution Procedure

### 1. Calculate Net Recombination Traffic U

**General SRH formula (Eq. 27.29):**
```
U = (n*p - n_i²) / [τ_p₀ (n + n₁) + τ_n₀ (p + p₁)]
```

Where:
- U = net recombination rate (positive for recombination, negative for generation)
- n, p = electron and hole densities
- n_i = intrinsic carrier density
- τ_n₀, τ_p₀ = electron and hole lifetime parameters
- n₁, p₁ = auxiliary densities (see step 2)

### 2. Determine Auxiliary Densities n₁ and p₁

Based on recombination center energy level (E_t) and intrinsic level (E_i):

**Eq. 27.31:**
```
n₁ = n_i * exp((E_t - E_i) / (kT))
```

**Eq. 27.32:**
```
p₁ = n_i * exp((E_i - E_t) / (kT))
```

### 3. Understand Sequential Nature

Physical interpretation:
- An electron from conduction band AND a hole from valence band must BOTH find their way to the recombination center
- The equation structure is of type (1/n + 1/p)⁻¹
- Both carrier types must be captured for complete recombination

### 4. Apply Simplified Relation (Equal Capture Coefficients)

If capture coefficients are equal (c_n = c_p = c):

**Eq. 27.33:**
```
U = c * N_r * (n*p - n_i²) / [n + p + 2n_i cosh((E_t - E_i)/(kT))]
```

## Key Variables

| Variable | Description |
|----------|-------------|
| U | Net recombination traffic (cm⁻³s⁻¹) |
| n, p | Electron and hole densities (cm⁻³) |
| n_i | Intrinsic carrier density (cm⁻³) |
| E_t | Energy level of recombination center (eV) |
| E_i | Intrinsic energy level (eV) |
| τ_n₀ | Electron lifetime parameter (s) |
| τ_p₀ | Hole lifetime parameter (s) |
| k | Boltzmann constant |
| T | Temperature (K) |

## Constraints
- Applies to deeper centers that may become recombination centers
- Non-degenerate semiconductor assumption
- Steady-state conditions

## Physical Interpretation

- **n₁**: Electron density when EFn = E_t
- **p₁**: Hole density when EFp = E_t
- When E_t = E_i (mid-gap): n₁ = p₁ = n_i
- Maximum recombination occurs when center is near mid-gap