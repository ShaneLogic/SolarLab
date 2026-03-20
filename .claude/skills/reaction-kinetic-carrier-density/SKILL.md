---
name: reaction-kinetic-carrier-density
description: Calculate steady-state electron and hole carrier densities from optical generation rates using reaction kinetic balance equations. Use this for determining carrier densities in semiconductors with recombination centers, analyzing photoconductive response, and solving for high-generation-rate limiting behavior.
---

# Reaction Kinetic Balance and Carrier Density

## When to Use
- Determining carrier densities from generation rates in steady state
- Analyzing semiconductors with recombination centers
- Computing photoconductive response
- Solving for carrier densities at high generation rates
- Modeling steady-state carrier populations

## Execution Procedure

### 1. Establish Balance Equations

Set up steady-state equations considering all competing transitions (Eqs 31.9-31.13):

**Note:** Terms in parenthesis (thermal excitation from deep levels) are usually negligible.

### 2. Apply Neutrality Condition

**Charge neutrality (Eq 31.14):**
```
n - p + N_r^- - N_d = 0
```

**Occupied recombination centers (Eq 31.15):**
```
N_r^- = N_r / [1 + (n c_cn)/(p c_cv)]
```

Where:
- n, p = electron and hole densities
- N_r^- = density of negatively charged recombination centers
- N_d = donor density
- c_cn, c_cv = capture coefficients for conduction/valence band

### 3. Simplify Balance Equations

**Electron balance (Eq 31.16):**
```
g_o = (n - n₀) c_cn n_r
```

**Hole balance (Eq 31.17):**
```
g_o = (p - p₀) c_cv n_r
```

Where:
- g_o = optical generation rate
- n₀, p₀ = equilibrium carrier densities
- n_r = density of recombination centers available for capture

### 4. Solve for Carrier Densities

**Simplified case (n_r ≈ N_r):**
```
Δp = g_o τ_p
```

Where minority carrier lifetime:
```
τ_p = 1/(N_r c_cv)
```

**General case:** Use implicit polynomial expression (Eq 31.21) derived by substituting trapped carrier densities and total neutrality.

### 5. High Generation Rate Limit (Bottleneck Equation)

When n ≈ p (very high generation rates):
```
n = p = √[g_o / (c_cr + c_rv)]
```

Carrier density is determined by the **smaller** of the two capture coefficients.

## Key Variables

| Variable | Description |
|----------|-------------|
| n | Electron density |
| p | Hole density |
| N_r | Density of recombination centers |
| c_cn | Capture coefficient for conduction band |
| c_cv | Capture coefficient for valence band |
| g_o | Optical generation rate |
| τ_p | Hole lifetime = 1/(N_r c_cv) |

## Constraints
- Steady-state condition
- Neglects thermal excitation from deep levels in simplified form
- Assumes known capture coefficients