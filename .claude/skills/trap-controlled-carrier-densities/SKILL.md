---
name: trap-controlled-carrier-densities
description: Calculate electron density in n-type photoconductors with significant trap densities. Use when modeling photoconductors where trap filling effects modify the standard carrier density relations, particularly in low generation regimes where trapped electrons dominate.
---

# Trap-Controlled Carrier Densities

## When to Use
Apply this skill when:
- Modeling n-type photoconductors with significant trap densities
- Analyzing materials where electron traps ($N_t$) and recombination centers ($N_r$) are present
- The generation rate is such that trap filling effects cannot be ignored
- You need to determine electron density ($n$) considering trap-controlled kinetics

## Prerequisites
- Understanding of reaction kinetic equations
- Knowledge of neutrality conditions in semiconductors
- Familiarity with carrier trapping and recombination processes

## Procedure

### 1. Define the Model
Identify the key components of your photoconductor system:
- Electron traps with density $N_t$
- Recombination centers with density $N_r$
- Optical generation rate $g_o$

### 2. Apply Predominant Trapping Assumption
Simplify the neutrality equation under the assumption of predominant carrier trapping:
- Use the approximation $n_t \approx n$ (trapped electron density ≈ free electron density)

### 3. Derive Modified Carrier Density Relation
Substitute $n_r$ (recombination center occupancy) and $p$ (hole density) into the balance equations to obtain the relation between $n$ and $g_o$.

The resulting equation is nonlinear in $n$ and includes a modified bottleneck relation in square brackets.

### 4. Analyze Generation Regimes
Determine which regime applies based on the relative magnitudes of $n$ and $n_t$:

**High Generation Regime** ($n \gg n_t$):
- Equation reverts to standard form
- $n$ rises proportionally to $g_o$

**Low Generation Regime** ($n_t \approx N_t \gg n$):
- Sublinear rise of $n$ with $g_o$
- Use the simplified formula: $n = g_o c_{rv} \frac{N_r c_{cr}}{c_{rv} N_r N_t - g_o}$

## Output
The skill returns the electron density $n$ considering trap filling effects.