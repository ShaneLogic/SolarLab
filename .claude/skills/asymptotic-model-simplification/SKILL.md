---
name: asymptotic-model-simplification
description: Simplify the recombination model configuration to enable direct comparison between numerical simulations and asymptotic analytic solutions. Use this when validating numerical models against analytic results in carrier transport problems, specifically for methylammonium lead tri-iodide or similar materials where monomolecular recombination approximations apply.
---

# Asymptotic Model Simplification

## When to Use This Skill
Use this skill when you need to:
- Compare numerical simulation results to asymptotic analytic solutions
- Validate numerical models against analytic expressions
- Analyze carrier transport in materials with hole-dominated recombination
- Obtain separate analytic expressions for charge carrier concentrations

## Prerequisites
Before applying this skill, ensure you have:
- Realistic parameter estimates from Eq. (18)
- A numerical model with SRH (Shockley-Read-Hall) recombination physics
- Understanding of the physical limits where electron density is not extremely small

## Procedure

### Step 1: Apply Realistic Parameters
Use the realistic parameter estimates provided in Eq. (18) for your model configuration.

### Step 2: Simplify Bulk Recombination
Modify the bulk recombination rate R(n, p) to be linear and monomolecular:

```
R(n, p) = gamma * p
```

where:
- `gamma` = 2.4 (recombination coefficient)
- `p` = local hole concentration

### Step 3: Eliminate Surface Recombination
Set both surface recombination rates to zero:
- `R_l(p) = 0` (left surface)
- `R_r(n) = 0` (right surface)

### Step 4: Verify the Approximation
Confirm that:
- Electron pseudo-lifetime is much less than hole pseudo-lifetime
- Material properties match monomolecular hole-dominated recombination behavior
- Electron density is not vanishingly small in the regions of interest

### Step 5: Compare Results
Run your simplified model and compare directly to the asymptotic analytic solutions.

## Key Variables

| Variable | Type | Description |
|----------|------|-------------|
| R(n, p) | Expression | Bulk recombination rate (simplified) |
| gamma | Float | Recombination coefficient (typically 2.4) |
| p | Variable | Local hole concentration |
| R_l(p) | Expression | Left surface recombination rate |
| R_r(n) | Expression | Right surface recombination rate |

## Limitations
This simplification breaks down where:
- Electron density is very small
- The monomolecular limit of the SRH law does not apply
- Material behavior deviates from hole-dominated recombination