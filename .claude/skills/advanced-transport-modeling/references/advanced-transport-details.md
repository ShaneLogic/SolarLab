# Advanced Transport Model Specifications

## Steric Effects Implementation Details

### Standard PNP Limitation
Classical Poisson-Nernst-Planck assumes:
```
P << P_lim
```
This breaks down at high ion densities where lattice site blocking becomes significant.

### Derivation of Modified Flux

Starting from electrochemical potential:
```
μ = μ_0 + k_B T ln(γ P) + qφ
```

With activity coefficient γ accounting for site occupancy:
```
γ = 1/(1 - θ)
```
where θ = P/P_lim is fractional occupancy.

Flux from gradient of μ:
```
F_P = -M P (∂μ/∂x)
```

Substituting and simplifying yields the modified drift term.

## Physical Constraints

**Steric Effects:**
- Only valid for P < P_lim (diverges at P = P_lim)
- NonlinearFP parameter must be set to 'Drift'
- Modifies only the drift term, diffusion term unchanged

**QFL Input:**
- Requires statistical integral S^-1 to be invertible
- Each transport layer can use different input type independently
- No constraint on mixing input types between layers

## Application Examples

### When Steric Effects Matter
- Halide vacancy migration in perovskite at high light intensities
- Ion accumulation near interfaces during bias stress
- Degradation scenarios with significant ion redistribution

### When QFL Input Is Useful
- Experimental data given in energy units (eV)
- Matching workfunction offsets directly
- Designing band alignment at interfaces
- Non-Boltzmann statistics (direct conversion is complex)

## Integration with Drift-Diffusion Model

**Vacancy flux J_P in continuity equation:**
```
∂P/∂t = -(1/q)(∂J_P/∂x) + G_P - R_P
```

Where J_P is the modified flux accounting for steric effects.

**Conservation:** Modified flux still conserves total charge density