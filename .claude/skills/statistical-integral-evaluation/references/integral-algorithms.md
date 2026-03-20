# Statistical Integral Evaluation Algorithms

## Variable Definitions

- ξ: Dimensionless quasi-Fermi level
- s: Gaussian width parameter (measures disorder)
- S(ξ): Statistical integral value
- S^-1: Inverse statistical integral
- g_c,v: Density of states for conduction/valence band
- E_c,v: Reference energy for conduction/valence band

## Pre-computation Algorithm

```matlab
% Generate lookup tables
xi_grid = linspace(xi_min, xi_max, N);
for i = 1:length(xi_grid)
    F(i) = numerical_integral(xi_grid(i));
end
% Trapezium rule integration for high accuracy
```

## PCHIP Interpolation

**Why PCHIP?**
- Ensures C¹ continuity (continuous first derivatives)
- Prevents overshoot/oscillations near steep gradients
- Critical for stability of DAE solver (ode15s)

**MATLAB Implementation:**
```matlab
S_interp = interp1(xi_grid, S_values, xi_input, 'pchip');
```

## Domain Constraints

| Integral Type | Valid Domain | Relative Error |
|---------------|--------------|----------------|
| Fermi-Dirac F(ξ) | ξ < 6 | < 0.1% |
| Gauss-Fermi Gs(ξ) | ξ < 2s, s ≥ 1 | < 0.1% |

## Model Selection Criteria

**Choose Blakemore (s=0) when:**
- Material has no structural disorder
- Conventional crystalline semiconductor
- Hopping transport is not significant

**Choose Gauss-Fermi (s>0) when:**
- Material has Gaussian band broadening
- Significant structural disorder
- Amorphous or disordered organic semiconductors

## Integration with Transport Equations

Statistical integrals appear in:
- Carrier density calculations: n = S(ξ)
- Quasi-Fermi level extraction: ξ = S^-1(n)
- Transport layer boundary conditions
- Density of states formulations