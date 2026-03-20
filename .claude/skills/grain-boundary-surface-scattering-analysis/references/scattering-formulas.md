# Detailed Formulas for Scattering Analysis

## Grain Boundary Potential Barrier

The space-charge triple layer at grain boundaries creates a potential barrier. From the integrated Poisson equation:

```
V_b = (e × Σ_i × L_D) / (ε × ε_0)
```

Where:
- e = electron charge
- Σ_i = interface surface charge density
- L_D = Debye length = √(ε × ε_0 × kT / (e² × n))
- ε = relative permittivity
- ε_0 = vacuum permittivity
- n = carrier concentration

## Temperature-Dependent Mobility

For thermally activated transport over barriers:

```
μ_b(T) = μ_0 × exp(-E_a / kT)
```

Where E_a = eV_b is the activation energy.

**Extraction method:**
1. Plot ln(μ) vs. 1/T (Arrhenius plot)
2. Slope = -E_a/k
3. E_a gives barrier height directly

## Surface Scattering - Complete Formula

For a thin platelet of thickness d, the mobility ratio is:

```
μ/μ_B = [1 - (1-s)/(1+s) × (λ/d) × (1 - exp(-d/λ))]⁻¹
```

**Limiting cases:**

| Condition | Result |
|-----------|--------|
| d >> λ | μ → μ_B (bulk behavior) |
| d << λ, s = 0 | μ → μ_B × (d/λ) |
| d << λ, s = 1 | μ → μ_B (no reduction) |

## Surface-Induced Relaxation Time

```
τ_s = (d/λ) × τ_B × (1+s)/(1-s)
```

For non-specular scattering (s = 0):
```
τ_s = (d/λ) × τ_B
```

## Graphical Solution

The mobility ratio μ/μ_B as a function of d/λ with s as family parameter:

- s = 1.0: Horizontal line at μ/μ_B = 1 (no reduction)
- s = 0.5: Moderate reduction at small d/λ
- s = 0.1: Significant reduction
- s = 0.0: Maximum reduction

(Reference: Fig. 18.4 in source material, after Sondheimer)
