---
name: statistical-integral-evaluation
description: Evaluate Fermi-Dirac, Gauss-Fermi, and Blakemore statistical integrals for carrier density calculations in semiconductor simulations. Use lookup table interpolation with PCHIP for fast, accurate computation. Apply when using non-Boltzmann statistics or modeling disorder in transport layers.
---

# Statistical Integral Evaluation

## When to Use
- Calculating carrier densities from quasi-Fermi levels in transport layers
- Using non-Boltzmann statistics (Fermi-Dirac or Gauss-Fermi distributions)
- Modeling materials with structural disorder (Gaussian band broadening)
- Evaluating Blakemore integrals for materials without hopping transport
- Enabling accurate carrier statistics while maintaining simulation performance

## Statistical Integral Types

### Blakemore Integral (Fermi-Dirac)
Use when Gaussian width parameter s = 0 (no structural disorder).

**Formula:** S(ξ) = ln[1 + exp(ξ)]

**Boltzmann Approximation:** If carriers follow Boltzmann distribution: S(ξ) = exp(ξ)

### Gauss-Fermi Integral
Use when Gaussian width s > 0 (structural disorder present).

## Evaluation Method: Lookup Table Interpolation

### Pre-computation (Simulation Start)

1. Generate lookup tables of dimensionless quasi-Fermi levels vs dimensionless carrier densities
2. Evaluate using numerical integration (trapezium rule) on high-resolution grid
3. Time cost:
   - ~39ms for Fermi-Dirac
   - ~30ms for Gauss-Fermi

### Evaluation (During Simulation)

1. Use interpolation instead of root-finding or direct integration
2. **Interpolation Method:** Piecewise Cubic Hermite Interpolating Polynomials (PCHIP)
3. **MATLAB Function:** `interp1` with 'pchip' option
4. **Rationale:** PCHIP ensures continuous first derivatives, essential for approximated derivatives within DAE system

### Accuracy Specifications
- Fermi-Dirac F(ξ): Relative error < 0.1% for domain {ξ < 6}
- Gauss-Fermi Gs(ξ): Relative error < 0.1% for domain {ξ < 2s, s ≥ 1}

## Physical Interpretation

### Density of States with Blakemore Model
```
ĝ(E) = g_c,v δ(E - E_c,v)
```
where:
- g_c,v: Density of states
- E_c,v: Reference energy
- δ: Dirac delta function

**Physical Meaning:** Material with no structural disorder (hopping transport)

## Performance Impact

- Overhead for 100 mV s^-1 J-V sweep: Only 0.54s
- Enables investigation of alternative statistical models without significant time penalty
- Accuracy maintained at < 0.1% relative error