# Theoretical Justification for Model Simplification

## Why Linear Recombination?
Asymptotic methods require linear recombination to obtain separate analytic expressions for charge carrier concentrations. Non-linear recombination terms make analytic solutions intractable.

## Monomolecular Limit Derivation
The monomolecular hole-dominated recombination is derived from the SRH (Shockley-Read-Hall) law:

```
R_SRH = (np - ni^2) / [τp(n + n1) + τn(p + p1)]
```

When the electron pseudo-lifetime (τn) is much less than the hole pseudo-lifetime (τp), and under appropriate doping conditions, this simplifies to:

```
R ≈ γp
```

## Material Applicability
This approximation is particularly good for:
- **Methylammonium lead tri-iodide (MAPbI3)**: The primary material for which this simplification was validated
- Other perovskite materials with similar carrier dynamics
- Systems where hole transport dominates electron behavior

## Physical Interpretation
The simplification assumes:
1. Electron populations reach quasi-equilibrium rapidly
2. Hole concentration determines the recombination rate
3. Bulk processes dominate over surface effects

## When the Approximation Fails
The monomolecular limit breaks down in scenarios where:
- Electron density n approaches zero (e.g., near depletion regions)
- Both carrier species have comparable pseudo-lifetimes
- Surface recombination becomes dominant
- High injection conditions apply