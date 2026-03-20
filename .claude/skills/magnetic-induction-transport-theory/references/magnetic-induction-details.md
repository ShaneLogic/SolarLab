# Magnetic Induction Transport Theory - Detailed Reference

## Equation References

### Eq. (16.5) - Iterative Solution
The Boltzmann equation with magnetic induction requires solving by iteration when including the term proportional to gradient of δf.

### Eq. (16.6) - Solution Near Conduction Band Bottom
For spherical equi-energy surfaces:
```
v = ℏk/mn
```

### Eq. (16.7) - Abbreviations
Mathematical abbreviations used in Eq. (16.6).

## Distribution Function Expansion

The full distribution function:
```
f = f(r, k, T(r), B, t)
```

This accounts for:
- Position (r)
- Wave vector (k)
- Temperature distribution (T(r))
- Magnetic induction (B)
- Time (t)

## Physical Interpretation

### Small vs Large Magnetic Fields

**Small B fields (perturbation regime)**:
- Electrical conductivity: Observed at small fields, treated as small perturbation
- Thermal conductivity: Observed at small fields, treated as small perturbation
- Only first-order terms considered

**Large B fields**:
- Magnetic effects (Hall effect, magnetoresistance) require RATHER LARGE FIELDS to become observable
- Higher-order terms become significant
- Observable magnetoelectric effects may appear

### Mathematical Consequence

For magnetic induction:
- Term proportional to gradient of δf MUST be taken into consideration
- This is different from pure electrical/thermal conductivity analysis
- Yields Eq. (16.5) which can be solved by iteration

## References

- Haug (1972)
- Madelung (1973)