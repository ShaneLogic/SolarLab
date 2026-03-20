# Magneto-resistance Analysis - Detailed Reference

## Historical Note

Discovered by W. Thomson in 1856.

## Physical Origin

- Higher magnetic induction → cannot ignore second-order terms (∝ B²)
- These terms cause REDUCTION in conductivity with increased magnetic induction
- Hall field compensates only for deflection of electrons with AVERAGE velocity
- Slower or faster electrons are more or less deflected
- Results in less favorable path average for carrier conductivity
- **IMPORTANT**: Scattering itself is NOT influenced by magnetic induction
- Yields information about anisotropy of effective mass (Glicksman 1958)

## Equation of Motion Method (Eq. 16.36)

Based on Brooks (1955), Seeger (1973). More suitable for quantitative evaluation than direct Boltzmann equation.

### With B = (0, 0, Bz) - Two Components (Eq. 16.37)
```
ωc = cyclotron frequency
```

### Complex Plane Representation (Eqs. 16.38-16.39)

### Integration and Drift Velocity (Eq. 16.40)
After multiplying by exp(iωct), shows oscillatory behavior.
Scattering interferes so only fraction of cycle completed for ωcτm < 1.

### Averaged Drift Velocity (Eq. 16.41)
Considering distribution of relaxation times.
First term (v0) drops out when averaging over all angles.

### Velocity Components (Eq. 16.42)
Separated real and imaginary parts of v and F.

### Current Densities (Eqs. 16.43-16.44)
```
j = envD
```
in z and y directions. Must average velocities.

## Magneto-resistance Coefficient (Eqs. 16.45-16.46)

For ωcτm ≲ 1 (generally fulfilled):
1. Neglect frequency dependence in denominators
2. With jy = 0, eliminate Fy from Eqs. (16.43) and (16.44)
3. Expression contains second-order term causing decrease of jz with increasing B

Final form:
```
jx = σFx{1 − f(Bx²)}
```

With ρ = 1/σ and e⟨τm⟩/mn = μn:
- Magneto-resistance coefficient essentially equals μ²n
- Except for term containing relaxation-time averages
- This term: Numerical factor depending on scattering mechanism
- Range: 0.38 to 2.15 (Seeger 1973)

## Special Cases

### Two-Carrier Case
- Straightforward (McKelvey 1966)
- Additive for both carriers
- Even if carriers are of opposite sign

### Non-spherical Equi-energy Surfaces
- Rather involved analysis
- Summarized by Conwell (1982)
- See also Beer (1963)

## Example: p-type Ge (Fig. 16.2)

Two carrier types:
- Light holes (pl, μl)
- Heavy holes (ph, μh)

Shows observed vs calculated magneto-resistance at 205K:
- Single carrier heavy hole model (dashed curve)
- Observed data (solid curve)

## Corbino Disk Enhancement (Eq. 16.48)

```
Δρ/ρ = (Δρ/ρ)f × [1 + (μH × B)²]
```

Where:
- μH is the Hall mobility
- (Δρ/ρ)f is the magneto-resistance change in a filament-type sample
- B is the magnetic induction

### Geometry Comparison

| Geometry | Magneto-resistance |
|----------|-------------------|
| Thin filament | Smallest effect |
| Corbino disk | Maximized effect (no surface charge compensation) |

## References

- Brooks (1955)
- Seeger (1973)
- Glicksman (1958)
- McKelvey (1966)
- Conwell (1982)
- Beer (1963)