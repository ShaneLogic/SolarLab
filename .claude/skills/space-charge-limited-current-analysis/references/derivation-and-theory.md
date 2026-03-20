# Space-Charge-Limited Current: Derivation and Theory

## Physical Basis

The space-charge-limited current (SCLC) is controlled by surplus carriers originating from a highly doped region. The behavior is analogous to current in a vacuum diode:
- Electrons are injected from the cathode (highly doped region)
- Carried to the anode following the electric field
- Limited by space charge near the injecting cathode
- Also referred to as "injected current"

## Key Approximations

For the condition n >> Nd₁:

### 1. Space Charge Approximation

```
ρ(x) ≈ en(x)  [Eq. 25.39]
```

The space charge density is dominated by the injected electron density.

### 2. Poisson Equation Simplification

```
dF/dx = en(x)/(εε₀)  [Eq. 25.40]
```

The Poisson equation becomes independent of doping density.

### 3. Drift Current Dominance

```
jn ≈ jn,Drift = en(x)μnF(x)  [Eq. 25.41]
```

Drift current dominates over diffusion current.

## Derivation Steps

1. Replace n(x) in the space charge approximation with the Poisson equation
2. Obtain the intermediate form:
   ```
   jn = -εε₀μnF(x)(dF/dx)  [Eq. 25.42]
   ```
3. Integrate after separating variables [Eq. 25.43]
4. Arrive at the final current-voltage characteristic [Eq. 25.44]

## Device Requirements

- Thin enough region 1 to have entire low-conducting region swamped with electrons
- Carrier density at injecting boundary sufficiently above bulk carrier density in region 1
- Can consist of homogeneous semiconductor of length L with injecting contact

## Example Parameters (Figure 25.11)

- μn = 100 cm²/Vs
- ε = 10
- Device thickness family parameter: L = 1, 1.2, 1.4, 1.6, 1.8, 2 × 10⁻⁵ cm (curves 1-6)
- Thinner device → steeper current increase with bias

## Key References

- Mott and Gurney (1940)
- Lampert (1956)
- Rose (1978a)
- Schappe et al. (1996)

## Equation Reference

| Equation | Description |
|----------|-------------|
| 25.39 | Space charge approximation |
| 25.40 | Poisson equation (doping independent) |
| 25.41 | Drift current dominance |
| 25.42 | Intermediate form |
| 25.43 | Integration step |
| 25.44 | Final I-V characteristic (nn+ junction) |
| 25.45 | I-V characteristic (homogeneous semiconductor) |