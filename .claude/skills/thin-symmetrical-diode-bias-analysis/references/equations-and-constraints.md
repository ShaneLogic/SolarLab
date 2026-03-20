# Key Equations and Constraints

## Fundamental Relationships

### Net Generation Rate
```
U = go - r
```
Where:
- U: Net generation rate (cm^-3 s^-1)
- go: Optical generation rate (cm^-3 s^-1)
- r: Recombination rate (cm^-3 s^-1)

### Current Gradient Relationship
```
djn/dx = e·r(x)
```
This equation determines the step-like slope of jn(x) and jp(x) profiles.

### Equilibrium Minority Carrier Density
```
n10,o = go·τp = n20,o = go·τn = 10^13 cm^-3
```
Under the specified conditions (go = 10^21 cm^-3 s^-1).

## Device Constraints

### Thin Device Approximation
This analysis applies specifically to thin devices. Thick devices exhibit:
- Different current saturation behavior
- Modified DRO-range characteristics
- Different quasi-Fermi level distributions

### Surface Conditions
Neutral surfaces (no surface recombination) are essential:
- Eliminates surface recombination current components
- Simplifies boundary conditions
- Allows focus on bulk and junction behavior

### Homogeneous Generation
The assumption of homogeneous generation rate:
- go = 10^21 cm^-3 s^-1 throughout the device
- Valid for uniform illumination conditions
- May not apply for focused or patterned illumination

## Recombination Center Density

The specified recombination center density:
- Nr = 10^17 cm^-3
- Determines carrier lifetimes
- Affects the magnitude of recombination rate r(x)

## Bias-Dependent Behavior Summary

| Condition | r(x) Behavior | U(x) Behavior | Current |
|-----------|---------------|---------------|----------|
| Reverse bias | Pulled down | U ≈ go | Saturation |
| Low forward | Slight increase | go > r in bulk | A > 1 |
| High forward | Large increase | r dominates | Monotonic |