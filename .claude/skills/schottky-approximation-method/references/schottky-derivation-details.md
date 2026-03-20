# Schottky Approximation - Detailed Derivation

## Complete Equation Set

### Governing Equations

**Eq. 26.4 - Space Charge:**
```
ρ(x) = e[Nd - n(x)] ≈ eNd (constant)
```
Valid when space charge becomes independent of n(x) in substantial fraction of junction region.

**Eq. 26.5 - Transport Equation:**
```
jn = en(x)μnF(x) + μnkT(dn/dx)
```

**Eq. 26.6 - Poisson Equation:**
```
dF/dx = ρ(x)/(εε₀) = eNd/(εε₀)
```

**Eq. 26.7 - Potential Relation:**
```
F(x) = -dψn/dx
```

### Field and Potential Solutions

**Eq. 26.8 - Field Distribution:**
```
F(x) = Fc + (eNd/εε₀)x
```

Where:
- Fc = maximum field at x = 0 (integration constant)
- For n-type with positive space charge (+eNd): Fc is negative
- Field decreases linearly with increasing x

**Eq. 26.9 - Potential Distribution:**
```
ψn(x) = ψn,D - Fc·x - (eNd/2εε₀)x²
```

Where:
- ψn,D = electron diffusion potential (appropriate for zero current)
- Integration constant for n-type: ψn,D is positive
- ψn decreases parabolically with increasing x

**Eq. 26.10 - Alternative Potential Expression:**
```
ψn(x) = ψn,D[1 - (x/LD)²]
```

**Eq. 26.20 - Debye Length:**
```
LD = √(εε₀kT/e²Nd)
```
LD is the characteristic length for changing ψn(x) and F(x).

### Finite Current Behavior

The solutions F(x) and ψn(x) have exactly the same form because the equations don't depend on jn. However:
- Integration constants become current-dependent
- Results in parallel shift of F(x) and ψn(x) in x with changing jn

### Barrier Layer Thickness

xD is defined from linear extrapolation of F(x) with F(xD) = 0:
- Indicated in Fig. 26.2b
- Computation method detailed in Sect. 26.2

### Example Values

Typical transition distance:
- xDt ≈ xD ≈ 8 × 10⁻⁶ cm

### Validity Range

| Condition | Valid |
|-----------|-------|
| Near electrode | Yes |
| Larger distances | Deviates |
| x > xD | Physically meaningless |
| xDt ≈ xD | Assumed |