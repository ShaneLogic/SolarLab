# Hall Effect Analysis - Detailed Reference

## Experimental Setup (Fig. 16.1)

- Two-dimensional sample (platelet shape)
- Electric field: F = (Fx, 0, 0)
- Magnetic induction: B = (0, 0, Bz)
- B-induced current initially in y-direction causes surface charging
- Polarization field develops until jy = 0

## Current Density Components

### Eqs. (16.26-16.28)
jx and jy expressed in terms of magneto-conductivity tensor.

### Magneto-conductivity Tensor Components
- Eq. (16.29): First component
- Eq. (16.30): Second component

## Key Equations

### Hall Voltage (Eq. 16.31)
Derived from Eq. (16.28) with jy = 0.

### Hall Angle (Eq. 16.32)
```
θH = arctan(Fy/Fx)
```

### Hall Constant (Eq. 16.33)
```
RH = VH/(jx × Bz)
RH ≈ 1/(en)
```
Numerical factor depends on scattering mechanism (order of 1).

### Hall Constant for Ellipsoidal Surfaces (Eq. 16.34)
More complex expression accounting for anisotropic effective mass (Herring 1955).

### Two-Carrier Hall Constant (Eq. 16.35)
For compensated semiconductors or when two carrier types present:
```
RH = (e1μ1²n1 + e2μ2²n2) / [e(μ1n1 + μ2n2)²]
```
Where (e1, e2) = (−e, +e) for electrons and holes respectively.

## Scattering Mechanisms and Mobility Ratios

| Scattering Type | μH/μ Ratio |
|----------------|------------|
| Acoustic mode | 3π/8 ≈ 1.18 |
| Ionized impurity | 1.7 |
| High defect density/T | 1 |

Reference: Mansfield (1956)

## Carrier Type Determination

- **NEGATIVE Hall constant** → n-type conduction (electrons majority)
- **POSITIVE Hall constant** → p-type conduction (holes majority)

Signs of e and μ carried in accordance with sign convention (Sect. 26.2.1).

## References

- Mansfield (1956)
- Herring (1955)
- Section 26.2.1 for sign conventions