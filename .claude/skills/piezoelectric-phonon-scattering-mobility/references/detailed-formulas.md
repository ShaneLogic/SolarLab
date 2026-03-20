# Piezoelectric Scattering Mobility Formulas

## Complete Mobility Formula (Eq. 17.16)

Meyer and Polder (1953) derived the mobility expression:

```
μ = (16√2π * ħ² * ε) / (3 * e² * K² * m*^(1/2) * (kT)^(1/2))
```

Where:
- ħ: Reduced Planck constant
- ε: Permittivity
- e: Elementary charge
- K: Electromechanical coupling constant
- m*: Effective mass
- k: Boltzmann constant
- T: Temperature

## Numerical Expression (Eq. 17.17)

For practical calculations:

```
μ = (constant) / (K² * T^(1/2))
```

The constant depends on material-specific parameters.

## Electromechanical Coupling Constant K²

```
K² = W_mechanical / W_total
```

K² can be expressed using piezoelectric and elastic constants:

```
K² = e_pz² / (ε * c_l)
```

Where:
- e_pz: Piezoelectric constant (~10⁻⁵ As/cm²)
- ε: Permittivity
- c_l: Longitudinal elastic constant

The relationship between tension T, stress S, and electric field F:

```
T = c_l * S - e_pz * F
```

## Temperature Dependence Comparison

| Scattering Mechanism | Temperature Scaling |
|---------------------|---------------------|
| Piezoelectric | T^(-1/2) |
| Acoustic deformation potential | T^(-3/2) |
| Ionized impurity | T^(+3/2) |

The T^(-1/2) dependence of piezoelectric scattering makes it intermediate between acoustic deformation potential and ionized impurity scattering.

## Key References

- Meyer, H.J.G., and Polder, D. (1953) - Original derivation
- Seeger, K. (1973) - "Semiconductor Physics"
- Zawadzki, W. (1980/1982) - Reviews on electron transport
- Nag, B.R. (1984) - "Electron Transport in Compound Semiconductors"