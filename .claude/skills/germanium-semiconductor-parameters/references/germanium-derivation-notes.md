### Parameter Derivation Notes

#### Effective Densities of States
Nc and Nv account for quantum mechanical density of states:
```
Nc = 2 * (2πm*n*kT/h²)^(3/2)
Nv = 2 * (2πm*p*kT/h²)^(3/2)
```

#### Intrinsic Carrier Density
```
ni² = Nc * Nv * exp(-Eg/kT)
```
At T = 300 K for Ge: ni = 2.265 × 10^13 cm⁻³

#### Thermal Velocity
```
vth = √(3kT/m*)
```
v* = 5.7 × 10^6 cm/s is the effective thermal velocity parameter used in transport equations.

#### Einstein Relation
```
Dn = μn * kT/e
Dp = μp * kT/e
```

### Temperature Dependence
These parameters are specified for T = 300 K. For other temperatures:
- ni increases exponentially with T
- μ decreases with T (approximately T^(-3/2))
- Eg decreases slightly with T

### Material Comparison
Ge vs Si at 300 K:
- Eg_Ge = 0.66 eV, Eg_Si = 1.12 eV
- ni_Ge = 2.3 × 10^13 cm⁻³, ni_Si = 1.0 × 10^10 cm⁻³
- μn_Ge = 3900 cm²/Vs, μn_Si = 1350 cm²/Vs