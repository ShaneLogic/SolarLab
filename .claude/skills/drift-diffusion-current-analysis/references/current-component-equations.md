# Current Component Equations

## Drift Current Equation

The drift current density for electrons is given by:

```
jn,drift = q * μn * n * E
```

Where:
- `q` = elementary charge (1.602 × 10⁻¹⁹ C)
- `μn` = electron mobility
- `n` = electron concentration
- `E` = electric field = -dψ/dx

In terms of conductivity:

```
σn = q * μn * n
jn,drift = σn * dψ/dx
```

## Diffusion Current Equation

The diffusion current density for electrons is:

```
jn,diffusion = q * Dn * dn/dx
```

Where:
- `Dn` = electron diffusion coefficient
- `dn/dx` = electron concentration gradient

## Einstein Relation

The diffusion coefficient and mobility are related by:

```
Dn = (kT/q) * μn
```

Where:
- `k` = Boltzmann constant (1.381 × 10⁻²³ J/K)
- `T` = absolute temperature
- `kT/q` ≈ 26 mV at room temperature (300K)

## Total Current

The total electron current density is:

```
jn = jn,drift + jn,diffusion
jn = q * μn * n * E + q * Dn * dn/dx
```

## Equation References

- Equation 25.33: Drift current formulation
- Equation 25.34: Diffusion current formulation

## Typical Values

| Parameter | Typical Value |
|-----------|---------------|
| Diffusion current at interface | ~10⁴ A/cm² |
| Thermal voltage (kT/q) at 300K | 26 mV |
| Electron mobility (Si) | ~1400 cm²/V·s |
| Hole mobility (Si) | ~450 cm²/V·s |