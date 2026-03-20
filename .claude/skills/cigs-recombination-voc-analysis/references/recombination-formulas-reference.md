# Recombination Analysis Reference Formulas

## Diode Saturation Current

The saturation current Jo is related to recombination:

### For SRH Recombination (A ≈ 1.5)
```
Jo ∝ exp(-Eg / 2kT)
```

### For Bulk Recombination (A ≈ 1)
```
Jo ∝ exp(-Eg / kT)
```

### For Interfacial Recombination (A ≥ 2)
```
Jo ∝ exp(-Φb / kT)
```

## Voltage-Temperature Relationship

The Voc vs T plot extrapolation:
- Linear region slope gives ideality factor
- Intercept at T = 0 gives Φb/e
- Compare Φb to Eg to identify recombination location

## Key Equations

### Ideality Factor from Temperature Dependence
```
A = (e/k) × (dVoc/dT)^(-1)
```

### Activation Energy
```
Ea = -k × d(ln Jo)/d(1/T)
```
For bulk-limited: Ea ≈ Eg
For interface-limited: Ea ≈ Φb < Eg
