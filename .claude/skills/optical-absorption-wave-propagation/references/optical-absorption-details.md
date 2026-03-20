# Optical Absorption - Detailed Derivations

## Damped Wave Equation

### Starting Point (Eq. 20.19)
Incorporating complex refractive index: ñ = n_r + iκ

### Wave Solution (Eq. 20.22)
```
E = E₀ exp[i(ωt - (ω/c)n_r x)] exp[-(ω/c)κ x]
```

### Phase Velocity
```
v_phase = c/n_r
```

## Absorption Coefficient Derivation

### Energy Flux (Poynting Vector)
```
S ∝ |E|² ∝ exp(-2(ω/c)κx)
```

### Conventional Expression
```
S ∝ exp(-α₀x)
```

### Equating Exponents
```
α₀x = 2(ω/c)κx
α₀ = 2ωκ/c
```

### Relation to Conductivity (Eq. 20.24)
From Maxwell's equations:
```
α₀ = σ/(n_r c ε₀)
```

## Urbach Tail

### Empirical Formula (Eq. 12.1)
```
α(E) = α₀ exp((E - E₀)/E₀)
```

### Physical Interpretation
- Represents exponential tail of density of states into band gap
- Caused by disorder and potential fluctuations
- Lifshitz tail concept from statistical mechanics

### Parameter Dependence
- α₀ and E₀ are NOT universal constants
- Must be determined experimentally for each material
- Depend on:
  - Semiconductor composition
  - Defect type and concentration
  - Processing conditions
  - Thermal treatment