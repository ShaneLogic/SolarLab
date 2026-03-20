# Quasi-Fermi Levels - Detailed Formulations

## Complete Equation Set

### Quasi-Fermi Level Definitions

**Electron Density (Eq 27.21):**
```
n = Nc / [1 + exp((Ec - EFn)/kT)]
```

**Hole Density (Eq 27.22):**
```
p = Nv / [1 + exp((EFp - Ev)/kT)]
```

Where:
- Nc, Nv = Effective density of states for conduction/valence bands
- Ec, Ev = Conduction/valence band edges
- k = Boltzmann constant
- T = Temperature
- EFn, EFp = Quasi-Fermi levels for electrons/holes

### Demarcation Line Equations

**Electron Demarcation Line (Eq 27.25):**
```
Ec - EDn = Ev - EFp + δi
```

**Correction term δi (Eq 27.26):**
```
δi = kT ln(s_ni√(m*n*)/s_pi√(m*p*))
```

**Hole Demarcation Line (Eq 27.27):**
```
EFp - Ev = Ec - EFn + δj
```

**Correction term δj:**
```
δj = kT ln(s_pj√(m*p*)/s_nj√(m*n*))
```

Where:
- s_ni, s_pi = Capture cross-sections for electrons/holes
- m*n*, m*p* = Effective masses for electrons/holes

### Physical Interpretation of Centers

| Center Type | Location | Predominant Transitions |
|-------------|----------|-------------------------|
| Electron traps | Close to conduction band | Communicate with conduction band only |
| Hole traps | Close to valence band | Communicate with valence band only |
| Recombination centers | Near middle of bandgap | Communicate between both bands |

### Capture Cross-Section Values

| Center Type | Cross-Section Range (cm²) |
|-------------|--------------------------|
| General range | 10⁻¹³ to 10⁻²² |
| Coulomb-repulsive | 10⁻²⁰ to 10⁻²² |
| Tightly bound | ≤ 10⁻¹⁸ |

### Intrinsic Carrier Density

**Mass action law at equilibrium (Eq 27.28):**
```
n₀p₀ = NvNc exp[-(Ec - Ev)/kT] = nᵢ²
```

Where nᵢ = intrinsic carrier density (constant for given bandgap and temperature)

### Position-Dependent Behavior

Critical observation: As one moves from electrode boundary into bulk:
- Demarcation lines EDn and EDp shift with respect to band edges
- Different regions involve different recombination centers
- Energy range of recombination centers changes significantly throughout space charge layers

This invalidates single-effective-center approximations in many device models.

### Thermal Equilibrium Condition

When no external excitation:
```
EFn = EFp = EF
EDn = EDp = ED
```
Result: No recombination center range exists in equilibrium.