# Detailed Physics Formulas

## Beer-Lambert Generation

```matlab
g(x) = ∫ φ₀(Eγ) * (1 - κ) * αabs(Eγ) * e^(-αabs(Eγ) * x) dEγ
```

Where:
- `φ₀`: Incident photon flux density
- `κ`: Reflectance
- `αabs`: Energy-dependent absorption coefficient
- Assumption: One electron-hole pair per photon

## SRH Trap Densities

```matlab
nt = ni * exp((Et - Ei) / (k * T))
pt = ni * exp((Ei - Et) / (k * T))
```

Where:
- `Et`: Trap energy level
- `Ei`: Intrinsic Fermi level
- `k`: Boltzmann constant
- `T`: Temperature

## Interface Decay Constants

```matlab
α = (q * (Ei - EFn)) / (k * T)
β = (q * (EFp - Ei)) / (k * T)
```

Where:
- `EFn`, `EFp`: Electron and hole quasi-Fermi levels
- `q`: Elementary charge

## VSR Consistency Check

```matlab
flux_diff = |Σ Rint - ∫ rvsr dx| / Σ Rint
if flux_diff > RelTol_vsr && Σ Rint > AbsTol_vsr
    warning('VSR approximation may be invalid')
end
```