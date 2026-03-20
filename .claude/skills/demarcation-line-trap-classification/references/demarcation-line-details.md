# Demarcation Line Details

## Derivation of Demarcation Lines

### Physical Basis

Demarcation lines are defined by the condition where transition rates between bands and trap states are equal. This determines whether a center primarily captures or emits carriers.

### Electron Demarcation Line (EDn)

The electron demarcation line is defined when the emission rate from trap to conduction band equals the capture rate from conduction band to trap:

```
etc = e_tc × p
```

Where:
- etc = Emission rate from trap to conduction band
- e_tc = Capture rate from conduction band to trap
- p = Hole density

From detailed balance considerations, this leads to:

```
EDn = EFn - kT × ln[(sn/sp) × (Nc/Nv) × exp(-(Ec - Ev)/2kT)]
```

### Hole Demarcation Line (EDp)

Similarly, the hole demarcation line is:

```
EDp = EFp + kT × ln[(sp/sn) × (Nv/Nc) × exp(-(Ec - Ev)/2kT)]
```

## Detailed Examples

### Example 1: Neutral Center

Consider a neutral center with:
- sn = 10⁻¹⁶ cm² (initial state)
- After electron capture, becomes negatively charged
- sp = 10⁻¹⁴ cm² (charged state)
- sn/sp = 10⁻²

At room temperature (T = 300 K):
- δi = kT × ln(10⁻²) = 0.026 × (-4.6) ≈ -0.12 eV

This shifts the electron demarcation line downward by 0.12 eV.

### Example 2: Hole Trap

For a hole trap with:
- sn/sp = 100

At room temperature:
- δj = kT × ln(100) = 0.026 × 4.6 ≈ +0.12 eV

This shifts the hole demarcation line upward by 0.12 eV.

### Example 3: n-type Material

In n-type material with narrow Ec - EFn:
- EFn is close to conduction band
- Wide range of electron traps (centers above EDn)
- Narrow range of hole traps (centers below EDp)

### Example 4: p-type Material

In p-type material:
- EFp is close to valence band
- Narrow range of electron traps
- Wide range of hole traps

## Capture Cross-Section Variations

### Typical Ranges

| Center Type | sn (cm²) | sp (cm²) | sn/sp |
|-------------|----------|----------|-------|
| Neutral donor | 10⁻¹⁶ | 10⁻¹⁴ | 10⁻² |
| Neutral acceptor | 10⁻¹⁴ | 10⁻¹⁶ | 10² |
| Deep level | 10⁻¹⁵ | 10⁻¹⁵ | 1 |
| Coulomb attractive | 10⁻¹² | 10⁻¹² | 1 |
| Coulomb repulsive | 10⁻²⁰ | 10⁻²⁰ | 1 |

### Occupancy Dependence

The capture cross-section ratio can change significantly with center occupancy:
- Neutral center: sn and sp may be similar
- After capturing an electron: becomes negatively charged, affecting sp
- After capturing a hole: becomes positively charged, affecting sn

## Temperature Effects

### δ Value Variation

At different temperatures:

| Temperature (K) | kT (eV) | Max δ Variation |
|-----------------|---------|-----------------|
| 77 (liquid N₂) | 0.0066 | ±0.15 eV |
| 300 (room) | 0.026 | ±0.6 eV |
| 600 | 0.052 | ±1.2 eV |

The demarcation lines spread over a wider range at higher temperatures.

## Material-Specific Considerations

### Silicon (Si)
- Band gap: 1.12 eV
- Nc ≈ 2.8 × 10¹⁹ cm⁻³
- Nv ≈ 1.04 × 10¹⁹ cm⁻³

### Gallium Arsenide (GaAs)
- Band gap: 1.42 eV
- Nc ≈ 4.7 × 10¹⁷ cm⁻³
- Nv ≈ 7.0 × 10¹⁸ cm⁻³

### Germanium (Ge)
- Band gap: 0.66 eV
- Nc ≈ 1.04 × 10¹⁹ cm⁻³
- Nv ≈ 6.0 × 10¹⁸ cm⁻³

## Limitations

1. **Steady-state only**: The method assumes steady-state conditions
2. **Single-level approximation**: Assumes single energy level for the center
3. **Neglects tunneling**: Does not account for tunneling effects
4. **Simplified capture model**: Assumes simple capture cross-section model

## Related Equations

Reference equations from source material:
- Eq. (22.28): Rate balance definition
- Eq. (22.29): Transition rate equality
- Eq. (22.30): Electron demarcation line
- Eq. (22.31): Hole demarcation line