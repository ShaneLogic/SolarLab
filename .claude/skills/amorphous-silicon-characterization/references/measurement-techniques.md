# Measurement Techniques and Theory

## Optical Absorption Measurement

### Methods

1. **Transmission/Reflection**
   - For α > 10^2 cm^-1
   - Standard spectrophotometry

2. **Photothermal Deflection Spectroscopy (PDS)**
   - For α down to 1 cm^-1
   - Sensitive to sub-gap absorption

3. **Constant Photocurrent Method (CPM)**
   - For α down to 0.1 cm^-1
   - Probes defect absorption

## Tauc Plot Theory

### Derivation
For parabolic bands:
```
α(hν) ∝ (hν - Eg)^2 / hν
```

Rearranging:
```
(αhν)^1/2 ∝ (hν - Eg)
```

### Validity Conditions
- Parabolic band edges assumed
- Direct transitions
- No excitonic effects

## Urbach Tail Analysis

### Urbach Rule
```
α(hν) = α0 × exp[(hν - E0) / EU]
```

Where EU = Urbach energy (tail width)

### Physical Origin
- Structural disorder
- Potential fluctuations
- Phonon interactions

## Mobility Edge Theory (Mott)

### Definition
Energy Ec (or Ev) where:
- States above: Extended (delocalized)
- States below: Localized

### Consequences
- Minimum metallic conductivity
- Hopping transport in tails
- Variable range hopping at low T

## Comparison of Methods

| Method | Advantages | Disadvantages |
|--------|------------|---------------|
| Tauc | Standard, widely used | Assumes parabolic bands |
| E04 | Simple, reproducible | Arbitrary definition |
| Electrical | Device relevant | Requires special measurement |

## a-SiGe Alloy Considerations

### Band Gap Variation
```
Eg(a-Si_xGe_{1-x}:H) ≈ 1.7 - 0.7x (eV)
```

### Band Tail Increase
- EC increases with Ge content
- Hole mobility further degraded
- Electron mobility also affected

### Applications
- Long-wavelength detectors
- Tandem solar cells
- Requires careful optimization