# Optical Fraction Calculation Details

## Spectral Data Sources

### Solar Illumination Spectrum (Fig. 12.2)
- Standard: AM1.5G (Global tilt)
- Wavelength range: Typically 300-2500 nm
- Units: W/m²/nm or photons/s/m²/nm
- Reference: ASTM G173-03

### Absorption Spectrum
- Material-specific
- Can be measured by:
  - Spectrophotometry (transmittance/reflectance)
  - Ellipsometry
  - Calculated from optical constants

## Integration Methods

### Numerical Integration
For discrete spectral data:
```
Fraction = Σᵢ [I(λᵢ) × A(λᵢ) × Δλ] / Σᵢ [I(λᵢ) × Δλ]
```

Where:
- i: wavelength index
- I(λᵢ): Solar intensity at wavelength λᵢ
- A(λᵢ): Absorption at wavelength λᵢ
- Δλ: Wavelength interval (constant)

### Common Fractions Calculated

1. **Photon Absorption Fraction**:
   - Uses photon flux spectrum
   - Useful for estimating Jsc

2. **Energy Absorption Fraction**:
   - Uses power spectrum (W/m²)
   - Useful for thermal analysis

3. **Wavelength-Weighted Fraction**:
   - Weights by detector response or AM1.5
   - Application-specific

## Error Sources

### Spectral Mismatch
- Different wavelength resolutions
- Interpolation errors
- Extrapolation beyond measured range

### Measurement Uncertainties
- Absorption measurement errors
- Solar spectrum variations
- Calibration uncertainties

## Extensions

### Including Reflection
```
A_effective(λ) = A(λ) × (1 - R(λ))
```
Where R(λ) is reflectance

### Multiple Layers
Sum contributions from each absorber layer

### Band-Limited
Restrict integration to specific wavelength range (e.g., 400-700 nm for visible)