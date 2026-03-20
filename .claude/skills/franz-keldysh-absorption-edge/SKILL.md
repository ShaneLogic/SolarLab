---
name: franz-keldysh-absorption-edge
description: Analyze optical absorption near semiconductor band edges under applied electric fields. Use when studying Franz-Keldysh effect, modulation spectroscopy, or field-dependent absorption behavior with photon energy hν < band gap Eg.
---

# Franz-Keldysh Absorption Edge Analysis

Determine absorption behavior and classify electric field regimes when analyzing optical absorption near semiconductor band edges under an applied electric field.

## When to Use
- Photon energy is near or below the band gap (hν < Eg)
- Electric field is applied to the semiconductor
- Performing modulation spectroscopy studies
- Analyzing oscillations or exponential tails in absorption spectra

## Classification of Field Ranges

### 1. Low-Field Range
- **Criterion**: Modulation of reflection ΔR/R < 10^-3
- **Use case**: Line-shape studies
- **Analysis method**: Third-derivative analysis

### 2. Medium-Field Range
- **Characteristic**: Electron acceleration predominates
- **Effect**: Determines line-shape broadening

### 3. High-Field Range
- **Additional effects**: Band-edge changes due to Stark effect
- **Analysis required**: Must account for Stark effect contributions

## Key Calculations

### Oscillation Behavior
- **Period**: ∥ F^(-1/3) where F is electric field
- **Amplitude**: Increases with electric field strength

### Exponential Absorption Tail (hν < Eg)
```
α(hν) ∥ exp[ -4/3 * (Eg - hν)^(3/2) / (eF * ħ * sqrt(2μ)) ]
```

This exponential extension indicates typical Franz-Keldysh effect allowing absorption below the band gap.

## Output
- Classification of field range (low, medium, high)
- Absorption coefficient behavior (oscillations or exponential tail)
- Quantitative relationships between field strength and absorption characteristics