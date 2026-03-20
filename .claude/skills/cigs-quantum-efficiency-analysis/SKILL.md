---
name: cigs-quantum-efficiency-analysis
description: Calculate external and internal quantum efficiency for Cu(InGa)Se2 solar cells and analyze optical and collection losses (shading, reflection, absorption, collection efficiency) to determine short-circuit current limitations. Use this when analyzing device performance, calculating Jsc, or investigating current losses.
---

# CIGS Quantum Efficiency Analysis

## When to Use
Apply this analysis when:
- Analyzing Cu(InGa)Se2 solar cell performance
- Calculating short-circuit current (Jsc)
- Investigating current loss mechanisms
- Determining collection efficiency
- Measuring spectral response

## Prerequisites
- Device structure (CdS/ZnO/CIGS)
- Bandgap data for Cu(InGa)Se2
- Measurement capability for QE vs wavelength

## Core Formulas

### External Quantum Efficiency (QEext)
```
QEext(λ,V) = [1 - R(λ)] × [1 - AZnO(λ)] × [1 - ACdS(λ)] × QEint(λ,V)
```

Where:
- **R**: Total reflection including grid shading
- **AZnO**: Absorption in ZnO layer
- **ACdS**: Absorption in CdS layer
- **QEint**: Internal quantum efficiency

### Internal Quantum Efficiency (QEint)
```
QEint(λ,V) = 1 - exp[-α(λ) × (W(V) + Ldiff)]
```

Where:
- **α**: Cu(InGa)Se2 absorption coefficient
- **W**: Space-charge width in Cu(InGa)Se2
- **Ldiff**: Minority-carrier diffusion length
- **Assumption**: All carriers generated in space-charge region are collected at 0V

## Loss Mechanism Analysis

Identify and quantify current losses reducing Jsc:

| Loss Type | Typical Value | Description |
|-----------|---------------|-------------|
| 1. Grid Shading | ~1.7 mA/cm² (4% area) | Front contact blocking light |
| 2. Front Reflection | ~3.8 mA/cm² | Surface reflection losses |
| 3. ZnO Absorption | ~1.8 mA/cm² | Free carrier (>900nm) and band-band (<400nm) |
| 4. CdS Absorption | ~0.8 mA/cm² | Increases below 2.42 eV |
| 5. Incomplete Generation | ~1.9 mA/cm² | Near bandgap in Cu(InGa)Se2 |
| 6. Incomplete Collection | ~0.4 mA/cm² | Recombination in Cu(InGa)Se2 |

## Voltage Bias Effects

### Reverse Bias (-1V)
- Increases space charge layer width
- Increases effective collection length
- Improves QE at longer wavelengths

### Forward Bias
- Reduces collection length
- Affects Fill Factor (FF) and Voc
- Simulates operating conditions

## Analysis Workflow

1. Measure QEext vs wavelength
2. Measure reflection spectrum
3. Calculate QEint using known layer thicknesses and absorption coefficients
4. Identify dominant loss mechanisms from the QE curve shape
5. Quantify current losses for each mechanism
6. Evaluate voltage bias dependence to assess collection efficiency

## Expected Result
Calculated Jsc and QE curve identifying optical and collection losses with quantitative breakdown of each loss mechanism.