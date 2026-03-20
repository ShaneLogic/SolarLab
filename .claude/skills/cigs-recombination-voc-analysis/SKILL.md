---
name: cigs-recombination-voc-analysis
description: Determine recombination mechanisms and theoretical Voc limits for Cu(InGa)Se2/CdS solar cells using diode equation analysis, ideality factors, and barrier height measurements. Use this when fitting diode curves, determining voltage limitations, or investigating performance bottlenecks.
---

# CIGS Recombination and Voc Analysis

## When to Use
Apply this analysis when:
- Fitting current-voltage (J-V) curves
- Determining open-circuit voltage limitations
- Investigating why Voc is below theoretical limit
- Analyzing recombination mechanisms
- Evaluating device quality

## Prerequisites
- Bandgap energy (Eg) of absorber
- Temperature (T) measurements
- J-V curve data

## Diode Equation
```
J = Jo × [exp(e(V - RsJ) / AkT) - 1] + G(V - RsJ) - Jl
```

Where:
- **Jo**: Diode saturation current
- **Rs**: Series resistance
- **G**: Shunt conductance
- **A**: Ideality factor (key indicator of recombination mechanism)
- **Jl**: Photocurrent density

## Open-Circuit Voltage Limits

### Temperature Extrapolation
- As T → 0: Voc → Eg / e
- For ideality factor 1 < A < 2: Voc(T→0) = Eg/e

### Barrier Height
- Measured barrier height: Φb
- For ideal devices: Φb = Eg

## Recombination Mechanism Identification

### Standard Case: SRH Recombination
- **Dominant mechanism**: Shockley-Read-Hall through deep trap states
- **Location**: Space-charge region of Cu(InGa)Se2 (where p ≈ n)
- **Ideality factor**: A = 1.5 ± 0.3

### High Efficiency Devices
- **Dominant mechanism**: Bulk recombination
- **Ideality factor**: A ≈ 1.1–1.3
- **Implication**: Interface does not limit Voc
- **Barrier height**: Φb = Eg

### Exception: Interfacial Recombination Dominant

**Conditions causing interfacial dominance:**
1. Wide-band-gap devices processed without Cu supply
   - Results in lower Voc
   - Barrier height: Φb < Eg

2. Diffusion barrier restricting Na diffusion from glass substrate
   - Voc reduced by ~120 mV from Eg
   - Na is critical for CIGS passivation

## Analysis Workflow

1. **Measure J-V curves** at multiple temperatures
2. **Extract ideality factor** from temperature-dependent analysis
3. **Determine barrier height** from activation energy
4. **Compare Φb to Eg**: If Φb = Eg, bulk recombination dominates
5. **Interpret ideality factor**:
   - A ≈ 1.5: SRH in space-charge region
   - A ≈ 1.1-1.3: Bulk recombination
6. **Check processing conditions**: Cu supply, Na availability
7. **Identify limiting mechanism**: Interface vs bulk

## Decision Guide

| Ideality Factor | Mechanism | Implication |
|-----------------|-----------|-------------|
| 1.5 ± 0.3 | SRH in space-charge | Standard performance |
| 1.1-1.3 | Bulk recombination | High efficiency possible |
| > 2 | Interfacial recombination | Interface issues, check Cu/Na |

## Expected Result
Determined recombination mechanism and theoretical Voc limit with identification of performance-limiting factors.