# Bandgap Materials Reference for Multijunction a-Si Cells

## Material Bandgap Summary

| Material | Bandgap (eV) | Role | Deposition Notes |
|----------|-------------|------|------------------|
| a-Si:H | ~1.9 | Front absorber | Standard PECVD process |
| a-SiGe:H (high Ge) | ~1.5-1.6 | Middle absorber | Ge content tunes bandgap |
| a-SiGe:H (low Ge) | ~1.3-1.4 | Bottom absorber | Higher Ge fraction |
| μc-Si | ~1.1-1.2 | Bottom absorber | Requires higher current density |

## Deposition Parameters

### a-Si:H Standard Deposition
- Frequency: 13.56 MHz RF discharge
- Substrate temperature: 150-250°C
- Typical deposition rate: 1-5 Å/s

### μc-Si Deposition Differences
- Same 13.56 MHz discharge equipment
- **Higher current density required** compared to a-Si:H
- Higher hydrogen dilution ratio
- May require VHF-PECVD for optimal quality

## Current Matching Guidelines

### Why Current Matching Matters
In a series-connected multijunction cell, the total current is limited by the junction generating the least current. All junctions must be designed to produce equal current for optimal efficiency.

### Thickness Adjustment Strategy
1. Calculate photon absorption in each layer
2. Adjust thickness to equalize photocurrent
3. Typical thickness ranges:
   - Top a-Si:H: 100-300 nm
   - Middle a-SiGe:H: 100-200 nm
   - Bottom μc-Si: 1-3 μm (thicker due to lower absorption coefficient)

## Efficiency Improvement Mechanisms

### Spectrum Splitting
- High-energy photons absorbed in front wide-bandgap layer
- Lower-energy photons transmitted to rear narrow-bandgap layers
- Reduces thermalization losses
- Better utilization of solar spectrum

### Stability Benefits
- a-Si:H exhibits Staebler-Wronski effect (light-induced degradation)
- Thinner a-Si:H layers in multijunction cells reduce degradation
- μc-Si is more stable than a-Si:H
- Multijunction designs can maintain efficiency better over time