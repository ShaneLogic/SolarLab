# Asymmetric Generation Rate Effects - Detailed Reference

## Example Parameters
- g1,o = 10^21 cm^-3 s^-1
- g2,o = 10^20 cm^-3 s^-1
- Factor of 10 difference between regions

## Detailed Analysis Results

### Junction Field Response
- Junction field is slightly increased with asymmetric decreased generation
- Caused by widening of the junction (see Fig. 32.17d)
- The widening effect redistributes the electric field across a larger region

### Open Circuit Voltage Impact
- **Measured Voc reduction**: 23mV
- **Expected reduction** (for uniform photon absorption decrease): 15mV
- **Actual reduction is larger** than expected due to asymmetric profile

### A-Factor Interpretation
- **Formula**: ΔVoc = (AkT/e) ln(g/go)
- **Calculated A-factor**: 1.7
- **Interpretation**: A > 1 indicates non-ideal behavior from asymmetric generation profile
- The A-factor quantifies how much the asymmetric generation deviates from ideal behavior

### Comparison to Symmetric Case
- Symmetric solution (s) shown for comparison in Fig. 32.17
- Asymmetric generation creates asymmetric carrier density profiles
- The carrier distribution becomes non-uniform across the device

## Physical Mechanism

When generation rates differ between regions:
1. Higher generation region produces more carriers
2. Lower generation region produces fewer carriers
3. This creates a carrier density gradient
4. The junction adjusts to accommodate this gradient
5. The adjustment results in junction widening and field redistribution
6. These changes lead to additional Voc reduction beyond simple photon counting

## Domain Context
- **Primary domain**: Photovoltaics
- **Related areas**: Optical generation, junction physics
- **Importance**: Medium
- **Figure reference**: Fig. 32.17

## Edge Cases

### Graded Generation
- If generation varies continuously rather than step-wise, this analysis does not apply
- Graded generation requires numerical integration or different analytical methods

### Symmetric Generation
- When g1,o = g2,o, use standard symmetric analysis
- A-factor should be approximately 1 for ideal symmetric devices

### Extreme Asymmetry
- Very large differences (>100x) may require additional considerations
- High asymmetry can lead to other non-ideal effects not captured by simple A-factor model