# Si Solar Cell I-V Analysis - Detailed Parameters

## Standard Device Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| go | 2·10^20 cm^-3 s^-1 | Optical generation rate |
| Nr1 | 10^17 cm^-3 | Recombination center density (region 1) |
| Nr2 | 10^16 cm^-3 | Recombination center density (region 2) |
| c | 10^-9 cm^-3 s^-1 | Capture coefficient |

## Figure References

### Figure 32.23 - Bias-Dependent Behavior

**Panel (a) and (b):** Carrier density distributions
- Curves 1 and 2 show onset of DRO-range appearance
- Indicates approach to current saturation

**Panel (c):** Recombination rate distribution
- Shows r ≈ go region near 5·10^-4 cm
- Demonstrates mid-bulk inactive region in thick devices

**Panel (d):** Current distribution
- Highly asymmetric with applied bias
- Key indicator of device behavior under bias

**Panel (e):** Hole current near left electrode
- Small shift in distribution near x = 0
- Surface recombination current reduction from 0.14 to 0.07 mA/cm²
- p(d1) reduction from 1.6·10^8 to 8·10^7 cm^-3

### Figure 32.24 - I-V Characteristic

- Computed characteristic very close to ideal
- Ideal curve shifted by 35.8 mA/cm²
- Saturation current jsc ≈ 35.5 mA/cm²

## Reverse Bias Current Cases

| Case | Current (mA/cm²) | Voltage (V) |
|------|------------------|-------------|
| Curve 1 | 35 | 0.365 |
| Curve 2 | 35.5 | 0.1 |

## Optimization Considerations

### Recombination-Related Parameters

- Still insufficiently understood in general
- May become decisive with specific doping configurations
- Performance improvement possible by shifting recombination overshoot into more benign region

### Design Challenges

1. **Geometrical Design:**
   - Electrode placement relative to active region
   - Optical access for carrier generation

2. **Electronic Design:**
   - Doping profile optimization
   - Minority carrier collection efficiency

3. **Balance Requirements:**
   - Electrode separation from photoelectric active region
   - Permit optical carrier generation close to junction
   - Achieve nearly perfect minority carrier collection