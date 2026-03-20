# nn+ Junction Large Doping Step - Detailed Parameters

## Variables

| Variable | Type | Description |
|----------|------|-------------|
| Nd1 | float | Donor density in lightly doped region (cm⁻³) |
| Nd2 | float | Donor density in highly doped region (cm⁻³) |
| jn | float | Current density (mA/cm² or kA) |

## Doping Step Scaling

- **Large doping step**: 10^5 to 10^6 ratio between highly doped (Nd2) and lightly doped (Nd1) regions
- Scale factor relates directly to step size relation
- For comparison: 10^6 vs 10 ratio
- General behavior similar to smaller step-size junctions but with scale factor equal to step size relation

## Carrier Transport Details

### Forward Bias Behavior
- More carriers swept from higher to lower doped region
- Substantial increase in carrier density permits much increased current in region 1
- Current enhancement proportional to doping step ratio

### Current Ratio Relationship
```
Forward/Reverse Current Ratio ∝ Nd2/Nd1
```
- The higher the Nd2/Nd1 ratio, the higher the ratio of forward to reverse current

### Rectification Threshold
```
Rectification Threshold Current ∝ 1/Nd1
```
- The lower Nd1, the lower the current at which rectification becomes noticeable

## Series-Resistance Limitation

### Condition
Current in nn+ junction becomes series-resistance limited when:
- Lightly doped region is too wide
- Limited number of carriers swept into it cannot sufficiently raise average free carrier concentration

### Physical Mechanism
- Wide lightly doped region → carriers spread over larger volume
- Average free carrier concentration remains low
- Conductivity remains limited by intrinsic carrier density
- Current becomes limited by series resistance rather than junction properties

## Practical Device Considerations

### Origin of nn+ Junctions
Most nn-junctions in practical devices result from:
1. Unintentional doping inhomogeneities
2. Intentional boundary layer doping

### Impact on I-V Characteristics
- Influence is generally small
- Significant impact only in:
  - Extreme cases involving high current densities
  - Extremely high doping density ratios

## Figure 25.10 Reference

### Current Density Family Parameters
- Curve 1: 30 mA/cm²
- Curve 2: 0 mA/cm²
- Curve 3: -30 mA/cm²

### Scaling for Lower Step Size
- Use spread of family parameter in kA
- Reduce abscissa scale by factor of 10^5

## Domain Context

**Tags**: semiconductor physics, junction theory, doping

**Importance**: High

**Related Concepts**:
- pn junction theory
- Carrier injection
- Series resistance effects
- Doping profiles
- Rectification behavior