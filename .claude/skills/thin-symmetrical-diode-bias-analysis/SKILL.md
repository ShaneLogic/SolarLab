---
name: Thin Symmetrical Diode Bias Analysis
description: Analyze carrier density profiles, current distributions, and recombination behavior in thin symmetrical pn-junction devices under forward or reverse bias. Use when modeling photodiodes with homogeneous optical generation, investigating current saturation phenomena, DRO-range effects, or diode quality factor behavior under bias conditions.
---

# Thin Symmetrical Diode Bias Analysis

## When to Use This Skill

Apply this skill when analyzing thin symmetrical pn-junction devices that:
- Have applied bias (forward or reverse)
- Feature neutral surfaces with no surface recombination
- Operate with homogeneous optical generation
- Require understanding of bias-dependent carrier and current behavior

## Prerequisites

Verify these conditions before analysis:
- Thin device approximation applies (thick devices show different behavior)
- Neutral surfaces (no surface recombination)
- Homogeneous generation rate: go = 10^21 cm^-3 s^-1
- Recombination center density: Nr = 10^17 cm^-3

## Device Configuration

1. Confirm thin symmetrical pn-junction geometry
2. Set neutral surface boundary conditions
3. Initialize homogeneous generation rate: go = 10^21 cm^-3 s^-1
4. Set recombination center density: Nr = 10^17 cm^-3

**Key insight**: Space charge, field, and potential distributions are determined by majority carriers only, maintaining the same qualitative shape as without light. Currents at which the junction is pulled open increase by EIGHT orders of magnitude (from dark current ~10^-10 A/cm² to photogenerated current ~10^-2 A/cm²).

## Reverse Bias Analysis

Execute this branch when analyzing devices under reverse bias:

### Recombination Rate Behavior
1. Observe r(x) pulled down with increased reverse bias
2. Calculate net generation rate: U = go - r
3. Verify maximum of r(x) decreases well below go
4. Confirm U(x) ≈ go (approaches homogeneous generation)

### Current Saturation Detection
1. Examine jn(x) or jp(x) distributions for saturation
2. Identify curve straightening (characteristic of saturation)
3. Confirm all minority carriers are pulled across junction

### Minority Carrier Density
- Track decrease below n10,o = goτp = n20,o = goτn = 10^13 cm^-3
- Monitor continued decrease with increasing reverse current

### DRO-Range Identification
The Domain of Reduced Output appears when approaching current saturation:
1. Look for strong drop of both quasi-Fermi levels
2. Identify slowly sloping, rather straight segment of n(x) and p(x)
3. Note typical spatial extent (e.g., between ±0.4 and ±1.8·10^-5 cm)
4. Characterize by: minor current increase with major bias changes

## Forward Bias Analysis

Execute this branch when analyzing devices under forward bias:

### Recombination Rate Behavior
1. Observe r(x) shifted upward with net forward current
2. Note most current generated in junction region

### Low Forward Current (up to ≈15 mA/cm²)
1. Verify go > r in bulk region
2. Identify small gr-current of minority carriers flowing toward junction
3. Note current flows into overshoot region
4. Calculate diode quality factor A > 1 from this behavior

### High Forward Current
1. Observe monotonic current distribution
2. Note bell-shaped r(x) distribution maintained

### Bell-Shaped r(x) Distribution Analysis
1. Confirm step-like slope of jn(x) and jp(x) per: djn/dx = er(x)
2. Track r(x) increase without bound with increased forward bias
3. Observe stretching of jn(x) and jp(x)
4. Note steeper slope in junction region near x=0

## Output

Generate the following results:
- Carrier density profiles n(x) and p(x)
- Current distributions jn(x) and jp(x)
- Recombination rate distribution r(x)
- Net generation rate U(x)
- Diode quality factor A (for forward bias)
- DRO-range characteristics (for reverse bias saturation)

## Key Variables

| Variable | Unit | Description |
|----------|------|-------------|
| U | cm^-3 s^-1 | Net generation rate (U = go - r) |
| go | cm^-3 s^-1 | Optical generation rate |
| r(x) | cm^-3 s^-1 | Recombination rate as function of position |
| A | dimensionless | Diode quality factor |