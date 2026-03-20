---
name: asymmetric-generation-analysis
description: Analyze the effects of asymmetric optical generation rates in pn-junction devices. Use this skill when generation rates differ between device regions (e.g., g1,o ≠ g2,o) in devices with non-uniform illumination or spatially varying optical generation. This skill helps quantify Voc reduction, junction field changes, and interpret non-ideal behavior using the diode quality factor (A-factor).
---

# Asymmetric Generation Analysis

## When to Use
Apply this analysis when:
- Working with pn-junction devices under non-uniform illumination
- Optical generation rates differ between device regions
- Spatially varying optical generation is present
- You need to quantify the impact of asymmetric generation on device performance

## Prerequisites
- Different generation rates in different device regions
- Homogeneous generation within each region (graded generation requires different analysis)

## Analysis Procedure

### 1. Identify Asymmetric Generation Configuration
- Determine generation rates for each region (g1,o, g2,o)
- Calculate the ratio between regions
- Note: A factor of 10 difference is a common asymmetric scenario

### 2. Analyze Junction Field Response
- Observe that the junction field increases with asymmetric generation
- This is caused by widening of the junction
- Compare to symmetric case baseline

### 3. Calculate Open Circuit Voltage Impact
- Measure the actual Voc reduction
- Compare to expected reduction from uniform photon absorption decrease
- Note: Actual reduction is typically larger than expected for asymmetric profiles

### 4. Interpret A-Factor
- Calculate A-factor using: ΔVoc = (AkT/e) ln(g/go)
- A > 1 indicates non-ideal behavior from asymmetric generation profile
- Typical values: A ≈ 1.7 for significant asymmetry

### 5. Compare to Symmetric Case
- Reference symmetric solution for baseline comparison
- Note that asymmetric generation creates asymmetric carrier density profiles

## Key Variables
- **g1,o**: Optical generation rate in region 1 (cm^-3 s^-1)
- **g2,o**: Optical generation rate in region 2 (cm^-3 s^-1)
- **A**: Diode quality factor (dimensionless)

## Expected Results
- Quantified Voc reduction (typically 20-25mV for significant asymmetry)
- Junction field changes
- A-factor value indicating degree of non-ideal behavior

## Constraints
- Assumes homogeneous generation within each region
- Graded generation profiles require different analysis methods
- Results are device-specific and depend on geometry and material properties