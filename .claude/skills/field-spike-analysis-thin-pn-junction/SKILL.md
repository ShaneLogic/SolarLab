---
name: field-spike-analysis-thin-pn-junction
description: Analyze field spike phenomena in thin asymmetrically doped Si pn-junction devices under bias. Use when working with semiconductor device simulations where electron density may exceed acceptor concentration near junction interfaces, or when investigating high-field effects in asymmetric doping configurations.
---

# Field Spike Analysis for Thin Asymmetric PN-Junctions

## When to Use This Skill

Use this skill when:
- Analyzing thin asymmetrically doped Si pn-junction devices
- Device has applied bias conditions
- Investigating field distribution anomalies near junction interfaces
- Checking for potential high-field effects in asymmetric structures

## Prerequisites

- Asymmetric doping configuration in the device
- Applied bias conditions
- Access to electron density n(x) and acceptor concentration Na data

## Procedure

### 1. Verify Device Configuration

Confirm the device is a thin asymmetrically doped Si pn-junction with:
- Reduced width of lowly doped region
- Applied bias conditions established

### 2. Identify Field Spike Presence

Examine the field distribution near the junction interface:
- Check if electron density n(x) exceeds acceptor concentration Na
- Field spike occurs at the location where this condition is met
- Look for sharp gradient in field distribution profile

### 3. Assess Spike Magnitude

- Determine if spike magnitude exceeds 60 kV/cm threshold
- Higher magnitudes indicate potential for significant field-dependent effects
- Document the peak field value for further analysis

### 4. Evaluate Impact on Potential Distribution

Assess the influence on device behavior based on operating conditions:
- **Open circuit conditions**: Area under spike is non-negligible; expect influence on potential
- **Other bias conditions**: Minimal influence on overall potential distribution
- **Band structure effect**: Causes slight steepening of Ec(x) near junction interface

### 5. Issue Alert and Recommendations

If field spike is detected:
- Alert that high-field effects may become significant
- Recommend considering field-dependent phenomena for complete analysis
- Note that basic analysis neglecting high-field effects may be insufficient

## Key Variables

| Variable | Unit | Description |
|----------|------|-------------|
| n(x) | cm⁻³ | Electron density as function of position |
| Na | cm⁻³ | Acceptor concentration |
| Ec(x) | eV | Conduction band edge energy |

## Output

- **Alert**: Identification of field spike presence and magnitude
- **Warning**: Potential high-field effects requiring advanced modeling
- **Recommendation**: Include field-dependent phenomena in complete characterization

## Constraints

- High-field effects are neglected in basic analysis
- Spike magnitude may trigger effects requiring advanced modeling
- Solution curves typically show no unexpected features beyond the field spike