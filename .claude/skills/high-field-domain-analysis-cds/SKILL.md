---
name: high-field-domain-analysis-cds
description: Analyze formation, detection, behavior, and measurement of high-field domains in CdS crystals and CdS/CdTe solar cells. Use when electric fields reach ~80 kV/cm, when observing domain formation, when measuring electronic properties using domain techniques, or when investigating work function changes via optical excitation.
---

# High-Field Domain Analysis in CdS

## When to Use
- Electric field reaches approximately 80 kV/cm in CdS or CdS/CdTe structures
- Electron current decreases more than linearly with increasing field
- Observing darkened regions in CdS crystals under monochromatic light
- Measuring work function changes via optical excitation
- Investigating carrier density and mobility as function of electric field
- Analyzing stationary vs nonstationary domain behavior

## Domain Formation Threshold

### Critical Conditions
1. **Field threshold**: ~80 kV/cm
2. **Current behavior**: Electron current decreases more than linearly with increasing field
3. **Mandatory formation**: Domain "must appear" once threshold is reached

### Domain Function
- Limits any increase of junction field
- Absorbs additional bias by increasing domain width (not field strength)
- Field limited to approximately 80-85 kV/cm

## Stationary Domain Conditions

Verify all four necessary conditions:

1. **Over-linear reduction**: Electron density reduction with field must be stronger than linear
2. **Boundary condition**: Electron density at electrode boundary must be low enough to fall into over-linear decrease region (Schottky barrier condition)
3. **Quadrant span**: Range between first and second singular points must span fourth quadrant
4. **Stationarity**: All external parameters must remain stationary

### Additional Condition for Anode-Adjacent Domains
- Field range must extend to singular point III (field excitation competes with field quenching)

## Domain Detection Methods

### Franz-Keldysh Effect
1. Use transmitted monochromatic light at absorption edge
2. Domain appears as darkened region in crystal
3. Field strength determined from slope of domain width vs bias

### Field-of-Direction Method
1. Apply transport and Poisson equations (Eq 37.1, 37.2)
2. Distinguish solution curves without numerical solving
3. Identify domain character when electron density decreases over-linearly

## Measurement Techniques

### Work Function Measurement
1. Establish high-field domain converts Schottky barrier to neutral contact
2. Confirm electron density at interface equals density within domain
3. Change optical excitation (light intensity)
4. Measure domain field via widening with applied voltage
5. Determine electron density from measured field and current
6. Infer work function change (typically 60 meV decrease per 10² intensity increase)

### Virtual Cathode Technique
1. Create shadow band in front of cathode
2. Position shadow a few Debye lengths beyond cathode
3. Vary boundary electron density (n*c) by reducing light intensity or adding quenching light
4. Produce sequential high-field domains with different boundary concentrations
5. Obtain neutrality electron density n1(F) as function of actual field

## Domain Behavior Analysis

### Stationary Domains
- Electric field remains constant within domain
- Electron density remains constant within domain
- At domain end, field and density decrease to low-field bulk value
- Space charge region shifts from blocking contact to anode side of domain
- Converts blocking cathode into neutral contact

### Nonstationary Domains
- Primary domains move slowly toward anode
- Sub-domains may form:
  - Direction: Opposite to primary domain
  - Speed: Much faster than primary domain
  - Appearance: Darker (higher field via Franz-Keldysh shift)
  - Frequency: ~0.5 Hz when coordinated
- Conductivity may invert to p-type in sub-domains
- Current oscillations correlate with sub-domain coordination

## Bias Evolution Behavior

1. **Forward to Voc**: Field at junction interface increases
2. **Field quenching**: Initiates first, then generates high-field domain
3. **Voc to reverse bias**: Field at junction interface remains constant
4. **Beyond Voc**: Additional voltage absorbed by expanding high-field domain

## Key Variables
- `domain_field`: Electric field within domain (~80-85 kV/cm)
- `domain_width`: Width of high-field domain
- `Voc`: Open circuit voltage
- `n*_c`: Boundary electron density at virtual cathode
- `n10`: Bulk electron density

## Expected Outcomes
- Field limited to ~85 kV/cm
- Excess voltage absorbed by domain width expansion
- Open circuit voltage can increase from <0.5V to 0.85V in optimized CdS/CdTe solar cells
- Unambiguous determination of electron density and mobility vs electric field