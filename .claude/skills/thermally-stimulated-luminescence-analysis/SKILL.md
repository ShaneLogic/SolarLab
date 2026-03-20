---
name: thermally-stimulated-luminescence-analysis
description: Perform Thermally Stimulated Luminescence (TSL) to analyze deep trap properties, and compare with Thermally Stimulated Conductivity (TSC). Use to determine trap depths, trap distributions, and identify thermal quenching artifacts in phosphors and semiconductors.
---

# Thermally Stimulated Luminescence Analysis

Analyze deep trap properties using temperature ramping techniques and distinguish between TSL and TSC behaviors.

## When to Use
- Determining deep trap properties (ionization energy, trap depth)
- Analyzing phosphors, luminophores, or materials with deep traps
- Characterizing trap distributions in semiconductors
- Distinguishing between TSL and TSC measurement results

## Thermally Stimulated Luminescence (TSL) Procedure

### 1. Estimate Residence Time

**Formula:**
```
t_res ≈ 1 / e_tc = 1 / (ν_t * exp(-(E_c - E_t) / kT))
```

**Example:** 1.5 eV trap in ZnS (3.6 eV gap):
- At 300K: ~millions of years
- At 600K: ~minutes

**Note:** Residence time > 1s at room temp considered phosphorescence

### 2. Preparation Steps
- Cool the phosphor
- Illuminate to fill traps (charging)

### 3. Heating Procedure

**Temperature ramp:**
```
T(t) = T_0 + a_T * t
```
Where:
- a_T = heating rate (dT/dt)
- T_0 = starting temperature

### 4. Measurement
Measure luminescence intensity as function of time (or temperature) → **Glow Curve**

### 5. Glow Curve Interpretation

**Curve features:**
- Rises: Electrons released from traps
- Maximum: Peak emission temperature
- Decreases: Traps depleted

**Analysis:**
- Maxima indicate specific trap levels or structured distributions
- Deconvolute curve to obtain trap distribution

## TSL vs TSC Comparison

### Single Trap Level
- Both TSL and TSC maxima occur at **same temperature**
- Reason: Both proportional to free carrier density

### Multiple Levels Involved
- **Differences appear** in curve shape and maxima position between TSL and TSC
- **Competition** exists between radiative (TSL) and non-radiative transitions
- **Temperature influences** competition (thermal quenching)

### Thermal Quenching / Cut-off

**Quenching temperature T_q:** Temperature where quenching transition becomes significant

**Cause:**
- Release of holes from hole traps
- Leads to desensitization (lowering of carrier lifetime)

**Result:**
- Steep decrease in current/luminescence
- **Warning:** Can be misinterpreted as missing part in electron-trap distribution

## Key Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Residence time | t_res | Time carrier stays in trap |
| Attempt-to-escape frequency | ν_t | Frequency factor for emission |
| Trap depth | E_c - E_t | Ionization energy |
| Heating rate | a_T | Rate of temperature increase |
| Quenching temperature | T_q | Onset of thermal quenching |

## Output
- Glow curve (Intensity vs Temperature)
- Trap depths identified from peak positions
- Trap distribution from curve deconvolution
- Identification of thermal quenching artifacts
- Distinction between TSL and TSC behaviors