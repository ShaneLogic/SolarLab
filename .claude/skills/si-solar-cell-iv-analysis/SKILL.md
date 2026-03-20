---
name: Si Solar Cell I-V Analysis Under Bias
description: Analyze current-voltage characteristics and saturation behavior of thick asymmetric Si pn-junction solar cells under applied bias. Use when evaluating solar cell performance, reverse bias behavior, current distribution asymmetry, or device optimization for Si cells with complete surface recombination at electrodes.
---

# Si Solar Cell I-V Analysis Under Bias

## When to Use This Skill

Use this skill when:
- Analyzing thick asymmetric Si pn-junction solar cells under non-zero bias
- Evaluating reverse bias current saturation behavior
- Computing I-V characteristic curves for Si solar cells
- Optimizing device parameters for photoelectric conversion efficiency
- Investigating surface recombination effects at electrodes

## Prerequisites

- Thick device configuration (d2 > Ln, base region thickness exceeds diffusion length)
- Asymmetric doping profile
- Non-vanishing bias conditions applied
- Complete surface recombination at both electrodes

## Constraints

- Model is rather crude and needs substantial refinement for detailed experimental comparison
- Results provide qualitative guidance rather than precise quantitative predictions

## Analysis Procedure

### 1. Verify Device Configuration

Confirm the device matches the thick asymmetric pn-junction model:
- Complete surface recombination at both electrodes
- Standard parameters: go = 2·10^20 cm^-3 s^-1, Nr1 = 10^17 cm^-3, Nr2 = 10^16 cm^-3, c = 10^-9 cm^-3 s^-1

### 2. Analyze Reverse Bias Behavior

**Recombination Rate Distribution:**
- With sufficient reverse bias, r(x) reduces well below go throughout the junction region
- Net generation rate U equals go up to a few diffusion lengths from junction and right surface

**Mid-Bulk Inactive Region:**
- For thick devices (d2 > Ln), identify region where r ≈ go
- This region separates near-junction and near-contact regions
- Near-junction region contributes to photovoltaic effect
- Near-contact region contributes to surface recombination current

**Current Distribution:**
- Current distribution becomes highly asymmetric with applied bias
- Analyze hole current distribution shifts near x = 0

### 3. Calculate Surface Recombination Current

At left electrode:
- Use jp(d1) = ev* p(d1) to calculate surface recombination current
- Compare to saturation current jsc (typically ~35.5 mA/cm²)
- Surface recombination current should be negligible compared to jsc

### 4. Determine Current Saturation

- Saturation is reached when DRO-range starts to appear
- Typical saturation current: jsc ≈ 35.5 mA/cm²
- Compare computed I-V curve to ideal characteristic (dashed curve shifted by ~35.8 mA/cm²)

### 5. Evaluate Device Optimization Opportunities

**Parameters Affecting Efficiency:**
- Electrode separation from photoelectric active region
- Optical carrier generation proximity to junction
- Minority carrier collection efficiency

**Design Balance Requirements:**
- Balance electrode separation vs. optical carrier generation access
- Aim for nearly perfect minority carrier collection
- Address both geometrical and electronic design challenges

## Key Variables

| Variable | Unit | Description |
|----------|------|-------------|
| jsc | mA/cm² | Short-circuit saturation current density |
| U | cm^-3 s^-1 | Net generation/recombination rate |
| d2 | cm | Base region thickness |
| Ln | cm | Diffusion length |

## Output

- I-V characteristic curve analysis
- Saturation current values
- Current distribution asymmetry assessment
- Device optimization recommendations