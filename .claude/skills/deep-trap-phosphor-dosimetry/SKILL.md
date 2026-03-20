---
name: deep-trap-phosphor-dosimetry
description: Measure radiation dosage using phosphor materials with deep electron traps. Use this skill when working with thermoluminescent dosimeters (TLDs) like CaF:Mn, when you need to quantify radiation exposure through glow curve analysis, or when analyzing materials that store trapped electrons for long-term dosimetry.
---

# Deep Trap Phosphor Dosimetry

This skill enables radiation dosage measurement using phosphor materials that trap electrons in deep energy levels. The trapped charge accumulates proportionally to radiation exposure and can be read out via thermoluminescence.

## When to Use

- Measuring radiation dose with thermoluminescent dosimeters (TLDs)
- Working with phosphor materials like CaF:Mn for dosimetry
- Analyzing glow curves to determine accumulated radiation exposure
- Implementing long-term radiation monitoring systems

## Prerequisites

- Phosphor material with deep trap availability
- Radiation exposure source (X-rays, gamma rays, etc.)
- Traps must be deep enough for long-term storage without significant fading

## Procedure

### 1. Excitation Phase

Expose the phosphor material to radiation:
- Place the dosimeter in the radiation field
- Ensure uniform exposure if possible
- The radiation creates electron-hole pairs, with electrons being trapped in deep energy levels
- The effect is cumulative—longer exposure or higher intensity fills more traps

### 2. Storage Phase

- Trapped electrons remain in deep traps for extended periods
- Deep traps prevent spontaneous release, enabling long-term dose storage
- Storage time depends on trap depth and material properties

### 3. Reading/Dosage Calculation

Perform glow curve analysis:
1. Heat the phosphor sample at a controlled rate
2. Measure the emitted light intensity (thermoluminescence) as a function of temperature
3. Record the glow curve
4. Calculate the area under the glow peak(s)
5. The integrated area is directly proportional to the absorbed radiation dose

### 4. Dosage Determination

- Compare the glow curve area to a calibration curve
- The calibration curve relates integrated intensity to known radiation doses
- Report the dosage in appropriate units (e.g., Roentgens, Gray)

## Key Variables

| Variable | Type | Description |
|----------|------|-------------|
| Dosage | Value | Measured radiation dose |
| Glow Curve Area | Value | Integrated intensity of the glow peak |

## Output

Radiation dosage value (e.g., in Roentgens or Gray)

## Notes

- The linear response range varies by material
- CaF:Mn demonstrates linearity over five orders of magnitude (0.1 to 10^4 R)
- Multiple glow peaks may indicate different trap depths
- Proper calibration is essential for accurate dose determination