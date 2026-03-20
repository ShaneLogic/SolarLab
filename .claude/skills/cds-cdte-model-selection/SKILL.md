---
name: cds-cdte-model-selection
description: Select and apply appropriate models (IFC, n-p, or n-i-p) for analyzing CdS/CdTe thin-film heterojunction solar cell current-voltage characteristics. Use when analyzing collection efficiency curves from I-V measurements, comparing model fits, characterizing cell performance parameters, or investigating degradation behavior in CdS/CdTe cells.
---

# CdS/CdTe Solar Cell Model Selection

## When to Use This Skill

Use this skill when:
- Analyzing CdS/CdTe thin-film heterojunction solar cells
- Comparing IFC, n-p, and n-i-p models for I-V characteristic fitting
- Determining collection efficiency parameters
- Investigating cell degradation behavior
- Characterizing cells prepared by sequential evaporation and treatments

## Prerequisites

- Current-voltage characteristics measured under AM1 and dark conditions
- CdTe layer thickness known (typically 1.7-7 μm)
- Contact type identified (VD: vacuum deposited, ED: etch deposited)

## Procedure

### 1. Prepare Measurement Data

- Calculate the difference of full AM1 characteristic minus dark curve
- This eliminates the influence of series resistance
- Use this corrected data for all model fitting

### 2. Apply and Compare Three Models

**IFC Model:**
- Fit using Eq. 35.3 (see references)
- Typical best-fit parameters: NA ≈ 2.5 × 10^15 cm^-3, V0 ≈ 0.72 eV
- Assessment: Does NOT approach measured curve completely

**n-p Model:**
- Fit using Eq. 35.3 (see references)
- Typical best-fit parameters: NA ≈ 1.6 × 10^16 cm^-3, V0 ≈ 0.84 eV
- Assessment: Does NOT approach measured curve completely

**n-i-p Model:**
- Fit using Eq. 35.5 (see references)
- Typical best-fit parameters: XC ≈ 22.5, V0 ≈ 0.84 eV
- Assessment: DOES approximate measured curve successfully
- Uses ONLY TWO adjustable parameters
- **Recommended as best approximation among the three models**

### 3. Select the Appropriate Model

- Select **n-i-p model** for CdS/CdTe cells when minimal parameters are desired
- The n-i-p model describes the physics of CdS/CdTe solar cell operation best

### 4. Interpret Parameters with Caution

⚠️ **CRITICAL WARNING:** Physical parameter values derived from model fitting may differ significantly from 'correct model' values.

- Do NOT read too much value into fitted parameters
- Fitted parameters are NOT equivalent to directly measured parameters (e.g., Hall mobility from Hall effect, optical absorption from direct experiments)
- Use fitted parameters for relative comparisons and characterization trends, not absolute physical values

### 5. Degradation Analysis (Optional)

For degradation studies:
- Compare parameters before and after thermal stress (e.g., 25 days at 80°C, open circuit)
- Cu/Ni contacts show different degradation behavior than Ni-only contacts
- Copper layer is essential to reduce degradation in open circuit condition

## Output Variables

| Variable | Type | Description |
|----------|------|-------------|
| model_type | string | Selected model: IFC, n-p, or n-i-p |
| fit_quality | float | Goodness of fit to measured characteristic |
| NA | float | Acceptor density (cm^-3) |
| XC | float | Collection length ratio parameter |
| V0 | float | Characteristic voltage (eV) |

## Key Constraints

- Models deal solely with collection efficiency curve derived from characteristics
- Physical parameter values derived may differ from 'correct model' values
- Systematic data analysis of large sample numbers is recommended for future cell improvement