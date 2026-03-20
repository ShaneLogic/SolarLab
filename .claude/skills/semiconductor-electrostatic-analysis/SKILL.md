---
name: semiconductor-electrostatic-analysis
description: Calculate electric fields, surface charges, and potential distributions in semiconductor devices under bias. Use this skill when analyzing space-charge regions, determining electrode surface charges for neutrality, or calculating total electric fields as superposition of built-in and applied fields.
---

# Semiconductor Electrostatic Analysis

Analyze electric fields and charge distributions in semiconductor devices, particularly focusing on surface charges at electrodes and the superposition of internal and external fields.

## When to Use
- Calculating surface charges at electrodes when quasi-neutrality breaks down in the bulk
- Determining total electric field in a biased semiconductor device
- Analyzing space-charge regions and internal fields
- Ensuring device neutrality under applied bias
- Working with current continuity and drift current in bulk regions

## Electrode Surface Charge Calculation

Use when quasi-neutrality breaks down in the bulk due to bias, requiring surface charges at electrodes to maintain device neutrality.

**Procedure:**
1. **Understand the charge origin:**
   - Missing charges required for device neutrality are located at the two electrodes
   - These charges split between both electrodes to provide drift current continuity in the bulk
2. **Express current in bulk (outside space-charge region):**
   - Current is given by drift alone: J = qμnF (Eq 25.15)
   - Continuity requires dJ/dx = 0, so the product nF must be constant
3. **Derive surface charges using Poisson's equation:**
   - Integrate Poisson's equation at the electrode boundary
   - Eliminate field F using the drift current equation: F = J/(qμn)
4. **Calculate surface charges:**
   - Left electrode (1): Q_s1 = J/(qμ n₁)
   - Right electrode (2): Q_s2 = J/(qμ n₂)
5. **Note the relationship:**
   - Surface charge is inversely proportional to carrier density at each electrode
   - If n₂ = 10×n₁, then σ₁ = 10×σ₂

## Superposition of External and Built-in Fields

Use when determining the total electric field in a device with applied bias.

**Procedure:**
1. **Calculate the internal (built-in) field (Fi):**
   - Use Poisson's equation: dFi/dx = ρ/ε
   - Where ρ is the space charge density from inhomogeneities in charged donors or acceptors
2. **Calculate the external field (Fe):**
   - Created by external bias applied to electrodes
   - Results in surface charge on electrodes with no space charge within semiconductor (in ideal simple model)
3. **Determine the total acting field (F):**
   - F = Fi + Fe
4. **Note on band slopes:**
   - Both external and internal fields result in the same slope of the bands
   - The distinction is usually not made in simple models (subscripts omitted)

## Key Variables
- **J**: Current density
- **Q_s**: Surface charge density at electrode (As/cm² or electrons/cm²)
- **μ**: Carrier mobility
- **n**: Electron density in the bulk
- **F**: Total electric field
- **Fi**: Internal (built-in) electric field
- **Fe**: External electric field from bias
- **ρ**: Space charge density
- **q**: Elementary charge

## Constraints
- Charges are split between both electrodes
- Drift current continuity requires nF = constant in bulk
- Both external and internal fields produce identical band slopes