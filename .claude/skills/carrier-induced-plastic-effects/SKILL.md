---
name: carrier-induced-plastic-effects
description: Analyze how carrier injection or optical generation affects dislocation behavior and mechanical strength in semiconductors and crystalline materials. Use this skill when evaluating the electroplastic, photoplastic, or cathodoplastic effects in II-VI compounds, when carriers are being injected or optically generated in materials with dislocations, or when predicting changes in mechanical properties due to carrier density modifications.
---

# Carrier-Induced Plastic Effects Analysis

## When to Use This Skill

Apply this skill when:
- Working with semiconductor materials containing dislocations
- Carriers are being injected or optically generated in the material
- Evaluating mechanical strength changes in II-VI compounds under carrier injection
- Analyzing dislocation climb behavior under electrical or optical excitation
- Investigating photoquenching effects on material properties

## Prerequisites

Before analysis, verify:
- Presence of dislocations in the material
- Available carrier injection mechanism OR optical generation capability
- Existence of defect centers capable of recharging

## Execution Steps

### 1. Identify the Carrier Generation Mechanism

Determine how carriers are being introduced:
- **Electrical injection** → Electroplastic effect
- **Optical generation** → Photoplastic effect
- **Cathode-related injection** → Cathodoplastic effect

The naming convention depends solely on the method of carrier density change induction.

### 2. Analyze the Primary Physical Changes

Monitor the following modifications:
- **Charge density change** per unit length of dislocation
- **Recharging of defect centers** during the process
- **Alteration of dislocation climb ability** (movement perpendicular to slip plane)

### 3. Predict Mechanical Property Changes

**Typical outcome:**
- Mechanical strength INCREASES
- Resistance to plastic deformation increases
- Material becomes harder to deform plastically

**Exception - Photoquenching:**
- Mechanical strength DECREASES
- Opposite effect of typical carrier-induced plastic effects

### 4. Consider Material-Specific Behavior

These effects are particularly pronounced in:
- **II-VI compound semiconductors** (elements from groups II and VI of periodic table)

Effect magnitude varies by material type.

## Key Variables to Track

- `charge_density`: Charge density per unit length of dislocation
- `carrier_density`: Density of charge carriers in the material
- `mechanical_strength`: Resistance to plastic deformation
- `dislocation_climb_ability`: Capability of dislocations to move perpendicular to slip plane

## Common Result Patterns

| Scenario | Mechanical Strength | Dislocation Climb |
|----------|---------------------|-------------------|
| Carrier injection | Increase | Altered |
| Optical generation | Increase | Altered |
| Photoquenching | Decrease | Altered |

## Constraints

- Photoquenching produces opposite effect (decrease in mechanical strength)
- Effect magnitude varies significantly by material type
- Requires presence of mobile dislocations in the material

## Historical Reference

First observed by Osip'yan and Savchenko. Key reference: Osip'yan et al. (1986).