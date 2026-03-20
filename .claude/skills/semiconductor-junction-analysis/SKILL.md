---
name: semiconductor-junction-analysis
description: Analyze electrical properties of semiconductor junctions and interface barriers. Use this skill when working with diodes, solar cells, transistors, or any device involving contact between dissimilar materials or doping regions. Applies to pn-junctions, heterojunctions, Schottky barriers, metal-semiconductor interfaces, and pin structures.
---

# Semiconductor Junction Analysis

## When to Use

Apply this skill when:
- Two materials or doping regions meet (trigger: junction formation)
- Analyzing diodes, solar cells, or transistors
- Calculating barrier heights, I-V curves, or capacitance
- Modeling metal-semiconductor contacts
- Evaluating heterojunction or homojunction devices

**Do NOT apply** for homogeneous single materials without interfaces.

## Analysis Workflow

### Step 1: Identify Junction Type

Classify the junction based on materials and structure:

1. **Homojunctions**: Same semiconductor material with different doping
   - pn-junction (most common)
   - pin structure (intrinsic layer between p and n)

2. **Heterojunctions**: Different semiconductor materials
   - Frontwall or backwall solar cell configurations
   - Hetero-boundaries at material interfaces

3. **Metal-Semiconductor Junctions**:
   - Schottky barrier (rectifying contact)
   - Ohmic contact (non-rectifying)

### Step 2: Determine Key Parameters

For each junction type, identify:
- Built-in potential (Vbi)
- Barrier height (φB for Schottky)
- Depletion width (W)
- Doping concentrations (NA, ND)

### Step 3: Apply Junction Physics

Execute appropriate calculations:

1. **Poisson Equation** - Solve for potential distribution:
   - Relates charge density to electric field
   - Foundation for depletion region analysis

2. **Space-Charge Analysis**:
   - Calculate depletion region width
   - Determine charge distribution
   - Evaluate space-charge-limited current if applicable

3. **Junction Capacitance**:
   - Depletion capacitance (voltage-dependent)
   - Diffusion capacitance (for forward bias)

### Step 4: Characterize Electrical Behavior

Generate output characteristics:
- I-V curve (rectifying behavior)
- C-V characteristics
- Barrier height determination
- Band alignment (valence band jump for heterojunctions)

## Quick Reference

| Junction Type | Key Feature | Primary Application |
|--------------|-------------|---------------------|
| pn-homojunction | Built-in potential | Diodes, transistors |
| pin | Intrinsic layer | Photodiodes, solar cells |
| Heterojunction | Band discontinuity | High-efficiency solar cells |
| Schottky barrier | Metal-semiconductor barrier | Fast diodes, contacts |

## Output Format

Provide results as:
- I-V characteristic curves
- Barrier height values
- Capacitance-voltage relationships
- Depletion region parameters