---
name: anode-adjacent-domain-analysis
description: Analyze and characterize anode-adjacent high-field domains in CdS crystals and solar cells. Use this skill when investigating high-bias semiconductor behavior, junction leakage problems in CdS/CdTe/CIS solar cells, or when current-voltage characteristics show pre-breakdown stabilization. Triggers on mentions of domain formation, singular points in field analysis, or solar cell junction optimization.
---

# Anode-Adjacent High-Field Domain Analysis

## When to Use

Apply this skill when:
- Analyzing CdS crystals under high bias conditions
- Investigating junction leakage in CdS/CdTe/CIS solar cells
- Current-voltage characteristics show unexpected stabilization in pre-breakdown range
- Field excitation competes with field quenching at high fields
- Transition from cathode-adjacent to anode-adjacent domain behavior

## Prerequisites

- High field range where field excitation competes with field quenching
- Domain fills entire crystal
- Transition from cathode-adjacent domain already occurred

## Analysis Procedure

### Step 1: Extend Neutrality Curve Analysis

Extend the neutrality curve in the field-of-direction to higher fields. Identify where the n1(F) curve levels off or increases, indicating field excitation competing with quenching.

### Step 2: Locate Singular Points

1. Identify the third singular point (III) at the intersection of n1(F) and n2(F)
2. Confirm the solution curve can no longer approach singular point I (bulk)
3. Verify the curve must connect points II and III

### Step 3: Characterize Domain Formation

Observe domain behavior:
- Domain starts at the anode
- Domain expands toward the cathode
- Bulk side of cathode-adjacent domain shrinks
- Only high-field horizontal branch at singular point II remains

### Step 4: Determine Field Strength

Calculate field strength from the slope of domain width vs bias:
- Typical anode-adjacent domain: ~135 kV/cm
- Compare with cathode-adjacent domain: ~80 kV/cm

### Step 5: Evaluate Stabilization Effects

Verify current stabilization:
- Current-voltage characteristic stabilizes in pre-breakdown range
- Current remains lower than expected
- Run-away current prevented (minimum energy principle)

## Solar Cell Application

For CdS/CdTe/CIS solar cells:

1. Verify domain limits field at junction interface below 80 kV/cm
2. Confirm junction leakage elimination
3. Measure open circuit voltage improvement (potential doubling)

## Key Variables

| Variable | Type | Description |
|----------|------|-------------|
| singular_point_III | Abstract Point | Intersection point for anode-adjacent domain |
| leakage_current | Current | Undesired current at junction |
| domain_field_strength | Field | Typically 135 kV/cm for anode-adjacent |

## Expected Results

- Anode-adjacent domain characterized at ~135 kV/cm
- Stabilized solar cell junction with reduced leakage
- Improved open circuit voltage performance