---
name: matlab-pdepe-limitations-avoidance
description: Identify scenarios where MATLAB's pdepe solver is unsuitable for PSC modeling, particularly multi-layer devices with internal boundary jump conditions. Use when evaluating solver compatibility for three-layer PSC models (ETL, Perovskite, HTL) with interface discontinuities in carrier currents.
---

# MATLAB pdepe Solver Limitations Avoidance

## When to Use This Skill

Use this skill when:
- Modeling three-layer PSC devices (ETL, Perovskite, HTL)
- Evaluating solver options for multi-layer systems
- Checking for internal boundary conditions
- Assessing physical realism of interface modeling

## Limitations Assessment Procedure

### 1. Check Model Structure

**Determine if the model requires:**
- Three-layer structure: ETL (electron transport layer), Perovskite, HTL (hole transport layer)
- Internal boundaries at material interfaces

### 2. Identify Interface Requirements

**Check for jump conditions at interfaces:**
- ETL/Perovskite interface
- Perovskite/HTL interface
- Discontinuities in carrier currents due to surface recombination

### 3. Evaluate pdepe Compatibility

**pdepe CANNOT be used when:**
- Jump conditions exist on internal boundaries
- Equations are not in required "standard form"
- Interface discontinuities require precise modeling

**Why pdepe fails:**
- Cannot accept jump conditions on internal boundaries
- Requires all equations in standard form
- Lacks flexibility for interface-specific physics

### 4. Recognize Unsuitable Workarounds

**Literature workaround (NOT recommended):**
- Artificially smearing surface recombination across "diffuse" interface regions

**Problems with workaround:**
- Physically unrealistic representation
- Large errors (sum of time-averaged errors ≈ 0.3 for N=900)
- Potential failure for long protocols like J-V scans

### 5. Decision Rule

**IF jump conditions exist:**
- Do NOT use pdepe solver
- Use Finite Element or Finite Difference schemes instead
- Implement explicit interface conditions

**IF no jump conditions:**
- pdepe may be suitable for single-layer models
- Still verify against performance benchmarks

## Variables

| Variable | Type | Description |
|----------|------|-------------|
| Layers | Integer | Number of physical layers in device |
| Interface Conditions | Boolean | Presence of jump conditions at internal boundaries |

## Key Constraints

- pdepe only applies to 1D spatial problems
- Requires standard form input for all equations
- Cannot model internal boundary discontinuities

## Error Magnitude

When pdepe is inappropriately used:
- Sum of time-averaged errors can reach **0.3** (at N=900 grid resolution)
- Errors increase with protocol duration (e.g., J-V scans)

## See Also

- `psc-numerical-method-selection` - for alternative methods (FE, FD)