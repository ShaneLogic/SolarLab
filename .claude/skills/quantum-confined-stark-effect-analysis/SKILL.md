---
name: quantum-confined-stark-effect-analysis
description: Analyze electric field effects on quantum wells, superlattices, and confined systems. Use when determining ionization conditions or calculating Stark shifts for excitons and hydrogen-like defects in confined geometries under applied electric fields.
---

# Quantum Confined Stark Effect Analysis

This skill analyzes the behavior of quantum wells, superlattices, and confined systems under applied electric fields, determining ionization conditions and calculating Stark energy shifts.

## When to Use This Skill

Apply this skill when:
- An electric field is applied to a quantum well or superlattice structure
- You need to determine if excitons or defects will ionize under a field
- You need to calculate Stark shift magnitudes (quadratic or linear regime)
- Analyzing hydrogen-like defects or excitons in confined geometries

## Prerequisites

Before applying this skill, ensure you have:
- Electric field magnitude (F)
- Quantum well width (l)
- Binding energy of the defect or exciton (E_b)
- Direction of electric field relative to layers

## Execution Workflow

### Step 1: Determine Field Direction

Identify whether the applied electric field is:
- **PARALLEL** to the superlattice/quantum well layers
- **NORMAL** to the superlattice/quantum well layers

The direction determines the applicable physical model and ionization thresholds.

### Step 2: Parallel Field Analysis

If the field is parallel to the layers:

1. Treat the system similar to bulk material behavior
2. Field ionization occurs when the energy gain exceeds binding energy
3. Typical ionization fields are on the order of 10 kV/cm

### Step 3: Normal Field Analysis

If the field is normal to the layers:

1. Quantum confinement permits much higher fields before ionization (typically > 10^5 V/cm)
2. Check ionization condition:
   
   `e * F * a_B > E_b`
   
   Where:
   - e is elementary charge
   - F is electric field
   - a_B is Bohr radius
   - E_b is binding energy

3. If ionization condition is NOT met, calculate Stark shift based on regime:
   
   **a) Quadratic Stark Regime:**
   - Binding energy changes as: ΔE_b ∝ F² * l⁴
   - Lower field regime
   
   **b) Linear Stark Regime:**
   - Binding energy changes with lesser slope: ΔE_b ∝ F * l³
   - Higher field regime

### Step 4: High Field Considerations

At very high fields, consider tunneling effects where carriers may tunnel out of the quantum well even when the ionization condition is not directly met.

## Output

This skill provides:
- Ionization status (ionized vs. confined)
- Magnitude of Stark shift
- Regime identification (quadratic vs. linear)
- Critical field values for transitions

## Constraints

- Behavior differs significantly based on field direction
- Parallel fields follow bulk-like behavior with lower ionization thresholds
- Normal fields leverage quantum confinement to achieve higher field tolerance