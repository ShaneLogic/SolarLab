---
name: semiconductor-energy-level-calculations
description: Calculate semiconductor energy band levels (conduction band, valence band, vacuum energy, bandgap) using electron affinity, ionization potential, and electrostatic potential. Use when working with semiconductor materials and energy band diagrams in Driftfusion simulations.
---

# Semiconductor Energy Level Calculations

## When to Use
Use this skill when you need to:
- Determine energy band positions for semiconductor materials
- Calculate conduction band and valence band energies
- Work with energy band diagrams in Driftfusion
- Analyze how electrostatic potential affects band energies

## Input Convention
**Critical:** Electron Affinity (Phi_EA) and Ionization Potential (Phi_IP) must be input and stored as NEGATIVE values.

## Calculation Procedure

1. **Calculate Electronic Bandgap (E_g):**
   ```
   E_g = Phi_IP - Phi_EA
   ```

2. **Determine Vacuum Energy (E_vac):**
   - Spatial changes in electrostatic potential (V) are reflected in E_vac
   ```
   E_vac = -q * V
   ```
   where q is the elementary charge

3. **Calculate Conduction Band Energy (E_CB):**
   ```
   E_CB = E_vac + Phi_EA
   ```

4. **Calculate Valence Band Energy (E_VB):**
   ```
   E_VB = E_vac - Phi_IP
   ```

## Output
Energy levels in electron volts (eV):
- Vacuum Energy (E_vac)
- Conduction Band Energy (E_CB)
- Valence Band Energy (E_VB)
- Electronic Bandgap (E_g)

## Physical Interpretation
Band energies include both:
- Molecular orbital energies of the solid
- Electrostatic potential from internal and external charge distributions

## Variables
- `Phi_EA`: Electron affinity (stored as negative, float)
- `Phi_IP`: Ionization potential (stored as negative, float)
- `V`: Electrostatic potential (float)
- `q`: Elementary charge (constant)