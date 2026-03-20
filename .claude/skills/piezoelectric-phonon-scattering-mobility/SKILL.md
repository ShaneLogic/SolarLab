---
name: piezoelectric-phonon-scattering-mobility
description: Calculate carrier mobility in piezoelectric crystals and semiconductors with ionic character at low temperatures. Use this when analyzing piezoelectric materials where lattice vibrations create dipole moments that scatter carriers, particularly in low-temperature regimes with low ionized impurity densities.
---

# Piezoelectric Phonon Scattering Mobility

## When to Use This Skill

Apply this skill when:
- Working with piezoelectric crystals or semiconductors with ionic character
- Analyzing carrier transport at low temperatures
- Ionized impurity density is low (making other scattering mechanisms less dominant)
- The material exhibits piezoelectric properties (K² ~ 10⁻³)

## Prerequisites

Before proceeding, verify:
- Electromechanical coupling constant K is known or can be calculated
- Operating in low temperature regime
- Low density of ionized impurities present
- Material is confirmed piezoelectric

## Procedure

### 1. Identify the Scattering Mechanism

In piezoelectric crystals:
- Longitudinal acoustic phonons cause alternating lattice compression and dilatation
- This oscillation generates a dipole moment
- The dipole creates an electric field parallel to the propagation direction
- Carriers scatter through interaction with this field

### 2. Determine Key Parameters

Obtain or calculate:
- **K**: Electromechanical coupling constant (typically ~10⁻³ for most semiconductors)
- **e_pz**: Piezoelectric constant (~10⁻⁵ As/cm²)
- **c_l**: Longitudinal elastic constant
- **Temperature T**: Current operating temperature

K represents the ratio of mechanical to total work in the piezoelectric material.

### 3. Calculate Mobility

Use the mobility formula (see references for complete derivation):
- The mobility follows μ ∝ T^(-1/2) temperature dependence
- This is stronger than acoustic deformation potential scattering (T^(-3/2))
- Apply Eq. 17.16 (Meyer and Polder 1953) with known constants
- Use numerical expression from Eq. 17.17 for practical calculations

### 4. Verify Results

Check that:
- Results show expected T^(-1/2) temperature scaling
- Mobility values are reasonable for the material system
- Piezoelectric scattering is indeed dominant (not competing with ionized impurity scattering)

## Key Variables

| Variable | Type | Description | Typical Value |
|----------|------|-------------|---------------|
| K | float | Electromechanical coupling constant | ~10⁻³ |
| e_pz | float | Piezoelectric constant | ~10⁻⁵ As/cm² |
| μ | float | Carrier mobility from piezoelectric scattering | Calculated |

## Constraints

- Only applicable to piezoelectric materials
- Competes with ionized impurity scattering at higher impurity densities
- Most significant at low temperatures when impurity scattering is minimized

## References

See `references/detailed-formulas.md` for:
- Complete Eq. 17.16 and Eq. 17.17
- Derivation of temperature dependence
- Relationship between K, e_pz, and c_l
- Comparison with other scattering mechanisms