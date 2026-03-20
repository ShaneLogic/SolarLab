# Detailed Equation Specifications

## Optical Generation (Equation 63)

```
G(x,t) = Is(t) Fph alpha exp(-alpha(2 + l x - 2))
```

**Physical Meaning:** Represents exponential decay of light intensity within the material.

**Constraints:** Only applies when wavelength-dependent profile (32) is NOT used.

## Bulk Recombination (Equation 64)

**Output:** R_bulk (bulk recombination rate)

**Units:** Typically cm^-3 s^-1

**Constraints:** Applies to recombination events within bulk volume only, not at interfaces.

## Interfacial Recombination (Equations 65 and 66)

**Equation 65 Structure:**
```
R_l(n+, p+) = beta k_EE n+ p+ - n_i^2 + nu k_EE n+ (p+ + n_i^+ + n_i) + nu_1 p+ E n^2 (i n+ + n_i)
```

**Variable Definitions:**
- n+: Electron concentration at interface (perovskite side)
- p+: Hole concentration at interface (perovskite side)
- ni: Intrinsic carrier density
- beta: Recombination coefficient
- k_EE: Constant related to electron-electron interactions
- nu: Frequency or velocity factor
- E: Electric field at interface

**Note:** This OCR reconstruction includes terms for band-to-band recombination and trap-assisted or field-dependent components.

**Software Context:** Rates are equivalent to IonMonger formulations but now coded specifically in terms of perovskite carrier concentrations adjacent to each interface.