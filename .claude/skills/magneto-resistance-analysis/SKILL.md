---
name: magneto-resistance-analysis
description: Calculate and analyze magneto-resistance effects in semiconductors under magnetic fields. Use when evaluating conductivity changes due to magnetic induction, including geometry-dependent effects like Corbino disk enhancement.
---

# Magneto-resistance Analysis

Use this skill when:
- Analyzing conductivity reduction in magnetic fields
- Evaluating transverse magneto-resistance in semiconductors
- Calculating geometry-dependent magneto-resistance (e.g., Corbino disk)
- Determining effective mass anisotropy from magneto-resistance data

## Determine Analysis Type

1. **Standard Transverse Magneto-resistance**:
   - Magnetic induction B perpendicular to electric field F
   - Sample has surfaces perpendicular to main current direction
   - Use when ωcτm ≳ 1 (cyclotron frequency × relaxation time)

2. **Corbino Disk Geometry**:
   - Circular sample with central hole electrode
   - Current flows from center to circumference
   - No surfaces perpendicular to main current direction
   - Maximizes magneto-resistance effect

## Calculate Transverse Magneto-resistance

1. **Check conditions**:
   - B = (0, 0, Bz), F perpendicular to B
   - Relaxation time distribution known
   - Scattering NOT influenced by magnetic induction

2. **Apply magneto-resistance formula** (for ωcτm ≲ 1):
   - jx = σFx{1 − f(Bx²)}
   - Magneto-resistance coefficient ≈ μ²n × numerical factor
   - Numerical factor range: 0.38 to 2.15 (depends on scattering)

3. **Special cases**:
   - Two-carrier systems: Additive for both carriers
   - Non-spherical equi-energy surfaces: See Conwell (1982)

## Calculate Corbino Disk Magneto-resistance

1. **Apply enhancement formula**:
   - Δρ/ρ = (Δρ/ρ)f × [1 + (μH × B)²]
   - Where (Δρ/ρ)f is magneto-resistance in filament-type sample

2. **Compare geometries**:
   - Thin filament: Smallest magneto-resistance
   - Corbino disk: Maximized magneto-resistance (no surface charge compensation)

## Interpret Results

1. Magneto-resistance provides information about:
   - Effective mass anisotropy
   - Scattering mechanisms
   - Carrier type and concentration

2. Physical origin:
   - Hall field compensates only for average velocity electrons
   - Slower/faster electrons are more/less deflected
   - Results in less favorable path average for conductivity

## Key Variables

- **ωc**: Cyclotron frequency (scalar)
- **τm**: Relaxation time (momentum) (scalar)
- **μn**: Electron mobility (scalar)
- **vD**: Drift velocity (vector)
- **μH**: Hall mobility (scalar)
- **B**: Magnetic induction (scalar or vector)
- **(Δρ/ρ)f**: Magneto-resistance change in filament-type sample (scalar)

## References

See `references/magneto-resistance-details.md` for detailed derivations, equation numbers, and example calculations.