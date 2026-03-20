---
name: hall-effect-analysis
description: Analyze Hall effect measurements in semiconductor samples to determine carrier type, concentration, and mobility. Use when performing Hall effect experiments or interpreting Hall voltage data.
---

# Hall Effect Analysis

Use this skill when:
- Measuring Hall effect in semiconductor samples
- Determining carrier type (n-type or p-type)
- Calculating carrier concentration from Hall measurements
- Computing Hall mobility and comparing to carrier mobility

## Verify Experimental Conditions

1. Confirm isothermal conditions
2. Verify magnetic induction is small enough for linear treatment
3. Check sample geometry (typically two-dimensional platelet)

## Apply Hall Effect Equations

1. **Hall Voltage (Eq. 16.31)**:
   - Determine VH from balancing field when jy = 0
   - VH = RH × jx × Bz

2. **Hall Constant (Eq. 16.33)**:
   - RH = VH/(jx × Bz)
   - RH ≈ 1/(en) with numerical factor depending on scattering mechanism

3. **Hall Angle (Eq. 16.32)**:
   - θH = arctan(Fy/Fx)
   - Permits direct measurement of Hall mobility μH

4. **Hall Mobility vs Carrier Mobility**:
   - μH is usually slightly smaller than μ
   - Ratio depends on scattering mechanism:
     * Acoustic mode scattering: μH/μ = 3π/8
     * Ionized impurity scattering: μH/μ = 1.7
     * Higher defect densities: μH/μ = 1

## Determine Carrier Properties

1. **Carrier Type**:
   - NEGATIVE RH → n-type conduction (electrons)
   - POSITIVE RH → p-type conduction (holes)

2. **Carrier Concentration**:
   - n ≈ 1/(e × RH) (with numerical factor correction)

3. **Special Cases**:
   - Ellipsoidal equi-energy surfaces: Use Eq. (16.34)
   - Two-carrier systems: Use Eq. (16.35)

## Key Variables

- **VH**: Hall voltage (scalar)
- **RH**: Hall constant (scalar)
- **μH**: Hall mobility (scalar)
- **μ**: Carrier mobility (scalar)
- **θH**: Hall angle (scalar)
- **Bz**: Magnetic induction in z-direction (scalar)

## References

See `references/hall-effect-details.md` for detailed equations, scattering mechanisms, and multi-carrier analysis.