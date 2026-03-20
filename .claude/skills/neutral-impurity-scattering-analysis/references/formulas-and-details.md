# Neutral Impurity Scattering - Detailed Formulas

## Scattering Cross Section Estimation

### Basic Approach
The scattering cross section for neutral impurities is approximately the size of the defect atom:
- Typical value: s_n ≈ 10^(-15) cm²

### Hydrogen Atom Analogy
The problem is analogous to electron scattering by hydrogen atoms.

### Detailed Formula (Eq. 17.31)
The scattering cross section s_n is estimated as:

s_n ∝ πa_qH²

Modified by scattering correction factor:
λ_DB/a_qH

where:
- λ_DB = De Broglie wavelength
- a_qH = Bohr radius

### Erginsoy Estimation
Erginsoy estimated:

s_n ≈ 20a_qH/|k|

## Collision Time and Mobility

### Gas-Kinetic Estimate (Eq. 17.32)
Using the standard gas-kinetic approach for collision time.

### Wave Vector Relation
With:
- |k| = 2π/λ_DB = mv/h
- v ≈ v_rms (root mean square velocity)

### Mobility Formula (Eq. 17.33-17.34)

μ = (formula independent of temperature)

**KEY RESULT**: This mobility is INDEPENDENT of temperature.

## Low Temperature Modification

### Blagosklonskaya Correction (Eq. 17.35)
For temperatures below 20K, the modification includes:
- Ionization energy E_i of the impurity
- Energy-dependent screening effects

Reference: Blagosklonskaya et al. (1970)

## Spin Effects

### Spin-Dependent Scattering
Scattering differs depending on whether the incident electron spin is:
- Parallel to electrons in scattering atom (triplet state)
- Antiparallel to electrons in scattering atom (singlet state)

## Multi-Valley Semiconductors

### Mattis and Sinha Analysis (1970)
For multi-valley semiconductors:
- Results similar to Blagosklonskaya at low temperatures
- Only slight mobility reduction at T > 10K

### References
- Figure 17.5 for visualization
- Norton and Levinstein (1972) for detailed analysis
- Seeger (1973) for scattering correction factor

## Variables Reference

| Variable | Type | Description |
|----------|------|-------------|
| s_n | float | Neutral defect scattering cross section (~10^(-15) cm²) |
| N_n | float | Neutral impurity density |
| a_qH | float | Bohr radius |
| λ_DB | float | De Broglie wavelength |
| E_i | float | Ionization energy of impurity (for low T correction) |