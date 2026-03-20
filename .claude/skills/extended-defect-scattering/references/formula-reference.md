# Extended Defect Scattering Formulas

## Debye Screening Length

```
L_D = √(εε₀·kT / e²·p₀)
```

**Variables:**
- `L_D`: Debye screening length (m)
- `ε`: Relative permittivity of material (dimensionless)
- `ε₀`: Vacuum permittivity (8.854 × 10⁻¹² F/m)
- `k`: Boltzmann constant (1.381 × 10⁻²³ J/K)
- `T`: Absolute temperature (K)
- `e`: Elementary charge (1.602 × 10⁻¹⁹ C)
- `p₀`: Hole density in bulk (m⁻³)

## Podor (1966) Dislocation Mobility Formula

```
μ_disl = 75 × (ε₀/ε_s)^(3/2) × (e/N·a_t²) × (m_n/m₀) × (kT/e) / √n
```

**Variables:**
- `μ_disl`: Mobility limited by dislocation scattering (cm²/V·s)
- `ε₀`: Vacuum permittivity
- `ε_s`: Semiconductor permittivity
- `e`: Elementary charge
- `N`: Dislocation density (cm⁻²)
- `a_t`: Distance between charged defect centers along dislocation core (cm)
- `m_n`: Effective mass of carriers
- `m₀`: Free electron rest mass
- `k`: Boltzmann constant
- `T`: Temperature (K)
- `n`: Carrier density (cm⁻³)

**Physical Interpretation:**
- Inverse dependence on dislocation density (N): More dislocations → more scattering
- Inverse square dependence on defect spacing (a_t²): Closer charges → stronger scattering
- Temperature dependence through kT factor
- Screening through permittivity ratio (ε₀/ε_s)^(3/2)
- Density dependence through 1/√n

## Anisotropic Mobility Relations

For aligned dislocation arrays:

```
μ_parallel ≈ μ_undeformed
μ_perpendicular ≪ μ_undeformed
```

Where:
- `μ_parallel`: Mobility in direction parallel to dislocation alignment
- `μ_perpendicular`: Mobility in direction perpendicular to dislocation alignment
- `μ_undeformed`: Mobility in material without dislocations

## Strain Field Contribution

At neutrality temperature (T₀) where core charge = 0:

```
μ_strain = f(N, deformation_potential, orientation)
```

This component persists even when core charge scattering is zero.

## Combined Mobility

When multiple scattering mechanisms are present:

```
1/μ_total = 1/μ_lattice + 1/μ_disl + 1/μ_strain + 1/μ_cluster
```

Where each term represents a different scattering mechanism.