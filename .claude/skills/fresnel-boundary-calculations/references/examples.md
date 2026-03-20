# Fresnel Calculations Examples

## Example 1: Normal Incidence at Air-Glass Interface

Given: n_air = 1, n_glass = 1.5

**Reflectance:**
```
R = ((1.5 - 1) / (1.5 + 1))² = (0.5/2.5)² = 0.04 = 4%
```

**Transmittance:**
```
T = 4 × 1.5 / (1.5 + 1)² = 6 / 6.25 = 0.96 = 96%
```

Check: R + T = 0.04 + 0.96 = 1 ✓

## Example 2: Normal Incidence at Air-Diamond Interface

Given: n_air = 1, n_diamond = 2.42

**Reflectance:**
```
R = ((2.42 - 1) / (2.42 + 1))² = (1.42/3.42)² = 0.1726 = 17.26%
```

**Transmittance:**
```
T = 4 × 2.42 / (2.42 + 1)² = 9.68 / 11.6164 = 0.8334 = 83.34%
```

## Example 3: Oblique Incidence at 45°

Given: n₁ = 1 (air), n₂ = 1.5 (glass), θ_i = 45°

**First, calculate transmission angle:**
```
θ_t = arcsin((n₁/n₂) × sinθ_i)
θ_t = arcsin((1/1.5) × sin(45°)) = arcsin(0.4714) = 28.1°
```

**Perpendicular polarization:**
```
r_⊥ = (1×cos45° - 1.5×cos28.1°) / (1×cos45° + 1.5×cos28.1°)
r_⊥ = (0.7071 - 1.323) / (0.7071 + 1.323) = -0.6159 / 2.030 = -0.303

R_⊥ = |r_⊥|² = 0.092 = 9.2%
```

**Parallel polarization:**
```
r_∥ = (1.5×cos45° - 1×cos28.1°) / (1.5×cos45° + 1×cos28.1°)
r_∥ = (1.0607 - 0.8829) / (1.0607 + 0.8829) = 0.1778 / 1.9436 = 0.0915

R_∥ = |r_∥|² = 0.0084 = 0.84%
```

Note: Parallel polarization has much lower reflection at 45°.