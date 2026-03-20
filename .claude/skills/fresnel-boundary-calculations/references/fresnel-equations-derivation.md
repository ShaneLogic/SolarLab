# Fresnel Equations Derivation

## Boundary Conditions

At the interface (z=0), the tangential components of E and H are continuous:

```
E_i_t + E_r_t = E_t_t
H_i_t + H_r_t = H_t_t
```

Where subscripts i, r, t denote incident, reflected, and transmitted waves.

## Fresnel Amplitude Coefficients

**Perpendicular polarization (s-polarization):**
```
r_⊥ = (n₁cosθ_i - n₂cosθ_t) / (n₁cosθ_i + n₂cosθ_t)

t_⊥ = (2n₁cosθ_i) / (n₁cosθ_i + n₂cosθ_t)
```

**Parallel polarization (p-polarization):**
```
r_∥ = (n₂cosθ_i - n₁cosθ_t) / (n₂cosθ_i + n₁cosθ_t)

t_∥ = (2n₁cosθ_i) / (n₂cosθ_i + n₁cosθ_t)
```

Using Snell's law: n₁sinθ_i = n₂sinθ_t

## Energy-Based Coefficients (R and T)

**Reflectance:**
```
R = |r|²
```

**Transmittance:**
```
T = (n₂cosθ_t / n₁cosθ_i) × |t|²
```

**For normal incidence (θ_i = θ_t = 0):**
```
R = ((n₂ - n₁) / (n₂ + n₁))²
T = 4n₁n₂ / (n₂ + n₁)²
```

**For air-dielectric interface (n₁ = 1):**
```
R = ((n_r2 - 1) / (n_r2 + 1))²
T = 4n_r2 / (n_r2 + 1)²
```

## Special Cases

**Brewster's angle (p-polarization):**
- When θ_i = θ_B, r_∥ = 0
- θ_B = arctan(n₂/n₁)

**Total internal reflection:**
- Occurs when n₁ > n₂ and θ_i > θ_c
- Critical angle: θ_c = arcsin(n₂/n₁)

## Equation References
- Eq 20.41-20.45 (general Fresnel equations)
- Eq 20.48-20.51 (reflectance and transmittance)