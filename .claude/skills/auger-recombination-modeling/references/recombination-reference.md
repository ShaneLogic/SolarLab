# Recombination Mechanisms Comparison

## Shockley-Read-Hall (SRH) Recombination

```
R_SRH = (n*p - n_i²) / [τ_p*(n + n₁) + τ_n*(p + p₁)]
```

- Dominant at low carrier densities
- Trap-mediated process
- Temperature-dependent via capture times

## Radiative Recombination

```
R_rad = B*(n*p - n_i²)
```

- Direct band-to-band process
- Proportional to np product
- B ≈ 10⁻¹⁰ cm³/s for typical perovskites

## Auger Recombination

```
R_Auger = (A_n*n + A_p*p)*(n*p - n_i²)
```

- Dominant at high carrier densities
- Scales with n²p or np²
- A ≈ 10⁻²⁸ to 10⁻³⁰ cm⁶/s for perovskites

## Relative Importance

| Condition | Dominant Mechanism | Why |
|-----------|-------------------|-----|
| Low light, room temp | SRH | Trap density high, carriers low |
| Moderate light | Radiative | Direct bandgap, moderate np |
| High light/high V_OC | Auger | Cubic scaling dominates |
| High defect density | SRH | Traps provide recombination centers |

## Typical Parameter Values

**Perovskite Materials:**
- `n_i`: ~10¹⁰ - 10¹² cm⁻³
- `B`: 10⁻¹⁰ - 10⁻⁹ cm³/s
- `A_n`, `A_p`: 10⁻²⁸ - 10⁻³⁰ cm⁶/s
- `τ_n`, `τ_p`: 10⁻⁷ - 10⁻⁵ s

**Measurement:**
- Extract ideality factor from J-V slope
- n_id ≈ 1: Radiative dominated
- n_id ≈ 2: SRH dominated  
- n_id > 2: Trap-assisted or Auger