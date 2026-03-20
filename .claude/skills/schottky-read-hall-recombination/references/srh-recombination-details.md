# Schottky-Read-Hall Recombination - Complete Derivation

## Full SRH Equation

### Eq. 27.29 - General Recombination Rate

```
U = (n*p - n_i²) / [τ_p₀ (n + n₁) + τ_n₀ (p + p₁)]
```

### Eq. 27.30 - Lifetime Parameters

```
τ_n₀ = 1 / (c_n N_r)
τ_p₀ = 1 / (c_p N_r)
```

Where:
- c_n, c_p = capture coefficients for electrons and holes (cm³/s)
- N_r = density of recombination centers (cm⁻³)

### Auxiliary Densities

**Eq. 27.31:**
```
n₁ = n_i * exp((E_t - E_i) / (kT))
```

**Eq. 27.32:**
```
p₁ = n_i * exp((E_i - E_t) / (kT))
```

### Simplified Form (Equal Capture Coefficients)

**Eq. 27.33:**
```
U = c N_r (n*p - n_i²) / [n + p + 2n_i cosh((E_t - E_i)/(kT))]
```

where c_n = c_p = c

## Special Cases

### 1. Thermal Equilibrium
```
n*p = n_i²
U = 0
```
No net recombination in equilibrium.

### 2. Mid-Gap Recombination Center (E_t = E_i)
```
n₁ = p₁ = n_i
cosh((E_t - E_i)/(kT)) = 1
```

```
U = c N_r (n*p - n_i²) / (n + p + 2n_i)
```

This gives maximum recombination rate for given carrier densities.

### 3. High Injection (n = p >> n_i)
```
U ≈ c N_r n / 2
```
Carrier lifetime: τ = 2 / (c N_r)

### 4. Low-Level Injection (n ≈ n₀, Δp << n₀)
For n-type semiconductor:
```
U ≈ (p - p₀) / τ_p₀
```

## Energy Dependence

The recombination rate depends strongly on the position of the recombination center:

```
U ∝ 1 / [n + p + 2n_i cosh((E_t - E_i)/(kT))]
```

- **Mid-gap centers**: Maximum recombination efficiency
- **Shallow centers**: Minimum recombination efficiency
- **Center near E_t = E_i**: n₁ = p₁ = n_i, optimal for recombination

## Capture Coefficients

| Center Type | Typical c_n, c_p (cm³/s) |
|-------------|--------------------------|
| Neutral centers | 10⁻⁸ to 10⁻⁷ |
| Coulomb-attractive | 10⁻⁷ to 10⁻⁶ |
| Coulomb-repulsive | 10⁻¹¹ to 10⁻⁹ |

## Lifetime Relationships

From Eq. 27.30:
```
τ_n₀ = 1/(c_n N_r)
τ_p₀ = 1/(c_p N_r)
```

The actual carrier lifetime depends on injection level and trap energy:
```
τ_n = (n₁ + Δn) / (c_n N_r [n₀ + p₀ + Δn + n₁ + p₁])
τ_p = (p₁ + Δp) / (c_p N_r [n₀ + p₀ + Δn + n₁ + p₁])
```

## Sequential Process Interpretation

The SRH equation can be rewritten as:
```
1/U = 1/U_n + 1/U_p
```

Where:
- U_n = rate limited by electron capture
- U_p = rate limited by hole capture

This shows the **sequential nature**: both electrons and holes must be captured for complete recombination.