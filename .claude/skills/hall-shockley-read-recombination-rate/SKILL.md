---
name: hall-shockley-read-recombination-rate
description: Calculate net recombination rate via single-type defect centers using the HSR formalism. Use this for analyzing steady-state recombination in semiconductors, determining carrier lifetimes, or modeling defect-mediated recombination processes.
---

# Hall-Shockley-Read (HSR) Net Recombination Rate

## When to Use
- Steady-state conditions with defect-mediated recombination
- Single type of recombination center dominant
- Calculating net carrier flow between bands
- Relating carrier densities to recombination rates

## Four Transition Rates
Identify the four fundamental transition rates for a center interacting with bands:

1. **Electron capture:** c_nt × n × (N_t - n_t)
2. **Electron emission:** e_tc × n_t
3. **Hole capture:** c_tv × p × n_t
4. **Hole emission:** e_vt × n_v × (N_t - n_t)

Where:
- n, p: electron and hole densities
- N_t: total density of recombination centers
- n_t: density of occupied centers
- c_nt, c_tv: capture coefficients
- e_tc, e_vt: emission coefficients

## Steady-State Condition
The sum of all four transition rates must vanish:
```
(rate_in) - (rate_out) = 0
```

This determines the trapped electron population:
```
n_t = (N_t × (c_nt × n + e_vt × n_v)) / (c_nt × n + c_tv × p + e_tc + e_vt × n_v)
```

## Auxiliary Parameters
Define the characteristic densities:
```
n₁ = N_c × exp[(E_t - E_c) / (kT)]
p₁ = N_v × exp[(E_v - E_t) / (kT)]
n_i² = n₁ × p₁
```
Where:
- N_c, N_v: effective density of states in conduction/valence bands
- E_t: energy of trap level
- n_i: intrinsic carrier density

## HSR Net Recombination Rate Formula
```
U = (np - n_i²) / [τ_p × (n + n₁) + τ_n × (p + p₁)]
```
Where:
- τ_n = 1/(c_cv × N_t): electron lifetime
- τ_p = 1/(c_cv × N_t): hole lifetime
- U: net recombination rate per unit volume

## Physical Interpretation

### Thermal Equilibrium
When np = n_i², U = 0 (net flow is zero, detailed balance).

### Steady State with Excitation
When HSR recombination dominates, U equals the generation rate g₀.

### Maximum Recombination Rate
Maximum occurs when:
- n = p (high-level injection)
- Trap level is near mid-gap (n₁ ≈ p₁ ≈ n_i)

## Simplified Cases

### High-Level Injection (n = p >> n_i)
```
U ≈ (n² - n_i²) / [(τ_n + τ_p) × (n + 2n_i)]
```

### Low-Level Injection (n >> p₀, p ≈ p₀)
```
U ≈ (n × p₀ - n_i²) / [τ_p × (n + n₁)]
```

## Key Insights
- The HSR model accounts for all four possible transitions at a defect
- The denominator contains terms for both electron and hole capture
- The rate is proportional to deviation from equilibrium (np - n_i²)
- Lifetime depends on trap energy position through n₁ and p₁
- Mid-gap traps are most efficient recombination centers