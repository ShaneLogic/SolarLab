# Detailed Derivation of Tunneling Current

## Step 1: Net Current Definition

The net tunneling current is defined as the difference between right-going and left-going currents:

```
j = e × ∫ Tₑ × [Nᵥ(E) × fₙ - N꜀(E) × fₚ] dE
```

Where:
- `e` = elementary charge
- `Tₑ` = transmission probability
- `Nᵥ(E)` = valence band density of states
- `N꜀(E)` = conduction band density of states
- `fₙ` = Fermi distribution on n-side
- `fₚ` = Fermi distribution on p-side

Note that: fₚ = 1 - fₙ

## Step 2: Incorporating k-space Considerations

Accounting for charge and velocity in k-space, the current density becomes:

```
j = (4πem/h³) × ∫ Tₑ × (fₙ - fₚ) dE
```

## Step 3: Integration Over Energy

Integrating over E_perp (which yields E_ℓ²) and applying approximations:

- Condition: Va >> kT/e and Va >> E/e
- Result simplifies to Fowler-Nordheim form

## Final Expression

```
j = A × F² × exp(-F₀/F)
```

Where:
- `A = e³ / (8πhΔE)`
- `F₀ = (4√(2m) × ΔE^(3/2)) / (3ħe)`
- `ΔE` = barrier height

## Critical Field Example

For typical parameters, the critical field for substantial tunneling (> 10⁻³ A/cm²) is approximately:

```
F_crit ≈ 1.5 × 10⁶ V/cm
```

This is derived from solving for F when j = 10⁻³ A/cm²:

```
F_crit = (4√(2m) × ΔE^(3/2)) / (3ħe × ln(A/j))
```

## Effect of Reduced Effective Mass

If mₙ = 0.1m₀:
- The critical field decreases by approximately a factor of 3
- This significantly enhances tunneling probability

## Variable Definitions

| Variable | Type | Description |
|----------|------|-------------|
| j | Current Density | Tunneling current density (A/cm²) |
| Tₑ | Probability | Transmission probability |
| F | Electric Field | Applied field (V/cm) |
| F₀ | Electric Field | Characteristic field constant |
| ΔE | Energy | Barrier height (eV) |
| m | Mass | Effective mass |
| ħ | Action | Reduced Planck constant |
| h | Action | Planck constant |