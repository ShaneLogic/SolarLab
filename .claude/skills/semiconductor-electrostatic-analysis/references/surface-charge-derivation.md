# Electrode Surface Charge Derivation

## Current Continuity

In the bulk region (outside space-charge region), current is purely drift:

```
J = qμnF    (Eq 25.15)
```

Current continuity requires:
```
dJ/dx = 0
```

Therefore:
```
nF = constant
```

## Surface Charge from Poisson's Equation

At the electrode boundary, integrate Poisson's equation:

```
∫(dF/dx)dx = ∫(ρ/ε)dx
```

The field discontinuity at the interface is related to surface charge:

```
F_out - F_in = Q_s/ε
```

## Eliminating Field

Using J = qμnF, solve for field:

```
F = J/(qμn)
```

Substitute into the surface charge expression:

```
Q_s = ε × (J/(qμn))
```

**Final expressions:**
```
Q_s1 = J/(qμ n₁)   (left electrode)
Q_s2 = J/(qμ n₂)   (right electrode)
```

## Numerical Example

Given:
- n₂ = 10 × n₁
- J = 2,000 A/cm²
- μ ≈ 1,000 cm²/(V·s) (typical electron mobility)
- q = 1.6 × 10⁻¹⁹ C

**Right electrode surface charge:**
```
Q_s2 = 2000 / (1.6×10⁻¹⁹ × 1000 × n₂)
```

For typical n₂ ≈ 10¹⁶ cm⁻³:
```
Q_s2 ≈ 1.25 × 10⁻⁹ As/cm²
```

In electrons/cm²:
```
Q_s2 ≈ 1.25 × 10⁻⁹ / (1.6 × 10⁻¹⁹) ≈ 7.8 × 10⁹ electrons/cm²
```

**Left electrode surface charge (10× higher due to n₁ = n₂/10):**
```
Q_s1 ≈ 7.8 × 10¹⁰ electrons/cm²
```