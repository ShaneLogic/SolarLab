# Electric Field Superposition in Semiconductors

## Internal (Built-in) Field

The built-in field arises from space charge due to doping inhomogeneities:

```
ρ(x) = q(p(x) - n(x) + N_D⁺(x) - N_A⁻(x))
```

From Poisson's equation:
```
dF_i/dx = ρ(x)/ε    (Eq 25.35)
```

Integrate to find internal field:
```
F_i(x) = (1/ε)∫₀ˣ ρ(x')dx' + F_i(0)
```

## External Field

The external field comes from applied bias V:

```
F_e = V/L
```

Where L is the device length, assuming uniform field in the ideal model.

More generally, for electrodes with surface charges Q_s1 and Q_s2:
```
F_e = (Q_s1 - Q_s2)/ε
```

## Total Field

```
F(x) = F_i(x) + F_e    (Eq 25.36)
```

## Band Diagram Interpretation

The band bending is related to the electric field:

```
dE_c/dx = -qF(x)
dE_v/dx = -qF(x)
```

Both F_i and F_e produce identical band slopes:
- Internal field: Bands bend due to space charge regions
- External field: Bands tilt uniformly across the device

Since both produce the same visual effect on band diagrams, the distinction is often omitted in simple models.

## Neutrality Condition

For overall device neutrality:
```
Q_s1 + Q_s2 + ∫ρ(x)dx = 0
```

The sum of surface charges must balance the net space charge in the device.