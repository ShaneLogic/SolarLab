# Detailed Equations and Examples

## Inverted Transport Equations (Eq. 16.16)

The coefficients are obtained by solving the Boltzmann equation for small perturbation:

With modified variables:
- F* = F − ∇φ
- w* = w − jφ/e

Results are listed in Table 16.1 (Conwell 1982).

## Classical Drude Derivation

**Step 1:** Set equal currents from electric field and thermal gradient

**Step 2:** Result for thermoelectric power:
```
α = ev(e)/(3nek)
```

**Step 3:** Replace specific heat:
```
ev(e) = (π²/2)(kT/EF)nk
```

**Step 4:** Final result:
Yields Eq. (16.19) except for factor of 2 due to insufficient scattering consideration.

## Typical Material Values (at T = 300K)

| Material | Seebeck Coefficient α (μV/K) |
|----------|-------------------------------|
| Na       | -8.3                          |
| K        | -15.6                         |
| Pt       | -4.4                          |
| Au       | +1.7                          |
| Cu       | +11.5                         |
| Li       | +0.2                          |

**Note:** k/e = 86 μV/K as reference scale

## References
- Conwell (1982): Table 16.1 for coefficient results
- Beer (1963): Comprehensive review of Wiedemann-Franz law
- Key equations: Eq. (16.16-16.19)