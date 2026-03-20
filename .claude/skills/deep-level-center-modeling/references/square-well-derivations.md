# Square Well Potential Model - Detailed Derivations

## Schrödinger Equation Solution

For a 1D rectangular well of depth V0 and width 2a:

```
V(x) = -V0 for |x| < a
V(x) = 0 for |x| > a
```

## Boundary Conditions

Wavefunction and its derivative must be continuous at x = ±a.

## Energy Eigenvalues

### Bound State Solutions

For even parity states:
```
√(2m(E+V0)/ħ²) * tan(√(2m(E+V0)/ħ²) * a) = √(2m|E|/ħ²)
```

For odd parity states:
```
-√(2m(E+V0)/ħ²) * cot(√(2m(E+V0)/ħ²) * a) = √(2m|E|/ħ²)
```

## Quadratic Scaling

In the deep well limit (V0 >> E), solutions approximate:

```
E_n ≈ (n²π²ħ²)/(8ma²) - V0
```

This shows the quadratic increase with quantum number n, contrasting with the 1/n² decrease for hydrogenic states.

## Coulomb Tail Corrections

For charged centers, include Coulomb potential V_C(r) = -e²/(4πεr) at large distances.

At low principal quantum numbers, the short-range core potential dominates.
At high principal quantum numbers, the Coulomb tail dominates.

Crossover point occurs when:

```
|V_core(r)| ≈ |V_Coulomb(r)|
```