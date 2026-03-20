# Wave Equation Solutions

## Non-Absorbing Media

**Undamped wave equation:**
```
∇²E - (με/c²) ∂²E/∂t² = 0
```

**Plane wave solution (x-direction propagation, y-polarization):**
```
E_y(x,t) = A exp[iω(t - x/v)]
```

Where:
- A = amplitude
- ω = angular frequency
- v = phase velocity

**Phase velocity relationship:**
```
v = c/√(με) = c/n_r
```

For non-magnetic dielectrics (μ=1):
```
n_r = √ε
```

## Absorbing Media

**Damped wave equation:**
```
∇²E - (με/c²) ∂²E/∂t² - (μσ/c²) ∂E/∂t = 0
```

**Complex plane wave solution:**
```
E_y(x,t) = A exp[iω(t - nx/c)]
```

Where n = n_r + ik is the complex refractive index.

**Substituting n = n_r + ik:**
```
E_y(x,t) = A exp(ωkx/c) exp[iω(t - n_r x/c)]
```

The first exponential term represents attenuation (absorption).

## Energy Calculations

**Poynting vector (instantaneous):**
```
S = E × H
```

**Time-averaged magnitude:**
```
|S| = (1/2) c ε n_r |E|²
```

**Total energy density:**
```
w = (1/2)(ε₀ε|E|² + μ₀μ|H|²)
```

For non-magnetic dielectrics:
```
w = ε|E|²
```

## Equation References
- Non-absorbing: Eq 20.5, 20.6, 20.11, 20.13, 20.15
- Absorbing: Eq 20.16, 20.19, 20.20