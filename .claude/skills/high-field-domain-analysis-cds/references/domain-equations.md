# Transport and Poisson Equations

## Governing Equations

### Transport Equation (Eq 37.1)
```
j = e * n * μ * F + e * D * (dn/dx)
```

### Poisson Equation (Eq 37.2)
```
dF/dx = ρ(x) / ε
```

## Field-of-Direction Method

The field-of-direction method allows distinguishing possible solution curves without numerical solving by analyzing the direction field in the (F, n) phase plane.

### Key Points
- When electron density decreases stronger than linearly, solution curve MUST have domain character
- Singular points determine possible solution branches
- Quadrant analysis determines if stationary domains can form

## Singular Points

- **Singular Point I**: Low-field equilibrium
- **Singular Point II**: Transition region
- **Singular Point III**: High-field region where field excitation competes with field quenching

## Gunne Effect Exclusion

The Gunne effect scenario is excluded from stationary domain analysis because:
- The range between singular points does NOT span the fourth quadrant
- This prevents formation of stationary domains
- Results in moving domains instead