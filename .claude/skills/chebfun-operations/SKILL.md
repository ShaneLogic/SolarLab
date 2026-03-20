---
name: chebfun-operations
description: Perform mathematical operations on chebfuns including integration, differentiation, and other calculus operations. Use when computing integrals, derivatives, or applying functions to chebfuns.
---

# Chebfun Operations

## Definite Integration

### Basic Integration
```matlab
sum(f)             % Integral over entire domain
sum(f, a, b)       % Integral over subdomain [a, b]
```

Method: Clenshaw-Curtis quadrature on the polynomial representation

## Indefinite Integration (cumsum)

### Cumulative Sum
```matlab
fint = cumsum(f)   % Indefinite integral starting at 0 at left endpoint
```

### Adjusting Integration Constant
```matlab
fint = fint - fint(0)    % Make value 0 at t=0
```

**Important Notes**:
- If `f` is a fun of length `n`, `cumsum(f)` returns a fun of length `n+1`
- For piecewise chebfuns, integration is performed on each smooth piece separately
- Default integral has value 0 at left endpoint
- For functions with singularities, start integral at appropriate point (e.g., Soldner's constant for logarithmic integral)

## Common Operations

Over 200 commands apply to chebfuns. View them with:
```matlab
methods chebfun
```

### Key Operations
- `sum(f)` - Definite integral
- `diff(f)` - Derivative
- `roots(f)` - Find zeros
- `abs(f)`, `acos(f)`, `norm(f)` - Standard math functions
- `mean(f)` - Average value

### Arithmetic Operators
All standard MATLAB operators are overloaded:
```matlab
f + g              % Addition
f * g              % Multiplication  
f.^2               % Power
1 ./ (1 + 25*x.^2) % Uses rdivide, plus, power, times
```

### Evaluation
```matlab
y = f(x)           % Evaluate chebfun at point x
```

## Use When

- Computing integrals (definite or indefinite) of functions
- Applying mathematical transformations to functions
- Evaluating functions at specific points
- Performing calculus operations numerically
- Combining functions arithmetically