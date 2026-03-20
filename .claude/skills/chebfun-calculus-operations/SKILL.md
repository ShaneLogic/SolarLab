---
name: chebfun-calculus-operations
description: Compute derivatives of chebfun objects using continuous finite differences, and apply non-smooth operations (abs, min, max, sign, round, floor, ceil) that introduce breakpoints at zeros or transitions. Use when you need to differentiate functions, handle functions with discontinuities, or create piecewise-defined functions from smooth components.
---

# Chebfun Calculus Operations

## Differentiation via diff

Compute the derivative of a chebfun using continuous analogue of finite differences.

### Basic Usage
```matlab
derivative = diff(f)
```

### Higher Order Derivatives
```matlab
% Second derivative
derivative_2 = diff(f, 2)

% Fourth derivative
derivative_4 = diff(f, 4)
```

### Handling Discontinuities

If the function has a jump, the derivative introduces a delta function with amplitude equal to the jump size.

### Inverse Relationships

```matlab
% Differentiating indefinite integral returns original function
diff(cumsum(f)) ≈ f

% Integrating derivative returns original plus constant
cumsum(diff(f)) ≈ f + C
% Add value at left endpoint to recover f exactly
```

### Stability Warning

Differentiation is an ill-posed problem:
- Errors in stable operations accumulate exponentially
- Successive derivatives lose information
- Differentiating a low-degree polynomial many times may eliminate the function entirely

### Example
```matlab
f = chebfun('sin(pi*x)')
f_prime = diff(f)
f_double_prime = diff(f, 2)

plot(f)
hold on
plot(f_prime, 'r--')
plot(f_double_prime, 'g:')
legend('f', "f'", "f''")
```

## Non-smooth Operations

Operations that introduce breakpoints/discontinuities.

### Absolute Value
```matlab
g = abs(f)
% Introduces breakpoints where f = 0
```

### Minimum and Maximum
```matlab
g = min(f, h)  % or max(f, h)
% Introduces breakpoints where f - h = 0
```

### Sign Function
```matlab
g = sign(f)
% Introduces breaks where f = 0
```

### Rounding Operations
```matlab
g = round(f)   % or floor(f), ceil(f)
% Introduces breaks based on rounding logic
```

### Mechanism

These commands use root finding to determine exactly where to place discontinuities:
- abs(f): zeros of f
- min/max(f,g): zeros of f - g
- sign(f): zeros of f
- round/floor/ceil: based on function value transitions

### Example
```matlab
f = chebfun('sin(pi*x)', [-1, 1])
g = abs(f)

% g has breakpoints at x = -1, 0, 1
plot(g)
% Notice the corners at the zeros of sin(pi*x)
```

### Piecewise Function Construction

```matlab
% Create piecewise function using min/max
f = chebfun('x.^2')
g = chebfun('1')
h = min(f, g)

% h equals x^2 for |x| <= 1, equals 1 for |x| > 1
plot(h)
```

## When to Use

- **diff(f)**: Compute first or higher-order derivatives
- **diff(f, n)**: Compute nth derivative
- **abs(f)**: Create function with absolute value, introduces corners at zeros
- **min/max(f,g)**: Create piecewise functions, introduces breaks at intersections
- **sign(f)**: Create sign function, introduces jumps at zeros
- **round/floor/ceil(f)**: Apply rounding operations, creates step functions

## Important Considerations

1. **Stability**: Repeated differentiation is ill-posed
2. **Delta functions**: Derivatives of discontinuous functions include delta functions
3. **Breakpoints**: Non-smooth operations automatically detect and place breakpoints
4. **Root finding**: Essential for determining where discontinuities occur