---
name: chebfun-integration-quadrature
description: Compute definite integrals of chebfun objects using FFT-based Clenshaw-Curtis quadrature, handle 2D integration over rectangular domains, and apply specialized quadrature rules (Gauss, Gauss-Jacobi) for high-precision numerical integration. Use when you need to integrate smooth functions, handle piecewise smooth integrands, or work with orthogonal polynomial quadrature nodes and weights.
---

# Chebfun Integration and Quadrature

## Definite Integration via sum Command

Use `sum(f)` to compute the definite integral of a chebfun over its domain.

### Basic Usage
```matlab
integral_value = sum(f)
```

### Algorithm
- Uses FFT-based version of Clenshaw-Curtis quadrature
- Applied to each smooth piece (fun) of the chebfun
- Results summed across all pieces

### Handling Special Cases

**Piecewise smooth functions:**
```matlab
f = chebfun('abs(x)', [-1, 1], 'splitting', 'on')
integral_value = sum(f)
```

**Functions with narrow spikes:**
```matlab
% Use minSamples flag to avoid missing narrow features
f = chebfun(@(x) spike_function(x), 'minSamples', 1000)
integral_value = sum(f)
```

**Infinite intervals:**
```matlab
f = chebfun(@(x) 1./x.^4, [1, inf])
integral_value = sum(f)  % May lose several digits of accuracy
```

**Divergent functions:**
```matlab
f = chebfun(@(x) 1./x.^0.9, [-1, 1], 'exps')
integral_value = sum(f)
```

## 2D Integration over Rectangles

### Basic Method
```matlab
% Construct chebfun with vectorize flag for 2D function
f = chebfun(@(x,y) function_handle(x,y), 'vectorize')

% Compute double integral
result = sum(sum(f))
```

### Recommendation
For better performance and plotting capabilities, use Chebfun2 instead of nested 1D chebfuns.

## Specialized Quadrature Rules

### Legendre Quadrature
```matlab
% Get quadrature nodes and weights
[s, w] = legpts(n)

% Compute integral
I = w * f(s)
```

### Chebyshev Quadrature
```matlab
% Get Chebyshev points
x = chebpts(n)

% Compute integral using built-in sum
I = sum(f)
```

### Jacobi Quadrature (for singularities at endpoints)
```matlab
% Get Jacobi quadrature nodes and weights
[s, w] = jacpts(n, alpha, beta)

% Compute integral
I = w * f(s)
```

### General Intervals
All quadrature operators work on general intervals `[a, b]`, not just `[-1, 1]`.

## When to Use Each Method

- **sum(f)**: Standard definite integration of chebfun objects
- **2D integration**: Use Chebfun2 for best results
- **Gauss quadrature**: High-precision integration with known polynomial basis
- **Jacobi quadrature**: Functions with endpoint singularities