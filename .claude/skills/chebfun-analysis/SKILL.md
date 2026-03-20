---
name: chebfun-analysis
description: Compute global extrema (minimum and maximum values) and vector norms (1-norm and infinity-norm) of chebfun objects. Use when analyzing the behavior, bounds, or magnitude of functions represented as chebfuns.
---

# Chebfun Analysis Operations

## Global Extrema

### Finding Minimum and Maximum Values

```matlab
% Create chebfun
f = chebfun(@(x) sin(x) + 0.1*x.^2, [-5, 5]);

% Get global minimum value
min_val = min(f);

% Get global maximum value  
max_val = max(f);
```

### Finding Both Value and Location

```matlab
% Get minimum value and its location
[min_val, min_pos] = min(f);

% Get maximum value and its location
[max_val, max_pos] = max(f);
```

### How It Works

Chebfun computes extrema by:
1. Evaluating f at all interval endpoints
2. Finding all zeros of the derivative f'
3. Evaluating f at these critical points
4. Selecting the global minimum/maximum

### Important Notes

- Only **one** position is returned even if extrema occur at multiple points
- Computing `min(f)` and `max(f)` separately is inefficient for large chebfuns
- Each call independently computes derivative and finds its zeros

### Efficiency Tip

For large chebfuns, use combined computation:
```matlab
% Faster alternative (if available)
[min_val, max_val, min_pos, max_pos] = minandmax(f);
```

## Vector Norms

### 1-Norm (Integral of Absolute Value)

```matlab
% Compute 1-norm
one_norm = norm(f, 1);
```

- Definition: `||f||_1 = ∫ |f(x)| dx`
- Chebfun computes by summing segments between zeros of f
- At zeros of f, |f(x)| typically has discontinuous slope

### Infinity-Norm (Maximum Absolute Value)

```matlab
% Compute infinity-norm
inf_norm = norm(f, inf);
```

- Definition: `||f||_inf = max |f(x)|`
- Computed as: `max(max(f), -min(f))`

### Comparison with Other Norms

```matlab
% For single chebfuns, 2-norm equals Frobenius norm
two_norm = norm(f, 2);
fro_norm = norm(f, 'fro');
% two_norm == fro_norm for single chebfuns
```

**Note**: For quasimatrices, 2-norm and Frobenius norm differ.

## When to Use

- Finding global bounds of a function
- Locating extreme points of a function
- Computing function magnitude measures
- Analyzing function size for error estimation
- Comparing function sizes in optimization
