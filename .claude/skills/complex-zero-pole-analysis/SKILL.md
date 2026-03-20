---
name: complex-zero-pole-analysis
description: Count zeros and poles of complex functions within closed regions using the argument principle, and locate specific zeros using Cauchy integrals. Use when analyzing the distribution of roots or finding precise locations of zeros in the complex plane.
---

# Complex Zero and Pole Analysis

## Counting Zeros and Poles (Argument Principle)

### The Formula

For a function f(z) with no zeros or poles on the boundary:

```matlab
% Define closed contour as chebfun z
z = chebfun(@(s) exp(1i*s), [0, 2*pi]);  % Unit circle

% Define function f(z)
f = @(z) z.^3 - 1;

% Compute N = (Zeros - Poles)
N = sum((diff(f(z)) ./ f(z))) / (2 * 1i * pi);
```

This derives from the argument principle:
```
N - P = (1/2πi) ∮ (f'(z)/f(z)) dz
```

### Interpretation

- **N > 0**: More zeros than poles inside contour
- **N < 0**: More poles than zeros inside contour
- **N = 0**: Equal number of zeros and poles, or none

If f has no poles, N is the exact count of zeros.

### Alternative Method: Change in Argument

```matlab
% Compute argument change along contour
arg_vals = angle(f(z));
arg_change = unwrap(arg_vals(end)) - unwrap(arg_vals(1));
N = arg_change / (2 * pi);
```

## Locating Zeros via Cauchy Integrals

### Zero Location Formula

For a contour enclosing a single zero:

```matlab
% Define contour enclosing region of interest
z = chebfun(@(s) 0.5 + 0.5*exp(1i*s), [0, 2*pi]);

% Define function with zero inside
f = @(z) z.^2 - 1;

% Compute zero location
z0 = sum(z .* (diff(f(z)) ./ f(z))) / (2 * 1i * pi);
```

### Verification

```matlab
% Check result
f(z0)  % Should be ~0

% Compare with standard root finding
roots(f)  % Should include z0
```

## Workflow

1. **Define region**: Create closed contour z(s) around region of interest
2. **Count zeros**: Use argument principle to find N = Z - P
3. **Locate zeros**: If N > 0 and region is small, compute precise location
4. **Verify**: Evaluate f(z0) or compare with roots(f)

## When to Use

- Counting how many zeros a function has in a region
- Finding precise locations of complex zeros
- Analyzing root distribution of polynomials or analytic functions
- Verifying absence of zeros in stability regions
