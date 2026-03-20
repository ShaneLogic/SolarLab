---
name: chebyshev-approximation
description: Work with Chebyshev series and interpolants, including extracting coefficients, understanding convergence properties, and constructing fixed-length approximations. Use when analyzing the mathematical foundation of chebfun approximations or needing explicit polynomial representations.
---

# Chebyshev Approximation

## Chebyshev Points (Second Kind)

```matlab
% N+1 Chebyshev points on [-1, 1]
N = 10;
j = 0:N;
x_j = -cos(j * pi / N);

% Special case: N=0 gives x_0 = 0
```

## Chebyshev Polynomials

```matlab
% Chebyshev polynomial of degree N
T_N = @(x) cos(N * acos(x));

% Examples:
% T_0(x) = 1
% T_1(x) = x
% T_2(x) = 2x^2 - 1
```

## Chebyshev Series

A smooth function f has unique expansion:

```
f(x) = Σ_{k=0}^∞ a_k T_k(x)
```

### Coefficient Formula

```matlab
% For k > 0:
a_k = (2/pi) * integral_{-1}^1 (f(x) * T_k(x) / sqrt(1-x^2)) dx

% For k = 0:
a_0 = (1/pi) * integral_{-1}^1 (f(x) / sqrt(1-x^2)) dx
```

## Chebyshev Interpolant

Chebfun constructs approximations via interpolants:

```
f_N(x) = Σ_{k=0}^N c_k T_k(x)
```

### Convergence

- As N → ∞: c_k → a_k (ignoring rounding errors)
- For finite N: c_k ≠ a_k

## Extracting Coefficients

### Chebyshev Coefficients

```matlab
f = chebfun(@(x) exp(x), [-1, 1]);

% Get Chebyshev coefficients (high-order first)
coeffs_cheb = chebcoeffs(f);
% Returns row vector: [c_N, c_{N-1}, ..., c_0]
```

### Monomial Coefficients

```matlab
% Get coefficients in monomial basis 1, x, x^2, ...
coeffs_mono = poly(f);
% Returns row vector: [a_N, a_{N-1}, ..., a_0]
```

### Non-Polynomial Functions

For functions not "really" polynomials:
```matlab
% First entry is typically just above machine precision
f = chebfun(@(x) sin(x), [-1, 1]);
coeffs_cheb = chebcoeffs(f);
% coeffs_cheb(1) ≈ eps * scale(f)
```

## Fixed-Length Interpolation

### Constructing Fixed-Length Chebfuns

```matlab
% Force specific length N (non-adaptive)
f_fixed = chebfun(@(x) exp(x), [-1, 1], 20);
length(f_fixed)  % Returns 20
```

### Gibbs Phenomenon

For functions not well-approximated by polynomials:

```matlab
% Function with discontinuity
f = chebfun(@(x) sign(x), [-1, 1], 50);
plot(f)  % Shows overshoot near x=0

% Measure overshoot amplitude
overshoot = max(f);
```

**Note**: Gibbs overshoot persists as N → ∞.

## When to Use

- Understanding chebfun approximation theory
- Extracting explicit polynomial coefficients
- Analyzing approximation quality
- Constructing fixed-degree polynomial approximations
- Studying convergence of Chebyshev series
- Working with discontinuous functions (Gibbs phenomenon)
