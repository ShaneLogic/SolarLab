# Chebyshev Approximation - Detailed Theory

## Chebyshev Points Derivation

Chebyshev points of the second kind:

```
x_j = cos(jπ/N), j = 0, 1, ..., N
```

These are the extrema of T_N(x) on [-1, 1].

## Orthogonality

Chebyshev polynomials are orthogonal with weight 1/√(1-x²):

```
∫_{-1}^1 T_m(x) T_n(x) / √(1-x²) dx =
    0, if m ≠ n
    π, if m = n = 0
    π/2, if m = n > 0
```

## Series vs Interpolant

### Series Coefficients (a_k)

```
a_k = (2/π) ∫_{-1}^1 f(x) T_k(x) / √(1-x²) dx
```

### Interpolant Coefficients (c_k)

Obtained by discrete cosine transform (DCT) of function values at Chebyshev points.

### Relationship

```
c_k = a_k + O(ρ^{-k})
```

where ρ > 1 depends on the analyticity region of f.

## Gibbs Phenomenon

For functions with jump discontinuities:

- Overshoot amplitude: ~9% of jump height
- Location: Near discontinuity
- Does not converge to zero as N → ∞

```matlab
% Example: sign function
f = chebfun(@(x) sign(x), [-1, 1], 100);
max(f)  % ≈ 1.09 (overshoot)
```

## Coefficient Decay

For analytic functions:
```
|a_k| ~ C ρ^{-k}
```

For functions with limited smoothness:
```
|a_k| ~ C k^{-α-1}
```

where α is the number of continuous derivatives.
