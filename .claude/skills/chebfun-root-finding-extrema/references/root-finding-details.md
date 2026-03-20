# Colleague Matrix Method

For each small piece of the chebfun, zeros are computed as eigenvalues of a colleague matrix. This is the analogue of the companion matrix for Chebyshev polynomials.

## Recursive Decomposition

When chebfun degree > 50:
1. Function is broken into smaller pieces
2. Process repeats recursively on each piece
3. Results combined

## Non-smooth Extrema Detection

For non-smooth functions, extrema are identified where:
- The derivative has a discontinuity
- The sign of the derivative changes across the discontinuity
- This corresponds to a "zero" of the derivative in a generalized sense

## Example: Intersection of Curves

```matlab
% Find where x = cos(x)
f = chebfun('x')
g = chebfun('cos(x)')
intersections = roots(f - g)

% Verify
fprintf('Intersection at x = %.15f\n', intersections)
fprintf('f(x) = %.15f, g(x) = %.15f\n', f(intersections), g(intersections))
```