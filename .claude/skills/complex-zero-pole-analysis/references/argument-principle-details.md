# Argument Principle - Mathematical Details

## Derivation

The argument principle states:

```
N - P = (1/2πi) ∮ (f'(z)/f(z)) dz
```

where:
- N = number of zeros (counting multiplicity)
- P = number of poles (counting multiplicity)
- Contour encloses the region of interest
- f(z) has no zeros or poles on the contour

## Chebfun Implementation

In Chebfun, with z parameterized by s:

```matlab
% f'(z) = (df/ds)(ds/dz)
% f'(z)/f(z) dz = (df/ds)/f * ds

% Therefore:
N = sum(diff(f(z)) ./ f(z)) / (2 * 1i * pi);
```

## Zero Location Formula

The location of a zero z0 inside a contour is:

```
z0 = (1/2πi) ∮ z * (f'(z)/f(z)) dz
```

In Chebfun:
```matlab
z0 = sum(z .* (diff(f(z)) ./ f(z))) / (2 * 1i * pi);
```

## Example: Polynomial Roots

```matlab
% Count roots of z^3 - 1 inside unit circle
z = chebfun(@(s) exp(1i*s), [0, 2*pi]);
f = @(z) z.^3 - 1;
N = sum((diff(f(z)) ./ f(z))) / (2 * 1i * pi);
% Result: 3 (all three roots on unit circle)
```
