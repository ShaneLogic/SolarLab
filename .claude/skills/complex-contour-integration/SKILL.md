---
name: complex-contour-integration
description: Compute contour integrals of complex functions along parameterized paths in the complex plane. Apply Cauchy's theorem for analytic functions and the residue theorem for meromorphic functions. Use when integrating complex functions along specific paths or around closed contours.
---

# Complex Contour Integration

## Computing Contour Integrals

### Define the Contour

Parameterize the contour as a chebfun z(s) where s is a real parameter:

```matlab
% Define contour parameter s over [0, 2*pi]
s = chebfun('s', [0, 2*pi]);

% Define contour z(s) as complex function of s
z = exp(1i * s);  % Unit circle

% Or define piecewise contours using join
z1 = chebfun(@(s) s, [0, 1]);  % Line from 0 to 1
z2 = chebfun(@(s) 1 + 1i*s, [0, 1]);  % Vertical line from 1 to 1+i
z = join(z1, z2);  % Combined contour
```

### Compute the Integral

```matlab
% Define the function f(z)
f = @(z) exp(-z.^2);

% Compute contour integral
I = sum(f(z) .* diff(z));
```

This implements the mathematical definition:
```
∫ f(z) dz = ∫ f(z(s)) z'(s) ds
```

## Cauchy's Theorem

### For Closed Contours

If f(z) is analytic everywhere inside and on a closed contour:

```matlab
% The integral equals zero
I = sum(f(z) .* diff(z));  % Result should be ~0
```

### Path Independence

For analytic functions, integrals between two points are path-independent:

```matlab
% Define two different paths from z1 to z2
path1 = chebfun(@(s) z1 + s*(z2-z1), [0, 1]);  % Straight line
path2 = join(...);  % Piecewise path

% Both integrals should be equal
I1 = sum(f(path1) .* diff(path1));
I2 = sum(f(path2) .* diff(path2));
% I1 ≈ I2
```

## Residue Theorem

### For Meromorphic Functions

If f(z) is meromorphic (analytic except for poles) inside a closed contour:

```matlab
% Compute contour integral
I = sum(f(z) .* diff(z));

% Sum of residues
SumResidues = I / (2 * pi * 1i);
```

### Residue Definition

The residue of f at z0 is the coefficient of the (z-z0)^(-1) term in the Laurent expansion.

**Example**: For f(z) = exp(z)/z^3:
- Laurent series: z^(-3) + z^(-2) + (1/2)z^(-1) + ...
- Residue at z=0: 1/2

### Application: Bernoulli Numbers

Bernoulli number B_k = k! × (k-th Taylor coefficient of z/(exp(z)-1))

```matlab
% Can be computed via residue theorem
f = @(z) z ./ (exp(z) - 1);
% Extract coefficient via contour integration
```

## When to Use

- Integrating complex functions along specific paths
- Verifying analyticity via Cauchy's theorem
- Computing residues of meromorphic functions
- Evaluating integrals using residue theorem
- Checking path independence of integrals
