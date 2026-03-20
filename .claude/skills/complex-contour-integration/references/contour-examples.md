# Contour Integration Examples

## Example 1: Integral of exp(-z^2)

```matlab
% Contour from 0 to -1+0.5i
s = chebfun('s', [0, 1]);
z = s * (-1 + 0.5i);

% Compute integral
I = sum(exp(-z.^2) .* diff(z));
```

## Example 2: Unit Circle Integral

```matlab
% Unit circle parameterization
s = chebfun('s', [0, 2*pi]);
z = exp(1i * s);

% Integral of 1/z around unit circle
I = sum((1./z) .* diff(z));
% Result: 2*pi*i
```

## Example 3: Verifying Cauchy's Theorem

```matlab
% Analytic function: sin(z)
f = @(z) sin(z);

% Closed contour: unit circle
s = chebfun('s', [0, 2*pi]);
z = exp(1i * s);

% Integral should be zero
I = sum(f(z) .* diff(z));
% Result: ~0 (within numerical tolerance)
```
