# Chebfun Ellipse Theory

## Mathematical Definition

For a chebfun of length L on interval [-1, 1]:

### Joukowski Map

```
J(z) = (z + 1/z) / 2
```

This maps the unit circle to the interval [-1, 1].

### Ellipse Parameter

```
r^(-L) = delta
```

where:
- r > 1 is the radius parameter
- L is the chebfun length
- delta = sqrt(eps) is the tolerance parameter

### Chebfun Ellipse

The Chebfun ellipse is the image of the circle |z| = r under J(z):

```
E = {J(z) : |z| = r}
```

## Root Filtering

```matlab
% All roots (including spurious)
all_roots = roots(f, 'all');

% Filtered roots (genuine only)
genuine_roots = roots(f, 'complex');

% Equivalent manual filtering
r = eps^(-1/L);  % or delta^(-1/L)
inside_ellipse = abs(all_roots + sqrt(all_roots.^2 - 1)) <= r + 1/r;
genuine_roots = all_roots(inside_ellipse);
```

## Accuracy Considerations

The ellipse boundary represents where the polynomial approximation becomes unreliable:

- Inside ellipse: Roots likely genuine
- Outside ellipse: Roots likely spurious
- Near boundary: Accuracy decreases

## Example

```matlab
% Function with known complex roots
f = chebfun(@(x) x.^3 - 1, [-1, 1]);

% All polynomial roots
all_roots = roots(f, 'all');
% Returns many complex roots

% Genuine roots
real_roots = roots(f, 'complex');
% Returns only roots near the interval
```
