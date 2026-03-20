# Norm Computation Details

## 1-Norm Implementation

The 1-norm is computed as:

```matlab
% Chebfun internally:
% 1. Find zeros of f
zeros_f = roots(f);
% 2. Add interval endpoints
breakpoints = [domain(f)(1), zeros_f, domain(f)(2)];
% 3. Integrate |f| on each segment
one_norm = 0;
for i = 1:length(breakpoints)-1
    segment = chebfun(@(x) abs(f(x)), [breakpoints(i), breakpoints(i+1)]);
    one_norm = one_norm + sum(segment);
end
```

## Infinity-Norm Formula

```matlab
inf_norm = max([max(f), -min(f)]);
```

This handles both positive and negative extrema correctly.

## Efficiency Considerations

For large chebfuns:

```matlab
% Inefficient - computes derivative twice
min_val = min(f);
max_val = max(f);

% More efficient - single derivative computation
[min_val, max_val] = minandmax(f);
```

## Multiple Extrema

If f attains its minimum at multiple points:

```matlab
f = chebfun(@(x) cos(x), [0, 4*pi]);
[min_val, min_pos] = min(f);
% min_pos returns only one of the locations
% (e.g., pi, not 3*pi)
```
