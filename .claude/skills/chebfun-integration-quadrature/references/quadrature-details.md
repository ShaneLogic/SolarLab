# Detailed Quadrature Information

## Clenshaw-Curtis Quadrature Algorithm

The FFT-based Clenshaw-Curtis quadrature was first described in Gentleman (1972). It provides high accuracy for smooth functions.

## Performance Comparison

For computing erf(1):
- MATLAB's erf code: fastest
- MATLAB's quadrature commands: slower
- Chebfun: competitive timing

## Hale-Townsend Algorithms

Used for Gauss and Gauss-Jacobi quadrature:
- More accurate than Golub-Welsch algorithm
- Faster computation

## Example: Integral with Spikes

```matlab
% Function with three spikes, each ten times narrower
f = chebfun(@(x) spike_function(x))

% Splitting off: global polynomial, large length, correct integral
result1 = sum(f)

% Splitting on: may miss narrowest spike
f_split = chebfun(@(x) spike_function(x), 'splitting', 'on')
result2 = sum(f_split)  % May be too small

% Fix with minSamples
f_fixed = chebfun(@(x) spike_function(x), 'splitting', 'on', 'minSamples', 100)
result3 = sum(f_fixed)  % Correct with smaller length
```

## Infinite Interval Accuracy

WARNING: Operations involving infinities are not always as accurate/robust as finite counterparts. Several digits of accuracy may be lost.