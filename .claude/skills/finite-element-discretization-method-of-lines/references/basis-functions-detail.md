# Basis Functions Detailed Reference

## Hat Function Definition

For a grid with points x_0, x_1, ..., x_N, x_{N+1}, the basis function φ_i(x) for interior point i is:

```
φ_i(x) = { (x - x_{i-1}) / h_{i-1}    for x ∈ [x_{i-1}, x_i]
           { (x_{i+1} - x) / h_i        for x ∈ [x_i, x_{i+1}]
           { 0                        otherwise
```

where h_{i-1} = x_i - x_{i-1} and h_i = x_{i+1} - x_i.

## Properties
- Compact support: φ_i(x) is non-zero only on [x_{i-1}, x_{i+1}]
- Partition of unity: Σ φ_i(x) = 1 for all x in the domain
- Piecewise linear interpolation at grid points: w(x_j) = Σ w_i(t) * φ_i(x_j) = w_j(t)

## Derivative of Basis Functions

```
φ'_i(x) = { 1/h_{i-1}    for x ∈ (x_{i-1}, x_i)
         { -1/h_i      for x ∈ (x_i, x_{i+1})
         { 0          otherwise
```