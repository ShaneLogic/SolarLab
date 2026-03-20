# Quasimatrix Norms - Detailed Reference

## 1-Norm Details

The 1-norm of a quasimatrix A is defined as the maximum column sum:

```
||A||_1 = max_j ||column_j||_1
```

For a column quasimatrix with columns c_1, c_2, ..., c_n:
```matlab
norm(A, 1) = max([norm(c_1, 1), norm(c_2, 1), ..., norm(c_n, 1)])
```

## Infinity-Norm Details

The infinity-norm is defined as the maximum row sum:

```
||A||_inf = max_i |row_i|
```

For a column quasimatrix, this becomes:
```matlab
norm(A, inf) = norm(sum(abs(A), 2), inf)
```

Example with columns 1, x, x^2, ..., x^5:
```matlab
A = chebfun(@(x) [1, x, x.^2, x.^3, x.^4, x.^5]);
norm(A, inf)  % Returns max of 1 + |x| + |x|^2 + ... + |x|^5 on [-1,1]
```

## Frobenius Norm Details

The Frobenius norm is computed via singular values:

```
||A||_F = sqrt(sum(σ_i^2))
```

where σ_i are the singular values of A.

## Limitations

- Condition number with 1-norm or inf-norm is not defined
- These norms rely on continuous integration, not discrete summation
