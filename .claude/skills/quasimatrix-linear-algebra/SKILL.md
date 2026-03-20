---
name: quasimatrix-linear-algebra
description: Perform linear algebra operations on quasimatrices (matrices with chebfuns as columns/rows), including QR factorization, least-squares solving via backslash, singular value decomposition (SVD), 2-norm computation, and condition number calculation. Use when working with systems of chebfuns, solving continuous least-squares problems, or analyzing the numerical properties of function spaces.
---

# Quasimatrix Linear Algebra

## Quasimatrix Fundamentals

A quasimatrix is analogous to a MATLAB matrix but with chebfuns as columns or rows.

### Construction
```matlab
% Column-oriented (default)
A = [chebfun('1'), chebfun('x'), chebfun('x.^2')]

% Row-oriented
B = [chebfun('sin(x)'); chebfun('cos(x)')]
```

### Basic Operations
```matlab
% Transpose
At = A'

% Inner product matrix
G = A' * A

% Access columns
col1 = A(:,1)
col2 = A(:,2)
```

## QR Factorization

Compute reduced (economy-size) QR factorization for quasimatrices.

### Basic Usage
```matlab
[Q, R] = qr(A, 0)
```

### Factorization Form
- A is m x n quasimatrix (m > n)
- Q is m x n quasimatrix with orthonormal columns
- R is n x n upper-triangular matrix
- Satisfies A = QR

### Computation Method
Uses quasimatrix analogue of Householder triangularization.

### Renormalization (e.g., for Legendre polynomials)
```matlab
% Impose P(1)=1 instead of norm 1
for j = 1:size(A,2)
    R(j,:) = R(j,:) * Q(1,j)
    Q(:,j) = Q(:,j) / Q(1,j)
end
```

### Inverse Interpretation
If A = QR, then A * R^(-1) = Q. Column k of R^(-1) contains coefficients for expanding column k of Q as a linear combination of columns of A.

## Backslash and Least-Squares

Solve linear systems or compute least-squares solutions.

### Basic Usage
```matlab
c = A \ b
```

### Behavior by Matrix Shape

**Square matrix:** Solves Ac = b

**Rectangular (more rows than columns):** Computes least-squares solution minimizing ||Ac - b||

### Continuous Least-Squares
For quasimatrices, this is a continuous computation involving integrals, not point evaluations.

### Interpretation
Vector c contains coefficients for the least-squares fit of b by a linear combination of columns of A.

### Property
The least-squares approximation by a polynomial of degree n to a continuous function f must intersect f at least n+1 times in the interval.

## Singular Value Decomposition (SVD)

Compute SVD for quasimatrices.

### Basic Usage
```matlab
[U, S, V] = svd(A)
```

### Factorization
- A = USV* (or AV = US)
- U is m x n with orthonormal columns
- S is diagonal with non-increasing non-negative singular values
- V is n x n orthogonal

### Geometric Interpretation
A maps the unit ball in R^n to a hyperellipsoid of dimension ≤ n.

## 2-Norm and Condition Number

### 2-Norm
```matlab
% CRITICAL: Must specify argument 2 for quasimatrices
norm_A = norm(A, 2)
```

**WARNING:** For quasimatrices, `norm(A)` defaults to Frobenius norm, not 2-norm.

### Condition Number
```matlab
cond_A = cond(A)
```

### Interpretation
- cond(A) >> 1: Ill-conditioned (e.g., monomials 1, x, ..., x^5)
- cond(A) = 1: Orthonormal basis (e.g., normalized Legendre polynomials)
- Chebyshev polynomials: Condition number close to 1

### Definition
Condition number is the ratio of largest to smallest singular value (eccentricity of hyperellipsoid).

## When to Use

- **Quasimatrices**: Organize multiple chebfuns into matrix structure
- **QR**: Orthogonalize function bases, compute orthonormal columns
- **Backslash**: Solve continuous least-squares problems
- **SVD**: Analyze numerical properties, compute low-rank approximations
- **norm(A,2)**: Compute operator norm for quasimatrices
- **cond(A)**: Assess numerical stability of function bases