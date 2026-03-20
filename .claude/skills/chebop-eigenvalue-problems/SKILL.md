---
name: chebop-eigenvalue-problems
description: Compute eigenvalues and eigenfunctions for linear differential operators, integral operators, and generalized eigenvalue systems using the overloaded eigs command. Handle systems of coupled ODEs using block operators. Use when solving spectral problems for differential equations, finding eigenmodes of operators, or working with coupled variable systems.
---

# Chebop Eigenvalue Problems

## Basic Eigenvalue Computation

Compute eigenvalues and eigenfunctions for linear operators.

### Basic Usage
```matlab
[V, D] = eigs(L, k)
```

### Parameters
- **L**: Linear chebop operator
- **k**: Number of eigenvalues to compute (optional, default: 6)
- **V**: Quasimatrix of eigenfunctions
- **D**: Diagonal matrix of eigenvalues

### Default Behavior
- Finds 6 eigenvalues if k not specified
- Targets eigenmodes that are 'most readily converged to' (approximately the smoothest ones)

### Example
```matlab
% Define operator: u'' = λu
L = chebop(-1, 1);
L.op = @(x,u) diff(u, 2);
L.bc = 'dirichlet';  % u(-1)=u(1)=0

% Compute 6 eigenvalues
[V, D] = eigs(L, 6);

% Plot eigenfunctions
plot(V)
legend('Mode 1', 'Mode 2', 'Mode 3', 'Mode 4', 'Mode 5', 'Mode 6')
```

## Generalized Eigenvalue Problems

Solve Au = λBu for two operators.

### Setup
```matlab
% Define two linear chebops
A = chebop(-1, 1);
B = chebop(-1, 1);

% Attach ALL boundary conditions to operator A
A.op = @(x,u) diff(u, 2);
A.bc = 'dirichlet';

B.op = @(x,u) u;
% B has no boundary conditions

% Compute generalized eigenvalues
[V, D] = eigs(A, B, k);
```

### Important Rule
Attach all boundary conditions to operator A, not B.

## Periodic Problems

```matlab
L = chebop(-pi, pi);
L.op = @(x,u) diff(u, 2);
L.bc = 'periodic';  % Set periodic boundary conditions

[V, D] = eigs(L, k);
```

## Block Operators and Systems of Equations

Handle coupled variables using block operators.

### Define Block Operator
```matlab
% System: u' = v, v' = -u
L = chebop(0, 2*pi);

% Define operator acting on [u; v]
L.op = @(x, u, v) [diff(u) - v; diff(v) + u];

% Boundary conditions
L.lbc = @(u, v) [u - 1; v];  % u(0)=1, v(0)=0
```

### Solve System
```matlab
% Solve boundary value problem
U = L \ 0;

% Extract individual variables
u = U(:,1);
v = U(:,2);

% Plot both variables
plot(U)
legend('u', 'v')
```

### Eigenvalue Problems for Systems

```matlab
% Define block operator for eigenvalue problem
L = chebop(0, 2*pi);
L.op = @(x, u, v) [diff(u) - v; diff(v) + u];
L.bc = 'periodic';

% Compute eigenvalues
[V, D] = eigs(L, k);

% V is quasimatrix with 2 columns per eigenmode
% First column: u component
% Second column: v component
```

### Visualization
```matlab
% Visualize operator structure
spy(L)

% Shows global vs local dependencies in the operator
```

## Constraints and Limitations

1. **Cannot find 'all' eigenvalues**: Operators have infinite spectrum
2. **Unreasonable requests may fail**: Asking for largest eigenvalue of differential operator may not converge
3. **Smoothest modes**: Default behavior targets smoothest eigenmodes

## Output Interpretation

### Standard Eigenvalue Problem
- **D**: Diagonal matrix of eigenvalues λ₁, λ₂, ..., λₖ
- **V**: Quasimatrix where column j is eigenfunction corresponding to λⱼ

### System of Equations
- **U**: ∞ × n quasimatrix (Chebfun with multiple columns)
- **U(:,j)**: Solution for variable j

### Generalized Eigenvalue Problem
- Same structure as standard problem
- Eigenvalues satisfy det(A - λB) = 0

## When to Use

- **eigs(L, k)**: Find eigenvalues/eigenfunctions of linear operator
- **eigs(A, B, k)**: Solve generalized eigenvalue problem Au = λBu
- **Block operators**: Handle systems of coupled ODEs
- **Periodic problems**: Set L.bc = 'periodic'
- **spy(L)**: Visualize operator structure for systems