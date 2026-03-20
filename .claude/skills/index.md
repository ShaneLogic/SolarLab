```markdown
# Chebfun Numerical Computing Skills

This collection of skills covers the Chebfun system for numerical computing with functions. Chebfun enables high-accuracy computations using Chebyshev polynomial approximations and spectral methods. These skills help users solve differential equations, perform complex analysis, compute integrals and derivatives, work with rational approximations, and conduct advanced mathematical operations on function representations.

## Available Skills

| Skill | Description | Use When |
|-------|-------------|----------|
| [chebfun-fundamentals](chebfun-fundamentals/SKILL.md) | Understand what chebfuns are, their philosophy, and how to construct them | Creating numerical representations of functions on intervals |
| [chebfun-resolution-strategies](chebfun-resolution-strategies/SKILL.md) | Configure sampling, splitting, and resolution preferences for chebfuns | Constructing chebfuns for functions with varying complexity, spikes, or piecewise smooth behavior |
| [chebfun-operations](chebfun-operations/SKILL.md) | Perform mathematical operations including integration, differentiation, and calculus operations | Computing integrals, derivatives, or applying functions to chebfuns |
| [chebfun-calculus-operations](chebfun-calculus-operations/SKILL.md) | Compute derivatives and apply non-smooth operations (abs, min, max, sign, etc.) | Computing derivatives or applying operations that introduce breakpoints |
| [chebfun-integration-quadrature](chebfun-integration-quadrature/SKILL.md) | Compute definite integrals using FFT-based Clenshaw-Curtis quadrature and specialized rules | Computing integrals, 2D integration, or applying specialized quadrature rules |
| [chebfun-analysis](chebfun-analysis/SKILL.md) | Compute global extrema and vector norms of chebfun objects | Analyzing behavior, bounds, or magnitude of functions |
| [chebfun-statistical-analysis](chebfun-statistical-analysis/SKILL.md) | Compute statistical measures including 2-norm, mean, standard deviation, and variance | Analyzing statistical properties of functions over intervals |
| [chebfun-root-finding-extrema](chebfun-root-finding-extrema/SKILL.md) | Find all zeros and identify local minima/maxima using the Boyd-Battles method | Locating all function zeros or finding extrema without explicit derivatives |
| [chebyshev-approximation](chebyshev-approximation/SKILL.md) | Work with Chebyshev series and interpolants, extract coefficients | Analyzing mathematical properties of Chebyshev approximations |
| [chebyshev-barycentric-interpolation](chebyshev-barycentric-interpolation/SKILL.md) | Evaluate Chebyshev polynomial interpolants using the barycentric formula | Evaluating Chebyshev interpolants at arbitrary points efficiently |
| [rational-approximation-methods](rational-approximation-methods/SKILL.md) | Compute rational function approximations p/q of type (m,n) for a given function | Approximating functions with rational functions or analyzing poles |
| [complex-root-finding](complex-root-finding/SKILL.md) | Find genuine complex roots using Chebfun ellipse filtering | Locating complex zeros of functions defined on real domains |
| [complex-zero-pole-analysis](complex-zero-pole-analysis/SKILL.md) | Count zeros and poles within closed regions using the argument principle | Analyzing distribution of roots or finding specific zeros |
| [complex-contour-integration](complex-contour-integration/SKILL.md) | Compute contour integrals along parameterized paths in the complex plane | Computing complex integrals or applying Cauchy's/residue theorems |
| [quasimatrix-linear-algebra](quasimatrix-linear-algebra/SKILL.md) | Perform linear algebra on quasimatrices (QR, SVD, least-squares) | Working with matrices of chebfuns or solving linear systems |
| [differential-equation-solver](differential-equation-solver/SKILL.md) | Solve linear and nonlinear differential equations using chebops | Solving boundary value problems, ODEs, integral equations, or PDEs |
| [chebop-eigenvalue-problems](chebop-eigenvalue-problems/SKILL.md) | Compute eigenvalues and eigenfunctions for differential operators | Solving eigenvalue problems or systems of coupled ODEs |
| [spectral-algorithms](spectral-algorithms/SKILL.md) | Configure spectral discretization methods for high-accuracy solving | Solving high-order differential equations or when default accuracy is insufficient |
| [frechet-derivative-computation](frechet-derivative-computation/SKILL.md) | Compute Fréchet derivatives using automatic differentiation | Determining variable dependencies or performing sensitivity analysis |
| [chebfun2-bibliographic-citation](chebfun2-bibliographic-citation/SKILL.md) | Extract bibliographic citations for Chebfun2 integration methods | Referencing the hybrid symbolic-numeric integration paper |

## Quick Navigation

### Getting Started
- **chebfun-fundamentals**: Learn the philosophy and construction of chebfuns
- **chebfun-resolution-strategies**: Configure sampling and splitting for complex functions

### Core Operations
- **chebfun-operations**: Basic mathematical operations on chebfuns
- **chebfun-calculus-operations**: Derivatives and non-smooth operations
- **chebfun-integration-quadrature**: Definite integrals and specialized quadrature

### Analysis & Statistics
- **chebfun-analysis**: Global extrema and vector norms
- **chebfun-statistical-analysis**: Mean, variance, and standard deviation
- **chebfun-root-finding-extrema**: Find zeros and local extrema

### Approximation Methods
- **chebyshev-approximation**: Chebyshev series and interpolants
- **chebyshev-barycentric-interpolation**: Efficient evaluation of Chebyshev interpolants
- **rational-approximation-methods**: Rational function approximations

### Complex Analysis
- **complex-root-finding**: Genuine complex roots with ellipse filtering
- **complex-zero-pole-analysis**: Count and locate zeros/poles using argument principle
- **complex-contour-integration**: Contour integrals and residue theorem

### Advanced Topics
- **quasimatrix-linear-algebra**: Linear algebra with chebfuns as columns/rows
- **differential-equation-solver**: Solve ODEs, BVPs, and PDEs
- **chebop-eigenvalue-problems**: Eigenvalue problems for differential operators
- **spectral-algorithms**: High-accuracy spectral discretization
- **frechet-derivative-computation**: Automatic differentiation for nonlinear operators

### Reference
- **chebfun2-bibliographic-citation**: Bibliographic information for Chebfun2 methods

## How to Use

These skills are designed to be invoked when working with Chebfun in MATLAB or Octave. When you need to:

1. **Create a function representation**: Start with `chebfun-fundamentals`
2. **Perform calculus operations**: Use `chebfun-calculus-operations` or `chebfun-integration-quadrature`
3. **Solve differential equations**: Use `differential-equation-solver` or `chebop-eigenvalue-problems`
4. **Analyze function properties**: Use `chebfun-analysis`, `chebfun-statistical-analysis`, or `chebfun-root-finding-extrema`
5. **Work with complex functions**: Use the complex analysis skills
6. **Handle difficult functions**: Configure with `chebfun-resolution-strategies` or `spectral-algorithms`

Each skill provides detailed guidance on syntax, parameters, and best practices for the specific operation.
```