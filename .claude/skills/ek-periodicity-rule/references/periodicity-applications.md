# E(k) Periodicity: Extended vs Reduced Zone

## Zone Schemes

### Extended Zone Scheme:
- Plot E(k) without applying periodicity
- Multiple bands appear in different k-ranges
- Each band is continuous and distinct

### Reduced Zone Scheme:
- Fold all bands into first Brillouin zone (-π/a to π/a)
- Uses periodicity E(k + 2π/a) = E(k)
- All bands appear within the same k-range

### Periodic Zone Scheme:
- Repeat first Brillouin zone pattern periodically
- Visualizes periodicity explicitly

## Example: 1D Monatomic Chain

For a simple 1D chain with lattice constant a:

**First Brillouin zone:** k ∈ [-π/a, π/a]

**Equivalent states:**
- k = 0.3π/a is equivalent to:
  - k = 0.3π/a + 2π/a = 2.3π/a
  - k = 0.3π/a - 2π/a = -1.7π/a
  - k = 0.3π/a + n(2π/a) for any integer n

## Relationship to Reciprocal Lattice

The periodicity vector G = 2π/a is the smallest reciprocal lattice vector in 1D.

In 3D, the generalization is:
```
E(k + G) = E(k)
```
where G is any reciprocal lattice vector.

## Physical Consequences

### Band Degeneracies:
- At zone boundaries (k = ±π/a), bands can touch or be separated by gaps
- Periodic boundary conditions create standing waves

### Band Folding:
- Larger real-space unit cells → smaller Brillouin zones
- Bands "fold" into the reduced zone according to periodicity
- Used to analyze superlattices and artificial periodic structures

### Umklapp Scattering:
- Electron scattering that involves reciprocal lattice vector
- k → k' + G where G ≠ 0
- Only possible because of periodicity