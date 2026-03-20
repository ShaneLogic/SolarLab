# Field-of-Direction Method: Detailed Procedures

## Mathematical Foundation

The field-of-direction method transforms a system of ordinary differential equations into a geometric problem in the phase plane.

### Original System
```
dn/dx = (1/μkT) * [j_n - eμnF]    (Transport)
dF/dx = (e/ε) * (n - N_D + N_A)   (Poisson)
```

### Phase Plane Interpretation

In the n-F plane:
- Each point represents a possible state (n, F)
- The direction vector [dn/dx, dF/dx] is unique at each point
- Solution curves follow these directions

## Constructing Auxiliary Curves

### Neutrality Curve n1(F)

Set dF/dx = 0:
```
n1(F) = N_D - N_A + trapped_charge(F)
```

For simple case:
```
n1(F) = N_D (constant)
```

With field quenching:
```
n1(F) = N_D - n_trapped(F)
```

### Drift Current Curve n2(F)

Set dn/dx = 0:
```
n2(F) = j_n / (eμF)
```

This represents constant current condition.

## Singular Point Analysis

### Singular Point I (Bulk)

Location: Intersection of n1(F) and n2(F)

Characteristics:
- Both derivatives zero
- Stable equilibrium
- Represents bulk semiconductor state

### Singular Point II (High-Field Domain)

Location: Second intersection when n1(F) decreases

Characteristics:
- Both derivatives zero
- Can be stable or unstable
- Represents domain boundary

## Solution Curve Topology

### For Blocking Cathode

1. Start: (nc, Fc) with nc < n1(Fc)
2. Curve rises through first quadrant
3. Crosses neutrality curve vertically
4. Enters second quadrant
5. Approaches singular point I

### For High-Field Domain

1. Solution extends from cathode at constant F
2. Maintains singular point II condition
3. Drops rapidly near anode
4. Approaches singular point I

## Three-Dimensional Extension

The full solution exists in (n, F, x) space:
- n-F plane: Phase portrait
- x-axis: Spatial coordinate
- Solution curve: Trajectory in 3D space

The field-of-direction is a projection onto the n-F plane.

## Practical Application Steps

1. **Plot n1(F)**: Calculate neutrality curve for the system
2. **Plot n2(F)**: Calculate drift current curve for given current
3. **Identify singular points**: Find intersections
4. **Draw direction arrows**: Sample the n-F plane
5. **Trace solution**: Follow arrows from boundary to bulk
6. **Check consistency**: Verify crossing rules are satisfied