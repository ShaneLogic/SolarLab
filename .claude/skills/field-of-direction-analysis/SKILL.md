---
name: Field-of-Direction Analysis
description: Apply graphical field-of-direction method to solve coupled transport and Poisson equations in semiconductors. Use when analyzing Schottky barriers, high-field domain formation, or seeking an alternative to numerical integration for semiconductor device equations.
---

# Field-of-Direction Analysis

## When to Use
- Solving coupled transport and Poisson equations graphically
- Analyzing Schottky barrier solutions
- Identifying high-field domain formation conditions
- Understanding singular points in semiconductor systems
- Seeking physical insight without full numerical integration

## Prerequisites
- Transport equation formulated for the device
- Poisson equation formulated
- Boundary conditions specified

## Procedure

### 1. Formulate the Equation System

For one-dimensional electron transport:

Transport equation:
```
dn/dx = f1(n, F)
```

Poisson equation:
```
dF/dx = f2(n, F)
```

### 2. Construct the Field-of-Directions

Project solution curves n(x) and F(x) into the n-F plane:
- Each point (n, F) has a unique direction angle
- Represent directions as short arrows in the n-F plane
- This creates the "field-of-directions" map

### 3. Identify Auxiliary Curves

**Neutrality curve n1(F)**:
- Where dF/dx ≡ 0
- Defines charge neutrality condition

**Drift current curve n2(F)**:
- Where dn/dx ≡ 0
- Defines constant current condition

These curves divide the n-F plane into four quadrants.

### 4. Analyze Quadrant Behavior

| Quadrant | Direction Arrow Behavior |
|----------|------------------------|
| First | Points up and right |
| Second | Points up and left |
| Third | Points down and left |
| Fourth | Points down and right |

**Crossing rules**:
- Neutrality curve: Can only be crossed vertically
- Drift current curve: Can only be crossed horizontally

### 5. Trace Solution Curves

For Schottky barrier analysis:

1. Start at boundary condition (e.g., nc < n1 for blocking electrode)
2. Follow the direction arrows
3. Select Fc precisely so solution arrives at bulk condition
4. Bulk condition: dn/dx = dF/dx ≡ 0

### 6. Identify Singular Points

**Singular Point I**:
- Intersection of neutrality and drift current curves
- Represents bulk equilibrium
- Constant electron density and field

**Singular Point II** (high-field domain):
- Appears when n1(F) decreases due to field quenching
- Forms when drift current curve crosses n1(F) again
- Indicates high-field domain development

### 7. Analyze High-Field Domain

When field quenching occurs:
- n1(F) curve decreases at higher fields
- If decrease is over-linear, high-field domain must form
- Trigger: n2(F) crosses n1(F) at given nc

Domain characteristics:
- Drift current curve stuck at singular point II
- Current saturates
- Solution extends flat from cathode
- Field drops within few Debye lengths to bulk (singular point I)

## Output
- Graphical solution showing n(x) and F(x) profiles
- Singular point locations
- Domain formation conditions
- Physical interpretation of solution behavior

## Advantages Over Numerical Integration

1. Provides physical insight into solution topology
2. Identifies critical points graphically
3. Shows domain formation conditions clearly
4. Reveals solution existence and uniqueness properties

## Constraints
- Requires graphical interpretation skills
- Deviates from classical numerical approaches
- Best suited for qualitative understanding
- Quantitative accuracy may require supplementary calculations