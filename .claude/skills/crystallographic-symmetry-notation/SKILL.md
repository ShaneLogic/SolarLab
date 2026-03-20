---
name: Crystallographic Symmetry Notation
description: Decode and interpret Hermann-Mauguin symmetry notation for crystallographic point groups and space groups. Use this skill when encountering symmetry symbols like '4/m', '3m', or '6/mmm' and needing to identify the rotational axes, mirror planes, and inversion centers they represent.
---

# Crystallographic Symmetry Notation

Decode Hermann-Mauguin symmetry symbols to identify crystallographic symmetry elements.

## When to Use

- Interpreting crystallographic point group symbols
- Analyzing molecular symmetry notation
- Converting symmetry descriptions to structural features
- Identifying symmetry operations in crystal structures

## Symbol Components

| Component | Meaning |
|-----------|---------|
| n (number) | n-fold rotational symmetry axis |
| m | Mirror plane |
| n̄ (overbar) | Inversion axis (roto-inversion) |
| I | Inversion center |

## Decoding Procedure

### Step 1: Identify Components

Parse the symbol into numbers (n) and letters (m).

### Step 2: Interpret Numbers

- **n** indicates rotational symmetry
- **n̄** (overbar) indicates roto-inversion
- The position of each number refers to a specific crystallographic direction

### Step 3: Interpret Mirror Planes

Determine mirror plane orientation based on position relative to rotational axis:

| Notation | Meaning |
|----------|---------|
| m (without subscript) | Mirror plane parallel to rotational axis |
| mₙ | Mirror plane perpendicular to n-fold axis |
| Repeated m | Symmetry about other orthogonal planes |

### Step 4: Analyze Symbol Structure

For multi-position symbols:
- **First position**: Primary axis symmetry
- **Second position**: Secondary axes or planes
- **Third position**: Tertiary symmetry elements

## Common Examples

| Symbol | Interpretation |
|--------|---------------|
| 1 | No symmetry (triclinic) |
| 1̄ | Inversion center only |
| m | Single mirror plane |
| 2 | Twofold rotation axis |
| 2/m | Twofold axis with perpendicular mirror |
| 3m | Threefold axis with parallel mirrors |
| 4/mmm | Fourfold axis, perpendicular mirror, two sets of parallel mirrors |
| 6/mmm | Sixfold axis with full hexagonal symmetry |

## Output

Provide:
1. List of symmetry elements present
2. Orientation of each mirror plane (parallel/perpendicular to axis)
3. Order of rotational axes
4. Presence or absence of inversion center