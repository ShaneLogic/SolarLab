---
name: Solid-State Bonding Analysis
description: Analyze and calculate bonding properties in crystalline solids. Use this skill when determining bond type (metallic, ionic, covalent, or mixed), calculating equilibrium distances, potential energies, lattice energies, or predicting structural properties based on bonding mechanisms.
---

# Solid-State Bonding Analysis

Determine bonding type and calculate associated energies and structural properties for crystalline materials.

## When to Use

- Analyzing bonding mechanisms in metals, ionic crystals, or covalent solids
- Calculating equilibrium distances between atoms
- Computing potential energy curves for ionic bonds
- Determining lattice binding energies via Madelung constants
- Applying Born-Haber cycle for thermodynamic lattice energy
- Predicting crystal structures from bonding characteristics
- Analyzing mixed ionic-covalent bonding in semiconductors

## Procedure

### 1. Identify Bonding Type

Classify the material based on constituent elements:

| Material Type | Bonding Character | Key Features |
|---------------|-------------------|--------------|
| Simple metals (alkali) | Metallic | Electron sea, nondirectional |
| Group I-VII compounds | Ionic | Charge transfer, electrostatic |
| Group IV elements | Covalent | Electron sharing, directional |
| III-V, II-VI compounds | Mixed | Partial ionicity |

### 2. Metallic Bonding Analysis

For simple metals (alkali metals):

1. **Model Mechanism**: Each atom donates valence electron(s) to form a delocalized electron sea
2. **Force Balance**: Attractive electron-ion interactions balance repulsive electron-electron and ion-ion interactions
3. **Key Properties**:
   - Nondirectional bonding (no preferred orientation)
   - Non-saturable bonding (not limited to specific neighbors)
   - Results in close-packed structures (coordination 8 or 12)
   - Low binding energy: ~1 eV/atom
   - High compressibility, mechanically soft

### 3. Ionic Bonding Calculations

#### Equilibrium Distance
```
re = rA + rB
```
Where rA and rB are characteristic atomic radii.

#### Potential Energy (Coulomb-Born Model)
```
V(r) = -(q²/r) + (B/r^m)
```
- First term: Coulomb attraction
- Second term: Born repulsion
- m: Born exponent (typically ~9, determined empirically)

#### Lattice Energy with Madelung Constant
```
U = (A × q² / r) × (1 - 1/m)
```
Where A is the Madelung constant (see references for values).

#### Born-Haber Cycle
Calculate lattice energy from experimental data:
```
H₀ = W_solid - [W_subl + W_ion + (1/2 × W_diss) + W_elaff]
```

### 4. Covalent Bonding Analysis

#### Bonding vs Antibonding States

Check spin alignment of approaching unpaired electrons:

- **Antiparallel spins** → Bonding state (Ψ+ = ΨA + ΨB)
  - Increased electron density in overlap region
  - Attractive force, lowest energy state

- **Parallel spins** → Antibonding state (Ψ- = ΨA - ΨB)
  - Electron density vanishes between atoms
  - Repulsive force, higher energy state

#### sp³ Hybridization (Group IV elements)

1. Promotion from s²p² to sp³ configuration
2. Linear combination: σi = 1/2(φs ± φpx ± φpy ± φpz)
3. Result: 4 tetrahedrally oriented orbitals (109.47° bond angle)

#### Valency-Structure Relationship
| Valency | Structure Type | Example |
|---------|----------------|----------|
| Monovalent | Diatomic molecules | H₂ |
| Divalent | Chains | S, Se |
| Trivalent | Layered lattices | As |
| Tetravalent | 3D networks | Si, C |

### 5. Mixed Bonding (III-V, II-VI Compounds)

Express as linear combination:
```
Ψ = aΨcov + bΨion
```

- Ionicity ratio b/a increases with electronegativity difference
- Electron pair shifted toward more electronegative atom

## Output

Provide:
1. Bond type classification
2. Calculated equilibrium distance and/or binding energy
3. Predicted structural properties
4. Applicable formulas with numerical results