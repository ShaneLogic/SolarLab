---
name: cdte-solar-cell-back-contact
description: Configure and process back electrodes for CdS/CdTe solar cells. Use this skill when fabricating back contacts, optimizing copper layer thickness, selecting between dry/wet processing methods, or performing pre-contact annealing for CdS/CdTe solar cell structures.
---

# CdTe Solar Cell Back Contact Configuration

## When to Use This Skill
Apply this procedure when:
- Fabricating back contacts for CdS/CdTe solar cells
- Optimizing cell performance through back electrode design
- Selecting processing methods for Te/Cu contacts
- Performing pre-contact annealing treatments
- Managing copper incorporation and doping

## Prerequisites
- CdS/CdTe sandwich structure is complete
- Back contact materials (Te, Cu, Au/Mo/Ni) are available
- Etching solutions (Br₂-methanol) prepared if using wet processing
- Annealing furnace with CdCl₂ and oxygen vapor capability

## Standard Back Electrode Structure

### Layer Configuration
1. **Layer 1**: Thin tellurium (Te) layer
2. **Layer 2**: Copper (Cu) layer - CRITICAL PARAMETER
3. **Layer 3**: Thick current-carrying metal contact
   - Options: Gold (Au), Molybdenum (Mo), or Nickel (Ni)

### Surface Preparation
- Etch surface in Br₂-methanol solution before layer deposition
- Ensure clean, oxide-free surface for optimal contact

## Copper Thickness Optimization

### For Dark Diode Characteristics
- **Minimum requirement**: 15nm Cu
- Below 15nm results in poor rectifying behavior
- Use thicker Cu layers when dark performance is critical

### For AM1 Illuminated Characteristics
- **Minimum requirement**: 2nm Cu
- Less sensitive to Cu thickness than dark operation
- Good conversion efficiencies achievable with thinner layers
- **Note**: There is significant asymmetry between dark and illuminated requirements

## Processing Methods Selection

### Dry Processing (Evaporation)
**Use when**: Minimizing series resistance is priority

**Procedure**:
1. Evaporate Cu and Te layers under vacuum
2. Ensure intimate contact with CdTe surface

**Results**:
- Ideality factor: A = 1.6
- Series resistance: 1.4 Ω·cm²

### Wet Processing (Chemical Etching)
**Use when**: Equipment limitations or specific surface chemistry requirements

**Procedure**:
1. Apply chemical etching approach for Te/Cu contact formation
2. Follow proper safety protocols for Br₂-methanol handling

**Results**:
- Ideality factor: A = 1.6
- Series resistance: 2.3 Ω·cm²

**Trade-off**: Higher series resistance than dry processing (2.3 vs 1.4 Ω·cm²)

## Pre-Contact Annealing

**Purpose**: Annealing, recrystallization, and doping of CdTe layer

**Procedure**:
1. Bake at 300-500°C before completing back electrode
2. Typical condition: 15 minutes at 350°C
3. Ambient: Vapor mixture of CdCl₂ and oxygen
4. Allow controlled cooling before proceeding with electrode deposition

## Copper Incorporation Management

### Incorporation Behavior
- Cu incorporates interstitially into CdTe and CdS layers
- Saturation level in CdS: approximately 100 ppm
- Electrical properties vary significantly with Cu reservoir at cell surface

### Surplus Copper Removal
Remove excess Cu using one of these methods:
- Bromine-methanol etch
- Hydrazine treatment
- Diluted acid wash

### Alternative Back Contact Configurations
Consider these alternatives when standard configuration is unsuitable:
- Cu and Hg doped graphite paste
- Graphite doped with Hg and Te

## Critical Constraints
- Back electrode has major influence on cell performance and degradation
- Copper thickness is the most critical parameter to control
- Processing method selection impacts series resistance
- Proper annealing is essential for optimal cell performance
- Copper management affects long-term stability