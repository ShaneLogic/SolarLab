---
name: solar-cell-device-completion
description: Complete solar cell fabrication by depositing metal grid contacts, defining cell area boundaries, and applying anti-reflection coating. Use this skill when TCO layer deposition is finished and the solar cell device needs final finishing steps before operation or testing.
---

# Solar Cell Device Completion

Complete the fabrication of thin-film solar cells by adding metal contacts, defining active cell area, and applying anti-reflection coating for maximum efficiency.

## Prerequisites

- Deposited TCO (Transparent Conductive Oxide) layer
- Clean processing environment for metal deposition

## Procedure

### 1. Metal Grid Contact Deposition

Deposit the front contact grid to collect current while minimizing light blockage.

**Adhesion Layer:**
- Evaporate Cr or Ni (tens of nanometers)
- Purpose: Prevent formation of high-resistance aluminum oxide at the interface

**Main Conductor:**
- Deposit Al to approximately 1 micrometer thickness
- Structure as a grid pattern to maximize light penetration

**Patterning Methods:**
- Aperture mask: Simpler, suitable for research-scale cells
- Photolithographic definition: Preferred for precise, high-efficiency cells

### 2. Cell Area Definition

Define the active cell area using one of the following methods:

| Method | Procedure | Best For |
|--------|-----------|----------|
| Mechanical scribing | Remove layers on top of Mo outside active area | Rapid prototyping |
| Laser patterning | Remove layers on top of Mo outside active area | Production scale |
| Photolithography + etching | Remove layers on top of Cu(InGa)Se₂ | High-precision cells |
| Masked TCO deposition | Deposit TCO through aperture mask | Simplified processing |

### 3. Anti-Reflection Coating (Optional)

Apply AR coating for highest-efficiency cells to minimize optical losses.

**Parameters:**
- Material: MgF₂ (evaporated)
- Thickness: ~100 nm

**Constraint:** Skip AR coating for modules that will be covered with glass or encapsulated, as the encapsulant provides sufficient optical matching.

## Output

Completed solar cell device ready for electrical characterization and operation.

## Key Variables

- Grid_Material: Cr/Ni/Al stack configuration
- AR_Material: MgF₂ for anti-reflection
- AR_Thickness: ~100 nm optimal thickness