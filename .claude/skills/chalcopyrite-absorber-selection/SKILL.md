---
name: chalcopyrite-absorber-selection
description: Select appropriate wide-band-gap chalcopyrite absorber materials for thin-film solar cells based on band-gap requirements. Use when designing single-junction or tandem solar cells, comparing Cu-based and Ag-based chalcopyrite compounds, or determining material suitability for specific Eg values.
---

# Chalcopyrite Absorber Material Selection

## When to Use

Use this skill when:
- Selecting absorber material for specific band-gap requirements
- Designing single-junction or tandem thin-film solar cells
- Comparing properties of different chalcopyrite compositions
- Determining processing requirements for specific materials

## Material Properties Lookup

### CuGaSe2
- Band-gap: Eg = 1.68 eV
- Application: Suited for tandem structures

### CuInS2
- Band-gap: Eg = 1.53 eV
- Application: Nearly optimum for single-junction solar cells
- Processing: Requires Cu-rich deposition, then etching away excess Cu (CuxS second phase) before CdS deposition

### Cu(InAl)Se2
- Band-gap: Eg = 1.15 eV
- Achieved efficiency: 17%

### CuAlSe2
- Band-gap: Eg = 2.7 eV

### (AgCu)(InGa)Se2
- Feature: Ag replacement of Cu enables band-gap increase while lowering alloy melting temperature
- Performance: Relatively high Voc reported

## Selection Procedure

1. Identify required band-gap (Eg) based on application:
   - Tandem structure: Target Eg ≈ 1.68 eV → CuGaSe2
   - Single-junction optimum: Target Eg ≈ 1.53 eV → CuInS2
   - Intermediate band-gap: Target Eg ≈ 1.15 eV → Cu(InAl)Se2
   - Wide band-gap: Target Eg ≈ 2.7 eV → CuAlSe2

2. Consider processing constraints:
   - CuInS2 requires additional etching step
   - (AgCu)(InGa)Se2 offers lower melting temperature

3. Verify efficiency targets against reference data

## Variables

- **Eg**: Band-gap energy in electron volts (eV) - primary selection criterion

## Constraints

- Specific efficiencies vary by deposition method and reference sources
- Always verify current efficiency records as technology advances