# TCO Layer Specifications Reference

## Sheet Resistance Requirements by Device Type

### Small-Area Cells
- **Target Sheet Resistance**: 20–50 ohms/sq
- **Typical Thickness Range**: 100–500 nm
- **Application Context**: Laboratory-scale devices, research applications

### Commercial Modules
- **Target Sheet Resistance**: 5–10 ohms/sq
- **Thickness Requirement**: Larger thicknesses than small-area cells
- **Application Context**: Production-scale solar modules requiring lower series resistance

## Physical Constraints

### Free Carrier Absorption
- As TCO thickness increases, free carrier concentration typically increases
- Higher carrier concentration leads to increased free carrier absorption
- This reduces optical transparency in the active wavelength range

### Resistance-Thickness Relationship
- Sheet resistance is inversely proportional to thickness
- Lower resistance requires thicker layers
- Thicker layers compound transparency losses

## Variables

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| Sheet_Resistance | resistance | ohms/sq | Resistance per square |
| Thickness | length | nm | Physical thickness of the TCO layer |

## Domain Tags
- Optimization
- Electrical Properties
- Optical Properties