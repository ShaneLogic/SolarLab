# CdS/CdTe Performance Parameter Reference

## Historical Development

- **1976**: Initial development by Bonnett
- **Early development**: Recognized as attractive thin-film cell technology
- **Mid-1990s**: Commercial panel production began
- **Current status**: Multi-gigawatt worldwide deployment from multibillion dollar industry

## Performance by Processing Method

Different processing methods yield varying performance characteristics:

### Efficiency Ranges
- Overall range: 8-16%
- High-efficiency cells: 14-18%
- Record cells: 16.5%+

### Open Circuit Voltage (Voc)
- Range: 500-860mV
- High-efficiency target: >840mV
- Factors affecting Voc:
  - CdTe carrier lifetime
  - Junction quality
  - Back contact barrier height

### Short Circuit Current (jsc)
- Range: 16-26mA/cm²
- High-efficiency target: >25mA/cm²
- Factors affecting jsc:
  - CdS window thickness (thinner = higher current)
  - CdTe absorption quality
  - Front contact transparency

### Fill Factor
- Range: 63-76%
- High-efficiency target: >75%
- Factors affecting FF:
  - Series resistance
  - Shunt resistance
  - Junction ideality factor

## Layer Thickness Guidelines

### CdS Window Layer
- Typical: 60nm
- Trade-offs:
  - Thinner: Higher jsc, risk of pinholes
  - Thicker: Better coverage, lower jsc

### CdTe Absorber Layer
- Typical: 2μm
- Minimum for good absorption: 1.5μm
- Thicker layers: Minimal benefit beyond 2μm due to carrier collection length

## Key Variables

| Variable | Type | Description |
|----------|------|-------------|
| CdS_thickness | numeric | Thickness of CdS window layer, typically 60nm |
| CdTe_thickness | numeric | Thickness of CdTe absorber layer, approximately 2μm |
| cell_efficiency | numeric | Power conversion efficiency, typically 14-18% for high-efficiency cells |