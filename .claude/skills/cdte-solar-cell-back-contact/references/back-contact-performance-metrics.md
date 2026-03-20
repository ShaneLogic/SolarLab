# Back Contact Performance Metrics

## Processing Method Comparison

| Method | Ideality Factor (A) | Series Resistance (Ω·cm²) | Advantages | Disadvantages |
|--------|-------------------|-------------------------|------------|---------------|
| Dry (Evaporation) | 1.6 | 1.4 | Lower series resistance, intimate contact | Requires vacuum equipment |
| Wet (Etching) | 1.6 | 2.3 | Simpler equipment setup | Higher series resistance |

## Copper Thickness Performance Data

### Dark Diode Characteristics
- **< 15nm Cu**: Poor rectifying behavior, high leakage currents
- **≥ 15nm Cu**: Reasonable diode characteristics, proper rectification
- **Optimal range**: 15-50nm for balanced performance

### AM1 Illuminated Characteristics
- **< 2nm Cu**: Degraded conversion efficiency
- **≥ 2nm Cu**: Good conversion efficiencies achievable
- **Optimal range**: 2-20nm for illuminated operation

### Performance Asymmetry
Figure 34.12 demonstrates the clear asymmetry between dark and illuminated requirements:
- Dark operation requires ~7.5x more Cu than illuminated operation
- This asymmetry must be considered for intended operating conditions

## Copper Incorporation Details

### Saturation Levels
- **CdS layer**: ~100 ppm Cu saturation
- **CdTe layer**: Varies with processing conditions
- **Surface reservoir**: Critical for long-term electrical properties

### Removal Methods Effectiveness

| Method | Effectiveness | Notes |
|--------|--------------|-------|
| Bromine-methanol | High | Standard etchant, requires careful handling |
| Hydrazine | Medium-High | Toxic, requires safety precautions |
| Diluted acids | Medium | May affect other layers if not controlled |

## Annealing Parameter Effects

### Temperature Effects
- **300°C**: Minimal recrystallization
- **350°C**: Optimal for grain growth and doping (15 min typical)
- **400-500°C**: Enhanced recrystallization, risk of Te loss

### Ambient Composition
- **CdCl₂ vapor**: Promotes grain growth and passivation
- **Oxygen**: Improves p-type doping in CdTe
- **Ratio**: Typically 1:1 to 2:1 CdCl₂:O₂ partial pressures

## Alternative Configurations

### Cu and Hg Doped Graphite Paste
- **Composition**: Graphite paste with Cu and Hg additives
- **Application**: Screen printing or doctor blade
- **Curing**: 200-300°C in inert atmosphere
- **Advantages**: Simpler processing, good for large areas

### Graphite Doped with Hg and Te
- **Composition**: Graphite with Hg and Te dopants
- **Application**: Similar to above
- **Performance**: Comparable to standard contacts for some applications

## Reference
Hegedus, S. and McCandless, B.E. (2005). CdTe solar cells. In: Luque, A. and Hegedus, S. (eds) Handbook of Photovoltaic Science and Engineering. Wiley, pp. 617-662.