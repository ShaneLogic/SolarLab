# Complete Transport Equation Specifications

## Perovskite Layer Variables

**Current Densities:**
- J_n: Electron current density
- J_p: Hole current density
- J_P: Ion vacancy current density

**Fields:**
- φ: Electric potential
- n: Electron density
- p: Hole density
- P: Anion vacancy density

**Constants:**
- N̂_0: Background cation vacancy density (immobile)
- ε: Permittivity
- q: Elementary charge

## Transport Layer Parameters

**ETL:**
- S_E: Statistical integral function (ETL-specific)
- g_E: Density of states factor (ETL)
- T_E: Temperature parameter (ETL)
- E_ct: Cathode workfunction

**HTL:**
- S_H: Statistical integral function (HTL-specific)
- g_H: Density of states factor (HTL)
- T_H: Temperature parameter (HTL)
- E_an: Anode workfunction

## Interface Variables

**R̂_l:** Interface recombination rate (dimensionally consistent with flux)

## Spatial Domains

- Perovskite: 0 < x < b
- ETL: -bE < x < 0  
- HTL: b < x < b+bH

## Physical Constraints

- Perovskite layer models all three carrier types (electrons, holes, vacancies)
- Transport layers model majority carriers only
- Background doping in perovskite is assumed uniform and constant
- Interfaces enforce flux continuity accounting for interfacial recombination