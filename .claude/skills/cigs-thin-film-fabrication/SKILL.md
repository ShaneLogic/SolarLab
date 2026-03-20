---
name: cigs-thin-film-fabrication
description: Use this skill when fabricating Cu(InGa)Se2 thin-film solar cells. Covers substrate selection, deposition methods, and TCO layer deposition for complete device fabrication.
---

# CIGS Thin Film Fabrication

## When to Use
- Fabricating Cu(InGa)Se2 thin-film solar cells
- Selecting substrates for CIGS deposition
- Choosing deposition methods for absorber layer
- Depositing transparent conducting oxide front contacts

## General Requirements
- Low cost
- High deposition/processing rate
- High compositional uniformity over large areas
- **Minimum thickness:** 1 μm for light absorption

## Substrate Selection

### Soda-Lime Window Glass
**Most commonly used substrate**

**Advantages:**
- Available in large quantities at low cost
- Contains Na for diffusion into cell
- Used to make highest-efficiency devices

**Thermal Properties:**
- **Deposition temperature:** 350-750°C (at least 350°C required)
- **Thermal expansion coefficient:** 9×10⁻⁶/K
- **Match with CIGS:** CIGS coefficient = 9×10⁻⁶/K
- **Result:** Little stress during cool-down

**Chemical Composition:**
- Contains oxides: Na2O, K2O, CaO
- Provides alkali to Cu(InGa)Se2

### Thermal Expansion Mismatch Effects

#### Lower Coefficient (e.g., borosilicate glass)
- Film under tensile stress during cool-down
- **Results:** Voids and micro-cracks

#### Higher Coefficient (e.g., polyimide)
- Film under compressive stress
- **Results:** Adhesion failures

### Sodium Incorporation Effects
- Influences microstructure with larger grains
- Higher degree of orientation with (112) parallel to glass surface

### Controlled Sodium Supply Methods
1. **Block sodium** from substrate with diffusion barrier (SiOx, Al2O3, SiN)
2. **Direct supply:** Deposit Na-containing precursor (NaF, ~10nm) onto Mo film
3. **Co-deposition:** Deposit Na with Cu(InGa)Se2
4. **Post-deposition:** Na treatment gives same performance increase

### Metal Foil Substrates
- Can withstand higher temperatures
- Electrically conductive
- **Stainless steel:** Most commonly used, highest efficiency flexible cells

## Deposition Methods

### Method 1: Simultaneous Vapor Deposition (Co-evaporation)

**Process:** Simultaneous deposition of Cu, In, Ga, and Se onto substrate
**Temperature range:** 450-600°C

**Evaporation Temperatures:**
- Cu: 1300-1400°C
- In: 1000-1100°C
- Ga: 1150-1250°C
- Se: 250-350°C

**Growth Strategy:**
- Bulk of film grown with Cu-rich overall composition
- Contains Cu_xSe phase in addition to Cu(InGa)Se2

**Advantage:** Flexibility to control film composition and band-gap
**Challenge:** Difficulty controlling desired Cu-evaporation

### Method 2: In-Line Process

**Process:** Substrate moves sequentially over constantly effusing sources

**Control Methods:**
- In situ flux measurement (electron impact, mass spectroscopy, atomic absorption)
- In situ film thickness measurement (quartz crystal, optical spectroscopy, XRF)
- Process monitoring for Cu-rich to Cu-poor transition (laser scattering, emissivity, IR transmission)

### Method 3: Precursor Reaction Processes (Two-Step Process)

**Process:**
1. Deposit precursor film containing Cu, In, and Ga
2. React at high temperature to form Cu(InGa)Se2 (selenization)

**Highest efficiency:** 16.5%

**Precursor Deposition Methods:**
- **Sputtering:** Easily scalable, good uniformity, high rates
- **Electro-deposition:** High material utilization at low cost
- **Particle ink/spray:** High utilization and uniformity

**Reaction Conditions:**
- **Agent:** H2Se at 400-500°C
- **Time:** Up to 60 min
- **Limitations:** Poor adhesion at longer times, excessive MoSe2 formation
- **Alternative:** Diethyl selenium (less toxic)

## TCO Materials and Deposition

### Material Selection
- **SnO2:** Requires undesired high temperatures > 250°C
- **ITO (In2O3:Sn):** Can be used, ZnO often favored for lower cost
- **ZnO:** Preferred material for Cu(InGa)Se2 solar cells

### Deposition Methods

#### ITO
- **Method:** Sputtering from ceramic ITO targets in Ar:O2 mixture
- **Rate:** 0.1 - 10 nm/s

#### ZnO:Al
- **Method:** rf magnetron sputtering from ceramic ZnO:Al2O3 targets (1-2 wt% Al2O3)
- **Alternative:** DC sputtering for higher deposition rates

#### Reactive DC Sputtering
- **Targets:** Al/Zn alloy targets
- **Advantage:** Lower costs
- **Challenge:** Precise process control due to hysteresis effect
- **Rate:** 5 - 10 nm/s

#### Chemical Vapor Deposition (CVD)
- **Reaction:** Water vapor and diethylzinc at atmospheric pressure
- **Doping:** Fluorine or boron