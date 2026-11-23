# SolarLab Physics & Protocol Contracts

This directory contains the **34 normative physical model, constitutive closure, schema, and execution-protocol specifications** enforced by `src/solarlab/` and verified by `tests/unit/`, `tests/integration/`, and `tests/refactor/`.

For cross-cutting solver architecture and extension rules, see [../DevelopmentGuide.md](../DevelopmentGuide.md) and [../PhysicsCorrectionsScopeV1.md](../PhysicsCorrectionsScopeV1.md).

## 1. Band Structure, Statistics & Heterointerfaces

| Specification | Scope |
|:--------------|:------|
| [DegenerateSemiconductorClosure.md](DegenerateSemiconductorClosure.md) | Fermi–Dirac vs Boltzmann carrier statistics and generalized Einstein relations |
| [BandGapNarrowingClosure.md](BandGapNarrowingClosure.md) | High-injection / heavy-doping band-gap narrowing ($\Delta E_g$) closure |
| [TwoSidedInterfaceTopology.md](TwoSidedInterfaceTopology.md) | Two-sided heterointerface trace topology (`TWO_SIDED_TRACE_V1`, `PHYSICAL_BOUNDARY_TWO_SIDED_V1`) |
| [InterfaceChargeClosurePolicy.md](InterfaceChargeClosurePolicy.md) | Equilibrium-referenced vs absolute interface sheet-charge Gauss closure |
| [PhysicalInterfaceCboScan.md](PhysicalInterfaceCboScan.md) | Conduction-band offset ($\Delta E_c$) scan protocol and numerical contraction gates |
| [WkbTunnellingFamilyContract.md](WkbTunnellingFamilyContract.md) | Nonlocal WKB barrier tunnelling constitutive family |
| [ResolvedIntrabandTunnellingContractV2.md](ResolvedIntrabandTunnellingContractV2.md) | Node-resolved intraband heterojunction tunnelling closure (V2) |

## 2. Bulk & Interface Defect Thermodynamics (DC, Transient & AC)

| Specification | Scope |
|:--------------|:------|
| [ExplicitDefectSchemaV1.md](ExplicitDefectSchemaV1.md) | Validated schema for explicit bulk defect declarations |
| [ExplicitInterfaceDefectSchemaV1.md](ExplicitInterfaceDefectSchemaV1.md) | Validated schema for explicit heterointerface defect declarations |
| [ExplicitDefectEnergyDistributionV2.md](ExplicitDefectEnergyDistributionV2.md) | Discrete, uniform, exponential, and Gaussian energetic defect quadrature |
| [ExplicitDefectSpatialProfileV3.md](ExplicitDefectSpatialProfileV3.md) | Non-uniform spatial trap profiles across layers and interfaces |
| [MonovalentDefectClosure.md](MonovalentDefectClosure.md) | Single-charge-transition SRH occupancy and space-charge coupling |
| [ChargedExplicitDefectQfDc.md](ChargedExplicitDefectQfDc.md) | Charged explicit defects in steady-state DC electrostatics |
| [BulkTrapEnergyDistributionClosure.md](BulkTrapEnergyDistributionClosure.md) | Multi-level bulk trap energy-distribution recombination closure |
| [MultivalentMetastableDefectContract.md](MultivalentMetastableDefectContract.md) | Amphoteric/multivalent correlation-energy ($U$) and metastable defect master equations |
| [DynamicTrapTransient.md](DynamicTrapTransient.md) | Non-quasi-static time-domain trap occupancy integration |
| [InterfaceDefectTransientIncrementContract.md](InterfaceDefectTransientIncrementContract.md) | Conservation-preserving incremental interface trap storage updates |
| [DynamicBulkDefectDeviceAc.md](DynamicBulkDefectDeviceAc.md) | Frequency-domain small-signal linearization of dynamic bulk traps |
| [DynamicInterfaceDefectDeviceAc.md](DynamicInterfaceDefectDeviceAc.md) | Frequency-domain small-signal linearization of dynamic interface traps |
| [DynamicDefectIonCombinedDeviceAc.md](DynamicDefectIonCombinedDeviceAc.md) | Joint small-signal AC coupling of mobile ions and dynamic bulk/interface defects |
| [InterfaceSrhIdentifiability.md](InterfaceSrhIdentifiability.md) | Fisher / profile-likelihood identifiability of $(v_n, v_p)$ interface recombination velocities |

## 3. Optics, External Circuits & Electrothermal Balance

| Specification | Scope |
|:--------------|:------|
| [CigsGradedOptics.md](CigsGradedOptics.md) | Depth-graded alloy composition $E_g(x)$ optical absorption binding (see also [../BeerLambertReferenceProtocolV1.md](../BeerLambertReferenceProtocolV1.md)) |
| [ExternalCircuitDc.md](ExternalCircuitDc.md) | Series/shunt parasitic resistance ($R_s, R_{\mathrm{sh}}$) terminal boundary coupling |
| [ThermalEnergyBalance.md](ThermalEnergyBalance.md) | Lumped convective/radiative electrothermal power ledger and transient coupling |

## 4. 1D Transient & Characterisation Protocols

| Specification | Scope |
|:--------------|:------|
| [ExperimentProtocol.md](ExperimentProtocol.md) | Canonical 1D sweep, preconditioning, and measurement protocol semantics |
| [FrozenIonSnapshotContract.md](FrozenIonSnapshotContract.md) | Zero-diffusivity electronic snapshot $J\text{–}V$ probes during ionic aging |
| [TpvOpenCircuitContract.md](TpvOpenCircuitContract.md) | Floating-port Maxwell total-current open-circuit integration and TPV decay identifiability |
| [IonSweepFigureReference.md](IonSweepFigureReference.md) | Dense mobile-ion waveform reference protocol (`calado2016_ion_sweep`) |

## 5. Two-Dimensional Microstructure & Mobile-Ion Contracts

| Specification | Scope |
|:--------------|:------|
| [TwodTransportContract.md](TwodTransportContract.md) | 2D $(N_y, N_x)$ Poisson + Scharfetter–Gummel assembly and 1D uniform-column reduction |
| [TwodGrainBoundaryAreaClosure.md](TwodGrainBoundaryAreaClosure.md) | Exact control-volume overlap blending for vertical grain-boundary SRH regions |
| [TwodTwoSidedInterfaceSrh.md](TwodTwoSidedInterfaceSrh.md) | Cross-node two-sided heterointerface SRH coupling in 2D |
| [TwodMobileIonTransient.md](TwodMobileIonTransient.md) | Conservative 2D Backward Euler single-mobile-ion time integration |
| [TwodMobileIonCurrent.md](TwodMobileIonCurrent.md) | 2D mobile-ion conduction + displacement Maxwell terminal current decomposition |
| [TwodJvExecutionProtocol.md](TwodJvExecutionProtocol.md) | Hash-bound `JV2DProtocol` and `DynamicJVPairProtocol2D` strict execution contracts |
