# Lumped Thermal Energy Balance

## Scope

The thermal layer uses one area-normalized control volume containing the
photovoltaic device and any lumped series/shunt elements already included in
the terminal power. It does not modify contact calibration or infer absorbed
power from incident irradiance. The caller must declare absorbed optical power
explicitly.

For a lumped temperature `T`, the registered first-law balance is

```text
C_A dT/dt = P_abs + P_internal - P_electrical
            - h (T - T_ambient)
            - emissivity * sigma * (T^4 - T_ambient^4)
```

All powers are in `W/m2`, the areal heat capacity `C_A` is in `J/(m2 K)`, and
`sigma = 5.670374419e-8 W/(m2 K4)`. Positive terminal electrical power leaves
the control volume and is subtracted exactly once. The current power-producing
contract therefore requires `0 <= P_electrical <= P_abs`; an independently
declared internal heat source cannot justify additional electrical export.

## Python API

```python
from perovskite_sim.experiments import (
    LumpedThermalProtocol,
    ThermalIntegrationProtocol,
    run_lumped_thermal_transient,
    solve_lumped_thermal_steady_state,
)

thermal = LumpedThermalProtocol(
    absorbed_optical_power_W_m2=800.0,
    ambient_temperature_K=300.0,
    areal_heat_capacity_J_m2_K=2000.0,
    heat_transfer_coefficient_W_m2_K=20.0,
    emissivity=0.85,
    maximum_temperature_K=500.0,
)
steady = solve_lumped_thermal_steady_state(
    thermal,
    terminal_electrical_export_W_m2=200.0,
)

integration = ThermalIntegrationProtocol(
    duration_s=200.0,
    initial_temperature_K=300.0,
    sample_count=101,
)
transient = run_lumped_thermal_transient(
    thermal,
    integration,
    terminal_electrical_export_W_m2=200.0,
)
```

Both protocols use exact-key schemas, canonical JSON, and SHA-256 identities.
Results retain those identities and a mapping hash. Transient output includes
the cumulative absorbed, internal, electrical, linear-rejection, and
radiative-rejection energies, plus the directly recomputed first-law residual.
All evidence arrays are immutable.

## Fail-Closed Conditions

The steady solver rejects an impossible electrical export, a positive heat
source with no declared rejection mechanism, a root outside the maximum
temperature envelope, or a residual above the frozen tolerance. The transient
solver additionally rejects non-finite or non-positive temperatures, envelope
violations, integration failures, and an energy ledger outside its registered
tolerance. Result constructors recompute the ledger from the frozen protocols,
so changing an array or a `certified` flag cannot create valid evidence.

## Evidence Boundary

This checkpoint is internally tested against exact linear-heating solutions,
radiative steady states, zero-heat limits, and cumulative first-law closure. It
remains usable as an independent constant-power balance.

## Steady Electrothermal Operating Point

`solve_electrothermal_operating_point` adds an opt-in steady feedback loop. At
every trial temperature it creates a fresh `DeviceStack`, runs the same strict
forward/reverse transient J-V protocol, applies the frozen series/shunt circuit,
selects one explicitly named branch's sampled terminal maximum-power point,
and evaluates the thermal residual. A bounded Brent root closes

```text
P_abs + P_internal - P_terminal,mpp(T) - P_rejection(T) = 0.
```

`ElectrothermalJVProtocol` freezes the grid, scan, voltage range, solver
tolerances, incident power, and fresh-state rule.
`ElectrothermalOperatingPointProtocol` freezes the branch, sampled-MPP rule,
root method, temperature tolerance, and iteration bound. The result includes
every evaluated temperature and its complete `ExperimentProtocol`, intrinsic
source hash, external mapping hash, terminal MPP, and first-law residual. The
backend exposes the same strict contract at
`POST /api/jv/electrothermal-operating-point`.

This is a protocol-conditioned steady operating point, not a joint
electrical-thermal transient DAE. Each temperature starts a new J-V history;
there is no thermal-memory or ion-state handoff between temperatures. The
model does not yet include a spatial heat equation, thermal contact
resistance, temperature-dependent optical constants, spectral thermalization,
or measured thermal parameters. A source-clean grid/tolerance certificate is
still required before the coupled layer is labeled internally certified.
