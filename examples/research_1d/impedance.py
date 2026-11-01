"""Run with python -m examples.research_1d.impedance (BLAS threads pinned to 1)."""

from __future__ import annotations

from examples.research_1d.steady import load_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.experiments.impedance import solve_impedance
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state


def main() -> None:
    compiled = compile_device_1d(load_device(), intervals_per_layer=12)
    controls = SteadyOptions()
    point = solve_steady_state(compiled, voltage_V=1.0, options=controls)
    result = solve_impedance(compiled, point, [0.0, 1.0, 1000.0], steady_options=controls)
    print("Ion-free fixed-reservoir small signal at 1 V DC; phasors use exp(+i omega t)")
    print("1 V normalization is a derivative per V, not a finite-amplitude excitation")
    print("frequency_Hz  admittance_density_S_m2  impedance_area_ohm_m2  maxwell_spread_A_m2_V")
    for frequency, admittance, impedance, spread in zip(
        result.frequency_Hz, result.admittance_density_S_m2,
        result.impedance_area_ohm_m2, result.maxwell_spread_A_m2_V,
    ):
        print(f"{frequency:g}  {admittance:.9g}  {impedance:.9g}  {spread:.6g}")
    print(f"working_state_identity={result.working_state_identity}")
    print(f"source_identity={result.source_identity}")


if __name__ == "__main__":
    main()
