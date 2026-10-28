"""Run the real new-core 1D dark and illuminated steady example.

From the repository root, with the package installed:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 python examples/research_1d/steady.py

Add --additional-recombination to inject a local analytic rate law without
changing the compiler, equation assembly, or nonlinear solver.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from solarlab.config.resolve_device import resolve_device
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.device.compile_1d import compile_device_1d
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument
from solarlab.physics.recombination.bulk import RecombinationRate
from solarlab.solvers.nonlinear.steady_1d import solve_equilibrium, solve_steady_state


def load_device(path: Path | None = None) -> PreparedDevice:
    """Prepare the explicit example catalog/configuration without legacy sources."""
    path = Path(__file__).with_name("device.json") if path is None else Path(path)
    source = SourceDocument(str(path), path.read_bytes())
    document = json.loads(source.content)
    catalog = {
        **document["default_catalog"],
        "evidence": [[source.id, source.sha256]],
    }
    return resolve_device(
        DeviceInput.model_validate(document["device"]),
        DefaultCatalog.from_mapping(catalog),
        ResourceLibrary(),
        sources=(source,),
    )


@dataclass(frozen=True)
class SaturatingRecombination:
    """Extra local net pair rate B (np - ni²)/(1 + (n+p)/n_sat).

    Both density tangents are analytic and the rate vanishes at mass action.
    This is an extension example, not a scientifically qualified model.
    """

    coefficient_m3_s: np.ndarray
    intrinsic_product_m6: np.ndarray
    saturation_density_m3: float
    model_id: str = field(init=False)

    def __post_init__(self) -> None:
        arrays = []
        for name in ("coefficient_m3_s", "intrinsic_product_m6"):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            if array.ndim != 1 or not np.all(np.isfinite(array)) or np.any(array < 0.0):
                raise ValueError("extension coefficients must be finite nonnegative arrays")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
            arrays.append(array)
        if arrays[0].shape != arrays[1].shape or not np.isfinite(self.saturation_density_m3) or self.saturation_density_m3 <= 0.0:
            raise ValueError("extension arrays must match and saturation density must be positive")
        digest = hashlib.sha256(repr(self.saturation_density_m3).encode())
        for array in arrays:
            digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
        object.__setattr__(self, "model_id", "example.saturating_recombination.v1:" + digest.hexdigest())

    def evaluate(self, n: np.ndarray, p: np.ndarray) -> RecombinationRate:
        excess = n * p - self.intrinsic_product_m6
        denominator = 1.0 + (n + p) / self.saturation_density_m3
        rate = self.coefficient_m3_s * excess / denominator
        d_n = self.coefficient_m3_s * (p - excess / (denominator * self.saturation_density_m3)) / denominator
        d_p = self.coefficient_m3_s * (n - excess / (denominator * self.saturation_density_m3)) / denominator
        return RecombinationRate(rate, d_n, d_p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--additional-recombination", action="store_true")
    arguments = parser.parse_args()
    prepared = load_device(arguments.config)
    device = compile_device_1d(prepared)
    if arguments.additional_recombination:
        absorber = np.zeros(device.node_count)
        for layer, mask in zip(device.parameters.layers, device.parameters.layer_masks):
            if layer.role.lower() == "absorber":
                absorber[mask] = 5.0e-15
        extra = SaturatingRecombination(absorber, device.parameters.intrinsic_product_m6, 1.0e21)
        device = compile_device_1d(prepared, additional_recombination=extra)
    equilibrium = solve_equilibrium(device)
    illuminated = solve_steady_state(device)
    summary = {
        "device_id": device.device_id,
        "compiled_identity": device.identity,
        "node_count": device.node_count,
        "geometry": device.geometry.name,
        "voltage_polarity": device.voltage_polarity,
        "built_in_potential_V": device.built_in_potential_V,
        "dark_max_abs_current_A_m2": float(np.max(np.abs(equilibrium.current_density_A_m2))),
        "short_circuit_current_x_A_m2": float(illuminated.current_density_A_m2[0]),
        "short_circuit_output_A_m2": float(-device.voltage_polarity * illuminated.current_density_A_m2[0]),
        "absorbed_photon_current_limit_A_m2": device.generation_limit_A_m2,
        "current_spread_A_m2": float(np.ptp(illuminated.current_density_A_m2)),
        "newton_iterations": illuminated.newton.iteration_count,
        "preparation_iterations": [result.iteration_count for result in illuminated.preparation],
        "density_basin_initializations": illuminated.density_basin_initializations,
        "gates": [
            {"name": gate.name, "passed": gate.passed, "value": gate.value, "limit": gate.limit}
            for gate in illuminated.physical_gates
        ],
        "qualification": "first runnable ion-free classical DC slice; not full scientific qualification",
    }
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
