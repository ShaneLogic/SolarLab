"""R1 fixed-position reconstruction and response comparison, without solves.

See docs/OneDimensionalMechanismR1SpatialV1.md. Carrier interpolation uses
constant-face-flux SG separately inside each homogeneous layer. Ionic storage
is reconstructed as conservative physical-cell averages, without mixing
materials or active-component boundaries. These are descriptive adapters, not
artifact acceptance or a numerical-convergence certificate.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from perovskite_sim.constants import K_B, Q
from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import (
    R1Response, compare_responses, validate_single_axis_comparison,
)
from perovskite_sim.validation.foundation_reference_refinement import _sample_sg_density


def _array(value, name, *, shape=None, positive=False, nonnegative=False):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(name + " must be real numeric data")
    result = np.array(raw, dtype=float, copy=True)
    if (shape is not None and result.shape != shape) or not np.all(np.isfinite(result)):
        raise ValueError(name + " has invalid shape or nonfinite data")
    if (positive and np.any(result <= 0)) or (nonnegative and np.any(result < 0)):
        raise ValueError(name + " violates its sign constraint")
    return result


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def sample_sg_density(grid_m, density_m3, potential_V, positions_m, thermal_voltage_V, *, carrier):
    """Sample one homogeneous layer, including exact side-specific endpoints.

    Within a homogeneous layer chi/Eg/DOS offsets are constant and cancel in
    SG voltage differences. Density is linear only in the zero-field limit.
    """
    if carrier not in ("electron", "hole"):
        raise ValueError("carrier must be electron or hole")
    grid = _array(grid_m, "grid_m")
    positions = _array(positions_m, "positions_m")
    if grid.ndim != 1 or positions.ndim != 1:
        raise ValueError("SG grid and positions must be vectors")
    density = _array(density_m3, "density_m3", shape=grid.shape, positive=True)
    potential = _array(potential_V, "potential_V", shape=grid.shape)
    thermal = _array(thermal_voltage_V, "thermal_voltage_V", shape=(), positive=True)
    return _sample_sg_density(grid, density, potential, positions, float(thermal),
                              drift_sign=1. if carrier == "electron" else -1.)


@dataclass(frozen=True)
class ConservativeIonProfile:
    """Piecewise linear cell-average representation with exact cell integrals.

    break_indices are interior face indices: slopes never use a neighbor
    across one. Boundary slopes are one-sided; interior slopes use a minmod
    limiter. A second bound prevents negative reconstructed endpoint values.
    The density itself is never clipped or renormalized.
    """

    faces_m: np.ndarray
    density_m3: np.ndarray
    break_indices: tuple[int, ...] = ()

    def __post_init__(self):
        faces = _array(self.faces_m, "faces_m")
        if faces.ndim != 1 or faces.size < 2 or np.any(np.diff(faces) <= 0):
            raise ValueError("ion profile requires strictly increasing physical faces")
        density = _array(self.density_m3, "density_m3", shape=(faces.size-1,), nonnegative=True)
        breaks = tuple(self.break_indices)
        if (any(type(index) is not int or not 0 < index < density.size for index in breaks)
                or tuple(sorted(set(breaks))) != breaks):
            raise ValueError("break_indices must be ordered unique interior face indices")
        width = np.diff(faces)
        center = faces[:-1]+width/2
        slopes = np.zeros_like(density)
        for start, stop in zip((0, *breaks), (*breaks, density.size)):
            if stop-start < 2:
                continue
            secants = np.diff(density[start:stop])/np.diff(center[start:stop])
            slopes[start], slopes[stop-1] = secants[0], secants[-1]
            for index in range(start+1, stop-1):
                left, right = secants[index-start-1:index-start+1]
                if left != 0 and right != 0 and np.signbit(left) == np.signbit(right):
                    central = (density[index+1]-density[index-1])/(center[index+1]-center[index-1])
                    slopes[index] = np.sign(left)*min(abs(2*left), abs(central), abs(2*right))
        half_changes = np.sign(slopes)*np.minimum(np.abs(slopes)*(width/2), density)
        slopes = half_changes/(width/2)
        if not np.all(np.isfinite(slopes)):
            raise ValueError("ion reconstruction slope arithmetic is nonfinite")
        object.__setattr__(self, "faces_m", _readonly(faces))
        object.__setattr__(self, "density_m3", _readonly(density))
        object.__setattr__(self, "break_indices", breaks)
        object.__setattr__(self, "centers_m", _readonly(center))
        object.__setattr__(self, "slopes_m4", _readonly(slopes))
        object.__setattr__(self, "half_changes_m3", _readonly(half_changes))

    def sample(self, positions_m, *, side="right"):
        """Select an explicitly named limit at an internal discontinuity."""
        positions = _array(positions_m, "positions_m")
        if positions.ndim != 1 or side not in ("left", "right"):
            raise ValueError("ion positions must be a vector with side left or right")
        if np.any(positions < self.faces_m[0]) or np.any(positions > self.faces_m[-1]):
            raise ValueError("ion sample lies outside the physical volume")
        cells = np.searchsorted(self.faces_m, positions, side=side)-1
        cells = np.clip(cells, 0, self.density_m3.size-1)
        fraction = (positions-self.faces_m[cells])/np.diff(self.faces_m)[cells]
        # Convex endpoint form avoids subtracting an inexact physical midpoint
        # when a limited endpoint is exactly zero. No density clipping occurs.
        left = self.density_m3[cells]-self.half_changes_m3[cells]
        right = self.density_m3[cells]+self.half_changes_m3[cells]
        values = (1-fraction)*left+fraction*right
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("ion reconstruction produced invalid density")
        return values

    def moments(self, left_m=None, right_m=None):
        """Return exact integrals of P and x*P over the selected physical span."""
        left = self.faces_m[0] if left_m is None else float(_array(left_m, "left_m", shape=()))
        right = self.faces_m[-1] if right_m is None else float(_array(right_m, "right_m", shape=()))
        if left < self.faces_m[0] or right > self.faces_m[-1] or left >= right:
            raise ValueError("moment interval must lie inside the physical volume")
        lo, hi = np.maximum(self.faces_m[:-1], left), np.minimum(self.faces_m[1:], right)
        active = hi > lo
        width = hi[active]-lo[active]
        midpoint = lo[active]+width/2
        center, density, slope = self.centers_m[active], self.density_m3[active], self.slopes_m4[active]
        local_mass = width*(density+slope*(midpoint-center))
        mass = float(np.sum(local_mass))
        first = float(np.sum(midpoint*local_mass+slope*width**3/12))
        if not np.isfinite(mass) or not np.isfinite(first):
            raise ValueError("ion moment arithmetic is nonfinite")
        return mass, first


def _record(prepared):
    record = prepared.to_dict() if hasattr(prepared, "to_dict") else prepared
    if not isinstance(record, dict) or record.get("schema") != "R1CommonStateV1":
        raise ValueError("spatial adapter requires an R1 common-state record")
    return record


def _geometry(prepared):
    record = _record(prepared)
    grid = record["grid"]
    x = _array(grid["coordinates_m"], "grid coordinates")
    if x.ndim != 1 or len(x) < 4 or np.any(np.diff(x) <= 0):
        raise ValueError("invalid R1 spatial grid")
    faces = _array(grid["faces_m"], "physical faces", shape=(len(x)+1,))
    widths = _array(grid["widths_m"], "physical widths", shape=x.shape, positive=True)
    interfaces = _array(grid["interface_positions_m"], "interface positions", shape=(1,))
    stack = record["physical_stack"]
    layers = stack["layers"]
    if len(layers) != 2 or stack["band_grading"]:
        raise ValueError("R1 spatial adapter supports two homogeneous electrical layers")
    if any(layer["role"] == "substrate" or layer["params"]["carrier_statistics"] != "maxwell_boltzmann"
           for layer in layers):
        raise ValueError("R1 SG adapter requires electrical Maxwell-Boltzmann layers")
    lengths = _array([layer["thickness"] for layer in layers], "layer thickness", positive=True)
    bounds = np.r_[0., np.cumsum(lengths)]
    if (not np.array_equal(interfaces, bounds[1:-1]) or x[0] != bounds[0] or x[-1] != bounds[-1]
            or faces[0] != x[0] or faces[-1] != x[-1] or not np.array_equal(np.diff(faces), widths)):
        raise ValueError("R1 physical layer, face and width identities disagree")
    from perovskite_sim.physics.physical_control_volume import physical_cell_faces
    if not np.array_equal(faces, physical_cell_faces(x, interfaces)):
        raise ValueError("R1 faces do not match physical control volumes")
    thermal = float(_array(stack["T"], "temperature", shape=(), positive=True))*K_B/Q
    background = _array(grid["background_positive_m3"], "background ions", shape=x.shape, positive=True)
    components = tuple(tuple(nodes) for nodes in grid["positive_components"])
    active = []
    cuts = {int(np.searchsorted(x, boundary)) for boundary in interfaces}
    for nodes in components:
        if (not nodes or any(type(i) is not int or not 0 <= i < len(x) for i in nodes)
                or nodes != tuple(range(nodes[0], nodes[-1]+1))):
            raise ValueError("ionic components must be contiguous physical cells")
        active.extend(nodes)
        cuts.update(index for index in (nodes[0], nodes[-1]+1) if 0 < index < len(x))
    if len(set(active)) != len(active) or active != grid["positive_nodes"]:
        raise ValueError("ionic component identity mismatch")
    positions = np.concatenate([left+(right-left)*np.arange(1, 18)/18
                                for left, right in zip(bounds[:-1], bounds[1:])])
    return record, x, faces, bounds, positions, thermal, background, components, tuple(sorted(cuts))


def sample_r1_state(prepared, state=None):
    """Reconstruct a saved snapshot at §10.2 positions and both interface sides.

    No material builder or solver is called. The adapter checks geometry and
    representation consistency; external artifact acceptance remains separate.
    """
    record, x, faces, bounds, positions, thermal, background, components, cuts = _geometry(prepared)
    state = record["state"] if state is None else state
    n = _array(state["n_m3"], "n_m3", shape=x.shape, positive=True)
    p = _array(state["p_m3"], "p_m3", shape=x.shape, positive=True)
    phi = _array(state["phi_V"], "phi_V", shape=x.shape)
    ions = _array(state["positive_m3"], "positive_m3", shape=x.shape, nonnegative=True)
    trace = _array(state["trace_state_m3"], "trace_state_m3", shape=(1, 4), positive=True)
    trace_phi = _array(state["trace_potential_V"], "trace_potential_V", shape=(1, 2))
    occupancy = _array(state["occupancy"], "occupancy", shape=(1,), nonnegative=True)
    if np.any(occupancy > 1):
        raise ValueError("occupancy must lie in [0,1]")
    sampled_n, sampled_p, sampled_phi = [], [], []
    for layer, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
        indices = np.flatnonzero((x >= left) & (x <= right))
        local_x, local_n, local_p, local_phi = [value[indices].copy() for value in (x, n, p, phi)]
        if layer == 0:
            local_x, local_n, local_p, local_phi = [np.r_[value, endpoint] for value, endpoint in
                zip((local_x, local_n, local_p, local_phi), (right, trace[0, 0], trace[0, 1], trace_phi[0, 0]))]
        else:
            local_x, local_n, local_p, local_phi = [np.r_[endpoint, value] for value, endpoint in
                zip((local_x, local_n, local_p, local_phi), (left, trace[0, 2], trace[0, 3], trace_phi[0, 1]))]
        queries = positions[17*layer:17*(layer+1)]
        sampled_n.extend(sample_sg_density(local_x, local_n, local_phi, queries, thermal, carrier="electron"))
        sampled_p.extend(sample_sg_density(local_x, local_p, local_phi, queries, thermal, carrier="hole"))
        sampled_phi.extend(np.interp(queries, local_x, local_phi))
    profile = ConservativeIonProfile(faces, ions, cuts)
    reference = ConservativeIonProfile(faces, background, cuts)
    inventory, centroid = [], []
    for nodes in components:
        mass, first = profile.moments(faces[nodes[0]], faces[nodes[-1]+1])
        if mass <= 0:
            raise ValueError("active ionic component has no inventory")
        inventory.append(mass)
        centroid.append(first/mass)
    values = {
        "position_m": positions, "n_m3": sampled_n, "p_m3": sampled_p,
        "phi_V": sampled_phi, "positive_m3": profile.sample(positions),
        "background_positive_m3": reference.sample(positions),
        "interface_position_m": bounds[1:-1], "interface_carriers_m3": trace,
        "interface_phi_V": trace_phi,
        "interface_positive_m3": np.column_stack((profile.sample(bounds[1:-1], side="left"),
                                                   profile.sample(bounds[1:-1], side="right"))),
        "interface_background_positive_m3": np.column_stack((reference.sample(bounds[1:-1], side="left"),
                                                              reference.sample(bounds[1:-1], side="right"))),
        "occupancy": occupancy, "ion_inventory_m2": inventory, "ion_centroid_m": centroid,
    }
    return MappingProxyType({key: _readonly(value) for key, value in values.items()})


@dataclass(frozen=True)
class R1SpatialResponses:
    """Sampled changes relative to the same grid's own saved 0- state."""

    responses: Mapping[str, R1Response]
    interface_responses: Mapping[str, R1Response]
    dc_potential: R1Response
    metadata: Mapping
    inventory_m2: np.ndarray
    dc_interface_potential: R1Response | None = None
    scope: str = "fixed_position_reconstruction_and_comparison_only"

    def __post_init__(self):
        object.__setattr__(self, "responses", MappingProxyType(dict(self.responses)))
        object.__setattr__(self, "interface_responses", MappingProxyType(dict(self.interface_responses)))
        object.__setattr__(self, "metadata", MappingProxyType(copy.deepcopy(dict(self.metadata))))
        object.__setattr__(self, "inventory_m2", _readonly(self.inventory_m2))


def spatial_responses_from_step(prepared, step):
    """Adapt finest saved accepted states at exactly the declared output times.

    Carrier changes are ln(n(t))-ln(n(0-)), not log of a density difference.
    Ion changes are (P(t)-P(0-))/P0. No amplitude division is applied to these
    §10.2 state responses. Zero time means 0+, distinct from the DC baseline.
    """
    record = _record(prepared)
    if (step.get("schema") != "R1ControlledStepV1" or step.get("prepared_sha256") != record["sha256"]
            or step.get("reference_sha256") != record["reference_sha256"]
            or step.get("intervals") != record["intervals"] or step.get("source") != record.get("source")):
        raise ValueError("step and prepared state identities disagree")
    times = _array(step["times_s"], "times_s")
    if times.ndim != 1 or not times.size or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("step times must identify 0+ then increasing physical times")
    levels = step["policy"]["refinement_substeps"]
    if not levels or any(type(level) is not int or level < 1 for level in levels):
        raise ValueError("step lacks declared positive integer refinement levels")
    rows = [row for row in step["accepted_steps"] if row["substeps"] == max(levels)]
    samples = []
    for time in times:
        matches = [row for row in rows if row["time_s"] == time]
        if len(matches) != 1:
            raise ValueError("each output time requires exactly one finest accepted snapshot")
        if matches[0]["phase"] != ("0+" if time == 0 else "accepted_regular_step"):
            raise ValueError("accepted row phase differs from its physical time")
        samples.append(sample_r1_state(record, matches[0]["state"]))
    base = sample_r1_state(record)
    positions = {"time_s": times, "position_m": base["position_m"]}
    interfaces = {"time_s": times, "interface_position_m": base["interface_position_m"]}
    arrays = {key: np.stack([sample[key] for sample in samples]) for key in base}
    responses = {
        "response_potential": R1Response(arrays["phi_V"]-base["phi_V"], positions),
        "carrier_log_density": R1Response(np.stack((np.log(arrays["n_m3"])-np.log(base["n_m3"]),
            np.log(arrays["p_m3"])-np.log(base["p_m3"])), axis=-1), positions, ("electron", "hole")),
        "ion_density_over_p0": R1Response((arrays["positive_m3"]-base["positive_m3"])/base["background_positive_m3"], positions),
        "trap_occupancy_change": R1Response(arrays["occupancy"]-base["occupancy"], interfaces),
        "ion_centroid_change": R1Response(arrays["ion_centroid_m"]-base["ion_centroid_m"],
            {"time_s": times, "active_component": np.arange(base["ion_centroid_m"].size)}),
    }
    side_responses = {
        "response_potential": R1Response(arrays["interface_phi_V"]-base["interface_phi_V"], interfaces, ("left", "right")),
        "carrier_log_density": R1Response(np.log(arrays["interface_carriers_m3"])-np.log(base["interface_carriers_m3"]),
            interfaces, ("n_left", "p_left", "n_right", "p_right")),
        "ion_density_over_p0": R1Response((arrays["interface_positive_m3"]-base["interface_positive_m3"])
            /base["interface_background_positive_m3"], interfaces, ("left", "right")),
    }
    metadata = {"stack_sha256": record["stack_sha256"], "reference_sha256": record["reference_sha256"],
                "source_sha256": (step.get("source") or {}).get("sha256"),
                "amplitude_V": step["amplitude_V"], "control": step["control_label"],
                "prepared_sha256": record["sha256"], "times_s": times.tolist(),
                "intervals": record["intervals"], "policy": step["policy"],
                "recorded_certificate_passed": step.get("certificate", {}).get("certified") is True,
                "carrier_resolution": "no independent carrier uncertainty supplied; signal resolution undetermined",
                "ionic_centroid_definition": "integral(x*P_reconstructed)/integral(P_reconstructed) over each active physical component"}
    return R1SpatialResponses(responses, side_responses,
        R1Response(base["phi_V"], {"position_m": base["position_m"]}), metadata, arrays["ion_inventory_m2"],
        dc_interface_potential=R1Response(base["interface_phi_V"],
            {"interface_position_m": base["interface_position_m"]}, ("left", "right")))


def compare_spatial_responses(left, right):
    """Apply existing §10.2 rules to equal physical states/times/positions.

    The result does not assert that only one numerical axis changed: policy
    identities are returned for the campaign to classify the comparison.
    """
    if not isinstance(left, R1SpatialResponses) or not isinstance(right, R1SpatialResponses):
        raise TypeError("comparison requires R1SpatialResponses")
    for key in ("stack_sha256", "reference_sha256", "source_sha256", "amplitude_V", "control"):
        if left.metadata[key] != right.metadata[key]:
            raise ValueError("physical experiment identity differs: " + key)
    bulk_names = {"response_potential", "carrier_log_density", "ion_density_over_p0",
                  "trap_occupancy_change", "ion_centroid_change"}
    side_names = {"response_potential", "carrier_log_density", "ion_density_over_p0"}
    if any(set(value.responses) != bulk_names or set(value.interface_responses) != side_names
           for value in (left, right)):
        raise ValueError("spatial comparison requires the complete response quantity inventory")
    if left.dc_interface_potential is None or right.dc_interface_potential is None:
        raise ValueError("spatial comparison requires both DC interface-side potentials")
    bulk = {key: compare_responses(key, value, right.responses[key]) for key, value in left.responses.items()}
    sides = {key: compare_responses(key, value, right.interface_responses[key])
             for key, value in left.interface_responses.items()}
    dc = compare_responses("dc_potential", left.dc_potential, right.dc_potential)
    dc_sides = compare_responses("dc_potential", left.dc_interface_potential, right.dc_interface_potential)
    return {"scope": "fixed_position_response_budget_comparison_only", "dc_potential": dc,
            "dc_interface_potential": dc_sides,
            "bulk": bulk, "interface_sides": sides,
            "within_compared_budgets": all(report["passed"] for report in (dc, dc_sides, *bulk.values(), *sides.values())),
            "convergence_passed": False,
            "left_metadata": dict(left.metadata), "right_metadata": dict(right.metadata),
            "limitations": ["no independent signal-resolution certificate", "no full three-axis or long-window acceptance"]}


def compare_convergence_responses(left, right, *, axis, electrical, input_verifications=None):
    """Derive a scoped single-axis verdict from all eight §10.2 quantities.

    The caller must obtain the two verification reports by verifying the
    exact source-bound input artifacts. They are not stored trajectory flags.
    Full-window/campaign acceptance remains a separate aggregate decision.
    """
    identity = validate_single_axis_comparison(left.metadata, right.metadata, axis=axis)
    spatial = compare_spatial_responses(left, right)
    electrical_names = {"regular_current": "regular_current_response", "integrated_charge": "integrated_charge_response"}
    for key, quantity in electrical_names.items():
        if key not in electrical or electrical[key].get("quantity") != quantity:
            raise ValueError("convergence requires both electrical response quantities")
    bulk, sides = spatial["bulk"], spatial["interface_sides"]
    reports = {
        "dc_potential": (spatial["dc_potential"], spatial["dc_interface_potential"]),
        "response_potential": (bulk["response_potential"], sides["response_potential"]),
        "carrier_log_density": (bulk["carrier_log_density"], sides["carrier_log_density"]),
        "ion_density_over_p0": (bulk["ion_density_over_p0"], sides["ion_density_over_p0"]),
        "trap_occupancy_change": (bulk["trap_occupancy_change"],),
        "ion_centroid_change": (bulk["ion_centroid_change"],),
        "regular_current_response": (electrical["regular_current"],),
        "integrated_charge_response": (electrical["integrated_charge"],),
    }
    quantity_passed = {key: all(report.get("passed") is True and report.get("failure_count") == 0
                                 and report.get("scalar_comparison_count", 0) > 0
                                 and isinstance(report.get("maximum_budget_ratio"), (int, float))
                                 and 0 <= report["maximum_budget_ratio"] <= 1 for report in items)
                       for key, items in reports.items()}
    verified = input_verifications is not None and len(input_verifications) == 2
    if verified:
        verified = all(isinstance(report, Mapping) and report.get("certified") is True
                       and report.get("content_matches_recomputed") is True
                       and metadata.get("recorded_certificate_passed") is True
                       for report, metadata in zip(input_verifications, (left.metadata, right.metadata)))
    passed = bool(verified and all(quantity_passed.values()))
    return {"schema": "R1SingleAxisConvergenceComparisonV1", "axis_check": identity,
            "spatial": spatial, "electrical": electrical, "quantity_passed": quantity_passed,
            "required_quantity_count": 8, "response_budgets_passed": all(quantity_passed.values()),
            "input_physics_verified": bool(verified), "within_compared_budgets": passed,
            "convergence_passed": passed, "full_window_convergence_certified": False,
            "scope": "verified_single_axis_comparison_only"}


__all__ = ["ConservativeIonProfile", "R1SpatialResponses", "sample_sg_density", "sample_r1_state",
           "compare_convergence_responses",
           "spatial_responses_from_step", "compare_spatial_responses"]
