"""Saved pair-state spatial responses, with rounding after physical changes.

This keeps the original homogeneous SG projection and conservative minmod
ion profile. Both words participate in interpolation, moments, logarithms
and baseline subtraction; only the public response values are rounded.
"""
from __future__ import annotations

import numpy as np

from ..physics.compensated import DD, log, sum as dd_sum
from .one_dimensional_mechanism_r1_precision import bernoulli_pair
from . import one_dimensional_mechanism_r1_pair_codec as codec


def stack(values, axis=0):
    return DD(np.stack([value.hi for value in values], axis=axis),
              np.stack([value.lo for value in values], axis=axis))


def concatenate(values):
    return DD(np.concatenate([value.hi for value in values]),
              np.concatenate([value.lo for value in values]))


def field(snapshot, name):
    return DD(snapshot["precision_" + name + "_hi"], snapshot["precision_" + name + "_lo"])


def _geometry(prepared):
    from .one_dimensional_mechanism_r1_spatial import _geometry as legacy_geometry
    record = prepared.to_dict() if hasattr(prepared, "to_dict") else prepared
    codec.decode_prepared(record, representation=codec.REPRESENTATION)
    # The validated seed carries the identical immutable geometric metadata.
    return (record, *legacy_geometry(record["seed_preparation"])[1:])


def sample_sg_pair(grid, density, potential, positions, thermal, *, carrier):
    """Original constant-face-flux SG formula evaluated in pair arithmetic."""
    grid, positions = np.asarray(grid, dtype=float), np.asarray(positions, dtype=float)
    if (carrier not in ("electron", "hole") or grid.ndim != 1 or positions.ndim != 1
            or len(grid) < 2 or np.any(np.diff(grid) <= 0)
            or density.shape != grid.shape or potential.shape != grid.shape
            or np.any(density <= 0) or np.any(positions < grid[0]) or np.any(positions > grid[-1])
            or not np.isfinite(grid).all() or not np.isfinite(positions).all()
            or not np.isfinite(thermal) or thermal <= 0):
        raise ValueError("invalid pair SG projection inputs")
    cells = np.clip(np.searchsorted(grid, positions, side="right")-1, 0, len(grid)-2)
    # Physical coordinates are the original binary64 grid; state arithmetic is DD.
    fraction = (positions-grid[cells])/(grid[cells+1]-grid[cells])
    xi = (potential[cells+1]-potential[cells]) * (1. if carrier == "electron" else -1.) / thermal
    weight = fraction * bernoulli_pair(xi) / bernoulli_pair(fraction*xi)
    values = (1-weight)*density[cells] + weight*density[cells+1]
    if np.any(values <= 0):
        raise ValueError("pair SG projection produced nonpositive density")
    return values


class PairIonProfile:
    """The original conservative profile with pair density, limiter and moments."""
    def __init__(self, faces, density, breaks=()):
        from .one_dimensional_mechanism_r1_spatial import ConservativeIonProfile
        # Validate the original geometric contract without using its rounded slopes.
        checked = ConservativeIonProfile(faces, density.hi, breaks)
        self.faces, self.width = checked.faces_m, np.diff(checked.faces_m)
        self.center, self.density = checked.centers_m, density
        if np.any(density < 0):
            raise ValueError("pair ion density must be nonnegative")
        slopes = [DD(0.) for _ in range(len(density))]
        for start, stop in zip((0, *breaks), (*breaks, len(density))):
            if stop-start < 2:
                continue
            secants = [(density[i+1]-density[i])/(self.center[i+1]-self.center[i])
                       for i in range(start, stop-1)]
            slopes[start], slopes[stop-1] = secants[0], secants[-1]
            for index in range(start+1, stop-1):
                left, right = secants[index-start-1:index-start+1]
                if bool(left != 0) and bool(right != 0) and bool(left < 0) == bool(right < 0):
                    central = (density[index+1]-density[index-1])/(self.center[index+1]-self.center[index-1])
                    magnitude = min(abs(2*left), abs(central), abs(2*right))
                    slopes[index] = -magnitude if bool(left < 0) else magnitude
        half = []
        for i, slope in enumerate(slopes):
            change = min(abs(slope)*(self.width[i]/2), density[i])
            half.append(-change if bool(slope < 0) else change if bool(slope != 0) else DD(0.))
        self.half = stack(half)
        self.slope = self.half/(self.width/2)

    def sample(self, positions, *, side="right"):
        positions = np.asarray(positions, dtype=float)
        if (positions.ndim != 1 or side not in ("left", "right") or not np.isfinite(positions).all()
                or np.any(positions < self.faces[0]) or np.any(positions > self.faces[-1])):
            raise ValueError("invalid pair ion sampling position")
        cells = np.clip(np.searchsorted(self.faces, positions, side=side)-1, 0, len(self.density)-1)
        fraction = (positions-self.faces[cells])/self.width[cells]
        values = (DD(1.)-fraction)*(self.density[cells]-self.half[cells]) + fraction*(self.density[cells]+self.half[cells])
        if np.any(values < 0):
            raise ValueError("pair ion profile produced negative density")
        return values

    def moments(self, left, right):
        if not self.faces[0] <= left < right <= self.faces[-1]:
            raise ValueError("pair moment interval is outside the physical volume")
        lo, hi = np.maximum(self.faces[:-1], left), np.minimum(self.faces[1:], right)
        active = hi > lo
        width, midpoint = hi[active]-lo[active], (lo[active]+(hi[active]-lo[active])/2)
        mass = width*(self.density[active]+self.slope[active]*(midpoint-self.center[active]))
        first = midpoint*mass+self.slope[active]*(DD(width)*width*width)/12
        return dd_sum(mass), dd_sum(first)


def conservative_centroids(prepared, state=None):
    record, _, faces, _, _, _, _, components, cuts = _geometry(prepared)
    snapshot = record["state"] if state is None else state
    codec.validate_snapshot(snapshot)
    return centroids_from_profile(faces, field(snapshot, "positive_m3"), components, cuts)


def centroids_from_profile(faces, density, components, cuts):
    """Pair centroids for already verified physical geometry and populations."""
    profile = PairIonProfile(faces, density, cuts)
    values = []
    for nodes in components:
        mass, first = profile.moments(faces[nodes[0]], faces[nodes[-1]+1])
        if bool(mass <= 0):
            raise ValueError("active pair ionic component has no inventory")
        values.append(first/mass)
    return stack(values)


def sample_pair_state(prepared, state=None):
    record, x, faces, bounds, positions, thermal, background, components, cuts = _geometry(prepared)
    state = record["state"] if state is None else state
    codec.validate_snapshot(state)
    n, p, phi, ions, trace, trace_phi, occupancy = [field(state, key) for key in
        ("n_m3", "p_m3", "phi_V", "positive_m3", "trace_state_m3", "trace_potential_V", "occupancy")]
    if np.any(n <= 0) or np.any(p <= 0) or np.any(trace <= 0) or np.any(occupancy < 0) or np.any(occupancy > 1):
        raise ValueError("pair spatial state violates population bounds")
    sampled_n, sampled_p, sampled_phi = [], [], []
    for layer, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
        indices = np.flatnonzero((x >= left) & (x <= right))
        local_x = x[indices]
        local = [value[indices] for value in (n, p, phi)]
        endpoints = (trace[0, 2*layer], trace[0, 2*layer+1], trace_phi[0, layer])
        if layer == 0:
            local_x = np.r_[local_x, right]
            local = [concatenate((value, endpoint.reshape(1))) for value, endpoint in zip(local, endpoints)]
        else:
            local_x = np.r_[left, local_x]
            local = [concatenate((endpoint.reshape(1), value)) for value, endpoint in zip(local, endpoints)]
        queries = positions[17*layer:17*(layer+1)]
        sampled_n.append(sample_sg_pair(local_x, local[0], local[2], queries, thermal, carrier="electron"))
        sampled_p.append(sample_sg_pair(local_x, local[1], local[2], queries, thermal, carrier="hole"))
        cells = np.clip(np.searchsorted(local_x, queries, side="right")-1, 0, len(local_x)-2)
        fraction = (queries-local_x[cells])/(local_x[cells+1]-local_x[cells])
        sampled_phi.append((DD(1.)-fraction)*local[2][cells]+fraction*local[2][cells+1])
    profile, reference = PairIonProfile(faces, ions, cuts), PairIonProfile(faces, DD(background), cuts)
    inventory, centroid = [], []
    for nodes in components:
        mass, first = profile.moments(faces[nodes[0]], faces[nodes[-1]+1])
        if bool(mass <= 0):
            raise ValueError("active pair ionic component has no inventory")
        inventory.append(mass)
        centroid.append(first/mass)
    return {"position_m": positions, "interface_position_m": bounds[1:-1],
        "n_m3": concatenate(sampled_n), "p_m3": concatenate(sampled_p), "phi_V": concatenate(sampled_phi),
        "positive_m3": profile.sample(positions), "background_positive_m3": reference.sample(positions),
        "interface_carriers_m3": trace, "interface_phi_V": trace_phi,
        "interface_positive_m3": stack((profile.sample(bounds[1:-1], side="left"), profile.sample(bounds[1:-1], side="right")), axis=1),
        "interface_background_positive_m3": stack((reference.sample(bounds[1:-1], side="left"), reference.sample(bounds[1:-1], side="right")), axis=1),
        "occupancy": occupancy, "ion_inventory_m2": stack(inventory), "ion_centroid_m": stack(centroid)}


def spatial_responses(prepared, step):
    from .one_dimensional_mechanism_r1_spatial import R1SpatialResponses
    from .one_dimensional_mechanism_r1_convergence import R1Response
    record = prepared.to_dict() if hasattr(prepared, "to_dict") else prepared
    codec.decode_prepared(record, representation=codec.REPRESENTATION)
    if (step.get("schema") != codec.STEP_SCHEMA or step.get("representation") != codec.REPRESENTATION
            or step.get("prepared_sha256") != record["sha256"]
            or step.get("reference_sha256") != record["reference_sha256"]
            or step.get("intervals") != record["intervals"] or step.get("source") != record.get("source")):
        raise ValueError("pair step and prepared state identities disagree")
    if step.get("sha256") != codec.digest({key: value for key, value in step.items() if key != "sha256"}):
        raise ValueError("pair step content digest differs")
    times = np.asarray(step["times_s"], dtype=float)
    if times.ndim != 1 or len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("pair spatial times must be zero then strictly increasing")
    finest = max(step["policy"]["refinement_substeps"])
    rows = [row for row in step["accepted_steps"] if row["substeps"] == finest]
    samples = []
    for time in times:
        selected = [row for row in rows if row["time_s"] == time]
        if len(selected) != 1 or selected[0]["phase"] != ("0+" if time == 0 else "accepted_regular_step"):
            raise ValueError("pair response requires exactly one correctly phased finest output row")
        samples.append(sample_pair_state(record, selected[0]["state"]))
    base = sample_pair_state(record)
    arrays = {key: stack([sample[key] for sample in samples]) for key in base if isinstance(base[key], DD)}
    positions = {"time_s": times, "position_m": base["position_m"]}
    interfaces = {"time_s": times, "interface_position_m": base["interface_position_m"]}
    def response(value, coordinates, components=()):
        return R1Response(value.to_float(), coordinates, components)
    responses = {
        "response_potential": response(arrays["phi_V"]-base["phi_V"], positions),
        "carrier_log_density": response(stack((log(arrays["n_m3"]/base["n_m3"]), log(arrays["p_m3"]/base["p_m3"])), axis=-1), positions, ("electron", "hole")),
        "ion_density_over_p0": response((arrays["positive_m3"]-base["positive_m3"])/base["background_positive_m3"], positions),
        "trap_occupancy_change": response(arrays["occupancy"]-base["occupancy"], interfaces),
        "ion_centroid_change": response(arrays["ion_centroid_m"]-base["ion_centroid_m"], {"time_s": times, "active_component": np.arange(base["ion_centroid_m"].size)})}
    sides = {
        "response_potential": response(arrays["interface_phi_V"]-base["interface_phi_V"], interfaces, ("left", "right")),
        "carrier_log_density": response(log(arrays["interface_carriers_m3"]/base["interface_carriers_m3"]), interfaces, ("n_left", "p_left", "n_right", "p_right")),
        "ion_density_over_p0": response((arrays["interface_positive_m3"]-base["interface_positive_m3"])/base["interface_background_positive_m3"], interfaces, ("left", "right"))}
    metadata = {"stack_sha256": record["stack_sha256"], "reference_sha256": record["reference_sha256"],
        "source_sha256": (step.get("source") or {}).get("sha256"), "amplitude_V": step["amplitude_V"],
        "control": step["control_label"], "prepared_sha256": record["sha256"], "times_s": times.tolist(),
        "intervals": record["intervals"], "policy": step["policy"],
        "recorded_certificate_passed": step.get("certificate", {}).get("certified") is True,
        "representation": codec.REPRESENTATION, "rounding_boundary": "after_pair_projection_and_response_difference",
        "carrier_resolution": "no independent carrier uncertainty supplied; signal resolution undetermined",
        "ionic_centroid_definition": "integral(x*P_reconstructed)/integral(P_reconstructed) over each active physical component"}
    return R1SpatialResponses(responses, sides, response(base["phi_V"], {"position_m": base["position_m"]}), metadata,
        arrays["ion_inventory_m2"].to_float(), dc_interface_potential=response(base["interface_phi_V"],
            {"interface_position_m": base["interface_position_m"]}, ("left", "right")))
