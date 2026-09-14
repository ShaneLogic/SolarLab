"""R1-0 research lane. See docs/OneDimensionalMechanismR1GeometryV1.md."""

from dataclasses import replace
import hashlib
import json

import numpy as np

from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    _build_qf_material, _prepare_two_sided_material, _research_charge_off_stack,
    build_two_sided_trace_grid,
)
from perovskite_sim.physics.interface_plane import FERMI_DIRAC_RICHARDSON
from perovskite_sim.physics.physical_control_volume import PHYSICAL_GEOMETRY_V1
from perovskite_sim.experiments.interface_defect_ion_transient import _InterfaceIonTransientSystem


class PhysicalInterfaceIonSystem(_InterfaceIonTransientSystem):
    """Lift the exact dielectric voltage change out of small Newton increments."""

    def set_voltage_lift(self, voltage, previous):
        from perovskite_sim.solver.mol import poisson_right_boundary
        drop = poisson_right_boundary(self.material, voltage)-previous.phi[-1]
        resistance = 1/self.material.poisson_factor.C
        self._lift = drop*np.r_[0., np.cumsum(resistance)]/np.sum(resistance)
        self._lift[-1] = drop
        self._lift_displacement = -drop/np.sum(resistance)
        self._trace_lift = np.empty((self.interface_count, 2))
        from perovskite_sim.constants import EPS_0
        for k, left in enumerate(self.left_nodes):
            c_left = EPS_0*self.material.eps_r[left]/self.material.iface_qss_left_distances_m[k]
            self._trace_lift[k, :] = self._lift[left]-self._lift_displacement/c_left
        self._lift_free_residual = False

    def _coordinates(self, coordinate, voltage):
        result = list(super()._coordinates(coordinate, voltage))
        if hasattr(self, "_lift"):
            result[0] = result[0].copy()
            result[1] = result[1].copy()
            result[0][1:-1] -= self._lift[1:-1]
            result[1][1:-1] += self._lift[1:-1]
            result[2][1:-1] += self._lift[1:-1]
            result[6] += self._trace_lift
        return tuple(result)

    def potential_increment(self, state, previous):
        value = super().potential_increment(state, previous)
        if hasattr(self, "_lift"):
            if self._lift_free_residual:
                value[-1] -= self._lift[-1]
            else:
                value[1:-1] += self._lift[1:-1]
        return value

    def _with_step_electrostatics(self, state):
        # The harmonic lift has exactly zero divergence and trace jump.
        # Keep the previous residual, while avoiding roundoff from L*lift.
        self._lift_free_residual = True
        try:
            return super()._with_step_electrostatics(state)
        finally:
            self._lift_free_residual = False

    def interface_current_sides(self, state, previous=None, dt=None):
        self._lift_free_residual = True
        try:
            conduction, displacement, _ = super().interface_current_sides(state, previous, dt)
        finally:
            self._lift_free_residual = False
        if dt is not None and hasattr(self, "_lift"):
            displacement += self.polarity*self._lift_displacement/dt
        return conduction, displacement, conduction+displacement


def build_r1_material(stack, intervals):
    """Keep the declared species and parameters; change only physical geometry."""
    from perovskite_sim.experiments.dynamic_defect_transient import classify_dynamic_defect_transient_capability
    # Reuse the strict source topology instead of widening a legacy capability.
    classify_dynamic_defect_transient_capability(stack)
    charge_off, microscopic = _research_charge_off_stack(stack)
    grid = build_two_sided_trace_grid(build_electrical_grid(stack, intervals), stack)
    material = _build_qf_material(grid, charge_off, defect_energy_quadrature_order=32)
    material = _prepare_two_sided_material(
        grid, charge_off, material, physical_boundary_volumes=True,
    )
    material = replace(
        material, N_iface_state=0, iface_state_v_th=1e5,
        iface_state_live_proj=True, iface_state_shared_occ=True,
        iface_state_physical_offsets=True, iface_qss_exclusive_transport=True,
        iface_qss_cross_transmission=1.0,
        iface_qss_transport_model=FERMI_DIRAC_RICHARDSON,
        iface_qss_allow_inexact_inner=True,
    )
    if (material.has_dual_ions or material.monovalent_bulk_defects is not None
            or len(microscopic.documents) != 1
            or len(material.iface_qss_interface_positions_m) != 1):
        raise ValueError("R1-0 supports one interface and positive ions only")
    return grid, material


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def solve_r1_dc(stack, intervals, reference=None):
    """Solve the same inventory-constrained DC equations on physical volumes."""
    from perovskite_sim.experiments.defect_ion_combined_impedance import (
        _build_ion_layout, _component_inventories, _DCSolveContext, _solve_combined_dc,
    )
    from perovskite_sim.experiments.quasi_fermi_steady_state import _QuasiFermiSystem
    from perovskite_sim.physics.contacts import require_contact_thermodynamic_certificate
    from perovskite_sim.physics.two_sided_interface import (
        TWO_SIDED_TRACE, solve_material_two_sided_interfaces_qss,
    )
    x, material = build_r1_material(stack, intervals)
    off, microscopic = _research_charge_off_stack(stack)
    layout = _build_ion_layout(material)
    target = _component_inventories(material.P_ion0, layout.positive_components, material.dx_cell)
    system = _QuasiFermiSystem(
        x, off, material, 0., interface_boundary=True,
        interface_topology=TWO_SIDED_TRACE, interface_transmission=1.,
        interface_transport_model=FERMI_DIRAC_RICHARDSON,
        interface_charge_reference_occupancy=reference,
        interface_charge_trap_density_m2=None if reference is None else microscopic.trap_density_m2,
        poisson_tolerance_V=1e-13, poisson_max_iterations=100,
    )
    contact = require_contact_thermodynamic_certificate(off, material)
    context = _DCSolveContext(x, off, material, system, layout, 0., 0., target, np.empty(0), contact)
    state, value = _solve_combined_dc(
        context, maximum_normalized_residual=1e-8,
        maximum_continuity_bound_A_m2=1e-4, maximum_ionic_face_current_A_m2=1e-6,
        maximum_inventory_error=1e-10, maximum_poisson_residual=1e-8,
        maximum_face_current_spread_A_m2=1e-4, max_nfev=1000,
    )
    if not state.certificate.certified:
        raise RuntimeError(f"R1 DC failed: {state.certificate.reasons}")
    local = solve_material_two_sided_interfaces_qss(
        material, off, value.y[:x.size], value.y[x.size:2*x.size], value.phi,
        cross_transmission=1., interface_transport_model=FERMI_DIRAC_RICHARDSON,
        fail_on_residual=True,
    )
    return x, material, state, local


def prepare_reference(stack, output, write_json):
    """Save every rung before applying the predeclared 1e-8 reference gate."""
    previous = None
    rungs = []
    for intervals in [32, 64, 128, 256, 512]:
        print(f"reference intervals={intervals}", flush=True)
        x, material, state, local = solve_r1_dc(stack, intervals)
        occupancy = np.asarray(local.occupancy)
        difference = None if previous is None else float(max(abs(occupancy-previous)))
        record = {
            "intervals": intervals, "grid_m": x, "faces_m": material.physical_cell_faces_m,
            "widths_m": material.dx_cell, "dc_state": state, "local_interface": local,
            "f_ref": occupancy, "difference": difference,
        }
        write_json(output / f"ReferenceGrid{intervals}V1.json", record)
        rungs.append({"intervals": intervals, "f_ref": occupancy.tolist(), "difference": difference})
        if intervals >= 128 and difference <= 1e-8:
            binding = {
                "schema": "ReferenceBindingV1", "geometry": PHYSICAL_GEOMETRY_V1,
                "f_ref": occupancy.tolist(), "reference_intervals": intervals,
                "rungs": rungs, "maximum_difference": 1e-8,
                "microscopic_documents": list(_research_charge_off_stack(stack)[1].document_sha256),
                "stack_sha256": hashlib.sha256(repr(stack).encode("utf-8")).hexdigest(),
                "scope": "synthetic_300K_dark_fixed_energy_positive_ion_reference",
            }
            binding["sha256"] = _digest(binding)
            write_json(output / "ReferenceBindingV1.json", binding)
            return binding
        previous = occupancy
    raise RuntimeError("f_ref failed the 1e-8 grid criterion at the 512-interval limit")


def validate_binding(binding, stack):
    if not isinstance(binding, dict):
        raise ValueError("R1 reference binding must be an object")
    payload = {k: v for k, v in binding.items() if k != "sha256"}
    if (binding.get("sha256") != _digest(payload)
            or binding.get("schema") != "ReferenceBindingV1"
            or binding.get("geometry") != PHYSICAL_GEOMETRY_V1
            or binding.get("microscopic_documents") != list(_research_charge_off_stack(stack)[1].document_sha256)
            or binding.get("stack_sha256") != hashlib.sha256(repr(stack).encode("utf-8")).hexdigest()):
        raise ValueError("R1 reference identity mismatch")
    f = np.asarray(binding.get("f_ref"), dtype=float)
    rungs = binding.get("rungs", [])
    if (f.shape != (1,) or not np.all(np.isfinite(f)) or np.any((f <= 0) | (f >= 1))
            or len(rungs) < 3 or not np.isfinite(rungs[-1]["difference"])
            or rungs[-1]["difference"] > 1e-8
            or rungs[-1]["intervals"] != binding["reference_intervals"]
            or rungs[-1]["f_ref"] != binding["f_ref"]):
        raise ValueError("R1 reference did not pass the refinement gate")
    intervals = [r["intervals"] for r in rungs]
    if intervals != [32, 64, 128, 256, 512][:len(intervals)]:
        raise ValueError("R1 reference ladder must follow 32..512 doubling")
    for previous, current in zip(rungs, rungs[1:]):
        difference = float(max(abs(np.asarray(current["f_ref"])-previous["f_ref"])))
        if not np.isfinite(difference) or difference != current["difference"]:
            raise ValueError("R1 reference differences do not match saved occupancies")


def validate_physical_material(grid, stack, material):
    from perovskite_sim.physics.physical_control_volume import physical_cell_faces
    from perovskite_sim.experiments.quasi_fermi_steady_state import _interface_positions
    from perovskite_sim.models.device import electrical_layers
    from perovskite_sim.physics.contacts import resolved_contact_velocities
    faces = physical_cell_faces(grid, _interface_positions(stack))
    thickness = sum(layer.thickness for layer in electrical_layers(stack))
    if (grid[0] != 0 or not np.isclose(grid[-1], thickness, rtol=1e-14, atol=0)
            or not np.array_equal(material.physical_cell_faces_m, faces)
            or not np.array_equal(material.dx_cell, np.diff(faces))
            or not np.array_equal(material.poisson_factor.h_cell, material.dx_cell[1:-1])
            or resolved_contact_velocities(stack) != (None,)*4):
        raise ValueError("R1 grid, volumes, Poisson storage, or pinned contacts mismatch")


def physical_step_record(system, state, previous, dt):
    """Independent Gauss and contact observations; no projected equal currents."""
    from perovskite_sim.constants import Q
    from perovskite_sim.physics.physical_control_volume import physical_contact_displacement
    from perovskite_sim.physics.recombination import total_recombination
    mat, w = system.material, system.widths
    rho, _ = system.system._bulk_space_charge_and_tangent(
        state.n, state.p, positive_ion_density_m3=state.positive,
        negative_ion_density_m3=state.negative,
    )
    displacement = -mat.poisson_factor.C*np.diff(state.phi)
    contact_d = physical_contact_displacement(displacement, rho, w)
    charge = float(np.dot(rho, w)+np.sum(state.sheet_charge))
    gauss_error = float(contact_d[1]-contact_d[0]-charge)
    # Pinned endpoint carrier reservoirs have zero density derivative.
    # Pair recombination contributes opposite corrections to Jn and Jp.
    recombination = total_recombination(
        state.n, state.p, mat.ni_sq, mat.tau_n, mat.tau_p,
        mat.n1, mat.p1, mat.B_rad, mat.C_n, mat.C_p,
    )
    jn = state.current_n[[0, -1]] + Q*recombination[[0, -1]]*w[[0, -1]]*[-1, 1]
    jp = state.current_p[[0, -1]] + Q*recombination[[0, -1]]*w[[0, -1]]*[1, -1]
    jc = jn+jp
    jd = np.zeros(2)
    charge_rate = 0.
    balance_error = 0.
    internal_total = state.conduction/system.polarity
    trap_storage_error = 0.
    if previous is not None:
        storage = system.storage_increment(state, previous)
        delta_rho = system._increment_charge_density(storage)
        delta_d = -mat.poisson_factor.C*np.diff(system.potential_increment(state, previous))
        jd = physical_contact_displacement(delta_d, delta_rho, w)/dt
        occupied = storage[2*system.interior_count:2*system.interior_count+system.interface_count]
        charge_rate = float((np.dot(delta_rho, w)-Q*np.sum(occupied))/dt)
        balance_error = float(charge_rate-(jc[0]-jc[1]))
        internal_total = system.transient_current_metrics(state, previous, dt)[1]/system.polarity
        trap_rate = state.rate[2*system.interior_count:2*system.interior_count+system.interface_count]
        trap_storage_error = float(max(abs(Q*(occupied/dt-trap_rate))))
    total = jc+jd
    all_total = np.r_[internal_total, total]
    scale = max(float(max(abs(all_total))), 1e-20)
    balance_scale = max(abs(charge_rate), abs(jc[0]-jc[1]), max(abs(state.conduction)), 1.)
    active = system.positive_nodes
    inventory = float(np.dot(state.positive[active], w[active]))
    return {
        "full_charge_C_m2": charge, "interior_charge_C_m2": system.integrated_charge(state),
        "rho_C_m3": rho, "physical_displacement_C_m2": contact_d,
        "gauss_error_C_m2": gauss_error, "gauss_normalized": abs(gauss_error)/(Q*1e15),
        "contact_electron_current_A_m2": jn, "contact_hole_current_A_m2": jp,
        "contact_ion_current_A_m2": np.zeros(2), "contact_conduction_A_m2": jc,
        "contact_displacement_A_m2": jd, "contact_maxwell_A_m2": total,
        "internal_maxwell_A_m2": internal_total,
        "charge_rate_A_m2": charge_rate, "charge_balance_error_A_m2": balance_error,
        "charge_balance_normalized": abs(balance_error)/balance_scale,
        "contact_internal_current_spread_relative": float(np.ptp(all_total))/scale,
        "trap_storage_error_A_m2": trap_storage_error,
        "inventory_m2": inventory, "inventory_relative_drift": abs(inventory/1e15-1),
    }


def physical_ac_observation(mat, system, value, positive, dynamic, electron, hole, ion):
    """Observe real contacts and both zero-volume interface traces before FD."""
    from perovskite_sim.constants import EPS_0, Q
    from perovskite_sim.physics.physical_control_volume import physical_contact_displacement
    from perovskite_sim.physics.recombination import total_recombination
    count = positive.size
    n, p = value.y[:count], value.y[count:2*count]
    rho, _ = system._bulk_space_charge_and_tangent(n, p, positive_ion_density_m3=positive)
    displacement = -mat.poisson_factor.C*np.diff(value.phi)
    contacts = physical_contact_displacement(displacement, rho, mat.dx_cell)
    rec = total_recombination(n, p, mat.ni_sq, mat.tau_n, mat.tau_p, mat.n1, mat.p1,
                              mat.B_rad, mat.C_n, mat.C_p)
    polarity = mat.junction_polarity
    e_contacts = electron[[0, -1]] + polarity*Q*rec[[0, -1]]*mat.dx_cell[[0, -1]]*[-1, 1]
    h_contacts = hole[[0, -1]] + polarity*Q*rec[[0, -1]]*mat.dx_cell[[0, -1]]*[1, -1]
    local = dynamic if dynamic is not None else value.interface_charge_qss
    for k in reversed(range(len(mat.iface_qss_interface_faces))):
        face = mat.iface_qss_interface_faces[k]
        left, right = mat.iface_qss_left_nodes[k], mat.iface_qss_right_nodes[k]
        cl = EPS_0*mat.eps_r[left]/mat.iface_qss_left_distances_m[k]
        cr = EPS_0*mat.eps_r[right]/mat.iface_qss_right_distances_m[k]
        off = (cl*value.phi[left]+cr*value.phi[right])/(cl+cr)
        trace = off + local.trace_potential_shift_V[k]
        flux = local.qss.bulk_flux_m2_s.reshape(-1, 4)[k]
        electron[face] = -polarity*Q*flux[2]
        hole[face] = polarity*Q*flux[3]
        displacement[face] = -cl*(trace[0]-value.phi[left])
        electron = np.insert(electron, face+1, polarity*Q*flux[0])
        hole = np.insert(hole, face+1, -polarity*Q*flux[1])
        ion = np.insert(ion, face+1, ion[face])
        displacement = np.insert(displacement, face+1, cr*(trace[1]-value.phi[right]))
    return (
        np.r_[e_contacts[0], electron, e_contacts[-1]],
        np.r_[h_contacts[0], hole, h_contacts[-1]],
        np.r_[0., ion, 0.],
        polarity*np.r_[contacts[0], displacement, contacts[-1]],
    )


def run_coupled(stack, intervals, binding, output, write_json, *, nonlinear_factor=1.):
    from perovskite_sim.experiments.interface_defect_ion_transient import (
        InterfaceDefectIonTransientPolicy, run_interface_defect_ion_device_transient,
    )
    validate_binding(binding, stack)
    x, material = build_r1_material(stack, intervals)
    records = []
    def observe(system, state, previous, dt, time, substeps, residual):
        record = physical_step_record(system, state, previous, dt)
        record.update(
            time_s=time, dt_s=dt, substeps=substeps, scaled_nonlinear_residual=residual,
            n_m3=state.n, p_m3=state.p, positive_m3=state.positive,
            phi_V=state.phi, occupancy=state.occupancy, sheet_charge_C_m2=state.sheet_charge,
            trace_potential_V=np.asarray([v.trace_potential for v in state.local]),
            trace_state_m3=np.asarray([v.state_m3 for v in state.local]),
            capture_m2_s=np.asarray([v.tangent.balance.capture_flux_m2_s for v in state.local]),
            poisson_residual_C_m2=state.poisson_residual,
            direct_poisson_residual_C_m2=state.direct_poisson_residual,
            local_residual=state.local_residual,
        )
        records.append(record)
        write_json(output / "AcceptedStepsV1.json", records)
    policy = InterfaceDefectIonTransientPolicy(
        maximum_newton_iterations=100, maximum_line_search_steps=40,
        maximum_near_acceptance_nonmonotone_steps=2,
        maximum_ion_inventory_relative_drift=1e-10,
    )
    fields = (
        "storage_relative_tolerance", "carrier_storage_atol_m3", "interface_storage_atol_m2",
        "ion_storage_atol_m3", "poisson_relative_tolerance", "poisson_atol_C_m2",
        "interface_algebraic_relative_tolerance", "interface_potential_atol_V",
        "interface_gauss_atol_C_m2", "interface_flux_atol_m2_s",
    )
    policy = replace(policy, **{key: getattr(policy, key)*nonlinear_factor for key in fields})
    write_json(output / "ProtocolV1.json", {
        "geometry": PHYSICAL_GEOMETRY_V1, "grid_m": x, "widths_m": material.dx_cell,
        "faces_m": material.physical_cell_faces_m, "reference": binding,
        "times_s": [0., 1e-8, 1e-6, 1e-4], "voltage_V": [0., .005, .005, .005],
        "policy": policy, "scope": "R1-0 short coupled geometry test; not R1-1 ideal-step separation",
    })
    result = run_interface_defect_ion_device_transient(
        x, stack, [0., 1e-8, 1e-6, 1e-4], [0., .005, .005, .005],
        mat=material, policy=policy, research_binding=binding, accepted_step_observer=observe,
    )
    write_json(output / "RawCoupledV1.json", result)
    limits = {"gauss_normalized": 1e-10, "charge_balance_normalized": 1e-10,
              "inventory_relative_drift": 1e-10, "contact_internal_current_spread_relative": 2e-6}
    maxima = {
        key: max(r[key] for r in records if r["dt_s"] > 0 or key not in ["contact_internal_current_spread_relative"])
        for key in limits
    }
    write_json(output / "PhysicalCertificateV1.json", {
        "maxima": maxima, "limits": limits, "engine": result.certificate,
        "certified": all(maxima[k] <= limits[k] for k in limits),
    })
    if not all(maxima[k] <= limits[k] for k in limits):
        raise RuntimeError(f"physical coupled gates failed: {maxima}")
