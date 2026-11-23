"""Unit tests for ``solarlab_research.precision`` (P07-01).

Covers:
* ``DD`` exact TwoSum/TwoProduct, immutability, normalized ``(hi, lo)`` roundtrip,
  and ``exp``/``expm1``/``log``/``log1p``/``sum`` to ``<= 1e-30`` relative error
  against 90-digit ``decimal.Decimal`` across sub-ULP and moderate inputs.
* ``DDArray`` and ``CompensatedArithmetic`` conformance with ``solarlab.contracts``
  (``PhysicalArray``, ``AssemblyArithmetic``, ``FieldProjection``, ``log_ratio``).
* ``freeze_coefficients``, ``evaluate_local_carrier_pair``, and ``to_float_tangent``:
  Fermi-table SHA-256 tamper rejection, exact compensated Jacobians vs high-precision
  central differences in ``DD``, and ``UnsupportedCompensatedInterfaceDomain``
  without binary64 fallback.
* ``PairStateAuthority``, ``ReferenceIncrementCoordinates``, ``AffineInputLiftCodec``,
  and ``decode_historical_r1_pair_state`` (plus tamper rejection).
* Boundary & registry hygiene: zero ``PhysicsModel`` registrations in
  ``solarlab.physics.registry`` from ``solarlab_research``, zero imports from
  ``perovskite_sim`` / ``solarlab_server`` / ``backend``, and zero private ``_``
  imports/attributes from ``solarlab``.
"""

from __future__ import annotations

import ast
from decimal import Decimal, localcontext
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from solarlab.contracts import (
    AssemblyArithmetic,
    ContractError,
    EquationSpec,
    Layout,
    ONE,
    PARTICLE,
    PhysicalArray,
    StateAuthority,
    Support,
    VOLT,
    VOLUME,
    VariableSpec,
)
from solarlab_research.precision import (
    AffineInputLiftCodec,
    CompensatedArithmetic,
    DD,
    DDArray,
    InputLiftCheckpointError,
    PairStateAuthority,
    ReferenceIncrementCoordinates,
    UnsupportedCompensatedInterfaceDomain,
    decode_historical_r1_pair_state,
    default_fd_table,
    evaluate_local_carrier_pair,
    exp,
    expm1,
    freeze_coefficients,
    log,
    log1p,
    sum as dd_sum,
    to_float_tangent,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESEARCH_SRC = REPO_ROOT / "src" / "solarlab_research"
HISTORICAL_PAIR_FIXTURE = (
    REPO_ROOT / "tests" / "refactor" / "fixtures" / "r1_pair_common_state_preparation.json"
)


def _dd_to_decimal(val: DD) -> Decimal:
    assert val.ndim == 0
    return Decimal(float(val.hi.item())) + Decimal(float(val.lo.item()))


def _rel_err_decimal(val: DD, ref: Decimal) -> float:
    got = _dd_to_decimal(val)
    if ref == 0:
        return float(abs(got))
    return float(abs(got - ref) / abs(ref))


def test_dd_exact_arithmetic_immutability_and_transcendentals() -> None:
    sub_ulp = 2.0**-60
    added = DD(1.0) + DD(sub_ulp)
    recovered = added - DD(1.0)
    assert float(recovered) == sub_ulp
    assert float(added.hi) == 1.0
    assert float(added.lo) == sub_ulp

    with pytest.raises(AttributeError):
        added.hi = np.asarray(2.0)  # type: ignore[misc]
    with pytest.raises((ValueError, TypeError)):
        added.hi[...] = 2.0

    pair = added.to_pair_dict()
    rebuilt = DD.from_pair_dict(pair, expected_shape=())
    assert rebuilt == added
    with pytest.raises(ValueError, match="not normalized"):
        DD.from_pair_dict({"hi": 1.0, "lo": 0.25})

    with localcontext() as ctx:
        ctx.prec = 90
        for x_val in (-1.25, -1.0e-18, 1.0e-18, 0.375, 3.5):
            x_dd = DD(x_val)
            x_dec = Decimal(x_val)

            ref_exp = x_dec.exp()
            assert _rel_err_decimal(exp(x_dd), ref_exp) <= 1.0e-30

            ref_expm1 = x_dec.exp() - Decimal(1)
            assert _rel_err_decimal(expm1(x_dd), ref_expm1) <= 1.0e-30

        for y_val in (0.125, 0.999999999999, 1.000000000001, 7.5):
            y_dd = DD(y_val)
            y_dec = Decimal(y_val)
            assert _rel_err_decimal(log(y_dd), y_dec.ln()) <= 1.0e-30

        for z_val in (-0.2, -1.0e-18, 1.0e-18, 0.2):
            z_dd = DD(z_val)
            z_dec = Decimal(z_val)
            assert _rel_err_decimal(log1p(z_dd), (Decimal(1) + z_dec).ln()) <= 1.0e-30

    terms = DD(np.array([1.0, 2.0**-60, -1.0, 2.0**-60], dtype=np.float64))
    total = dd_sum(terms)
    assert float(total) == pytest.approx(2.0**-59, rel=0.0, abs=1.0e-32)


def _accepts_physical_array(arr: PhysicalArray) -> PhysicalArray:
    return arr


def _accepts_arithmetic(arith: AssemblyArithmetic) -> AssemblyArithmetic:
    return arith


def _accepts_authority(auth: StateAuthority) -> StateAuthority:
    return auth


def test_dd_array_and_compensated_arithmetic_contracts() -> None:
    base = DDArray.from_parts([1.0e22, 2.0e22, 4.0e22])
    perturbed = DDArray(base.values + base.values * DD([1.0e-18, -2.0e-18, 0.5]))
    assert _accepts_physical_array(base) is base

    ratio = perturbed.log_ratio(base)
    assert float(ratio.values[0]) == pytest.approx(1.0e-18, rel=1.0e-12)
    assert float(ratio.values[1]) == pytest.approx(-2.0e-18, rel=1.0e-12)
    assert float(ratio.values[2]) == pytest.approx(np.log(1.5), rel=1.0e-15)

    proj = perturbed.project_to_float(authority_identity="auth-1")
    assert proj.authority_identity == "auth-1"
    assert proj.evaluator_identity == "dd-high-word-projection-v1"
    assert proj.absolute_error_bound is not None
    assert np.all(np.asarray(proj.absolute_error_bound.values) >= np.abs(perturbed.values.lo))

    arith = CompensatedArithmetic()
    assert _accepts_arithmetic(arith) is arith
    mat = np.array([[1.0, 1.0], [2.0, -1.0]], dtype=np.float64)
    vec = DD(np.array([1.0, -1.0], dtype=np.float64), np.array([2.0**-60, 0.0], dtype=np.float64))
    prod = arith.matmul(mat, vec)
    assert float(prod[0]) == pytest.approx(2.0**-60, rel=0.0, abs=1.0e-32)
    assert float(prod[1]) == pytest.approx(3.0, rel=1.0e-15)

    scattered = arith.scatter_add(arith.zeros((3,)), np.array([1, 1], dtype=np.intp), DD([1.0, 2.0**-60]))
    assert float(scattered[1] - DD(1.0)) == pytest.approx(2.0**-60, rel=0.0, abs=1.0e-32)


def _sample_frozen_coefficients() -> Any:
    geometry = {"left_distance_m": 2.5e-8, "right_distance_m": 2.0e-8}
    physics = {
        "thermal_voltage_V": 0.02585199102456544,
        "temperature_K": 300.0,
        "transmission": 1.0,
        "D_n_left_m2_s": 5.0e-6,
        "D_p_left_m2_s": 5.0e-6,
        "D_n_right_m2_s": 1.5e-6,
        "D_p_right_m2_s": 1.5e-6,
        "N_C_left_m3": 2.0e25,
        "N_V_left_m3": 2.0e25,
        "N_C_right_m3": 4.0e25,
        "N_V_right_m3": 3.0e25,
        "n1_left_m3": 1.0e17,
        "p1_left_m3": 2.0e17,
        "n1_right_m3": 1.0e17,
        "p1_right_m3": 2.0e17,
        "richardson_n_A_m2_K2": 1.2e6,
        "richardson_p_A_m2_K2": 1.2e6,
        "surface_recombination_velocity_n_m_s": 80.0,
        "surface_recombination_velocity_p_m_s": 90.0,
        "conduction_band_step_eV": 0.08,
        "hole_transport_step_eV": 0.25,
    }
    return geometry, physics, freeze_coefficients(geometry, physics)


def test_two_sided_interface_fd_tamper_jacobians_and_domain_guards() -> None:
    geometry, physics, coeffs = _sample_frozen_coefficients()
    table, sha = default_fd_table()
    assert coeffs.fd_table_sha256 == sha

    with pytest.raises(ValueError, match="identity mismatch"):
        freeze_coefficients(geometry, physics, fd_table=table, fd_table_sha256="0" * 64)

    state_d = DD(np.array([1.2e19, 2.5e18, 3.0e20, 1.5e16], dtype=np.float64))
    trace_v = DD(np.array([0.12, 0.14], dtype=np.float64))
    bulk_d = DD(np.array([1.4e19, 2.2e18, 3.3e20, 1.2e16], dtype=np.float64))
    bulk_v = DD(np.array([0.11, 0.155], dtype=np.float64))
    occ = DD(0.37)

    ev = evaluate_local_carrier_pair(
        state_density=state_d,
        trace_potential=trace_v,
        bulk_density=bulk_d,
        bulk_potential=bulk_v,
        occupancy=occ,
        capture_multiplier=1.0,
        coefficients=coeffs,
    )
    projected = to_float_tangent(ev)
    assert projected.balance.jacobian_log_state_m2_s.shape == (4, 4)
    assert projected.balance.jacobian_trace_potential_m2_s_V.shape == (4, 2)
    assert projected.balance.jacobian_bulk_coordinates.shape == (4, 6)
    assert ev.metadata["fallback_used"] is False

    eps = DD(1.0e-10)
    # 4x4 log-state Jacobian check via central difference in DD
    jac_log = projected.balance.jacobian_log_state_m2_s
    for col in range(4):
        hi_p, lo_p = state_d.hi.copy(), state_d.lo.copy()
        hi_m, lo_m = state_d.hi.copy(), state_d.lo.copy()
        up = state_d[col] * exp(eps)
        dn = state_d[col] * exp(-eps)
        hi_p[col], lo_p[col] = up.hi, up.lo
        hi_m[col], lo_m[col] = dn.hi, dn.lo
        ev_p = evaluate_local_carrier_pair(
            state_density=DD.from_normalized_parts(hi_p, lo_p),
            trace_potential=trace_v,
            bulk_density=bulk_d,
            bulk_potential=bulk_v,
            occupancy=occ,
            capture_multiplier=1.0,
            coefficients=coeffs,
        )
        ev_m = evaluate_local_carrier_pair(
            state_density=DD.from_normalized_parts(hi_m, lo_m),
            trace_potential=trace_v,
            bulk_density=bulk_d,
            bulk_potential=bulk_v,
            occupancy=occ,
            capture_multiplier=1.0,
            coefficients=coeffs,
        )
        fd_col = ((ev_p.balance["residual_m2_s"] - ev_m.balance["residual_m2_s"]) / (DD(2.0) * eps)).to_float()
        assert np.allclose(jac_log[:, col], np.asarray(fd_col), rtol=1.0e-9, atol=1.0e-9)

    # 4x2 trace-potential Jacobian check
    jac_tr = projected.balance.jacobian_trace_potential_m2_s_V
    for col in range(2):
        hi_p, lo_p = trace_v.hi.copy(), trace_v.lo.copy()
        hi_m, lo_m = trace_v.hi.copy(), trace_v.lo.copy()
        up = trace_v[col] + eps
        dn = trace_v[col] - eps
        hi_p[col], lo_p[col] = up.hi, up.lo
        hi_m[col], lo_m[col] = dn.hi, dn.lo
        ev_p = evaluate_local_carrier_pair(
            state_density=state_d,
            trace_potential=DD.from_normalized_parts(hi_p, lo_p),
            bulk_density=bulk_d,
            bulk_potential=bulk_v,
            occupancy=occ,
            capture_multiplier=1.0,
            coefficients=coeffs,
        )
        ev_m = evaluate_local_carrier_pair(
            state_density=state_d,
            trace_potential=DD.from_normalized_parts(hi_m, lo_m),
            bulk_density=bulk_d,
            bulk_potential=bulk_v,
            occupancy=occ,
            capture_multiplier=1.0,
            coefficients=coeffs,
        )
        fd_col = ((ev_p.balance["residual_m2_s"] - ev_m.balance["residual_m2_s"]) / (DD(2.0) * eps)).to_float()
        assert np.allclose(jac_tr[:, col], np.asarray(fd_col), rtol=1.0e-9, atol=1.0e-9)

    # Occupancy derivative check
    ev_occ_p = evaluate_local_carrier_pair(
        state_density=state_d,
        trace_potential=trace_v,
        bulk_density=bulk_d,
        bulk_potential=bulk_v,
        occupancy=occ + eps,
        capture_multiplier=1.0,
        coefficients=coeffs,
    )
    ev_occ_m = evaluate_local_carrier_pair(
        state_density=state_d,
        trace_potential=trace_v,
        bulk_density=bulk_d,
        bulk_potential=bulk_v,
        occupancy=occ - eps,
        capture_multiplier=1.0,
        coefficients=coeffs,
    )
    fd_occ = (
        (ev_occ_p.balance["capture_flux_m2_s"] - ev_occ_m.balance["capture_flux_m2_s"]) / (DD(2.0) * eps)
    ).to_float()
    assert np.allclose(projected.capture_flux_occupancy_derivative_m2_s, np.asarray(fd_occ), rtol=1.0e-9, atol=1.0e-9)

    # Domain guards raise UnsupportedCompensatedInterfaceDomain without silent fallback
    degenerate_state = DD(np.array([1.5e25, 2.5e18, 3.0e20, 1.5e16], dtype=np.float64))
    with pytest.raises(UnsupportedCompensatedInterfaceDomain):
        evaluate_local_carrier_pair(
            state_density=degenerate_state,
            trace_potential=trace_v,
            bulk_density=bulk_d,
            bulk_potential=bulk_v,
            occupancy=occ,
            capture_multiplier=1.0,
            coefficients=coeffs,
        )

    extreme_trace = DD(np.array([50.0, 0.14], dtype=np.float64))
    with pytest.raises(UnsupportedCompensatedInterfaceDomain):
        evaluate_local_carrier_pair(
            state_density=state_d,
            trace_potential=extreme_trace,
            bulk_density=bulk_d,
            bulk_potential=bulk_v,
            occupancy=occ,
            capture_multiplier=1.0,
            coefficients=coeffs,
        )


def test_state_authority_reference_coordinates_and_codecs() -> None:
    support = Support("nodes", "cell", (2,))
    iface = Support("iface", "global", (1,))
    layout = Layout(
        supports=(support, iface),
        variables=(
            VariableSpec("phi", "electrostatics", "nodes", (2,), VOLT),
            VariableSpec("n", "transport", "nodes", (2,), PARTICLE / VOLUME),
            VariableSpec("f_t", "interface", "iface", (1,), ONE),
        ),
        equations=(
            EquationSpec("eq_phi", "electrostatics", "nodes", (2,), ONE),
            EquationSpec("eq_n", "transport", "nodes", (2,), ONE),
            EquationSpec("eq_f", "interface", "iface", (1,), ONE),
        ),
    )
    coords = ReferenceIncrementCoordinates.from_float_anchor(
        layout,
        {
            "phi": np.array([0.1, 0.2], dtype=np.float64),
            "n": np.array([1.0e21, 2.0e21], dtype=np.float64),
            "f_t": np.array([0.4], dtype=np.float64),
        },
        multiplicative_variables=("n",),
    )
    p0 = coords.point(np.zeros(layout.size, dtype=np.float64), time=0.0, inputs=[0.005])
    assert p0.state.authority is not None
    assert _accepts_authority(p0.state.authority) is p0.state.authority
    assert isinstance(p0.state.authority, PairStateAuthority)

    dy = np.array([1.0e-18, -2.0e-18, 3.0e-18, -1.0e-18, 5.0e-18], dtype=np.float64)
    p1, inc = coords.advance(p0, dy, time=1.0e-9, inputs=[0.005])
    # layout sorts variables alphabetically: ('f_t', 'n', 'phi') -> offsets: f_t=[0:1], n=[1:3], phi=[3:5]
    f_delta = inc.field("f_t")
    n_delta = inc.field("n")
    phi_delta = inc.field("phi")
    assert isinstance(f_delta, DDArray) and isinstance(n_delta, DDArray) and isinstance(phi_delta, DDArray)
    assert float(f_delta.values[0]) == pytest.approx(1.0e-18, rel=1.0e-12)
    assert float(phi_delta.values[0]) == pytest.approx(-1.0e-18, rel=1.0e-12)
    assert float(phi_delta.values[1]) == pytest.approx(5.0e-18, rel=1.0e-12)

    ec_delta = inc.electrochemical_delta(
        p0,
        p1,
        "n",
        "phi",
        np.array([[0, 1]], dtype=np.intp),
        0.02585,
        potential_sign=-1,
        arithmetic=CompensatedArithmetic(),
    )
    # dy on 'n' is [-2e-18, 3e-18], dy on 'phi' is [-1e-18, 5e-18]
    expected_ec = (3.0e-18 - (-2.0e-18)) + (-1.0) * (5.0e-18 - (-1.0e-18)) / 0.02585
    assert isinstance(ec_delta, DD)
    assert float(ec_delta[0]) == pytest.approx(expected_ec, rel=1.0e-11)

    rebased_map, p_rebased = coords.rebase(p1)
    assert np.all(p_rebased.y == 0.0)
    assert isinstance(p_rebased.state.authority, PairStateAuthority)
    assert isinstance(p1.state.authority, PairStateAuthority)
    assert np.all(p_rebased.state.authority.dd_field("phi").values == p1.state.authority.dd_field("phi").values)

    codec = AffineInputLiftCodec()
    encoded = codec.encode_point(p1)
    decoded_map, p_decoded = codec.decode_with_map(
        encoded, layout=layout, multiplicative_variables=("n",)
    )
    assert decoded_map.layout.identity == layout.identity
    assert isinstance(p_decoded.state.authority, PairStateAuthority)
    assert np.all(p_decoded.state.authority.dd_field("n").values == p1.state.authority.dd_field("n").values)

    tampered = json.loads(json.dumps(encoded))
    tampered["snapshot"]["represented_fields"]["phi"]["lo"][0] = 0.0
    with pytest.raises(InputLiftCheckpointError):
        codec.decode_point(tampered, layout=layout)


def test_decode_historical_r1_pair_state_and_tamper_rejection() -> None:
    record = json.loads(HISTORICAL_PAIR_FIXTURE.read_text(encoding="utf-8"))
    decoded = decode_historical_r1_pair_state(record)
    assert len(decoded) == 17
    assert "phi_V" in decoded and "n_m3" in decoded and "occupancy" in decoded
    assert isinstance(decoded["phi_V"], DD)

    tampered = json.loads(json.dumps(record))
    tampered["state"]["precision_phi_V_lo"][0] += 1.0e-25
    with pytest.raises(InputLiftCheckpointError):
        decode_historical_r1_pair_state(tampered)

    with pytest.raises(ContractError):
        PairStateAuthority(layout_identity="x", values={}, anchor_reference="")


def test_research_boundary_and_registry_hygiene() -> None:
    public_api = json.loads(
        (REPO_ROOT / "src" / "solarlab" / "reproducibility" / "PublicAPI.json").read_text(
            encoding="utf-8"
        )
    )
    allowed_public_imports = set(public_api["public_imports"])
    forbidden_roots = {"perovskite_sim", "solarlab_server", "backend"}

    py_files = sorted(RESEARCH_SRC.rglob("*.py"))
    assert py_files, "expected solarlab_research python modules"

    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in forbidden_roots, f"{path} imports forbidden {alias.name}"
                    if root == "solarlab":
                        assert alias.name in allowed_public_imports, (
                            f"{path} imports {alias.name} missing from PublicAPI.json"
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                assert root not in forbidden_roots, f"{path} imports forbidden {node.module}"
                if root == "solarlab":
                    for alias in node.names:
                        assert not alias.name.startswith("_"), (
                            f"{path} imports private symbol {node.module}.{alias.name}"
                        )
                        fq = f"{node.module}.{alias.name}"
                        assert (
                            node.module in allowed_public_imports or fq in allowed_public_imports
                        ), f"{path} import {fq} not declared in PublicAPI.json"
                        # Precision injection must NOT register PhysicsModel / ModelDefinition.
                        if path.is_relative_to(RESEARCH_SRC / "precision"):
                            assert alias.name not in {"ModelDefinition", "ModelRegistry"}, (
                                f"{path} must not register physics models ({alias.name})"
                            )
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "solarlab":
                    assert not node.attr.startswith("_"), f"{path} accesses private solarlab.{node.attr}"
