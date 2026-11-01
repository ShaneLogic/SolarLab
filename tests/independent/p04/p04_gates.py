"""Frozen GateSpec registry for the R3 independent P04-01 / P04-07 oracles.

Every acceptance number used by ``tests/independent/p04`` is declared here,
before any W3a/W3b candidate output exists, with the fields V1 Section 14.3
requires (ID, quantity/unit, scope, reference identity, norm, absolute and
relative limits, fixed scale, mask, zero/weak-signal rule, reference error,
independent source, version).  A gate with a missing field is invalid
(``test_p04_gates.py`` enforces this).  Tolerances are derived from IEEE-754
binary64 rounding analyses or exact identities -- never from the candidate.
Changing a value is a prospective GateSpec revision (new ``version``) that
needs independent review before the affected run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

EPS = float(np.finfo(float).eps)  # 2.220446049250313e-16


@dataclass(frozen=True)
class OracleGate:
    gate_id: str
    task: str
    quantity: str
    unit: str
    scope: str
    reference: str
    norm: str
    abs_limit: float
    rel_limit: float
    scale: str
    mask: str
    zero_rule: str
    reference_error: str
    source: str
    version: str = "V1"


_G = OracleGate
GATES: tuple[OracleGate, ...] = (
    # ------------------------------------------------------------- P04-01
    _G("P0401-BERN-VAL", "P04-01", "Bernoulli B(x)", "1", "binary64 |x| <= 700",
       "p04_oracle_bernoulli.bernoulli_hp (100 digits)", "max relative", 0.0, 4 * EPS,
       "|B_ref(x)|", "finite x", "B > 0 on the scope; no zero branch",
       "< 1e-90 relative (decimal, exact input)",
       "definition B=x/(e^x-1); Scharfetter-Gummel 1969"),
    _G("P0401-BERN-TAIL", "P04-01", "Bernoulli B(x) far tail", "1",
       "700 < x <= 1e5 (B < 7e-302) and -1e5 <= x < -700",
       "bernoulli_hp", "max abs (x>700); max relative (x<-700)", 1e-300, 4 * EPS,
       "|B_ref| for x<-700", "finite x",
       "x>700: 0 <= B <= 1e-300 accepted (historical zero branch, deviation LD-01)",
       "< 1e-90 relative", "definition; physically inert (density ratio e^700 unreachable)"),
    _G("P0401-BERN-IDENT", "P04-01", "B(-x) - B(x) - x", "1", "0 < x <= 700",
       "exact identity (I1)", "max abs, exact rational evaluation of the float outputs",
       0.0, 2.0, "ulp(B(-x)) (rel_limit counts ulps)", "finite x",
       "large branch must be recovered by stable addition; B(50) must not vanish",
       "exact", "identity B(-x)=x+B(x); V1 P04-01 step 3"),
    _G("P0401-DBERN-VAL", "P04-01", "Bernoulli derivative B'(x)", "1", "|x| <= 700",
       "bernoulli_derivative_hp", "max relative", 1e-300, 1e-11, "|B'_ref(x)|",
       "finite x", "B'(0) = -1/2 exactly; abs floor only where |B'| < 1e-300",
       "< 1e-90 relative",
       "definition; legacy measured 3.3e-12 (2 eps/|x| cancellation at the 1e-4 switch)"),
    _G("P0401-DBERN-IDENT", "P04-01", "B'(-x) + B'(x) + 1", "1", "0 < x <= 700",
       "exact identity (I2)", "max abs", 1e-11, 0.0, "1 (dimensionless)", "finite x",
       "none", "exact", "derivative of (I1)"),
    _G("P0401-DBERN-SHAPE", "P04-01", "B' range and monotonicity", "1", "|x| <= 1e5",
       "convexity of B", "pointwise", 0.0, 0.0, "n/a", "sorted finite grid",
       "-1 <= B' <= 0, non-decreasing in x, B'(0) == -0.5", "exact", "B'' > 0"),
    _G("P0401-SG-EQ", "P04-01", "SG particle flux at Boltzmann equilibrium", "m^-2 s^-1",
       "1D faces, |xi| <= 200, electrons and holes",
       "c = c0 exp(-z psi/V_T) (exact zero flux)",
       "per face |F| / ((D/h)(B(z xi)c_i + B(-z xi)c_{i+1}))", 0.0, 8.0,
       "8 eps (16 + |psi_i/V_T| + |psi_{i+1}/V_T| + |xi|) (rel_limit is the factor)",
       "all faces", "weak signal: referred to the SG leg sum, never to |F|",
       "rounding bound derived in p04_oracle_bernoulli.sg_equilibrium_bound",
       "B(-y)/B(y) = e^y"),
    _G("P0401-SG-VAL", "P04-01", "SG particle flux value", "m^-2 s^-1",
       "1D faces, |xi| <= 200", "sg_particle_flux_hp (100 digits)",
       "per face |F - F_ref| / leg sum", 0.0, 64 * EPS, "(D/h)(B(z xi)c_i + B(-z xi)c_{i+1})",
       "all faces", "leg-sum scale (cancellation-aware)", "< 1e-90",
       "equation (SG)"),
    _G("P0401-SG-JAC", "P04-01", "SG flux derivatives d/dc_i, d/dc_{i+1}, d/dpsi_i, d/dpsi_{i+1}",
       "m^-2 s^-1 per unit", "1D faces, |xi| <= 200",
       "analytic (SG) derivative with bernoulli_hp / bernoulli_derivative_hp",
       "per entry relative to |entry| + leg-derivative scale", 0.0, 1e-10,
       "|d_ref| + (D/h)(|B'(z xi)|c_i + |B'(-z xi)|c_{i+1})/V_T for psi entries",
       "all faces", "golden equality never substitutes for this check", "< 1e-80",
       "differentiation of (SG); validated against decimal central differences"),
    _G("P0401-FACE-MEAN", "P04-01", "harmonic face mean (eps, D)", "as input",
       "positive and zero nodal values", "2ab/(a+b) = exact series of two half cells",
       "relative", 0.0, 4 * EPS, "|exact|", "all faces",
       "a = 0 or b = 0 must give exactly 0 (no NaN)", "exact rational", "series resistance"),
    _G("P0401-POI-EXACT", "P04-01", "Poisson potential, exactly representable cases", "V",
       "piecewise linear (series capacitor, sheet charge) and single-layer uniform charge, "
       "non-uniform meshes", "p04_oracle_electrostatics.layered_constant_charge_solution",
       "max abs over nodes", 1e-10, 0.0, "max(1 V, max|phi_exact|) (abs_limit multiplies it)",
       "all nodes", "gauge fixed by the two Dirichlet values", "exact closed form",
       "secant slope of a quadratic = midpoint derivative; constant D"),
    _G("P0401-POI-CAP", "P04-01", "series capacitance from discrete faces", "F m^-2",
       "charge-free layered dielectric", "1/sum d_k/(eps0 eps_k)", "relative", 0.0, 1e-13,
       "C_exact", "all faces", "none", "fsum, < 1e-15",
       "series capacitor"),
    _G("P0401-POI-GEOM", "P04-01", "historical 1D effective-plane shift", "V",
       "legacy_1d_integer_endpoint_v1 channel", "exact solution of the structure with planes at "
       "x_I - h_{I-1}/2", "max abs over nodes", 1e-10, 0.0, "max(1 V, max|phi|)",
       "all nodes", "the unshifted analytic must differ by >= 100 x the limit (discrimination)",
       "exact", "right-owner node rule + harmonic face mean (solver/mol.py, physics/poisson.py)"),
    _G("P0401-POI-MMS", "P04-01", "observed convergence order (max nodal error)", "1",
       "manufactured D = F0 cos(pi x/L) + F1, 3 layers, eps contrast 3/24.1/10, non-uniform "
       "nested refinement x2", "ManufacturedPoisson.potential", "log2 ratio of max errors",
       0.0, 1.8, "finest refinement pair (rel_limit = required minimum order)",
       "all nodes", "historical channel with eps contrast is expected first order: "
       "0.8 <= p <= 1.5 recorded as geometry difference, not an algorithm failure",
       "closed form", "Roache MMS; standard FV truncation analysis"),
    _G("P0401-GAUSS-JUMP", "P04-01", "Gauss jump D+ - D- at an interface sheet charge", "C m^-2",
       "two-sided trace with a zero-volume interface node", "sigma", "relative", 0.0, 1e-12,
       "|sigma| + |D|", "interface faces", "sigma = 0 must give continuity", "exact",
       "div D = rho; V1 P04-01 step 4"),
    _G("P0401-GAUSS-SINK", "P04-01", "TermSink Gauss balance vs algebraic residual", "C m^-2",
       "N-class TermSink assembly, interior and contact rows", "oracle FV residual with the "
       "operator's own C and w; Gauss over the end cells", "per row |a - b| / row abs-sum",
       0.0, 8 * EPS, "sum of absolute terms of the row", "all rows",
       "contact rows are a physical D only for geometric contact cells",
       "exact", "added V1 at review R3-2 before the candidate was evaluated"),
    _G("P0401-MEAS", "P04-01", "total control-volume measure", "m (1D), m^2 (2D)",
       "each named geometry", "expected_total_measure closed forms", "abs of fsum difference",
       0.0, 1e-14, "L (1D) or L_x L_y (2D)", "all cells",
       "historical 1D total is L + (h_0 + h_last)/2 by definition", "exact",
       "geometry definitions in p04_oracle_geometry"),
    _G("P0401-LAYER-MEAS", "P04-01", "per-material measure (R1 two-sided)", "m",
       "r1_physical_boundary_two_sided_v1", "layer thickness", "relative", 0.0, 1e-14, "d_k",
       "cells of each material", "none", "exact", "physical_cell_faces definition"),
    _G("P0401-SUBSTRATE", "P04-01", "substrate-prefix interface alignment", "index, m",
       "stacks with 0..2 substrate layers", "electrical_alignment", "exact index; relative position",
       0.0, 1e-15, "position", "electrical interfaces",
       "no electrical interface may map to a substrate boundary", "exact",
       "models/device.py electrical_interfaces contract (E10.1 regression)"),
    _G("P0401-PORT", "P04-01", "port outward flux and conservation", "m^-2 s^-1 x area",
       "uniform face flux", "outward normals -x (left), +x (right)", "exact", 0.0, 0.0,
       "n/a", "ports", "left + right outward = 0 for divergence-free flux", "exact",
       "V1 Section 15.1 TerminalPort normal/area"),
    # ------------------------------------------------------------- P04-07
    _G("P0407-REC-EQ", "P04-07", "net SRH / radiative / Auger rate at equilibrium", "m^-3 s^-1",
       "n p = ni^2 (exactly representable and rounded)", "0 exactly",
       "|R| / gross", 0.0, 8 * EPS, "gross = (n p + ni^2) x channel coefficient",
       "all nodes", "weak signal: gross scale; exactly representable inputs give 0.0 on the "
       "historical formula (bitwise M expectation)", "exact rational",
       "detailed balance"),
    _G("P0407-REC-VAL", "P04-07", "bulk rate value", "m^-3 s^-1",
       "MB lifetime SRH, B, Auger, 1e-30..1e30 m^-3", "p04_oracle_recombination (Fraction)",
       "|R - R_ref| / (|R_ref| + gross)", 0.0, 8 * EPS, "|R_ref| + gross", "all inputs",
       "gross scale near equilibrium", "exact", "Shockley-Read-Hall; standard B, Auger laws"),
    _G("P0407-REC-DER", "P04-07", "bulk rate derivatives dR/dn, dR/dp", "s^-1",
       "as REC-VAL", "exact rational derivative", "|d - d_ref| / (|d_ref| + gross_d)",
       0.0, 32 * EPS, "|d_ref| + gross_d", "all inputs",
       "derivatives are checked independently of any golden", "exact", "analytic differentiation"),
    _G("P0407-REC-LIMIT", "P04-07", "low/high injection limits", "1", "p-type, Delta/p0 <= 1e-6; "
       "n = p >= 1e6 n1", "exact closed-form ratio", "|ratio - 1|", 8 * EPS, 2.0,
       "closed-form asymptotic error, e.g. (Delta + p1 + (tau_p/tau_n)(n0 + Delta + n1))/p0 "
       "(rel_limit is the factor on it)", "n/a", "none", "exact",
       "R -> Delta/tau_n, Delta/(tau_n+tau_p), B p0 Delta, B Delta^2, C_p p0^2 Delta, (C_n+C_p) Delta^3"),
    _G("P0407-REC-ZERO", "P04-07", "disabled channel identity", "m^-3 s^-1",
       "B = 0, C_n = C_p = 0, absent SRH instance", "0.0", "bitwise", 0.0, 0.0, "n/a",
       "all nodes", "disabled channel adds exactly 0.0 to rate and both derivatives; NaN forbidden",
       "exact", "V1 P04-07 acceptance 'zero channel strict identity'"),
    _G("P0407-REC-NODOUBLE", "P04-07", "explicit defect replaces lifetime SRH", "m^-3 s^-1",
       "explicit-defect nodes", "explicit species rate only", "as REC-VAL", 0.0, 8 * EPS,
       "|R_ref| + gross", "explicit nodes; lifetime nodes unchanged",
       "a 2x rate (double count) is a failure regardless of tolerance", "exact",
       "V1 P04-07 step 1"),
    _G("P0407-TRAP", "P04-07", "trap density profile and tau(x)", "m^-3, s",
       "exponential/gaussian, edge both/left/right", "p04_oracle_recombination.*_ref",
       "relative", 0.0, 16 * EPS, "|N_t|", "layer nodes",
       "L_d <= 0 or sigma <= 0 -> uniform bulk exactly; tau floor at N_bulk",
       "closed form", "physics/traps.py documented semantics (F-05 normalisation)"),
    _G("P0407-BL-TELE", "P04-07", "discrete absorbed photons sum_i G_i w_i", "m^-2 s^-1",
       "any mesh", "Phi (1 - exp(-tau_d,total)) of the declared discrete optical depth",
       "relative", 0.0, 1e-14, "Phi (1 - exp(-tau))", "all nodes", "Phi = 0 -> exactly 0",
       "exact", "telescoping; legacy RTOL_FP provenance (tests/unit/physics/test_generation.py)"),
    _G("P0407-BL-BOUND", "P04-07", "photon ceiling", "m^-2 s^-1", "any mesh, any alpha >= 0",
       "Phi", "strict inequality sum G w <= Phi", 0.0, 0.0, "n/a", "all nodes",
       "no tolerance", "exact", "photon conservation"),
    _G("P0407-BL-LAYER", "P04-07", "per-layer absorbed photons (monochromatic BL)", "m^-2 s^-1",
       "control volumes aligned with material planes", "equation (BL)", "relative", 0.0, 1e-13,
       "A_k", "cells of each layer", "non-absorbing layer: exactly 0", "closed form",
       "Beer-Lambert"),
    _G("P0407-BL-DIR", "P04-07", "incidence direction", "m^-3 s^-1", "same stack lit from x=L",
       "mirrored stack lit from x=0", "relative", 0.0, 1e-14, "max G", "all nodes", "none",
       "exact symmetry", "V1 P04-07 step 3"),
    _G("P0407-TMM-RTA", "P04-07", "R + T + sum_k A_k - 1", "1", "coherent stack, normal incidence",
       "independent TMM (Poynting and closed-form |E|^2 integral)", "max abs over wavelength",
       1e-12, 0.0, "1", "all wavelengths", "lossless stack: sum A_k = 0 exactly up to 1e-15",
       "< 1e-14", "Born & Wolf 1.6; Byrnes arXiv:1603.02720"),
    _G("P0407-TMM-CLOSED", "P04-07", "closed-form R, T", "1",
       "Fresnel interface, Airy slab, quarter-wave AR, half-wave absentee", "closed forms",
       "max abs", 1e-14, 0.0, "1", "all wavelengths", "QWOT R must be <= 1e-28", "exact",
       "thin-film optics"),
    _G("P0407-TMM-BUDGET", "P04-07", "device photon budget", "m^-2 s^-1",
       "spectral TMM with electrical + parasitic layers", "incident = R + T + A_elec + A_parasitic",
       "relative", 0.0, 1e-12, "incident photon flux", "all layers",
       "parasitic (substrate/junction) absorption never enters G", "exact",
       "V1 P04-07 step 3"),
    _G("P0407-UNITS", "P04-07", "spectral unit conversion", "m^-2 s^-1 m^-1",
       "W m^-2 nm^-1 -> photons per m", "Phi = E 1e9 lambda/(h c)", "relative", 0.0, 4 * EPS,
       "|Phi|", "all wavelengths", "per-nm vs per-m differs by exactly 1e9", "exact",
       "SI exact h, c; data/am15g.csv header"),
    _G("P0407-TMM-CELL", "P04-07", "TMM per-cell absorbed photons", "m^-2 s^-1",
       "dual cells, any mesh", "independent closed-form cumulative absorptance x spectral "
       "trapezoid", "relative", 0.0, 1e-11, "max cell value", "all cells",
       "sum telescopes to the electrical absorbed budget", "< 1e-13",
       "solver/mol.py _compute_tmm_generation semantics"),
    # ------------------------------------------------------------- golden
    _G("P04-GOLDEN-M", "P04-01/P04-07", "historical arrays", "as recorded",
       "M-class migration of a recorded case", "fixtures/P04LegacyGoldenV1.json (legacy "
       "b1f953c)", "bitwise (float.hex equality)", 0.0, 0.0, "n/a", "all entries",
       "-0.0 vs 0.0 counts as a difference only if signbit differs", "exact",
       "V1 Section 1.2 M row"),
    _G("P04-GOLDEN-N", "P04-01/P04-07", "historical arrays", "as recorded",
       "N-class replacement of a recorded case", "fixtures/P04LegacyGoldenV1.json",
       "per entry |new-old| <= abs + rel |old|", 0.0, 1e-12, "|old|", "all entries",
       "golden agreement alone never passes a case: every oracle gate must pass too",
       "legacy is a historical reference, not truth", "V1 Section 1.2 N row"),
)


def gate(gate_id: str) -> OracleGate:
    for item in GATES:
        if item.gate_id == gate_id:
            return item
    raise KeyError(gate_id)


def as_records() -> list[dict[str, object]]:
    return [asdict(item) for item in GATES]


def missing_fields(item: OracleGate) -> list[str]:
    out = []
    for f in fields(OracleGate):
        value = getattr(item, f.name)
        if isinstance(value, str) and not value.strip():
            out.append(f.name)
        if isinstance(value, float) and not np.isfinite(value):
            out.append(f.name)
    return out
