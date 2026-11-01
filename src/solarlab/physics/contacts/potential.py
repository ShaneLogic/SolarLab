"""Poisson contact-potential sources, independent of carrier exchange (M class).

The signed contact potential ``V_bi_bc = phi(right) - phi(left)`` comes from
one explicitly selected source:

* ``semiconductor_work_function`` -- ``W_left - W_right`` of the two contact
  faces from :func:`~solarlab.physics.contacts.reservoirs.semiconductor_work_function`;
* ``metal_work_function`` -- ``W_left - W_right`` of two declared metals;
* ``legacy_manual`` -- the configured magnitude, oriented by the historical
  band-derived endpoint estimate;
* ``None`` -- the compatibility sentinel of old configurations: with
  ``flat_band_contacts`` it uses the historical estimate itself, otherwise it
  behaves as ``legacy_manual``.

The historical estimate (``legacy_band_contact_potential``) evaluates each
contact face's Fermi level with the configured scalar ``n_i`` and the 300 K
thermal voltage, and returns the manual value when every electrical layer has
``chi = Eg = 0``. Carrier exchange (Dirichlet, blocking, Robin) never selects
the potential source. Positive forward bias reduces ``|V_bi_bc|`` for either
junction orientation through ``junction_polarity``. All values are
bit-identical to the legacy engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

from solarlab.physics.contacts.reservoirs import ContactMaterial, semiconductor_work_function
from solarlab.physics.statistics.thermal import REFERENCE_THERMAL_VOLTAGE_V

__all__ = [
    "BUILT_IN_POTENTIAL_MODES", "BuiltInPotentialSources", "ContactPotentialBoundary",
    "LEGACY_MANUAL", "LegacyBandContact", "METAL_WORK_FUNCTION",
    "SEMICONDUCTOR_WORK_FUNCTION", "junction_polarity", "legacy_band_contact_potential",
    "legacy_band_fermi_level", "resolve_built_in_potential_mode",
]

SEMICONDUCTOR_WORK_FUNCTION = "semiconductor_work_function"
METAL_WORK_FUNCTION = "metal_work_function"
LEGACY_MANUAL = "legacy_manual"
BUILT_IN_POTENTIAL_MODES = (SEMICONDUCTOR_WORK_FUNCTION, METAL_WORK_FUNCTION, LEGACY_MANUAL)


@dataclass(frozen=True, slots=True)
class LegacyBandContact:
    """Inputs of the historical endpoint Fermi-level estimate at one contact face."""

    electron_affinity_eV: float
    band_gap_eV: float
    donor_density_m3: float
    acceptor_density_m3: float
    intrinsic_density_m3: float


def legacy_band_fermi_level(contact: LegacyBandContact) -> float:
    """Historical Fermi level below vacuum [eV] from ``E_i = chi + Eg/2``.

    Uses the two-branch majority density and the 300 K thermal voltage.
    """
    ni = contact.intrinsic_density_m3
    e_i = contact.electron_affinity_eV + contact.band_gap_eV / 2.0
    net = 0.5 * (contact.donor_density_m3 - contact.acceptor_density_m3)
    disc = math.sqrt(net * net + ni * ni)
    if contact.donor_density_m3 > contact.acceptor_density_m3:
        n = net + disc
        return e_i - REFERENCE_THERMAL_VOLTAGE_V * math.log(n / ni)
    if contact.acceptor_density_m3 > contact.donor_density_m3:
        hole = -net + disc
        return e_i + REFERENCE_THERMAL_VOLTAGE_V * math.log(hole / ni)
    return e_i


def legacy_band_contact_potential(
    left: LegacyBandContact,
    right: LegacyBandContact,
    *,
    electrical_layer_bands: Sequence[tuple[float, float]],
    manual_V_bi: float,
) -> float:
    """Historical signed estimate ``E_F,left - E_F,right`` [V].

    ``electrical_layer_bands`` holds ``(chi, Eg)`` of every electrical layer
    that has material parameters; when all are zero the manual value is
    returned unchanged (old configurations without band data).
    """
    if not electrical_layer_bands:
        return manual_V_bi
    if all(chi == 0.0 and gap == 0.0 for chi, gap in electrical_layer_bands):
        return manual_V_bi
    return legacy_band_fermi_level(left) - legacy_band_fermi_level(right)


def resolve_built_in_potential_mode(mode: str | None, *, flat_band_contacts: bool) -> str:
    """Public mode, including the compatibility sentinel's inference."""
    if mode is not None:
        if mode not in BUILT_IN_POTENTIAL_MODES:
            raise ValueError(f"unknown built-in-potential mode {mode!r}")
        return mode
    if flat_band_contacts:
        return SEMICONDUCTOR_WORK_FUNCTION
    return LEGACY_MANUAL


@dataclass(frozen=True)
class BuiltInPotentialSources:
    """Every input a contact-potential source may read; each is evaluated lazily.

    ``left_band``/``right_band`` feed the historical estimate,
    ``left_material``/``right_material`` the semiconductor work functions and
    ``work_function_*_eV`` the metal mode. A source that is not selected is
    never evaluated, exactly as in the historical resolution.
    """

    mode: str | None
    manual_V_bi: float
    flat_band_contacts: bool = False
    left_band: LegacyBandContact | None = None
    right_band: LegacyBandContact | None = None
    electrical_layer_bands: tuple[tuple[float, float], ...] = ()
    left_material: ContactMaterial | None = None
    right_material: ContactMaterial | None = None
    work_function_left_eV: float | None = None
    work_function_right_eV: float | None = None

    def __post_init__(self) -> None:
        if self.mode is not None and self.mode not in BUILT_IN_POTENTIAL_MODES:
            raise ValueError(f"unknown built-in-potential mode {self.mode!r}")

    @property
    def resolved_mode(self) -> str:
        return resolve_built_in_potential_mode(
            self.mode, flat_band_contacts=self.flat_band_contacts)

    def historical_estimate(self) -> float:
        """The deprecated band-derived endpoint estimate (``compute_V_bi``)."""
        if not self.electrical_layer_bands:
            return self.manual_V_bi
        if all(chi == 0.0 and gap == 0.0 for chi, gap in self.electrical_layer_bands):
            return self.manual_V_bi
        if self.left_band is None or self.right_band is None:
            raise ValueError("the historical estimate needs both contact faces")
        return legacy_band_contact_potential(
            self.left_band, self.right_band,
            electrical_layer_bands=self.electrical_layer_bands,
            manual_V_bi=self.manual_V_bi,
        )

    def semiconductor_potential(self) -> float:
        """``W_left - W_right`` of the two semiconductor contact faces."""
        if self.left_material is None or self.right_material is None:
            raise ValueError(
                "semiconductor_work_function requires material parameters on both "
                "electrical contact layers"
            )
        W_left = semiconductor_work_function(self.left_material)
        W_right = semiconductor_work_function(self.right_material)
        return W_left - W_right

    def metal_potential(self) -> float:
        """Signed ``W_left - W_right`` from explicit metal contacts."""
        if self.work_function_left_eV is None or self.work_function_right_eV is None:
            raise ValueError("metal_work_function requires both contact work functions")
        return float(self.work_function_left_eV - self.work_function_right_eV)

    def _oriented_manual(self) -> float:
        orientation = -1.0 if self.historical_estimate() < 0.0 else 1.0
        return orientation * abs(float(self.manual_V_bi))

    def poisson_potential(self) -> float:
        """Signed ``V_bi_bc`` for the Poisson boundary."""
        if self.mode is None:
            if self.flat_band_contacts:
                return self.historical_estimate()
            return self._oriented_manual()
        if self.mode == LEGACY_MANUAL:
            return self._oriented_manual()
        if self.mode == SEMICONDUCTOR_WORK_FUNCTION:
            return self.semiconductor_potential()
        return self.metal_potential()

    def operating_potential(self) -> float:
        """Signed potential for operating defaults; the sentinel keeps the estimate."""
        if self.mode is None:
            return self.historical_estimate()
        return self.poisson_potential()


def junction_polarity(V_bi_bc: float) -> float:
    """``-1`` for an n-contact-left orientation (negative ``V_bi_bc``), else ``+1``."""
    return -1.0 if V_bi_bc < 0.0 else 1.0


@dataclass(frozen=True, slots=True)
class ContactPotentialBoundary:
    """Dirichlet potentials of the two contacts for an applied bias."""

    V_bi_bc: float
    left_V: float = 0.0

    @property
    def junction_polarity(self) -> float:
        return junction_polarity(self.V_bi_bc)

    def right(self, V_app: float) -> float:
        """Right contact potential; positive forward bias reduces ``|V_bi_bc|``."""
        return float(self.V_bi_bc - self.junction_polarity * float(V_app))

    def values(self, V_app: float) -> tuple[float, float]:
        return self.left_V, self.right(V_app)
