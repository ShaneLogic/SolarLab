"""Unit tests for perovskite_sim.models.mode."""
from __future__ import annotations

import pytest

from perovskite_sim.models.mode import (
    SimulationMode,
    LEGACY,
    FAST,
    FULL,
    resolve_mode,
)


class TestPresets:
    def test_legacy_disables_all_upgrades(self):
        assert LEGACY.name == "legacy"
        assert LEGACY.use_thermionic_emission is False
        assert LEGACY.use_tmm_optics is False
        assert LEGACY.use_dual_ions is False
        assert LEGACY.use_trap_profile is False
        assert LEGACY.use_temperature_scaling is False
        assert LEGACY.use_photon_recycling is False
        assert LEGACY.use_field_dependent_mobility is False
        assert LEGACY.use_selective_contacts is False

    def test_full_enables_all_upgrades(self):
        assert FULL.name == "full"
        assert FULL.use_thermionic_emission is True
        assert FULL.use_tmm_optics is True
        assert FULL.use_dual_ions is True
        assert FULL.use_trap_profile is True
        assert FULL.use_temperature_scaling is True
        assert FULL.use_photon_recycling is True
        assert FULL.use_field_dependent_mobility is True
        assert FULL.use_selective_contacts is True

    def test_fast_enables_build_once_upgrades(self):
        """FAST turns on every Phase 1/2/3.1 (build-once) upgrade."""
        assert FAST.name == "fast"
        assert FAST.use_thermionic_emission is True
        assert FAST.use_tmm_optics is True
        assert FAST.use_dual_ions is True
        assert FAST.use_trap_profile is True
        assert FAST.use_temperature_scaling is True
        assert FAST.use_photon_recycling is True

    def test_fast_disables_per_rhs_upgrades(self):
        """FAST keeps field-mobility / selective contacts off because they
        break the MaterialArrays build-once invariant and run per-RHS."""
        assert FAST.use_field_dependent_mobility is False
        assert FAST.use_selective_contacts is False

    def test_fast_is_strictly_between_legacy_and_full(self):
        """Every FAST flag is >= the LEGACY flag and <= the FULL flag."""
        for field in (
            "use_thermionic_emission",
            "use_tmm_optics",
            "use_dual_ions",
            "use_trap_profile",
            "use_temperature_scaling",
            "use_photon_recycling",
            "use_field_dependent_mobility",
            "use_selective_contacts",
        ):
            legacy_val = bool(getattr(LEGACY, field))
            fast_val = bool(getattr(FAST, field))
            full_val = bool(getattr(FULL, field))
            # legacy <= fast <= full, treating False < True.
            assert legacy_val <= fast_val <= full_val, (
                f"Tier monotonicity broken for flag {field}: "
                f"legacy={legacy_val}, fast={fast_val}, full={full_val}"
            )

    def test_simulation_mode_is_frozen(self):
        with pytest.raises(Exception):
            LEGACY.use_thermionic_emission = True  # type: ignore[misc]


class TestResolveMode:
    def test_none_resolves_to_full(self):
        assert resolve_mode(None) is FULL

    @pytest.mark.parametrize(
        "name,expected",
        [("legacy", LEGACY), ("fast", FAST), ("full", FULL)],
    )
    def test_string_resolves_to_preset(self, name, expected):
        assert resolve_mode(name) is expected

    def test_case_insensitive(self):
        assert resolve_mode("LEGACY") is LEGACY
        assert resolve_mode("Full") is FULL

    def test_passthrough_simulation_mode(self):
        custom = SimulationMode(
            name="full",
            use_thermionic_emission=False,
            use_tmm_optics=True,
            use_dual_ions=False,
            use_trap_profile=True,
            use_temperature_scaling=True,
        )
        assert resolve_mode(custom) is custom

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown simulation mode"):
            resolve_mode("turbo")

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            resolve_mode(42)  # type: ignore[arg-type]
