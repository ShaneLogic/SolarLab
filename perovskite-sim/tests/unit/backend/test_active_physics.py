from backend.main import _describe_active_physics
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams


def _minimal_stack(mode: str) -> DeviceStack:
    """A single trivial layer is enough to construct a DeviceStack that
    resolve_mode() can consume; we only care about the mode string.
    """
    params = MaterialParams(
        eps_r=10.0,
        mu_n=1e-4,
        mu_p=1e-4,
        D_ion=0.0,
        P_lim=1e26,
        P0=1e24,
        ni=1e15,
        tau_n=1e-9,
        tau_p=1e-9,
        n1=1e15,
        p1=1e15,
        B_rad=0.0,
        C_n=0.0,
        C_p=0.0,
        alpha=0.0,
        N_A=0.0,
        N_D=0.0,
    )
    layer = LayerSpec(name="L", role="absorber", thickness=1e-7, params=params)
    return DeviceStack(layers=(layer,), V_bi=1.0, Phi=1.4e21, T=300.0, mode=mode)


def test_full_mode_string_lists_all_upgrades():
    s = _describe_active_physics(_minimal_stack("full"))
    assert "FULL" in s
    assert "TE" in s
    assert "TMM" in s
    assert "dual ions" in s
    assert "T-scaling" in s
    # Phase 3.x extras only labelled when on — FULL has them all on.
    assert "photon recycling" in s
    assert "μ(E)" in s
    assert "Robin contacts" in s


def test_legacy_mode_string_lists_no_upgrades():
    s = _describe_active_physics(_minimal_stack("legacy"))
    assert "LEGACY" in s
    assert "Beer-Lambert" in s
    assert "uniform" in s
    assert "T=300K" in s
    # Phase 3.x extras should not be claimed at all.
    assert "photon recycling" not in s
    assert "μ(E)" not in s
    assert "Robin contacts" not in s


def test_fast_mode_string_shows_build_once_upgrades():
    """FAST tier enables every build-once upgrade (TE, TMM, dual ions,
    trap profile, T-scaling, PR) but keeps the per-RHS 3.2/3.3 hooks off.
    """
    s = _describe_active_physics(_minimal_stack("fast"))
    assert "FAST" in s
    # Build-once physics fragments all present.
    assert "TE" in s
    assert "TMM" in s
    assert "dual ions" in s
    assert "trap profile" in s
    assert "T-scaling" in s
    assert "photon recycling" in s
    # Per-RHS hooks stay off in FAST.
    assert "μ(E)" not in s
    assert "Robin contacts" not in s
