from perovskite_sim.models.config_loader import load_device_from_yaml


def test_nip_loads_three_layers():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    assert len(stack.layers) == 3


def test_pin_loads_three_layers():
    stack = load_device_from_yaml("configs/pin_MAPbI3.yaml")
    assert len(stack.layers) == 3


def test_absorber_has_ions():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    absorber = next(l for l in stack.layers if l.role == "absorber")
    assert absorber.params.D_ion > 0


def test_incoherent_field_loaded_from_yaml(tmp_path):
    """The `incoherent: true` YAML field must land on MaterialParams.incoherent."""
    yaml_text = '''
device: {V_bi: 1.1, Phi: 2.5e21}
layers:
  - name: glass
    role: substrate
    thickness: 1.0e-3
    eps_r: 2.25
    mu_n: 0.0
    mu_p: 0.0
    ni: 0.0
    N_D: 0.0
    N_A: 0.0
    D_ion: 0.0
    P_lim: 1.0e30
    P0: 0.0
    tau_n: 1.0e-9
    tau_p: 1.0e-9
    n1: 1.0
    p1: 1.0
    B_rad: 0.0
    C_n: 0.0
    C_p: 0.0
    alpha: 0.0
    optical_material: glass
    incoherent: true
  - name: MAPbI3
    role: absorber
    thickness: 400e-9
    eps_r: 24.1
    mu_n: 2e-4
    mu_p: 2e-4
    ni: 3.2e13
    N_D: 0.0
    N_A: 0.0
    D_ion: 1e-16
    P_lim: 1.6e27
    P0: 1.6e24
    tau_n: 1e-6
    tau_p: 1e-6
    n1: 3.2e13
    p1: 3.2e13
    B_rad: 5e-22
    C_n: 1e-42
    C_p: 1e-42
    alpha: 1.3e7
'''
    p = tmp_path / "tmm.yaml"
    p.write_text(yaml_text)
    stack = load_device_from_yaml(str(p))
    assert stack.layers[0].params.incoherent is True
    assert stack.layers[0].params.optical_material == "glass"
    assert stack.layers[1].params.incoherent is False  # default
