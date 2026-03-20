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
