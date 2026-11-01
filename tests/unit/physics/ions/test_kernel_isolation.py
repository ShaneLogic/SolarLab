"""Callable ion kernels and inherited semantics without legacy imports."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from solarlab.physics import ions


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.mark.parametrize("diffusion_only", [False, True])
def test_uniform_equilibrium_zero_diffusivity_and_shared_zero_partner(diffusion_only):
    x = np.array([0.0, 1e-9, 4e-9, 1e-8])
    density = np.full(4, 3e26)
    phi = np.zeros(4)
    kwargs = dict(steric_diffusion_only=diffusion_only, P_lim_node=1.6e27)
    for function in (ions.ion_continuity_rhs, ions.ion_continuity_rhs_neg):
        np.testing.assert_array_equal(function(x, phi, density, 1e-16, 0.025852, 1.6e27, **kwargs), 0.0)
        np.testing.assert_array_equal(
            function(x, np.array([0.0, 0.1, -0.1, 0.2]), density, 0.0, 0.025852, 1.6e27, **kwargs), 0.0,
        )
        own = function(x, phi, density, 1e-16, 0.025852, 1.6e27, **kwargs)
        shared = function(x, phi, density, 1e-16, 0.025852, 1.6e27, P_other_node=np.zeros(4), **kwargs)
        assert own.tobytes() == shared.tobytes()


def test_fresh_process_runs_all_kernel_families_with_legacy_and_tests_blocked():
    script = """
import importlib.abc, sys
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'perovskite_sim', 'backend', 'tests'}:
            raise RuntimeError('forbidden core dependency: ' + fullname)
sys.meta_path.insert(0, BlockImports())
import numpy as np
from solarlab.physics import ions
x=np.array([0., 1e-8, 3e-8]); dx=np.diff(x)
phi=np.array([0., .01, .03]); P=np.array([1e24, 2e24, 3e24])
for mode in (False, True):
    for sign in (-1., 1.):
        assert np.all(np.isfinite(ions.ion_face_flux(phi,P,dx,1e-16,.025852,1.6e27,steric_diffusion_only=mode,drift_sign=sign)))
        assert np.all(np.isfinite(ions.ion_face_flux_jacobian(phi,P,dx,1e-16,.025852,1.6e27,steric_diffusion_only=mode,drift_sign=sign).flux))
    for rhs in (ions.ion_continuity_rhs, ions.ion_continuity_rhs_neg):
        assert np.all(np.isfinite(rhs(x,phi,P,1e-16,.025852,1.6e27,steric_diffusion_only=mode,cell_widths_m=np.array([.5e-8,1.5e-8,1e-8]))))
assert np.isfinite(ions.ion_flux_steric(phi[:2],P[:2],dx[0],1e-16,.025852,1.6e27))
capacity=np.full(3,1e27); reference=.1*capacity
logit=np.full(3,np.log(.1)-np.log1p(-.1)); coordinate=np.array([-.1,0.,.1])
assert np.all(np.isfinite(ions.single_site_density(reference,capacity,logit,coordinate)))
assert np.all(np.isfinite(ions.shifted_logistic_density(reference,capacity,logit,coordinate,label='positive-ion')))
assert all(np.all(np.isfinite(v)) for v in ions.shared_site_densities(capacity,logit,logit,coordinate,-coordinate))
assert not any(name.split('.')[0] in {'perovskite_sim','backend','tests'} for name in sys.modules)
print('all ionic constitutive kernel families ran with legacy/tests imports blocked')
"""
    environment = {
        **os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[4],
        env=environment, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "kernel families ran" in result.stdout
