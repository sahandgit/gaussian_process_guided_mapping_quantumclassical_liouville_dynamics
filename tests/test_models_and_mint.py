import numpy as np

from gp_mint_qcle.Models import TullyModel, TullyParams
from gp_mint_qcle.Mint import PBMEMIntDynamics, PBMEMIntParams, pack_z


def test_tully_dual_derivatives_match_finite_difference():
    model = TullyModel(TullyParams.defaults("dual"))
    R = np.array([-2.0, -0.3, 0.7, 2.5], dtype=float)
    eps = 1.0e-6
    dH_fd = (model.diabatic_potential(R + eps) - model.diabatic_potential(R - eps)) / (2 * eps)
    dH = model.d_diabatic_potential_dR(R)
    np.testing.assert_allclose(dH, dH_fd, rtol=2e-6, atol=2e-8)


def test_mint_short_step_energy_conservation():
    model = TullyModel(TullyParams.defaults("dual"))
    dyn = PBMEMIntDynamics(model=model, params=PBMEMIntParams(mass=2000.0, hbar=1.0))
    z0 = pack_z(-15.0, 40.0, np.array([np.sqrt(3.0), 0.0]), np.array([0.0, 1.0]))
    e0 = dyn.energy(z0)
    z1 = dyn.step(z0, 0.1)
    e1 = dyn.energy(z1)
    np.testing.assert_allclose(e1, e0, rtol=1e-8, atol=1e-8)
