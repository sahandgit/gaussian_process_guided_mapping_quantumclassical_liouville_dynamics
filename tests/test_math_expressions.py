from __future__ import annotations

# --- UTF-8 console safety: prevent UnicodeEncodeError on Windows cp1252 ---
# The wrapped self-tests below print non-ASCII physics notation (α, ρ̂, Δ, →, ħ).
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
# --------------------------------------------------------------------------

r"""
test_math_expressions.py
========================

Independent verification of the *mathematical expressions* in the pipeline.

Each test pins one analytic formula against an independent reference — a central
finite difference, an eigen-decomposition, a Gauss--Hermite quadrature, JAX
forward-mode autodiff, or a conservation law that the discrete map must respect.
The intent is that a regression in any closed-form derivative, moment, or
integrator invariant fails loudly here.

Dependency tiers
----------------
* torch-free maths (Models, Mint, Monodromy, Operator FD engine, ProductMoments,
  qcle_grid_tully, Sampling) run everywhere NumPy/SciPy/JAX are present.
* torch-gated maths (the GP density surrogate derivatives, the SEO profile
  derivatives, and the GP-based coupling-term FD test) are guarded with
  ``pytest.importorskip("torch")`` so they run in the full research environment
  and skip cleanly where PyTorch is absent.

Run with:  pytest -v test_math_expressions.py
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _central_fd(func, x, eps):
    """Central finite difference of a vector->vector (or ->matrix) function."""
    return (np.asarray(func(x + eps)) - np.asarray(func(x - eps))) / (2.0 * eps)


def _gauss_hermite_moment(mu, var, power, n=24):
    """
    Independent E[X^power] for X ~ N(mu, var) via probabilists' Gauss--Hermite
    quadrature (exact for polynomials up to degree 2n-1).  Used to cross-check
    the closed-form Gaussian raw moments in ProductMoments.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(n)   # weight exp(-t^2/2)
    norm = np.sqrt(2.0 * np.pi)
    mu = np.atleast_1d(np.asarray(mu, float))
    var = np.atleast_1d(np.asarray(var, float))
    out = np.empty_like(mu)
    for i in range(mu.size):
        x = mu[i] + np.sqrt(var[i]) * nodes
        out[i] = np.sum(weights * x ** power) / norm
    return out


# ===========================================================================
# 1. Tully model potentials and their analytic derivatives  (Models.py)
# ===========================================================================

class TestTullyModelExpressions:
    def setup_method(self):
        from Models import TullyModel, TullyParams
        self.m = TullyModel(TullyParams.defaults("dual"))
        self.R = np.linspace(-4.0, 4.0, 17)

    def test_diabatic_element_derivatives_match_fd(self):
        m = self.m
        eps = 1.0e-6
        for dfn, vfn, name in [(m.dV11_dR, m.V11, "V11"),
                               (m.dV22_dR, m.V22, "V22"),
                               (m.dV12_dR, m.V12, "V12")]:
            fd = _central_fd(vfn, self.R, eps)
            np.testing.assert_allclose(dfn(self.R), fd, rtol=1e-5, atol=1e-7,
                                       err_msg=f"d{name}/dR analytic vs FD")

    def test_diabatic_matrix_derivative_matches_fd(self):
        m = self.m
        eps = 1.0e-6
        fd = _central_fd(m.diabatic_potential, self.R, eps)
        np.testing.assert_allclose(m.d_diabatic_potential_dR(self.R), fd,
                                   rtol=1e-5, atol=1e-8)

    def test_diabatic_matrix_is_symmetric(self):
        Vd = np.asarray(self.m.diabatic_potential(self.R))
        np.testing.assert_allclose(Vd, np.transpose(Vd, (0, 2, 1)), atol=1e-14)

    def test_adiabatic_energies_are_eigenvalues(self):
        Vd = np.asarray(self.m.diabatic_potential(self.R))
        E = np.sort(np.asarray(self.m.adiabatic_energies(self.R)), axis=-1)
        w = np.sort(np.linalg.eigvalsh(Vd), axis=-1)
        np.testing.assert_allclose(E, w, atol=1e-12)

    def test_adiabatic_states_orthonormal(self):
        S = np.asarray(self.m.adiabatic_states(self.R))          # (..., 2, 2)
        gram = np.einsum("...ki,...kj->...ij", S, S)
        eye = np.broadcast_to(np.eye(2), gram.shape)
        np.testing.assert_allclose(gram, eye, atol=1e-10)

    def test_adiabatic_gap_nonnegative(self):
        assert np.min(self.m.adiabatic_gap(self.R)) >= 0.0


# ===========================================================================
# 2. PBME--MInt integrator invariants  (Mint.py)
# ===========================================================================

class TestMIntExpressions:
    def setup_method(self):
        from Mint import PBMEMIntDynamics, pack_z, unpack_z
        self.dyn = PBMEMIntDynamics()
        self.pack_z, self.unpack_z = pack_z, unpack_z
        rng = np.random.default_rng(7)
        self.z0 = np.array([-1.3, 18.0, 0.9, -0.4, 0.3, 0.6])
        self.dt = 0.5

    def test_pack_unpack_roundtrip(self):
        R, P, r, p = self.unpack_z(self.z0)
        z = self.pack_z(R, P, r, p)
        np.testing.assert_array_equal(z, self.z0)

    def test_energy_conserved_along_trajectory(self):
        dyn, z, dt = self.dyn, self.z0.copy(), self.dt
        e0 = float(dyn.energy(z))
        for _ in range(200):
            z = dyn.step(z, dt)
        assert abs(float(dyn.energy(z)) - e0) < 1e-7

    def test_mapping_radius_conserved_along_trajectory(self):
        dyn, z, dt = self.dyn, self.z0.copy(), self.dt
        m0 = float(dyn.mapping_radius_sq(z))
        for _ in range(200):
            z = dyn.step(z, dt)
        assert abs(float(dyn.mapping_radius_sq(z)) - m0) < 1e-11

    def test_step_jacobian_is_symplectic(self):
        dyn = self.dyn
        J = np.asarray(dyn.compute_step_jacobian(self.z0, self.dt))
        Om = np.asarray(dyn.omega_matrix())
        resid = J.T @ Om @ J - Om
        assert np.linalg.norm(resid) < 1e-6
        assert float(dyn.symplectic_defect(J)) < 1e-6

    def test_step_jacobian_unit_determinant(self):
        J = np.asarray(self.dyn.compute_step_jacobian(self.z0, self.dt))
        assert abs(float(np.linalg.det(J)) - 1.0) < 1e-6

    def test_time_reversal_symmetry(self):
        dyn = self.dyn
        z_fwd = dyn.step(self.z0, self.dt)
        z_back = dyn.step(z_fwd, -self.dt)
        np.testing.assert_allclose(z_back, self.z0, atol=1e-10)


# ===========================================================================
# 3. Monodromy / half-leg backward map geometry  (Monodromy.py)
# ===========================================================================

class TestMonodromyExpressions:
    def setup_method(self):
        from Mint import PBMEMIntDynamics
        from Monodromy import (MonodromyTools, _QCLE_COLUMNS, _QCLE_PAIRS,
                               _QCLE_TRIPLES)
        self.dyn = PBMEMIntDynamics()
        self.tools = MonodromyTools(self.dyn)
        self.cols, self.pairs, self.triples = (_QCLE_COLUMNS, _QCLE_PAIRS,
                                               _QCLE_TRIPLES)
        self.dt = 0.5
        rng = np.random.default_rng(0)   # fixed -> deterministic tolerances
        self.Z = np.column_stack([
            rng.uniform(-2, 2, 6), rng.uniform(5, 25, 6),
            rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 6),
            rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 6)])

    def test_numpy_jax_step_consistency(self):
        from Monodromy import check_mint_jax_consistency
        res = check_mint_jax_consistency(self.dyn)
        assert res["ok"], res
        assert res["max_abs_diff"] < 1e-11

    def test_backward_half_step_geometry_autodiff_vs_fd(self):
        """
        The previously-unverified gap: cross-check the JAX forward-mode autodiff
        tensors of the backward half-step map Y(Z)=Φ^0_{-Δt/2}(Z) against a
        central finite-difference evaluation, order by order (Jacobian J,
        Hessian pairs H, third-order triples T).
        """
        jax = pytest.importorskip("jax")  # noqa: F841 (exact path needs JAX)
        Y_a, J_a, H_a, T_a = self.tools.midpoint_geometry(self.Z, self.dt)
        Y_f, J_f, H_f, T_f = self.tools.midpoint_required_tensors(
            self.Z, self.dt, self.cols, self.pairs, self.triples)

        def _maxdiff(da, df):
            return max(float(np.max(np.abs(np.asarray(da[k]) - np.asarray(df[k]))))
                       for k in da)

        assert float(np.max(np.abs(Y_a - Y_f))) < 1e-10   # value
        assert _maxdiff(J_a, J_f) < 1e-6                   # 1st order
        assert _maxdiff(H_a, H_f) < 5e-4                   # 2nd order
        assert _maxdiff(T_a, T_f) < 2e-2                   # 3rd order

    def test_one_step_forward_monodromy_symplectic(self):
        from Monodromy import test_one_step_monodromy_matrix
        out = test_one_step_monodromy_matrix(
            self.dyn, dt=self.dt, eps=1e-7, precision=6,
            sym_tol=1e-6, det_tol=1e-6)   # raises AssertionError on failure
        assert out["symplectic_fro"] < 1e-6

    def test_backward_half_step_jacobian_symplectic(self):
        from Monodromy import test_midpoint_jacobian_symplecticity
        out = test_midpoint_jacobian_symplecticity(
            self.dyn, dt=self.dt, eps=1e-7, precision=6,
            sym_tol=1e-6, det_tol=1e-6)   # raises AssertionError on failure
        assert out is not None


# ===========================================================================
# 4. Gaussian raw moments closed form  (ProductMoments.py)
# ===========================================================================

class TestProductMomentExpressions:
    def test_normal_raw_moments_vs_gauss_hermite(self):
        from ProductMoments import _normal_moment
        mu = np.array([0.3, -1.2, 2.5])
        var = np.array([0.5, 2.0, 0.05])
        for power in range(5):
            analytic = _normal_moment(mu, var, power)
            reference = _gauss_hermite_moment(mu, var, power, n=24)
            np.testing.assert_allclose(analytic, reference, rtol=1e-10, atol=1e-12,
                                       err_msg=f"E[X^{power}] closed form")

    def test_normal_moment_rejects_high_order(self):
        from ProductMoments import _normal_moment
        with pytest.raises(ValueError):
            _normal_moment(np.array([0.0]), np.array([1.0]), 5)

    def test_static_product_closed_form_norm_and_moments(self):
        # Mirrors the analytic Gaussian-product identities used in production.
        from ProductMoments import (product_norm_raw,
                                     product_quadratic_mapping_moments,
                                     product_kkt_moments)
        pytest.importorskip("torch")  # _FakeProduct fixture lives in the torch suite
        from test_pipeline_core import _FakeProduct
        gp = _FakeProduct()
        assert abs(product_norm_raw(gp) - 8.0 * np.pi / 27.0) < 1e-13
        assert abs(product_quadratic_mapping_moments(gp)["mapping_radius_sq"]
                   - 4.0) < 1e-13
        assert abs(product_kkt_moments(gp)["trace"] - 1.0) < 1e-13


# ===========================================================================
# 5. Grid-QCLE reference solver invariants  (qcle_grid_tully.py)
# ===========================================================================

class TestGridQCLEExpressions:
    def setup_method(self):
        import dataclasses
        from qcle_grid_tully import QCLEGridSolver, QCLEGridParams
        from Models import TullyModel, TullyParams
        params = dataclasses.replace(QCLEGridParams(), n_R=96, n_P=96)
        self.solver = QCLEGridSolver(
            model=TullyModel(TullyParams.defaults("dual")), params=params)
        self.state0 = self.solver.initial_diabat_gaussian(
            R0=-10.0, P0=20.0, sigma_R=1.0, init_state=0)

    def test_populations_sum_equals_trace(self):
        s = self.solver
        p0, p1 = s.populations(self.state0)
        assert abs((p0 + p1) - s.trace(self.state0)) < 1e-12

    def test_trace_and_energy_conserved_under_propagation(self):
        s = self.solver
        tr0 = s.trace(self.state0)
        E0 = s.energy_components(self.state0)["E"]
        _, snaps = s.propagate(self.state0, dt=0.5, n_steps=40,
                               save_every=40, verbose=False)
        stf = snaps[-1]
        assert abs(s.trace(stf) - tr0) < 1e-10
        assert abs(s.energy_components(stf)["E"] - E0) < 1e-6


# ===========================================================================
# 6. SEO-signed MMST sampler statistics  (Sampling.py)
# ===========================================================================

class TestSamplingExpressions:
    def test_seo_signed_sample_means_match_wavepacket(self):
        from Sampling import (GaussianWavePacketParams, MappingInitParams,
                              MMSTSampler)
        sampler = MMSTSampler(
            GaussianWavePacketParams(R0=[-8.0], P0=[30.0], sigma_R=[1.0], hbar=1.0),
            MappingInitParams(nstates=2, init_state=0, hbar=1.0, gamma=0.5))
        rng = np.random.default_rng(3)
        s = sampler.sample_seo_signed(n_samples=200_000, rng=rng)
        R = np.asarray(s.R, float).reshape(-1)
        P = np.asarray(s.P, float).reshape(-1)
        w = np.asarray(s.target_density, float).reshape(-1)
        # Importance-weighted phase-space means recover the wavepacket centre.
        # Signed SEO weights are a high-variance estimator, so the acceptance
        # band is the estimator's own weighted standard error rather than a
        # hard-coded window: a genuine bias shows up as many sigma, while
        # ordinary MC scatter stays within a few.
        sw = np.sum(w)
        R_mean = np.sum(w * R) / sw
        P_mean = np.sum(w * P) / sw
        se_R = np.sqrt(np.sum((w * (R - R_mean)) ** 2)) / abs(sw)
        se_P = np.sqrt(np.sum((w * (P - P_mean)) ** 2)) / abs(sw)
        assert abs(R_mean - (-8.0)) < 5.0 * se_R, (R_mean, se_R)
        assert abs(P_mean - 30.0) < 5.0 * se_P, (P_mean, se_P)


# ===========================================================================
# 7. GP density surrogate derivatives  (GPDerivatives.py) -- torch-gated
# ===========================================================================

class TestGPDensityDerivatives:
    def test_gp_grad_hess_third_vs_finite_difference(self):
        pytest.importorskip("torch")
        from GPDerivatives import test_gp_derivatives_against_finite_differences
        out = test_gp_derivatives_against_finite_differences(feature_zscore=False)
        assert out["max_abs_grad_error"] < 1e-6
        assert out["max_abs_hess_error"] < 1e-4
        assert out["max_abs_third_error"] < 5e-3
        assert out["max_hessian_antisymmetry"] < 1e-12
        assert out["max_third_antisymmetry"] < 1e-12

    def test_gp_derivatives_with_feature_zscoring(self):
        pytest.importorskip("torch")
        from GPDerivatives import test_gp_derivatives_against_finite_differences
        out = test_gp_derivatives_against_finite_differences(feature_zscore=True)
        assert out["max_abs_grad_error"] < 1e-6
        assert out["max_abs_hess_error"] < 1e-4
        assert out["max_abs_third_error"] < 5e-3


# ===========================================================================
# 8. SEO profile derivatives  (GP_Density.seo_profile_derivs) -- torch-gated
# ===========================================================================

class TestSEOProfileDerivatives:
    def test_seo_profile_first_and_second_derivatives_vs_fd(self):
        pytest.importorskip("torch")
        from GP_Density import seo_profile_derivs
        hbar, init_state = 1.0, 0
        # The SEO profile lives on the 4-D mapping coordinate x = (r0, r1, p0, p1);
        # feed a batch of shape (N, 4), not a 1-D scalar sweep.
        rng = np.random.default_rng(0)
        x0 = rng.uniform(-1.2, 1.2, size=(7, 4))
        eps = 1e-6
        g, dg, d2g = seo_profile_derivs(x0, hbar, init_state, 2)
        assert g.shape == (7,)
        assert dg.shape == (7, 4)
        assert d2g.shape == (7, 4, 4)
        # Central finite differences along each of the 4 mapping axes.
        dg_fd = np.zeros_like(dg)
        d2g_fd = np.zeros_like(d2g)
        for a in range(4):
            e = np.zeros(4); e[a] = eps
            g_p, dg_p, _ = seo_profile_derivs(x0 + e, hbar, init_state, 2)
            g_m, dg_m, _ = seo_profile_derivs(x0 - e, hbar, init_state, 2)
            dg_fd[:, a] = (g_p - g_m) / (2.0 * eps)
            d2g_fd[:, :, a] = (dg_p - dg_m) / (2.0 * eps)
        np.testing.assert_allclose(dg, dg_fd, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(d2g, d2g_fd, rtol=1e-5, atol=1e-7)
        # Analytic Hessian must be symmetric in its two derivative indices.
        np.testing.assert_allclose(d2g, np.transpose(d2g, (0, 2, 1)), atol=1e-12)


# ===========================================================================
# 9. QCLE excess-operator coupling term  (Operator.py) -- torch-gated
# ===========================================================================

class TestExcessOperatorCouplingTerm:
    def test_coupling_term_vs_finite_difference(self):
        pytest.importorskip("torch")
        pytest.importorskip("jax")
        from Operator import test_coupling_term_against_finite_differences
        out = test_coupling_term_against_finite_differences()
        # The routine raises AssertionError internally on failure; sanity-check
        # that it returned a diagnostics dict.
        assert isinstance(out, dict)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
