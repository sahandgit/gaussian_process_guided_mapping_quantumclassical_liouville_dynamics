from __future__ import annotations

import json
import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Collector import Collector, Snapshot, StepDiagnostics
from FigureCatalog import build_figure_catalog
from KDEDensity import ProjectedNuclearGP, ProjectedNuclearGPConfig
from Mint import PBMEMIntDynamics
from Models import TullyModel, TullyParams
from ProductMoments import (product_kkt_moments, product_norm_raw,
                            product_quadratic_mapping_moments)
from Reproducibility import array_fingerprint
from ReviewerValidation import campaign_cases, seo_basis_matrix
from qcle_grid_tully import QCLEGridParams, QCLEGridSolver


class _FakeInner:
    _initial_fit_done = True
    _alpha = np.array([1.0])
    lengthscales = np.ones(6)
    sigma_f = 1.0
    raw_training_centers = np.zeros((1, 6))
    dynamics = PBMEMIntDynamics()


class _FakeProduct:
    _inner = _FakeInner()
    _hbar = 1.0
    _init_state = 0
    _nstates = 2
    _footpoints = None


class PipelineCoreTests(unittest.TestCase):
    def test_tully_derivatives_match_centered_difference(self):
        model = TullyModel(TullyParams.defaults("dual"))
        R = np.linspace(-2.0, 2.0, 11)
        eps = 1.0e-6
        fd = ((model.diabatic_potential(R + eps)
               - model.diabatic_potential(R - eps)) / (2.0 * eps))
        np.testing.assert_allclose(model.d_diabatic_potential_dR(R), fd,
                                   rtol=2e-7, atol=2e-9)

    def test_mint_preserves_energy_and_mapping_radius(self):
        dyn = PBMEMIntDynamics()
        z = np.array([-1.0, 20.0, 1.0, 0.2, 0.1, 0.7])
        z1 = dyn.step(z, 0.5)
        self.assertLess(abs(float(dyn.energy(z1) - dyn.energy(z))), 1e-9)
        self.assertLess(abs(float(np.dot(z1[2:], z1[2:])
                                  - np.dot(z[2:], z[2:]))), 1e-12)

    def test_static_product_closed_form(self):
        gp = _FakeProduct()
        self.assertAlmostEqual(product_norm_raw(gp), 8.0*np.pi/27.0, places=13)
        qm = product_quadratic_mapping_moments(gp)
        self.assertAlmostEqual(qm["mapping_radius_sq"], 4.0, places=13)
        km = product_kkt_moments(gp)
        self.assertAlmostEqual(km["trace"], 1.0, places=13)

    def test_collector_roundtrip_preserves_product_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            stem = str(Path(td) / "run")
            c = Collector("midpoint", {"paired_initial_cloud": True})
            c.record_diagnostics(StepDiagnostics(0, 0.0, 0.0, 1.0, 1e-4,
                                                  np.ones(6), 0.0,
                                                  {"raw_norm_drift": 0.0}))
            c.record_snapshot(Snapshot(
                0, 0.0, np.zeros((2,6)), np.ones(2), np.ones(2),
                1.0, 1e-4, np.ones(6), is_product=True,
                product_hbar=1.0, product_init_state=0,
                product_nstates=2, product_g_floor_rel=1e-3,
                geometric_measure=np.array([0.2, 0.3]),
            ))
            c.save(stem)
            loaded = Collector.load(stem)
            self.assertTrue(loaded["meta"]["run_metadata"]["paired_initial_cloud"])
            self.assertTrue(loaded["snapshots"][0].is_product)
            self.assertEqual(loaded["snapshots"][0].product_init_state, 0)
            np.testing.assert_allclose(loaded["snapshots"][0].geometric_measure,
                                       [0.2, 0.3])

    def test_fingerprint_is_content_and_shape_sensitive(self):
        a=np.arange(12,dtype=float).reshape(3,4)
        self.assertEqual(array_fingerprint(a),array_fingerprint(a.copy()))
        self.assertNotEqual(array_fingerprint(a),array_fingerprint(a.reshape(4,3)))

    def test_seo_projection_basis_has_four_columns_and_full_rank(self):
        rng=np.random.default_rng(2); B=seo_basis_matrix(rng.normal(size=(100,4)))
        self.assertEqual(B.shape,(100,4)); self.assertEqual(np.linalg.matrix_rank(B),4)

    def test_campaign_includes_refinements_and_replication(self):
        cases=campaign_cases(100,0.5,[1,2,3],20.0)
        self.assertTrue(any(c.n_train==200 for c in cases))
        self.assertTrue(any(c.dt==0.25 for c in cases))
        self.assertGreaterEqual(len({c.seed for c in cases}),3)

    def test_grid_qcle_one_step_preserves_trace(self):
        params=QCLEGridParams(R_min=-15,R_max=15,n_R=64,
                              P_min=-10,P_max=30,n_P=64)
        solver=QCLEGridSolver(TullyModel(TullyParams.defaults("dual")),params)
        state=solver.initial_diabat_gaussian(-5,8,1,0)
        trace0=solver.trace(state); trace1=solver.trace(solver.step(state,0.01))
        self.assertLess(abs(trace1-trace0),1e-13)

    def test_projected_pbme_gp_matches_common_support_kde(self):
        """PBME marginal comparison must differ only by sparse-GP error."""
        rng = np.random.default_rng(17)
        n = 500
        Z = np.zeros((n, 6), dtype=float)
        Z[:, 0] = rng.normal(-8.0, 1.2, n)
        Z[:, 1] = rng.normal(18.0, 0.7, n)
        Z[:, 2:] = rng.normal(size=(n, 4))
        # Focused PBME cancellation: omega_i*y_i = 1/N.
        omega = np.ones(n)
        y = np.full(n, 1.0/n)
        projected = ProjectedNuclearGP(
            ProjectedNuclearGPConfig(max_inducing=128)).fit_from_cloud(
                Z, omega, y, dim_pair=(0, 1))
        R = np.linspace(-12.0, -4.0, 90)
        P = np.linspace(15.5, 20.5, 90)
        kde = projected.kde_grid(R, P)
        gp = projected.gp_grid(R, P)
        # NumPy >=2.4 removes np.trapz.  Do not place it in an eager getattr
        # default because that expression is evaluated even when trapezoid
        # exists.
        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        integrate = lambda a: float(trap(trap(a, R, axis=1), P, axis=0))
        kde /= integrate(kde)
        gp /= integrate(gp)
        e1 = integrate(np.abs(gp-kde))
        self.assertLess(e1, 0.02)
        self.assertAlmostEqual(
            projected.metadata()["target_raw_mass"], 1.0, places=12)

    def test_plotting_sources_cannot_emit_visible_headers(self):
        """No figure or axes title may re-enter either plotting module."""
        # Support both the packaged ``tests/test_pipeline_core.py`` layout
        # and a user copying this file directly beside the modules on
        # Windows.  The old unconditional ``parents[1]`` searched one level
        # too high in the latter layout.
        here = Path(__file__).resolve().parent
        root = here if (here/"Visualization.py").exists() else here.parent
        self.assertTrue((root/"Visualization.py").exists(),
                        f"Could not locate plotting modules relative to {__file__}")
        for filename in ("Visualization.py", "Compare_gp_se_qcle.py"):
            tree = ast.parse((root/filename).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr == "suptitle":
                    self.fail(f"Visible figure header found in {filename}:{node.lineno}")
                if node.func.attr == "set_title" and node.args:
                    arg = node.args[0]
                    is_empty = isinstance(arg, ast.Constant) and arg.value == ""
                    self.assertTrue(is_empty,
                                    f"Visible axes header found in {filename}:{node.lineno}")

    def test_breathing_refit_source_has_transactional_full_state_restore(self):
        """The long-run singularity guard must not regress to ell-only restore."""
        here = Path(__file__).resolve().parent
        root = here if (here/"GP_Density.py").exists() else here.parent
        source = (root/"GP_Density.py").read_text(encoding="utf-8")
        start = source.index("    def _breathing_optimize_lengthscales(")
        end = source.index("\n    def refit(", start)
        breathing = source[start:end]
        self.assertIn('"log_lengthscales":', breathing)
        self.assertIn('"log_sigma_n":', breathing)
        self.assertIn("_restore_best_state()", breathing)
        self.assertIn("line_search_fn=None", breathing)
        self.assertNotIn('line_search_fn="strong_wolfe"', breathing)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                         "PyTorch is not installed in the lightweight test environment")
    def test_cholesky_records_scale_aware_jitter_for_duplicate_points(self):
        """A singular duplicate-point covariance is repaired and audited."""
        import torch
        from GP_Density import GPDensity, GPDensityConfig

        gp = GPDensity(
            GPDensityConfig(jitter=0.0, constraints_enabled=False),
            PBMEMIntDynamics(),
        )
        Ky = torch.ones((4, 4), dtype=torch.float64)
        L = gp._cholesky(Ky)
        self.assertTrue(bool(torch.isfinite(L).all().item()))
        self.assertGreater(gp.last_cholesky_adaptive_jitter, 0.0)
        self.assertGreater(gp.last_cholesky_attempts, 1)
        self.assertTrue(np.isfinite(gp.last_cholesky_min_eigenvalue))

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                         "PyTorch is not installed in the lightweight test environment")
    def test_cholesky_audits_nearly_duplicate_support_points(self):
        """A near-singular RBF matrix from nearly coincident points stays finite."""
        import torch
        from GP_Density import GPDensity, GPDensityConfig

        gp = GPDensity(
            GPDensityConfig(jitter=0.0, constraints_enabled=False),
            PBMEMIntDynamics(),
        )
        support = torch.tensor(
            [[0.0], [1.0e-14], [1.0], [2.0]],
            dtype=torch.float64,
        )
        squared_distance = (
            support.square()
            + support.square().T
            - 2.0 * support @ support.T
        ).clamp_min(0.0)
        Ky = torch.exp(-0.5 * squared_distance)
        L = gp._cholesky(Ky)
        self.assertTrue(bool(torch.isfinite(L).all().item()))
        self.assertGreaterEqual(gp.last_cholesky_adaptive_jitter, 0.0)
        self.assertGreaterEqual(gp.last_cholesky_attempts, 1)
        self.assertTrue(np.isfinite(gp.last_cholesky_min_eigenvalue))

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                         "PyTorch is not installed in the lightweight test environment")
    def test_derivative_bundle_query_batching_is_numerically_identical(self):
        """Large-query memory batching must not change GP derivatives."""
        from GPDerivatives import rho_derivative_bundle
        from GP_Density import GPDensity, GPDensityConfig

        rng = np.random.default_rng(9182)
        Z = rng.normal(size=(400, 6))
        y = np.sin(Z[:, 0]) + 0.2 * Z[:, 1] * Z[:, 2]
        gp = GPDensity(GPDensityConfig(
            n_opt_steps=0, constraints_enabled=False,
            refit_hyper_policy="frozen",
        ), PBMEMIntDynamics())
        gp.fit(Z, y, moment_targets=None, apply_constraints=False)
        Y = rng.normal(size=(1900, 6))  # 760,000 pairs triggers batching.
        batched = rho_derivative_bundle(gp, Y)
        direct_parts = [
            rho_derivative_bundle(gp, Y[start:start + 200])
            for start in range(0, len(Y), 200)
        ]
        direct = tuple(
            np.concatenate([part[index] for part in direct_parts], axis=0)
            for index in range(3)
        )
        for observed, expected in zip(batched, direct):
            np.testing.assert_allclose(
                observed, expected, rtol=1.0e-13, atol=1.0e-14
            )

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                         "PyTorch is not installed in the lightweight test environment")
    def test_breathing_refit_restores_noise_and_lengthscales_on_bad_candidate(self):
        """A rejected adaptive candidate cannot poison the next alpha solve."""
        import torch
        from GP_Density import GPDensity, GPDensityConfig

        cfg = GPDensityConfig(
            init_log_lengthscales=np.zeros(6),
            init_log_sigma_n=-2.5,
            use_loocv=True,
            breathing_anchor_policy="initial",
        )
        gp = GPDensity(cfg, PBMEMIntDynamics())
        gp._initial_fit_done = True
        gp._initial_log_sigma_f_anchor = 0.0
        gp._initial_log_sigma_n_anchor = -2.5
        gp._initial_log_lengthscales_anchor = np.zeros(6)
        Z = torch.zeros((8, 6), dtype=torch.float64)
        y = torch.ones(8, dtype=torch.float64)
        ell0 = gp.log_lengthscales.detach().clone()
        sn0 = gp.log_sigma_n.detach().clone()

        calls = {"n": 0}

        def flaky_objective(_Z, _y):
            calls["n"] += 1
            # Baseline and optimizer closure are valid; validation of the
            # proposed post-update state simulates a failed Cholesky trial.
            if calls["n"] >= 3:
                raise RuntimeError("synthetic non-positive-definite candidate")
            return (torch.sum((gp.log_lengthscales - 0.3) ** 2)
                    + (gp.log_sigma_n + 2.0) ** 2)

        gp._loo_cv_loss = flaky_objective
        history = gp._breathing_optimize_lengthscales(
            Z, y, n_steps=1, prior_weight=0.0, prior_clip=1.0)

        self.assertEqual(history, [])
        self.assertTrue(gp.last_breathing_failed)
        self.assertEqual(gp.last_breathing_failure_code, 5)
        self.assertEqual(gp.breathing_failure_count, 1)
        self.assertTrue(torch.equal(gp.log_lengthscales.detach(), ell0))
        self.assertTrue(torch.equal(gp.log_sigma_n.detach(), sn0))


if __name__ == "__main__":
    unittest.main()
