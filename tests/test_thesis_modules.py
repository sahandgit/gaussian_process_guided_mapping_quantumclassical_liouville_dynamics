from __future__ import annotations

r"""
test_thesis_modules.py
======================

Pytest wrapper for the three thesis-revision modules:

    thesis_analysis.py        one-week items: figures + tables
    seo_coefficient_gp.py     Route B: projection-preserving representation
    conservative_excess.py    conservative / weak-form excess discretization

Each module carries its own ``run_self_test``; these tests exercise those and
add independent checks of the properties the examiner asked us to demonstrate.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import conservative_excess as ce
import seo_coefficient_gp as seo
import thesis_analysis as ta
import thesis_closure as tc
from Compare_gp_se_qcle import qcle_boundary_masses


# ---------------------------------------------------------------------------
# Module self-tests
# ---------------------------------------------------------------------------

def test_thesis_analysis_self_test():
    ta.run_self_test()


def test_seo_coefficient_gp_self_test():
    seo.run_self_test()


def test_conservative_excess_self_test():
    ce.run_self_test()


def test_thesis_closure_self_test():
    tc.run_self_test()


def test_qcle_boundary_gate_uses_physical_marginals():
    rho = np.zeros((20, 20))
    rho[10, 10] = 2.0
    # Equal and opposite Wigner lobes at an R boundary cancel in the physical
    # R marginal but remain visible in the separate abs(W) ringing diagnostic.
    rho[0, 8] = 1.0
    rho[0, 9] = -1.0

    edge = qcle_boundary_masses(rho, dR=0.5, dP=0.25)

    assert edge["marginal_R"] == pytest.approx(0.0)
    assert edge["phase_space_R"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# thesis_closure -- three-level reference, audit, nesting
# ---------------------------------------------------------------------------

def test_three_level_assembly_recovers_order():
    lv = [tc.ReferenceLevel("coarse", 0.2, 100, 2048, {"P0": 4.0}, {}),
          tc.ReferenceLevel("fine", 0.1, 200, 4096, {"P0": 1.0}, {}),
          tc.ReferenceLevel("finer", 0.05, 400, 8192, {"P0": 0.25}, {})]
    res = tc._assemble_reference("tdse", lv, "both")
    assert res["status"] == "COMPLETE"
    assert float(res["observables"]["P0"]["p_obs"]) == pytest.approx(2.0)
    assert res["observables"]["P0"]["ratio"] == pytest.approx(4.0)


def test_two_levels_refuses_to_report_convergence():
    lv = [tc.ReferenceLevel("a", 0.2, 100, 2048, {"P0": 1.0}, {}),
          tc.ReferenceLevel("b", 0.1, 200, 4096, {"P0": 0.5}, {})]
    res = tc._assemble_reference("tdse", lv, "both")
    assert tc.NOT_COMPUTED in str(res["status"])


def test_figure_audit_matches_real_sidecar_schema(tmp_path):
    """
    Regression: the audit must use the pipeline's actual sidecar keys
    (figure/normalization/scale_policy/run_metadata), not an invented schema.
    A null run_metadata is the substantive failure, and must be reported as
    such rather than as a generic 'missing fields'.
    """
    root = tmp_path / "root"
    (root / "f").mkdir(parents=True)

    ok = root / "f" / "ok.png"; ok.write_bytes(b"x")
    Path(str(ok) + ".meta.json").write_text(json.dumps({
        **{k: "v" for k in tc.SIDECAR_FIELDS},
        "run_metadata": {k: 1 for k in tc.PROVENANCE_FIELDS}}),
        encoding="utf-8")

    # exactly the shape found in results/**/*.pdf.meta.json
    bad = root / "f" / "bad.png"; bad.write_bytes(b"x")
    Path(str(bad) + ".meta.json").write_text(json.dumps({
        "figure": "bad.png", "title": "P0",
        "normalization": "stated by axis and legend labels",
        "scale_policy": "shared across compared methods",
        "data_sources": [], "run_metadata": None,
        "deviations_from_run_configuration": None}), encoding="utf-8")

    nom = root / "f" / "none.png"; nom.write_bytes(b"x")

    rows = tc.audit_figures([root])
    by = {Path(r["figure"]).name: r for r in rows}
    assert by["ok"]["status"] == "PASS"
    assert "run_metadata empty" in by["bad"]["status"]
    assert by["bad"]["provenance_missing"] == list(tc.PROVENANCE_FIELDS)
    assert by["none"]["has_sidecar"] is False

    summ = tc.figure_audit_report(rows, tmp_path / "out")
    assert summ["n_figures"] == 3 and summ["n_pass"] == 1
    assert summ["n_empty_run_metadata"] == 1
    assert (tmp_path / "out" / "figure_audit.tex").exists()

    # Regression: the summary fields must actually reach the JSON on disk,
    # not merely exist on the in-memory dict.
    on_disk = json.loads(
        (tmp_path / "out" / "figure_audit.json").read_text(encoding="utf-8"))
    for k in ("n_no_sidecar", "n_empty_run_metadata", "missing_field_counts"):
        assert k in on_disk, k
    assert on_disk["n_empty_run_metadata"] == 1
    assert on_disk["n_no_sidecar"] == 1


def test_figure_audit_deduplicates_pdf_png_pairs(tmp_path):
    root = tmp_path / "r"; (root / "f").mkdir(parents=True)
    for ext in (".pdf", ".png"):
        (root / "f" / f"same{ext}").write_bytes(b"x")
    assert len(tc.audit_figures([root])) == 1


def test_observed_order_refuses_zero_numerator():
    """Coarse and mid identical -> log2(0/b) = -inf; must be refused."""
    import reviewer_closure_campaign as rcc
    p, why = rcc.observed_order(np.array([1.0]), np.array([1.0]),
                                np.array([0.5]))
    assert p is None and "numerator underflow" in why


def test_nested_plan_levels_are_strict_subsets(tmp_path):
    plan = tc.write_nested_support_plan(tmp_path, n_max=120,
                                        levels=(30, 60, 120), seeds=(11,))
    m = plan["per_seed"]["11"]
    s30, s60, s120 = set(m["30"]), set(m["60"]), set(m["120"])
    assert s30 < s60 < s120
    assert (len(s30), len(s60), len(s120)) == (30, 60, 120)


# ---------------------------------------------------------------------------
# seo_coefficient_gp -- the two structural defects are fixed
# ---------------------------------------------------------------------------

def test_seo_basis_matches_pipeline_definition():
    """Must agree with ReviewerValidation.seo_basis_matrix bit-for-bit."""
    rv = pytest.importorskip("ReviewerValidation")
    rng = np.random.default_rng(3)
    x = rng.uniform(-1.5, 1.5, size=(25, 4))
    np.testing.assert_allclose(seo.seo_basis(x, 1.0),
                               rv.seo_basis_matrix(x, 1.0), atol=1e-12)


def test_seo_hessian_matches_finite_differences():
    """Analytic mapping second derivatives vs central FD."""
    rng = np.random.default_rng(5)
    x = rng.uniform(-1.0, 1.0, size=(4, 4))
    H = seo.seo_basis_hessian(x, 1.0)
    h = 1e-4
    for i in range(x.shape[0]):
        for a in range(seo.N_BASIS):
            for u in range(4):
                for v in range(4):
                    eu = np.zeros(4); eu[u] = h
                    ev = np.zeros(4); ev[v] = h
                    f = lambda z: float(seo.seo_basis(z.reshape(1, 4), 1.0)[0, a])
                    fd = (f(x[i] + eu + ev) - f(x[i] + eu - ev)
                          - f(x[i] - eu + ev) + f(x[i] - eu - ev)) / (4 * h * h)
                    assert abs(H[i, a, u, v] - fd) < 5e-5


def test_seo_representation_has_zero_leakage():
    """
    The defect the examiner identified: the product ansatz leaks 27-95% out of
    the SEO span. A field built in the exact basis must leak nothing.
    """
    rng = np.random.default_rng(7)
    Xb = rng.uniform(-2, 2, size=(30, 2))
    C = rng.standard_normal((30, 4))
    gp = seo.SEOCoefficientSurrogate(sigma_n2=1e-12).fit(Xb, C)
    probes = rng.uniform(-1.5, 1.5, size=(300, 4))
    for q in ([0.0, 0.0], [1.0, -0.5], [-1.7, 1.3]):
        assert gp.projection_residual(np.array(q), probes) < 1e-10


def test_seo_dc_dP_is_analytic_and_correct():
    rng = np.random.default_rng(11)
    Xb = rng.uniform(-2, 2, size=(25, 2))
    C = np.column_stack([np.sin(Xb[:, 1]), np.cos(Xb[:, 1]),
                         Xb[:, 0] * 0.3, Xb[:, 1] * 0.2])
    gp = seo.SEOCoefficientSurrogate(sigma_n2=1e-12).fit(Xb, C)
    q = np.array([[0.4, 0.2]])
    eps = 1e-6
    fd = (gp.coefficients(q + np.array([[0.0, eps]]))[0]
          - gp.coefficients(q - np.array([[0.0, eps]]))[0]) / (2 * eps)
    np.testing.assert_allclose(gp.dc_dP(q)[0], fd, atol=1e-6)


def test_seo_excess_source_linear_in_hamiltonian_derivative():
    rng = np.random.default_rng(13)
    Xb = rng.uniform(-1, 1, size=(20, 2))
    gp = seo.SEOCoefficientSurrogate(sigma_n2=1e-12).fit(
        Xb, rng.standard_normal((20, 4)))
    n = 6
    Xq = rng.uniform(-1, 1, size=(n, 2))
    xq = rng.uniform(-1, 1, size=(n, 4))
    dh = rng.standard_normal((n, 2, 2)) * 0.05
    q1 = gp.excess_source(Xq, xq, dh)
    q3 = gp.excess_source(Xq, xq, 3.0 * dh)
    np.testing.assert_allclose(q3, 3.0 * q1, atol=1e-12)
    assert np.all(np.isfinite(q1))


# ---------------------------------------------------------------------------
# conservative_excess -- exact discrete conservation
# ---------------------------------------------------------------------------

def test_conservation_holds_for_arbitrary_flux():
    """
    The decisive property: 1^T A = 0 regardless of the flux values, so the
    normalization functional is an exact left null vector of the generator.
    """
    grid = ce.MomentumGrid(-8.0, 8.0, 48)
    rng = np.random.default_rng(17)
    for _ in range(5):
        A = ce.conservative_generator(grid, rng.standard_normal((49, 48)))
        assert ce.conservation_residual(A) < 1e-10


def test_mass_preserved_under_explicit_evolution():
    grid = ce.MomentumGrid(-8.0, 8.0, 48)
    rng = np.random.default_rng(19)
    A = ce.conservative_generator(grid, rng.standard_normal((49, 48)))
    rho = np.exp(-0.5 * (grid.centres / 1.5) ** 2)
    m0 = float(np.sum(rho) * grid.dP)
    r = rho.copy()
    for _ in range(500):
        r = r + 5e-4 * (A @ r)
    assert abs(float(np.sum(r) * grid.dP) - m0) < 1e-10


def test_weak_form_constant_test_function_is_null():
    grid = ce.MomentumGrid(-5.0, 5.0, 32)
    rng = np.random.default_rng(23)
    Psi = np.vstack([np.ones(32), grid.centres, grid.centres ** 2])
    W = ce.weak_form_matrix(grid, Psi, rng.standard_normal((33, 32)))
    assert np.max(np.abs(W[0])) < 1e-10        # conservation in weak form
    assert np.max(np.abs(W[1])) > 0.0


def test_grid_geometry():
    g = ce.MomentumGrid(-1.0, 1.0, 4)
    assert g.dP == pytest.approx(0.5)
    np.testing.assert_allclose(g.centres, [-0.75, -0.25, 0.25, 0.75])
    np.testing.assert_allclose(g.faces, [-1.0, -0.5, 0.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# thesis_analysis -- table machinery
# ---------------------------------------------------------------------------

def test_latex_table_emits_booktabs_and_handles_empty():
    rows = [{"a": 1, "b": 2.5}, {"a": 3, "b": float("nan")}]
    tex = ta.latex_table(rows, ["a", "b"], "cap", "tab:x")
    assert r"\begin{table}" in tex and r"\bottomrule" in tex
    assert "cap" in tex and "tab:x" in tex
    empty = ta.latex_table([], ["a"], "cap", "tab:y")
    assert ta.NOT_COMPUTED in empty or empty.strip().startswith("%")


def test_manifest_value_finds_nested_leaf():
    man = {"config": {"nested": {"l2_regularization": 0.05}}, "other": 1}
    assert ta.manifest_value(man, "l2_regularization") == 0.05
    assert ta.manifest_value(man, "absent") is None
    assert ta.manifest_value(None, "anything") is None


def test_observed_order_recovers_second_order(tmp_path):
    """Synthetic exact 2nd-order ladder -> p_obs = 2."""
    t = np.linspace(0.0, 5.0, 41)
    base = np.cos(t)
    root = tmp_path / "reviewer_closure_x"
    for dt, off in ((0.5, 4.0), (0.25, 1.0), (0.125, 0.25)):
        d = root / "step7_dt_P020" / f"seed11_dt{dt:g}"
        d.mkdir(parents=True)
        np.savez(d / "midpoint.npz", t=t, lw_P0=base + off * 1e-3)
    rows = ta.compute_observed_order(root)
    assert len(rows) == 1
    assert float(rows[0]["p_obs"]) == pytest.approx(2.0, abs=1e-6)


def test_replication_statistics(tmp_path):
    t = np.linspace(0.0, 1.0, 11)
    root = tmp_path / "reviewer_closure_y"
    for seed, v in ((11, 1.0), (29, 2.0), (47, 3.0)):
        d = root / "step9_repl_P0100" / f"seed{seed}"
        d.mkdir(parents=True)
        np.savez(d / "midpoint.npz", t=t, lw_P0=np.full_like(t, v),
                 raw_norm_drift=np.zeros_like(t))
    rows = ta.compute_replication_statistics(root)
    lw = [r for r in rows if r["quantity"] == "lw_P0"][0]
    assert lw["n_seeds"] == 3
    assert lw["mean"] == pytest.approx(2.0)
    assert lw["sample_sd"] == pytest.approx(1.0)
    assert lw["standard_error"] == pytest.approx(1.0 / np.sqrt(3))


def test_raw_conservation_separates_kinds(tmp_path):
    t = np.linspace(0.0, 3000.0, 301)
    root = tmp_path / "reviewer_closure_z"
    d = root / "step9_repl_P020" / "seed11"
    d.mkdir(parents=True)
    np.savez(d / "midpoint.npz", t=t, raw_norm_drift=1e-9 * t,
             km_normalization=np.ones_like(t))
    rows = ta.compute_raw_conservation(root)
    kinds = {r["kind"] for r in rows}
    assert "raw" in kinds and "self-normalized" in kinds
    raw = [r for r in rows if r["quantity"] == "raw_norm_drift"][0]
    assert raw["endpoint"] == pytest.approx(3e-6)
    assert raw["pre_interaction_max_abs"] is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
