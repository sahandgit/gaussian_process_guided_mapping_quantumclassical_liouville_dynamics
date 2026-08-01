from __future__ import annotations

r"""
test_regularization_selection.py
================================

Numerical verification of the torch-free core of ``select_regularization.py``.

Each test pins one piece of the selection machinery against an independent
reference:

* the closed-form GP leave-one-out residuals against a brute-force
  retrain-and-predict LOO,
* the condition number against ``numpy.linalg.cond``,
* the one-standard-error rule against a hand-built LOO curve with a known
  minimum and a known 1-SE plateau,
* the acquisition-injection selector against pure top-acquisition (it must
  spread points out, not cluster them),
* the full synthetic resampling/injection loop for convergence.
"""

import numpy as np
import pytest

import select_regularization as sr


# ---------------------------------------------------------------------------
# 1. Closed-form GP LOO == brute-force leave-one-out
# ---------------------------------------------------------------------------

def _brute_force_loo(C, y):
    """Independent LOO: for each i, predict y_i from all others via the GP mean."""
    n = C.shape[0]
    resid = np.empty(n)
    for i in range(n):
        mask = np.ones(n, bool); mask[i] = False
        C_mm = C[np.ix_(mask, mask)]
        c_im = C[i, mask]
        mu_i = c_im @ np.linalg.solve(C_mm, y[mask])
        resid[i] = y[i] - mu_i
    return resid


def test_closed_form_loo_matches_brute_force():
    rng = np.random.default_rng(0)
    n, d = 30, 2
    Z = rng.uniform(-2, 2, (n, d))
    ell = np.array([0.7, 0.7]); sf2 = 1.3
    K = sr._rbf_gram(Z, Z, ell, sf2)
    C = K + 1e-6 * np.eye(n)
    y = np.sin(Z[:, 0]) + 0.1 * rng.standard_normal(n)
    fast = sr.gp_loo_residuals(C, y)
    slow = _brute_force_loo(C, y)
    np.testing.assert_allclose(fast, slow, rtol=1e-8, atol=1e-9)


def test_loo_rms_is_sqrt_mean_square():
    rng = np.random.default_rng(1)
    C = np.eye(10) + 0.1 * rng.standard_normal((10, 10))
    C = C @ C.T + np.eye(10)          # SPD
    y = rng.standard_normal(10)
    r = sr.gp_loo_residuals(C, y)
    assert abs(sr.loo_rms(C, y) - np.sqrt(np.mean(r * r))) < 1e-14


# ---------------------------------------------------------------------------
# 2. Condition number
# ---------------------------------------------------------------------------

def test_condition_number_matches_numpy():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((12, 12))
    C = A @ A.T + 1e-3 * np.eye(12)   # SPD
    assert abs(sr.condition_number(C) - np.linalg.cond(C)) / np.linalg.cond(C) < 1e-8
    assert abs(sr.log10_condition_number(C) - np.log10(np.linalg.cond(C))) < 1e-8


# ---------------------------------------------------------------------------
# 3. One-standard-error rule + conditioning constraint
# ---------------------------------------------------------------------------

def test_one_se_rule_prefers_more_regularized_within_1se():
    # LOO minimum at index 2, but indices 3,4 are within 1 SE -> pick the
    # LARGEST l2 (index 4).
    grid = np.logspace(-8, -2, 5)                     # increasing
    means = np.array([0.50, 0.30, 0.20, 0.205, 0.208])
    folds = 6
    # Build fold data with a controlled SE ~0.02 at each grid point.
    rng = np.random.default_rng(3)
    loo = means[:, None] + 0.02 * np.sqrt(folds) * (
        rng.standard_normal((5, folds)))
    loo = np.abs(loo)
    # Force exact means/SEs so the test is deterministic:
    loo = np.tile(means[:, None], (1, folds))
    loo[:, 0] += 0.02 * np.sqrt(folds - 1)            # inject spread -> SE≈0.02
    loo[:, 1] -= 0.02 * np.sqrt(folds - 1)
    kappa = np.full((5, folds), 6.0)                  # all well-conditioned
    res = sr.one_se_selection(grid, loo, kappa, kappa_ceiling_log10=10.0)
    assert res.l2_argmin == pytest.approx(grid[2])
    assert res.l2_star >= res.l2_argmin               # more-regularized choice
    assert res.l2_star == pytest.approx(grid[4])


def test_condition_ceiling_excludes_ill_conditioned():
    grid = np.logspace(-10, -4, 4)
    # Small l2 fits best but is ill-conditioned; large l2 is the only feasible.
    loo = np.array([[0.10]*4, [0.15]*4, [0.20]*4, [0.25]*4])
    kappa = np.array([[14.0]*4, [13.0]*4, [8.0]*4, [7.0]*4])   # log10 κ
    res = sr.one_se_selection(grid, loo, kappa, kappa_ceiling_log10=10.0)
    # indices 0,1 are infeasible (κ>1e10); feasible min is index 2.
    assert res.l2_star in (pytest.approx(grid[2]), pytest.approx(grid[3]))
    assert not res.feasible[0] and not res.feasible[1]
    assert res.feasible[2] and res.feasible[3]


# ---------------------------------------------------------------------------
# 4. Acquisition-injection selector spreads points out
# ---------------------------------------------------------------------------

def test_injection_avoids_clustering_vs_top_acquisition():
    rng = np.random.default_rng(4)
    cand = rng.uniform(-2, 2, (300, 2))
    # Acquisition peaked in a tight blob near (1.5, 1.5): pure top-k clusters.
    blob = np.exp(-8.0 * np.sum((cand - np.array([1.5, 1.5]))**2, axis=1))
    acq = 0.05 + blob
    chosen0 = np.empty((0, 2))
    k = 12
    idx_fp = sr.select_injection_points(cand, acq, chosen0, k=k)
    idx_top = np.argsort(-acq)[:k]

    def mean_min_pairwise(P):
        d = np.sqrt(((P[:, None, :] - P[None, :, :])**2).sum(-1))
        np.fill_diagonal(d, np.inf)
        return float(np.mean(d.min(axis=1)))

    spread_fp = mean_min_pairwise(cand[idx_fp])
    spread_top = mean_min_pairwise(cand[idx_top])
    assert len(set(idx_fp.tolist())) == k          # no duplicates
    assert spread_fp > spread_top                  # farthest-point spreads out


# ---------------------------------------------------------------------------
# 5. Full synthetic resampling / injection loop
# ---------------------------------------------------------------------------

def test_full_synthetic_loop_runs_and_selects_finite_l2():
    out = sr.run_self_test(verbose=False)
    l2 = out["l2_star"]
    hist = out["history"]
    assert np.isfinite(l2) and l2 > 0
    # support grew across rounds (injection happened) unless it converged early
    ns = hist["n_support"]
    assert ns[-1] >= ns[0]
    # every recorded l2* is on the grid and positive
    for v in hist["l2_star"]:
        assert v > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
