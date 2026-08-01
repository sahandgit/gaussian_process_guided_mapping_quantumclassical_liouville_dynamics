from __future__ import annotations

# --- UTF-8 console safety (see run.py) ---
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
# -----------------------------------------

r"""
select_regularization.py
========================

Choose the GP surrogate's ``l2_regularization`` (the hyperparameter-shrinkage
weight in ``GPDensityConfig``) by **nested resampling with active point
injection**, judged by **leave-one-out predictive RMS** subject to a
**condition-number ceiling**.

Why this design
---------------
The reported symptom — "with regularization the results change significantly" —
is the signature of a high-variance, under-supported surrogate.  The remedy is
not to pick the ``l2`` that best fits one cloud (circular), but to:

  1.  Score each candidate ``l2`` by out-of-sample error (closed-form GP LOO,
      the same estimator used by ``GPDensity._loo_cv_loss``), averaged over
      several **independent resampled clouds**.
  2.  Apply the **one-standard-error rule** (Breiman): take the *most
      regularized* ``l2`` whose mean LOO-RMS is within one standard error of the
      grid minimum.  This deliberately favors the stabler surrogate.
  3.  Reject any ``l2`` whose Gram condition number exceeds a ceiling, so the
      Cholesky stays well posed.
  4.  Wrap all of that in an **injection loop**: after selecting ``l2*`` at the
      current support, inject new points where the posterior variance is largest
      (farthest-point-filtered so they do not cluster), refit, and re-select.
      The trustworthy ``l2*`` is the fixed point — the value that stops moving,
      and whose LOO-RMS stops improving, as points are injected.  If ``l2*``
      keeps shrinking with every injection, the ridge was masking missing
      support and injection (not regularization) is the cure.

Two layers
----------
* A **pure-NumPy core** (``gp_loo_residuals``, ``condition_number``,
  ``one_se_selection``, ``select_injection_points``, ``resampling_injection_loop``)
  that is dependency-free and unit-tested in ``test_regularization_selection.py``.
* A **production adapter** (``GPDensityScorer``) that wires the core to the real
  ``GPDensity`` / ``MMSTSampler`` (requires PyTorch).  Run it with
  ``python select_regularization.py --production``.

A synthetic, torch-free demonstration runs with ``--self_test`` (or ``--demo``)
so the whole loop can be exercised without the heavy stack.
"""

import argparse
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

FloatArray = np.ndarray


# ===========================================================================
# Pure-NumPy core  (unit-tested; no torch/jax dependency)
# ===========================================================================

def gp_loo_residuals(C: FloatArray, y: FloatArray) -> FloatArray:
    r"""
    Closed-form leave-one-out residuals of a GP regressor.

    For a GP with (noise-augmented) Gram matrix ``C = K + σ_n² I`` and targets
    ``y``, the exact leave-one-out prediction error at point *i* is

        ê_i = α_i / [C⁻¹]_{ii},          α = C⁻¹ y

    (Rasmussen & Williams, eq. 5.12).  This is the identity ``GPDensity`` uses
    in ``_loo_cv_loss``; here it is recomputed in NumPy so the selection logic
    is testable without torch.  Verified against a brute-force retrain in the
    accompanying test suite.

    Parameters
    ----------
    C : (N, N)  symmetric positive-definite Gram matrix *including* the noise
        diagonal (i.e. what the code calls ``Ky``).
    y : (N,)    targets.

    Returns
    -------
    resid : (N,)  leave-one-out residuals  y_i − μ_{−i}(x_i).
    """
    C = np.asarray(C, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = C.shape[0]
    L = np.linalg.cholesky(C)
    alpha = np.linalg.solve(C, y)
    # diag(C⁻¹) = column-sum of squares of L⁻¹ (since C⁻¹ = L⁻ᵀ L⁻¹).
    Linv = np.linalg.solve_triangular(L, np.eye(n), lower=True) \
        if hasattr(np.linalg, "solve_triangular") else _solve_lower(L, np.eye(n))
    Cinv_diag = np.sum(Linv * Linv, axis=0)
    Cinv_diag = np.clip(Cinv_diag, 1.0e-30, None)
    return alpha / Cinv_diag


def _solve_lower(L: FloatArray, B: FloatArray) -> FloatArray:
    """Forward-substitution solve L X = B for lower-triangular L (NumPy fallback)."""
    from scipy.linalg import solve_triangular as _st
    return _st(L, B, lower=True)


def loo_rms(C: FloatArray, y: FloatArray) -> float:
    """Root-mean-square leave-one-out predictive error (= sqrt of ``_loo_cv_loss``)."""
    r = gp_loo_residuals(C, y)
    return float(np.sqrt(np.mean(r * r)))


def condition_number(C: FloatArray) -> float:
    """Spectral condition number κ = λ_max / λ_min of a symmetric PD matrix."""
    w = np.linalg.eigvalsh(np.asarray(C, dtype=np.float64))
    lo = max(float(w[0]), 1.0e-300)
    return float(w[-1] / lo)


def log10_condition_number(C: FloatArray) -> float:
    return float(np.log10(condition_number(C)))


@dataclass
class SelectionResult:
    l2_star: float
    l2_argmin: float
    grid: FloatArray
    mean_loo: FloatArray            # mean LOO-RMS per l2 (over folds)
    se_loo: FloatArray             # standard error per l2
    median_log_kappa: FloatArray   # conditioning per l2
    feasible: FloatArray           # bool mask: log κ ≤ ceiling
    kappa_ceiling: float

    def as_dict(self) -> Dict:
        return {
            "l2_star": self.l2_star,
            "l2_argmin": self.l2_argmin,
            "kappa_ceiling_log10": self.kappa_ceiling,
            "grid": [float(x) for x in self.grid],
            "mean_loo_rms": [float(x) for x in self.mean_loo],
            "se_loo_rms": [float(x) for x in self.se_loo],
            "median_log10_kappa": [float(x) for x in self.median_log_kappa],
            "feasible": [bool(x) for x in self.feasible],
        }


def one_se_selection(
    l2_grid: Sequence[float],
    loo_rms_folds: FloatArray,       # (n_l2, n_folds)
    log_kappa_folds: FloatArray,     # (n_l2, n_folds)
    kappa_ceiling_log10: float = 10.0,
) -> SelectionResult:
    r"""
    One-standard-error rule with a condition-number constraint.

    Among the ``l2`` values whose *median* log₁₀ κ is at or below the ceiling,
    find the grid minimum of mean LOO-RMS, then return the **largest** ``l2``
    (most regularized / most stable) whose mean LOO-RMS is within one standard
    error of that minimum.

    Rationale: the 1-SE rule trades a statistically-insignificant amount of
    fit for a meaningfully more stable surrogate — exactly what damps the
    observable's sensitivity to the cloud.
    """
    grid = np.asarray(l2_grid, dtype=np.float64)
    loo = np.asarray(loo_rms_folds, dtype=np.float64)
    kap = np.asarray(log_kappa_folds, dtype=np.float64)
    order = np.argsort(grid)
    grid, loo, kap = grid[order], loo[order], kap[order]

    n_folds = loo.shape[1]
    mean_loo = loo.mean(axis=1)
    se_loo = loo.std(axis=1, ddof=1) / np.sqrt(max(n_folds, 1)) if n_folds > 1 \
        else np.zeros_like(mean_loo)
    med_kappa = np.median(kap, axis=1)
    feasible = med_kappa <= float(kappa_ceiling_log10)

    if not np.any(feasible):
        # No l2 satisfies conditioning; fall back to the best-conditioned one.
        i_star = int(np.argmin(med_kappa))
        return SelectionResult(float(grid[i_star]), float(grid[i_star]), grid,
                               mean_loo, se_loo, med_kappa, feasible,
                               float(kappa_ceiling_log10))

    masked_mean = np.where(feasible, mean_loo, np.inf)
    i_min = int(np.argmin(masked_mean))
    threshold = mean_loo[i_min] + se_loo[i_min]

    # Largest l2 within the threshold AND feasible.
    within = feasible & (mean_loo <= threshold)
    i_star = int(np.max(np.where(within)[0])) if np.any(within) else i_min
    return SelectionResult(float(grid[i_star]), float(grid[i_min]), grid,
                           mean_loo, se_loo, med_kappa, feasible,
                           float(kappa_ceiling_log10))


def select_injection_points(
    cand_Z: FloatArray,
    cand_acq: FloatArray,
    chosen_Z: FloatArray,
    k: int,
    length_scales: Optional[FloatArray] = None,
) -> np.ndarray:
    r"""
    Pick ``k`` new support points by greedy acquisition × farthest-point.

    At each step choose the candidate maximizing ``acq · min_dist_to_selected``
    (distances in optionally length-scaled coordinates), then add it to the
    selected set.  This concentrates new points where the posterior is
    uncertain (large ``acq``) while spreading them out so they do not pile up in
    one region.

    Returns the row indices into ``cand_Z`` of the chosen points.
    """
    cand_Z = np.asarray(cand_Z, dtype=np.float64)
    acq = np.asarray(cand_acq, dtype=np.float64).reshape(-1)
    ls = np.ones(cand_Z.shape[1]) if length_scales is None \
        else np.asarray(length_scales, dtype=np.float64).reshape(-1)
    ls = np.where(ls > 0, ls, 1.0)

    ref = np.asarray(chosen_Z, dtype=np.float64) / ls
    candS = cand_Z / ls

    def min_dist(points_S, ref_S):
        if ref_S.shape[0] == 0:
            return np.full(points_S.shape[0], np.inf)
        d2 = np.sum(points_S**2, axis=1)[:, None] \
            + np.sum(ref_S**2, axis=1)[None, :] \
            - 2.0 * points_S @ ref_S.T
        return np.sqrt(np.clip(d2.min(axis=1), 0.0, None))

    chosen_idx: List[int] = []
    ref_S = ref.copy()
    available = np.ones(cand_Z.shape[0], dtype=bool)
    # Normalize acquisition to [0,1] so the product is scale-free.
    a = acq - acq.min()
    a = a / (a.max() + 1e-30)
    for _ in range(min(k, cand_Z.shape[0])):
        md = min_dist(candS, ref_S)
        # First pick (empty reference) -> all distances are +inf; fall back to
        # pure acquisition by giving every candidate the same unit distance so
        # 0*inf never arises.
        finite = np.isfinite(md)
        if not np.any(finite):
            md = np.ones_like(md)
        else:
            md = np.where(finite, md, md[finite].max())
        score = a * md
        score[~available] = -np.inf
        j = int(np.argmax(score))
        chosen_idx.append(j)
        available[j] = False
        ref_S = np.vstack([ref_S, candS[j][None, :]])
    return np.array(chosen_idx, dtype=np.int64)


@dataclass
class InjectionHistory:
    n_support: List[int] = field(default_factory=list)
    l2_star: List[float] = field(default_factory=list)
    loo_at_star: List[float] = field(default_factory=list)
    log_kappa_at_star: List[float] = field(default_factory=list)
    selections: List[Dict] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "n_support": self.n_support,
            "l2_star": self.l2_star,
            "loo_rms_at_star": self.loo_at_star,
            "log10_kappa_at_star": self.log_kappa_at_star,
            "per_round_selection": self.selections,
        }

    def converged(self, rtol_l2: float = 0.5, rtol_loo: float = 0.05) -> bool:
        """
        Fixed-point test: l2* stable to within ``rtol_l2`` in log10 and LOO-RMS
        improving by less than ``rtol_loo`` relatively over the last round.
        """
        if len(self.l2_star) < 2:
            return False
        dl2 = abs(np.log10(self.l2_star[-1] + 1e-300)
                  - np.log10(self.l2_star[-2] + 1e-300))
        dloo = abs(self.loo_at_star[-1] - self.loo_at_star[-2]) \
            / (abs(self.loo_at_star[-2]) + 1e-30)
        return (dl2 <= rtol_l2) and (dloo <= rtol_loo)


# Signature of the fit-and-score callback the loop needs.
#   fit_score_fn(l2, fold_seed, support_state) ->
#       dict(loo_rms=float, log_kappa=float,
#            cand_Z=(P,D), cand_acq=(P,), length_scales=(D,))
FitScoreFn = Callable[[float, int, "SupportState"], Dict]


@dataclass
class SupportState:
    """Opaque handle describing the current support set; adapter-defined."""
    Z: FloatArray
    extra: Dict = field(default_factory=dict)


def resampling_injection_loop(
    fit_score_fn: FitScoreFn,
    inject_fn: Callable[["SupportState", np.ndarray], "SupportState"],
    initial_state: "SupportState",
    l2_grid: Sequence[float],
    n_rounds: int = 4,
    n_folds: int = 5,
    k_inject: int = 50,
    kappa_ceiling_log10: float = 10.0,
    base_seed: int = 0,
    verbose: bool = True,
) -> Tuple[float, InjectionHistory]:
    """
    Outer active-resampling loop.  Returns the final ``l2*`` and the history.

    ``fit_score_fn`` fits the surrogate at a given ``l2`` on a resampled cloud
    (indexed by ``fold_seed``) at the current support and returns its LOO-RMS,
    log₁₀ κ, and a candidate pool with acquisition scores for injection.
    ``inject_fn`` returns a new ``SupportState`` given the chosen candidate rows.
    """
    grid = np.asarray(l2_grid, dtype=np.float64)
    state = initial_state
    hist = InjectionHistory()

    for rnd in range(n_rounds):
        loo_folds = np.zeros((grid.size, n_folds))
        kap_folds = np.zeros((grid.size, n_folds))
        last_pool: Dict = {}
        for gi, l2 in enumerate(grid):
            for fi in range(n_folds):
                out = fit_score_fn(float(l2), base_seed + 1000 * rnd + fi, state)
                loo_folds[gi, fi] = out["loo_rms"]
                kap_folds[gi, fi] = out["log_kappa"]
                last_pool = out  # keep a representative pool for injection
        sel = one_se_selection(grid, loo_folds, kap_folds, kappa_ceiling_log10)

        # LOO / kappa at the selected l2 (mean over folds).
        i_star = int(np.argmin(np.abs(grid - sel.l2_star)))
        loo_star = float(loo_folds[i_star].mean())
        kap_star = float(np.median(kap_folds[i_star]))

        hist.n_support.append(int(state.Z.shape[0]))
        hist.l2_star.append(sel.l2_star)
        hist.loo_at_star.append(loo_star)
        hist.log_kappa_at_star.append(kap_star)
        hist.selections.append(sel.as_dict())

        if verbose:
            print(f"[round {rnd}] N={state.Z.shape[0]:5d}  "
                  f"l2*={sel.l2_star:.3e} (argmin {sel.l2_argmin:.3e})  "
                  f"LOO-RMS={loo_star:.4e}  log10κ={kap_star:.2f}")

        if hist.converged():
            if verbose:
                print(f"[converged] l2* fixed point reached at round {rnd}.")
            break
        if rnd < n_rounds - 1:
            # Inject where the (representative) pool is most uncertain.
            pool_out = fit_score_fn(sel.l2_star, base_seed + rnd, state)
            idx = select_injection_points(
                pool_out["cand_Z"], pool_out["cand_acq"], state.Z,
                k=k_inject, length_scales=pool_out.get("length_scales"))
            state = inject_fn(state, pool_out["cand_Z"][idx])

    return hist.l2_star[-1], hist


# ===========================================================================
# Synthetic torch-free adapter  (for --self_test / --demo and unit tests)
# ===========================================================================

def _rbf_gram(A: FloatArray, B: FloatArray, ell: FloatArray, sf2: float) -> FloatArray:
    A = A / ell
    B = B / ell
    d2 = (np.sum(A*A, 1)[:, None] + np.sum(B*B, 1)[None, :] - 2.0 * A @ B.T)
    return sf2 * np.exp(-0.5 * np.clip(d2, 0.0, None))


def make_synthetic_scorer(D: int = 2, sf2: float = 1.0, ell: float = 0.5,
                          sigma_n2_base: float = 1e-8, noise: float = 0.03,
                          target=None):
    """
    A torch-free stand-in for ``GPDensityScorer`` that mimics the interface:
    an RBF GP on a signed target.  ``l2`` here acts as an added Gram-diagonal
    ridge (a legitimate regularizer in this toy), so the loop's behaviour can be
    validated end-to-end without the heavy stack.
    """
    ell_v = np.full(D, ell)
    if target is None:
        target = lambda Z: np.sin(2.0 * Z[:, 0]) * np.exp(-0.25 * np.sum(Z**2, 1))

    def sample_cloud(n, seed):
        rng = np.random.default_rng(seed)
        Z = rng.uniform(-2.0, 2.0, size=(n, D))
        y = target(Z) + noise * rng.standard_normal(n)
        return Z, y

    def fit_score_fn(l2, fold_seed, state: SupportState):
        Z = state.Z
        rng = np.random.default_rng(fold_seed)
        y = target(Z) + noise * rng.standard_normal(Z.shape[0])
        K = _rbf_gram(Z, Z, ell_v, sf2)
        C = K + (sigma_n2_base + l2) * np.eye(Z.shape[0])
        rms = loo_rms(C, y)
        lk = log10_condition_number(C)
        # candidate pool + acquisition = posterior std
        cand, _ = sample_cloud(400, fold_seed + 7)
        Kss = sf2 * np.ones(cand.shape[0])
        Ks = _rbf_gram(cand, Z, ell_v, sf2)
        v = np.linalg.solve(C, Ks.T)
        var = np.clip(Kss - np.sum(Ks * v.T, axis=1), 0.0, None)
        return dict(loo_rms=rms, log_kappa=lk, cand_Z=cand,
                    cand_acq=np.sqrt(var), length_scales=ell_v)

    def inject_fn(state: SupportState, new_Z):
        return SupportState(Z=np.vstack([state.Z, new_Z]), extra=state.extra)

    return sample_cloud, fit_score_fn, inject_fn


def run_self_test(verbose: bool = True) -> Dict:
    """End-to-end synthetic run of the full selection + injection loop."""
    sample_cloud, fit_score_fn, inject_fn = make_synthetic_scorer()
    Z0, _ = sample_cloud(80, seed=1)
    state0 = SupportState(Z=Z0)
    l2_grid = np.logspace(-10, -2, 9)
    l2_star, hist = resampling_injection_loop(
        fit_score_fn, inject_fn, state0, l2_grid,
        n_rounds=4, n_folds=5, k_inject=40, kappa_ceiling_log10=10.0,
        base_seed=0, verbose=verbose)
    if verbose:
        print("\nfinal l2* =", l2_star)
    return {"l2_star": l2_star, "history": hist.as_dict()}


# ===========================================================================
# Production adapter  (requires PyTorch + the pipeline GP stack)
# ===========================================================================

def make_gpdensity_scorer(
    R0: float = 1.2, P0: float = 8.0, sigma_R: float = 1.0,
    n_support: int = 300, n_opt_steps: int = 60,
):
    """
    Build a ``fit_score_fn`` / ``inject_fn`` backed by the real ``GPDensity``.

    Requires torch (imported lazily).  LOO-RMS and log₁₀ κ are recomputed from
    the fitted GP's Cholesky factor ``_L_Ky`` and weight vector ``_alpha`` using
    the same closed form as ``GPDensity._loo_cv_loss``; the injection acquisition
    is the posterior std from ``predict_with_variance``.
    """
    import torch  # noqa: F401  (fail fast with a clear error if absent)
    from .Sampling import GaussianWavePacketParams, MappingInitParams, MMSTSampler
    from .Mint import PBMEMIntParams, PBMEMIntDynamics, pack_z
    from .Models import TullyModel, TullyParams
    from .GP_Density import GPDensity, GPDensityConfig

    model = TullyModel(TullyParams.defaults("dual"))
    dynamics = PBMEMIntDynamics(model=model, params=PBMEMIntParams(mass=2000.0, hbar=1.0))
    sampler = MMSTSampler(
        GaussianWavePacketParams(R0=[R0], P0=[P0], sigma_R=[sigma_R], hbar=1.0),
        MappingInitParams(nstates=2, init_state=0, hbar=1.0, gamma=0.5))

    def _sample(n, seed):
        s = sampler.sample_seo_signed(n_samples=int(n), rng=np.random.default_rng(seed))
        return pack_z(s.R, s.P, s.r, s.p), np.asarray(s.target_density, float)

    def _loo_and_kappa(gp):
        import numpy as _np
        L = gp._L_Ky.detach().cpu().numpy()
        alpha = gp._alpha.detach().cpu().numpy().reshape(-1)
        Linv = _solve_lower(L, _np.eye(L.shape[0]))
        Cinv_diag = _np.clip(_np.sum(Linv * Linv, axis=0), 1e-30, None)
        resid = alpha / Cinv_diag
        rms = float(_np.sqrt(_np.mean(resid * resid)))
        C = L @ L.T
        lk = log10_condition_number(C)
        return rms, lk

    def fit_score_fn(l2, fold_seed, state: SupportState):
        Z = state.Z
        _, y = _sample(Z.shape[0], fold_seed)   # resample labels on this support
        cfg = GPDensityConfig(
            n_opt_steps=n_opt_steps, l2_regularization=float(l2),
            fix_sigma_n=True, init_log_sigma_n=-8.0,
            reinit_lengthscales=True, constraints_enabled=False,
            interpolate_targets=False)
        gp = GPDensity(cfg, dynamics=dynamics)
        gp.fit(Z_train=Z, y_train=y, moment_targets={},
               optimize=(n_opt_steps > 0), apply_constraints=False)
        rms, lk = _loo_and_kappa(gp)
        cand, _ = _sample(400, fold_seed + 7)
        _, var = gp.predict_with_variance(cand)
        length_scales = np.exp(gp.log_lengthscales.detach().cpu().numpy())
        return dict(loo_rms=rms, log_kappa=lk, cand_Z=np.asarray(cand, float),
                    cand_acq=np.sqrt(np.clip(np.asarray(var, float), 0, None)),
                    length_scales=length_scales)

    def inject_fn(state: SupportState, new_Z):
        return SupportState(Z=np.vstack([state.Z, np.asarray(new_Z, float)]),
                            extra=state.extra)

    Z0, _ = _sample(n_support, seed=12345)
    return SupportState(Z=Z0), fit_score_fn, inject_fn


def run_production(args) -> Dict:
    state0, fit_score_fn, inject_fn = make_gpdensity_scorer(
        R0=args.R0, P0=args.P0, sigma_R=args.sigma_R,
        n_support=args.n_support, n_opt_steps=args.n_opt_steps)
    l2_grid = np.logspace(args.l2_log_min, args.l2_log_max, args.l2_points)
    l2_star, hist = resampling_injection_loop(
        fit_score_fn, inject_fn, state0, l2_grid,
        n_rounds=args.rounds, n_folds=args.folds, k_inject=args.inject,
        kappa_ceiling_log10=args.kappa_ceiling, base_seed=args.seed, verbose=True)
    result = {"l2_star": l2_star, "history": hist.as_dict()}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"[saved] {args.out}")
    print(f"\nRECOMMENDED l2_regularization = {l2_star:.3e}")
    return result


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true",
                   help="Run the torch-free synthetic loop and exit.")
    p.add_argument("--demo", action="store_true", help="Alias for --self_test.")
    p.add_argument("--production", action="store_true",
                   help="Run the real GPDensity-backed selection (requires torch).")
    p.add_argument("--l2_log_min", type=float, default=-10.0)
    p.add_argument("--l2_log_max", type=float, default=-2.0)
    p.add_argument("--l2_points", type=int, default=9)
    p.add_argument("--rounds", type=int, default=4, help="Injection rounds.")
    p.add_argument("--folds", type=int, default=5, help="Resampled clouds per l2.")
    p.add_argument("--inject", type=int, default=50, help="Points injected per round.")
    p.add_argument("--kappa_ceiling", type=float, default=10.0,
                   help="Max allowed log10 condition number.")
    p.add_argument("--n_support", type=int, default=300)
    p.add_argument("--n_opt_steps", type=int, default=60)
    p.add_argument("--R0", type=float, default=1.2)
    p.add_argument("--P0", type=float, default=8.0)
    p.add_argument("--sigma_R", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    if args.production:
        run_production(args)
    else:
        # default to the self-test if nothing else requested
        run_self_test(verbose=True)


if __name__ == "__main__":
    main()
