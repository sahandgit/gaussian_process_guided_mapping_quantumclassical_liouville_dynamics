from __future__ import annotations

"""
GP_Density.py
=============

Differentiable Wigner-density surrogate on the 6D MMST phase space

    z = (R, P, r0, r1, p0, p1)

with physical moment constraints enforced via a KKT / Schur-complement
projection applied on top of the unconstrained GP fit.

Intended use inside the pipeline:

    from .Sampling import MMSTSampler
    from .Mint import PBMEMIntDynamics, pack_z
    from .GP_Density import GPDensity, GPDensityConfig

    # 1. Draw support points (z0, y0) at t = 0 from the exact sampler.
    sampler = MMSTSampler(classical_params, mapping_params)
    s       = sampler.sample_seo_signed(n_samples=n, rng=rng)
    Z0      = pack_z(s.R, s.P, s.r, s.p)
    y0      = s.target_density           # signed Wigner values

    # 2. Evaluate the initial-ensemble energy under the signed target
    #    using the proposal-sampling importance weights.
    E0 = float(np.dot(s.weight, dynamics.energy(Z0)) / np.sum(s.weight))

    # 3. Fit the GP at t = 0 (hyperparameters + KKT correction).
    gp = GPDensity(GPDensityConfig(), dynamics=dynamics)
    gp.fit(Z_train=Z0, y_train=y0,
           moment_targets={"normalization": 1.0,
                           "trace":         1.0,
                           "energy":        E0})

    # 4. At any later time, propagate support points with MInt and refit.
    Z_k = dynamics.propagate(Z0, dt=0.5, n_steps=k)[-1]
    gp.refit(Z_train=Z_k, y_train=y0,
             moment_targets={"normalization": 1.0,
                             "trace":         1.0,
                             "energy":        E0})

    # 5. Predict the density at arbitrary query points.
    rho_hat = gp.predict(Z_query)

Design notes
------------
* ARD-RBF kernel, ZeroMean.  Signed targets are handled natively.
* Hyperparameters (sigma_f, lengthscales, sigma_n) are log-parameterized.
  The initial fit uses full-batch L-BFGS on MLL or LOO-CV; adaptive refits
  use bounded, transactional projected L-BFGS.
  MAD-based lengthscale initialization (one lengthscale per coordinate
  block if requested).
* Moment integrals exploit the ARD tensor product:
      – the mapping (P, r, p) dimensions are done analytically
        (Gaussian × polynomial moments),
      – the nuclear R dimension is done with Gauss–Hermite quadrature
        against V0(R) and h_{αβ}(R), which are supplied by the
        TullyModel owned by PBMEMIntDynamics.
* KKT projection uses the noise-aware metric K_y = K + σ_n² I so that
  the constrained solution is the closest (in the GP posterior norm)
  to the unconstrained α_0 = K_y^{-1} y.
"""

# --- UTF-8 console safety: prevent UnicodeEncodeError on Windows cp1252 ---
# Banners/diagnostics below print non-ASCII physics notation (α, ρ̂, Δ, →, ħ).
# Reconfigure the console streams to UTF-8 so direct execution of this module
# does not abort under Windows' default cp1252 encoding.  No-op where unsupported.
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
# --------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Literal

import copy

import numpy as np
from numpy.typing import ArrayLike, NDArray
from numpy.polynomial.hermite import hermgauss

import torch
from torch import Tensor

from .Models import TullyModel
from .Mint import D, PBMEMIntDynamics


FloatArray = NDArray[np.float64]
MomentName = Literal["normalization", "trace", "energy"]

_DEFAULT_DTYPE = torch.float64
_DEFAULT_DEVICE = torch.device("cpu")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GPDensityConfig:
    """
    Hyperparameter + optimizer settings for the density GP.

    This configuration is now geared toward kernel ridge regression rather than
    near-interpolation.  The production path is:
      * finite diagonal noise is always present in Ky,
      * σ_n is trainable by default,
      * lengthscales are updated by optimization rather than MAD refits,
      * early stopping is driven by a deterministic validation split,
      * an optional L2 penalty regularizes the log-hyperparameters.
    """
    init_log_sigma_f: float = 0.0
    init_log_lengthscales: Optional[ArrayLike] = None
    # Larger starting diagonal noise so the fit begins in the ridge-regression
    # regime instead of near-exact interpolation.
    init_log_sigma_n: float = -2.5
    jitter: float = 1.0e-6

    n_gh: int = 16
    lr: float = 1e-3
    n_opt_steps: int = 400
    mll_tol: float = 1.0e-7
    # Production default: allow σ_n to adapt with the geometry.
    fix_sigma_n: bool = False

    # Keep the flag for backward compatibility, but do not reinitialize
    # lengthscales from MAD in the production pipeline.
    reinit_lengthscales: bool = False
    constraint_ridge: float = 1.0e-10

    log_ls_floor: float = -2.0
    log_ls_ceiling: float = 2.0
    log_sn_floor: float = -8.0
    log_sn_ceiling: float = 1.0

    # RNS (target normalization) is removed.  Targets are always used in physical
    # units.  Set fix_sigma_n=True and choose init_log_sigma_n to achieve the
    # desired SNR: init_log_sigma_n = log(σ_n_physical) e.g. -10 for σ_n≈4.5e-5.

    feature_zscore: bool = False
    recompute_feature_zscore: bool = False

    # Keep for backward compatibility; no longer used in the production path.
    interpolate_targets: bool = False
    fixed_interp_noise: float = 1.0e-12

    # Optimizer-side stabilization / regularization.
    validation_fraction: float = 0.10
    min_validation_points: int = 32
    validation_split_seed: int = 12345
    early_stop_patience: int = 30
    early_stop_min_delta: float = 1.0e-6
    l2_regularization: float = 1.0e-6

    # -------------------------------------------------------------------------
    # Hyperparameter-optimization objective
    # -------------------------------------------------------------------------
    # use_loocv = False (default): optimize marginal log-likelihood (MLL).
    #   Fast, analytically justified, but has a well-known oversmoothing bias:
    #   MLL prefers large lengthscales because larger ℓ increases the entropy
    #   term, "explaining" variance as noise.  In breathing refits this drives
    #   monotone ℓ growth → flat density → R² collapse.
    #
    # use_loocv = True (recommended): optimize leave-one-out cross-validation
    #   loss  L_LOO = (1/N) Σ_i (α_i / [K_y^{-1}]_{ii})².
    #   This DIRECTLY minimizes the training-set prediction error and has NO
    #   oversmoothing bias: the optimizer is penalized for predictions that
    #   deviate from each y_i, so it cannot grow ℓ for free.  Consequence:
    #   R² on the training set stays near 1 throughout dynamics as long as the
    #   kernel is expressive enough to represent ρ at the current cloud geometry.
    #   Same O(N³) cost as MLL; can be used for both initial fit and breathing.
    use_loocv: bool = False

    constraints_enabled: bool = True

    # -------------------------------------------------------------------------
    # Refit-time hyperparameter policy
    # -------------------------------------------------------------------------
    # Three policies are supported and orthogonal to `fix_sigma_n`:
    #
    #   "frozen"     (DEFAULT): lock (σ_f, ℓ, σ_n) at the initial fit
    #                values; every refit only rebuilds Ky and solves for α.
    #                Low cost.  The kernel cannot track the support cloud's
    #                geometric evolution, so fit quality eventually degrades
    #                as the cloud phase-mixes far from its initial shape.
    #                In smoke tests for the Tully dual model this is the
    #                strongest baseline for up to t ~ 50 with dt=1.
    #
    #   "breathing"  (opt-in):  lock (σ_f, σ_n) but allow lengthscales to
    #                breathe with the cloud geometry, subject to a quadratic
    #                shrinkage prior centered on the t=0 lengthscales.  Use
    #                this when the frozen fit visibly degrades (R² falling
    #                below ~0.99) in long runs — the breathing refit tracks
    #                legitimate geometric evolution without letting MLL
    #                drift ℓ monotonically outward.
    #
    #                IMPORTANT: breathing is only useful if the prior weight
    #                is strong enough to resist MLL's preference for larger
    #                lengthscales (the oversmoothing tendency).  w=1 is too
    #                weak and produces monotonic drift.  The default w=10
    #                is empirically stable for n_train ~ 200-500; larger
    #                n_train may need a proportionally larger weight.
    #
    #   "free"       (diagnostic): re-optimize every log-hyperparameter at
    #                every refit.  Expensive and susceptible to MLL-driven
    #                oversmoothing over long runs — diagnostic use only.
    #
    #   "adaptive"   (recommended for long focused-mode runs through avoided
    #                crossings): a CHEAP wrapper around "frozen" that only
    #                triggers breathing optimization when the current ℓ
    #                can no longer interpolate the labels to machine
    #                precision.  Algorithm:
    #
    #                  1. Skip the optimizer entirely (zero work).
    #                  2. Rebuild K_y at current ℓ and solve α₀.
    #                  3. Measure the unconstrained training-fit RMS.
    #                  4. If RMS ≤ adaptive_fit_rms_target: done.  This is
    #                     the common case after a step where the cloud
    #                     simply translated (Liouville flow does not
    #                     change the kernel's representational ability).
    #                  5. If RMS > target: invoke `_breathing_optimize_
    #                     lengthscales` with `refit_opt_steps` iterations.
    #                     Anchored to current ℓ, prior-shrinkage protected.
    #
    #                Empirically on 2400-step Tully focused: ~99% of steps
    #                take the no-op branch (microseconds); the ~1% during
    #                crossing where the cloud bifurcates trigger a brief
    #                breathing refit and recover machine-precision fit.
    #                Total wall cost <2× frozen, but kernel adapts when
    #                physics demands.  This is the recommended policy for
    #                long-time QCLE runs.
    #
    # `refit_opt_steps` is the bounded L-BFGS outer-iteration budget used at each refit under
    # "breathing", "adaptive" (when triggered), or "free"; the initial
    # fit always uses `n_opt_steps`.
    refit_hyper_policy: str = "breathing"     # "frozen" | "breathing" | "adaptive" | "free"
    refit_opt_steps: int = 25
    # Adaptive policy's fit-quality trigger.  If `last_free_fit_rms` after
    # a frozen refit exceeds this threshold, a breathing optimization is
    # invoked to recover sub-threshold fit.  Default 1e-6 corresponds to
    # ~6 decimal digits of interpolation accuracy — comfortably above
    # the ~1e-9 LOO-CV-converged baseline but tight enough to detect
    # genuine kernel-cloud mismatch (e.g. wavepacket bifurcation at an
    # avoided crossing) before it corrupts third-derivative-based Q.
    adaptive_fit_rms_target: float = 1.0e-6
    # Adaptive policy's cloud-spread trigger.  The fit_rms target above
    # detects label-interpolation failure at support points, but an
    # ARD-RBF kernel can interpolate sparse labels perfectly even when
    # its third derivatives BETWEEN support points have become wild
    # (this is what corrupts Q during/after avoided crossings — the
    # cloud bifurcates into multiple lobes separated by ≫ℓ, and the
    # kernel's high-order derivatives oscillate between them while
    # fit_rms at the lobe centers stays at machine precision).
    #
    # The cloud-spread trigger catches this directly: on each
    # informative (non-pinned) axis d, compute the ratio
    #     ratio_d = Var(Z[:, d]) / ℓ_d²
    # If any ratio exceeds adaptive_cloud_ratio_target, the kernel
    # bandwidth has fallen behind the cloud's geometric spread —
    # trigger breathing.  Default 4.0 corresponds to cloud std ≈ 2 ℓ_d,
    # which is the regime where ARD-RBF derivative quality starts to
    # degrade meaningfully.  Pinned axes (e.g. focused mode's mapping
    # axes whose variance is locked by the Casimir geometry) are
    # excluded from the check because the pin gradient mask would
    # suppress any breathing on them anyway.
    adaptive_cloud_ratio_target: float = 4.0
    # Per-trigger LBFGS budget for the adaptive policy.  Distinct from
    # `refit_opt_steps`, which is the budget for the unconditional
    # breathing policy.  Adaptive may fire every step during a crossing,
    # so each burst is cheap (5 projected outer LBFGS updates, with one
    # gradient and one post-projection validation per update).  Five updates
    # are enough to move ℓ_d materially in log
    # space per refit, which empirically tracks the cloud spread
    # evolution at dt=0.5 around the Tully dual crossing.
    adaptive_opt_steps: int = 5
    # Minimum number of refits between two adaptive triggers.  Even when
    # the cloud-ratio trigger condition holds continuously (e.g. through
    # the entire post-crossing region), breathing optimization is
    # invoked at most once every `adaptive_cooldown` refits.  This caps
    # the per-run wall cost: with cooldown=20 and dt=0.5, breathing
    # fires every ~10 time-units when triggered, which is much shorter
    # than the cloud's own bifurcation timescale (~100 time-units for
    # the Tully model).  Set to 1 to fire every step (expensive); set
    # to 0 to disable adaptive (equivalent to "frozen").
    adaptive_cooldown: int = 20
    # Shrinkage prior on lengthscales for "breathing" mode:
    #     L = nll/N + w · mean((log ℓ − log ℓ_0)²)
    # and a hard trust-region clip  |log ℓ − log ℓ_0| ≤ clip  applied after
    # every projected L-BFGS step.
    #
    # Tuning history:
    #   w=10, clip=0.5  — too tight: on the Tully dual-crossing model the
    #                     lengthscales did not move at all over 1000 steps
    #                     (verified from NPZ: ℓ identical to 6 decimals),
    #                     so the cloud's 2.5× P-spread during the crossing
    #                     was not tracked and 1D marginals faded after the
    #                     wavepacket left the initial-fit region.
    #   w=1.0, clip=1.0 — current default: allows ℓ to grow/shrink by a
    #                     factor of ~e in either direction before being
    #                     clipped, with the nll term dominant enough that
    #                     the cloud spread is actually tracked.
    #   w=0, clip=∞    — equivalent to "free" mode, leads to MLL-driven
    #                     oversmoothing over long runs.
    lengthscale_prior_weight: float = 1.0
    lengthscale_prior_clip: float = 1.0
    # When True, breathing updates ONLY nuclear lengthscales ℓ_R and ℓ_P
    # (dims 0, 1); mapping lengthscales ℓ_{r0}, ℓ_{r1}, ℓ_{p0}, ℓ_{p1}
    # (dims 2-5) are pinned at the initial-fit anchor.
    #
    # HISTORICAL RATIONALE (kept here as a warning, not a justification):
    # the previous default was True with the argument that MInt conserves
    # the mapping Casimir r²+p² exactly so the mapping cloud spread never
    # changes.  That argument is incorrect in two ways:
    #   (1) the distribution over (r_α, p_α) is NOT the same as its
    #       aggregate radius — the shell rotates under coupled dynamics,
    #       and the optimal per-axis ℓ_{r_α}, ℓ_{p_α} tracks the
    #       projected-shell marginal geometry, which does change;
    #   (2) finite-dt MInt has numerical drift in the mapping sector.
    # Empirically, running with breathe_nuclear_only=True across an
    # 800-step run of Tully-dual showed the full 6-vector of ℓ_d frozen
    # to displayed precision while the cloud spread (especially in P)
    # grew by 3.6×, forcing ‖α‖ to absorb all the geometry change and
    # producing 130–160% step-to-step jumps in the reconstructed ρ̂(z).
    # Setting breathe_nuclear_only=False lets all six ℓ_d track the
    # cloud; the Casimir-conservation concern is addressed by the
    # shrinkage prior and trust-region clip instead.
    breathe_nuclear_only: bool = False

    # -------------------------------------------------------------------------
    # Breathing prior anchor — STATIC vs CLOUD-TRACKING
    # -------------------------------------------------------------------------
    # The breathing optimizer minimizes
    #     L(log ℓ) = LOOCV(log ℓ) + w · ‖log ℓ − log ℓ_anchor‖² / D'
    # where D' is the number of free dims.  The choice of `log ℓ_anchor`
    # determines whether breathing TRACKS cloud-geometry evolution or
    # locks lengthscales to their initial-fit values.
    #
    # Three policies are supported:
    #   "initial"     : log ℓ_anchor = log ℓ_0  (the lengthscales at first fit).
    #                   Legacy behavior.  Stable but causes fit_rms to grow
    #                   monotonically when the cloud broadens — the prior
    #                   actively prevents lengthscales from growing with the
    #                   data, even if LOOCV would prefer larger ℓ.  Observed
    #                   in long PBME runs near avoided crossings: fit_rms
    #                   creeping up while reg stays at ~1e-10 and ℓ stays
    #                   pinned to t=0 values.
    #
    #   "cloud_mad"   : log ℓ_anchor[d] = log(κ · MAD_d) where MAD_d is the
    #                   robust scale estimate (1.4826·median |z_d - median(z_d)|)
    #                   of the CURRENT support cloud, and κ is a heuristic
    #                   bandwidth factor (default 0.4 — standard for ARD-RBF
    #                   regression on Gaussian-like data).  The anchor moves
    #                   step-by-step with the cloud.  Best fidelity but can
    #                   be noisy if the cloud has occasional outliers.
    #
    #   "ewma"        : exponentially-weighted moving average of cloud_mad,
    #                   initialized at log ℓ_0 (so behaviour at t=0 matches
    #                   "initial").  As the cloud evolves, the anchor
    #                   smoothly tracks the new geometry with time constant
    #                   ≈ 1/(1−β) ≈ 10 refits at the default β=0.9.  This
    #                   is the recommended default: it tracks long-term
    #                   geometry change without reacting to single-step
    #                   sampling jitter.
    #
    # The cloud-MAD computation reads `Z_train` as the optimizer sees it,
    # i.e. the NORMALIZED training data when feature_zscore=True (so MAD
    # is dimensionless and the anchor lives in the same normalized log-
    # space as `log_lengthscales`).  When feature_zscore=False, MAD is
    # in physical units and the anchor is too.  Either way the anchor
    # has the same units as the parameter it shrinks toward.
    breathing_anchor_policy: str = "ewma"
    breathing_anchor_mad_factor: float = 0.4
    breathing_anchor_ewma_beta: float = 0.9

    # -------------------------------------------------------------------------
    # Per-dimension prior weight (default: scalar prior_weight applied to all)
    # -------------------------------------------------------------------------
    # The breathing prior `w · ‖log ℓ - log ℓ_anchor‖² / D_free` shrinks
    # log ℓ toward the anchor.  For some problems the right regularization
    # pressure is the same on every dimension (default behavior); for
    # others — especially Tully-like dynamics where the wavepacket enters
    # an avoided crossing — the nuclear dimensions need to resolve fine
    # de-Broglie-scale oscillations (ξ_R = 2π/P_typical) while the mapping
    # dimensions only see smooth Gaussian-shaped clouds.
    #
    # If set, ``lengthscale_prior_weight_per_dim`` is a length-D array that
    # OVERRIDES the scalar ``lengthscale_prior_weight`` per dimension.
    # Suggested values for Tully-like dynamics with P ≈ 30 a.u.:
    #
    #     prior_weight_per_dim = [0.01, 0.01, 0.1, 0.1, 0.1, 0.1]
    #
    # i.e. 10× looser regularization on (R, P) than on (r₀, r₁, p₀, p₁).
    # The mapping-dim Casimir conservation argument keeps those tight; the
    # nuclear dims are free to pull ℓ down toward whatever LOOCV prefers,
    # which in the de-Broglie regime is ξ_R/2 ≈ 0.1 a.u. (well below the
    # cloud-MAD anchor of κ·σ_R ≈ 0.4 with κ=0.4).
    #
    # When ``None``, falls back to the scalar ``lengthscale_prior_weight``.
    lengthscale_prior_weight_per_dim: Optional[ArrayLike] = None

    # -------------------------------------------------------------------------
    # Mini-batch hyperparameter optimization
    # -------------------------------------------------------------------------
    # When set to a positive integer B, each Adam step in the optimizer loop
    # uses a random subsample of B training points instead of the full N.
    # This reduces per-step Cholesky cost from O(N³) to O(B³) — roughly a
    # (N/B)³ speedup in the loop — while the final alpha solve still uses
    # all N points.
    #
    # Accuracy notes:
    #   • LOO-CV on a mini-batch is an unbiased estimator of pointwise
    #     prediction error on that batch; across steps it covers the full
    #     training set in expectation (stochastic LOO-CV).
    #   • MLL on a mini-batch is stochastic MLL (standard for large GPs).
    #   • Recommended starting point: B = min(N, 256) for N ≥ 500.
    #     Smaller B gives larger speedup but noisier gradients.
    # Set to None (default) to use full-batch optimization (original behavior).
    mini_batch_size: Optional[int] = None


# =============================================================================
# Small helpers
# =============================================================================

def _as_tensor(x: ArrayLike) -> Tensor:
    arr = np.asarray(x, dtype=np.float64)
    if not arr.flags.writeable:
        arr = arr.copy()
    return torch.as_tensor(arr, dtype=_DEFAULT_DTYPE, device=_DEFAULT_DEVICE)


def _as_numpy(t: Tensor) -> FloatArray:
    return t.detach().cpu().numpy().astype(np.float64, copy=False)


def _mad_lengthscales(Z: FloatArray) -> FloatArray:
    """
    Legacy helper retained for backward compatibility.
    Production refits use optimizer-driven updates instead of MAD reinitialization.
    """
    med = np.median(Z, axis=0, keepdims=True)
    mad = np.median(np.abs(Z - med), axis=0)
    ls  = 1.4826 * mad
    ls  = np.where(ls > 1.0e-3, ls, 1.0e-3)
    return ls


def _std_lengthscales(Z: FloatArray) -> FloatArray:
    """Per-dimension standard-deviation initializer for ARD lengthscales."""
    sd = np.std(np.asarray(Z, dtype=np.float64), axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1.0e-3), sd, 1.0)
    return sd.astype(np.float64)


def _zscore_stats(Z: FloatArray) -> Tuple[FloatArray, FloatArray]:
    """Feature-space z-score statistics with safe floors."""
    mu = np.mean(Z, axis=0)
    sd = np.std(Z, axis=0)
    sd = np.where(sd > 1.0e-12, sd, 1.0)
    return mu.astype(np.float64), sd.astype(np.float64)


# =============================================================================
# ARD-RBF kernel (functional, autograd-friendly)
# =============================================================================

def _ard_gram(Z1: Tensor, Z2: Tensor,
              log_sigma_f: Tensor, log_lengthscales: Tensor) -> Tensor:
    """
    K(Z1, Z2)_{ij} = σ_f^2 exp(-0.5 Σ_d (Z1_id - Z2_jd)^2 / ℓ_d^2)

    Z1 shape (N1, D), Z2 shape (N2, D), returns (N1, N2).
    """
    inv_ls = torch.exp(-log_lengthscales)            # (D,)
    Z1s = Z1 * inv_ls                                # (N1, D)
    Z2s = Z2 * inv_ls                                # (N2, D)

    # squared distance via (a-b)^2 = a^2 - 2ab + b^2 for better vectorization
    d2 = (Z1s * Z1s).sum(dim=1, keepdim=True) \
       + (Z2s * Z2s).sum(dim=1, keepdim=True).t() \
       - 2.0 * (Z1s @ Z2s.t())
    d2 = d2.clamp_min(0.0)

    return torch.exp(2.0 * log_sigma_f - 0.5 * d2)


# =============================================================================
# Moment-integral utilities (ARD-RBF over 6D MMST phase space)
# =============================================================================

class _MomentIntegrator:
    """
    Evaluate the linear-functional matrix

        A_{i,j} = ∫ k(z, Z_j) ψ_i(z) dz,   i ∈ {norm, trace, energy}

    The physical trace moment uses the MMST symbols
        c_{aa} = (1/2ℏ)(r_a² + p_a² - ℏ),
    so that
        c_{00} + c_{11} = (1/2ℏ)(r_0²+r_1²+p_0²+p_1² - 2ℏ).

    for the ARD-RBF kernel on z = (R, P, r0, r1, p0, p1).

    The mapping / nuclear-momentum dimensions are done analytically
    (Gaussian × polynomial).  For the default dual Tully model, the R
    dependence is also integrated analytically because V0(R) and h_{αβ}(R)
    are Gaussian functions of R.  For the other Tully models, the R
    integrals fall back to Gauss–Hermite quadrature.

    All per-training-point integrals are batched along the leading axis.
    """

    _SQRT_2PI = float(np.sqrt(2.0 * np.pi))

    def __init__(self,
                 dynamics: PBMEMIntDynamics,
                 n_gh: int = 16) -> None:
        self.dynamics = dynamics
        self.model    = dynamics.model
        self.mass     = float(dynamics.params.mass)
        self.hbar     = float(dynamics.params.hbar)

        # Physicist's Gauss–Hermite: ∫ e^{-x^2} f(x) dx ≈ Σ w_q f(ξ_q).
        xi, wq = hermgauss(int(n_gh))
        self._gh_nodes   = xi.astype(np.float64)
        self._gh_weights = wq.astype(np.float64)

    # --- nuclear potential pieces from the model --------------------------
    def _V0_and_h(self, R: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """
        Given R of shape (...,), return (V0 shape (...), h shape (..., 2, 2))
        with h symmetric traceless (convention matching Mint._trace_traceless).
        """
        V = self.model.diabatic_potential(R)           # (..., 2, 2)
        if V.ndim == 2:                                # scalar R guard
            V = V[None, ...]
        V0 = 0.5 * (V[..., 0, 0] + V[..., 1, 1])
        h = V.copy()
        h[..., 0, 0] -= V0
        h[..., 1, 1] -= V0
        # symmetrize defensively
        h = 0.5 * (h + np.swapaxes(h, -1, -2))
        return V0, h

    # --- exact Gaussian integrals for the dual model -----------------------
    def _gaussian_times_gaussian_integral(self, mu: FloatArray, ell: float, beta: float) -> FloatArray:
        r"""
        J_0(beta; mu, ell) = ∫ exp(-(R-mu)^2/(2 ell^2)) exp(-beta R^2) dR
                           = √(2π) ell / √(1 + 2 beta ell^2)
                             · exp[-beta mu^2 / (1 + 2 beta ell^2)].
        """
        mu = np.asarray(mu, dtype=np.float64)
        denom = 1.0 + 2.0 * float(beta) * float(ell) * float(ell)
        return (self._SQRT_2PI * float(ell) / np.sqrt(denom)) * np.exp(-(float(beta) * mu * mu) / denom)

    def _R_integrals_dual_exact(
        self,
        R_train: FloatArray,
        ell_R: float,
    ) -> Dict[str, FloatArray]:
        r"""
        Exact R-integrals for the dual avoided-crossing model.

        Dual model:
            V11(R) = 0,
            V22(R) = -A exp(-B R^2) + E0,
            V12(R) =  C exp(-D R^2).

        Therefore
            V0(R)  = (E0 - A exp(-B R^2))/2,
            h00(R) = (A exp(-B R^2) - E0)/2 = -V0(R),
            h11(R) = (E0 - A exp(-B R^2))/2 =  V0(R),
            h01(R) = C exp(-D R^2).
        """
        p = self.model.params
        Z = np.full(R_train.shape[0], self._SQRT_2PI * float(ell_R), dtype=np.float64)
        J_B = self._gaussian_times_gaussian_integral(R_train, ell_R, float(p.B))
        J_D = self._gaussian_times_gaussian_integral(R_train, ell_R, float(p.D))

        V0  = 0.5 * float(p.E0) * Z - 0.5 * float(p.A) * J_B
        h00 = -V0
        h11 = V0
        h01 = float(p.C) * J_D
        return {
            "Z":   Z,
            "V0":  V0,
            "h00": h00,
            "h01": h01,
            "h11": h11,
        }

    # --- kernel integrals against V0(R), h_{αβ}(R) over R  ----------------
    def _R_integrals(
        self,
        R_train: FloatArray,
        ell_R: float,
    ) -> Dict[str, FloatArray]:
        """
        For each training R_j, compute

            IR^{f}(R_j) = ∫ exp(-(R-R_j)^2 / (2 ℓ_R^2)) f(R) dR

        for f ∈ {1, V0, h00, h01, h11} via Gauss–Hermite.

        Returns a dict with keys 'Z', 'V0', 'h00', 'h01', 'h11', each
        an array of shape (N,).

        For the dual model these integrals are evaluated in closed form.
        For the simple/extended models they fall back to Gauss–Hermite.
        """
        if getattr(self.model.params, "kind", None) == "dual":
            return self._R_integrals_dual_exact(R_train, ell_R)

        N = R_train.shape[0]
        xi = self._gh_nodes                             # (Q,)
        wq = self._gh_weights                           # (Q,)
        Q  = xi.size
        sqrt2_ell = np.sqrt(2.0) * ell_R

        # R_eval_{j, q} = R_j + sqrt(2) ℓ_R ξ_q
        R_eval = R_train[:, None] + sqrt2_ell * xi[None, :]     # (N, Q)
        V0, h  = self._V0_and_h(R_eval.reshape(-1))              # (N*Q,), (N*Q, 2, 2)
        V0 = V0.reshape(N, Q)
        h  = h.reshape(N, Q, 2, 2)

        prefac = sqrt2_ell                           # √2 ℓ_R
        # Zeroth-moment (no f) = √(2π) ℓ_R  [constant in j; still returned per-j]
        Z_const = self._SQRT_2PI * ell_R
        IR_V0  = prefac * (wq[None, :] * V0).sum(axis=1)
        IR_h00 = prefac * (wq[None, :] * h[..., 0, 0]).sum(axis=1)
        IR_h01 = prefac * (wq[None, :] * h[..., 0, 1]).sum(axis=1)
        IR_h11 = prefac * (wq[None, :] * h[..., 1, 1]).sum(axis=1)

        return {
            "Z":   np.full(N, Z_const, dtype=np.float64),
            "V0":  IR_V0,
            "h00": IR_h00,
            "h01": IR_h01,
            "h11": IR_h11,
        }

    # --- the big A-matrix -------------------------------------------------
    def build_A(
        self,
        Z_train: FloatArray,
        log_sigma_f: float,
        log_lengthscales: FloatArray,
        moments: Tuple[MomentName, ...],
    ) -> FloatArray:
        """
        Build A of shape (len(moments), N).  Rows are ordered to match `moments`.

        Dimension ordering of lengthscales follows z = (R, P, r0, r1, p0, p1).
        """
        assert Z_train.ndim == 2 and Z_train.shape[1] == D, \
            f"Z_train must have shape (N, {D}); got {Z_train.shape}"

        N = Z_train.shape[0]
        ls = np.exp(log_lengthscales).astype(np.float64)
        ell_R, ell_P = ls[0], ls[1]
        ell_r0, ell_r1, ell_p0, ell_p1 = ls[2], ls[3], ls[4], ls[5]
        sigma2_f = float(np.exp(2.0 * log_sigma_f))

        # 1D marginal norms Z_d = √(2π) ℓ_d and base product
        Z_R  = self._SQRT_2PI * ell_R
        Z_P  = self._SQRT_2PI * ell_P
        Z_r0 = self._SQRT_2PI * ell_r0
        Z_r1 = self._SQRT_2PI * ell_r1
        Z_p0 = self._SQRT_2PI * ell_p0
        Z_p1 = self._SQRT_2PI * ell_p1
        G    = Z_R * Z_P * Z_r0 * Z_r1 * Z_p0 * Z_p1                  # scalar

        # unpack training points
        R_tr  = Z_train[:, 0]
        P_tr  = Z_train[:, 1]
        r0_tr = Z_train[:, 2]
        r1_tr = Z_train[:, 3]
        p0_tr = Z_train[:, 4]
        p1_tr = Z_train[:, 5]

        # squared-moment shifts M^{(2)}_d(x_j) = x_j^2 + ℓ_d^2
        M2_r0 = r0_tr * r0_tr + ell_r0 * ell_r0
        M2_r1 = r1_tr * r1_tr + ell_r1 * ell_r1
        M2_p0 = p0_tr * p0_tr + ell_p0 * ell_p0
        M2_p1 = p1_tr * p1_tr + ell_p1 * ell_p1
        M2_P  = P_tr  * P_tr  + ell_P  * ell_P

        # R-dependent integrals (only needed if energy is requested)
        if "energy" in moments:
            R_ints = self._R_integrals(R_tr, ell_R)

        rows: List[FloatArray] = []
        for m in moments:
            if m == "normalization":
                # ψ = 1  ⇒  A_j = σ_f² G  (constant in j)
                row = np.full(N, sigma2_f * G, dtype=np.float64)

            elif m == "trace":
                # ψ(z) = c_00 + c_11
                #      = (1/2ℏ)(r0² + r1² + p0² + p1² - 2ℏ)
                trace_core = (
                    (M2_r0 + M2_r1 + M2_p0 + M2_p1) / (2.0 * self.hbar)
                    - 1.0
                )
                row = sigma2_f * G * trace_core

            elif m == "energy":
                # H(z) = P²/(2M) + V0(R) + (1/(2ℏ)) [r^T h(R) r + p^T h(R) p]
                # — kinetic piece:
                #   ∫ k(z,z_j) P²/(2M) dz = (σ_f² / 2M) · (G/Z_P) · (Z_P · M2_P)
                #                         = (σ_f² / 2M) · G · M2_P
                A_kin = sigma2_f * G * (M2_P / (2.0 * self.mass))

                # — V0(R) piece (replace Z_R by Z_R^{V0})
                G_no_R = G / Z_R
                A_V0   = sigma2_f * G_no_R * R_ints["V0"]

                # — electronic piece (r^T h r + p^T h p)
                # mapping moments for each (α,β) block
                # Use the PHYSICAL mapping radii r_α² + p_α² (no kernel
                # bandwidth ℓ_r² added), not M2_rα = r_α² + ℓ_rα².
                #
                # Rationale: the energy H contains (r^T h r + p^T h p)/(2ℏ),
                # whose kernel integral is ∫k(z,Z_j)(r0²+p0²)dz = σ_f²G(r0j²+ℓ_r0²).
                # For focused mode the physical density is delta-supported
                # on two circles with DIFFERENT radii (R_active≠R_inactive),
                # so ℓ_active≠ℓ_inactive.  The asymmetric bandwidth shift
                # (ℓ_r0²+ℓ_p0²) − (ℓ_r1²+ℓ_p1²) = 2(ℓ_active²−ℓ_inactive²) ≠ 0
                # breaks the exact cancellation A_V0+A_el=0 that the
                # traceless h satisfies at the physical circle radii, giving
                # a spurious energy deficit of ≈ −(ℓ_active²−ℓ_inactive²)V0/(2ℏ)
                # ≈ −0.0125 a.u. for the dual crossing default parameters.
                #
                # Fix: use the physical (bandwidth-free) values so the
                # cancellation is exact and ⟨H⟩_GP = E_phys.
                M_00 = r0_tr * r0_tr + p0_tr * p0_tr   # physical: r0² + p0²
                M_11 = r1_tr * r1_tr + p1_tr * p1_tr   # physical: r1² + p1²
                M_01 = r0_tr * r1_tr + p0_tr * p1_tr   # cross-term (no ℓ² anyway)

                # kernel integrals in the mapping space are G_no_R already ×
                # the corresponding mapping moment, so we multiply by
                # G_no_R / G * G = G_no_R.  We already have σ_f² in the prefactor.
                A_el = (sigma2_f / (2.0 * self.hbar)) * G_no_R * (
                      R_ints["h00"] * M_00
                    + 2.0 * R_ints["h01"] * M_01
                    + R_ints["h11"] * M_11
                )

                row = A_kin + A_V0 + A_el

            else:
                raise ValueError(f"Unknown moment name {m!r}")

            rows.append(row)

        return np.stack(rows, axis=0)


# =============================================================================
# Main GP density class
# =============================================================================

class GPDensity:
    """
    Constrained GP density surrogate on the 6D MMST phase space.

    Usage pattern:
        gp = GPDensity(GPDensityConfig(), dynamics)
        gp.fit(Z0, y0, moment_targets={...})
        ...
        gp.refit(Z_k, y0, moment_targets={...})  # Liouville-preserved y
        rho = gp.predict(Z_query)
    """

    # ----- construction --------------------------------------------------
    def __init__(self,
                 config: GPDensityConfig,
                 dynamics: PBMEMIntDynamics) -> None:

        self.config   = config
        self.dynamics = dynamics
        self.moment_integrator = _MomentIntegrator(dynamics, n_gh=config.n_gh)

        # Hyperparameters as leaf tensors (log-parameterized).
        self.log_sigma_f = torch.tensor(float(config.init_log_sigma_f),
                                        dtype=_DEFAULT_DTYPE, requires_grad=True)
        if config.init_log_lengthscales is None:
            ls_init = np.zeros(D, dtype=np.float64)      # overwritten at fit()
        else:
            ls_init = np.asarray(config.init_log_lengthscales,
                                 dtype=np.float64).reshape(-1)
            if ls_init.size != D:
                raise ValueError(f"init_log_lengthscales must have size {D}.")
        self.log_lengthscales = torch.tensor(ls_init, dtype=_DEFAULT_DTYPE,
                                             requires_grad=True)
        self.log_sigma_n = torch.tensor(float(config.init_log_sigma_n),
                                        dtype=_DEFAULT_DTYPE,
                                        requires_grad=(not config.fix_sigma_n))

        # Regularization anchors for optimizer-side hyperparameter control.
        self._reg_anchor_log_sigma_f = float(config.init_log_sigma_f)
        self._reg_anchor_log_lengthscales = np.asarray(
            config.init_log_lengthscales if config.init_log_lengthscales is not None else np.zeros(D, dtype=np.float64),
            dtype=np.float64,
        ).reshape(-1)
        if self._reg_anchor_log_lengthscales.size != D:
            self._reg_anchor_log_lengthscales = np.zeros(D, dtype=np.float64)
        self._reg_anchor_log_sigma_n = float(config.init_log_sigma_n)

        # Fit state.
        self._Z_train: Optional[Tensor]    = None    # RAW physical training centers (external API)
        self._Z_train_norm: Optional[Tensor] = None  # Internal kernel coordinates
        self._feature_mean: Optional[Tensor] = None
        self._feature_std: Optional[Tensor] = None
        self._y_train: Optional[Tensor]    = None    # physical labels (no scaling applied)
        self._y_raw:   Optional[Tensor]    = None    # alias; kept for API compat
        self._y_scale: float               = 1.0    # always 1.0 — RNS normalization removed
        self._sigma_n_normalized: float    = float(np.exp(2.0*0.0))**0.5
        self._L_Ky:    Optional[Tensor]    = None   # Cholesky factor
        self._alpha:   Optional[Tensor]    = None   # constrained coefficients (physical)
        self._alpha0:  Optional[Tensor]    = None   # unconstrained (physical)
        self._A:       Optional[Tensor]    = None   # moment matrix (m, N)
        self._b:       Optional[Tensor]    = None   # moment targets (m,)
        self._moment_order: Tuple[MomentName, ...] = ()

        # Freeze flags retained for backward compatibility, but the production
        # pipeline no longer calls freeze_hypers().
        self._hypers_frozen: bool = False
        self._feature_stats_frozen: bool = False

        # "Initial-fit anchor" for the breathing refit policy.  After the first
        # fit() completes, we pin (σ_f, σ_n, ℓ)_0 here.  Subsequent refits under
        # refit_hyper_policy="breathing" then:
        #   * hold σ_f at its anchor while σ_n floats unless fix_sigma_n=True,
        #   * allow log ℓ (and, by default, log σ_n) to move in a bounded
        #     L-BFGS solve with a shrinkage prior
        #     ‖log ℓ − log ℓ_0‖² that pulls ℓ back to its initial scale.
        # The policy is set per-refit (see refit(... hyper_policy=...)) so the
        # same GP can be driven in multiple modes from the same state.
        self._initial_fit_done: bool = False
        self._initial_log_sigma_f_anchor: float = float(config.init_log_sigma_f)
        self._initial_log_sigma_n_anchor: float = float(config.init_log_sigma_n)
        self._initial_log_lengthscales_anchor: FloatArray = np.zeros(D, dtype=np.float64)

        # -----------------------------------------------------------------
        # Lengthscale-pinning state (label-information-rank contract).
        # -----------------------------------------------------------------
        # `_pin_mask[d] = True` ⇒ axis d's lengthscale is held fixed at
        # `_pin_log_ls_norm[d]` (stored in the SAME normalized/physical
        # units that `log_lengthscales` lives in) for the entire run.
        # The sampler decides this at construction time via
        # `pin_lengthscales(LabelInformation)`; once pinned, axes stay
        # pinned until `unpin_lengthscales()` is called.
        #
        # Enforcement points (must all respect the pin):
        #   * optimize_hyperparameters: gradient on pinned axes is masked
        #     to zero AFTER backward(); pinned values are also restored
        #     by hand at each outer step (defense-in-depth against any
        #     optimizer state that survived the mask).
        #   * _project_log_hypers_: pinned axes overwritten with the pin
        #     value after every projection (cheap idempotent restore).
        #   * _breathing_optimize_lengthscales: same masking applied.
        #   * _compute_breathing_anchor: anchor for pinned axes IS the
        #     pin value (so the shrinkage term degenerates to zero
        #     gradient on pinned axes; harmless).
        self._pin_mask: NDArray[np.bool_] = np.zeros(D, dtype=bool)
        self._pin_log_ls_norm: FloatArray = np.zeros(D, dtype=np.float64)
        self._pin_anchor_phys: FloatArray = np.full(D, np.nan, dtype=np.float64)
        self._pin_source: str = "none"
        # See pin_lengthscales for purpose; declared here for type stability.
        self._pin_redundant_moments: Tuple[str, ...] = ()
        # Default True for backward compatibility (no pin → behave as before).
        self._pin_apply_kkt: bool = True
        # None → respect cfg.use_loocv; bool overrides.
        self._pin_use_loocv: Optional[bool] = None

        # EWMA-tracked anchor for the breathing prior (used when
        # config.breathing_anchor_policy == "ewma").  Initialized to the
        # first-fit lengthscales by the first call to _compute_breathing_anchor,
        # then exponentially blended toward 0.4·MAD(cloud) at every
        # subsequent refit.  None until first refit; set in
        # _compute_breathing_anchor.
        self._ewma_log_lengthscales_anchor: Optional[FloatArray] = None
        # Last anchor actually used (for diagnostics / inspection).  Stored
        # in the SAME (normalized or physical) units as log_lengthscales.
        self._last_breathing_anchor: Optional[FloatArray] = None

        # Diagnostics: fit quality, optimizer losses, and early-stop state.
        self.last_fit_rms: float = float("nan")
        self.last_fit_mae: float = float("nan")
        self.last_fit_r2:  float = float("nan")
        self.last_free_fit_rms: float = float("nan")
        self.last_free_fit_mae: float = float("nan")
        self.last_free_fit_r2:  float = float("nan")
        self.constraint_delta_rmse: float = float("nan")
        self.constraint_delta_mae: float = float("nan")
        self.constraint_delta_r2:  float = float("nan")
        self.last_opt_total_loss: float = float("nan")
        self.last_opt_nll_loss: float = float("nan")
        self.last_opt_reg_loss: float = float("nan")
        self.last_opt_train_mae: float = float("nan")
        self.last_opt_train_r2: float = float("nan")
        self.last_opt_val_mae: float = float("nan")
        self.last_opt_val_r2: float = float("nan")
        self.last_opt_steps: int = 0
        self.last_opt_best_step: int = -1
        self.last_opt_early_stopped: bool = False
        # Numerical factorization diagnostics. These remain separate from
        # both sigma_n and the scientific L2 regularization parameter.
        self.last_cholesky_adaptive_jitter: float = 0.0
        self.last_cholesky_effective_jitter: float = float(config.jitter)
        self.last_cholesky_attempts: int = 0
        self.last_cholesky_min_eigenvalue: float = float("nan")
        # Breathing-refit safety diagnostics.  A failed adaptive trial is not
        # a failed dynamics step: the entire last-known-good hyperparameter
        # state is restored transactionally and propagation continues.
        self.last_breathing_failed: bool = False
        self.last_breathing_failure_reason: str = ""
        self.last_breathing_failure_code: int = 0
        self.breathing_failure_count: int = 0
        # Diagnostic: number of singular values dropped in the last
        # KKT projection (rank-deficient constraint rows).  Should be 0
        # under healthy operation.
        self._kkt_dropped_rank: int = 0

    # ----- convenience accessors ----------------------------------------
    @property
    def norm_lengthscales(self) -> FloatArray:
        return _as_numpy(torch.exp(self.log_lengthscales))

    @property
    def lengthscales(self) -> FloatArray:
        ell = self.norm_lengthscales
        if self.config.feature_zscore and self._feature_std is not None:
            ell = ell * _as_numpy(self._feature_std)
        return np.asarray(ell, dtype=np.float64)

    @property
    def sigma_f(self) -> float:
        return float(torch.exp(self.log_sigma_f).item())

    @property
    def sigma_n(self) -> float:
        """Noise standard deviation in physical units.
        RNS (target normalization) was removed — y_scale == 1.0 always,
        so sigma_n and sigma_n_normalized are identical.  sigma_n is the
        single source of truth; sigma_n_normalized is kept as a thin alias
        so callers that still reference it (e.g. Dynamics diagnostics) do
        not break, but it is marked deprecated.
        """
        return float(torch.exp(self.log_sigma_n).item())

    @property
    def sigma_n_normalized(self) -> float:
        """Deprecated alias for sigma_n.  y_scale == 1.0 always (RNS removed)."""
        return self.sigma_n

    @property
    def n_train(self) -> int:
        return 0 if self._Z_train is None else int(self._Z_train.shape[0])

    @property
    def raw_training_centers(self) -> FloatArray:
        if self._Z_train is None:
            raise RuntimeError("GP not fitted.")
        return _as_numpy(self._Z_train)

    def _normalize_features_np(self, Z: FloatArray) -> FloatArray:
        Z = np.asarray(Z, dtype=np.float64)
        if not self.config.feature_zscore:
            return Z
        if self._feature_mean is None or self._feature_std is None:
            raise RuntimeError("Feature normalizer not initialized.")
        mu = _as_numpy(self._feature_mean)
        sd = _as_numpy(self._feature_std)
        return (Z - mu) / sd

    def _normalize_features_t(self, Z: Tensor) -> Tensor:
        if not self.config.feature_zscore:
            return Z
        if self._feature_mean is None or self._feature_std is None:
            raise RuntimeError("Feature normalizer not initialized.")
        return (Z - self._feature_mean) / self._feature_std

    # ----- label-information-rank lengthscale pinning -------------------
    def pin_lengthscales(self, label_info: "LabelInformation") -> None:
        """
        Apply a sampler-provided `LabelInformation` contract to this GP.

        Effect
        ------
        Every axis d with `label_info.information_rank[d] < 1.0` is:

          * Set to ``log(anchor_lengthscales[d])`` in normalized kernel
            coordinates (z-score-aware: if `feature_zscore=True`, the
            physical anchor is divided by the per-axis std before taking
            the log; if `feature_zscore=False`, anchor in == anchor out).
          * Excluded from the L-BFGS optimizer's effective gradient via
            a mask applied inside `closure()` after `loss.backward()`.
          * Restored to its anchor value by `_project_log_hypers_` after
            every optimization step (defense in depth against any
            optimizer state that survives the gradient mask).
          * Also restored by the breathing-refit optimizer at every step.

        Axes with rank == 1 remain freely estimated by data — the
        standard GP behavior.

        When called
        -----------
        Typically once, before the initial `fit()`, from
        `Dynamics.build_initial_state`.  The sampler reports its own
        contract via `MMSTSamples.label_information`.

        Idempotence
        -----------
        Calling `pin_lengthscales` again replaces the previous contract
        wholesale; partial pins are not currently supported (one
        contract per GP).  To remove pinning, call `unpin_lengthscales`.

        Diagnostics
        -----------
        The active pin state is queryable via `self.pinned_mask` (alias
        of `self._pin_mask`) and `self.pinned_log_lengthscales`.
        """
        from .Sampling import LabelInformation as _LI   # local import; avoid cycle
        if not isinstance(label_info, _LI):
            raise TypeError(
                f"pin_lengthscales expects a Sampling.LabelInformation; got {type(label_info)}."
            )
        if label_info.D != D:
            raise ValueError(
                f"LabelInformation has dim {label_info.D} but GP expects {D}."
            )

        mask = label_info.pinned_mask.copy()
        anchor_phys = label_info.anchor_lengthscales.copy()

        # Convert PHYSICAL anchors → NORMALIZED kernel units (the space
        # log_lengthscales lives in).  Without feature z-scoring these
        # are equal.  With z-scoring, the kernel sees z/std, so a
        # physical anchor ℓ corresponds to log-kernel anchor log(ℓ/std).
        log_ls_norm = np.zeros(D, dtype=np.float64)
        if self.config.feature_zscore and self._feature_std is not None:
            std = _as_numpy(self._feature_std)
        else:
            std = np.ones(D, dtype=np.float64)

        for d in range(D):
            if mask[d]:
                if not (np.isfinite(anchor_phys[d]) and anchor_phys[d] > 0.0):
                    raise ValueError(
                        f"pin_lengthscales: axis {d} marked pinned but anchor "
                        f"is not finite-positive: {anchor_phys[d]}."
                    )
                log_ls_norm[d] = float(np.log(anchor_phys[d] / std[d]))
            else:
                log_ls_norm[d] = np.nan  # value never read; mask is False

        self._pin_mask         = mask
        self._pin_log_ls_norm  = log_ls_norm
        self._pin_anchor_phys  = anchor_phys
        self._pin_source       = label_info.source
        # Cache the sampler's redundant-moment declaration.  fit() and
        # refit() will silently drop these names from any moment_targets
        # they receive — this avoids rank-deficient KKT systems caused by
        # Casimir invariants (e.g. trace under focused sampling).
        self._pin_redundant_moments: Tuple[str, ...] = tuple(label_info.redundant_moments)
        # Whether this sampler's labels are interpretable as physical
        # density.  False for focused (labels are positive proxy
        # W_cl·K_focus, not ρ_phys); the KKT projection is then skipped
        # entirely and α = α₀ (unconstrained LS solution).  See
        # LabelInformation.apply_kkt docstring for the full rationale.
        self._pin_apply_kkt: bool = bool(label_info.apply_kkt)
        # Optional override of the hyperparameter-optimization loss.
        # None  → respect GPDensityConfig.use_loocv (legacy behavior).
        # True  → force LOO-CV regardless of config.
        # False → force MLL regardless of config.
        # Set from LabelInformation.recommended_loss; both factories
        # default to "loocv" (empirically much better — see docstring).
        if label_info.recommended_loss is None:
            self._pin_use_loocv: Optional[bool] = None
        else:
            self._pin_use_loocv = (label_info.recommended_loss == "loocv")

        # Apply the pin immediately so subsequent fit/refit sees the
        # right starting point.  Also update the "regularization anchor"
        # so the existing L2 reg term doesn't pull pinned axes away
        # from their pin value.
        with torch.no_grad():
            for d in range(D):
                if mask[d]:
                    self.log_lengthscales[d] = float(log_ls_norm[d])
                    self._reg_anchor_log_lengthscales[d] = float(log_ls_norm[d])

        # Runge-pathology guard.  If the pinned lengthscales are SMALLER
        # than the cloud spread on the same axis, the Gram matrix K will
        # be near-diagonal and α = K⁻¹y will oscillate violently between
        # support points (classic RKHS Runge phenomenon).  This produces
        # LOO/fit residual ratios of 10⁴-10⁵ and renders bath marginals
        # noisy.  We compare σ_d = std(Z_d) against the pinned ℓ_d in
        # PHYSICAL units on every pinned axis and warn if σ/ℓ > 0.9.
        # Threshold chosen so a healthy fit (σ/ℓ ≈ 0.47 for ℓ = 1.5 R
        # and circular SEO sampling) stays well below; pathological
        # values (σ/ℓ > 1) trigger.
        if hasattr(self, "_train_X") and self._train_X is not None:
            Z_phys = _as_numpy(self._train_X)
            for d in range(D):
                if not mask[d]:
                    continue
                sigma_d = float(np.std(Z_phys[:, d]))
                ell_d   = float(anchor_phys[d])
                if ell_d > 0 and sigma_d / ell_d > 0.9:
                    print(
                        f"[GP_Density] WARNING: pinned axis {d} is in the "
                        f"Runge-pathology regime: σ(Z_d) = {sigma_d:.4f}, "
                        f"ℓ_pin = {ell_d:.4f}, σ/ℓ = {sigma_d/ell_d:.3f}. "
                        f"Kernel will be near-diagonal; expect α to be "
                        f"signed and LOO≫fit.  Consider increasing the "
                        f"anchor multiplier in LabelInformation.focused_2state."
                    )

    def unpin_lengthscales(self) -> None:
        """Remove the current pinning contract.  All axes become freely learnable."""
        self._pin_mask = np.zeros(D, dtype=bool)
        self._pin_log_ls_norm = np.zeros(D, dtype=np.float64)
        self._pin_anchor_phys = np.full(D, np.nan, dtype=np.float64)
        self._pin_source = "none"
        self._pin_redundant_moments = ()
        self._pin_apply_kkt = True
        self._pin_use_loocv = None

    @property
    def pinned_mask(self) -> NDArray[np.bool_]:
        """Boolean mask: True on axes whose lengthscale is sampler-pinned."""
        return self._pin_mask.copy()

    @property
    def pinned_log_lengthscales(self) -> FloatArray:
        """Anchor log-lengthscales in kernel-internal (normalized) units.
        Entries on free axes are NaN."""
        out = np.full(D, np.nan, dtype=np.float64)
        out[self._pin_mask] = self._pin_log_ls_norm[self._pin_mask]
        return out

    @property
    def pin_source(self) -> str:
        return self._pin_source

    def _filter_redundant_moments(
        self,
        moment_targets: Optional[Dict[MomentName, float]],
    ) -> Optional[Dict[MomentName, float]]:
        """
        Strip any moments that the active pin contract declares as
        redundant Casimirs.

        Background
        ----------
        When the sampler's support geometry locks a Casimir invariant
        (e.g. focused MMST sampling places every cloud point on the
        product of two circles r_α²+p_α² = const, making the trace
        functional c₀₀+c₁₁ a constant across all training points), the
        corresponding KKT moment row becomes a scalar multiple of the
        normalization row.  The Schur complement S = A Ky⁻¹ Aᵀ becomes
        rank-deficient; constraint_ridge "saves" the linear-algebra solve
        but only by returning a solution that is approximately
        ridge × (some garbage direction in the redundant subspace),
        producing the constant norm=0.6 / trace=1.19 pathology.

        The fix is to NOT enforce a constraint that is geometrically
        already satisfied.  This helper makes the filtering automatic
        and silent: each sampler declares its Casimirs in
        LabelInformation.redundant_moments, and every fit / refit
        honors that.
        """
        if moment_targets is None or not self._pin_redundant_moments:
            return moment_targets
        return {k: v for k, v in moment_targets.items()
                if k not in self._pin_redundant_moments}

    def freeze_hypers(self) -> None:
        """
        Lock σ_f, lengthscales, σ_n at their current values.  All subsequent
        refit() calls will then:
          - skip MLL optimization,
          - skip MAD-based lengthscale reinitialization,
          - only recompute the Cholesky factor of Ky on the new Z and solve
            for α (and KKT-project if moment_targets are supplied).

        This is the right behavior during dynamics: the physical density's
        correlation lengths are set by the initial wavepacket, not by the
        geometric dispersion of the support-point cloud as trajectories
        phase-mix.  Re-optimizing MLL at every step causes monotone
        lengthscale growth ("oversmoothing") because MLL sees the cloud
        stretch and prefers to explain the rougher response as smooth-plus-
        noise — exactly the pathology in long FLV runs.

        Important: if feature z-scoring is enabled, the feature statistics
        define part of the kernel map z -> z_norm.  They must therefore be
        frozen together with (sigma_f, ell, sigma_n); otherwise refits with
        recomputed z-score statistics silently alter the physical kernel.
        """
        with torch.no_grad():
            self.log_sigma_f.requires_grad_(False)
            self.log_lengthscales.requires_grad_(False)
            self.log_sigma_n.requires_grad_(False)
        self._hypers_frozen = True
        self._feature_stats_frozen = True

    # ----- kernel / Cholesky helpers ------------------------------------
    def _Ky(self, Z: Tensor) -> Tensor:
        K = _ard_gram(Z, Z, self.log_sigma_f, self.log_lengthscales)
        N = Z.shape[0]
        sn2 = torch.exp(2.0 * self.log_sigma_n)
        return K + (sn2 + self.config.jitter) * torch.eye(N, dtype=_DEFAULT_DTYPE, device=Z.device)

    def _cholesky(self, Ky: Tensor) -> Tensor:
        """Return a stable Cholesky factor without concealing invalid trials.

        A finite RBF covariance is positive semidefinite analytically, but
        round-off can leave a very small negative eigenvalue.  Retry such
        matrices with a monotonically increasing *positive* diagonal jitter.
        Non-finite matrices are optimizer failures, not conditioning events,
        and are rejected explicitly so the caller can restore a safe state.
        """
        if Ky.ndim != 2 or Ky.shape[0] != Ky.shape[1]:
            raise ValueError(f"Ky must be square; got shape {tuple(Ky.shape)}")
        if not bool(torch.isfinite(Ky).all().item()):
            raise RuntimeError("GP covariance Ky contains non-finite values")

        # Eliminate harmless asymmetry accumulated by BLAS before factorizing.
        Ky_sym = 0.5 * (Ky + Ky.transpose(-2, -1))
        N = int(Ky_sym.shape[0])
        eye = torch.eye(N, dtype=Ky_sym.dtype, device=Ky_sym.device)
        diag_scale = torch.mean(torch.abs(torch.diagonal(Ky_sym)))
        scale = max(1.0, float(diag_scale.detach().item()))
        # O(N^2) Gershgorin lower bound used as a conservative minimum-
        # eigenvalue estimate; a full eigendecomposition inside every L-BFGS
        # closure would duplicate the cubic Cholesky cost.
        diag = torch.diagonal(Ky_sym)
        radii = torch.sum(torch.abs(Ky_sym), dim=1) - torch.abs(diag)
        min_eig = float(torch.min(diag - radii).detach().item())
        self.last_cholesky_min_eigenvalue = min_eig

        # The first attempt adds nothing. Subsequent attempts follow the
        # scale-aware reviewer ladder s*10^j, j=-12,...,-3.
        attempted: List[float] = []
        last_info = -1
        for rel in (0.0,) + tuple(10.0 ** j for j in range(-12, -2)):
            eps = rel * scale
            attempted.append(eps)
            trial = Ky_sym if eps == 0.0 else Ky_sym + eps * eye
            L, info = torch.linalg.cholesky_ex(trial, check_errors=False)
            last_info = int(torch.max(info).detach().item())
            if last_info == 0 and bool(torch.isfinite(L).all().item()):
                self.last_cholesky_adaptive_jitter = float(eps)
                self.last_cholesky_effective_jitter = (
                    float(self.config.jitter) + float(eps)
                )
                self.last_cholesky_attempts = len(attempted)
                return L
        self.last_cholesky_adaptive_jitter = float(attempted[-1])
        self.last_cholesky_effective_jitter = (
            float(self.config.jitter) + float(attempted[-1])
        )
        self.last_cholesky_attempts = len(attempted)

        raise RuntimeError(
            "GP covariance remained non-positive-definite after positive "
            f"jitter escalation through {attempted[-1]:.3e} "
            f"(last cholesky info={last_info})"
        )

    @staticmethod
    def _chol_solve(L: Tensor, b: Tensor) -> Tensor:
        # solves Ky x = b  given  L L^T = Ky
        return torch.cholesky_solve(b.reshape(-1, 1) if b.ndim == 1 else b, L).squeeze(-1)

    # ----- marginal log-likelihood --------------------------------------
    def _neg_mll(self, Z: Tensor, y: Tensor) -> Tensor:
        Ky = self._Ky(Z)
        L  = self._cholesky(Ky)
        alpha = torch.cholesky_solve(y.reshape(-1, 1), L).reshape(-1)
        N = y.shape[0]
        quad = 0.5 * torch.dot(y, alpha)
        logdet = torch.sum(torch.log(torch.diagonal(L)))
        const = 0.5 * N * float(np.log(2.0 * np.pi))
        return quad + logdet + const

    # ----- LOO-CV loss (preferred over MLL for near-R²=1) ----------------
    def _loo_cv_loss(self, Z: Tensor, y: Tensor) -> Tensor:
        """
        Leave-one-out cross-validation loss for ARD-RBF GP.

        L_LOO = (1/N) Σ_i (α_i / [K_y^{-1}]_{ii})²

        Derivation:
            The LOO prediction error for point i is  ê_i = α_i / [K_y^{-1}]_{ii},
            where [K_y^{-1}]_{ii} = Σ_k (L^{-1})_{ki}²  (L lower-triangular Cholesky).

        This directly penalizes pointwise prediction errors and has NO oversmoothing
        bias, unlike MLL which prefers large ℓ to explain variance as entropy.
        Using LOO-CV for hyperparameter optimization keeps R² near 1 throughout
        dynamics; MLL monotonically grows ℓ in breathing refits (bug 4).

        Cost: O(N³) — same as MLL (dominated by triangular solve for L^{-1}).
        """
        Ky  = self._Ky(Z)
        L   = self._cholesky(Ky)
        alpha = torch.cholesky_solve(y.reshape(-1, 1), L).reshape(-1)
        # Compute L^{-1} to get diag(K_y^{-1}) = row-norms of L^{-1}.
        I_N = torch.eye(L.shape[0], dtype=L.dtype, device=L.device)
        Linv = torch.linalg.solve_triangular(L, I_N, upper=False)  # (N, N)
        Ky_inv_diag = torch.sum(Linv * Linv, dim=0).clamp_min(1.0e-30)  # (N,)
        loo_resid = alpha / Ky_inv_diag          # LOO prediction errors (N,)
        return torch.mean(loo_resid * loo_resid)

    # ----- hyperparameter optimization ----------------------------------
    def _project_log_hypers_(self) -> None:
        """Clip log-hyperparameters into their configured floors/ceilings,
        then restore pinned-axis values from the lengthscale-pin contract.

        When ``fix_sigma_n=True`` the log_sigma_n leaf has requires_grad=False
        and is intentionally pinned at the user-supplied init_log_sigma_n.
        Clamping it here would silently override that choice whenever the
        chosen init falls outside [log_sn_floor, log_sn_ceiling] — which
        actually happens with the run.py defaults (init=-10 vs floor=-8).
        Skip the σ_n clamp in that case so the user's pinned value is honored.

        Lengthscale pinning: any axis with ``self._pin_mask[d]`` is RESTORED
        to ``self._pin_log_ls_norm[d]`` after the clamp.  This is the
        last-line-of-defense enforcement; the optimizer also masks gradients
        on those axes (see `optimize_hyperparameters`), but if any code path
        leaks an update past the gradient mask it will be erased here.
        """
        cfg = self.config
        with torch.no_grad():
            self.log_lengthscales.clamp_(cfg.log_ls_floor, cfg.log_ls_ceiling)
            if not cfg.fix_sigma_n:
                self.log_sigma_n.clamp_(cfg.log_sn_floor, cfg.log_sn_ceiling)
            # Restore pinned axes (overwrite any drift from optimizer
            # state or the clamp above).  No-op when no pins are set.
            if bool(self._pin_mask.any()):
                pin_idx = np.where(self._pin_mask)[0]
                pin_vals = torch.as_tensor(
                    self._pin_log_ls_norm[pin_idx],
                    dtype=self.log_lengthscales.dtype,
                    device=self.log_lengthscales.device,
                )
                self.log_lengthscales[torch.as_tensor(
                    pin_idx, dtype=torch.long,
                    device=self.log_lengthscales.device,
                )] = pin_vals

    def _regularization_loss(self) -> Tensor:
        lam = float(getattr(self.config, "l2_regularization", 0.0))
        if lam <= 0.0:
            return torch.zeros((), dtype=_DEFAULT_DTYPE, device=_DEFAULT_DEVICE)
        reg = (self.log_sigma_f - self._reg_anchor_log_sigma_f) ** 2
        reg = reg + torch.mean((self.log_lengthscales - _as_tensor(self._reg_anchor_log_lengthscales)) ** 2)
        reg = reg + (self.log_sigma_n - self._reg_anchor_log_sigma_n) ** 2
        return torch.as_tensor(lam, dtype=_DEFAULT_DTYPE, device=_DEFAULT_DEVICE) * reg

    def _validation_split(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Deterministic validation split.

        Uses a seeded random permutation rather than `np.linspace` because
        after MInt propagation the cloud is no longer in "random iid" order
        (support points were already shuffled when drawn, but any sorting
        induced by the healthiest-cloud selector, signed-SEO filtering, or
        subsequent processing can bias a strided index).  A fixed-seed
        permutation gives the same split for the same (n, config) pair while
        avoiding the order-bias.
        """
        frac = float(np.clip(getattr(self.config, "validation_fraction", 0.0), 0.0, 0.5))
        min_val = int(max(0, getattr(self.config, "min_validation_points", 0)))
        if frac <= 0.0 or n < max(4, min_val + 2):
            idx = np.arange(n, dtype=np.int64)
            return idx, np.empty(0, dtype=np.int64)
        n_val = max(min_val, int(np.ceil(frac * n)))
        n_val = min(max(1, n_val), n - 2)

        seed = int(getattr(self.config, "validation_split_seed", 12345))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n).astype(np.int64)
        val_idx = np.sort(perm[:n_val])
        train_idx = np.sort(perm[n_val:])
        return train_idx, val_idx

    def _fit_metrics_from_subset(self,
                                 Z_ref: Tensor,
                                 Z_basis: Tensor,
                                 y_ref_raw: Tensor,
                                 y_basis_train: Tensor) -> Tuple[float, float]:
        """
        Compute held-out (or train) MAE / R² for the current hyperparameters.

        `y_basis_train` must be the *uncentered* scaled RHS so that
        α_scaled from Cholesky-solve matches what _compute_unconstrained_alpha
        would compute.  We un-scale back to physical units via `* y_scale`
        (no additive offset) before comparing to y_ref_raw.
        """
        Ky = self._Ky(Z_basis)
        L = self._cholesky(Ky)
        alpha_scaled = torch.cholesky_solve(y_basis_train.reshape(-1, 1), L).reshape(-1)
        with torch.no_grad():
            K_ref = _ard_gram(Z_ref, Z_basis, self.log_sigma_f, self.log_lengthscales)
            # _y_scale == 1.0 (RNS removed); alpha_scaled == alpha in physical units
            y_pred = K_ref @ alpha_scaled
            resid  = y_pred - y_ref_raw
            mae = float(torch.mean(torch.abs(resid)).item())
            y0 = y_ref_raw - torch.mean(y_ref_raw)
            denom = float(torch.sum(y0 ** 2).item())
            if denom <= 0.0:
                r2 = float("nan")
            else:
                ss_res = float(torch.sum(resid ** 2).item())
                r2 = 1.0 - ss_res / denom
        return mae, r2

    def optimize_hyperparameters(self,
                                 Z_train: Tensor,
                                 y_train: Tensor,
                                 n_steps: Optional[int] = None,
                                 verbose: bool = False) -> List[float]:
        """
        Maximise the marginal log-likelihood (MLL) using full-batch L-BFGS with
        strong Wolfe line search.  L-BFGS is a quasi-Newton method that converges
        in O(10-50) outer iterations on smooth objectives — far fewer than Adam —
        without stochastic noise.  Mini-batching is intentionally not used.

        n_steps: maximum L-BFGS outer iterations (each does up to _MAX_ITER
        internal line-search evaluations).  Early stopping terminates as soon as
        the validation MAE does not improve for early_stop_patience consecutive
        outer steps, restoring the best hyperparameter state found.
        """
        if self._hypers_frozen:
            return []

        cfg = self.config
        n_steps = cfg.n_opt_steps if n_steps is None else int(n_steps)
        if n_steps <= 0:
            self.last_opt_steps = 0
            return []

        params = [self.log_sigma_f, self.log_lengthscales]
        if not cfg.fix_sigma_n:
            params.append(self.log_sigma_n)

        n = int(Z_train.shape[0])
        tr_idx_np, va_idx_np = self._validation_split(n)
        tr_idx = torch.as_tensor(tr_idx_np, dtype=torch.long, device=Z_train.device)
        va_idx = torch.as_tensor(va_idx_np, dtype=torch.long, device=Z_train.device)
        Z_tr = Z_train.index_select(0, tr_idx)
        y_tr = y_train.index_select(0, tr_idx)

        # Sampler-supplied LabelInformation may override the config's
        # loss choice (see `LabelInformation.recommended_loss`).  This
        # respects the principle that the sampler knows its labels'
        # statistical structure better than a generic config default.
        if self._pin_use_loocv is not None:
            use_loocv = bool(self._pin_use_loocv)
        else:
            use_loocv = bool(getattr(cfg, "use_loocv", False))

        _MAX_ITER = 20  # line-search evaluations per outer L-BFGS step
        opt = torch.optim.LBFGS(
            params, lr=1.0, max_iter=_MAX_ITER,
            tolerance_grad=1.0e-9, tolerance_change=1.0e-11,
            history_size=10, line_search_fn="strong_wolfe",
        )

        history: List[float] = []
        best_metric = float("inf")
        best_step   = -1
        best_state  = {
            "log_sigma_f":      self.log_sigma_f.detach().clone(),
            "log_lengthscales": self.log_lengthscales.detach().clone(),
            "log_sigma_n":      self.log_sigma_n.detach().clone(),
        }
        patience  = int(max(1, getattr(cfg, "early_stop_patience",  30)))
        min_delta = float(max(0.0, getattr(cfg, "early_stop_min_delta", 1.0e-6)))
        no_improve    = 0
        early_stopped = False

        for k, attr in [("last_opt_total_loss", float("nan")),
                         ("last_opt_nll_loss",   float("nan")),
                         ("last_opt_reg_loss",   float("nan")),
                         ("last_opt_train_mae",  float("nan")),
                         ("last_opt_train_r2",   float("nan")),
                         ("last_opt_val_mae",    float("nan")),
                         ("last_opt_val_r2",     float("nan"))]:
            setattr(self, k, attr)

        _nan_strikes = 0           # consecutive NaN steps before giving up
        _MAX_NAN_STRIKES = 5

        for outer in range(n_steps):
            def closure() -> Tensor:
                opt.zero_grad(set_to_none=True)
                # Robust objective (2026-07): with sigma_n free, the strong-
                # Wolfe line search may probe hyperparameters where K_y loses
                # positive-definiteness; the Cholesky then RAISES inside the
                # optimizer's internal evaluation, bypassing the NaN-strike
                # recovery below.  Convert the failure into a NaN loss so the
                # existing best-state-restore + optimizer-reset machinery
                # handles it — this is what makes joint optimization of
                # (sigma_f, sigma_n, all lengthscales) safe without pinning.
                try:
                    nll = (self._loo_cv_loss(Z_tr, y_tr)
                           if use_loocv else self._neg_mll(Z_tr, y_tr))
                    reg = self._regularization_loss()
                    loss = nll + reg
                except (torch.linalg.LinAlgError, RuntimeError) as _e:
                    if "positive-definite" not in str(_e) and \
                       not isinstance(_e, torch.linalg.LinAlgError):
                        raise
                    return torch.tensor(float("nan"),
                                        dtype=self.log_sigma_f.dtype)
                # NaN guard: if the loss is non-finite (Cholesky failure,
                # log(0), or numerical overflow) do NOT call backward() —
                # propagating NaN gradients poisons the L-BFGS quasi-Newton
                # curvature estimate and makes all subsequent steps produce
                # NaN too.  Return the NaN loss so the caller can detect it
                # and reset the optimizer state.
                if not torch.isfinite(loss):
                    return loss
                loss.backward()
                # Mask gradient on pinned axes — they are not optimized
                # parameters.  Without this the L-BFGS line search would
                # move them and the post-step _project_log_hypers_ restore
                # would create spurious "curvature" in the optimizer's
                # quasi-Newton state.
                if self.log_lengthscales.grad is not None and bool(self._pin_mask.any()):
                    mask_t = torch.as_tensor(
                        (~self._pin_mask).astype(np.float64),
                        dtype=self.log_lengthscales.grad.dtype,
                        device=self.log_lengthscales.grad.device,
                    )
                    self.log_lengthscales.grad.mul_(mask_t)
                return loss

            try:
                loss_t = opt.step(closure)
                total_loss = (float(loss_t.detach())
                              if loss_t is not None else float("nan"))
            except (IndexError, RuntimeError, torch.linalg.LinAlgError):
                # torch's _strong_wolfe bracket handling itself fails when
                # the closure returns NaN mid-line-search (empty bracket →
                # IndexError), or a Cholesky failure escapes on the very
                # first evaluation.  Treat exactly like a NaN loss: restore
                # the best state and rebuild the optimizer below.
                total_loss = float("nan")
            self._project_log_hypers_()

            # If the step produced a NaN loss, the L-BFGS internal state is
            # contaminated.  Restore the best known hyperparameters and
            # reinitialize the optimizer so subsequent steps start from a
            # clean quasi-Newton history.  Count consecutive NaN strikes; if
            # they persist (e.g. the entire optimization landscape is
            # degenerate) break early to avoid wasted iterations.
            if not np.isfinite(total_loss):
                _nan_strikes += 1
                with torch.no_grad():
                    self.log_sigma_f.copy_(best_state["log_sigma_f"])
                    self.log_lengthscales.copy_(best_state["log_lengthscales"])
                    self.log_sigma_n.copy_(best_state["log_sigma_n"])
                self._project_log_hypers_()
                # Reinitialize optimizer from the restored (clean) state.
                opt = torch.optim.LBFGS(
                    params, lr=1.0, max_iter=_MAX_ITER,
                    tolerance_grad=1.0e-9, tolerance_change=1.0e-11,
                    history_size=10, line_search_fn="strong_wolfe",
                )
                if _nan_strikes >= _MAX_NAN_STRIKES:
                    break
                history.append(total_loss)
                continue
            _nan_strikes = 0
            history.append(total_loss)

            with torch.no_grad():
                nll_loss = float((self._loo_cv_loss(Z_tr, y_tr)
                                  if use_loocv else
                                  self._neg_mll(Z_tr, y_tr)).item())
                reg_loss = float(self._regularization_loss().item())

            train_mae, train_r2 = self._fit_metrics_from_subset(
                Z_tr, Z_tr,
                self._y_raw.index_select(0, tr_idx),
                self._y_train.index_select(0, tr_idx),
            )
            if va_idx.numel() > 0:
                val_mae, val_r2 = self._fit_metrics_from_subset(
                    Z_train.index_select(0, va_idx), Z_tr,
                    self._y_raw.index_select(0, va_idx),
                    self._y_train.index_select(0, tr_idx),
                )
                metric = val_mae
            else:
                val_mae, val_r2 = float("nan"), float("nan")
                metric = total_loss

            if metric < (best_metric - min_delta):
                best_metric = metric;  best_step = outer;  no_improve = 0
                best_state = {
                    "log_sigma_f":      self.log_sigma_f.detach().clone(),
                    "log_lengthscales": self.log_lengthscales.detach().clone(),
                    "log_sigma_n":      self.log_sigma_n.detach().clone(),
                }
                self.last_opt_total_loss = total_loss
                self.last_opt_nll_loss   = nll_loss
                self.last_opt_reg_loss   = reg_loss
                self.last_opt_train_mae  = train_mae
                self.last_opt_train_r2   = train_r2
                self.last_opt_val_mae    = val_mae
                self.last_opt_val_r2     = val_r2
            else:
                no_improve += 1

            if verbose and (outer % max(1, n_steps // 10) == 0
                            or outer == n_steps - 1):
                print(f"  [L-BFGS/MLL] step {outer:3d}  loss={total_loss:.6e}  "
                      f"nll={nll_loss:.6e}  reg={reg_loss:.6e}  "
                      f"val_mae={val_mae:.6e}  "
                      f"σ_f={self.sigma_f:.3e}  σ_n={self.sigma_n:.3e}")
            if no_improve >= patience:
                early_stopped = True
                break

        with torch.no_grad():
            self.log_sigma_f.copy_(best_state["log_sigma_f"])
            self.log_lengthscales.copy_(best_state["log_lengthscales"])
            self.log_sigma_n.copy_(best_state["log_sigma_n"])
        self._project_log_hypers_()
        self.last_opt_steps         = len(history)
        self.last_opt_best_step     = best_step
        self.last_opt_early_stopped = bool(early_stopped)
        return history
    def _set_training_data(self, Z_train: FloatArray, y_train: FloatArray) -> None:
        Zt = np.asarray(Z_train, dtype=np.float64)
        yt = np.asarray(y_train, dtype=np.float64).reshape(-1)
        if Zt.ndim != 2 or Zt.shape[1] != D:
            raise ValueError(f"Z_train must have shape (N, {D}); got {Zt.shape}.")
        if yt.shape[0] != Zt.shape[0]:
            raise ValueError(f"y_train length {yt.shape[0]} mismatches N = {Zt.shape[0]}.")

        # Feature z-score: compute once on the first fit, unless the caller
        # explicitly requests recomputation *and* the feature map has not been
        # frozen by freeze_hypers().  We always keep RAW centers in self._Z_train
        # so moments/observables stay in physical coordinates.
        recompute_feature_stats = (self.config.recompute_feature_zscore
                                   and (not self._feature_stats_frozen))
        need_feature_stats = (self._feature_mean is None or self._feature_std is None
                              or recompute_feature_stats)
        if need_feature_stats:
            mu, sd = _zscore_stats(Zt)
            self._feature_mean = _as_tensor(mu)
            self._feature_std  = _as_tensor(sd)

        Z_norm = self._normalize_features_np(Zt)
        self._Z_train = _as_tensor(Zt)
        self._Z_train_norm = _as_tensor(Z_norm)

        # RNS target normalization removed.  Targets always in physical units.
        # _y_scale stays 1.0 (set in __init__).
        self._y_raw   = _as_tensor(yt)
        self._y_train = self._y_raw

    def _training_fit_metrics(self, alpha: Tensor) -> Tuple[float, float, float]:
        # Predicted training values are y_pred = K(Z_train, Z_train) @ alpha
        # (no σ_n² and no jitter — both come from Ky, not from the model
        # itself; predict() also evaluates K alone).
        #
        # The previous shortcut "y_pred = y_train - σ_n² * alpha" relied on
        # Ky @ alpha == y_train, which is true ONLY for the unconstrained
        # alpha0 = Ky^{-1} y.  After the KKT/Schur projection alpha is moved
        # off that line, so Ky @ alpha != y_train and the shortcut produces
        # incorrect fit metrics for the constrained path — affecting
        # last_fit_rms / mae / r2 and the constraint_delta_* diagnostics.
        # It also silently dropped the jitter offset for both alphas.
        #
        # Fix: rebuild K(Z, Z) once via _ard_gram (matches predict()).
        # Cost is one O(N²·D) kernel build, negligible vs. the upstream
        # O(N³) Cholesky, paid once per fit/refit.
        with torch.no_grad():
            K = _ard_gram(self._Z_train_norm, self._Z_train_norm,
                          self.log_sigma_f, self.log_lengthscales)
            y_pred = K @ alpha
            resid = y_pred - self._y_raw
            rms = float(torch.sqrt(torch.mean(resid ** 2)).item())
            mae = float(torch.mean(torch.abs(resid)).item())
            y0 = self._y_raw - torch.mean(self._y_raw)
            denom = float(torch.sum(y0 ** 2).item())
            if denom <= 0.0:
                r2 = float("nan")
            else:
                ss_res = float(torch.sum(resid ** 2).item())
                r2 = 1.0 - ss_res / denom
        return rms, mae, r2

    def _compute_unconstrained_alpha(self) -> None:
        Ky = self._Ky(self._Z_train_norm)
        L  = self._cholesky(Ky)
        with torch.no_grad():
            # Solve K_y α = y.  Since _y_scale == 1.0 (RNS removed),
            # y_train is already in physical units; no rescaling needed.
            alpha0 = torch.cholesky_solve(
                self._y_train.reshape(-1, 1), L).reshape(-1)
        self._L_Ky   = L.detach()
        self._alpha0 = alpha0

    def _apply_kkt_projection(self,
                              moment_targets: Dict[MomentName, float]) -> None:
        """
        Build the moment matrix A and targets b for the requested moments, then
        perform the Schur-complement projection of alpha0 onto A α = b.
        """
        if self._alpha0 is None or self._L_Ky is None or self._Z_train is None:
            raise RuntimeError("Internal state not initialized; call fit() first.")

        # If the active LabelInformation contract declares its labels are
        # NOT interpretable as physical density (e.g. focused sampling
        # where labels are W_cl·K_focus, not ρ_phys), skip KKT entirely.
        # The unconstrained least-squares α₀ already fits the labels
        # perfectly (typically R² > 0.999); attempting to force the GP
        # integral to a physical value would destroy this fit because
        # the integral lives in label units, not physical units, and
        # the discrepancy is O(1) not O(sampling_noise).  Physical
        # observables come from the cloud Riemann sum independently.
        if not self._pin_apply_kkt:
            self._A = None
            self._b = None
            self._alpha = self._alpha0.clone()
            self._moment_order = ()
            self._kkt_dropped_rank = 0
            return

        order: Tuple[MomentName, ...] = tuple(moment_targets.keys())
        if (len(order) == 0) or (not self.config.constraints_enabled):
            self._A = None
            self._b = None
            self._alpha = self._alpha0.clone()
            self._moment_order = ()
            return

        A_np = self.moment_integrator.build_A(
            Z_train=_as_numpy(self._Z_train),
            log_sigma_f=float(self.log_sigma_f.item()),
            log_lengthscales=np.log(self.lengthscales),
            moments=order,
        )
        b_np = np.array([float(moment_targets[m]) for m in order], dtype=np.float64)

        A = _as_tensor(A_np)                               # (m, N)
        b = _as_tensor(b_np)                               # (m,)
        L = self._L_Ky

        # Ky^{-1} A^T  of shape (N, m)
        KyInv_AT = torch.cholesky_solve(A.t(), L)
        # Schur matrix S = A Ky^{-1} A^T  of shape (m, m)
        S = A @ KyInv_AT
        S = S + self.config.constraint_ridge * torch.eye(S.shape[0], dtype=S.dtype)

        rhs = A @ self._alpha0 - b                         # (m,)

        # Rank-revealing solve via SVD.
        # ---------------------------------------------------------------
        # The naive solve `lam = solve(S, rhs)` fails silently when S is
        # rank-deficient (Casimir invariants make moment rows linearly
        # dependent; cf. focused sampling making trace ∝ normalization).
        # With constraint_ridge=1e-12 (default), linalg.solve returns a
        # solution dominated by the null-space direction scaled by 1/ridge,
        # giving the constant-norm-0.6 / constant-trace-1.19 pathology
        # observed in long focused-mode runs.
        #
        # Robust fix: SVD S = U Σ Vᵀ; pseudoinverse drops directions whose
        # singular value falls below max(σ)·rcond.  For full-rank S this is
        # identical to direct solve to within roundoff.  For rank-r < m it
        # returns the minimum-norm λ that satisfies the well-posed
        # sub-system A·correction = (A·α₀ - b) projected onto the row space
        # of A — exactly what we want when one of the moments is a Casimir
        # already automatically satisfied by α₀.
        S_np = _as_numpy(S)
        rhs_np = _as_numpy(rhs)
        U, sv, Vt = np.linalg.svd(S_np, full_matrices=False)
        sv_max = float(sv.max()) if sv.size > 0 else 0.0
        rcond = 1.0e-10  # matches torch.linalg.pinv default rtol
        sv_threshold = sv_max * rcond
        sv_inv = np.where(sv > sv_threshold, 1.0 / sv, 0.0)
        lam_np = Vt.T @ (sv_inv * (U.T @ rhs_np))
        lam = torch.as_tensor(lam_np, dtype=S.dtype)
        # Diagnostic: number of effectively-zero singular values
        self._kkt_dropped_rank = int(np.sum(sv <= sv_threshold))

        correction = KyInv_AT @ lam                        # (N,)

        self._A = A.detach()
        self._b = b.detach()
        self._alpha = (self._alpha0 - correction).detach()
        self._moment_order = order

    def fit(self,
            Z_train: ArrayLike,
            y_train: ArrayLike,
            moment_targets: Optional[Dict[MomentName, float]] = None,
            optimize: bool = True,
            verbose: bool = False,
            apply_constraints: Optional[bool] = None) -> None:
        """
        Full initial fit:
          1. cache training data
          2. (optionally) initialize lengthscales from training MAD
          3. (optionally) optimize hyperparameters by MLL
          4. compute unconstrained α0
          5. KKT-project onto the requested moment constraints
        """
        self._set_training_data(Z_train, y_train)
        cfg = self.config

        # Drop any moments the sampler declared as redundant Casimirs
        # under its current LabelInformation pin contract.  See
        # `_filter_redundant_moments` for details on why this is needed.
        moment_targets = self._filter_redundant_moments(moment_targets)

        # Initialize lengthscales once from the cloud spread when the user did
        # not provide them explicitly.  Subsequent changes come from optimizer updates.
        # If a pinning contract was set via `pin_lengthscales`, restore the
        # pinned axes after the cloud-spread init (it would otherwise overwrite them).
        need_ls_init = (cfg.init_log_lengthscales is None) and (self.n_train > 0)
        if need_ls_init:
            ls0 = _std_lengthscales(_as_numpy(self._Z_train_norm))
            with torch.no_grad():
                self.log_lengthscales.copy_(torch.log(_as_tensor(ls0)))
            self._reg_anchor_log_lengthscales = np.log(ls0)
            # If a pin is active, axes in the pin must NOT take the cloud-std init;
            # restore them to their pinned values.
            if bool(self._pin_mask.any()):
                with torch.no_grad():
                    for d in range(D):
                        if self._pin_mask[d]:
                            self.log_lengthscales[d] = float(self._pin_log_ls_norm[d])
                            self._reg_anchor_log_lengthscales[d] = float(self._pin_log_ls_norm[d])

        self._project_log_hypers_()

        if optimize:
            # Physical-scale labels (RNS removed, y_scale==1) — no centering needed.
            self.optimize_hyperparameters(self._Z_train_norm, self._y_train,
                                          verbose=verbose)

        self._compute_unconstrained_alpha()
        self.last_free_fit_rms, self.last_free_fit_mae, self.last_free_fit_r2 = self._training_fit_metrics(self._alpha0)

        if moment_targets is None:
            moment_targets = {}
        # normalize ordering
        allowed: Tuple[MomentName, ...] = ("normalization", "trace", "energy")
        ordered = {m: float(moment_targets[m]) for m in allowed if m in moment_targets}
        if apply_constraints is not None:
            old = self.config.constraints_enabled
            self.config.constraints_enabled = bool(apply_constraints)
            try:
                self._apply_kkt_projection(ordered)
            finally:
                self.config.constraints_enabled = old
        else:
            self._apply_kkt_projection(ordered)

        self.last_fit_rms, self.last_fit_mae, self.last_fit_r2 = self._training_fit_metrics(self._alpha)
        self.constraint_delta_rmse = self.last_fit_rms - self.last_free_fit_rms
        self.constraint_delta_mae = self.last_fit_mae - self.last_free_fit_mae
        self.constraint_delta_r2 = self.last_fit_r2 - self.last_free_fit_r2

        # Pin the initial-fit anchor AFTER the first successful fit.  The
        # breathing-refit policy reads these values to (a) hold σ_f and σ_n
        # fixed during refits, and (b) regularize lengthscales back toward
        # their initial-fit values.  We never touch the anchor after this.
        if not self._initial_fit_done:
            self._initial_log_sigma_f_anchor = float(self.log_sigma_f.detach().item())
            self._initial_log_sigma_n_anchor = float(self.log_sigma_n.detach().item())
            self._initial_log_lengthscales_anchor = _as_numpy(self.log_lengthscales).copy()
            self._initial_fit_done = True

    def _compute_breathing_anchor(self, Z_used: Tensor) -> Tensor:
        """
        Return the per-dim log-lengthscale anchor for the breathing prior,
        in the SAME space (normalized or physical) as ``self.log_lengthscales``.

        Three policies, selected by ``self.config.breathing_anchor_policy``:

          * ``"initial"`` — frozen at the first-fit lengthscales.  Stable but
            does not track cloud-geometry change; this is what causes
            fit_rms to creep up monotonically over a long run when the
            cloud broadens (the prior actively prevents ℓ from following).

          * ``"cloud_mad"`` — the per-dim robust scale estimator
            ``MAD_d = 1.4826 · median(|z_d − median(z_d)|)`` of the current
            cloud, scaled by ``config.breathing_anchor_mad_factor`` (default
            0.4 — the textbook ARD-RBF bandwidth heuristic).  Anchor moves
            with the cloud at every refit; can be jittery.

          * ``"ewma"`` — exponentially-weighted moving average of cloud_mad,
            initialized at log ℓ_0 (first-fit lengthscales) so behaviour at
            t=0 is identical to "initial".  As the cloud evolves, the
            anchor smoothly tracks the new geometry with time constant
            ``≈ 1/(1−β) ≈ 10`` refits at the default β=0.9.  Recommended
            default: tracks long-term geometry change without reacting to
            single-step sampling jitter.

        ``Z_used`` is the cloud the optimizer is actually consuming (i.e.
        normalized when feature_zscore=True), so MAD computed here lives in
        the same space as ``self.log_lengthscales`` and no rescaling is
        needed.  The returned tensor is detached and on the same device as
        the parameter it shrinks toward.
        """
        policy = str(getattr(self.config, "breathing_anchor_policy", "ewma")).lower()

        if policy == "initial":
            anchor_np = self._initial_log_lengthscales_anchor
        else:
            # Robust scale per dimension on the cloud the optimizer sees.
            Z_np = _as_numpy(Z_used)                                  # (N, D)
            med = np.median(Z_np, axis=0)                             # (D,)
            mad = 1.4826 * np.median(np.abs(Z_np - med[None, :]), axis=0)
            mad = np.maximum(mad, 1.0e-12)                            # avoid log(0)
            kappa = float(getattr(self.config, "breathing_anchor_mad_factor", 0.4))
            cloud_mad_log = np.log(kappa * mad)                       # (D,)

            if policy == "cloud_mad":
                anchor_np = cloud_mad_log
            elif policy == "ewma":
                if self._ewma_log_lengthscales_anchor is None:
                    # Seed at log ℓ_0 so the very first refit reproduces
                    # legacy "initial" behaviour exactly.
                    self._ewma_log_lengthscales_anchor = (
                        self._initial_log_lengthscales_anchor.copy()
                    )
                beta = float(getattr(self.config,
                                     "breathing_anchor_ewma_beta", 0.9))
                self._ewma_log_lengthscales_anchor = (
                    beta * self._ewma_log_lengthscales_anchor
                    + (1.0 - beta) * cloud_mad_log
                )
                anchor_np = self._ewma_log_lengthscales_anchor.copy()
            else:
                raise ValueError(
                    f"Unknown breathing_anchor_policy: {policy!r} "
                    f"(expected 'initial', 'cloud_mad', or 'ewma')"
                )

        # Optionally clamp the anchor to the global log-lengthscale floor /
        # ceiling so a pathological cloud (e.g. singular MAD) cannot send
        # the prior off to extreme values.
        floor = float(self.config.log_ls_floor)
        ceil  = float(self.config.log_ls_ceiling)
        anchor_np = np.clip(anchor_np, floor, ceil)

        self._last_breathing_anchor = anchor_np.copy()
        return torch.as_tensor(anchor_np, dtype=_DEFAULT_DTYPE)

    def _breathing_optimize_lengthscales(self,
                                         Z_train: Tensor,
                                         y_train: Tensor,
                                         n_steps: int,
                                         prior_weight: float,
                                         prior_clip: float,
                                         verbose: bool = False) -> List[float]:
        """
        Transactional per-refit update using projected L-BFGS.

        Minimises
            L(log ℓ) = -log p(y | Z; σ_f₀, σ_n, ℓ)
                     + Σ_d w_d (log ℓ_d - log ℓ_anchor_d)²  (shrinkage prior)

        subject to a trust-region clip |log ℓ_d - log ℓ_anchor_d| ≤ prior_clip
        applied after each outer L-BFGS step.  σ_f is pinned at its initial-fit
        anchor; σ_n floats inside its configured bounds unless
        ``fix_sigma_n=True``.  Each optimizer step is evaluated and accepted as
        a complete state (ℓ, σ_n).  If the objective, hyperparameters, or
        covariance become invalid, the full last-known-good state is restored
        and only this breathing burst is abandoned.  The dynamics step then
        continues with a valid GP.

        The refit intentionally uses one projected L-BFGS update per outer step
        without a strong-Wolfe line search.  Strong-Wolfe probes unconstrained
        temporary points before the post-step projection is applied; in long
        focused PBME runs those probes can overflow an ARD length scale or noise
        parameter and feed a non-finite matrix to Cholesky.  Projected L-BFGS
        preserves the quasi-Newton history while ensuring every evaluated
        candidate has first passed the configured bounds and trust region.
        """
        n_steps = int(max(0, n_steps))
        self.last_breathing_failed = False
        self.last_breathing_failure_reason = ""
        self.last_breathing_failure_code = 0
        if n_steps == 0:
            return []

        if not self._initial_fit_done:
            raise RuntimeError(
                "GPDensity._breathing_optimize_lengthscales called before fit(); "
                "the initial-fit anchor is not set."
            )

        # Pin σ_f and (optionally) σ_n.  Project before taking the entry
        # snapshot so even a manually modified caller state respects the global
        # floors/ceilings and the label-information pin contract.
        with torch.no_grad():
            self.log_sigma_f.copy_(torch.tensor(
                self._initial_log_sigma_f_anchor, dtype=_DEFAULT_DTYPE))
            self.log_sigma_f.requires_grad_(False)
            if self.config.fix_sigma_n:
                self.log_sigma_n.copy_(torch.tensor(
                    self._initial_log_sigma_n_anchor, dtype=_DEFAULT_DTYPE))
                self.log_sigma_n.requires_grad_(False)
            else:
                self.log_sigma_n.requires_grad_(True)
            self.log_lengthscales.requires_grad_(True)
        self._project_log_hypers_()

        anchor = self._compute_breathing_anchor(Z_train).to(_DEFAULT_DTYPE)
        clip   = float(max(0.0, prior_clip))

        # Per-dim prior weights
        per_dim = getattr(self.config, "lengthscale_prior_weight_per_dim", None)
        if per_dim is not None:
            w_vec = torch.as_tensor(
                np.asarray(per_dim, dtype=np.float64), dtype=_DEFAULT_DTYPE
            ).clamp(min=0.0)
            if w_vec.numel() != D:
                raise ValueError(f"lengthscale_prior_weight_per_dim must have length {D}")
        else:
            w_vec = torch.full((D,), float(max(0.0, prior_weight)), dtype=_DEFAULT_DTYPE)

        nuclear_only = bool(getattr(self.config, "breathe_nuclear_only", False))
        if nuclear_only:
            mapping_anchor_vals = anchor[2:].detach().clone()

        # See optimize_hyperparameters for rationale on the pin override.
        if self._pin_use_loocv is not None:
            use_loocv = bool(self._pin_use_loocv)
        else:
            use_loocv = bool(getattr(self.config, "use_loocv", False))

        def _prior_term() -> Tensor:
            if nuclear_only:
                d_dev = self.log_lengthscales[:2] - anchor[:2]
                return torch.sum(w_vec[:2] * d_dev * d_dev) / 2.0
            d_dev = self.log_lengthscales - anchor
            return torch.sum(w_vec * d_dev * d_dev) / float(D)

        opt_params = [self.log_lengthscales]
        if not self.config.fix_sigma_n:
            opt_params.append(self.log_sigma_n)

        # One deterministic quasi-Newton update per outer call.  Optimizer state
        # persists between calls to step(), so curvature history is retained.
        # A conservative learning rate plus the explicit trust-region projection
        # replaces unsafe unconstrained strong-Wolfe trial evaluations.
        _LBFGS_LR = 0.25
        opt = torch.optim.LBFGS(
            opt_params, lr=_LBFGS_LR, max_iter=1,
            tolerance_grad=1.0e-9, tolerance_change=1.0e-11,
            history_size=10, line_search_fn=None,
        )

        history: List[float] = []
        best_state = {
            "log_lengthscales": self.log_lengthscales.detach().clone(),
            "log_sigma_n": self.log_sigma_n.detach().clone(),
        }
        best_loss = float("inf")
        best_nll  = float("nan")
        best_prior = float("nan")

        def _restore_best_state() -> None:
            with torch.no_grad():
                self.log_lengthscales.copy_(best_state["log_lengthscales"])
                self.log_sigma_n.copy_(best_state["log_sigma_n"])
                self.log_sigma_f.copy_(torch.tensor(
                    self._initial_log_sigma_f_anchor,
                    dtype=self.log_sigma_f.dtype,
                    device=self.log_sigma_f.device,
                ))
            self._project_log_hypers_()

        def _reject(code: int, reason: str) -> None:
            self.last_breathing_failed = True
            self.last_breathing_failure_code = int(code)
            self.last_breathing_failure_reason = str(reason)
            self.breathing_failure_count += 1
            _restore_best_state()
            if verbose:
                print("  [breathing/L-BFGS] rejected adaptive refit; "
                      "restored last finite (ℓ, σ_n) state: "
                      f"{reason}")

        def _objective_values() -> Tuple[float, float, float]:
            with torch.no_grad():
                nll_t = (self._loo_cv_loss(Z_train, y_train)
                         if use_loocv else self._neg_mll(Z_train, y_train))
                prior_t = _prior_term()
                total_t = nll_t + prior_t
            vals = (float(total_t.item()), float(nll_t.item()),
                    float(prior_t.item()))
            if not all(np.isfinite(v) for v in vals):
                raise RuntimeError("non-finite breathing objective")
            return vals

        # Establish that the entry state is usable and make it the baseline
        # candidate.  This is also the state restored if no proposed update is
        # better or if a later candidate is invalid.
        try:
            best_loss, best_nll, best_prior = _objective_values()
        except (RuntimeError, torch.linalg.LinAlgError) as exc:
            _reject(1, f"invalid entry objective: {exc}")
            self.last_opt_total_loss = float("nan")
            self.last_opt_nll_loss = float("nan")
            self.last_opt_reg_loss = float("nan")
            self.last_opt_steps = 0
            self.last_opt_best_step = -1
            return []

        best_step = -1

        for step in range(n_steps):
            def closure() -> Tensor:
                opt.zero_grad(set_to_none=True)
                try:
                    nll = (self._loo_cv_loss(Z_train, y_train)
                           if use_loocv else self._neg_mll(Z_train, y_train))
                    prior = _prior_term()
                    loss  = nll + prior
                except (RuntimeError, torch.linalg.LinAlgError):
                    return torch.tensor(float("nan"),
                                        dtype=self.log_lengthscales.dtype,
                                        device=self.log_lengthscales.device)
                # Skip backward on non-finite loss so no invalid gradient can
                # enter the quasi-Newton history.
                if not torch.isfinite(loss):
                    return loss
                loss.backward()
                # Mask gradient on pinned axes (label-information-rank
                # contract).  Identical to the initial-fit optimizer.
                if self.log_lengthscales.grad is not None and bool(self._pin_mask.any()):
                    mask_t = torch.as_tensor(
                        (~self._pin_mask).astype(np.float64),
                        dtype=self.log_lengthscales.grad.dtype,
                        device=self.log_lengthscales.grad.device,
                    )
                    self.log_lengthscales.grad.mul_(mask_t)
                return loss

            try:
                loss_t = opt.step(closure)
                step_loss = (float(loss_t.detach().item())
                             if loss_t is not None else float("nan"))
            except (IndexError, RuntimeError, torch.linalg.LinAlgError) as exc:
                _reject(2, f"optimizer step failed: {exc}")
                break

            if not np.isfinite(step_loss):
                _reject(3, "optimizer returned a non-finite loss")
                break

            # Reject NaN/Inf before projection: torch.clamp deliberately leaves
            # NaN unchanged, so it cannot repair a poisoned optimizer leaf.
            leaves_finite = (
                bool(torch.isfinite(self.log_lengthscales).all().item())
                and bool(torch.isfinite(self.log_sigma_n).all().item())
                and bool(torch.isfinite(self.log_sigma_f).all().item())
            )
            if not leaves_finite:
                _reject(4, "optimizer proposed non-finite hyperparameters")
                break

            # Trust-region clip + mapping-dim restore + pin restore
            with torch.no_grad():
                delta = self.log_lengthscales - anchor
                if clip > 0.0:
                    delta = delta.clamp(-clip, clip)
                self.log_lengthscales.copy_(anchor + delta)
                if nuclear_only:
                    self.log_lengthscales[2:].copy_(mapping_anchor_vals)
                self.log_lengthscales.clamp_(
                    self.config.log_ls_floor, self.config.log_ls_ceiling)
                # Restore pinned axes — last line of defense.  See
                # _project_log_hypers_ for rationale.
                if bool(self._pin_mask.any()):
                    pin_idx = np.where(self._pin_mask)[0]
                    pin_vals = torch.as_tensor(
                        self._pin_log_ls_norm[pin_idx],
                        dtype=self.log_lengthscales.dtype,
                        device=self.log_lengthscales.device,
                    )
                    self.log_lengthscales[torch.as_tensor(
                        pin_idx, dtype=torch.long,
                        device=self.log_lengthscales.device,
                    )] = pin_vals
            self._project_log_hypers_()  # includes the σ_n bounds

            # Validate the projected candidate itself.  The value returned by
            # torch LBFGS is the pre-update closure loss, so best-state selection
            # must use this post-update objective instead.
            try:
                candidate_loss, nll_v, prior_v = _objective_values()
            except (RuntimeError, torch.linalg.LinAlgError) as exc:
                _reject(5, f"projected candidate is invalid: {exc}")
                break
            history.append(candidate_loss)

            if candidate_loss < best_loss:
                best_loss  = candidate_loss
                best_nll   = nll_v
                best_prior = prior_v
                best_step  = step
                best_state = {
                    "log_lengthscales": self.log_lengthscales.detach().clone(),
                    "log_sigma_n": self.log_sigma_n.detach().clone(),
                }

            if verbose and (step == 0 or step == n_steps - 1
                            or step % max(1, n_steps // 5) == 0):
                ell_dev = float(torch.max(
                    torch.abs(self.log_lengthscales - anchor)).item())
                print(f"  [breathing/L-BFGS] step {step:3d}  "
                      f"loss={candidate_loss:.6e}  nll/N={nll_v:.6e}  "
                      f"prior={prior_v:.6e}  max|Δlog ℓ|={ell_dev:.3e}")

        _restore_best_state()

        self.last_opt_total_loss = best_loss
        self.last_opt_nll_loss   = best_nll
        self.last_opt_reg_loss   = best_prior
        self.last_opt_steps      = len(history)
        self.last_opt_best_step  = best_step
        self.last_opt_early_stopped = False
        for attr in ("last_opt_train_mae", "last_opt_train_r2",
                     "last_opt_val_mae",   "last_opt_val_r2"):
            setattr(self, attr, float("nan"))
        return history
    def refit(self,
              Z_train: ArrayLike,
              y_train: ArrayLike,
              moment_targets: Optional[Dict[MomentName, float]] = None,
              optimize: bool = True,
              reinit_lengthscales: Optional[bool] = None,
              verbose: bool = False,
              apply_constraints: Optional[bool] = None,
              hyper_policy: Optional[str] = None) -> None:
        """
        Refit at a later time step.  y_train should still be the t=0 density
        values (Liouville conservation along trajectories) for the PBME
        scheme, or the midpoint-updated y_new = y - dt·Q for the QCLE
        scheme.

        Hyperparameter behavior during refits is controlled by
        `hyper_policy` (falling back to `self.config.refit_hyper_policy`):

        *   "frozen"    — lock (σ_f, ℓ, σ_n) at their anchor values.  Only
                         rebuild Ky and solve for α.  Legacy behavior.

        *   "breathing" — lock σ_f at the anchor; update ℓ and, unless
                         ``fix_sigma_n=True``, σ_n with bounded L-BFGS on
                         MLL/LOO plus a quadratic shrinkage prior toward the
                         anchor ℓ_0.  This is the recommended default for
                         long propagation.

        *   "free"      — re-optimize every hyperparameter.  Expensive and
                         susceptible to MLL-driven oversmoothing; diagnostic
                         only.

        The legacy `optimize` flag is honored only under "free".  The
        `reinit_lengthscales` flag is silently ignored (MAD-based
        reinitialization is no longer part of the production path).
        """
        self._set_training_data(Z_train, y_train)

        # Same redundant-moment filtering as in fit(): drop Casimirs the
        # sampler declared in its LabelInformation pin contract so the
        # KKT system stays rank-non-deficient.
        moment_targets = self._filter_redundant_moments(moment_targets)

        policy = str(hyper_policy if hyper_policy is not None
                     else getattr(self.config, "refit_hyper_policy", "breathing")).strip().lower()
        if policy not in ("frozen", "breathing", "adaptive", "free"):
            raise ValueError(f"Unknown refit hyper_policy={policy!r}")

        # Per-refit status (the cumulative counter is intentionally retained).
        # Without this reset a single rejected adaptive burst would be reported
        # as a failure on every subsequent cooldown step.
        self.last_breathing_failed = False
        self.last_breathing_failure_reason = ""
        self.last_breathing_failure_code = 0

        # Legacy freeze_hypers() is still honored: it forces "frozen".
        if self._hypers_frozen:
            policy = "frozen"

        if policy == "frozen":
            # Pin σ_f, σ_n, ℓ at the initial-fit anchor so that short-term
            # wiggles in the current leaf-tensor values (from prior breathing
            # refits, for example) don't accumulate.
            if self._initial_fit_done:
                with torch.no_grad():
                    self.log_sigma_f.copy_(torch.tensor(self._initial_log_sigma_f_anchor, dtype=_DEFAULT_DTYPE))
                    self.log_sigma_n.copy_(torch.tensor(self._initial_log_sigma_n_anchor, dtype=_DEFAULT_DTYPE))
                    self.log_lengthscales.copy_(torch.as_tensor(self._initial_log_lengthscales_anchor, dtype=_DEFAULT_DTYPE))
            # Record zero-cost no-opt diagnostics.
            self.last_opt_total_loss = float("nan")
            self.last_opt_nll_loss = float("nan")
            self.last_opt_reg_loss = float("nan")
            self.last_opt_steps = 0
            self.last_opt_best_step = -1
            self.last_opt_early_stopped = False
        elif policy == "breathing":
            self._project_log_hypers_()
            self._breathing_optimize_lengthscales(
                self._Z_train_norm, self._y_train,
                n_steps=int(self.config.refit_opt_steps),
                prior_weight=float(self.config.lengthscale_prior_weight),
                prior_clip=float(self.config.lengthscale_prior_clip),
                verbose=verbose,
            )
        elif policy == "adaptive":
            # Adaptive: skip optimization unless the cloud has spread
            # significantly beyond the current kernel bandwidth on the
            # informative (non-pinned) axes.  Trigger detail:
            #
            # The fit_rms diagnostic alone is INSUFFICIENT: an ARD-RBF
            # kernel can interpolate labels at support points to
            # machine precision (fit_rms ~ 1e-9) even when the cloud has
            # bifurcated into multiple lobes — and in that regime the
            # kernel's *third derivatives* between lobes are unphysical
            # (this is what corrupts the Q operator).  The correct
            # signal is the ratio of cloud variance to kernel bandwidth
            # squared on each informative axis:
            #
            #     ratio_d = Var(Z[:, d]) / ℓ_d²
            #
            # When ratio_d > target_ratio (default 4 — i.e. cloud std
            # has grown to >2ℓ_d), the kernel can no longer support
            # smooth interpolation across the cloud's full spread.
            # Trigger breathing.
            #
            # On pinned axes (e.g. focused mode's mapping axes) Var is
            # locked by the Casimir geometry, so the ratio is bounded
            # and breathing on those axes is suppressed by the pin
            # mechanism anyway — but we exclude them from the trigger
            # check for clarity.
            self._project_log_hypers_()
            Z_np = _as_numpy(self._Z_train_norm if self.config.feature_zscore
                             else self._Z_train)
            ell_np = self.lengthscales
            informative = ~self._pin_mask
            target_ratio = float(getattr(self.config,
                                         "adaptive_cloud_ratio_target", 4.0))
            cooldown = int(max(1, getattr(self.config,
                                          "adaptive_cooldown", 20)))
            # Initialize per-instance cooldown counter on first hit.
            if not hasattr(self, "_adaptive_cooldown_remaining"):
                self._adaptive_cooldown_remaining: int = 0
            # Decrement cooldown each refit; only check the trigger when
            # cooldown has expired.
            trigger_condition = False
            if int(informative.sum()) > 0 and Z_np.shape[0] >= 2:
                var_d = Z_np.var(axis=0)
                ratios = var_d[informative] / (ell_np[informative] ** 2 + 1e-30)
                trigger_condition = bool(np.any(ratios > target_ratio))
            triggered = bool(trigger_condition and
                             self._adaptive_cooldown_remaining <= 0)
            if self._adaptive_cooldown_remaining > 0:
                self._adaptive_cooldown_remaining -= 1
            self._adaptive_triggered_last_refit: bool = triggered
            if triggered:
                # Breathe just enough to recover sub-target fit.  The
                # `_breathing_optimize_lengthscales` routine will respect
                # the LabelInformation pin (mapping ℓ stays anchored) and
                # zero gradients on pinned axes.  α₀ is rebuilt by the
                # post-policy block below.
                #
                # The adaptive policy uses a smaller per-trigger LBFGS
                # budget than the breathing policy: adaptive may fire
                # often, so each burst must be cheap.  5 LBFGS outer
                # iterations is enough to move ℓ_d by ~0.5 in log space
                # per refit, which tracks the cloud spread evolution.
                adaptive_steps = int(getattr(self.config,
                                             "adaptive_opt_steps", 5))
                self._breathing_optimize_lengthscales(
                    self._Z_train_norm, self._y_train,
                    n_steps=adaptive_steps,
                    prior_weight=float(self.config.lengthscale_prior_weight),
                    prior_clip=float(self.config.lengthscale_prior_clip),
                    verbose=verbose,
                )
                # Start cooldown: don't re-trigger for the next
                # `cooldown` refits, even if ratios still exceed target.
                self._adaptive_cooldown_remaining = cooldown
            else:
                # No-op path: record zero-cost diagnostics, matching
                # what the "frozen" branch produces.
                self.last_opt_total_loss = float("nan")
                self.last_opt_nll_loss = float("nan")
                self.last_opt_reg_loss = float("nan")
                self.last_opt_steps = 0
                self.last_opt_best_step = -1
                self.last_opt_early_stopped = False
        else:   # "free"
            self._project_log_hypers_()
            if optimize:
                self.optimize_hyperparameters(self._Z_train_norm, self._y_train,
                                              verbose=verbose)

        self._compute_unconstrained_alpha()
        self.last_free_fit_rms, self.last_free_fit_mae, self.last_free_fit_r2 = self._training_fit_metrics(self._alpha0)

        if moment_targets is None:
            moment_targets = {}
        allowed: Tuple[MomentName, ...] = ("normalization", "trace", "energy")
        ordered = {m: float(moment_targets[m]) for m in allowed if m in moment_targets}
        if apply_constraints is not None:
            old = self.config.constraints_enabled
            self.config.constraints_enabled = bool(apply_constraints)
            try:
                self._apply_kkt_projection(ordered)
            finally:
                self.config.constraints_enabled = old
        else:
            self._apply_kkt_projection(ordered)

        self.last_fit_rms, self.last_fit_mae, self.last_fit_r2 = self._training_fit_metrics(self._alpha)
        self.constraint_delta_rmse = self.last_fit_rms - self.last_free_fit_rms
        self.constraint_delta_mae = self.last_fit_mae - self.last_free_fit_mae
        self.constraint_delta_r2 = self.last_fit_r2 - self.last_free_fit_r2

    # ----- prediction ----------------------------------------------------
    def _require_fit(self) -> None:
        if self._alpha is None or self._Z_train is None or self._Z_train_norm is None:
            raise RuntimeError("GPDensity is not fitted; call fit() first.")

    def predict(self, Z: ArrayLike) -> FloatArray:
        """Posterior mean density ρ̂(z*) = k(z*, Z_train) @ α."""
        self._require_fit()
        Z_np = np.asarray(Z, dtype=np.float64)
        single = (Z_np.ndim == 1)
        if single:
            if Z_np.shape[0] != D:
                raise ValueError(f"Single z must have size {D}; got {Z_np.shape}.")
            Z_np = Z_np.reshape(1, D)
        elif Z_np.ndim != 2 or Z_np.shape[1] != D:
            raise ValueError(f"Z must have shape (N, {D}); got {Z_np.shape}.")

        Z_t = _as_tensor(self._normalize_features_np(Z_np))
        with torch.no_grad():
            k_star = _ard_gram(Z_t, self._Z_train_norm,
                               self.log_sigma_f, self.log_lengthscales)
            mean = k_star @ self._alpha
        out = _as_numpy(mean)
        return out[0] if single else out

    def predict_with_variance(self, Z: ArrayLike) -> Tuple[FloatArray, FloatArray]:
        """
        Return (mean, variance) at the query points.

        NB: the variance is the *unconstrained* GP posterior variance.
        Incorporating the KKT constraints into the posterior covariance
        is possible but not yet implemented; the mean is the full
        constrained projection.
        """
        self._require_fit()
        Z_np = np.asarray(Z, dtype=np.float64).reshape(-1, D)
        Z_t = _as_tensor(self._normalize_features_np(Z_np))
        with torch.no_grad():
            k_star = _ard_gram(Z_t, self._Z_train_norm,
                               self.log_sigma_f, self.log_lengthscales)      # (M, N)
            mean = k_star @ self._alpha

            v = torch.cholesky_solve(k_star.t(), self._L_Ky)                 # (N, M)
            diag_kk = torch.exp(2.0 * self.log_sigma_f) * torch.ones(
                Z_t.shape[0], dtype=_DEFAULT_DTYPE)                          # k(z*,z*)
            var = diag_kk - torch.sum(k_star * v.t(), dim=1)
            var = var.clamp_min(0.0)
        return _as_numpy(mean), _as_numpy(var)

    def solve_K(self, B: ArrayLike) -> FloatArray:
        """
        Solve the GP gram-matrix system  K x = B  with the cached
        Cholesky factor (regularised by σ_n² as built during fit).

        Required by Dynamics.MidpointScheme to build the linear label-
        ODE generator  A = -L K⁻¹  where L = compute_L_matrix(...).

        Parameters
        ----------
        B : (N,) or (N, M) array.

        Returns
        -------
        x : same shape as B.
        """
        if self._L_Ky is None or self._Z_train is None:
            raise RuntimeError("GP must be fitted before solve_K.")
        B_np = np.asarray(B, dtype=np.float64)
        squeeze_back = (B_np.ndim == 1)
        B_t = _as_tensor(B_np if not squeeze_back else B_np[:, None])
        with torch.no_grad():
            x_t = torch.cholesky_solve(B_t, self._L_Ky)
        x = _as_numpy(x_t)
        return x[:, 0] if squeeze_back else x

    # ----- diagnostics ---------------------------------------------------
    def compute_moment_values(self) -> Dict[str, float]:
        """
        Compute ⟨ψ_i⟩ under the current surrogate, for all three physical
        moments (not just the constrained ones). Useful for diagnostics
        and unit tests.

        Result is also cached in self._last_norm_value for cheap access via
        the `normalization` property, avoiding repeated O(N) evaluations
        when the same value is needed multiple times in one step.
        """
        self._require_fit()
        all_moments: Tuple[MomentName, ...] = ("normalization", "trace", "energy")
        A_np = self.moment_integrator.build_A(
            Z_train=_as_numpy(self._Z_train),
            log_sigma_f=float(self.log_sigma_f.item()),
            log_lengthscales=np.log(self.lengthscales),
            moments=all_moments,
        )
        alpha_np = _as_numpy(self._alpha)
        vals = A_np @ alpha_np
        result = {name: float(v) for name, v in zip(all_moments, vals)}
        self._last_norm_value: float = result["normalization"]
        return result

    @property
    def normalization(self) -> float:
        """Return the cached GP normalization ∫ρ̂ dz from the last call to
        compute_moment_values().  If never called, computes it now.
        Use this instead of compute_moment_values()['normalization'] when
        only the normalization is needed to avoid redundant O(N) evaluations.
        """
        if not hasattr(self, "_last_norm_value"):
            self.compute_moment_values()
        return self._last_norm_value

    def report(self) -> str:
        lines = [
            "GPDensity state",
            f"  n_train         = {self.n_train}",
            f"  σ_f             = {self.sigma_f:.6e}",
            f"  σ_n             = {self.sigma_n:.6e}",
            f"  lengthscales    = {self.lengthscales}",
            f"  constrained     = {self._moment_order}",
            f"  free_fit_rms    = {self.last_free_fit_rms:.6e}",
            f"  free_fit_mae    = {self.last_free_fit_mae:.6e}",
            f"  fit_rms         = {self.last_fit_rms:.6e}",
            f"  fit_mae         = {self.last_fit_mae:.6e}",
        ]
        if self._alpha is not None:
            moms = self.compute_moment_values()
            lines.append(f"  ⟨normalization⟩ = {moms['normalization']:.6e}")
            lines.append(f"  ⟨trace⟩         = {moms['trace']:.6e}")
            lines.append(f"  ⟨energy⟩        = {moms['energy']:.6e}")
        return "\n".join(lines)


# =============================================================================
# Self-tests
# =============================================================================

def _self_test() -> None:
    """
    End-to-end consistency test:

      1. Draw SEO-signed samples at t = 0 from MMSTSampler.
      2. Compute E_0 from the initial ensemble.
      3. Fit GPDensity with all three moment constraints.
      4. Verify that the constrained moments are reproduced to high accuracy.
      5. Refit after 10 MInt steps and verify the energy moment still lands
         on E_0 (Liouville-preserved y, KKT-projected).
      6. Predict on training points; compare to the noisy GP posterior.
    """
    import time
    from .Sampling import (GaussianWavePacketParams, MappingInitParams,
                          MMSTSampler)
    from .Mint import PBMEMIntParams, pack_z
    from .Models import TullyModel, TullyParams

    rng = np.random.default_rng(0)

    # --- 1. pipeline objects ----------------------------------------------
    model = TullyModel(TullyParams.defaults("dual"))
    dynamics = PBMEMIntDynamics(model=model,
                                params=PBMEMIntParams(mass=2000.0, hbar=1.0))

    classical_params = GaussianWavePacketParams(
        R0=[-8.0], P0=[30.0], sigma_R=[1.0], hbar=1.0)
    mapping_params = MappingInitParams(nstates=2, init_state=0, hbar=1.0, gamma=0.5)
    sampler = MMSTSampler(classical_params, mapping_params)

    # --- 2. initial support points ----------------------------------------
    n_train = 400
    s = sampler.sample_seo_signed(n_train, rng=rng)
    Z0 = pack_z(s.R, s.P, s.r, s.p)
    y0 = s.target_density

    E0 = float(np.dot(s.weight, dynamics.energy(Z0)) / np.sum(s.weight))
    print(f"[self-test] n_train = {n_train}   E0 = {E0:.6e}")
    print(f"[self-test] y0 range = [{y0.min():.3e}, {y0.max():.3e}]   "
          f"fraction negative = {np.mean(y0 < 0):.3f}")

    # --- 3. fit ------------------------------------------------------------
    cfg = GPDensityConfig(n_opt_steps=300, lr=5.0e-2,
                          fix_sigma_n=True, init_log_sigma_n=-6.0,
                          use_loocv=True,
                          reinit_lengthscales=True)
    gp = GPDensity(cfg, dynamics=dynamics)

    t0 = time.time()
    gp.fit(Z_train=Z0, y_train=y0,
           moment_targets={"normalization": 1.0, "trace": 1.0, "energy": E0},
           optimize=True, verbose=False)
    t1 = time.time()
    print(f"[self-test] fit time = {t1 - t0:.2f} s")
    print(gp.report())

    moms = gp.compute_moment_values()
    print(f"[self-test] moment residuals (|⟨ψ⟩ - target|):")
    print(f"            normalization : {abs(moms['normalization'] - 1.0):.3e}")
    print(f"            trace         : {abs(moms['trace']         - 1.0):.3e}")
    print(f"            energy        : {abs(moms['energy']        - E0):.3e}")

    # --- 4. predict on training points and on a held-out random batch -----
    rho_hat_train = gp.predict(Z0)
    rms_train = float(np.sqrt(np.mean((rho_hat_train - y0)**2)))
    print(f"[self-test] training RMS : {rms_train:.3e}  "
          f"(vs std(y0) = {np.std(y0):.3e})")

    # held-out: draw fresh samples from the same initial density, evaluate
    s_val = sampler.sample_seo_signed(200, rng=rng)
    Z_val = pack_z(s_val.R, s_val.P, s_val.r, s_val.p)
    y_val = s_val.target_density
    rho_hat_val = gp.predict(Z_val)
    rms_val = float(np.sqrt(np.mean((rho_hat_val - y_val)**2)))
    print(f"[self-test] held-out RMS : {rms_val:.3e}")

    # --- 5. refit after propagation (Liouville: y unchanged) --------------
    n_steps = 10
    dt = 0.2
    Z_k = dynamics.propagate(Z0, dt=dt, n_steps=n_steps)[-1]
    # Energy of propagated ensemble must (nearly) equal E0 for symplectic MInt
    E_k = float(np.mean(dynamics.energy(Z_k)))
    print(f"[self-test] after {n_steps} MInt steps: E_k = {E_k:.6e}  "
          f"(ΔE = {E_k - E0:.2e})")

    gp.refit(Z_train=Z_k, y_train=y0,
             moment_targets={"normalization": 1.0, "trace": 1.0, "energy": E0},
             optimize=True, verbose=False)
    print(gp.report())
    moms_k = gp.compute_moment_values()
    print(f"[self-test] post-refit moment residuals:")
    print(f"            normalization : {abs(moms_k['normalization'] - 1.0):.3e}")
    print(f"            trace         : {abs(moms_k['trace']         - 1.0):.3e}")
    print(f"            energy        : {abs(moms_k['energy']        - E0):.3e}")


# =============================================================================
# Reference-profile ("product") surrogate  rho_hat(z) = g(x) * mu(z)
# =============================================================================
# Merged from the former GP_DensityProduct module (2026-07-05) to keep the
# pipeline's original file structure.  The excess mapping-QCLE operator
# contracts SECOND MAPPING DERIVATIVES of the density; focused sampling
# carries no radial mapping information, so a plain GP's mapping curvature is
# an anchor-lengthscale artifact and the operator input is suppressed (~527x
# vs the exact analytic iL' rho).  Factoring rho_hat = g(x)*mu(z) with g the
# analytic SEO mapping profile makes that curvature exact; the GP carries only
# the smooth modulation.  The Leibniz-rule operator/flux that consume this
# surrogate live in Operator.py (product_Q_at_points / product_flux_at_points).
# GP-analytic moments are NaN by design in this surrogate (see the class).
# =============================================================================
# =============================================================================
# Analytic SEO mapping profile and its derivatives
# =============================================================================

def seo_profile_derivs(
    x: ArrayLike,
    hbar: float = 1.0,
    init_state: int = 0,
    nstates: int = 2,
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """
    g, dg/dx (N,4), d2g/dx dx (N,4,4) for the SEO Wigner profile of the
    occupied state.  Mapping coordinate layout x = (r0, r1, p0, p1);
    occupied-state coordinates are x-indices (init_state, 2+init_state).

        g = A * E * q,   A = (pi hbar)^{-nstates},
        E = exp(-|x|^2/hbar),
        q = (2/hbar)(x_s^2 + x_{2+s}^2) - 1 .
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    N = x.shape[0]
    A = (np.pi * hbar) ** (-nstates)
    E = np.exp(-np.sum(x * x, axis=1) / hbar)                       # (N,)
    ar, ap = init_state, 2 + init_state
    q = (2.0 / hbar) * (x[:, ar] ** 2 + x[:, ap] ** 2) - 1.0        # (N,)

    # dq/dx_a: (4/hbar) x_a on occupied indices, else 0.
    dq = np.zeros((N, 4))
    dq[:, ar] = (4.0 / hbar) * x[:, ar]
    dq[:, ap] = (4.0 / hbar) * x[:, ap]
    # d2q: (4/hbar) delta_ab on occupied diagonal.
    d2q = np.zeros((N, 4, 4))
    d2q[:, ar, ar] = 4.0 / hbar
    d2q[:, ap, ap] = 4.0 / hbar

    # dE/dx_a = -(2 x_a/hbar) E ; d2E = [ (4 x_a x_b/hbar^2)
    #                                    - (2/hbar) delta_ab ] E
    two_x = (2.0 / hbar) * x                                        # (N,4)
    g   = A * E * q
    dg  = A * E[:, None] * (dq - two_x * q[:, None])                # (N,4)
    eye = np.eye(4)[None, :, :]
    d2g = A * E[:, None, None] * (
        d2q
        - two_x[:, :, None] * dq[:, None, :]
        - two_x[:, None, :] * dq[:, :, None]
        + (two_x[:, :, None] * two_x[:, None, :]
           - (2.0 / hbar) * eye) * q[:, None, None]
    )                                                               # (N,4,4)
    return g, dg, d2g


def _g_safe(g: FloatArray, floor_rel: float) -> FloatArray:
    """Signed floor: |g| >= floor_rel * max|g| (transform regularizer)."""
    floor = floor_rel * float(np.max(np.abs(g))) if g.size else 0.0
    return np.where(np.abs(g) >= floor, g, np.where(g >= 0.0, floor, -floor))


# =============================================================================
# The product surrogate
# =============================================================================

class GPDensityProduct:
    """
    rho_hat(z) = g(x) * mu(z), mu the inner GPDensity fitted to y/g.

    Delegates every attribute it does not override to the inner GP (so
    Collector, diagnostics, and deepcopy in MidpointScheme work
    unchanged); overrides the label transform, prediction, and the
    moment interface.  ``_is_product = True`` is the dispatch flag used
    by Operator.compute_Q_at_points / compute_flux_at_points and
    Dynamics.MidpointScheme._rho_at.
    """

    _is_product = True

    def __init__(self, gp, hbar: float = 1.0, init_state: int = 0,
                 nstates: int = 2, g_floor_rel: float = 1.0e-3) -> None:
        # NOTE: use object.__setattr__-free plain attributes; __getattr__
        # only fires for MISSING attributes, so overrides below win.
        self._inner = gp
        self._hbar = float(hbar)
        self._init_state = int(init_state)
        self._nstates = int(nstates)
        self._g_floor_rel = float(g_floor_rel)

    # -- delegation ------------------------------------------------------
    def __getattr__(self, name):
        # Called only when normal lookup fails -> delegate to inner GP.
        return getattr(self._inner, name)

    def __deepcopy__(self, memo):
        new = GPDensityProduct.__new__(GPDensityProduct)
        new._inner = copy.deepcopy(self._inner, memo)
        new._hbar = self._hbar
        new._init_state = self._init_state
        new._nstates = self._nstates
        new._g_floor_rel = self._g_floor_rel
        new._footpoints = (None if getattr(self, "_footpoints", None) is None
                           else self._footpoints.copy())
        new._foot_jac = (None if getattr(self, "_foot_jac", None) is None
                         else self._foot_jac.copy())
        return new

    # -- profile helpers ---------------------------------------------------
    def profile_at(self, Z: ArrayLike) -> FloatArray:
        """
        g at the current point.  In the STATIC mode (default) this is the
        t=0 SEO profile g(x) evaluated at the current mapping coordinates.
        In the TRANSPORTED mode (Rung 2, enabled by attach_footpoints /
        transport_footpoints), the profile rides the MInt flow: it is the
        birth-time profile g(x^0) evaluated at each support point's stored
        footpoint mapping coordinate, so the analytic curvature the QCLE
        operator differentiates follows the Hamiltonian backbone exactly
        rather than being frozen at its t=0 shape.
        """
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        fp = getattr(self, "_footpoints", None)
        if fp is not None:
            if fp.shape[0] != Z.shape[0]:
                raise ValueError(
                    f"TRANSPORTED profile: footpoint records are matched to "
                    f"rows POSITIONALLY, but got {Z.shape[0]} query rows vs "
                    f"{fp.shape[0]} footpoints.  Evaluating the transported "
                    f"profile at a subset, grid, or reordered point set is "
                    f"ill-defined without an explicit row->trajectory map.  "
                    f"(Previously this fell back SILENTLY to the static "
                    f"profile — wrong by design; found 2026-07-10 via an "
                    f"FD check on a support subset.)  Pass the full aligned "
                    f"support cloud, or use profile_at_footindex().")
            g, _, _ = seo_profile_derivs(fp[:, 2:6], self._hbar,
                                         self._init_state, self._nstates)
            return g
        g, _, _ = seo_profile_derivs(Z[:, 2:6], self._hbar,
                                     self._init_state, self._nstates)
        return g

    def profile_at_footindex(self, Z: ArrayLike, foot_index) -> FloatArray:
        """Transported-profile value for query rows Z that correspond to
        footpoint records ``foot_index`` (explicit row->trajectory map, for
        subset evaluations).  Static mode ignores the index."""
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        fp = getattr(self, "_footpoints", None)
        if fp is None:
            g, _, _ = seo_profile_derivs(Z[:, 2:6], self._hbar,
                                         self._init_state, self._nstates)
            return g
        idx = np.asarray(foot_index, dtype=int).reshape(-1)
        if idx.shape[0] != Z.shape[0]:
            raise ValueError("foot_index length must match Z rows.")
        g, _, _ = seo_profile_derivs(fp[idx][:, 2:6], self._hbar,
                                     self._init_state, self._nstates)
        return g

    def profile_derivs_current(self, Z: ArrayLike) -> Tuple[FloatArray,
                                                            FloatArray,
                                                            FloatArray]:
        """
        (g, dg, d2g) of the profile with respect to the CURRENT phase-space
        coordinates z = (R,P,r0,r1,p0,p1), returned as dg (N,6) and
        d2g (N,6,6) so the Leibniz operator can consume mapping AND
        bath-momentum profile derivatives uniformly.

        STATIC mode: g depends on mapping coords only; dg is nonzero on
        dims 2..5, d2g on the mapping block; all bath entries zero — this
        reproduces the original behaviour exactly.

        TRANSPORTED mode: g = g0(x^0(z)), x^0 the footpoint mapping
        coordinate.  The footpoint depends on the current point through the
        inverse MInt map, whose mapping-block Jacobian J = dx^0/dz (N,4,6)
        and Hessian are supplied by the transport bookkeeping.  Then
            dg/dz_a       = sum_m (dg0/dx^0_m) J_{m,a}
            d2g/dz_a dz_b = sum_mn (d2g0) J_{m,a} J_{n,b}
                            + sum_m (dg0/dx^0_m) H_{m,a,b}
        With the linear frozen-R mapping rotation, H is small; v1 transport
        keeps the first (Jacobian) term exactly and neglects the mixed
        Hessian term H (second order in the rotation angle per step), which
        is the same order already dropped by the O((iL')^2) midpoint
        truncation.
        """
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        N = Z.shape[0]
        fp = getattr(self, "_footpoints", None)
        Jm = getattr(self, "_foot_jac", None)     # (N,4,6) dx^0_map/dz

        if fp is not None and Jm is not None and fp.shape[0] != N:
            raise ValueError(
                f"TRANSPORTED profile derivatives: positional footpoint "
                f"match requires {fp.shape[0]} rows, got {N}.  A subset/"
                f"grid/reordered evaluation is ill-defined; the old code "
                f"fell back SILENTLY to the static profile here (found "
                f"2026-07-10).  Pass the full aligned support cloud.")
        if fp is None or Jm is None:
            # STATIC profile: derivatives only on the mapping block.
            g, dg4, d2g4 = seo_profile_derivs(Z[:, 2:6], self._hbar,
                                              self._init_state, self._nstates)
            dg = np.zeros((N, 6)); dg[:, 2:6] = dg4
            d2g = np.zeros((N, 6, 6)); d2g[:, 2:6, 2:6] = d2g4
            return g, dg, d2g

        # TRANSPORTED profile via chain rule through the stored footpoint
        # mapping Jacobian.
        g, dg4, d2g4 = seo_profile_derivs(fp[:, 2:6], self._hbar,
                                          self._init_state, self._nstates)
        dg = np.einsum("nm,nma->na", dg4, Jm)                     # (N,6)
        d2g = np.einsum("nmk,nma,nkb->nab", d2g4, Jm, Jm)         # (N,6,6)
        return g, dg, d2g

    # -- transport bookkeeping (Rung 2) -----------------------------------
    def attach_footpoints(self, Z0: ArrayLike) -> None:
        """
        Initialise transported-profile mode: record the t=0 support
        coordinates as footpoints and seed the footpoint->current mapping
        Jacobian as identity on the mapping block.  Call once at t=0 with
        the initial support cloud.
        """
        Z0 = np.atleast_2d(np.asarray(Z0, dtype=np.float64))
        self._footpoints = Z0.copy()
        N = Z0.shape[0]
        # dx^0_map/dz : (N,4,6); at t=0 identity onto mapping dims 2..5.
        J = np.zeros((N, 4, 6))
        J[:, 0, 2] = 1.0; J[:, 1, 3] = 1.0
        J[:, 2, 4] = 1.0; J[:, 3, 5] = 1.0
        self._foot_jac = J

    def transport_footpoints(self, B_map: ArrayLike) -> None:
        """
        Advance the footpoint mapping Jacobian by one MInt step.  B_map is
        the (N,4,4) forward mapping-block Jacobian dx'_map/dx_map of the
        step just applied to the support points (the linear frozen-R
        rotation).  The footpoint map composes inversely:
            J_new = J_old @ B_map^{-1}   (acting on mapping dims of z)
        The footpoint COORDINATES themselves stay fixed (a footpoint is a
        birth-time label); only the Jacobian relating current-point
        variations to footpoint variations evolves.
        """
        if getattr(self, "_foot_jac", None) is None:
            return
        B = np.asarray(B_map, dtype=np.float64)
        Binv = np.linalg.inv(B)                                   # (N,4,4)
        # J is (N,4,6): footpoint-map derivative wrt z.  Only its mapping
        # columns (2..5) rotate; bath columns stay zero under the linear
        # frozen-R map.
        Jm = self._foot_jac[:, :, 2:6]                            # (N,4,4)
        self._foot_jac[:, :, 2:6] = np.einsum("nij,njk->nik", Binv, Jm)

    def _transform_labels(self, Z: ArrayLike, y: ArrayLike) -> FloatArray:
        g = self.profile_at(Z)
        gs = _g_safe(g, self._g_floor_rel)
        return np.asarray(y, dtype=np.float64).reshape(-1) / gs

    # -- fit / refit / predict --------------------------------------------
    @staticmethod
    def _strip_moment_targets(kwargs: dict) -> dict:
        """Chapter 4 contract: NO hard moment (KKT) correction on the product
        surrogate.  The inner mu-GP's KKT rows are vanilla kernel integrals
        (no profile g) with physical targets — enforcing them would constrain
        int psi mu = b_phys while the physical moment is int psi g mu, an
        O(1) unit/structure mismatch.  Focused sampling already skips KKT via
        LabelInformation.apply_kkt=False; this strip makes the product
        surrogate safe under ANY sampling mode (e.g. seo_signed, where
        apply_kkt=True would silently mis-project the inner alpha)."""
        if kwargs.get("moment_targets") is not None:
            kwargs = dict(kwargs)
            kwargs["moment_targets"] = None
        return kwargs

    def fit(self, Z_train, y_train, *args, **kwargs):
        return self._inner.fit(Z_train,
                               self._transform_labels(Z_train, y_train),
                               *args, **self._strip_moment_targets(kwargs))

    def refit(self, Z_train, y_train, *args, **kwargs):
        return self._inner.refit(Z_train,
                                 self._transform_labels(Z_train, y_train),
                                 *args, **self._strip_moment_targets(kwargs))

    def predict_with_variance(self, Z: ArrayLike):
        """Density-space (mean, variance): mean = g*mu; variance = g^2 Var[mu]
        (fixed profile scales the modulation covariance, Sigma_rho = g g'
        Sigma_f — thesis Ch.4 Eq. gp-product-density-covariance).  Without
        this override, __getattr__ delegation returned the MODULATION
        variance, mis-stating density uncertainty by the factor g^2."""
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        mu, var = self._inner.predict_with_variance(Z)
        g = self.profile_at(Z)
        return g * np.asarray(mu).reshape(-1), \
               (g * g) * np.asarray(var).reshape(-1)

    def predict(self, Z: ArrayLike) -> FloatArray:
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        mu = np.asarray(self._inner.predict(Z),
                        dtype=np.float64).reshape(-1)
        return self.profile_at(Z) * mu

    # -- moments: exact closed form via ProductMoments (2026-07-10) --------
    # (replaces the V1 NaN stub; the NaN design predated ProductMoments.
    #  Raw contract matches vanilla compute_moment_values: UNNORMALIZED
    #  kernel integrals of {1, c00+c11, H} against rho_hat = g*mu.)
    def compute_moment_values(self, *args, **kwargs) -> Dict[str, float]:
        from .ProductMoments import product_kkt_moments
        km = product_kkt_moments(self)
        return {
            "normalization": km["normalization_raw"],
            "trace":         km["trace_raw"],
            "energy":        km["energy_raw"],
        }

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    torch.set_default_dtype(_DEFAULT_DTYPE)
    _self_test()
