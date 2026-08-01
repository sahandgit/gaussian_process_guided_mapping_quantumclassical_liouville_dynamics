from __future__ import annotations

"""
Dynamics.py
===========

Lean time-stepping module.

Contents
--------
*   SimulationState       : current (Z, y, GP, t, step_index).
*   DynamicsConfig        : run-level knobs.
*   DynamicsScheme (ABC)  : per-step update interface.
*   PBMEScheme            : pure Liouville transport (y unchanged).
*   MidpointScheme        : PBME transport + QCLE midpoint correction.
*   Simulation            : orchestrates: propagate, measure, record.
*   build_scheme          : factory ("pbme" | "midpoint") → scheme.

Module responsibilities
-----------------------
*   Operator.py      builds the geometric data (Y, ∂_R h̄, A, B, C).
*   GPDerivatives.py supplies ρ_{,a}, ρ_{,ab}, ρ_{,abc} at Y.
*   Dynamics.py      (this file) performs the final tensor contraction
                     and advances the simulation state.
*   Observables.py   computes physics diagnostics after each step.
*   Collector.py     serialises diagnostics and GP snapshots.
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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import copy
import time

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from .Mint import D, pack_z, PBMEMIntDynamics
from .GP_Density import GPDensity, GPDensityConfig
from .Sampling import (
    GaussianWavePacketParams, MappingInitParams, MMSTSampler,
)
from .Operator import QCLECorrection, compute_L_matrix, compute_flux_at_points
# NOTE (corrected 2026-07): the comment that used to sit here claimed
# "Operator.compute_Q handles the full chain rule via JAX autodiff" and that
# GPDerivatives was therefore unused in production.  Neither claim is
# accurate: compute_Q/compute_Q_at_points use the closed-form intrinsic-Y
# derivative path (no autodiff-on-composition — see Operator.py's module
# docstring), and GPDerivatives.rho_derivative_bundle IS now used in
# production below, to evaluate grad(rho_hat) for the flow-correction
# displacement.
from .GPDerivatives import rho_derivative_bundle
from .Monodromy import _I_P
from .Observables import compute_all
from .Collector import Collector, Snapshot, StepDiagnostics
from .Reproducibility import build_run_metadata


FloatArray = NDArray[np.float64]


def _signed_expectation(values: ArrayLike, weights: ArrayLike, *, name: str) -> float:
    """Monte Carlo expectation under the signed target density."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if v.shape != w.shape:
        raise ValueError(f"{name}: values and weights must have matching shapes; got {v.shape} and {w.shape}.")
    denom = float(np.sum(w))
    if not np.isfinite(denom) or abs(denom) <= 1.0e-14:
        raise ValueError(f"{name}: signed weight sum is too small or non-finite ({denom!r}).")
    numer = float(np.dot(w, v))
    return numer / denom


def _signed_weight_summary(weights: Optional[ArrayLike]) -> Dict[str, float]:
    """Diagnostics of the original sampling weights, distinct from carried labels y."""
    if weights is None:
        return {}
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    n = int(w.size)
    if n == 0:
        return {}
    S = float(np.sum(w))
    S_abs = float(np.sum(np.abs(w)))
    S2 = float(np.sum(w * w))
    if S2 <= 0.0 or not np.isfinite(S2):
        ess = float("nan")
        abs_ess = float("nan")
    else:
        ess = (S * S) / S2
        abs_ess = (S_abs * S_abs) / S2
    chi = abs(S) / S_abs if S_abs > 0.0 and np.isfinite(S_abs) else float("nan")
    return {
        "weight_sum": S,
        "abs_weight_sum": S_abs,
        "ess": float(ess),
        "ess_frac": float(ess / n) if np.isfinite(ess) else float("nan"),
        "abs_ess": float(abs_ess),
        "abs_ess_frac": float(abs_ess / n) if np.isfinite(abs_ess) else float("nan"),
        "cancel_ratio": float(chi),
        "neg_frac": float(np.mean(w < 0.0)),
        "pos_frac": float(np.mean(w > 0.0)),
    }



def _support_mapping_observables(Z: ArrayLike, hbar: float) -> Dict[str, FloatArray]:
    """Physical MMST observables evaluated directly on the support cloud."""
    Zb = np.asarray(Z, dtype=np.float64).reshape(-1, D)
    r0 = Zb[:, 2]; r1 = Zb[:, 3]
    p0 = Zb[:, 4]; p1 = Zb[:, 5]
    P0 = (r0 * r0 + p0 * p0 - hbar) / (2.0 * hbar)
    P1 = (r1 * r1 + p1 * p1 - hbar) / (2.0 * hbar)
    coh_re = (r0 * r1 + p0 * p1) / (2.0 * hbar)
    coh_im = (r0 * p1 - r1 * p0) / (2.0 * hbar)
    return {
        "P0": P0,
        "P1": P1,
        "trace": P0 + P1,
        "coh_re": coh_re,
        "coh_im": coh_im,
    }


def _weighted_support_diagnostics(Z: ArrayLike,
                                  dynamics: PBMEMIntDynamics,
                                  omega: Optional[ArrayLike],
                                  y:     Optional[ArrayLike]) -> Dict[str, float]:
    """
    Cloud Riemann-sum diagnostics on the transported support cloud.

    Every quantity here is a phase-space integral evaluated as

        ⟨A⟩(t) = Σ_i ω_i A(z_i(t)) y_i(t),

    with the FROZEN geometric measure ω_i = 1/(N q(z_i^0)) and the live
    density values y_i(t) (frozen at ρ_0 for PBME, QCLE-corrected for
    midpoint).  No self-normalization, no IS reweighting.

    The mapping-radius diagnostic is a sharp physics control: MInt
    conserves r_0² + p_0² + r_1² + p_1² along EACH trajectory exactly,
    so its density-weighted expectation is conserved exactly under PBME
    and conserved up to QCLE corrections under midpoint.
    """
    if omega is None or y is None:
        return {}
    Zb     = np.asarray(Z,     dtype=np.float64).reshape(-1, D)
    omega_ = np.asarray(omega, dtype=np.float64).reshape(-1)
    y_     = np.asarray(y,     dtype=np.float64).reshape(-1)
    oy     = omega_ * y_                                     # (N,)

    E   = np.asarray(dynamics.energy(Zb), dtype=np.float64).reshape(-1)
    obs = _support_mapping_observables(Zb, hbar=float(dynamics.params.hbar))

    mapping_radius_sq = (Zb[:, 2] ** 2 + Zb[:, 3] ** 2
                         + Zb[:, 4] ** 2 + Zb[:, 5] ** 2)
    R_arr = Zb[:, 0]; P_arr = Zb[:, 1]

    out = {
        "cloud_weighted_energy":            float(np.dot(oy, E)),
        "cloud_weighted_P0":                float(np.dot(oy, obs["P0"])),
        "cloud_weighted_P1":                float(np.dot(oy, obs["P1"])),
        "cloud_weighted_trace":             float(np.dot(oy, obs["trace"])),
        "cloud_weighted_coh_re":            float(np.dot(oy, obs["coh_re"])),
        "cloud_weighted_coh_im":            float(np.dot(oy, obs["coh_im"])),
        "cloud_weighted_mapping_radius_sq": float(np.dot(oy, mapping_radius_sq)),
        "cloud_weighted_R_mean":            float(np.dot(oy, R_arr)),
        "cloud_weighted_P_mean":            float(np.dot(oy, P_arr)),
        "cloud_weighted_R_sq":              float(np.dot(oy, R_arr * R_arr)),
        "cloud_weighted_P_sq":              float(np.dot(oy, P_arr * P_arr)),
    }
    out["cloud_weighted_R_var"] = out["cloud_weighted_R_sq"] - out["cloud_weighted_R_mean"] ** 2
    out["cloud_weighted_P_var"] = out["cloud_weighted_P_sq"] - out["cloud_weighted_P_mean"] ** 2

    # ------------------------------------------------------------------
    # Self-normalised label-weighted estimators  lw_*
    # ------------------------------------------------------------------
    # The raw cloud_weighted_* quantities are Riemann sums
    #     Σ_i ω_i y_i(t) A(z_i(t))
    # without dividing by ∫ρ̂ dz ≈ Σ_i ω_i y_i(t).
    # Under the midpoint QCLE scheme, y_i accumulates corrections and
    # Σ ω_i y_i can drift from 1.  The self-normalised version
    #     lw_A = Σ ω_i y_i A_i / Σ ω_i y_i
    # is the proper density-weighted expectation and remains bounded
    # by the Casimir/KKT structure even when labels drift.
    cloud_norm = float(np.sum(oy))      # Σ ω_i y_i  ≈  ∫ ρ̂ dz  (should be ≈ 1)
    out["cloud_norm"] = cloud_norm
    _D = cloud_norm if abs(cloud_norm) > 1.0e-15 else 1.0

    out["lw_P0"]              = out["cloud_weighted_P0"]              / _D
    out["lw_P1"]              = out["cloud_weighted_P1"]              / _D
    out["lw_P_sum"]           = (out["cloud_weighted_P0"]
                                 + out["cloud_weighted_P1"])           / _D
    out["lw_trace"]           = out["cloud_weighted_trace"]           / _D
    out["lw_energy"]          = out["cloud_weighted_energy"]          / _D
    out["lw_coh_re"]          = out["cloud_weighted_coh_re"]          / _D
    out["lw_coh_im"]          = out["cloud_weighted_coh_im"]          / _D
    out["lw_mapping_radius_sq"] = out["cloud_weighted_mapping_radius_sq"] / _D
    out["lw_R_mean"]          = out["cloud_weighted_R_mean"]          / _D
    out["lw_P_mean"]          = out["cloud_weighted_P_mean"]          / _D

    return out


# =============================================================================
# Statistical health helpers
# =============================================================================

def _cloud_center_variance(Z: ArrayLike) -> Dict[str, FloatArray]:
    """Sample mean/variance of the current support cloud in each phase-space dimension."""
    Zb = np.asarray(Z, dtype=np.float64).reshape(-1, D)
    return {
        "mean": np.mean(Zb, axis=0),
        "var": np.var(Zb, axis=0),
        "std": np.std(Zb, axis=0),
        "min": np.min(Zb, axis=0),
        "max": np.max(Zb, axis=0),
    }


def _gp_signal_noise_diagnostics(gp: GPDensity, y_pred: ArrayLike) -> Dict[str, float]:
    """Signal/noise summaries for the current surrogate.

    Important: sigma_f lives in the GP's normalized target space, while sigma_n
    is often reported in physical units.  To avoid a misleading constant SNR,
    we report both normalized-space and physical-space diagnostics explicitly.
    """
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    sigma_f_norm = float(getattr(gp, "sigma_f_normalized", gp.sigma_f))
    sigma_n_norm = float(max(getattr(gp, "sigma_n_normalized", gp.sigma_n), 1.0e-300))
    sigma_n_raw = float(max(gp.sigma_n, 1.0e-300))
    pred_rms = float(np.sqrt(np.mean(yp ** 2))) if yp.size else float('nan')
    pred_abs_mean = float(np.mean(np.abs(yp))) if yp.size else float('nan')
    return {
        "sigma_f_normalized": sigma_f_norm,
        "sigma_n_normalized": sigma_n_norm,
        "sigma_f_over_sigma_n": sigma_f_norm / sigma_n_norm,
        "pred_rms_over_sigma_n": pred_rms / sigma_n_raw if np.isfinite(pred_rms) else float('nan'),
        "pred_abs_mean_over_sigma_n": pred_abs_mean / sigma_n_raw if np.isfinite(pred_abs_mean) else float('nan'),
    }


def _gp_surrogate_center_variance(gp: GPDensity) -> Dict[str, FloatArray]:
    """Analytic first and second coordinate moments of the GP surrogate in physical coordinates.

    When the signed-weight sum (normalization) is near zero due to ESS collapse,
    the moment integrals are dominated by numerical noise.  In that regime we
    return NaN for var/std rather than a clamped negative number, so downstream
    code (Visualization, trust-region checks) sees an explicit signal rather
    than a silently wrong value.
    """
    if gp._alpha is None or gp._Z_train is None:
        raise RuntimeError('GP must be fitted before surrogate moment diagnostics.')
    alpha = np.asarray(gp._alpha.detach().cpu().numpy() if hasattr(gp._alpha, 'detach') else gp._alpha, dtype=np.float64).reshape(-1)
    Ztr_obj = gp.raw_training_centers if hasattr(gp, 'raw_training_centers') else gp._Z_train
    Ztr = np.asarray(Ztr_obj.detach().cpu().numpy() if hasattr(Ztr_obj, 'detach') else Ztr_obj, dtype=np.float64)
    ell = np.asarray(gp.lengthscales, dtype=np.float64).reshape(-1)
    G = (float(gp.sigma_f) ** 2) * float(np.prod(np.sqrt(2.0 * np.pi) * ell))
    aG = alpha * G
    norm_raw = float(getattr(gp, "normalization",
                             gp.compute_moment_values().get("normalization", 1.0)))
    # When norm_raw is near zero the signed-weight sum has collapsed due to
    # ESS failure.  Dividing by a near-zero norm produces moments that are
    # numerically unbounded; return NaN explicitly so callers can detect the
    # condition rather than receiving a clamped-negative variance that
    # propagates silently (e.g. as a negative gp_var_P in the log, or as
    # NaN from sqrt in Visualization.py).
    if abs(norm_raw) <= 1.0e-10:
        nan_vec = np.full(Ztr.shape[1], float("nan"), dtype=np.float64)
        return {"mean": nan_vec.copy(), "var": nan_vec.copy(),
                "std": nan_vec.copy(), "norm_raw": norm_raw}
    norm = norm_raw
    mean = np.sum(aG[:, None] * Ztr, axis=0) / norm
    second = np.sum(aG[:, None] * (Ztr ** 2 + ell[None, :] ** 2), axis=0) / norm
    var = second - mean ** 2
    # Small negative values (|var| < 1e-10) are numerical roundoff from
    # catastrophic cancellation in second - mean²; clamp to zero.
    # Large negative values indicate a genuine numerical failure (e.g. the
    # kernel matrix is near-singular so alpha has large sign-alternating
    # entries); expose as NaN so the caller knows the moment is unreliable.
    _roundoff_tol = 1.0e-10
    var = np.where(
        var >= 0.0,
        var,
        np.where(np.abs(var) <= _roundoff_tol, 0.0, float("nan")),
    )
    std = np.where(np.isfinite(var), np.sqrt(np.maximum(var, 0.0)), float("nan"))
    return {"mean": mean, "var": var, "std": std, "norm_raw": norm_raw}


def _gp_density_trust_region(gp: GPDensity, k_sigma: float = 2.0) -> Dict[str, FloatArray]:
    """Simple kernel-support trust region: [min(Z)-kℓ, max(Z)+kℓ] in each coordinate."""
    Ztr_obj = gp.raw_training_centers if hasattr(gp, 'raw_training_centers') else gp._Z_train
    Ztr = np.asarray(Ztr_obj.detach().cpu().numpy() if hasattr(Ztr_obj, 'detach') else Ztr_obj, dtype=np.float64)
    ell = np.asarray(gp.lengthscales, dtype=np.float64).reshape(-1)
    zmin = np.min(Ztr, axis=0)
    zmax = np.max(Ztr, axis=0)
    return {
        "lower": zmin - k_sigma * ell,
        "upper": zmax + k_sigma * ell,
        "kernel_halfwidth": k_sigma * ell,
    }


def _alpha_health(gp: GPDensity) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Current coefficient diagnostics + sign-faithfulness test.

    Faithfulness-Test #1 (alpha sign sanity)
    ----------------------------------------
    Under focused sampling, all labels y_i = K_focus · W_cl(R_i, P_i) are
    strictly non-negative.  If the GP is faithfully representing the
    cloud the coefficients α = K_y^{-1} y will then also be predominantly
    non-negative (small negative wiggle from kernel off-diagonals is
    normal; large negative α at many points indicates the kernel can no
    longer cover the cloud and is sign-oscillating to interpolate).

    Reported metrics:
        alpha_neg_frac           — fraction of α_i with α_i < 0
        alpha_neg_l1_frac        — |α_neg|_1 / |α|_1.  Mass-fraction in
                                   the negative coefficients; bounded
                                   triangle invariant for the magnitude of
                                   sign cancellation.  This is the
                                   strongest signal: if alpha_neg_l1_frac
                                   is small, even many sign flips amount
                                   to little net unphysical contribution.
        alpha_min_to_max_ratio   — α_min / α_max (signed ratio).  −1 means
                                   the largest negative coefficient is
                                   as big as the largest positive one
                                   (full sign cancellation regime).

    For seo_signed sampling the labels can be ±-signed by construction,
    so this test is vacuous there — we report the metrics but the
    threshold for "the GP is in trouble" applies only when the active
    sampler is focused (apply_kkt=False on the pin contract).

    Returns (metric_dict, alpha_array).
    """
    alpha = np.asarray(gp._alpha.detach().cpu().numpy() if hasattr(gp._alpha, 'detach') else gp._alpha,
                       dtype=np.float64).reshape(-1)
    out: Dict[str, float] = {
        "alpha_mean": float(np.mean(alpha)),
        "alpha_std": float(np.std(alpha)),
        "alpha_l1": float(np.sum(np.abs(alpha))),
        "alpha_l2": float(np.linalg.norm(alpha)),
        "alpha_linf": float(np.max(np.abs(alpha))),
    }
    # Sign-faithfulness metrics.
    n = int(alpha.size)
    if n > 0:
        a_max = float(np.max(alpha))
        a_min = float(np.min(alpha))
        l1    = float(np.sum(np.abs(alpha)))
        l1_neg = float(np.sum(np.abs(alpha[alpha < 0.0]))) if np.any(alpha < 0.0) else 0.0
        out["alpha_neg_frac"]         = float(np.mean(alpha < 0.0))
        out["alpha_neg_l1_frac"]      = float(l1_neg / l1) if l1 > 0.0 else 0.0
        # Use signed ratio (negative when α_min < 0 < α_max, zero or
        # positive when sign is uniform).  Divide by max(|α_max|, |α_min|)
        # to keep it in [-1, 1].
        denom = max(abs(a_max), abs(a_min), 1.0e-300)
        out["alpha_min_to_max_ratio"] = float(a_min / denom)
    else:
        out["alpha_neg_frac"] = float("nan")
        out["alpha_neg_l1_frac"] = float("nan")
        out["alpha_min_to_max_ratio"] = float("nan")
    # Active-pin flag: True iff this GP is operating under a positive-label
    # contract (focused mode).  Test 1's thresholds should be applied only
    # when this flag is False (apply_kkt=False ⇒ labels are positive-proxy
    # densities, so α sign-oscillation is genuinely a faithfulness failure).
    out["alpha_test_active"] = float(0.0 if bool(getattr(gp, "_pin_apply_kkt", True)) else 1.0)
    return out, alpha


def _gp_vs_cloud_moment_agreement(obs: Dict[str, float]) -> Dict[str, float]:
    """
    Faithfulness-Test #2 (GP-integral vs cloud-Riemann agreement)
    -------------------------------------------------------------
    The KKT-constrained surrogate has analytic integrals
        gpi_A ≡ ∫ ρ̂(z) A(z) dz                  (key: ``km_A`` in obs)
    and the support cloud carries the dual estimator
        clw_A ≡ Σ_i ω_i y_i A(z_i(t))            (key: ``cloud_weighted_A`` in obs)
    For a faithful surrogate these two evaluate the same physical integral
    in two different ways and must agree up to sampling noise / Q residue.
    A growing gap is the signal that the GP is no longer telling the cloud's
    story.  This test reports the absolute difference per anchored moment.
    Tolerances depend on Σ ω_i y_i: when the cloud Riemann sum stays near
    1 the absolute differences are interpretable; when the cloud sum has
    drifted the relative difference is the safer reading.

    Returns a dict with keys
        gp_vs_cloud_norm_abs, gp_vs_cloud_trace_abs, gp_vs_cloud_energy_abs,
        gp_vs_cloud_norm_rel, gp_vs_cloud_trace_rel, gp_vs_cloud_energy_rel
    that the caller appends to the step's `obs` dict.  Missing input keys
    yield NaN entries (e.g. when constraints_enabled=False).
    """
    res: Dict[str, float] = {}
    pairs = (
        ("norm",   "km_normalization", "cloud_norm"),
        ("trace",  "km_trace",         "cloud_weighted_trace"),
        ("energy", "km_energy",        "cloud_weighted_energy"),
    )
    for short, gp_key, cloud_key in pairs:
        gp_v    = obs.get(gp_key, float("nan"))
        cloud_v = obs.get(cloud_key, float("nan"))
        if (not np.isfinite(gp_v)) or (not np.isfinite(cloud_v)):
            res[f"gp_vs_cloud_{short}_abs"] = float("nan")
            res[f"gp_vs_cloud_{short}_rel"] = float("nan")
            continue
        diff_abs = abs(gp_v - cloud_v)
        denom    = max(abs(gp_v), abs(cloud_v), 1.0e-300)
        res[f"gp_vs_cloud_{short}_abs"] = float(diff_abs)
        res[f"gp_vs_cloud_{short}_rel"] = float(diff_abs / denom)
    return res


def _delta_alpha_health(alpha_prev: Optional[FloatArray], alpha_cur: FloatArray) -> Dict[str, float]:
    if alpha_prev is None:
        return {
            "delta_alpha_mean": float('nan'),
            "delta_alpha_std": float('nan'),
            "delta_alpha_l2": float('nan'),
            "delta_alpha_linf": float('nan'),
            "delta_alpha_rel_l2": float('nan'),
        }
    da = np.asarray(alpha_cur, dtype=np.float64).reshape(-1) - np.asarray(alpha_prev, dtype=np.float64).reshape(-1)
    prev_l2 = float(np.linalg.norm(alpha_prev))
    da_l2 = float(np.linalg.norm(da))
    return {
        "delta_alpha_mean": float(np.mean(da)),
        "delta_alpha_std": float(np.std(da)),
        "delta_alpha_l2": da_l2,
        "delta_alpha_linf": float(np.max(np.abs(da))),
        "delta_alpha_rel_l2": da_l2 / max(prev_l2, 1.0e-300),
    }


def _surrogate_faithfulness(gp, Z: ArrayLike, y: ArrayLike,
                             omega: Optional[ArrayLike],
                             weight: Optional[ArrayLike] = None) -> Dict[str, float]:
    """
    Battery of per-step surrogate-vs-cloud faithfulness diagnostics.

    All quantities below are cheap (O(N²) at most, no extra GP fits) and
    serve as in-pipeline guards against three failure modes:

      (i)   FIT FAILURE
            The surrogate cannot represent the current cloud — either
            because the kernel is too smooth (lengthscale too long) or
            two support points have coalesced and K is near-singular.
            Signal:  large LOO residuals at a few points, large cond(K).

      (ii)  COVERAGE COLLAPSE
            The cloud has bifurcated and only a few trajectories are
            "alive" (carry significant weight), so the GP is leaning on
            a tiny effective sample.
            Signal:  ESS / N → 0.

      (iii) MOMENT INCONSISTENCY
            The GP-analytic integrals of ρ̂ and the cloud Riemann sums
            of the labels y disagree on quantities both can compute.
            Signal:  moment_drift_R, moment_drift_P grow monotonically.

    Mathematical content
    --------------------
    Leave-one-out residuals via the standard Cholesky shortcut.  For a
    GP with coefficient α = K⁻¹y, removing point i and predicting at
    z_i gives the residual

        r_i = y_i − ŷ_{−i}(z_i) = α_i / (K⁻¹)_{ii}

    and the LOO predictive variance σ²_{LOO,i} = 1 / (K⁻¹)_{ii}.  This
    requires only the diagonal of K⁻¹, obtained from the cached
    Cholesky factor as Σ_k (L⁻¹)²_{ki} via one triangular solve of
    L⁻¹ I.  Cost: O(N²).

    Effective sample sizes — for α, the standard
        ESS(α) = (Σ α)² / Σ α²            ∈ [1, N]
    measured directly on the GP coefficient vector tells how many
    points dominate the representation.  For y·ω, the equivalent for
    the cloud probability sum.

    K conditioning — λ_max(K)/λ_min(K).  We use np.linalg.cond which
    falls back to SVD; sufficient at N ≲ 2000.

    Moment drifts — for each constraint M ∈ {1, R, P}, compute
    analytically ∫ ψ_M ρ̂ dz (closed-form for Gaussian kernels) and the
    cloud Riemann sum Σ_i ω_i y_i ψ_M(z_i).  The gap is a direct
    surrogate-vs-cloud inconsistency.  For probability (ψ_M = 1) the
    GP integral comes from the GP's own ``compute_moment_values``;
    here we use the simpler ``predict`` cross-check on the support
    cloud (the GP's predict-vs-y residual at the supports themselves
    is the strongest signal of fit failure on the cloud).

    Parameters
    ----------
    gp    : GPDensity surrogate (must be already fitted)
    Z     : (N, D) support-point coordinates
    y     : (N,)  support labels (the values the GP was fit to)
    omega : (N,)  geometric measure ω_i = 1/(N q(z_i^0)) or None

    Returns
    -------
    Dict[str, float] of named diagnostic values, all guaranteed finite
    or NaN (no infs).  Keys are prefixed ``faith_`` for grep-ability.
    """
    out: Dict[str, float] = {}

    Z_np = np.asarray(Z, dtype=np.float64).reshape(-1, 6)
    y_np = np.asarray(y, dtype=np.float64).reshape(-1)
    N = int(y_np.size)
    out["faith_N"] = float(N)

    # ── 1. ESS of α (coverage collapse signal) ────────────────────────────
    # Standard sign-robust ESS:
    #     ESS = (Σ|w|)² / Σ w²
    # This bounds the effective sample size in [1, N] regardless of sign;
    # the alternative (Σw)²/Σw² collapses to <1 for high-cancellation
    # ensembles, which is informative but harder to interpret as a count.
    # We report both so the user can read sign cancellation directly:
    #   ess_alpha       (sign-robust, the "coverage" number)
    #   ess_alpha_naive (unsigned (Σα)²/Σα², measures coverage AND sign
    #                    alignment — drops below 1 when α is a near-zero
    #                    sum of large alternating values)
    try:
        alpha = gp._alpha.detach().cpu().numpy() if hasattr(gp._alpha, "detach") \
                else np.asarray(gp._alpha, dtype=np.float64)
        alpha = alpha.reshape(-1)
        s1_abs = float(np.sum(np.abs(alpha)))
        s1     = float(np.sum(alpha))
        s2     = float(np.sum(alpha * alpha))
        out["faith_ess_alpha"]       = (s1_abs * s1_abs) / s2 if s2 > 1e-300 else float("nan")
        out["faith_ess_alpha_frac"]  = out["faith_ess_alpha"] / max(N, 1)
        out["faith_ess_alpha_naive"] = (s1 * s1) / s2 if s2 > 1e-300 else float("nan")
        # Sign-cancellation ratio: 1.0 = no cancellation, → 0 = total cancellation
        out["faith_alpha_sign_align"] = abs(s1) / s1_abs if s1_abs > 1e-300 else float("nan")
    except Exception:
        out["faith_ess_alpha"]       = float("nan")
        out["faith_ess_alpha_frac"]  = float("nan")
        out["faith_ess_alpha_naive"] = float("nan")
        out["faith_alpha_sign_align"] = float("nan")

    # ── 2. ESS of ω·y (cloud probability concentration) ───────────────────
    if omega is not None:
        w = (np.asarray(omega, dtype=np.float64).reshape(-1) * y_np)
        s1_abs = float(np.sum(np.abs(w)))
        s1     = float(np.sum(w))
        s2     = float(np.sum(w * w))
        out["faith_ess_wy"]      = (s1_abs * s1_abs) / s2 if s2 > 1e-300 else float("nan")
        out["faith_ess_wy_frac"] = out["faith_ess_wy"] / max(N, 1)
        out["faith_wy_sign_align"] = abs(s1) / s1_abs if s1_abs > 1e-300 else float("nan")
    else:
        out["faith_ess_wy"]      = float("nan")
        out["faith_ess_wy_frac"] = float("nan")
        out["faith_wy_sign_align"] = float("nan")

    # ── 3. LOO residuals via Cholesky (fit-failure signal) ───────────────
    # diag(K⁻¹) is needed.  GP exposes solve_K(B) which gives K⁻¹ B;
    # diag(K⁻¹) = solve_K(I).diagonal() — one Cholesky multi-RHS solve.
    try:
        if hasattr(gp, "solve_K"):
            Kinv_diag = np.diag(gp.solve_K(np.eye(N)))          # (N,)
            Kinv_diag = np.asarray(Kinv_diag, dtype=np.float64)
            # Safe division: where (K⁻¹)_ii ≤ 0 (numerical, shouldn't happen
            # for PD K but does at near-singular K), mark as NaN.
            ok = Kinv_diag > 1.0e-300
            r_loo  = np.where(ok, alpha / np.where(ok, Kinv_diag, 1.0),
                              float("nan"))
            v_loo  = np.where(ok, 1.0 / np.where(ok, Kinv_diag, 1.0),
                              float("nan"))
            r_finite = r_loo[np.isfinite(r_loo)]
            if r_finite.size:
                out["faith_loo_rms"]   = float(np.sqrt(np.mean(r_finite**2)))
                out["faith_loo_max"]   = float(np.max(np.abs(r_finite)))
                out["faith_loo_med"]   = float(np.median(np.abs(r_finite)))
                # Standardized residual: r / σ_LOO.  Should be ~ N(0,1) for
                # a well-calibrated GP.  We report the fraction outside ±3σ.
                v_finite = v_loo[np.isfinite(v_loo) & (v_loo > 0)]
                if v_finite.size:
                    sig = np.sqrt(v_finite)
                    z_std = r_finite[: sig.size] / sig
                    out["faith_loo_std_max"] = float(np.max(np.abs(z_std)))
                    out["faith_loo_n_3sig"]  = float(np.sum(np.abs(z_std) > 3.0))
                else:
                    out["faith_loo_std_max"] = float("nan")
                    out["faith_loo_n_3sig"]  = float("nan")
            else:
                out["faith_loo_rms"] = float("nan")
                out["faith_loo_max"] = float("nan")
                out["faith_loo_med"] = float("nan")
                out["faith_loo_std_max"] = float("nan")
                out["faith_loo_n_3sig"]  = float("nan")
        else:
            for k in ("faith_loo_rms", "faith_loo_max", "faith_loo_med",
                      "faith_loo_std_max", "faith_loo_n_3sig"):
                out[k] = float("nan")
    except Exception:
        for k in ("faith_loo_rms", "faith_loo_max", "faith_loo_med",
                  "faith_loo_std_max", "faith_loo_n_3sig"):
            out[k] = float("nan")

    # ── 4. Conditioning of K (numerical-stability signal) ─────────────────
    # We avoid forming K explicitly: cond(K) = (λ_max/λ_min) and the
    # Cholesky factor L has λ_i(K) = L_ii²·..., but extracting eigenvalues
    # from L requires extra work.  Cheap proxy: ratio of largest to
    # smallest diagonal of L (an underestimate of cond(K) but trends with
    # it; useful as a relative-time-series quantity).
    try:
        L_Ky = getattr(gp, "_L_Ky", None)
        if L_Ky is not None:
            d = L_Ky.diagonal().detach().cpu().numpy() \
                if hasattr(L_Ky, "detach") \
                else np.asarray(L_Ky.diagonal(), dtype=np.float64)
            d = np.asarray(d, dtype=np.float64)
            dmin = float(np.min(np.abs(d))) if d.size else 1.0
            dmax = float(np.max(np.abs(d))) if d.size else 1.0
            # cond(K) ≥ (dmax/dmin)² for a Cholesky factor; report this lower bound
            out["faith_cond_K_lo"] = (dmax / dmin) ** 2 if dmin > 1e-300 else float("inf")
            out["faith_cond_K_lo_log10"] = (
                float(np.log10(out["faith_cond_K_lo"]))
                if np.isfinite(out["faith_cond_K_lo"]) and out["faith_cond_K_lo"] > 0
                else float("inf")
            )
        else:
            out["faith_cond_K_lo"] = float("nan")
            out["faith_cond_K_lo_log10"] = float("nan")
    except Exception:
        out["faith_cond_K_lo"] = float("nan")
        out["faith_cond_K_lo_log10"] = float("nan")

    # ── 5. Predict-at-support residual (fit failure on the cloud itself) ─
    # The surrogate is fit to the EFFECTIVE labels that the scheme actually
    # handed it.  For the midpoint/QCLE scheme these are w⊙y (the frozen
    # labels y scaled by the per-point correction weight w); ``state.y`` is
    # deliberately kept frozen at y, so comparing predict(Z) against the raw
    # ``y`` would report w⊙y − y — a weight artefact, NOT a fit error, and
    # would spuriously inflate this metric whenever w deviates from 1.
    # Comparing against the effective labels reports the GP's TRUE residual
    # against what it fit.  For schemes with no correction (weight=None, e.g.
    # PBME) y_eff ≡ y, so behaviour is unchanged.
    try:
        if weight is not None:
            w_np = np.asarray(weight, dtype=np.float64).reshape(-1)
            y_eff = y_np * w_np if w_np.shape == y_np.shape else y_np
        else:
            y_eff = y_np
        y_pred = gp.predict(Z_np)
        r = y_pred - y_eff
        out["faith_predict_rms"]  = float(np.sqrt(np.mean(r * r)))
        out["faith_predict_max"]  = float(np.max(np.abs(r)))
        # Relative — normalised by RMS of the effective labels themselves so
        # the number is interpretable across runs and physical units.
        denom = float(np.sqrt(np.mean(y_eff * y_eff))) or 1.0
        out["faith_predict_rms_rel"] = out["faith_predict_rms"] / denom
    except Exception:
        out["faith_predict_rms"]     = float("nan")
        out["faith_predict_max"]     = float("nan")
        out["faith_predict_rms_rel"] = float("nan")

    return out


_DIM_LABELS = ("R", "P", "r0", "r1", "p0", "p1")

# =============================================================================
# Run-level state
# =============================================================================

@dataclass
class SimulationState:
    """
    Live state threaded through the dynamics loop.

    Z  : (N, D)   support-point coordinates at the current time.
    y  : (N,)     density labels at the support points.
                  Under pure Liouville (PBME), y_i = ρ(Z_i^0) is frozen;
                  under QCLE, y_i is updated by the midpoint correction.
    gp : GPDensity  surrogate fitted on (Z, y), consistent with moment targets.
    t  : float    current physical time.
    step_index : int  number of completed dynamics steps.
    moment_targets : {"normalization": 1, "trace": 1, "energy": E_0}.
    initial_proposal_density / initial_target_density / initial_weight:
        Optional sampling diagnostics retained from the initial signed-SEO
        draw so the exact Monte Carlo estimator used to set moment targets can
        be audited later from saved snapshots.
    sampling_*:
        Metadata describing how the initial support cloud was chosen. This keeps
        the sampling policy connected to the actual propagated state instead of
        being lost at the run-driver level.
    """
    Z:              FloatArray
    y:              FloatArray
    gp:             GPDensity
    t:              float
    step_index:     int
    moment_targets: Dict[str, float]
    initial_proposal_density: Optional[FloatArray] = None
    initial_target_density: Optional[FloatArray] = None
    initial_weight: Optional[FloatArray] = None
    # Frozen geometric measure ω_i = 1/(N q(z_i^0)) for cloud Riemann sums.
    # Set once at t=0 from the initial sampler; never updated.
    geometric_measure: Optional[FloatArray] = None
    # Per-trajectory QCLE correction weights w_i, dynamic.  Initialised to
    # ones(N) by the MidpointScheme at first step; the GP fits to
    # (w * y) as the empirical signed-density label.  state.y stays
    # frozen at the t=0 sampled values; only w evolves via the Heun or
    # Cayley weight rules.  None for legacy schemes that don't carry w.
    correction_weight: Optional[FloatArray] = None
    # Per-init diagnostics about the proposal and ω clipping.
    sampling_diagnostics: Optional[Dict[str, float]] = None
    sampling_mode: str = "seo_signed"
    sampling_healthiest: bool = False
    sampling_n_candidates: int = 1
    sampling_jackknife_blocks: int = 0
    sampling_best_candidate_index: Optional[int] = None
    sampling_score: Optional[float] = None
    sampling_signed_ess: Optional[float] = None
    sampling_cancellation_ratio: Optional[float] = None


@dataclass
class DynamicsConfig:
    scheme:               str                  # "pbme" | "midpoint"
    dt:                   float
    n_steps:              int
    snapshot_every:       int   = 5             # 0 disables snapshots
    include_abs_integral: bool  = False         # expensive MC L¹ diagnostic
    verbose:              bool  = True
    detailed_verbose:     bool  = False         # print full GP + invariant detail each step
    output_dir:           str   = "results"
    run_name:             str   = "run"
    # Hyperparameter policy during propagation.
    #
    # The physical correlation lengths of ρ(z) are set at t=0 by the initial
    # wavepacket (σ_R) and the SEO Gaussian envelope (ℏ).  Two failure modes:
    #
    #   * Hard-freezing every hyperparameter ("frozen" mode — the production
    #     default).  Fit quality eventually degrades as the support cloud
    #     phase-mixes, because the kernel's notion of locality cannot follow
    #     the cloud.  For short-to-medium runs this is still the strongest
    #     baseline.
    #
    #   * Fully re-optimizing every hyperparameter at each refit ("free"
    #     mode).  MLL sees rougher responses at fixed labels and prefers to
    #     explain them as smooth-plus-noise — driving ℓ upward (oversmoothing)
    #     and steadily flattening the density representation.
    #
    # The "breathing" middle path pins σ_f and σ_n at the initial-fit anchor
    # but lets lengthscales adapt with a shrinkage prior.  Useful for long
    # runs where the frozen fit visibly degrades (R² below ~0.99).  Set via
    # `GPDensityConfig.refit_hyper_policy`.
    #
    # `freeze_hypers_after_initial_fit=True` here forces the legacy
    # `freeze_hypers()` hard lock (overriding the GP policy).  It is retained
    # for backwards compatibility but is now redundant with
    # `refit_hyper_policy="frozen"`.
    freeze_hypers_after_initial_fit: bool = False
    # Midpoint-only optional safeguard.  For scientific diagnostics this should
    # generally remain disabled so the raw pulled-back operator is tested.
    # The corrected csz label scheme already conserves probability to machine
    # precision per step, so q_clip is no longer needed in production runs
    # — but we keep it as an OFF-by-default opt-in.
    apply_q_clip:         bool  = False
    q_clip_frac:          float = 0.3
    q_clip_abs:           float = 1.0e-12
    # Derivative-smoothing scale for the QCLE operator (accepted for
    # compatibility with run.py; ignored in old_files_2 since the operator
    # does not yet expose q_sigma_n_scale).
    q_derivative_sigma_n_scale: float = 1.0

    # ─── Flow-correction parameters ─────────────────────────────────────
    # CORRECTED 2026-07: this comment used to claim flow correction "is
    # ALWAYS applied", while MidpointScheme.__init__ simultaneously
    # documented the same parameters as dead legacy knobs ("the L β = Q /
    # v_corr machinery is gone") — a direct contradiction between the two
    # Continuity-form flow correction (2026-07): flow_fraction f routes
    # the excess-term flux between support displacement (dz_P = f dt J/rho)
    # and the weight ODE (rate Q + f u dP(rho)) — same PDE at every f.
    # OFF by default (f=0, the pure weight scheme). See
    # MidpointScheme docstring and Operator.compute_flux_at_points.
    flow_correction_axes:         str   = "P_only"    # or "all"
    flow_correction_grad_floor:   float = 1.0e-8
    flow_correction_step_cap:     float = 0.5
    flow_fraction:                float = 0.0         # 0 = unchanged behaviour

    # MidpointScheme weight-update variant.  "midpoint" = explicit-
    # midpoint Heun (k¹ then k²); "cayley" = symmetric Cayley map
    # (1 + Δt/2 σ)/(1 − Δt/2 σ).  Both update the per-trajectory
    # correction weight w via a midpoint refit of the GP.  Only consulted
    # when label_scheme="weight".
    weight_scheme:                str   = "midpoint"   # or "cayley"

    # Label-integrator variant.  "weight" (default) = the scalar
    # Heun/Cayley scheme above, unchanged.  "linear" = experimental
    # Crank-Nicolson integrator of the linear label-product ODE
    # b_dot = A b, A = L K^-1 (see MidpointScheme docstring). New in
    # 2026-07; not yet validated against finite differences.
    label_scheme:                 str   = "weight"     # or "linear"

    # -------------------------------------------------------------------------
    # Signed-weight ESS resampling (opt-in).
    # -------------------------------------------------------------------------
    # When the signed-SEO initial cloud has ESS/N fall below `ess_resample_threshold`,
    # the labels carried along trajectories stop being a reliable Monte Carlo
    # estimator of the density.  This shows up as `km_normalization` drifting
    # away from 1, which divides into every KKT-normalized physical moment and
    # produces the catastrophic P_α blowup seen in long runs.
    #
    # If `enable_ess_resampling=True`, then whenever essf < threshold, the
    # scheme re-evaluates labels:
    #
    #     y_new ← ρ̂(Z_current)      (use the current GP surrogate as the
    #                                "truth" against which new samples will
    #                                be drawn)
    #
    # and refits the GP on (Z_current, y_new).  This destroys the carried-along
    # Liouville labels, but restores the surrogate to a self-consistent state.
    # It's a sequential-importance-resampling move: the estimator becomes
    # biased toward whatever bias the surrogate has at the resampling instant,
    # in exchange for preventing unbounded variance growth.
    #
    # Rate-limited: after a resample, suppress further resamples for
    # `ess_resample_cooldown` steps so the system can actually settle.
    #
    # Default off so the pathology is visible in baseline runs.
    enable_ess_resampling: bool   = False
    ess_resample_threshold: float = 0.05      # essf below this triggers
    ess_resample_cooldown:  int   = 25        # min steps between resamples
    ess_resample_max:       int   = 100       # hard cap on total resamples

    # -------------------------------------------------------------------------
    # Density representation: single GP vs density-difference.
    # -------------------------------------------------------------------------
    # When False (default), the pipeline uses a single GPDensity surrogate
    # that carries the full label vector y directly.  This is the legacy
    # architecture.
    #
    # When True, the pipeline uses a GPDensityDiff surrogate that
    # decomposes ρ̂ = ρ̂_0^{transported} + δ̂.  The baseline ρ̂_0 is frozen
    # at t=0; only the correction δ̂ is refit each step, with training
    # targets δ_i = y_i(t) - y_i(0) that start identically at zero.  This
    # prevents the KKT-constrained refit from growing oscillatory α
    # coefficients to reconcile drifted y with pinned moment constraints.
    #
    # See GP_DensityDiff.py for the full architecture.
    use_density_diff: bool = False

    # -------------------------------------------------------------------------
    # Sampling-variance reduction options.
    # -------------------------------------------------------------------------
    # The cloud Riemann sum  ⟨A⟩ = Σ_i ω_i y_i A(z_i(t))  has per-step
    # variance proportional to  Σ_i ω_i² y_i² A_i² .  With the default
    # signed-SEO sampling, ω_i = 1/(N q_i) is unbounded in tails (q is
    # Gaussian; |w_poly| grows quadratically), causing tail points to
    # dominate the estimator and destabilize trace conservation under
    # midpoint propagation.  Two opt-in remedies:
    #
    # (1) Tail-clip ω: cap ω_i at the `omega_clip_quantile` empirical
    #     quantile.  Cheap, biased (drops the very tail of the integrand),
    #     bias is logged.  Set `omega_clip_quantile=None` to disable.
    #
    # (2) Better proposal: sample from |ρ_0| via rejection on the polynomial
    #     factor (`abs_target=True`).  Then ω_i y_i = ±Z_abs/N and the cloud
    #     sum is bounded — variance is purely from sign cancellation among
    #     A(z_i), which is intrinsic and not a sampling artifact.  Slower at
    #     init (rejection sampling) but gives uniform per-step variance.
    #     Set `abs_cap_quantile` to control the rejection ceiling.
    #
    # Either or both may be enabled.  The two are independent fixes for
    # different aspects of the same problem (heavy-tailed importance ratios).
    omega_clip_quantile:  Optional[float] = None        # e.g. 0.99 → clip top 1%
    abs_target:           bool            = False
    abs_cap_quantile:     float           = 0.999


# =============================================================================
# Scheme interface
# =============================================================================

class DynamicsScheme(ABC):
    """
    One step of the integrator.

    Concrete implementations must advance state.Z, state.y, state.gp,
    state.t, and state.step_index, then return a diagnostics dict.
    """

    name: str = "base"

    def __init__(self, dynamics: PBMEMIntDynamics, dt: float) -> None:
        self.dynamics = dynamics
        self.dt       = float(dt)

    @abstractmethod
    def step(self, state: SimulationState) -> Dict:
        ...


# =============================================================================
# PBME scheme  (pure Liouville transport)
# =============================================================================

class PBMEScheme(DynamicsScheme):
    """
    Pure Liouville transport.

        Z_new = Φ^0_{Δt}(Z_old)          (MInt forward step)
        y     unchanged                   (Liouville: ρ constant along trajectories)
        GP    FROZEN — not refitted       (density is exactly conserved; cloud
                                           Riemann sums lw_* carry time evolution
                                           through the propagated positions Z_new)
    """
    name = "pbme"

    def step(self, state: SimulationState) -> Dict:
        # -------- MInt forward propagation -----------------------------------
        t_mint = time.time()
        Z_new = np.asarray(
            self.dynamics.step(state.Z, self.dt), dtype=np.float64
        )
        mint_wall = time.time() - t_mint

        # -------- α refit with frozen hyperparameters ------------------------
        # PBME density is Liouville-conserved: ρ(z,t) = ρ₀(Φ_{-t}(z)).
        # Hyperparameters (ℓ, σ_f, σ_n) are frozen at their initial-fit values.
        # The coefficient vector α IS re-solved at each step via
        #   α_t = K_y(Z_t)^{-1} y,  then KKT-projected onto A(Z_t)α = b,
        # so that dp_trace = 1 and dp_P0 remain correctly constrained as the
        # cloud translates.  Without this step, the moment-matrix A(Z_t) drifts
        # from A(Z₀) and dp_trace would no longer equal 1.
        t_refit = time.time()
        state.gp.refit(
            Z_train=Z_new, y_train=state.y,
            moment_targets=state.moment_targets,
        )
        refit_wall = time.time() - t_refit

        state.Z          = Z_new
        state.t         += self.dt
        state.step_index += 1
        return {
            "Q": None, "Y": None,
            "mint_wall":     mint_wall,
            "operator_wall": 0.0,
            "refit_wall":    refit_wall,
        }


# =============================================================================
# Diagnostics helpers shared by the corrected MidpointScheme
#
# Both functions expose machinery that already existed and was already
# being computed (or trivially computable) elsewhere in the pipeline, but
# was never surfaced: the KKT residual lives inside every GP refit
# (GP_Density._apply_kkt_projection), and the flow displacement is a direct
# application of the GP's own analytic gradient (GPDerivatives.
# rho_derivative_bundle), which is already validated against finite
# differences.
# =============================================================================

def _kkt_residual_norm(gp: GPDensity) -> float:
    """
    ||A alpha - b|| for the moment-constraint system solved by the GP's
    KKT / Schur-complement projection (see GPDensity._apply_kkt_projection).

    A, b, and the corrected alpha are already stored on the GP object after
    every fit/refit; this just reads them.  Returns 0.0 (a genuine zero, not
    a placeholder) when no moment constraints were active on the last fit.
    """
    A = getattr(gp, "_A", None)
    b = getattr(gp, "_b", None)
    alpha = getattr(gp, "_alpha", None)
    if A is None or b is None or alpha is None:
        return 0.0
    with torch.no_grad():
        resid = A @ alpha - b
    return float(torch.linalg.norm(resid).item())


def _flow_displacement_removed():
    """REMOVED (2026-07): the Taylor-inversion displacement
    dz = -Q grad/|grad|^2 was a per-step value-matching device with no
    flux structure.  The flow channel now uses the exact continuity form
    u_P = J_P/rho with compressibility carried by the weight rate
    k = Q + f*u*dP(rho); see MidpointScheme._continuity_velocity and
    Operator.compute_flux_at_points."""
    raise NotImplementedError


class MidpointScheme(DynamicsScheme):
    """
    QCLE midpoint integrator with per-trajectory weight updates.

    Discretisation
    --------------
    The mapping-QCLE evolution is split into

        ∂_t ρ_m  =  − iℒ_m° ρ_m   −  iℒ_m' ρ_m
                    (PBME Liouville)   (QCLE correction)

    and the integrator alternates **PBME half-steps** (via MInt) with a
    **weight update** that absorbs the QCLE coupling.  Trajectories carry
    a frozen sign-bearing label

        y_i := ρ_0(z_i⁰)     (set once at sampling, never updated)

    and a dynamic correction weight  w_i  (init w_i = 1).  The GP fits to
    the empirical signed-density labels  (w_i · y_i).

    One step,  t → t + Δt:

        Step 1.  Z^{n+1/2} = Φ_{Δt/2}^{MInt}(Z^n)
        Step 2.  k¹_i = −(iℒ_m' ρ̂^n)(Z_i^n)          (Q at current GP, current pts)
        Step 3.  w_i^{n+1/2,*} = w_i^n + (Δt/2) · k¹_i / y_i      [HEUN]
            or:  σ_i = k¹_i / ρ̂^n(Z_i^n)
                 w_i^{n+1/2,*} = (1 + Δt/2 σ_i)/(1 − Δt/2 σ_i) · w_i^n  [CAYLEY]
        Step 4.  Refit GP_mid to (Z^{n+1/2}, w^{n+1/2,*} · y) + KKT
        Step 5.  k²_i = −(iℒ_m' ρ̂_mid)(Z_i^{n+1/2})  (Q at midpoint GP, midpoint pts)
        Step 6.  w_i^{n+1} = w_i^n + Δt · k²_i / y_i             [HEUN]
            or:  σ_i = k²_i / ρ̂_mid(Z_i^{n+1/2})
                 w_i^{n+1} = (1 + Δt/2 σ_i)/(1 − Δt/2 σ_i) · w_i^n   [CAYLEY]
        Step 7.  Z^{n+1} = Φ_{Δt/2}^{MInt}(Z^{n+1/2})
        Step 8.  Refit GP_new to (Z^{n+1}, w^{n+1} · y) + KKT

    Crucial property: α is determined ENTIRELY by the refit (K_y⁻¹ on the
    weighted labels, then KKT projection).  No β-solve, no L β = Q
    machinery.  The QCLE coupling enters only through w — UNLESS
    ``flow_fraction > 0`` (below), in which case part of Q instead moves
    the support points directly.

    Two weight-update variants, runtime selectable:

    * ``weight_scheme="midpoint"``: explicit-midpoint Heun (k¹ then k²).
      Source slope evaluated as  ̇w_i = k_i / y_i  where  k_i = −Q_i.
    * ``weight_scheme="cayley"``: symmetric Cayley map (1+x)/(1−x).
      Local rate  σ_i = k_i / ρ̂(z_i).  Time-symmetric to 2nd order;
      bounded for arbitrary real σ.

    Flow correction (``flow_fraction``), restored 2026-07, revised 2026-07
    -------------------------------------------------------
    Previously ``flow_correction_axes/grad_floor/step_cap`` were accepted
    but never used (see git history / prior docstring: "the L β = Q / v_corr
    machinery is gone") — despite ``DynamicsConfig`` separately and
    self-contradictorily documenting flow correction as "ALWAYS applied".
    That contradiction, plus the memory of an earlier ``flow_fraction`` /
    ``correction_mode`` design meant to split Q between the flow and weight
    channels precisely to avoid double-counting, is why this exists.

    CONTINUITY FORM (2026-07, final design — supersedes both the
    pre-split-rate and realized-increment designs).  The excess term is
    exactly a bath-momentum divergence, Q = -d(J_P)/dP with
    J_P = (hbar/8) dh_bar : (D_rr + D_pp) rho  [NBK JCP 133, 134115
    (2010), Eq. (10); the flux form follows because dh_bar depends on R
    only].  The mapping-QCLE is therefore an exact continuity equation
    with hydrodynamic velocity u_P = J_P/rho on top of the
    divergence-free Hamiltonian field.  The corrected flow is
    COMPRESSIBLE: Liouville's theorem does not hold along it and is not
    enforced; the density obeys D(rho)/Dt = -rho d(u_P)/dP.  The
    ``flow_fraction`` f in [0,1] routes the SAME correction between two
    exact Lagrangian representations of that equation:
        dz/dt = v_H + f*u_P*e_P,      d(w*y)/dt = Q + f*u_P*d(rho)/dP.
    f=0 is the pure weight scheme; f=1 is pure flow with the weights
    carrying only the compressibility factor; every f solves the same
    PDE to the order of the scheme (verified: observable differences
    across f in {0, 0.5, 1} are O(dt^2)-small fractions of the
    correction signal).  f is a representation dial, not physics —
    larger f trades weight-channel ESS degradation (w excursions shrink
    monotonically with f) for support-geometry perturbation.

    Label integrator (``label_scheme="linear"``), new 2026-07
    -------------------------------------------------------
    An alternative to the scalar per-trajectory Heun/Cayley update above.
    Operator.compute_L_matrix already establishes  Q = L alpha  and, since
    alpha = K^{-1} b with b_i := w_i y_i,  b_dot = Q = L K^{-1} b = A b  is
    an exact LINEAR ODE for the label-product vector b (frozen-coefficient
    within a step, since L, K depend on the current GP fit).  With
    ``label_scheme="linear"``, b is advanced by the symmetric (Cayley/
    Crank-Nicolson) Pade approximant to exp(dt A) instead of by the scalar
    per-point Heun/Cayley rule — the direct matrix generalisation of
    ``weight_scheme="cayley"``, now coupling trajectories through the
    off-diagonal kernel structure of L.  This is a genuinely new
    implementation (nothing to restore — no prior code built A = L K^{-1}
    anywhere), built entirely from already-verified pieces
    (``compute_L_matrix``, ``GPDensity.solve_K``); it has NOT been checked
    against finite-difference or small-N convergence tests the way the
    rest of the operator has, so treat it as experimental until you do.
    Default remains ``label_scheme="weight"`` (current behaviour,
    unchanged).
    """
    name = "midpoint"

    def __init__(self, dynamics: PBMEMIntDynamics, dt: float,
                 operator: Optional[QCLECorrection] = None,
                 apply_q_clip: bool = False,
                 q_clip_frac: float = 0.3,
                 q_clip_abs:  float = 1.0e-12,
                 flow_correction_axes: str = "P_only",
                 flow_correction_grad_floor: float = 1.0e-8,
                 flow_correction_step_cap: float = 0.5,
                 flow_fraction: float = 0.0,
                 weight_scheme: str = "midpoint",
                 label_scheme: str = "weight") -> None:
        """
        Parameters
        ----------
        operator : QCLECorrection
            Provides analytic Q.  Created lazily if None.
        weight_scheme : {"midpoint", "cayley"}
            * "midpoint" : explicit-midpoint Heun on  ̇w = −Q/y
            * "cayley"   : symmetric Cayley  (1+Δt/2 σ)/(1−Δt/2 σ)
                           where σ = −Q/ρ̂  (2nd-order, time-symmetric,
                           bounded for any real σ).
            Only consulted when ``label_scheme="weight"``.
        flow_fraction : float in [0, 1], default 0.0
            Fraction of the REALIZED weight-update increment (evaluated
            with the full, unsplit rate) that is converted into a
            support-point displacement afterward and pulled back out of
            the weight change, rather than a pre-allocated share of the
            raw rate — see the class docstring. 0.0 (default) reproduces
            the exact pre-existing behaviour — nothing changes for callers
            who don't set this. NOT combined with ``label_scheme="linear"``
            (raises if both are active — the linear b-ODE already
            consumes all of Q through the label channel, and splitting
            it against a position displacement on top has not been
            derived).
        apply_q_clip, q_clip_frac, q_clip_abs:
            LEGACY — accepted for back-compat with older configs.  Still
            unused.
        flow_correction_axes, flow_correction_grad_floor,
        flow_correction_step_cap:
            Continuity-flow controls (2026-07).  The flux J_P lives on
            the bath-P axis only, so ``axes`` is retained for config
            compatibility but the displacement is intrinsically P-only.
            ``grad_floor`` now floors |rho| (the only denominator left,
            in u = J/rho); ``step_cap`` clips |dz_P| per leg.  All
            inert when ``flow_fraction == 0``.
        label_scheme : {"weight", "linear"}
            * "weight" (default): the Steps 1-8 scalar Heun/Cayley scheme
              above, unchanged.
            * "linear": the experimental Crank-Nicolson label-ODE
              integrator on b = w*y — see the class docstring.
        """
        super().__init__(dynamics, dt)
        self.operator = operator if operator is not None \
                        else QCLECorrection(dynamics)
        self.apply_q_clip = bool(apply_q_clip)
        self.q_clip_frac  = float(q_clip_frac)
        self.q_clip_abs   = float(q_clip_abs)
        self.flow_correction_axes        = flow_correction_axes
        self.flow_correction_grad_floor  = float(flow_correction_grad_floor)
        self.flow_correction_step_cap    = float(flow_correction_step_cap)
        self.flow_fraction = float(flow_fraction)
        if not (0.0 <= self.flow_fraction <= 1.0):
            raise ValueError(f"flow_fraction must be in [0, 1]; got {flow_fraction!r}")

        if weight_scheme not in ("midpoint", "cayley"):
            raise ValueError(
                f"weight_scheme must be 'midpoint' or 'cayley'; "
                f"got {weight_scheme!r}")
        self.weight_scheme = weight_scheme

        if label_scheme not in ("weight", "linear", "strang"):
            raise ValueError(
                f"label_scheme must be 'weight', 'linear' or 'strang'; "
                f"got {label_scheme!r}")
        if label_scheme in ("linear", "strang") and self.flow_fraction > 0.0:
            raise ValueError(
                "flow_fraction > 0 with label_scheme='linear'/'strang' is not "
                "supported: the linear b-ODE already routes all of Q through "
                "the label channel; splitting it against a flow displacement "
                "has not been derived. Use one mechanism at a time.")
        self.label_scheme = label_scheme

    # ------------------------------------------------------------------
    # Helper: evaluate ρ̂(Z) directly from a GP surrogate
    # ------------------------------------------------------------------
    @staticmethod
    def _rho_at(gp: GPDensity, Z: FloatArray) -> FloatArray:
        """ρ̂(Z_i) for each support point Z_i, computed analytically via the
        kernel evaluation k(Z_i, Z_train_j) · α_j.  Product surrogates
        (ρ̂ = g·μ) route through their own predict, which applies the
        analytic mapping profile."""
        if getattr(gp, "_is_product", False):
            return np.asarray(gp.predict(Z), dtype=np.float64).reshape(-1)
        from .GPDerivatives import _prepare
        K_t, _V_t, _lam_t, alpha_t, _W_t, _single = _prepare(gp, Z)
        with torch.no_grad():
            rho_t = (K_t @ alpha_t)
        return rho_t.detach().cpu().numpy().astype(np.float64)

    def _continuity_velocity(self, J: FloatArray, rho: FloatArray) -> FloatArray:
        """
        Hydrodynamic momentum velocity u_P = J_P / ρ̂ of the excess-term
        continuity form, with a signed density floor.  The floor is the
        larger of the configured absolute value (flow_correction_grad_floor,
        reinterpreted: it now floors |ρ̂|, the only denominator left in the
        continuity formulation) and a 1e-6 relative fraction of max|ρ̂| —
        u is a flux-over-density ratio and is only meaningful where the
        surrogate carries density.
        """
        J   = np.asarray(J,   dtype=np.float64).reshape(-1)
        rho = np.asarray(rho, dtype=np.float64).reshape(-1)
        floor = max(float(self.flow_correction_grad_floor),
                    1.0e-6 * float(np.max(np.abs(rho))) if rho.size else 0.0)
        rho_safe = np.where(np.abs(rho) >= floor, rho,
                            np.where(rho >= 0.0, floor, -floor))
        return J / rho_safe

    def _continuity_displacement(self, u: FloatArray, dt_leg: float,
                                 shape) -> tuple:
        """
        Displacement dz for one Strang leg: dz_P = f · dt_leg · u_P, clipped
        at ±flow_correction_step_cap.  The excess flux lives on the bath-P
        axis only, so the displacement is intrinsically P-only — no axes
        choice remains.
        """
        step = self.flow_fraction * dt_leg * np.asarray(u, dtype=np.float64)
        cap = float(self.flow_correction_step_cap)
        n_capped = int(np.sum(np.abs(step) > cap)) if np.isfinite(cap) else 0
        if np.isfinite(cap):
            step = np.clip(step, -cap, cap)
        dz = np.zeros(shape, dtype=np.float64)
        dz[:, 1] = step
        return dz, n_capped

    @staticmethod
    def _probability_drift(state: SimulationState, db: FloatArray) -> float:
        """
        |Delta integral rho_hat dz|, estimated via the frozen cloud
        Riemann sum omega_i = 1/(N q(z_i^0)) already carried on
        ``state.geometric_measure`` (the same estimator used for the
        physical lw_*/dp_* observables elsewhere in the pipeline),
        applied to the actual label-product increment db = Delta(w*y).

        Returns 0.0 (genuine, not a placeholder) if no geometric measure
        was recorded at init (e.g. a non signed-SEO sampling mode).
        """
        if state.geometric_measure is None:
            return 0.0
        omega = state.geometric_measure
        return float(np.abs(np.sum(omega * db)))

    def step(self, state: SimulationState) -> Dict:
        if self.label_scheme == "linear":
            return self._step_linear(state)
        if self.label_scheme == "strang":
            return self._step_strang(state)
        return self._step_weight(state)

    def _step_weight(self, state: SimulationState) -> Dict:
        # ------------------------------------------------------------------
        # State initialisation: if state.correction_weight is None we are
        # at the first step.  Initialise w = ones(N).  state.y is the
        # full signed-SEO Wigner value from sampling — we leave it as is.
        # ------------------------------------------------------------------
        if state.correction_weight is None:
            state.correction_weight = np.ones(state.Z.shape[0], dtype=np.float64)
        w_n = state.correction_weight
        y_n = state.y       # frozen sign-bearing label
        flow_on = self.flow_fraction > 0.0

        # ------------------------------------------------------------------
        # Step 1 — PBME half-step (first leg of Strang splitting).
        # ------------------------------------------------------------------
        t_mint = time.time()
        Z_half = np.asarray(
            self.dynamics.step(state.Z, 0.5 * self.dt), dtype=np.float64
        )
        mint_wall = time.time() - t_mint

        # ------------------------------------------------------------------
        # Step 2 — k¹ = −(iℒ_m' ρ̂^n)(Z^n)
        #
        # Q evaluated at the CURRENT support positions Z^n directly.
        # We use build_at_points (no internal backward half-step) because
        # the Heun label ODE  d(w·y)/dt = +Q  requires the rate at the
        # actual trajectory location z_i^n, not at Φ_{-Δt/2}(z_i^n).
        # Using operator.build() here would apply an additional Φ_{-Δt/2}
        # pullback, evaluating k¹ at z^{n-1/2} instead of z^n and
        # breaking second-order consistency of the Strang splitting.
        # ------------------------------------------------------------------
        t_op = time.time()
        data_1 = self.operator.build_at_points(state.Z, gp=state.gp)
        Q_1   = np.asarray(data_1.Q, dtype=np.float64)
        operator_wall_1 = time.time() - t_op
        # ------------------------------------------------------------------
        # CONTINUITY-FORM FLOW (2026-07, replaces the Taylor-inversion
        # splice).  The excess term is exactly a bath-momentum divergence,
        #     Q = -d(J_P)/dP,   J_P = (hbar/8) dh_bar : (D_rr + D_pp) rho,
        # so the full mapping-QCLE is the continuity equation
        #     drho/dt + div(rho v_H) + d(J_P)/dP = 0
        # with hydrodynamic velocity u = J_P/rho on the P axis only.  The
        # corrected flow is COMPRESSIBLE: along dz/dt = v_H + f*u*e_P the
        # exact carried-value rate is
        #     D(rho)/Dt = Q + f * u * d(rho)/dP        (any f in [0,1]),
        # which reduces to the pure weight scheme at f=0 and to pure flow
        # with compressibility weights (D rho/Dt = -rho dU/dP) at f=1.
        # Every f solves the SAME PDE to the order of the scheme — f is a
        # representation dial, not physics.  Liouville's theorem does NOT
        # hold for the corrected flow and is not enforced; the weight
        # factor carries exactly the compressibility (NBK 2010, Sec. III:
        # the excess flux is the back reaction on the environment).
        # ------------------------------------------------------------------
        if flow_on:
            J_1, rho_1, dPrho_1 = compute_flux_at_points(
                state.Z, state.gp, self.dynamics)
            u_1 = self._continuity_velocity(J_1, rho_1)
            k1 = Q_1 + self.flow_fraction * u_1 * dPrho_1
        else:
            u_1 = None
            k1 = Q_1

        # ------------------------------------------------------------------
        # Step 3 — predicted midpoint weight  w^{n+1/2,*}
        #
        # Compute the FULL (unsplit) update first, using the true rate k1 —
        # not a pre-scaled-down fraction of it.  This matters for
        # weight_scheme="cayley", whose map is nonlinear in its rate
        # argument: feeding it (1-f)*k1 from the start does NOT equal
        # taking (1-f) of the realized increment produced by the true k1.
        # Flow is then derived from what the update actually did, per your
        # point that flow should evolve from the realized density change
        # rather than a rate split decided in advance.
        # ------------------------------------------------------------------
        # Numerical-safety floor on |y| only to avoid division by exactly
        # zero — does NOT change behaviour away from exact zeros.
        y_safe = np.where(np.abs(y_n) > 0.0, y_n,
                          np.sign(y_n) + (y_n == 0.0).astype(np.float64))

        if self.weight_scheme == "midpoint":
            # Heun: ẇ_i = k_i / y_i  with the Lagrangian rate k (includes
            # the f·u·∂_Pρ advective compensation when flow is on).
            w_dot_1 = k1 / y_safe
            w_half_star = w_n + 0.5 * self.dt * w_dot_1
            sigma_diag_1 = w_dot_1
        else:  # cayley
            rho_at_Zn = self._rho_at(state.gp, state.Z)
            rho_safe_1 = np.where(np.abs(rho_at_Zn) > 0.0, rho_at_Zn,
                                  1.0 + 0.0 * rho_at_Zn)
            sigma_1 = k1 / rho_safe_1
            w_half_star = ((1.0 + 0.5 * self.dt * sigma_1) /
                           (1.0 - 0.5 * self.dt * sigma_1)) * w_n
            sigma_diag_1 = sigma_1

        # Stage-1 displacement: half-step advection along the continuity
        # velocity, dz_P = f · (Δt/2) · u.  The flux J_P lives on the P
        # axis ONLY — the old axes ambiguity is gone; the physics fixes
        # the direction.
        if flow_on:
            dz_1, n_capped_1 = self._continuity_displacement(
                u_1, 0.5 * self.dt, state.Z.shape)
        else:
            dz_1 = np.zeros_like(state.Z)
            n_capped_1 = 0

        # The predictor midpoint cloud used to fit gp_mid picks up the
        # stage-1 flow displacement (zero when flow_fraction == 0).
        Z_half_fc = Z_half + dz_1

        # ------------------------------------------------------------------
        # Step 4 — refit GP_mid on (Z^{n+1/2}, w^{n+1/2,*} · y) + KKT.
        #
        # Deep-copy the current GP so the midpoint fit doesn't perturb
        # the persistent state.gp before the final refit overwrites it
        # (Step 8).
        # ------------------------------------------------------------------
        t_refit_mid = time.time()
        gp_mid = copy.deepcopy(state.gp)
        target_half = w_half_star * y_n
        gp_mid.refit(
            Z_train=Z_half_fc, y_train=target_half,
            moment_targets=state.moment_targets,
        )
        refit_wall_mid = time.time() - t_refit_mid

        # ------------------------------------------------------------------
        # Step 5 — k² = −(iℒ_m' ρ̂_mid)(Z^{n+1/2})
        #
        # Q evaluated directly at the (flow-corrected) midpoint positions.
        # Same reasoning as k¹: build_at_points avoids the internal
        # Φ_{-Δt/2} pullback.  Using build() here would place the
        # evaluation at Φ_{-Δt/2}(Z^{n+1/2}) = Z^n, collapsing k₂ onto
        # nearly the same location as k₁ and destroying the two-stage
        # Heun correction.
        # ------------------------------------------------------------------
        t_op2 = time.time()
        data_2 = self.operator.build_at_points(Z_half_fc, gp=gp_mid)
        Q_2   = np.asarray(data_2.Q, dtype=np.float64)
        operator_wall_2 = time.time() - t_op2
        if flow_on:
            J_2, rho_2, dPrho_2 = compute_flux_at_points(
                Z_half_fc, gp_mid, self.dynamics)
            u_2 = self._continuity_velocity(J_2, rho_2)
            k2 = Q_2 + self.flow_fraction * u_2 * dPrho_2
        else:
            u_2 = None
            rho_2 = None
            k2 = Q_2

        # ------------------------------------------------------------------
        # Step 6 — final weight update  w^{n+1} with the stage-2 Lagrangian
        # rate; stage-2 displacement covers the second half-step, so the
        # per-step total is the Heun trapezoid f·Δt·(u¹+u²)/2.
        # ------------------------------------------------------------------
        if self.weight_scheme == "midpoint":
            w_dot_2 = k2 / y_safe
            w_new = w_n + self.dt * w_dot_2
            sigma_diag_2 = w_dot_2
        else:  # cayley
            rho_at_Zmid = rho_2 if rho_2 is not None else self._rho_at(gp_mid, Z_half_fc)
            rho_safe_2  = np.where(np.abs(rho_at_Zmid) > 0.0, rho_at_Zmid,
                                   1.0 + 0.0 * rho_at_Zmid)
            sigma_2 = k2 / rho_safe_2
            w_new = ((1.0 + 0.5 * self.dt * sigma_2) /
                     (1.0 - 0.5 * self.dt * sigma_2)) * w_n
            sigma_diag_2 = sigma_2

        if flow_on:
            dz_2, n_capped_2 = self._continuity_displacement(
                u_2, 0.5 * self.dt, Z_half_fc.shape)
        else:
            dz_2 = np.zeros_like(Z_half_fc)
            n_capped_2 = 0

        # ------------------------------------------------------------------
        # Step 7 — PBME half-step (second leg of Strang splitting), then
        # commit the stage-2 flow displacement onto the final positions
        # (zero when flow_fraction == 0).
        # ------------------------------------------------------------------
        t_mint2 = time.time()
        Z_new = np.asarray(
            self.dynamics.step(Z_half_fc, 0.5 * self.dt), dtype=np.float64
        )
        mint_wall_2 = time.time() - t_mint2
        Z_new_fc = Z_new + dz_2

        # Transported reference profile (Rung 2): advance the footpoint
        # Jacobian by the composition of the two half-step mapping-block
        # maps, so g rides the exact MInt flow.  No-op unless the surrogate
        # is a product surrogate with transport attached.
        if getattr(state.gp, "_is_product", False) and \
                getattr(state.gp, "_foot_jac", None) is not None:
            B1 = self.dynamics.mapping_block_jacobian(state.Z, 0.5 * self.dt)
            B2 = self.dynamics.mapping_block_jacobian(Z_half_fc, 0.5 * self.dt)
            state.gp.transport_footpoints(np.einsum("nij,njk->nik", B2, B1))

        # ------------------------------------------------------------------
        # Step 8 — refit GP_new on (Z^{n+1}, w^{n+1} · y) + KKT.
        # ------------------------------------------------------------------
        t_refit = time.time()
        target_new = w_new * y_n
        state.gp.refit(
            Z_train=Z_new_fc, y_train=target_new,
            moment_targets=state.moment_targets,
        )
        refit_wall = time.time() - t_refit

        # ------------------------------------------------------------------
        # State commit
        # ------------------------------------------------------------------
        state.Z                 = Z_new_fc
        state.correction_weight = w_new
        # state.y stays frozen
        state.t                += self.dt
        state.step_index       += 1

        # ------------------------------------------------------------------
        # Diagnostics
        # ------------------------------------------------------------------
        dw  = w_new - w_n
        db  = dw * y_n   # Delta(w*y) — well-defined regardless of scheme
        # Continuity-flow diagnostics: velocity magnitude and the density
        # denominator that regularizes it (replaces the old grad stats).
        if flow_on:
            fc_u_max   = float(np.max(np.abs(np.concatenate([u_1, u_2]))))
            fc_rho_min = float(np.min(np.abs(rho_2)))
        else:
            fc_u_max   = float("nan")
            fc_rho_min = float("nan")
        ret = {
            "Q":             Q_2,            # Q at the midpoint, used in final update
            "Q_applied":     -k2,            # what was actually applied: -k2 ≡ Q_2
            "Y":             data_2.Y,
            "n_q_clipped":   0,
            "n_q_nonfinite": int(np.sum(~np.isfinite(Q_2))),
            "n_q_overshoot": 0,
            "apply_q_clip":  self.apply_q_clip,
            "mint_wall":     float(mint_wall + mint_wall_2),
            "operator_wall": float(operator_wall_1 + operator_wall_2),
            "refit_wall":    float(refit_wall + refit_wall_mid),
            "weight_scheme": self.weight_scheme,
            # Weight-update diagnostics
            "w_min":         float(np.min(w_new)),
            "w_max":         float(np.max(w_new)),
            "w_mean":        float(np.mean(w_new)),
            "w_abs_max":     float(np.max(np.abs(w_new))),
            "dw_max":        float(np.max(np.abs(dw))),
            "dw_rms":        float(np.sqrt(np.mean(dw ** 2))),
            "sigma1_max":    float(np.max(np.abs(sigma_diag_1))),
            "sigma2_max":    float(np.max(np.abs(sigma_diag_2))),
            "k1_max":        float(np.max(np.abs(k1))),
            "k2_max":        float(np.max(np.abs(k2))),
            # Label-side diagnostics: well-defined for ANY scheme (they
            # describe how the label-product b = w*y evolved), so these
            # are now real numbers rather than NaN/0 placeholders.
            "label_scheme_id":         0.0 if self.weight_scheme == "midpoint" else 1.0,
            "omega_A_residual_norm":   _kkt_residual_norm(state.gp),
            "label_dy_max":            float(np.max(np.abs(db))),
            "label_dy_rms":            float(np.sqrt(np.mean(db ** 2))),
            "label_probability_drift": self._probability_drift(state, db),
            # Flow-correction diagnostics: genuinely zero when
            # flow_fraction == 0, real when it's on.
            "fc_applied":              float(self.flow_fraction),
            "fc_dz_max":               float(np.max(np.abs(dz_2))) if flow_on else 0.0,
            "fc_dz_rms":               float(np.sqrt(np.mean(dz_2 ** 2))) if flow_on else 0.0,
            "fc_n_capped":             float(n_capped_1 + n_capped_2),
            "fc_u_max":                fc_u_max,
            "fc_rho_min":              fc_rho_min,
        }
        return ret

    # ------------------------------------------------------------------
    # Exact-exponential Strang leg (label_scheme="strang"), new 2026-07
    # ------------------------------------------------------------------
    def _build_label_generator(self, Z: FloatArray, gp) -> FloatArray:
        """
        Assemble the leg-constant label-ODE generator A with  b_dot = A b
        at the frozen support cloud Z under the current surrogate fit.

        Vanilla GP:    rho_hat = k . alpha, alpha = K_y^{-1} b
                       -> A = L K_y^{-1},           L = compute_L_matrix.
        Product GP:    rho_hat = g * mu, inner alpha = K_y^{-1}(b / g_s)
                       -> A = L_prod K_y^{-1} diag(1/g_s),
                       L_prod = compute_L_matrix_product (static AND
                       transported profile — the footpoint Jacobian is
                       frozen between MInt legs, so A is leg-constant).

        A depends on the support positions and kernel hyperparameters
        ONLY — not on b.  Both are frozen for the duration of the middle
        splitting leg, so  b_dot = A b  is an exactly linear, autonomous
        ODE and  expm(dt A)  is its EXACT propagator (no intra-leg
        time-discretisation error at the surrogate level).
        """
        N = Z.shape[0]
        I_N = np.eye(N, dtype=np.float64)
        if getattr(gp, "_is_product", False):
            from .Operator import compute_L_matrix_product
            from .GP_Density import _g_safe
            L = compute_L_matrix_product(Z, gp, self.dynamics)
            Kinv = gp.solve_K(I_N)                       # inner mu-GP K_y^{-1}
            g_s = _g_safe(gp.profile_at(Z), gp._g_floor_rel)
            return (L @ Kinv) / g_s[None, :]
        L = compute_L_matrix(Z, gp=gp, dt=0.0, dynamics=self.dynamics)
        Kinv = gp.solve_K(I_N)
        return L @ Kinv

    def _step_strang(self, state: SimulationState) -> Dict:
        """
        Exact-leg Strang factorisation of the mapping-QCLE generator,

            exp(iL dt) = exp(iL0 dt/2) exp(iL' dt) exp(iL0 dt/2) + O(dt^3),

        with EVERY leg evaluated exactly (at the surrogate level):

        * iL0 legs: the MInt map Phi_{dt/2} — symplectic, so Liouville's
          theorem holds exactly for the trajectory flow and the labels
          b = w*y are invariant along its characteristics (the density is
          carried, not re-integrated).  No new approximation.
        * iL' leg: support points FROZEN.  In the RKHS representation the
          leg is the linear autonomous ODE  b_dot = A b  (A built by
          ``_build_label_generator`` at the half-drifted cloud), solved by
          the exact matrix exponential  b <- expm(dt A) b.  This replaces
          the Heun / Crank-Nicolson time-discretisation of the middle leg
          with its exact exponential — the only remaining time error of
          the composite step is the Strang commutator O(dt^3 [iL0,[iL0,iL']])
          per step (O(dt^2) global).

        Liouville accounting: the point flow is purely Hamiltonian
        (MInt), hence exactly volume-preserving; the excess operator iL'
        is a third-order dispersive (Kramers-Moyal D^(3)) term that is NOT
        a flow, has no characteristics, and provably makes the corrected
        continuity flow compressible — routing all of it through the
        label channel is the ONLY representation in which the ensemble
        measure remains exactly Liouvillian while the density still obeys
        the full generator.

        Sequence (t -> t + dt):
          1. Z_half = Phi_{dt/2}(Z^n)                     [exact iL0]
          2. refit gp_half on (Z_half, b^n) + KKT         [mid-slice fit]
          3. A = generator(Z_half, gp_half)               [leg-constant]
          4. b^{n+1} = expm(dt A) b^n                     [exact iL']
          5. Z^{n+1} = Phi_{dt/2}(Z_half)                 [exact iL0]
          6. transport footpoints (product_transported)
          7. refit state.gp on (Z^{n+1}, b^{n+1}) + KKT
        One operator build + one expm per step (vs two operator builds
        for the Heun/CN schemes).
        """
        from scipy.linalg import expm

        if state.correction_weight is None:
            state.correction_weight = np.ones(state.Z.shape[0], dtype=np.float64)
        w_n = state.correction_weight
        y_n = state.y
        y_safe = np.where(np.abs(y_n) > 0.0, y_n,
                          np.sign(y_n) + (y_n == 0.0).astype(np.float64))
        b_n = w_n * y_n

        # ---- leg 1: exact iL0 half-step -------------------------------
        t_mint = time.time()
        Z_half = np.asarray(self.dynamics.step(state.Z, 0.5 * self.dt),
                            dtype=np.float64)
        mint_wall = time.time() - t_mint

        # ---- mid-slice refit (labels unchanged: Liouville along iL0) ---
        t_refit_mid = time.time()
        gp_half = copy.deepcopy(state.gp)
        gp_half.refit(Z_train=Z_half, y_train=b_n,
                      moment_targets=state.moment_targets)
        refit_wall_mid = time.time() - t_refit_mid

        # ---- leg 2: exact exponential of the frozen-cloud iL' ----------
        t_op = time.time()
        A = self._build_label_generator(Z_half, gp_half)
        operator_wall = time.time() - t_op
        k1 = A @ b_n                       # rate at leg entry (diagnostic)
        t_exp = time.time()
        E_dtA = expm(self.dt * A)
        b_new = E_dtA @ b_n
        expm_wall = time.time() - t_exp
        k2 = A @ b_new                     # rate at leg exit (diagnostic)
        w_new = b_new / y_safe

        # ---- leg 3: exact iL0 half-step -------------------------------
        t_mint2 = time.time()
        Z_new = np.asarray(self.dynamics.step(Z_half, 0.5 * self.dt),
                           dtype=np.float64)
        mint_wall_2 = time.time() - t_mint2

        # Transported reference profile (Rung 2): compose the two exact
        # half-step mapping-block maps, as in _step_weight.
        if getattr(state.gp, "_is_product", False) and \
                getattr(state.gp, "_foot_jac", None) is not None:
            B1 = self.dynamics.mapping_block_jacobian(state.Z, 0.5 * self.dt)
            B2 = self.dynamics.mapping_block_jacobian(Z_half, 0.5 * self.dt)
            state.gp.transport_footpoints(np.einsum("nij,njk->nik", B2, B1))

        # ---- final refit ------------------------------------------------
        t_refit = time.time()
        state.gp.refit(Z_train=Z_new, y_train=b_new,
                       moment_targets=state.moment_targets)
        refit_wall = time.time() - t_refit

        state.Z                 = Z_new
        state.correction_weight = w_new
        state.t                += self.dt
        state.step_index       += 1

        db = b_new - b_n
        dw = w_new - w_n
        # amplification of the exact leg propagator on this cloud
        bn_norm = float(np.linalg.norm(b_n))
        amp = float(np.linalg.norm(b_new)) / bn_norm if bn_norm > 0 else float("nan")
        ret = {
            "Q":             k2,
            "Q_applied":     k2,
            "Y":             Z_half,
            "n_q_clipped":   0,
            "n_q_nonfinite": int(np.sum(~np.isfinite(b_new))),
            "n_q_overshoot": 0,
            "apply_q_clip":  self.apply_q_clip,
            "mint_wall":     float(mint_wall + mint_wall_2),
            "operator_wall": float(operator_wall + expm_wall),
            "refit_wall":    float(refit_wall + refit_wall_mid),
            "weight_scheme": self.weight_scheme,
            "w_min":         float(np.min(w_new)),
            "w_max":         float(np.max(w_new)),
            "w_mean":        float(np.mean(w_new)),
            "w_abs_max":     float(np.max(np.abs(w_new))),
            "dw_max":        float(np.max(np.abs(dw))),
            "dw_rms":        float(np.sqrt(np.mean(dw ** 2))),
            # generator-scale proxies (matrix scheme: not per-point scalars)
            "sigma1_max":    float(np.max(np.abs(A))),
            "sigma2_max":    amp,          # ||expm(dt A) b|| / ||b||
            "k1_max":        float(np.max(np.abs(k1))),
            "k2_max":        float(np.max(np.abs(k2))),
            "label_scheme_id":         3.0,
            "omega_A_residual_norm":   _kkt_residual_norm(state.gp),
            "label_dy_max":            float(np.max(np.abs(db))),
            "label_dy_rms":            float(np.sqrt(np.mean(db ** 2))),
            "label_probability_drift": self._probability_drift(state, db),
            "fc_applied":              0.0,
            "fc_dz_max":               0.0,
            "fc_dz_rms":               0.0,
            "fc_n_capped":             0.0,
            "fc_u_max":                float("nan"),
            "fc_rho_min":              float("nan"),
        }
        return ret

    def _step_linear(self, state: SimulationState) -> Dict:
        """
        Experimental label-ODE integrator (``label_scheme="linear"``, see
        the class docstring).  Advances b = w*y by the Cayley/Crank-
        Nicolson Pade approximant to exp(dt A), A = L K^{-1}, instead of
        the scalar per-point Heun/Cayley rule used by ``_step_weight``.
        Not yet validated against finite differences or small-N
        convergence — treat as a research prototype.
        """
        if state.correction_weight is None:
            state.correction_weight = np.ones(state.Z.shape[0], dtype=np.float64)
        w_n = state.correction_weight
        y_n = state.y
        y_safe = np.where(np.abs(y_n) > 0.0, y_n,
                          np.sign(y_n) + (y_n == 0.0).astype(np.float64))
        b_n = w_n * y_n
        N = state.Z.shape[0]
        I_N = np.eye(N, dtype=np.float64)

        # ---- Step 1: PBME half-step ----------------------------------
        t_mint = time.time()
        Z_half = np.asarray(self.dynamics.step(state.Z, 0.5 * self.dt),
                            dtype=np.float64)
        mint_wall = time.time() - t_mint

        # ---- Predictor: A^n = L(Z^n) K^{-1}(state.gp), half CN step ----
        # dt=0.0 -> half_tau=0 -> the internal backward pullback collapses
        # to the identity, matching the same no-pullback convention already
        # validated for Q via build_at_points/compute_Q_at_points (Q
        # evaluated directly at the trajectory point, not a pulled-back
        # midpoint).
        t_op = time.time()
        L_n = compute_L_matrix(state.Z, gp=state.gp, dt=0.0, dynamics=self.dynamics)
        Kinv_n = state.gp.solve_K(I_N)
        A_n = L_n @ Kinv_n
        operator_wall_1 = time.time() - t_op
        k1 = A_n @ b_n     # == Q at Z^n by construction (Q = L alpha = L K^-1 b)

        M_half = 0.25 * self.dt * A_n
        b_half = np.linalg.solve(I_N - M_half, (I_N + M_half) @ b_n)
        w_half_star = b_half / y_safe

        # ---- Refit midpoint GP -----------------------------------------
        t_refit_mid = time.time()
        gp_mid = copy.deepcopy(state.gp)
        gp_mid.refit(Z_train=Z_half, y_train=b_half,
                    moment_targets=state.moment_targets)
        refit_wall_mid = time.time() - t_refit_mid

        # ---- Corrector: A_mid = L(Z_half) K^{-1}(gp_mid), full CN step --
        t_op2 = time.time()
        L_mid = compute_L_matrix(Z_half, gp=gp_mid, dt=0.0, dynamics=self.dynamics)
        Kinv_mid = gp_mid.solve_K(I_N)
        A_mid = L_mid @ Kinv_mid
        operator_wall_2 = time.time() - t_op2
        k2 = A_mid @ b_half

        M_full = 0.5 * self.dt * A_mid
        b_new = np.linalg.solve(I_N - M_full, (I_N + M_full) @ b_n)
        w_new = b_new / y_safe

        # ---- Step 7: PBME half-step -------------------------------------
        t_mint2 = time.time()
        Z_new = np.asarray(self.dynamics.step(Z_half, 0.5 * self.dt),
                           dtype=np.float64)
        mint_wall_2 = time.time() - t_mint2

        # ---- Step 8: refit -----------------------------------------------
        t_refit = time.time()
        state.gp.refit(Z_train=Z_new, y_train=b_new,
                       moment_targets=state.moment_targets)
        refit_wall = time.time() - t_refit

        state.Z                 = Z_new
        state.correction_weight = w_new
        state.t                += self.dt
        state.step_index       += 1

        db = b_new - b_n
        dw = w_new - w_n
        ret = {
            "Q":             k2,
            "Q_applied":     k2,
            "Y":             Z_half,
            "n_q_clipped":   0,
            "n_q_nonfinite": int(np.sum(~np.isfinite(k2))),
            "n_q_overshoot": 0,
            "apply_q_clip":  self.apply_q_clip,
            "mint_wall":     float(mint_wall + mint_wall_2),
            "operator_wall": float(operator_wall_1 + operator_wall_2),
            "refit_wall":    float(refit_wall + refit_wall_mid),
            "weight_scheme": self.weight_scheme,
            "w_min":         float(np.min(w_new)),
            "w_max":         float(np.max(w_new)),
            "w_mean":        float(np.mean(w_new)),
            "w_abs_max":     float(np.max(np.abs(w_new))),
            "dw_max":        float(np.max(np.abs(dw))),
            "dw_rms":        float(np.sqrt(np.mean(dw ** 2))),
            # Not per-point scalars in this scheme; report the generator's
            # operator norm proxy (max abs entry) so the key stays populated.
            "sigma1_max":    float(np.max(np.abs(A_n))),
            "sigma2_max":    float(np.max(np.abs(A_mid))),
            "k1_max":        float(np.max(np.abs(k1))),
            "k2_max":        float(np.max(np.abs(k2))),
            "label_scheme_id":         2.0,
            "omega_A_residual_norm":   _kkt_residual_norm(state.gp),
            "label_dy_max":            float(np.max(np.abs(db))),
            "label_dy_rms":            float(np.sqrt(np.mean(db ** 2))),
            "label_probability_drift": self._probability_drift(state, db),
            "fc_applied":              0.0,
            "fc_dz_max":               0.0,
            "fc_dz_rms":               0.0,
            "fc_n_capped":             0.0,
            "fc_u_max":                float("nan"),
            "fc_rho_min":              float("nan"),
        }
        return ret

# =============================================================================
# Scheme factory
# =============================================================================

def build_scheme(name: str, dynamics: PBMEMIntDynamics, dt: float,
                 operator: Optional[QCLECorrection] = None,
                 apply_q_clip: bool = False,
                 q_clip_frac: float = 0.3,
                 q_clip_abs:  float = 1.0e-12,
                 flow_correction_axes: str = "P_only",
                 flow_correction_grad_floor: float = 1.0e-8,
                 flow_correction_step_cap: float = 0.5,
                 flow_fraction: float = 0.0,
                 weight_scheme: str = "midpoint",
                 label_scheme: str = "weight") -> DynamicsScheme:
    if name.lower() == "pbme":
        return PBMEScheme(dynamics, dt)
    if name.lower() == "midpoint":
        return MidpointScheme(dynamics, dt, operator=operator,
                              apply_q_clip=apply_q_clip,
                              q_clip_frac=q_clip_frac,
                              q_clip_abs=q_clip_abs,
                              flow_correction_axes=flow_correction_axes,
                              flow_correction_grad_floor=flow_correction_grad_floor,
                              flow_correction_step_cap=flow_correction_step_cap,
                              flow_fraction=flow_fraction,
                              weight_scheme=weight_scheme,
                              label_scheme=label_scheme)
    raise ValueError(f"Unknown scheme: {name!r}")


# =============================================================================
# Simulation orchestrator
# =============================================================================

class Simulation:
    """
    Run a dynamics scheme from t=0 to t=n_steps·dt, recording diagnostics
    at each step and periodic GP snapshots.
    """

    def __init__(self, config: DynamicsConfig, state: SimulationState,
                 dynamics: Optional[PBMEMIntDynamics] = None,
                 operator: Optional[QCLECorrection] = None,
                 run_metadata: Optional[Dict] = None) -> None:
        self.config    = config
        self.state     = state
        self.dynamics  = (dynamics if dynamics is not None
                          else state.gp.dynamics)

        # Guard: DynamicsConfig.use_density_diff is informational only — the
        # actual density architecture is determined by the runtime type of
        # state.gp.  Enforce consistency so the flag does not silently lie.
        gp_is_diff = hasattr(state.gp, "gp0") and hasattr(state.gp, "gp_delta")
        cfg_wants_diff = bool(getattr(config, "use_density_diff", False))
        if cfg_wants_diff != gp_is_diff:
            raise ValueError(
                f"DynamicsConfig.use_density_diff={cfg_wants_diff!r} but "
                f"state.gp is {'GPDensityDiff' if gp_is_diff else 'GPDensity'}. "
                "Pass use_density_diff=True to DynamicsConfig when state.gp "
                "is a GPDensityDiff, and False otherwise."
            )

        # Legacy hard-freeze path.  When enabled, lock (σ_f, ℓ, σ_n) and the
        # feature z-score statistics so every refit only rebuilds Ky and solves
        # for α.  The default is now False so that GPDensityConfig.refit_hyper_policy
        # (typically "breathing") is honored.  Set this to True only for
        # diagnostic comparisons against the legacy behavior.
        if getattr(config, "freeze_hypers_after_initial_fit", False):
            if not getattr(self.state.gp, "_hypers_frozen", False):
                self.state.gp.freeze_hypers()

        self.scheme    = build_scheme(
            config.scheme, self.dynamics, config.dt,
            operator=operator,
            apply_q_clip=config.apply_q_clip,
            q_clip_frac=config.q_clip_frac,
            q_clip_abs=config.q_clip_abs,
            flow_correction_axes=config.flow_correction_axes,
            flow_correction_grad_floor=config.flow_correction_grad_floor,
            flow_correction_step_cap=config.flow_correction_step_cap,
            flow_fraction=getattr(config, "flow_fraction", 0.0),
            weight_scheme=getattr(config, "weight_scheme", "midpoint"),
            label_scheme=getattr(config, "label_scheme", "weight"),
        )
        metadata = build_run_metadata(
            config=config, state=state, dynamics=self.dynamics,
            extra=run_metadata,
        )
        self.collector = Collector(scheme_name=self.scheme.name,
                                   run_metadata=metadata)
        self._last_alpha_for_diag: Optional[FloatArray] = None
        self._raw_drift_reference: Dict[str, float] = {}

    # -------------------------------------------------------------------------
    # Build initial state
    # -------------------------------------------------------------------------
    @staticmethod
    def build_initial_state(
        n_train:          int,
        classical_params: GaussianWavePacketParams,
        mapping_params:   MappingInitParams,
        gp_config:        GPDensityConfig,
        seed:             Optional[int] = 0,
        dynamics:         Optional[PBMEMIntDynamics] = None,
        *,
        sampling_mode:    str = "seo_signed",
        healthiest_sampling: bool = False,
        n_candidates:     int = 8,
        jackknife_blocks: int = 8,
        focused_random_angles: bool = True,
        use_density_diff: bool = False,
        density_diff_config: Optional["GPDensityDiffConfig"] = None,
        abs_target:       bool  = False,
        abs_cap_quantile: float = 0.999,
        omega_clip_quantile: Optional[float] = None,
        surrogate:        str = "gp",
        product_g_floor_rel: float = 1.0e-3) -> SimulationState:
        rng     = np.random.default_rng(seed)
        sampler = MMSTSampler(classical_params, mapping_params)

        # Always request return_selection=True so the return type is always
        # HealthySampleSelection regardless of healthiest_sampling.  When
        # healthiest_sampling=False the sampler wraps the raw MMSTSamples in
        # a sentinel HealthySampleSelection(report=None, best_index=0).
        if sampling_mode == "seo_signed":
            selection = sampler.sample_seo_signed(
                n_samples=n_train,
                rng=rng,
                healthiest=healthiest_sampling and not abs_target,
                n_candidates=n_candidates,
                jackknife_blocks=jackknife_blocks,
                return_selection=True,
                abs_target=abs_target,
                abs_cap_quantile=abs_cap_quantile,
            )
        elif sampling_mode == "focused":
            if abs_target:
                raise ValueError(
                    "abs_target=True is only meaningful for sampling_mode='seo_signed'."
                )
            selection = sampler.sample_focused(
                n_samples=n_train,
                rng=rng,
                random_angles=focused_random_angles,
                healthiest=healthiest_sampling,
                n_candidates=n_candidates,
                jackknife_blocks=jackknife_blocks,
                return_selection=True,
            )
        else:
            raise ValueError("sampling_mode must be 'seo_signed' or 'focused'.")

        samples = selection.samples

        Z = pack_z(samples.R, samples.P, samples.r, samples.p)
        if samples.target_density is None:
            raise ValueError(
                "Initial GP construction requires target_density labels. "
                "Focused initialization currently provides support points but not "
                "the exact density labels needed by the surrogate fit."
            )
        y = np.asarray(samples.target_density, dtype=np.float64).reshape(-1)
        proposal_density = (
            None if samples.proposal_density is None
            else np.asarray(samples.proposal_density, dtype=np.float64).reshape(-1)
        )
        target_density = y.copy()
        weight = (
            None if samples.weight is None
            else np.asarray(samples.weight, dtype=np.float64).reshape(-1)
        )

        dyn = dynamics if dynamics is not None else PBMEMIntDynamics()
        if use_density_diff:
            # Density-difference architecture: wrap two GPDensity instances
            # in GPDensityDiff.  Use the same gp_config for both baseline
            # and correction unless a custom density_diff_config was passed.
            from .GP_DensityDiff import GPDensityDiff, GPDensityDiffConfig
            if density_diff_config is None:
                density_diff_config = GPDensityDiffConfig(
                    base_config=gp_config,
                    delta_config=gp_config,
                )
            gp = GPDensityDiff(density_diff_config, dynamics=dyn)
        else:
            gp = GPDensity(gp_config, dynamics=dyn)

        if surrogate in ("product", "product_transported"):
            # Reference-profile surrogate (2026-07-04, finger-test fix):
            # rho_hat = g_SEO(x) * GP.  The analytic profile supplies the
            # exact mapping curvature the QCLE operator differentiates —
            # restoring the operator input that focused sampling cannot
            # inform (measured ~527x suppression on the plain GP).  The
            # GP is fitted to y/g; with focused sampling g is constant on
            # the focus torus, so the transform is exactly benign at t=0.
            # 'product_transported' (Rung 2) additionally rides the profile
            # along the exact MInt flow so the operator stays accurate as
            # the density's mapping structure evolves, not only at t=0.
            # Static-product analytic moments are supplied by ProductMoments;
            # transported-product global moments deliberately use cloud sums.
            from .GP_Density import GPDensityProduct
            if use_density_diff:
                raise ValueError("surrogate='product*' is not combined with "
                                 "use_density_diff in v1.")
            gp = GPDensityProduct(gp, hbar=mapping_params.hbar,
                                  init_state=mapping_params.init_state,
                                  nstates=mapping_params.nstates,
                                  g_floor_rel=product_g_floor_rel)
            if surrogate == "product_transported":
                gp.attach_footpoints(Z)
        elif surrogate != "gp":
            raise ValueError(f"Unknown surrogate {surrogate!r}; use 'gp', "
                             "'product', or 'product_transported'.")

        # ---------------------------------------------------------------
        # Apply the sampler's label-information-rank contract to the GP.
        #
        # The sampler tells the GP which phase-space axes its y-labels
        # actually constrain (rank=1) and which are intrinsically
        # constant along the data manifold (rank=0).  Axes with rank=0
        # have their ARD lengthscales pinned at physical anchors
        # provided in `samples.label_information.anchor_lengthscales`,
        # and are excluded from the GP's MLL/LOO-CV gradient throughout
        # the run (initial fit + every refit).
        #
        # Without this, focused sampling silently breaks: the four
        # mapping lengthscales receive zero data gradient (labels are
        # constant on the focus manifold by construction) and drift to
        # whatever the prior allows, producing 100-1000× over-magnified
        # marginals and sign-changing KKT-projected α (the "narrow band
        # + Mexican-hat" pathology).  With the pin, those four axes are
        # held at their physically motivated bandwidths (R_focus/2) and
        # only the two nuclear axes (R, P) are estimated from data —
        # which is exactly what the data supports.
        #
        # seo_signed sampling reports `LabelInformation.all_free()`,
        # which is a no-op pin (all axes free), so this path adds zero
        # overhead to legacy behavior.
        #
        # The pin handles density-diff transparently: GPDensityDiff
        # exposes the same pin_lengthscales API on its base + delta GPs.
        # ---------------------------------------------------------------
        label_info = getattr(samples, "label_information", None)
        if label_info is not None:
            if hasattr(gp, "pin_lengthscales"):
                gp.pin_lengthscales(label_info)
            else:
                # Density-difference path: pin both sub-GPs.
                if hasattr(gp, "base_gp") and hasattr(gp.base_gp, "pin_lengthscales"):
                    gp.base_gp.pin_lengthscales(label_info)
                if hasattr(gp, "delta_gp") and hasattr(gp.delta_gp, "pin_lengthscales"):
                    gp.delta_gp.pin_lengthscales(label_info)

        energies = np.asarray(dyn.energy(Z), dtype=np.float64).reshape(-1)
        if weight is None:
            # Focused mode: no proposal, no IS weights — every trajectory
            # contributes equally with weight 1/N to the energy target.
            # _signed_expectation reduces to the plain arithmetic mean
            # when weights are uniform, so we substitute ones(N) here.
            weight_for_energy = np.ones_like(energies, dtype=np.float64)
        else:
            weight_for_energy = weight
        E_target = _signed_expectation(energies, weight_for_energy,
                                       name="initial energy target")

        # Physical reasoning for the moment-target split
        # ------------------------------------------------
        # At t=0 the SEO-signed initial condition gives ⟨c00+c11⟩ = 1.  We
        # anchor the initial fit with {normalization, trace, energy}.
        #
        # WHY TRACE IS CONSERVED — correcting the previous (wrong) comment:
        #
        # The mapping Casimir  C = r_0² + r_1² + p_0² + p_1²  is an EXACT
        # algebraic invariant of the MInt mapping rotation.  This means
        #     c_00 + c_11 = C/(2ℏ) - 1
        # is preserved along every single trajectory.  Its expectation under
        # ANY Liouville-transported (PBME) or QCLE-corrected density is
        # therefore also an invariant:
        #     d/dt ⟨c_00+c_11⟩ = 0   (both PBME and QCLE).
        #
        # QCLE verification: the QCLE midpoint correction iL'_m contains only
        # third-order derivatives ∂³/∂P∂r_α∂r_β and ∂³/∂P∂p_α∂p_β.  Applying
        # the adjoint to (c_00+c_11), which is quadratic in mapping vars and
        # independent of P, yields zero.  So ⟨c_00+c_11⟩ is conserved exactly.
        #
        # BUG in previous version: dropping the trace constraint from
        # propagation_moment_targets caused the KKT projection to leave
        # ∫ ρ̂ (c_00+c_11) dz unconstrained.  As the breathing mode updates
        # the mapping lengthscales ℓ_{r_α}, ℓ_{p_α}, the analytic moment
        # integrals shift (M² = r_{α,j}² + ℓ_{r_α}² depends on ℓ), causing
        # the computed trace to drift far from 1 — reaching −1.5 by t=730
        # in test runs — which corrupts every population observable.
        #
        # FIX: keep "trace": 1.0 in propagation_moment_targets for all time.
        initial_moment_targets = {
            "normalization": 1.0,
            "trace": 1.0,
            "energy": E_target,
        }
        # Why normalization=1.0 is consistent with the MC energy target
        # ---------------------------------------------------------------
        # Energy uses a self-normalized IS estimator:
        #   E_target = Σ_i w_i E_i / Σ_i w_i
        # For normalization, the same self-normalized IS estimator gives:
        #   norm_target = Σ_i w_i · 1 / Σ_i w_i = 1.0  (exactly, by construction).
        # So hardcoding 1.0 IS the MC estimate — there is no mixing inconsistency.
        # (Unnormalized IS would give 1 + O(1/√N) ≠ 1; we use self-normalized IS
        # throughout, so both paths agree on 1.0.)
        propagation_moment_targets = {
            "normalization": 1.0,
            "trace": 1.0,       # conserved invariant — must stay constrained
            "energy": E_target,
        }

        gp.fit(Z_train=Z, y_train=y, moment_targets=initial_moment_targets)

        # Frozen geometric measure for the cloud Riemann sum
        #     ⟨A⟩(t) = Σ_i ω_i A(z_i(t)) y_i(t),
        # where  ω_i = 1/(N q(z_i^0))  and q is the (positive) proposal.
        # Liouville (det J = 1) keeps q(z_i(t)) = q(z_i^0) along trajectories,
        # so this measure is correct for all later times without update.
        #
        # Focused mode (no proposal) is the degenerate case: by construction
        # the trajectories are sampled FROM the target density itself, so
        # there is no separate proposal q.  We then set ω_i = 1/N (uniform)
        # and the IS estimator <A> = Σ ω·A·y reduces to the canonical
        # focused Monte Carlo average  <A> = (1/N) Σ A(z_i) y_i.
        N = int(Z.shape[0])
        if proposal_density is not None:
            omega_raw = (1.0 / (N * np.asarray(proposal_density, dtype=np.float64).reshape(-1)))
        else:
            # No proposal → uniform Monte Carlo weights.
            omega_raw = np.full(N, 1.0 / N, dtype=np.float64)

        # Optional ω clipping: cap ω at the `omega_clip_quantile` empirical
        # quantile.  Bounded above by ω_max ≡ Q_p(ω) so tail points cannot
        # dominate cloud Riemann sums.  Introduces a small bias which we
        # log via the saved diagnostics fields.
        omega = omega_raw
        omega_clip_diag: Dict[str, float] = {}
        if (omega_raw is not None) and (omega_clip_quantile is not None):
            qcap = float(omega_clip_quantile)
            if not (0.0 < qcap < 1.0):
                raise ValueError("omega_clip_quantile must be in (0,1).")
            omega_max = float(np.quantile(omega_raw, qcap))
            omega = np.minimum(omega_raw, omega_max)
            n_clipped = int(np.sum(omega_raw > omega_max))
            mass_lost = float(np.sum(omega_raw - omega))   # ∫ q-weight removed
            mass_total = float(np.sum(omega_raw))
            omega_clip_diag = {
                "omega_clip_quantile":     qcap,
                "omega_clip_max":          omega_max,
                "omega_clip_n":             float(n_clipped),
                "omega_clip_mass_frac":    (mass_lost / mass_total) if mass_total > 0 else 0.0,
                "omega_max_raw":           float(np.max(omega_raw)),
                "omega_max_used":          float(np.max(omega)),
            }
        elif omega_raw is not None:
            omega_clip_diag = {
                "omega_clip_quantile":     float("nan"),
                "omega_clip_max":          float("nan"),
                "omega_clip_n":            0.0,
                "omega_clip_mass_frac":    0.0,
                "omega_max_raw":           float(np.max(omega_raw)),
                "omega_max_used":          float(np.max(omega_raw)),
            }
        # Add a flag for which proposal was used
        omega_clip_diag["abs_target"] = 1.0 if abs_target else 0.0
        omega_clip_diag["abs_cap_quantile"] = (
            float(abs_cap_quantile) if abs_target else float("nan"))

        return SimulationState(
            Z=Z, y=y, gp=gp, t=0.0, step_index=0,
            moment_targets=propagation_moment_targets,
            initial_proposal_density=proposal_density,
            initial_target_density=target_density,
            initial_weight=weight,
            geometric_measure=omega,
            sampling_diagnostics=omega_clip_diag,
            sampling_mode=sampling_mode,
            sampling_healthiest=bool(healthiest_sampling),
            sampling_n_candidates=(int(n_candidates) if healthiest_sampling else 1),
            sampling_jackknife_blocks=(int(jackknife_blocks) if healthiest_sampling else 0),
            sampling_best_candidate_index=int(selection.best_index),
            sampling_score=(None if selection.report is None
                            else float(selection.report.score)),
            sampling_signed_ess=(None if selection.report is None
                                 else float(selection.report.signed_ess)),
            sampling_cancellation_ratio=(None if selection.report is None
                                         else float(selection.report.cancellation_ratio)),
        )

    # -------------------------------------------------------------------------
    # Measurement
    # -------------------------------------------------------------------------
    def _measure(self, scheme_diag: Dict,
                 wall_time: float) -> StepDiagnostics:
        cfg = self.config
        s   = self.state

        # ------------------------------------------------------------------
        # Two distinct notions of "fit RMS on support" — keep both so they
        # can be inspected independently.
        #
        # 1. train_fit_rms : RMS of the GP's training residual.
        #
        #    For VANILLA GPDensity, this is straightforward:
        #      ‖predict(Z) - state.y‖
        #    measures how well the surrogate fits its labels.
        #
        #    For GPDensityDiff, predict(Z) at support points returns the
        #    EXACT identity y0 + δ_GP_predict(Z) ≈ state.y by Liouville,
        #    so ‖predict(Z) - state.y‖ is structurally zero and useless as
        #    a diagnostic.  The meaningful metric for diff-GP is the
        #    δ-GP's own training residual:
        #      ‖gp_delta.predict(Z) - delta‖    where delta = state.y - y0.
        #    This is what the breathing optimizer is actually minimizing
        #    and what reflects the surrogate's per-step fit quality.
        #
        # 2. liouville_rms : RMS between predict(Z) and the INITIAL signed-
        #    SEO target densities y_0.  For PBME this measures how well the
        #    propagated surrogate represents the Liouville-transported
        #    density (a physics signal).  For midpoint this is NOT
        #    meaningful in general because the QCLE update breaks strict
        #    Liouville equality.
        # ------------------------------------------------------------------
        y_pred = s.gp.predict(s.Z)

        # Effective labels actually fitted by the surrogate this step.  The
        # midpoint/QCLE scheme refits the GP on  w⊙y  (the frozen sign-bearing
        # labels y scaled by the evolving per-point correction weight w) while
        # deliberately keeping ``state.y`` frozen at y.  Measuring the training
        # residual against the raw ``state.y`` therefore reports  w⊙y − y  — a
        # correction-weight artefact, NOT a fit error — which is exactly what
        # inflated fit_rms_on_support to ~1e-2 (and predict_rms/|y| to >1)
        # while the GP's true residual against what it fit was ~1e-7.  Schemes
        # with no correction (PBME) leave correction_weight=None ⇒ y_eff ≡ y,
        # so their diagnostics are unchanged.
        _w = s.correction_weight
        if _w is not None:
            _w = np.asarray(_w, dtype=np.float64).reshape(-1)
            y_eff = s.y * _w if _w.shape == np.asarray(s.y).reshape(-1).shape else s.y
        else:
            y_eff = s.y

        # Detect diff-GP and compute fit_rms against the δ-GP's residual.
        is_diff_gp = (hasattr(s.gp, "gp0") and hasattr(s.gp, "gp_delta"))
        if is_diff_gp:
            # δ targets at the current support: delta = y_eff - y0.  The δ-GP
            # is refit by GPDensityDiff.refit on (full target w⊙y) − y0, so the
            # comparison label here must likewise be y_eff − y0 (not y − y0) to
            # report the δ-GP's TRUE training residual rather than the weight
            # artefact w⊙y − y.
            delta_targets = y_eff - np.asarray(s.gp.y0, dtype=np.float64)
            # δ-GP predictions at the same support.  This is the SAME quantity
            # the breathing/Adam loop minimizes inside _fit_alpha_only, so the
            # number that comes out here is directly comparable to the
            # gp_delta.last_fit_rms diagnostic.
            delta_pred = np.asarray(s.gp.gp_delta.predict(s.Z), dtype=np.float64)
            train_fit_rms = float(np.sqrt(np.mean((delta_pred - delta_targets) ** 2)))
        else:
            train_fit_rms = float(np.sqrt(np.mean((y_pred - y_eff) ** 2)))

        if s.initial_target_density is not None:
            y0_arr = np.asarray(s.initial_target_density, dtype=np.float64).reshape(-1)
            liou_res = np.asarray(y_pred, dtype=np.float64).reshape(-1) - y0_arr
            liouville_rms = float(np.sqrt(np.mean(liou_res ** 2)))
            liouville_max = float(np.max(np.abs(liou_res)))
            _y0_rms = float(np.sqrt(np.mean(y0_arr ** 2)))
            liouville_rel = liouville_rms / _y0_rms if _y0_rms > 0.0 else float("nan")
            # Corrected residual: for the midpoint scheme the QCLE update
            # deliberately breaks strict Liouville constancy — the physically
            # meaningful transport residual is against the QCLE-predicted
            # carried value w*y0, not raw y0.  For PBME (w absent) the two
            # coincide, so ONE key serves both schemes in comparison figures.
            liou_res_c = np.asarray(y_pred, dtype=np.float64).reshape(-1) \
                         - np.asarray(y_eff, dtype=np.float64).reshape(-1)
            liouville_rms_corr = float(np.sqrt(np.mean(liou_res_c ** 2)))
        else:
            liouville_rms = liouville_max = liouville_rel = float("nan")
            liouville_rms_corr = float("nan")
        # fit_rms_on_support is kept for backward compatibility and always
        # equals train_fit_rms (the training residual of the most recent fit,
        # interpreted appropriately for the surrogate type).
        fit_rms = train_fit_rms

        # CRITICAL (2026-07 fix): the cloud Riemann sums inside compute_all
        # are documented as "Σ_i ω_i y_i(t) A(z_i(t)) — tracks the QCLE
        # correction through y_i(t)".  That contract dates from the era when
        # the midpoint scheme updated y in place.  Since the architecture
        # moved to frozen y + per-trajectory correction weight w, the live
        # density value at z_i is  w_i·y_i, and passing the FROZEN s.y here
        # silently removed the correction from every cloud_*/lw_* observable
        # (midpoint cloud populations were identical to PBME by construction;
        # only the GP-analytic km_/dp_/gpi_ moments saw the correction).
        # y_eff (= w⊙y, computed above; ≡ y when correction_weight is None,
        # i.e. PBME) is the label vector every estimator must see.
        obs = compute_all(
            gp=s.gp, Z=s.Z, Q=scheme_diag.get("Q"), dt=cfg.dt,
            y=y_eff, dynamics=self.dynamics,
            omega=s.geometric_measure,
            include_abs_integral=cfg.include_abs_integral,
        )
        # Extra correction bookkeeping: raw operator contraction vs actually
        # applied update (the latter only differs if q-clipping is enabled).
        Q_applied = scheme_diag.get("Q_applied")
        if Q_applied is None:
            Q_applied = scheme_diag.get("Q")
        cs_applied = compute_all(
            gp=s.gp, Z=s.Z, Q=Q_applied, dt=cfg.dt,
            y=y_eff, dynamics=self.dynamics,
            omega=s.geometric_measure,
            include_abs_integral=False,
        )
        for key in ("cs_q_rms", "cs_q_max", "cs_dtq_rms", "cs_dtq_max", "cs_dq_over_y_rms", "cs_dq_over_y_max"):
            if key in cs_applied:
                obs[f"applied_{key}"] = float(cs_applied[key])
        obs["n_q_clipped"] = float(scheme_diag.get("n_q_clipped", 0))
        obs["n_q_nonfinite"] = float(scheme_diag.get("n_q_nonfinite", 0))
        obs["n_q_overshoot"] = float(scheme_diag.get("n_q_overshoot", 0))
        obs["apply_q_clip"] = float(1 if scheme_diag.get("apply_q_clip", False) else 0)
        # Posterior variance of the third-derivative operator is not evaluated
        # by the production path.  Saving an explicit flag prevents the mean-Q
        # curve from being misreported as an uncertainty calculation.
        obs["operator_variance_computed"] = 0.0

        # Cloud-transport diagnostics — Riemann sums in (omega, y).
        # Same y_eff contract as compute_all above: the cloud-weighted
        # energy/trace must see the live density value w⊙y, not frozen y.
        support_diag = _weighted_support_diagnostics(
            s.Z, self.dynamics, s.geometric_measure, y_eff)
        for k, v in support_diag.items():
            obs[k] = float(v)

        # Reviewer-facing cumulative RAW conservation curves.  These are not
        # self-normalized and are referenced to the actual step-0 estimate,
        # so a flat curve cannot be manufactured by dividing through the live
        # norm.  Self-normalized lw_* quantities remain available separately.
        drift_sources = {
            "raw_norm": "cloud_norm",
            "raw_energy": "cloud_weighted_energy",
            "raw_trace": "cloud_weighted_trace",
            "raw_mapping_radius_sq": "cloud_weighted_mapping_radius_sq",
        }
        for label, key in drift_sources.items():
            current = float(obs.get(key, float("nan")))
            if label not in self._raw_drift_reference and np.isfinite(current):
                self._raw_drift_reference[label] = current
            reference = self._raw_drift_reference.get(label, float("nan"))
            drift = current - reference
            scale = max(abs(reference), np.finfo(float).tiny)
            obs[f"{label}_initial"] = reference
            obs[f"{label}_drift"] = drift
            obs[f"{label}_relative_drift"] = drift / scale

        # Product-profile floor audit: both its absolute scale and the fraction
        # of support labels whose y/g transformation is regularized are saved.
        if getattr(s.gp, "_is_product", False):
            try:
                g_profile = np.asarray(s.gp.profile_at(s.Z), dtype=np.float64)
                g_floor = float(getattr(s.gp, "_g_floor_rel", 0.0)
                                * np.max(np.abs(g_profile)))
                obs["product_g_floor_abs"] = g_floor
                obs["product_g_floor_rel"] = float(getattr(s.gp, "_g_floor_rel", 0.0))
                obs["product_g_floor_fraction"] = float(np.mean(np.abs(g_profile) < g_floor))
            except Exception:
                obs["product_g_floor_abs"] = float("nan")
                obs["product_g_floor_rel"] = float(getattr(s.gp, "_g_floor_rel", float("nan")))
                obs["product_g_floor_fraction"] = float("nan")

        # MInt-invariant check on the support cloud: r_0²+p_0²+r_1²+p_1²
        # is conserved along EVERY trajectory by the mapping rotation, so its
        # density-weighted expectation
        #     ⟨|r|²+|p|²⟩(t) = Σ_i ω_i y_i (r²+p²)_i
        # is conserved exactly under PBME (where y is frozen) and conserved
        # up to QCLE corrections under midpoint.  Constancy is a direct
        # cloud-level Casimir check.
        Zb = np.asarray(s.Z, dtype=np.float64).reshape(-1, D)
        mapping_radius_sq_per_point = (Zb[:, 2] ** 2 + Zb[:, 3] ** 2
                                        + Zb[:, 4] ** 2 + Zb[:, 5] ** 2)
        if s.geometric_measure is not None:
            _omega = np.asarray(s.geometric_measure, dtype=np.float64).reshape(-1)
            # Effective labels: exactly conserved under PBME (w ≡ 1, y frozen),
            # conserved up to the QCLE correction (through w) under midpoint —
            # which is the physically meaningful statement of the Casimir check.
            _y     = np.asarray(y_eff,               dtype=np.float64).reshape(-1)
            obs["cloud_mapping_radius_sq_mean"] = float(
                np.dot(_omega * _y, mapping_radius_sq_per_point))
            # Self-normalised version (stable even when Σω_iy_i drifts)
            cloud_norm_msr = float(np.sum(_omega * _y))
            _D_msr = cloud_norm_msr if abs(cloud_norm_msr) > 1e-15 else 1.0
            obs["lw_mapping_radius_sq_msr"] = obs["cloud_mapping_radius_sq_mean"] / _D_msr
        else:
            obs["cloud_mapping_radius_sq_mean"] = float(
                np.mean(mapping_radius_sq_per_point))
        # Trajectory-level (unweighted) bounds — independent integrator check.
        obs["cloud_mapping_radius_sq_std"] = float(np.std(mapping_radius_sq_per_point))
        obs["cloud_mapping_radius_sq_min"] = float(np.min(mapping_radius_sq_per_point))
        obs["cloud_mapping_radius_sq_max"] = float(np.max(mapping_radius_sq_per_point))

        init_sw = _signed_weight_summary(s.initial_weight)
        for k, v in init_sw.items():
            obs[f"init_sw_{k}"] = float(v)
        if "cloud_weighted_energy" in support_diag:
            obs["cloud_weighted_energy_err"] = float(support_diag["cloud_weighted_energy"] - s.moment_targets["energy"])
        if "cloud_weighted_trace" in support_diag:
            # The electronic trace is anchored to 1 at t=0 by the SEO initial
            # condition but is NOT a Liouville invariant; the deviation from 1
            # is therefore a physics signal (population transfer) rather than
            # a bookkeeping error.  Report it as such, for diagnostics only.
            obs["cloud_weighted_trace_err"] = float(support_diag["cloud_weighted_trace"] - 1.0)

        obs["gp_free_fit_rms"] = float(getattr(s.gp, "last_free_fit_rms", float("nan")))
        obs["gp_free_fit_mae"] = float(getattr(s.gp, "last_free_fit_mae", float("nan")))
        obs["gp_free_fit_r2"] = float(getattr(s.gp, "last_free_fit_r2", float("nan")))
        obs["gp_fit_mae"] = float(getattr(s.gp, "last_fit_mae", float("nan")))
        obs["gp_fit_r2"] = float(getattr(s.gp, "last_fit_r2", float("nan")))
        obs["gp_train_fit_rms"] = float(train_fit_rms)
        # Adaptive-trigger observability (2026-07): the refit policy
        # re-optimizes hyperparameters only when the cloud outgrows the
        # kernel, Var(Z_d)/ell_d^2 > adaptive_cloud_ratio_target (4.0 by
        # default, i.e. cloud std > 2*ell).  Export the per-axis ratio for
        # the free bath axes so "why are the lengthscales flat" is
        # answerable from a figure: flat hyperparameters with ratio << 4
        # is the policy working, not a frozen fit.
        try:
            _gp_for_ell = getattr(s.gp, "gp_delta", s.gp)
            _ell = np.asarray(_gp_for_ell.lengthscales,
                              dtype=np.float64).reshape(-1)
            _var = np.var(np.asarray(s.Z, dtype=np.float64), axis=0)
            obs["adapt_ratio_R"] = float(_var[0] / (_ell[0] ** 2 + 1e-30))
            obs["adapt_ratio_P"] = float(_var[1] / (_ell[1] ** 2 + 1e-30))
            obs["adapt_triggered"] = float(
                bool(getattr(_gp_for_ell, "_adaptive_triggered_last_refit",
                             False)))
            obs["adapt_refit_failed"] = float(
                bool(getattr(_gp_for_ell, "last_breathing_failed", False)))
            obs["adapt_refit_failure_code"] = float(
                int(getattr(_gp_for_ell, "last_breathing_failure_code", 0)))
            obs["adapt_refit_failure_count"] = float(
                int(getattr(_gp_for_ell, "breathing_failure_count", 0)))
        except Exception:
            obs["adapt_ratio_R"] = float("nan")
            obs["adapt_ratio_P"] = float("nan")
            obs["adapt_triggered"] = float("nan")
            obs["adapt_refit_failed"] = float("nan")
            obs["adapt_refit_failure_code"] = float("nan")
            obs["adapt_refit_failure_count"] = float("nan")

        obs["gp_liouville_rms"] = float(liouville_rms)
        obs["gp_liouville_max"] = float(liouville_max)
        obs["gp_liouville_rel"] = float(liouville_rel)
        obs["gp_liouville_rms_corrected"] = float(liouville_rms_corr)
        obs["gp_constraint_delta_rmse"] = float(getattr(s.gp, "constraint_delta_rmse", float("nan")))
        obs["gp_constraint_delta_mae"] = float(getattr(s.gp, "constraint_delta_mae", float("nan")))
        obs["gp_constraint_delta_r2"] = float(getattr(s.gp, "constraint_delta_r2", float("nan")))
        obs["gp_opt_total_loss"] = float(getattr(s.gp, "last_opt_total_loss", float("nan")))
        obs["gp_opt_nll_loss"] = float(getattr(s.gp, "last_opt_nll_loss", float("nan")))
        obs["gp_opt_reg_loss"] = float(getattr(s.gp, "last_opt_reg_loss", float("nan")))
        obs["gp_opt_train_mae"] = float(getattr(s.gp, "last_opt_train_mae", float("nan")))
        obs["gp_opt_train_r2"] = float(getattr(s.gp, "last_opt_train_r2", float("nan")))
        obs["gp_opt_val_mae"] = float(getattr(s.gp, "last_opt_val_mae", float("nan")))
        obs["gp_opt_val_r2"] = float(getattr(s.gp, "last_opt_val_r2", float("nan")))
        obs["gp_opt_steps"] = float(getattr(s.gp, "last_opt_steps", float("nan")))
        obs["gp_opt_best_step"] = float(getattr(s.gp, "last_opt_best_step", float("nan")))
        obs["gp_opt_early_stopped"] = float(1 if getattr(s.gp, "last_opt_early_stopped", False) else 0)

        # GP coefficient health
        alpha_stats, alpha_cur = _alpha_health(s.gp)
        for k, v in alpha_stats.items():
            obs[k] = float(v)
        for k, v in _delta_alpha_health(self._last_alpha_for_diag, alpha_cur).items():
            obs[k] = float(v)
        self._last_alpha_for_diag = alpha_cur.copy()

        # ── Surrogate-vs-cloud FAITHFULNESS BATTERY ────────────────────
        # Per-step diagnostics: ESS, LOO residuals, cond(K), predict
        # residual.  Reads from gp + (Z, y, ω).  All "faith_*" keys.
        # ``correction_weight`` is forwarded so the predict-residual term is
        # measured against the EFFECTIVE labels w⊙y the GP actually fit
        # (None for schemes without a correction, e.g. PBME).
        for k, v in _surrogate_faithfulness(
                s.gp, s.Z, s.y, s.geometric_measure,
                weight=s.correction_weight).items():
            obs[k] = float(v)

        # Faithfulness-Test #2: GP-integral vs cloud-Riemann moment agreement.
        # Must be invoked AFTER both km_* (from compute_all_observables) and
        # cloud_weighted_* (from _weighted_support_diagnostics) have been
        # written to obs.
        for k, v in _gp_vs_cloud_moment_agreement(obs).items():
            obs[k] = float(v)

        # Six-dimensional extension diagnostic: parallel KDE surrogate.
        # Build a Gaussian KDE from the same effective (Z, omega, y_eff)
        # contract the GP sees and
        # emit its analytic moment integrals as `kde_*` keys.  The KDE
        # is non-negative by construction (for focused mode where
        # ω·y ≥ 0) so its moments are reliable physical readings.
        # Comparing `km_*` (GP) to `kde_*` (KDE) measures sensitivity to
        # the GP's off-support 6D extension.  It is not used as the physical
        # PBME nuclear-density comparison; that uses ProjectedNuclearGP on
        # the saved geometric measure.
        # DOES NOT participate in dynamics; the QCLE pipeline still
        # uses the GP.
        if s.geometric_measure is not None:
            try:
                from .KDEDensity import build_kde_from_gp
                kde = build_kde_from_gp(
                    s.gp, omega=np.asarray(s.geometric_measure, dtype=np.float64),
                    y=np.asarray(y_eff, dtype=np.float64),
                )
                kde_moms = kde.compute_moment_values()
                for k, v in kde_moms.items():
                    obs[f"kde_{k}"] = float(v)
                # Also report GP-vs-KDE relative differences (parallel
                # to the gp_vs_cloud comparison).
                for short, gp_key, kde_key in (
                    ("norm",   "km_normalization", "kde_normalization"),
                    ("trace",  "km_trace",         "kde_trace"),
                    ("energy", "km_energy",        "kde_energy"),
                ):
                    gp_v  = obs.get(gp_key,  float("nan"))
                    kde_v = obs.get(kde_key, float("nan"))
                    if np.isfinite(gp_v) and np.isfinite(kde_v):
                        denom = max(abs(gp_v), abs(kde_v), 1.0e-300)
                        obs[f"gp_vs_kde_{short}_abs"] = float(abs(gp_v - kde_v))
                        obs[f"gp_vs_kde_{short}_rel"] = float(abs(gp_v - kde_v) / denom)
                    else:
                        obs[f"gp_vs_kde_{short}_abs"] = float("nan")
                        obs[f"gp_vs_kde_{short}_rel"] = float("nan")
            except Exception as _e:
                # KDE diagnostic is fully optional; any failure must not
                # block the run.  Log a NaN sentinel so the missing key
                # is detectable in postprocessing.
                obs["kde_normalization"] = float("nan")
                obs["kde_trace"]         = float("nan")
                obs["kde_energy"]        = float("nan")

        # Statistical geometry of the cloud and the surrogate
        cloud_stats = _cloud_center_variance(s.Z)
        gp_stats = _gp_surrogate_center_variance(s.gp)
        trust = _gp_density_trust_region(s.gp, k_sigma=2.0)
        snr = _gp_signal_noise_diagnostics(s.gp, y_pred)
        for i, lbl in enumerate(_DIM_LABELS):
            obs[f"cloud_mean_{lbl}"] = float(cloud_stats["mean"][i])
            obs[f"cloud_var_{lbl}"] = float(cloud_stats["var"][i])
            obs[f"cloud_std_{lbl}"] = float(cloud_stats["std"][i])
            obs[f"cloud_min_{lbl}"] = float(cloud_stats["min"][i])
            obs[f"cloud_max_{lbl}"] = float(cloud_stats["max"][i])
            obs[f"gp_mean_{lbl}"] = float(gp_stats["mean"][i])
            obs[f"gp_var_{lbl}"] = float(gp_stats["var"][i])
            obs[f"gp_std_{lbl}"] = float(gp_stats["std"][i])
            obs[f"trust_lo_{lbl}"] = float(trust["lower"][i])
            obs[f"trust_hi_{lbl}"] = float(trust["upper"][i])
            obs[f"trust_halfwidth_{lbl}"] = float(trust["kernel_halfwidth"][i])
        for k, v in snr.items():
            obs[k] = float(v)

        # Per-step wall-time breakdown (seconds).  The scheme returns these
        # keys explicitly; they default to 0.0 at step 0 or if the scheme did
        # not supply them.  The total scheme step cost is reported as
        # `step_wall_total` so it can be plotted against its three components.
        obs["mint_wall"]     = float(scheme_diag.get("mint_wall",     0.0))
        obs["operator_wall"] = float(scheme_diag.get("operator_wall", 0.0))
        obs["refit_wall"]    = float(scheme_diag.get("refit_wall",    0.0))
        obs["step_wall_total"] = float(wall_time)
        obs["measure_wall"] = float(max(0.0, wall_time
                                        - obs["mint_wall"]
                                        - obs["operator_wall"]
                                        - obs["refit_wall"]))

        # Flow-correction and label-integrator diagnostics.
        #
        # The midpoint scheme returns these schema-stable keys on every step
        # (NaN/0 for variants that carry no explicit flow correction or label
        # ODE, e.g. the weight-based Cayley/Heun scheme).  They were previously
        # dropped here, so Visualization.py had nothing to plot.  Persist them
        # verbatim so the flow-correction (`fc_*`) and label-integrator
        # (`label_*`) figures can render — and show real curves the moment a
        # scheme that DOES drive them is selected.
        for _k in ("fc_applied", "fc_dz_max", "fc_dz_rms", "fc_n_capped",
                   "fc_u_max", "fc_rho_min",
                   "label_scheme_id", "omega_A_residual_norm",
                   "label_dy_max", "label_dy_rms", "label_probability_drift",
                   # Weight-channel activity: the correction-weight integrator
                   # (Heun/Cayley on w) IS the label integrator of the default
                   # midpoint scheme.  Persisting these makes its per-step
                   # activity visible in the label-integrator figure set
                   # instead of only the derived Δ(w·y) statistics.
                   "dw_rms", "dw_max", "w_min", "w_max", "w_mean",
                   "w_abs_max", "sigma1_max", "sigma2_max",
                   "k1_max", "k2_max"):
            if _k in scheme_diag:
                obs[_k] = float(scheme_diag[_k])

        return StepDiagnostics(
            step_index=s.step_index, t=s.t, wall_time=wall_time,
            sigma_f=s.gp.sigma_f, sigma_n=s.gp.sigma_n,
            lengthscales=s.gp.lengthscales.copy(),
            fit_rms_on_support=fit_rms,
            values=obs,
        )

    def _snapshot(self) -> Snapshot:
        gp    = self.state.gp
        alpha = np.asarray(
            gp._alpha.detach().cpu().numpy()
            if hasattr(gp._alpha, "detach") else gp._alpha,
            dtype=np.float64,
        ).reshape(-1)
        fmean = None if getattr(gp, "_feature_mean", None) is None else np.asarray(gp._feature_mean.detach().cpu().numpy() if hasattr(gp._feature_mean, "detach") else gp._feature_mean, dtype=np.float64).reshape(-1)
        fstd = None if getattr(gp, "_feature_std", None) is None else np.asarray(gp._feature_std.detach().cpu().numpy() if hasattr(gp._feature_std, "detach") else gp._feature_std, dtype=np.float64).reshape(-1)

        # Density-difference extras (None for vanilla GPDensity).
        is_diff = hasattr(gp, "gp0") and hasattr(gp, "gp_delta")
        alpha_base = None; y0_snap = None
        sigma_f_base = None; sigma_n_base = None; ell_base = None
        delta_snap = None
        if is_diff:
            a_b = gp.gp0._alpha
            alpha_base = np.asarray(
                a_b.detach().cpu().numpy() if hasattr(a_b, "detach") else a_b,
                dtype=np.float64,
            ).reshape(-1).copy()
            y0_snap = None if gp.y0 is None else gp.y0.copy()
            sigma_f_base = float(gp.gp0.sigma_f)
            sigma_n_base = float(gp.gp0.sigma_n)
            ell_base = np.asarray(gp.gp0.lengthscales, dtype=np.float64).copy()
            if y0_snap is not None:
                # δ targets actually fitted by the δ-GP are (w⊙y − y0), not
                # (y − y0) — same contract as the fit_rms diagnostic above.
                _w_d = self.state.correction_weight
                _y_live = (self.state.y * _w_d
                           if _w_d is not None
                           and np.asarray(_w_d).shape == np.asarray(self.state.y).shape
                           else self.state.y)
                delta_snap = np.asarray(_y_live, dtype=np.float64) - y0_snap

        # Note: Snapshot.y stores the raw label vector; Snapshot.alpha
        # stores the *correction* alpha when is_density_diff=True (so that
        # downstream code that reads a single alpha sees the quantity
        # directly relevant to fit diagnostics on δ).  The baseline α is
        # in alpha_base.
        feature_zscore_flag = False
        cfg = getattr(gp, "config", None)
        if cfg is not None:
            feature_zscore_flag = bool(getattr(cfg, "feature_zscore", False))
        else:
            # density-diff: inherit from the correction GP's config
            try:
                feature_zscore_flag = bool(getattr(gp.gp_delta.config, "feature_zscore", False))
            except Exception:
                feature_zscore_flag = False

        # Snapshot.y contract (Collector docstring): "the effective label
        # vector actually fitted by the GP, i.e. correction_weight * raw
        # initial y".  Storing frozen state.y here (the previous behaviour)
        # broke every downstream consumer that reads snap.y as the live
        # density value — the faithfulness cloud-KDE panels, the GP
        # reconstruction path, and the Compare_gp_se_qcle cloud density —
        # all of which then silently rendered the UNcorrected density for
        # midpoint runs.  The raw sampled labels remain recoverable as
        # snap.y / snap.weight (weight carries w when present).
        _w_snap = self.state.correction_weight
        y_snap = (self.state.y * _w_snap
                  if _w_snap is not None
                  and np.asarray(_w_snap).shape == np.asarray(self.state.y).shape
                  else self.state.y)
        return Snapshot(
            step_index=self.state.step_index, t=self.state.t,
            Z=self.state.Z.copy(), y=np.asarray(y_snap, dtype=np.float64).copy(),
            alpha=alpha.copy(), sigma_f=gp.sigma_f, sigma_n=gp.sigma_n,
            lengthscales=gp.lengthscales.copy(),
            feature_mean=fmean, feature_std=fstd,
            feature_zscore=feature_zscore_flag,
            proposal_density=(None if self.state.initial_proposal_density is None
                              else self.state.initial_proposal_density.copy()),
            target_density=(None if self.state.initial_target_density is None
                            else self.state.initial_target_density.copy()),
            weight=(None if (self.state.correction_weight is None
                              and self.state.initial_weight is None)
                    else (self.state.correction_weight.copy()
                          if self.state.correction_weight is not None
                          else self.state.initial_weight.copy())),
            geometric_measure=(None if self.state.geometric_measure is None
                               else self.state.geometric_measure.copy()),
            # Diff-GP extras
            is_density_diff=is_diff,
            alpha_base=alpha_base,
            y0=y0_snap,
            sigma_f_base=sigma_f_base,
            sigma_n_base=sigma_n_base,
            lengthscales_base=ell_base,
            delta=delta_snap,
            is_product=bool(getattr(gp, "_is_product", False)),
            product_hbar=(float(getattr(gp, "_hbar"))
                          if getattr(gp, "_is_product", False) else None),
            product_init_state=(int(getattr(gp, "_init_state"))
                                if getattr(gp, "_is_product", False) else None),
            product_nstates=(int(getattr(gp, "_nstates"))
                             if getattr(gp, "_is_product", False) else None),
            product_g_floor_rel=(float(getattr(gp, "_g_floor_rel"))
                                 if getattr(gp, "_is_product", False) else None),
            product_transported=bool(
                getattr(gp, "_is_product", False)
                and getattr(gp, "_footpoints", None) is not None),
        )

    # -------------------------------------------------------------------------
    # Run loop
    # -------------------------------------------------------------------------
    def run(self) -> None:
        cfg = self.config

        # step-0 diagnostics
        t0    = time.time()
        diag0 = self._measure(scheme_diag={"Q": None}, wall_time=0.0)
        self.collector.record_diagnostics(diag0)
        if cfg.snapshot_every:
            self.collector.record_snapshot(self._snapshot())

        if cfg.verbose:
            v = diag0.values
            print(f"[{self.scheme.name}] step 0   t=0.000  "
                  f"norm={v['km_normalization']:+.3e}  "
                  f"E_phys={v.get('lw_energy', v.get('spe_E_density', float('nan'))):+.6e}  "
                  f"<H>={v.get('lw_energy', v.get('km_energy', float('nan'))):+.6e}  "
                  f"P0={v.get('lw_P0', v.get('cloud_weighted_P0', float('nan'))):+.4f}  P1={v.get('lw_P1', v.get('cloud_weighted_P1', float('nan'))):+.4f}  "
                  f"Psum={v.get('lw_P_sum', v.get('cloud_weighted_trace', float('nan'))):+.4f}  "
                  f"essf_c={v.get('ce_ess_c_frac', float('nan')):.3e}  "
                  f"|c|max={v.get('ce_max_abs_c_frac', float('nan')):.3e}  "
                  f"essf_w={v.get('init_sw_ess_frac', float('nan')):.3e}  "
                  f"essf_y={v.get('sw_ess_frac', float('nan')):.3e}  "
                  f"chi={v.get('sw_cancel_ratio', float('nan')):.3e}  "
                  f"fit_rms={diag0.fit_rms_on_support:.3e}  fit_mae={v.get('gp_fit_mae', float('nan')):.3e}  "
                  f"fit_r2={v.get('gp_fit_r2', float('nan')):+.3e}  "
                  f"liou_rms={v.get('gp_liouville_rms', float('nan')):.3e}  "
                  f"loss={v.get('gp_opt_total_loss', float('nan')):.3e}  reg={v.get('gp_opt_reg_loss', float('nan')):.3e}  "
                  f"Rμ={v.get('cloud_mean_R', float('nan')):+.3e}  Pμ={v.get('cloud_mean_P', float('nan')):+.3e}  "
                  f"VR={v.get('cloud_var_R', float('nan')):.3e}  VP={v.get('cloud_var_P', float('nan')):.3e}  "
                  f"σn={diag0.sigma_n:.3e}  σn(norm)={v.get('sigma_n_normalized', float('nan')):.3e}  "
                  f"snr={v.get('sigma_f_over_sigma_n', float('nan')):.3e}"
                  + (f"  α-neg={v.get('alpha_neg_frac', float('nan')):.3f}"
                     f"  α-negL1={v.get('alpha_neg_l1_frac', float('nan')):.3f}"
                     f"  α-min/max={v.get('alpha_min_to_max_ratio', float('nan')):+.3f}"
                     if v.get('alpha_test_active', 0.0) > 0.5 else "")
                  + f"  Δnorm={v.get('gp_vs_cloud_norm_rel', float('nan')):.2e}"
                  + f"  ΔE={v.get('gp_vs_cloud_energy_rel', float('nan')):.2e}")

        # ESS / Q warnings are rate-limited so a long degenerate run doesn't
        # produce thousands of identical warnings.  We print the first time
        # each threshold is crossed and then again every ~50 steps while the
        # condition persists, so the user notices without being flooded.
        _ess_warned_at = -999
        _q_warned_at = -999
        _last_resample_step = -10_000
        _total_resamples = 0
        ESS_WARNING_THRESHOLD = 0.02      # 2% ESS is dangerous
        ESS_WARNING_INTERVAL = 50         # steps
        Q_OVERSHOOT_WARNING_INTERVAL = 25
        banner_ess = False
        banner_q = False

        # Make the ESS-resampling policy reachable from the hot path without
        # repeated getattr.  Defaults preserve old behavior.
        rs_enabled = bool(getattr(cfg, "enable_ess_resampling", False))
        rs_threshold = float(getattr(cfg, "ess_resample_threshold", 0.05))
        rs_cooldown = int(getattr(cfg, "ess_resample_cooldown", 25))
        rs_max = int(getattr(cfg, "ess_resample_max", 100))

        for k in range(1, cfg.n_steps + 1):
            ts          = time.time()
            scheme_diag = self.scheme.step(self.state)
            wall        = time.time() - ts

            diag = self._measure(scheme_diag=scheme_diag, wall_time=wall)
            self.collector.record_diagnostics(diag)

            if cfg.snapshot_every and (k % cfg.snapshot_every == 0
                                       or k == cfg.n_steps):
                self.collector.record_snapshot(self._snapshot())

            # ------ Health warnings (rate-limited, stderr-ish format) ------
            v = diag.values
            # ESS diagnostic: use the INITIAL SAMPLING WEIGHT ESS (init_sw_ess_frac)
            # for health warnings and ESS-triggered resampling.
            # sw_ess_frac (label-vector ESS) is inherently tiny for a signed
            # Wigner density — it measures sign cancellation in the DENSITY VALUES
            # y_i, not in the sampling weights.  For PBME this stays constant
            # throughout the run and firing a warning every step is misleading.
            essf_label = v.get("sw_ess_frac", float("nan"))          # label ESS (sign cancellation)
            essf        = v.get("init_sw_ess_frac", essf_label)      # sampling weight ESS (correct sentinel)
            n_overshoot = int(v.get("n_q_overshoot", 0))
            n_nonfinite = int(v.get("n_q_nonfinite", 0))

            # ------ ESS-triggered relabelling (opt-in) ------
            # Use the initial SAMPLING WEIGHT ESS (init_sw_ess_frac) to judge
            # whether the Monte-Carlo estimator has collapsed, not the label ESS.
            if (rs_enabled
                and np.isfinite(essf) and essf < rs_threshold
                and (k - _last_resample_step) >= rs_cooldown
                and _total_resamples < rs_max):

                # ARCHITECTURAL CAVEAT for density-difference surrogates.
                # GPDensityDiff is built on the invariant that
                #     y(t) = y0 + Σ (Δt · QCLE corrections),
                # i.e. δ(t) = y(t) - y0 represents the accumulated QCLE
                # corrections only.  ESS resampling assigns
                #     y_new := ρ̂(Z) = y0 + δ_old + (any GP fit residual)
                # which is no longer "QCLE corrections accumulated since t=0"
                # in any rigorous sense — it folds in everything that has
                # drifted, including kernel-swap approximation error and
                # lengthscale breathing.  The diff-GP architecture's
                # interpretation of δ is broken from this step forward.
                #
                # Two reasonable responses, neither perfect:
                #   (a) refuse to resample and let the run proceed with the
                #       degraded ESS (current behavior, with a loud warning);
                #   (b) reset the baseline by refitting gp0 on (Z, y_new),
                #       losing the t=0 anchor but restoring the invariant.
                # We pick (a): warn and skip.  If you need (b), turn
                # enable_ess_resampling off and run with --density_mode full.
                gp_is_diff = (hasattr(self.state.gp, "gp0")
                              and hasattr(self.state.gp, "gp_delta"))
                if gp_is_diff:
                    if cfg.verbose:
                        print("#" * 72)
                        print(f"[{self.scheme.name}] step {k:4d}: ESS/N={essf:.2e} would trigger "
                              f"resample, but the surrogate is GPDensityDiff —")
                        print(f"  resampling would break the δ = (accumulated QCLE corrections) "
                              f"invariant and silently corrupt the baseline.")
                        print(f"  SKIPPING the resample.  Run with --density_mode full if ESS "
                              f"resampling is required.")
                        print("#" * 72)
                    # Mark the would-be event but do not apply it.
                    diag.values["ess_resample_blocked_diffgp"] = 1.0
                    # Bump the cooldown so we don't spam this every step.
                    _last_resample_step = k
                else:
                    # Relabel using the current GP surrogate evaluated on the
                    # current support cloud: y_new = ρ̂(Z).  The surrogate is
                    # the self-consistent "truth" we trust more than the
                    # drifted signed weights.  Then refit so α matches the
                    # new labels.  The refit honors the hyperparameter policy
                    # selected by the user.
                    y_new_relabel = np.asarray(self.state.gp.predict(self.state.Z),
                                               dtype=np.float64)
                    # Guardrail: if predict produced any non-finite values
                    # (shouldn't happen, but check) skip the resample.
                    n_bad = int(np.sum(~np.isfinite(y_new_relabel)))
                    if n_bad == 0:
                        self.state.y = y_new_relabel
                        self.state.gp.refit(
                            Z_train=self.state.Z, y_train=self.state.y,
                            moment_targets=self.state.moment_targets,
                        )
                        _last_resample_step = k
                        _total_resamples += 1
                        # Mark this step as a resample event so it can be plotted
                        # or cross-referenced later.  Write into the just-recorded
                        # diag in-place.
                        diag.values["ess_resampled"] = 1.0
                        if cfg.verbose:
                            print(f"[{self.scheme.name}] step {k:4d}: ESS={essf:.2e} → "
                                  f"RELABEL via ρ̂(Z); total resamples = {_total_resamples}")
                    else:
                        if cfg.verbose:
                            print(f"[{self.scheme.name}] step {k:4d}: wanted to resample but "
                                  f"GP predict returned {n_bad} non-finite values; skipped.")

            # 1. ESS collapse (sampling-side pathology — no GP fix applies)
            if (np.isfinite(essf) and essf < ESS_WARNING_THRESHOLD
                and (k - _ess_warned_at) >= ESS_WARNING_INTERVAL):
                if not banner_ess:
                    print("#" * 72)
                    print(f"[{self.scheme.name}] WARNING: signed-weight ESS collapse detected.")
                    print(f"  Monte Carlo estimates under the signed target are now dominated")
                    n_samples_current = int(self.state.Z.shape[0])
                    print(f"  by ~{max(0, int(essf * n_samples_current))} effective samples of {n_samples_current}.")
                    print(f"  Normalization and population diagnostics may be unreliable.")
                    if rs_enabled:
                        print(f"  ESS resampling is ENABLED (total resamples so far: {_total_resamples}).")
                    else:
                        print(f"  Consider: larger n_train, --enable_ess_resampling, or")
                        print(f"  enabling the normalization KKT constraint during propagation.")
                    print("#" * 72)
                    banner_ess = True
                else:
                    print(f"[{self.scheme.name}] step {k:4d}: ESS/N = {essf:.2e} (still below {ESS_WARNING_THRESHOLD:.0%}).")
                _ess_warned_at = k

            # 2. Q correction overshooting |y| at many points (midpoint-only
            # pathology; means single-step corrections exceed the label they
            # modify, which nearly always indicates numerical trouble).
            if (self.scheme.name == "midpoint" and n_overshoot > 0
                and (k - _q_warned_at) >= Q_OVERSHOOT_WARNING_INTERVAL):
                frac = n_overshoot / max(1, self.state.Z.shape[0])
                if not banner_q and frac > 0.05:
                    print("#" * 72)
                    print(f"[{self.scheme.name}] WARNING: Q overshoot at step {k}.")
                    print(f"  |dt·Q| exceeded |y| at {n_overshoot} of {self.state.Z.shape[0]} points ({frac:.1%}).")
                    print(f"  Single-step corrections are flipping label signs somewhere.")
                    print(f"  Consider: --apply_q_clip with --q_clip_frac 0.5, or reducing dt.")
                    print("#" * 72)
                    banner_q = True
                elif frac > 0.05:
                    print(f"[{self.scheme.name}] step {k:4d}: Q overshoot at {frac:.1%} of points.")
                if frac > 0.05:
                    _q_warned_at = k

            if n_nonfinite > 0:
                print(f"[{self.scheme.name}] step {k:4d}: {n_nonfinite} non-finite Q entries trapped and zeroed.")
            if v.get("adapt_refit_failed", 0.0) > 0.5:
                _reason = str(getattr(
                    self.state.gp, "last_breathing_failure_reason",
                    "invalid adaptive GP candidate"))
                print(f"[{self.scheme.name}] step {k:4d}: adaptive GP refit "
                      f"rejected safely (code "
                      f"{int(v.get('adapt_refit_failure_code', 0))}); "
                      f"restored last finite state. {_reason}")
            # ----------------------------------------------------------------

            if cfg.verbose:
                print(f"[{self.scheme.name}] step {k:4d}  "
                      f"t={self.state.t:7.3f}  "
                      f"norm={v['km_normalization']:+.3e}  "
                      f"E_phys={v.get('lw_energy', v.get('spe_E_density', float('nan'))):+.6e}  "
                      f"<H>={v.get('lw_energy', v.get('km_energy', float('nan'))):+.6e}  "
                      f"P0={v.get('lw_P0', v.get('cloud_weighted_P0', float('nan'))):+.4f}  P1={v.get('lw_P1', v.get('cloud_weighted_P1', float('nan'))):+.4f}  "
                      f"Psum={v.get('lw_P_sum', v.get('cloud_weighted_trace', float('nan'))):+.4f}  "
                      f"essf_c={v.get('ce_ess_c_frac', float('nan')):.3e}  "
                      f"|c|max={v.get('ce_max_abs_c_frac', float('nan')):.3e}  "
                      f"essf_w={v.get('init_sw_ess_frac', float('nan')):.3e}  "
                      f"essf_y={v.get('sw_ess_frac', float('nan')):.3e}  "
                      f"chi={v.get('sw_cancel_ratio', float('nan')):.3e}  "
                      f"q_rms={v['cs_q_rms']:.3e}  "
                      f"fit_rms={diag.fit_rms_on_support:.3e}  fit_mae={v.get('gp_fit_mae', float('nan')):.3e}  "
                      f"fit_r2={v.get('gp_fit_r2', float('nan')):+.3e}  "
                      f"liou_rms={v.get('gp_liouville_rms', float('nan')):.3e}  "
                      f"loss={v.get('gp_opt_total_loss', float('nan')):.3e}  reg={v.get('gp_opt_reg_loss', float('nan')):.3e}  "
                      f"Rμ={v.get('cloud_mean_R', float('nan')):+.3e}  Pμ={v.get('cloud_mean_P', float('nan')):+.3e}  "
                      f"VR={v.get('cloud_var_R', float('nan')):.3e}  VP={v.get('cloud_var_P', float('nan')):.3e}  "
                      f"σn={diag.sigma_n:.3e}  σn(norm)={v.get('sigma_n_normalized', float('nan')):.3e}  "
                      f"snr={v.get('sigma_f_over_sigma_n', float('nan')):.3e}"
                      # Faithfulness metrics — appended at end of line so existing parsers
                      # don't break.  Test 1 (alpha sign) is only meaningful when the
                      # sampler declares positive labels (apply_kkt=False → focused).
                      + (f"  α-neg={v.get('alpha_neg_frac', float('nan')):.3f}"
                         f"  α-negL1={v.get('alpha_neg_l1_frac', float('nan')):.3f}"
                         f"  α-min/max={v.get('alpha_min_to_max_ratio', float('nan')):+.3f}"
                         if v.get('alpha_test_active', 0.0) > 0.5 else "")
                      # Test 2 (GP-vs-cloud moments) is always meaningful when both
                      # moments are computed (constraints_enabled=True).
                      + f"  Δnorm={v.get('gp_vs_cloud_norm_rel', float('nan')):.2e}"
                      + f"  ΔE={v.get('gp_vs_cloud_energy_rel', float('nan')):.2e}"
                      # ── Surrogate-faithfulness battery ─────────────────
                      # ESS(α): coverage of GP coefficients (low → faithfulness collapse)
                      # LOO:    leave-one-out RMS / max residual (large → fit failure)
                      # logκ:   lower-bound log10 cond(K) (large → numerical singularity)
                      # 3σ:     count of LOO standardised residuals outside ±3σ
                      + f"  ESS(α)f={v.get('faith_ess_alpha_frac', float('nan')):.2f}"
                      + f"  LOO_rms={v.get('faith_loo_rms', float('nan')):.2e}"
                      + f"  LOO_max={v.get('faith_loo_max', float('nan')):.2e}"
                      + f"  logκ≥{v.get('faith_cond_K_lo_log10', float('nan')):.1f}"
                      + f"  3σ={int(v.get('faith_loo_n_3sig', 0))}"
                      + f"  pred_rms={v.get('faith_predict_rms', float('nan')):.2e}"
                      + (f"  adapt=REJECT({int(v.get('adapt_refit_failure_code', 0))})"
                         if v.get("adapt_refit_failed", 0.0) > 0.5 else ""))
                if cfg.detailed_verbose:
                    _print_detailed_step(self.scheme.name, self.state.step_index, self.state.t, diag)

        if cfg.verbose:
            total = time.time() - t0
            print(f"[{self.scheme.name}] done in {total:.1f}s")
            self._print_faithfulness_report()

    def _print_faithfulness_report(self) -> None:
        """
        Run-level summary of surrogate-faithfulness diagnostics.

        Aggregates the per-step `faith_*` keys collected by `_measure`
        and prints a one-block report card.  Trips a warning flag if
        any of the following criteria are exceeded:

          • ESS(α)/N drops below 0.1 at any step  (coverage collapse)
          • LOO_max exceeds 10·median LOO AND any |z|>3σ  (local fit failure)
          • log10 cond(K) exceeds 12              (numerical singularity)
          • predict_rms_rel exceeds 0.5           (fit doesn't track cloud)

        The report goes to stdout so it appears in your usual run log;
        the raw per-step time series are saved to the NPZ as faith_*
        columns for offline plotting.
        """
        arr = self.collector.as_arrays()
        if not arr:
            return
        def _safe(key: str) -> np.ndarray:
            v = arr.get(key)
            if v is None: return np.array([], dtype=np.float64)
            v = np.asarray(v, dtype=np.float64).ravel()
            return v[np.isfinite(v)]

        ess_alpha = _safe("faith_ess_alpha_frac")
        ess_wy    = _safe("faith_ess_wy_frac")
        loo_rms   = _safe("faith_loo_rms")
        loo_max   = _safe("faith_loo_max")
        loo_n3sig = _safe("faith_loo_n_3sig")
        cond_lo   = _safe("faith_cond_K_lo_log10")
        pred_rms  = _safe("faith_predict_rms")
        pred_rel  = _safe("faith_predict_rms_rel")

        def _line(label: str, vals: np.ndarray, fmt: str = "{:+.3e}") -> str:
            if vals.size == 0:
                return f"  {label:24s} (no data)"
            return (f"  {label:24s} min={fmt.format(float(vals.min()))}  "
                    f"med={fmt.format(float(np.median(vals)))}  "
                    f"max={fmt.format(float(vals.max()))}")

        print(f"[{self.scheme.name}] ── Faithfulness report ──────────────────────────────")
        print(_line("ESS(α)/N",           ess_alpha, "{:.3f}"))
        print(_line("ESS(ω·y)/N",         ess_wy,    "{:.3f}"))
        print(_line("LOO residual RMS",   loo_rms))
        print(_line("LOO residual max",   loo_max))
        print(_line("LOO #|z|>3σ",        loo_n3sig, "{:.0f}"))
        print(_line("log10 cond(K) (LB)", cond_lo,   "{:+.2f}"))
        print(_line("predict RMS",        pred_rms))
        print(_line("predict RMS / |y|",  pred_rel,  "{:.3e}"))

        # Warning flags (criteria from the diagnostic block docstring).
        warns: List[str] = []
        if ess_alpha.size and float(ess_alpha.min()) < 0.10:
            warns.append(f"ESS(α)/N dropped to {float(ess_alpha.min()):.3f}"
                         f" (< 0.10 → coverage collapse)")
        # LOO warning (calibrated 2026-07): the raw ratio max(LOO_max) /
        # median(LOO_rms) compares a run-wide extreme (max over ~steps × N
        # deletion tests) against a typical per-step RMS.  Under a PERFECT
        # Gaussian-residual null this ratio already sits at ≈ √(2 ln M) ≈ 5
        # for M ~ 5·10⁵ draws, so the fixed 10× threshold trips on ~2×-null
        # tails that the GP's own predictive variance fully covers.  Require
        # corroboration by the calibrated statistic — the z-scored LOO count
        # |z| > 3σ — before calling it a local fit failure.  A large raw
        # ratio with zero 3σ outliers is reported as a note, not a warning.
        if loo_max.size and loo_rms.size and \
           float(loo_max.max()) > 10.0 * float(np.median(loo_rms)):
            n3sig_total = int(loo_n3sig.sum()) if loo_n3sig.size else 0
            if n3sig_total > 0:
                warns.append(f"LOO max = {float(loo_max.max()):.2e} exceeded "
                             f"10·median(LOO_rms) = {10*float(np.median(loo_rms)):.2e}"
                             f" with {n3sig_total} point(s) beyond 3σ"
                             f" → local fit failure")
            else:
                print(f"[{self.scheme.name}]   note: LOO max/median(rms) = "
                      f"{float(loo_max.max())/float(np.median(loo_rms)):.1f} "
                      f"(> 10), but 0 points beyond 3σ of the GP predictive "
                      f"uncertainty — extreme-value tail, not a fit failure.")
        if cond_lo.size and float(cond_lo.max()) > 12.0:
            warns.append(f"log10 cond(K) peaked at {float(cond_lo.max()):.1f}"
                         f" (> 12 → numerical singularity)")
        if pred_rel.size and float(pred_rel.max()) > 0.5:
            warns.append(f"predict_rms / |y| peaked at {float(pred_rel.max()):.2f}"
                         f" (> 0.5 → surrogate isn't tracking the cloud)")
        if warns:
            print(f"[{self.scheme.name}]   *** FAITHFULNESS WARNINGS ***")
            for w in warns:
                print(f"[{self.scheme.name}]     - {w}")
        else:
            print(f"[{self.scheme.name}]   no faithfulness warnings — "
                  f"surrogate appears to track the cloud throughout the run.")
        print(f"[{self.scheme.name}] ─────────────────────────────────────────────────────")

    def save(self) -> str:
        path = f"{self.config.output_dir}/{self.config.run_name}"
        return self.collector.save(path)


# =============================================================================
# Detailed connector diagnostics / tests
# =============================================================================

def _print_detailed_step(prefix: str,
                         step_index: int,
                         t: float,
                         diag: StepDiagnostics) -> None:
    v = diag.values
    def _fmt_vec(prefix_key: str) -> str:
        return "[" + ", ".join(f"{lbl}={v.get(f'{prefix_key}_{lbl}', float('nan')):+.3e}" for lbl in _DIM_LABELS) + "]"
    print(f"[{prefix}] step={step_index:4d}  t={t:9.6f}")
    print(f"  GP moments:   <1>={v.get('km_normalization', float('nan')):+.12e}  "
          f"<tr>={v.get('km_trace', float('nan')):+.12e}  "
          f"<H>={v.get('lw_energy', v.get('km_energy', float('nan'))):+.12e}")
    print(f"  GP fit:       free_rms={v.get('gp_free_fit_rms', float('nan')):.6e}  "
          f"free_mae={v.get('gp_free_fit_mae', float('nan')):.6e}  free_r2={v.get('gp_free_fit_r2', float('nan')):+.6e}  "
          f"fit_rms={diag.fit_rms_on_support:.6e}  fit_mae={v.get('gp_fit_mae', float('nan')):.6e}  fit_r2={v.get('gp_fit_r2', float('nan')):+.6e}")
    print(f"  GP delta:     Δrms={v.get('gp_constraint_delta_rmse', float('nan')):+.6e}  "
          f"Δmae={v.get('gp_constraint_delta_mae', float('nan')):+.6e}  Δr2={v.get('gp_constraint_delta_r2', float('nan')):+.6e}")
    print(f"  GP hypers:    sigma_f(norm)={v.get('sigma_f_normalized', float('nan')):.6e}  "
          f"sigma_n(norm)={v.get('sigma_n_normalized', float('nan')):.6e}  "
          f"sigma_n(raw)={diag.sigma_n:.6e}  SNR_sf/sn={v.get('sigma_f_over_sigma_n', float('nan')):.6e}  "
          f"SNR_pred/sn={v.get('pred_rms_over_sigma_n', float('nan')):.6e}  ell={diag.lengthscales}")
    print(f"  GP opt:       loss={v.get('gp_opt_total_loss', float('nan')):.6e}  nll={v.get('gp_opt_nll_loss', float('nan')):.6e}  "
          f"reg={v.get('gp_opt_reg_loss', float('nan')):.6e}  train_mae={v.get('gp_opt_train_mae', float('nan')):.6e}  "
          f"val_mae={v.get('gp_opt_val_mae', float('nan')):.6e}  best_step={int(v.get('gp_opt_best_step', -1.0))}  "
          f"early_stop={bool(int(v.get('gp_opt_early_stopped', 0.0)))}")
    print(f"  Alpha:        mean={v.get('alpha_mean', float('nan')):+.6e}  std={v.get('alpha_std', float('nan')):.6e}  "
          f"l1={v.get('alpha_l1', float('nan')):.6e}  l2={v.get('alpha_l2', float('nan')):.6e}  linf={v.get('alpha_linf', float('nan')):.6e}")
    print(f"  Delta alpha:  mean={v.get('delta_alpha_mean', float('nan')):+.6e}  std={v.get('delta_alpha_std', float('nan')):.6e}  "
          f"l2={v.get('delta_alpha_l2', float('nan')):.6e}  linf={v.get('delta_alpha_linf', float('nan')):.6e}  "
          f"rel_l2={v.get('delta_alpha_rel_l2', float('nan')):.6e}")
    print(f"  Observables:  P0={v.get('lw_P0', v.get('cloud_weighted_P0', float('nan'))):+.12e}  P1={v.get('lw_P1', v.get('cloud_weighted_P1', float('nan'))):+.12e}  "
          # Use km_trace (KKT-constrained analytic integral, always = 1.0 when
          # constraints are active) rather than dp_trace (GP quadratic-moment
          # path that relies on _quadratic_mapping_moments and is corrupted by
          # near-singular K when the P-variance clamps to a negative value).
          f"trace={v.get('km_trace', v.get('dp_trace', float('nan'))):+.12e}  |rho01|={v.get('dc_coh_abs', float('nan')):+.12e}")
    if 'cloud_weighted_energy' in v:
        print(f"  Cloud diag:   Ew={v['cloud_weighted_energy']:+.12e}  ΔEw={v.get('cloud_weighted_energy_err', float('nan')):+.6e}  "
              f"Trw={v.get('cloud_weighted_trace', float('nan')):+.12e}")
    print(f"  Fit weights:  sum={v.get('sw_weight_sum', float('nan')):+.12e}  abs_sum={v.get('sw_abs_weight_sum', float('nan')):+.12e}  "
          f"ESS={v.get('sw_ess', float('nan')):.6e}  ESS/N={v.get('sw_ess_frac', float('nan')):.6e}  "
          f"chi={v.get('sw_cancel_ratio', float('nan')):.6e}  neg_frac={v.get('sw_neg_frac', float('nan')):.6e}")
    print(f"  Init weights: ESS={v.get('init_sw_ess', float('nan')):.6e}  ESS/N={v.get('init_sw_ess_frac', float('nan')):.6e}  "
          f"chi={v.get('init_sw_cancel_ratio', float('nan')):.6e}  neg_frac={v.get('init_sw_neg_frac', float('nan')):.6e}")
    print(f"  Cloud center: {_fmt_vec('cloud_mean')}")
    print(f"  Cloud var:    {_fmt_vec('cloud_var')}")
    print(f"  GP center:    {_fmt_vec('gp_mean')}")
    print(f"  GP var:       {_fmt_vec('gp_var')}")
    # ── Surrogate faithfulness battery ─────────────────────────────────
    # ESS:           coverage of the cloud and α (low → faithfulness collapse)
    # LOO:           Cholesky-shortcut leave-one-out RMS / max residual
    # cond(K)_lo:    log10 lower bound on K's condition number
    # predict_rms:   surrogate-vs-cloud RMS on the support cloud itself
    print(f"  Faithfulness: ESS(α)={v.get('faith_ess_alpha_frac', float('nan')):.3f}  "
          f"ESS(ωy)={v.get('faith_ess_wy_frac', float('nan')):.3f}  "
          f"LOO_rms={v.get('faith_loo_rms', float('nan')):.3e}  "
          f"LOO_max={v.get('faith_loo_max', float('nan')):.3e}  "
          f"log10cond(K)≥{v.get('faith_cond_K_lo_log10', float('nan')):.2f}  "
          f"pred_rms={v.get('faith_predict_rms', float('nan')):.3e}  "
          f"n_3σ={int(v.get('faith_loo_n_3sig', 0))}")
    print(f"  Trust region: lower={_fmt_vec('trust_lo')}  upper={_fmt_vec('trust_hi')}")
    print(f"  Correction:   raw_q_rms={v.get('cs_q_rms', float('nan')):.6e}  raw_q_max={v.get('cs_q_max', float('nan')):.6e}  "
          f"applied_q_rms={v.get('applied_cs_q_rms', float('nan')):.6e}  n_clipped={int(v.get('n_q_clipped', 0.0))}")


def run_connector_diagnostic(
    scheme: str,
    n_train: int = 400,
    n_steps: int = 20,
    dt: float = 0.2,
    seed: int = 0,
    snapshot_every: int = 0,
    include_abs_integral: bool = False,
    apply_q_clip: bool = False,
    verbose: bool = True,
    detailed_verbose: bool = True,
    classical_params: Optional[GaussianWavePacketParams] = None,
    mapping_params: Optional[MappingInitParams] = None,
    gp_config: Optional[GPDensityConfig] = None,
    dynamics: Optional[PBMEMIntDynamics] = None,
) -> Dict[str, float]:
    """
    End-to-end diagnostic on the module that connects sampling, GP fit/refit,
    PBME transport, midpoint QCLE correction, and observable extraction.

    Returns a summary dict with max drifts/errors.  By default q-clipping is
    disabled so the raw midpoint operator can be assessed scientifically.
    """
    if classical_params is None:
        classical_params = GaussianWavePacketParams(R0=[-8.0], P0=[30.0], sigma_R=[1.0], hbar=1.0)
    if mapping_params is None:
        mapping_params = MappingInitParams(nstates=2, init_state=0, hbar=1.0, gamma=0.5)
    if gp_config is None:
        gp_config = GPDensityConfig(n_opt_steps=120, fix_sigma_n=False, init_log_sigma_n=-2.5,
                                    feature_zscore=True, recompute_feature_zscore=False,
                                    reinit_lengthscales=False, interpolate_targets=False)
    dyn = dynamics if dynamics is not None else PBMEMIntDynamics()

    state = Simulation.build_initial_state(
        n_train=n_train,
        classical_params=classical_params,
        mapping_params=mapping_params,
        gp_config=gp_config,
        seed=seed,
        dynamics=dyn,
    )
    cfg = DynamicsConfig(
        scheme=scheme, dt=dt, n_steps=n_steps, snapshot_every=snapshot_every,
        include_abs_integral=include_abs_integral, verbose=False,
        detailed_verbose=detailed_verbose, apply_q_clip=apply_q_clip,
        output_dir="results", run_name=f"diagnostic_{scheme}",
    )
    sim = Simulation(cfg, state, dynamics=dyn)

    # step 0: measure without stepping — pass a minimal but complete scheme_diag
    # that mirrors what the production run loop sees at k=0 (before any step).
    # All correction keys default to zero/nan via _measure's .get() calls, which
    # is correct — there is no Q yet.  Using a real dict instead of a hand-rolled
    # partial dict ensures any new keys added to _measure don't silently miss here.
    step0_diag = {
        "Q": None, "Q_applied": None, "Y": None,
        "n_q_clipped": 0, "n_q_nonfinite": 0, "n_q_overshoot": 0,
        "apply_q_clip": apply_q_clip,
        "mint_wall": 0.0, "operator_wall": 0.0, "refit_wall": 0.0,
    }
    diag0 = sim._measure(step0_diag, 0.0)
    if verbose:
        _print_detailed_step(f"connector/{scheme}", 0, state.t, diag0)

    max_weighted_dE = abs(diag0.values.get("cloud_weighted_energy_err", 0.0))
    max_gp_norm_err = abs(diag0.values.get("km_normalization", 1.0) - 1.0)
    max_gp_trace_err = abs(diag0.values.get("km_trace", 1.0) - 1.0)
    max_gp_energy_err = abs(diag0.values.get("km_energy", state.moment_targets["energy"]) - state.moment_targets["energy"])
    max_q_raw = 0.0
    max_q_applied = 0.0
    total_clipped = 0

    for k in range(1, n_steps + 1):
        ts = time.time()
        scheme_diag = sim.scheme.step(sim.state)
        wall = time.time() - ts
        diag = sim._measure(scheme_diag, wall)

        max_weighted_dE = max(max_weighted_dE, abs(diag.values.get("cloud_weighted_energy_err", 0.0)))
        max_gp_norm_err = max(max_gp_norm_err, abs(diag.values.get("km_normalization", 1.0) - 1.0))
        max_gp_trace_err = max(max_gp_trace_err, abs(diag.values.get("km_trace", 1.0) - 1.0))
        max_gp_energy_err = max(max_gp_energy_err, abs(diag.values.get("km_energy", sim.state.moment_targets["energy"]) - sim.state.moment_targets["energy"]))
        max_q_raw = max(max_q_raw, float(diag.values.get("cs_q_max", 0.0)))
        max_q_applied = max(max_q_applied, float(diag.values.get("applied_cs_q_max", 0.0)))
        total_clipped += int(diag.values.get("n_q_clipped", 0.0))

        if verbose:
            _print_detailed_step(f"connector/{scheme}", k, sim.state.t, diag)

    summary = {
        "target_weighted_energy": float(sim.state.moment_targets["energy"]),
        "max_abs_cloud_weighted_dE": float(max_weighted_dE),
        "max_abs_gp_norm_err": float(max_gp_norm_err),
        "max_abs_gp_trace_err": float(max_gp_trace_err),
        "max_abs_gp_energy_err": float(max_gp_energy_err),
        "max_abs_raw_q": float(max_q_raw),
        "max_abs_applied_q": float(max_q_applied),
        "total_q_clipped": int(total_clipped),
    }
    print("\nSummary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:28s} = {v:.12e}")
        else:
            print(f"  {k:28s} = {v}")
    return summary


def test_pbme_sample_fit_propagate(**kwargs) -> Dict[str, float]:
    summary = run_connector_diagnostic("pbme", apply_q_clip=False, **kwargs)
    if not np.isfinite(summary["max_abs_cloud_weighted_dE"]):
        raise AssertionError("PBME diagnostic produced non-finite weighted cloud energy drift.")
    if summary["max_abs_gp_norm_err"] > 1.0e-10:
        raise AssertionError("PBME GP normalization deviated from target beyond tolerance.")
    if summary["max_abs_gp_trace_err"] > 1.0e-10:
        raise AssertionError("PBME GP trace deviated from target beyond tolerance.")
    if summary["max_abs_gp_energy_err"] > 1.0e-10:
        raise AssertionError("PBME GP energy deviated from target beyond tolerance.")
    return summary


def test_midpoint_sample_fit_propagate(**kwargs) -> Dict[str, float]:
    # No clipping by default: we want to see the raw pulled-back midpoint QCLE dynamics.
    return run_connector_diagnostic("midpoint", apply_q_clip=False, **kwargs)


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    print("\n[dynamics/test_pbme_sample_fit_propagate]")
    test_pbme_sample_fit_propagate(n_train=250, n_steps=8, dt=0.2)
    print("\n[dynamics/test_midpoint_sample_fit_propagate]")
    test_midpoint_sample_fit_propagate(n_train=250, n_steps=8, dt=0.2)
