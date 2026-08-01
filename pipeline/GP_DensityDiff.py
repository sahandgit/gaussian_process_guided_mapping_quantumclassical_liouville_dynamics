from __future__ import annotations

"""
GP_DensityDiff.py
=================

Density-difference surrogate.

Motivation
----------
The single-GP surrogate in `GP_Density.py` has an unpleasant property: when
the carried-label vector y drifts (as in the midpoint QCLE scheme, where
y_new = y_old - dt·Q has no Liouville invariant backing it), the KKT-
constrained alpha solve is forced to reconcile a drifted label vector
against unit-valued moment constraints. On Tully dual-crossing at t >~ 500,
that produces oscillatory alpha of norm ~30, unphysical populations, and
visible red/blue ringing in the 2D marginals, even while the training-point
fit R^2 is still 1.00000.

The density-difference representation sidesteps this.  At time t we write

    rho_hat(z, t) = rho_hat_0(Phi_{-t}^0(z)) + delta_hat(z, t)

where

    * rho_hat_0   is the t=0 baseline GP, FROZEN after the initial fit.
    * Phi_{-t}^0  is the backward classical PBME flow (pure MInt; this
                  is the flow that transports support points forward in
                  time).
    * delta_hat   is a correction GP re-fit each step to the targets
                      delta_i = y_i(t) - y_i(0)
                  which are identically zero at t=0 and grow only as
                  the midpoint QCLE correction accumulates.

Key property used everywhere
----------------------------
By construction, support points transport along the backward flow:
    Phi_{-t}^0 (Z_i(t)) = Z_i(0).
So the baseline contribution AT SUPPORT POINTS is just
    rho_hat_0(Z_i(0)) = y_i(0),
and therefore the training targets for delta_hat are
    delta_i = y_i(t) - y_i(0)
without any backward-flow evaluation.  This is the "zero difference at
time zero" that makes the decomposition trivial at training time.

For OFF-support queries (which Q's third derivatives need at the pulled-
back midpoint Y(Z)), the baseline contribution is
    rho_hat_0(Phi_{-t}^0(z))
and its derivatives follow from the chain rule through the backward-flow
Jacobian, Hessian, and third-derivative tensors.

Linearity of the moment integrators
-----------------------------------
The ARD-RBF moment integrals ∫ψ_m(z) k(z, Z_j) dz are LINEAR in the alpha
vector.  So for any analytic moment ψ_m,

    <ψ_m>_{rho_hat(t)} = [A_m^{(0)} · alpha^{(0)}] + [A_m^{(δ)} · alpha^{(δ)}]

modulo the fact that the baseline A rows use the SUPPORT CENTERS OF THE
BASELINE GP, i.e. Z^{(0)}.  For moments whose integrand is position-
dependent (r², p², energy), we exploit a subtle but critical property:

    ∫ f(z) rho_hat_0(Phi_{-t}^0(z)) dz = ∫ f(Phi_t^0(w)) rho_hat_0(w) dw

by change of variables, with |det J(Phi_t^0)| = 1 since the classical
MInt flow is symplectic.  So the baseline moment at time t is the FROZEN
baseline density's moment against the image of f under the FORWARD flow.
For r² and p², that image is not simple -- but for normalization, trace,
and mapping-radius-squared it IS simple:

    * normalization:   ∫1 · rho_hat_0 dz  = unchanged                  (1)
    * mapping r²+p²:   r²+p² is exactly conserved by MInt per-particle
                       → the moment is unchanged                        (2)
    * trace sum rule:  c_00 + c_11 = (r²+p²)/ℏ - 1 is conserved        (3)
    * energy (exact H): energy is exactly conserved by PBME per-particle → unchanged (4)

This means the baseline contributions to the four KKT-relevant moments
(normalization, trace, energy) AND to ⟨r²+p²⟩ are COMPUTABLE ONCE AT t=0
AND FROZEN FOREVER.  For r²_alpha and p²_alpha individually -- needed
for diabatic populations -- the MInt mapping rotation does mix them, so
the baseline contribution has to be recomputed each step via moment
integrals against the *initial* GP centers but with an updated center
list reflecting the forward-evolved Z^{(0)}.  (We handle that correctly
below.)

Interface
---------
GPDensityDiff exposes the subset of the GPDensity API that downstream
code actually uses:

    fit(Z_train, y_train, moment_targets, ...)
    refit(Z_train, y_train, moment_targets, ...)
    predict(Z)
    compute_moment_values()
    freeze_hypers()
    report()
    sigma_f, sigma_n, lengthscales  (from the δ-GP; the baseline values
                                     are captured in .gp0_snapshot)

It also exposes:

    .gp0         — the frozen baseline GP
    .gp_delta    — the active correction GP
    .y0          — the frozen initial labels
    .Z0          — the frozen initial support centers

Downstream hooks
----------------
GPDerivatives.py is extended with a `DensityLike` protocol so
rho_derivative_bundle can accept either a GPDensity or a GPDensityDiff.
The diff path sums the two kernel evaluations and chain-rules the
baseline derivatives through the backward flow Jacobian/Hessian/third
tensors built by MonodromyTools.

Observable layer
----------------
Observables.py is extended to dispatch through a new helper
_quadratic_mapping_moments_split(gp_like) that, for a diff-GP, builds
two A rows (one on Z_delta, one on Z_0_forwarded) and sums.  The sum is
always taken BEFORE dividing by the normalization -- so populations use
    (A_r²·α) / (A_norm·α)  =  (A_r²^(0)·α^(0) + A_r²^(δ)·α^(δ)) / (...)
"""

from dataclasses import dataclass, replace as _dc_replace
from typing import Any, Dict, Optional, Tuple

import copy
import numpy as np
from numpy.typing import ArrayLike, NDArray

import torch

from GP_Density import (
    GPDensity, GPDensityConfig, _as_numpy, _as_tensor, _ard_gram,
    MomentName, D, _DEFAULT_DTYPE,
)

FloatArray = NDArray[np.float64]


# =============================================================================
# Config
# =============================================================================

@dataclass
class GPDensityDiffConfig:
    """
    Configuration for the density-difference surrogate.

    The baseline and correction GPs can in principle use different
    GPDensityConfig settings (the baseline is typically fit with tighter
    hyperparameter optimization since it is frozen forever; the correction
    is refit every step and uses the breathing policy).  By default both
    inherit the same settings.

    Attributes
    ----------
    base_config : GPDensityConfig
        Config for the baseline GP rho_hat_0.  Typical setting:
        refit_hyper_policy='frozen' (irrelevant since we freeze_hypers()
        it immediately), n_opt_steps=200+, fix_sigma_n=True.

    delta_config : GPDensityConfig
        Config for the correction GP delta_hat.  Typical setting:
        refit_hyper_policy='breathing' (the correction's lengthscales
        should breathe with the cloud), fix_sigma_n=True with an
        init_log_sigma_n matched to the scale of expected QCLE
        corrections.

    enforce_zero_moment_targets : bool, default True
        When True, the correction GP is refit with KKT constraints that
        are identically zero (normalization, trace, energy contributions
        of delta relative to baseline are zero at t=0 and should stay
        small).  When False, the correction GP uses exact residual
        targets computed from the baseline's frozen moments.

    freeze_baseline : bool, default True
        If True, call freeze_hypers() on the baseline GP after its
        initial fit.  Since the baseline is never refit, this is purely
        belt-and-suspenders but makes the API explicit.
    """
    base_config: GPDensityConfig = None
    delta_config: GPDensityConfig = None
    enforce_zero_moment_targets: bool = True
    freeze_baseline: bool = True

    def __post_init__(self):
        if self.base_config is None:
            self.base_config = GPDensityConfig()
        if self.delta_config is None:
            self.delta_config = GPDensityConfig()


# =============================================================================
# The density-difference surrogate
# =============================================================================

class GPDensityDiff:
    """
    Density-difference surrogate.

    rho_hat(z, t) = rho_hat_0(Phi_{-t}^0(z))  +  delta_hat(z, t)

    Carries two GPDensity instances (baseline + correction) and a frozen
    copy of the initial labels y0 and support centers Z0.  The backward
    flow is never explicitly needed at support points by construction.

    Usage
    -----
    Initial fit (called by Simulation.build_initial_state):

        diff = GPDensityDiff(cfg, dynamics=dynamics)
        diff.fit(Z_train=Z, y_train=y, moment_targets={...})

    Per-step refit (called by MidpointScheme / PBMEScheme .step()):

        diff.refit(Z_train=Z_new, y_train=y_new, moment_targets={...})

    Prediction at arbitrary query points z*:

        rho_vals = diff.predict(z_star)      # returns rho_hat_0 + delta_hat
    """

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    def __init__(self, config: GPDensityDiffConfig, dynamics) -> None:
        self.config = config
        self.dynamics = dynamics
        # Give gp0 and gp_delta SEPARATE config objects so that the
        # `apply_constraints is not None` branch in GPDensity.fit/refit
        # (which transiently mutates `self.config.constraints_enabled`)
        # cannot leak between the two surrogates.  When the same config
        # object was passed to both, this mutation pattern was safe in
        # single-threaded sequential code (try/finally restored the value
        # before each call returned) but fragile under any future async
        # path or shared-state inspection.  `dataclasses.replace` makes a
        # shallow copy with all fields preserved — exactly what we need.
        base_cfg  = _dc_replace(config.base_config)
        delta_cfg = _dc_replace(config.delta_config)
        self.gp0      = GPDensity(base_cfg,  dynamics=dynamics)
        self.gp_delta = GPDensity(delta_cfg, dynamics=dynamics)

        # Frozen initial state (populated by .fit)
        self.Z0:  Optional[FloatArray] = None
        self.y0:  Optional[FloatArray] = None

        # Baseline moment values at t=0 (frozen forever)
        self._baseline_moment_values_0: Dict[str, float] = {}

        # Whether fit() has been called
        self._initial_fit_done: bool = False

    # -------------------------------------------------------------------------
    # GPDensity-compatible surface attributes
    #
    # Downstream code reads .sigma_f, .sigma_n, .lengthscales, ._alpha, etc.
    # For diagnostic/logging purposes we expose the CORRECTION GP's values,
    # since the baseline values are by construction frozen and equal to the
    # snapshot in self.gp0.
    # -------------------------------------------------------------------------
    @property
    def sigma_f(self) -> float:
        return self.gp_delta.sigma_f

    @property
    def sigma_n(self) -> float:
        return self.gp_delta.sigma_n

    @property
    def lengthscales(self) -> FloatArray:
        return self.gp_delta.lengthscales

    @property
    def _alpha(self):
        return self.gp_delta._alpha

    @property
    def _Z_train(self):
        return self.gp_delta._Z_train

    @property
    def _Z_train_norm(self):
        return self.gp_delta._Z_train_norm

    @property
    def _L_Ky(self):
        # Expose correction GP Cholesky factor so the generic faithfulness
        # diagnostics can compute LOO residuals and conditioning for δ.
        return self.gp_delta._L_Ky

    def solve_K(self, B):
        # Forward to correction GP.  In the diff representation LOO diagnostics
        # are for δ̂, while predict-at-support diagnostics below are for the
        # full density ρ̂₀+δ̂.
        return self.gp_delta.solve_K(B)

    @property
    def _y_train(self):
        # Exposes the delta-GP training targets (not y itself).  Downstream
        # code that reads _y_train for fit-RMS and similar diagnostics will
        # see delta, which is the relevant residual for the correction fit.
        return self.gp_delta._y_train

    @property
    def raw_training_centers(self) -> FloatArray:
        return self.gp_delta.raw_training_centers

    @property
    def _hypers_frozen(self) -> bool:
        return bool(getattr(self.gp_delta, "_hypers_frozen", False))

    @property
    def log_sigma_f(self):   return self.gp_delta.log_sigma_f
    @property
    def log_sigma_n(self):   return self.gp_delta.log_sigma_n
    @property
    def log_lengthscales(self): return self.gp_delta.log_lengthscales
    @property
    def _feature_mean(self):    return self.gp_delta._feature_mean
    @property
    def _feature_std(self):     return self.gp_delta._feature_std

    # -------------------------------------------------------------------------
    # Compatibility aliases and sampler label-information contract
    # -------------------------------------------------------------------------
    @property
    def base_gp(self):
        """Alias used by older Dynamics.py fallback code."""
        return self.gp0

    @property
    def delta_gp(self):
        """Alias used by older Dynamics.py fallback code."""
        return self.gp_delta

    @property
    def _pin_apply_kkt(self) -> bool:
        """Expose the correction GP's KKT policy to diagnostics."""
        return bool(getattr(self.gp_delta, "_pin_apply_kkt", True))

    @property
    def _pin_mask(self):
        return getattr(self.gp_delta, "_pin_mask", None)

    def pin_lengthscales(self, label_info) -> None:
        """Apply the sampler's LabelInformation contract to both sub-GPs.

        This is essential for focused sampling.  Focused labels are positive
        proxy labels, not the physical density itself, so LabelInformation
        sets apply_kkt=False and pins unsupported mapping lengthscales.  If the
        wrapper does not forward this contract, both gp0 and gp_delta keep
        KKT constraints enabled and the constrained alpha solve is forced away
        from the support labels.
        """
        self.gp0.pin_lengthscales(label_info)
        self.gp_delta.pin_lengthscales(label_info)

    def unpin_lengthscales(self) -> None:
        self.gp0.unpin_lengthscales()
        self.gp_delta.unpin_lengthscales()

    # -------------------------------------------------------------------------
    # Per-refit diagnostics — forwarded from gp_delta.
    #
    # The CORRECTION GP (gp_delta) is the only one being actively refit
    # during a run; gp0 is frozen at t=0 and its diagnostics never update.
    # When run.py / Dynamics.py read these via getattr(s.gp, …), they
    # need the diff-GP wrapper to expose them — otherwise getattr falls
    # back to NaN and the per-step log shows fit_mae=nan, fit_r2=nan,
    # loss=nan, reg=nan from step 0 (visible in long PBME-diff logs).
    # All names match GPDensity's public attributes one-for-one.
    # -------------------------------------------------------------------------
    @property
    def last_fit_rms(self) -> float:        return self.gp_delta.last_fit_rms
    @property
    def last_fit_mae(self) -> float:        return self.gp_delta.last_fit_mae
    @property
    def last_fit_r2(self) -> float:         return self.gp_delta.last_fit_r2
    @property
    def last_free_fit_rms(self) -> float:   return self.gp_delta.last_free_fit_rms
    @property
    def last_free_fit_mae(self) -> float:   return self.gp_delta.last_free_fit_mae
    @property
    def last_free_fit_r2(self) -> float:    return self.gp_delta.last_free_fit_r2
    @property
    def constraint_delta_rmse(self) -> float: return self.gp_delta.constraint_delta_rmse
    @property
    def constraint_delta_mae(self) -> float:  return self.gp_delta.constraint_delta_mae
    @property
    def constraint_delta_r2(self) -> float:   return self.gp_delta.constraint_delta_r2
    @property
    def last_opt_total_loss(self) -> float: return self.gp_delta.last_opt_total_loss
    @property
    def last_opt_nll_loss(self) -> float:   return self.gp_delta.last_opt_nll_loss
    @property
    def last_opt_reg_loss(self) -> float:   return self.gp_delta.last_opt_reg_loss
    @property
    def last_opt_train_mae(self) -> float:  return self.gp_delta.last_opt_train_mae
    @property
    def last_opt_train_r2(self) -> float:   return self.gp_delta.last_opt_train_r2
    @property
    def last_opt_val_mae(self) -> float:    return self.gp_delta.last_opt_val_mae
    @property
    def last_opt_val_r2(self) -> float:     return self.gp_delta.last_opt_val_r2
    @property
    def last_opt_steps(self) -> int:        return self.gp_delta.last_opt_steps
    @property
    def last_opt_best_step(self) -> int:    return self.gp_delta.last_opt_best_step
    @property
    def last_opt_early_stopped(self) -> bool: return self.gp_delta.last_opt_early_stopped

    # -------------------------------------------------------------------------
    # Fit / refit
    # -------------------------------------------------------------------------
    def fit(self,
            Z_train: ArrayLike,
            y_train: ArrayLike,
            moment_targets: Optional[Dict[MomentName, float]] = None,
            optimize: bool = True,
            verbose: bool = False,
            apply_constraints: Optional[bool] = None) -> None:
        """
        Initial fit at t=0.

          1. Fit baseline GP rho_hat_0 to (Z, y) with the user-provided
             moment targets (same as a vanilla GPDensity would).
          2. Freeze baseline: we never refit it.
          3. Snapshot Z0 = Z, y0 = y.
          4. Initialize the correction GP with zero labels (delta_i ≡ 0)
             and zero-valued KKT targets.  Its alpha solves trivially to 0.
             This costs one GP fit but guarantees the diff machinery is
             exercised at t=0, so any bug in the decomposition shows up
             immediately rather than first at t=1.
        """
        Z_np = np.asarray(Z_train, dtype=np.float64)
        y_np = np.asarray(y_train, dtype=np.float64)
        if Z_np.ndim != 2 or Z_np.shape[1] != D:
            raise ValueError(f"Z_train must have shape (N, {D}); got {Z_np.shape}")
        if y_np.shape != (Z_np.shape[0],):
            raise ValueError(f"y_train must be shape ({Z_np.shape[0]},); got {y_np.shape}")

        # ---- Baseline fit
        self.gp0.fit(
            Z_train=Z_np, y_train=y_np,
            moment_targets=moment_targets,
            optimize=optimize, verbose=verbose,
            apply_constraints=apply_constraints,
        )
        if self.config.freeze_baseline:
            self.gp0.freeze_hypers()

        # ---- Snapshot initial state
        self.Z0 = Z_np.copy()
        self.y0 = y_np.copy()
        self._baseline_moment_values_0 = self.gp0.compute_moment_values()

        # ---- Initial delta fit: delta_i ≡ 0 everywhere.
        # We still want the correction GP to have legitimate hyperparameters
        # (fitted to a near-zero target signal means the optimizer will land
        # on arbitrary lengthscales — so we INHERIT the baseline's fitted
        # hypers and then skip optimization for this first delta fit).
        self._inherit_hypers_from_baseline()
        # Prevent GPDensity.fit() from overwriting the inherited baseline
        # lengthscales with a fresh cloud-std initialization on the identically
        # zero delta labels.  The correction GP's initial kernel must be the
        # same physical kernel as the fitted baseline; otherwise the first
        # nonzero delta refit starts from arbitrary zero-label bandwidths.
        self.gp_delta.config.init_log_lengthscales = _as_numpy(
            self.gp_delta.log_lengthscales
        ).copy()
        delta0 = np.zeros_like(y_np)
        zero_targets = self._compute_zero_delta_targets(moment_targets)
        # apply_constraints is respected; if None, GPDensity applies them iff
        # config.constraints_enabled is True.
        self.gp_delta.fit(
            Z_train=Z_np, y_train=delta0,
            moment_targets=zero_targets,
            optimize=False,                    # don't optimize on zero labels
            verbose=verbose,
            apply_constraints=apply_constraints,
        )

        self._initial_fit_done = True

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
        Per-step refit at t > 0.

        The baseline is NEVER refit.  We compute delta_i = y_i - y0_i at
        the current support centers (which — since support points transport
        along the classical flow under PBME and midpoint-QCLE — are the
        forward-transported initial centers) and fit the correction GP.
        """
        if not self._initial_fit_done:
            raise RuntimeError("GPDensityDiff.fit must be called before refit.")

        Z_np = np.asarray(Z_train, dtype=np.float64)
        y_np = np.asarray(y_train, dtype=np.float64)
        if Z_np.shape != self.Z0.shape:
            raise ValueError(
                f"Z_train shape {Z_np.shape} does not match initial "
                f"support shape {self.Z0.shape}. "
                f"The density-difference representation assumes the support "
                f"cloud size is conserved."
            )

        # Compute the correction targets
        delta = y_np - self.y0

        # Translate the user's moment targets (which are for the TOTAL
        # density rho_hat) into moment targets for delta alone.
        delta_targets = self._split_moment_targets(moment_targets)

        # Dispatch to the correction GP's refit
        self.gp_delta.refit(
            Z_train=Z_np, y_train=delta,
            moment_targets=delta_targets,
            optimize=optimize,
            reinit_lengthscales=reinit_lengthscales,
            verbose=verbose,
            apply_constraints=apply_constraints,
            hyper_policy=hyper_policy,
        )

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------
    def predict(self, Z: ArrayLike) -> FloatArray:
        """
        rho_hat(z, t) = rho_hat_0(Phi_{-t}^0(z)) + delta_hat(z, t).

        AT SUPPORT POINTS (where Z matches the current support cloud
        and so Phi_{-t}^0(Z_i) = Z0_i exactly by construction of MInt-
        propagated centers), we use the EXACT identity

            rho_hat_0(Z0_i) = y0_i

        rather than the kernel-swap approximation.  This matters: in
        long PBME-diff runs the kernel-swap k_b(Z_t, Z_t) @ alpha_b
        drifts away from y0 as soon as Z_t ≠ Z_0, because the kernel
        matrix K(Z_t, Z_t) ≠ K(Z_0, Z_0) and so alpha_b is no longer
        the right linear-system solution at the new centers.  The
        observed symptom in the per-step log is fit_rms creeping up
        every step (e.g. 4.5e-5 → 5.9e-5 over 5 steps with VR/VP
        unchanged but Rμ moving 0.08 a.u.) — i.e. precisely matching
        the cloud translation, not the breathing prior.

        FOR OFF-SUPPORT QUERIES, we keep the kernel-swap approximation
        because there is no free Phi_{-t}^0 evaluation available at
        arbitrary z.  The error is O(|z - Z_t|³) per kernel Taylor
        expansion (== O(dt³) at the QCLE midpoints Y(Z) which are
        within dt/2 of Z), which is consistent with the overall scheme
        truncation order.
        """
        if not self._initial_fit_done:
            raise RuntimeError("GPDensityDiff must be fitted first.")

        Z_np = np.asarray(Z, dtype=np.float64)
        single = (Z_np.ndim == 1)
        if single:
            Z_np = Z_np.reshape(1, D)

        # -- Detect support-point queries ---------------------------------
        # If Z matches the current support cloud (same shape, exact array),
        # we can use the closed-form identity rho_hat_0(Z0_i) = y0_i.
        # We require BOTH same shape AND same number of rows AND same
        # values (within float64 noise) — the strict array-equality check
        # is cheap (O(N·D)) and makes false positives effectively
        # impossible in practice.
        Z_support = _as_numpy(self.gp_delta._Z_train)
        is_support_query = (
            Z_np.shape == Z_support.shape
            and np.array_equal(Z_np, Z_support)
        )

        if is_support_query:
            # Exact identity at support points: rho_hat_0(Phi_{-t}(Z_t,i)) = y0_i.
            rho_base = np.asarray(self.y0, dtype=np.float64).copy()
        else:
            # Off-support: kernel-swap approximation.
            rho_base = self._evaluate_baseline_transported(Z_np)

        # Correction contribution — direct.
        rho_delta = self.gp_delta.predict(Z_np)

        out = rho_base + rho_delta
        return out[0] if single else out

    def _evaluate_baseline_transported(self, Z: ArrayLike) -> FloatArray:
        """
        APPROXIMATE evaluation of rho_hat_0(Phi_{-t}^0(z)) at query z.

        The approximation:
            k(z*, Z_t) @ alpha_0  ≈  rho_hat_0(Phi_{-t}^0(z*))

        is EXACT at z* = Z_t (support points), because Phi_{-t}^0(Z_t) = Z_0
        by construction and k(Z_t, Z_t) @ alpha_0 = y_0.  For OFF-SUPPORT
        queries z* ≠ Z_t — which is exactly where the QCLE operator needs
        the baseline, at midpoints Y — the ARD-RBF kernel is translation-
        invariant in Euclidean space but NOT invariant under the nonlinear
        MInt flow.  The error is O(|z* - Z_t|³) per kernel Taylor expansion
        around each center and grows with dt.

        Production justification: for small dt the midpoints Y(Z) differ
        from Z by O(dt), so the approximation error is O(dt³).  This is
        consistent with the overall midpoint-scheme truncation order.
        Accept as-is; do not rewrite as "exact" in comments or docstrings.

        Feature-zscore note: this function uses raw physical coordinates
        throughout (both query and centers).  The kernel is therefore
        evaluated with log_ell = log(gp0.lengthscales) where
        gp0.lengthscales returns physical-unit lengthscales.  Do NOT
        substitute gp0.log_lengthscales (normalized-space) here.
        """
        gp0 = self.gp0
        Z_np = np.asarray(Z, dtype=np.float64)
        single = (Z_np.ndim == 1)
        if single:
            Z_np = Z_np.reshape(1, D)

        # Use the correction GP's current centers as the transported
        # baseline centers (they are identical by construction at support
        # points). Apply the baseline GP's kernel and alpha.
        Z_centers_t = _as_numpy(self.gp_delta._Z_train)

        # Physical-unit lengthscales: gp0.lengthscales multiplies the
        # stored ell_norm by _feature_std when feature_zscore=True, so
        # this is correct regardless of the zscore setting.
        log_ell_raw = torch.log(_as_tensor(gp0.lengthscales))

        Zq_t = _as_tensor(Z_np)
        Zc_t = _as_tensor(Z_centers_t)
        with torch.no_grad():
            k_star = _ard_gram(Zq_t, Zc_t, gp0.log_sigma_f, log_ell_raw)
            mean = k_star @ gp0._alpha
        out = _as_numpy(mean)
        return out[0] if single else out

    # -------------------------------------------------------------------------
    # Moment values
    # -------------------------------------------------------------------------
    def compute_moment_values(self) -> Dict[str, float]:
        """
        <psi_m>_{rho_hat(t)} = <psi_m>_{rho_hat_0_transported} + <psi_m>_{delta_hat}

        For normalization and energy (both exactly conserved by PBME along
        trajectories), the baseline contribution equals its frozen t=0
        value. For trace (which is c_00 + c_11 = (r²+p²)/ℏ - 1; r²+p² is
        exactly conserved algebraically by MInt), same.

        So we CAN legitimately use the frozen baseline moment values for
        the three KKT moments without re-integrating. For general moments
        (r² individually, etc.) we use the moment integrator against the
        current transported centers.
        """
        base_vals = self.gp0.compute_moment_values()
        delta_vals = self.gp_delta.compute_moment_values()
        return {k: base_vals[k] + delta_vals[k] for k in base_vals}

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _inherit_hypers_from_baseline(self) -> None:
        """Copy baseline GP hypers into the correction GP as a warm start."""
        with torch.no_grad():
            self.gp_delta.log_sigma_f.copy_(self.gp0.log_sigma_f)
            self.gp_delta.log_sigma_n.copy_(self.gp0.log_sigma_n)
            self.gp_delta.log_lengthscales.copy_(self.gp0.log_lengthscales)

    def _compute_zero_delta_targets(
        self,
        total_targets: Optional[Dict[MomentName, float]],
    ) -> Dict[MomentName, float]:
        """Moment targets for delta at t=0 are all zero."""
        if total_targets is None:
            return {}
        return {k: 0.0 for k in total_targets}

    def _split_moment_targets(
        self,
        total_targets: Optional[Dict[MomentName, float]],
    ) -> Dict[MomentName, float]:
        """
        Turn moment targets for rho_hat into moment targets for delta_hat.

        Since <psi_m>_{rho_hat} = <psi_m>_{rho_hat_0} + <psi_m>_{delta},
        and the frozen baseline's moment value equals the user's target
        at t=0, the delta-GP's target is

            <psi_m>_{delta} = user_target - baseline_moment_value_0

        For PBME-conserved moments (normalization, trace, energy) the
        baseline-value is exactly conserved, so the delta-target stays
        zero forever.  That's the enforce_zero_moment_targets=True
        regime and is what we use in practice.
        """
        if total_targets is None:
            return {}
        if self.config.enforce_zero_moment_targets:
            return {k: 0.0 for k in total_targets}
        out = {}
        for k, v in total_targets.items():
            base_v = float(self._baseline_moment_values_0.get(k, 0.0))
            out[k] = float(v) - base_v
        return out

    # -------------------------------------------------------------------------
    # API shims
    # -------------------------------------------------------------------------
    def freeze_hypers(self) -> None:
        """Freezes both baseline and correction hypers. No-op on baseline
        (already frozen after fit)."""
        if not self.gp0._hypers_frozen:
            self.gp0.freeze_hypers()
        self.gp_delta.freeze_hypers()

    def report(self) -> str:
        return (
            "GPDensityDiff state\n"
            "  --- baseline ---\n"
            + "  " + self.gp0.report().replace("\n", "\n  ") + "\n"
            "  --- correction (delta) ---\n"
            + "  " + self.gp_delta.report().replace("\n", "\n  ") + "\n"
            "  --- split ---\n"
            f"  |y0|_inf                = {float(np.max(np.abs(self.y0))) if self.y0 is not None else float('nan'):.3e}\n"
            f"  |delta|_inf             = {float(np.max(np.abs(self.gp_delta._y_train.detach().cpu().numpy() if hasattr(self.gp_delta._y_train, 'detach') else self.gp_delta._y_train))) if self.gp_delta._y_train is not None else float('nan'):.3e}\n"
        )

    # -------------------------------------------------------------------------
    # Support-point shortcut for predictions (used by observable paths)
    # -------------------------------------------------------------------------
    def predict_at_support(self) -> FloatArray:
        """
        Fast path: rho_hat at the current support cloud = y0 + delta_fit
        (up to baseline & delta fit residuals). This avoids the kernel
        evaluation in `predict`.
        """
        if not self._initial_fit_done:
            raise RuntimeError("GPDensityDiff must be fitted first.")
        delta_at_support = self.gp_delta.predict(_as_numpy(self.gp_delta._Z_train))
        return self.y0 + delta_at_support