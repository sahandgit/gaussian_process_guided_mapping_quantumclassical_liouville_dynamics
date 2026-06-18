from __future__ import annotations

"""
KDEDensity.py
=============

Gaussian KDE surrogate used **only as a diagnostic comparison** against
the GP surrogate.  Does NOT participate in dynamics: the QCLE Q operator,
Operator.py, GPDerivatives.py, and Mint.py continue to use the ARD-RBF
GP surrogate exclusively.

What this is
------------
A standard signed-weight Gaussian kernel estimator on the 6D phase-
space cloud with per-trajectory weights ω_i · y_i (where the y_i may
be either non-negative for focused-mode labels or signed for seo_signed
labels and/or QCLE-corrected midpoint labels):

    ρ̂_KDE(z) = (1/(2π)^(D/2) · ∏_d h_d) Σ_i (ω_i y_i) ·
                  ∏_d exp(-½(z_d - Z_{i,d})² / h_d²)

with per-axis bandwidth h_d set by Silverman's rule on the cloud std:
    h_d = (4/(D+2))^{1/(D+4)} · σ_d(Z) · N^{-1/(D+4)}.

Sign behaviour
--------------
The estimator is mathematically a signed Gaussian mixture and inherits
its sign structure DIRECTLY from the weights ω_i · y_i.  Three regimes:

  1. Focused IC with frozen labels (PBME):  ω_i · y_i ≥ 0 by sampler
     construction.  ρ̂_KDE is non-negative everywhere — this is a
     property of the labels, not a feature of the KDE class itself.

  2. seo_signed IC:  w_i = ±1 ⇒ ω_i · y_i takes both signs.  ρ̂_KDE
     develops negative regions wherever the negative-weight cluster
     locally dominates, reproducing the signed Wigner structure to
     the extent the cloud samples it.  This is the regime where the
     KDE meaningfully represents a Wigner-like distribution.

  3. Focused IC with QCLE corrections (midpoint):  y_i(t) starts ≥ 0
     but Q can drive individual y_i past zero.  ρ̂_KDE then has
     localised negative regions at those trajectory positions — same
     local behaviour as case 2.

What the KDE does NOT do
------------------------
It does not impose ANY positivity constraint.  If you want a strictly-
positive density surrogate (e.g. for a probability-density
interpretation that requires ρ̂ ≥ 0 everywhere), this is the wrong tool.
Use a non-negative KDE (clip negative weights to zero), a log-KDE,
or a constrained-GP formulation instead.

What the KDE compares well against
----------------------------------
The GP surrogate `ρ̂_GP(z) = σ_f² Σ_i α_i ∏_d exp(-½(z_d-Z_{i,d})²/ℓ_d²)`
has the SAME signed-Gaussian-mixture form as ρ̂_KDE.  The only
mathematical difference is how the coefficients are determined:
  *  GP: α = K_y⁻¹ y (or KKT-projected variant) — chosen so the
         surrogate INTERPOLATES the labels at the support points.
  *  KDE: weights = ω_i · y_i directly — no interpolation, just
         smoothing of the label field by Gaussian bumps.
Both can take negative values when fed signed labels.  Both are smooth.
Their disagreement at high cloud-vs-kernel mismatch tells you the GP's
exact-interpolation requirement has driven α into wild sign oscillation
between support points — a regression artifact, not a physical feature.

What this is NOT
----------------
*   Not a replacement for the GP in the dynamics.  The ARD-RBF kernel
    and its derivatives drive QCLE corrections; that code path is
    untouched.
*   Not a tunable Bayesian object.  No marginal likelihood, no LOO-CV,
    no KKT projection.  It's a deterministic transform of the cloud.
*   Not exact-interpolating.  KDE oversmooths labels at support points
    by design — that's the trade for guaranteed positivity.

Used by
-------
*   Diagnostic moment integrals reported alongside the GP (km_* vs
    kde_*) so we can see when the GP and the KDE disagree, which is
    exactly when the GP has stopped being faithful to the cloud.
*   The faithfulness 2D-marginal plot (Visualization.plot_faithfulness_
    2d_marginal): GP-integral / KDE-integral / difference panels.
*   Faithfulness tests (test_faithfulness.test_kde_is_nonnegative).

Closed-form moment integrals
----------------------------
With a product-Gaussian kernel of bandwidth h_d on each axis, every
polynomial moment of ρ̂_KDE has a closed form analogous to the ARD-RBF
GP case (replace α_i with ω_i y_i, replace ℓ_d with h_d, drop the
σ_f² prefactor):

    ∫ ρ̂_KDE dz                                   = Σ_i (ω_i y_i)
    ∫ z_d · ρ̂_KDE dz                              = Σ_i (ω_i y_i) Z_{i,d}
    ∫ z_d² · ρ̂_KDE dz                             = Σ_i (ω_i y_i) (Z_{i,d}² + h_d²)
    ∫ (r_α² + p_α² - ℏ)/(2ℏ) · ρ̂_KDE dz           = (...mapping moment...)
    ∫ H(R) · ρ̂_KDE dz                             = Σ_i (ω_i y_i) (∫ K_h(R-R_i) H(R) dR)
                                                  (for Tully H this is
                                                   the convolution with
                                                   a 1D Gaussian; we
                                                   evaluate at training
                                                   R_i and add a 2nd-
                                                   moment correction.)

These are implemented in `KDEDensity.compute_moment_values` so the
diagnostic layer can call it the same way as `GPDensity.compute_moment_
values`.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from .Mint import D, PBMEMIntDynamics, pack_z

FloatArray = NDArray[np.float64]


# =============================================================================
# Bandwidth selection
# =============================================================================

def silverman_bandwidth(Z: FloatArray, axis_std_floor: float = 1.0e-6) -> FloatArray:
    r"""
    Silverman's rule of thumb for product-kernel multivariate KDE.

        h_d = (4 / (D+2))^{1/(D+4)} · σ_d · N^{-1/(D+4)}

    For D=6 the multiplicative prefactor (4/8)^{1/10} ≈ 0.933.  N^{-1/10}
    decays slowly with sample size, so for N=300 ÷ 1000 the bandwidth
    sits around 0.4·σ_d ÷ 0.5·σ_d.

    A per-axis floor `axis_std_floor` keeps h_d strictly positive when
    one axis is sample-degenerate (e.g. mapping axes locked on the
    focus manifold to a circle of radius √3 — their std is non-zero
    but very small).
    """
    N = max(int(Z.shape[0]), 2)
    sigma = np.maximum(np.std(Z, axis=0), axis_std_floor)
    return (4.0 / (D + 2)) ** (1.0 / (D + 4)) * sigma * N ** (-1.0 / (D + 4))


# =============================================================================
# KDE surrogate
# =============================================================================

@dataclass
class KDEDensityConfig:
    """
    Knobs for the KDE diagnostic surrogate.

    bandwidth_floor : minimum h_d allowed on any axis.  Defaults to 1e-3
        in physical units.  Prevents zero-bandwidth on degenerate axes
        (e.g. r_α + p_α² locked on the focus circle has near-constant
        radius; without the floor the per-axis std on r or p alone is
        small but nonzero, giving a Silverman bandwidth that's too narrow
        for meaningful averaging).
    bandwidth_anchor : optional per-axis bandwidth override.  Pass a
        length-D array; entries set to NaN fall back to Silverman.
        Useful for matching the GP's pinned focus anchors so the KDE
        and GP have the same effective mapping-axis bandwidth.
    """
    bandwidth_floor: float = 1.0e-3
    bandwidth_anchor: Optional[FloatArray] = None


class KDEDensity:
    r"""
    Diagnostic-only Gaussian KDE surrogate.

    Construction is cheap: build_from_cloud(Z, ω·y) stores the cloud and
    computes h_d once via Silverman.  All evaluations are direct sums
    over the N training points.

    The surrogate is non-negative iff all (ω_i y_i) are non-negative,
    which holds in focused mode by construction.  For seo_signed mode
    the KDE can take negative values (mirrors the underlying signed
    sampler) — same as the GP can.  Positivity guarantees are a
    property of the labels, not of the surrogate.
    """

    def __init__(self,
                 config: Optional[KDEDensityConfig] = None,
                 dynamics: Optional[PBMEMIntDynamics] = None):
        self.config = config if config is not None else KDEDensityConfig()
        self.dynamics = dynamics if dynamics is not None else PBMEMIntDynamics()
        self.hbar = float(self.dynamics.params.hbar)
        # Lazily set in build_from_cloud
        self._Z: Optional[FloatArray] = None
        self._w: Optional[FloatArray] = None         # ω_i · y_i (the KDE weights)
        self._h: Optional[FloatArray] = None         # per-axis bandwidth (D,)

    # ----------------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------------
    def build_from_cloud(self,
                         Z: FloatArray,
                         omega: FloatArray,
                         y: FloatArray) -> None:
        """
        Build the KDE from the support cloud Z and frozen geometric
        measure ω, with current labels y.

        omega and y are kept as separate inputs (rather than passing the
        product) so callers can be explicit about which is which.
        """
        Z_arr = np.asarray(Z,     dtype=np.float64).reshape(-1, D)
        w_om  = np.asarray(omega, dtype=np.float64).reshape(-1)
        w_y   = np.asarray(y,     dtype=np.float64).reshape(-1)
        if not (Z_arr.shape[0] == w_om.size == w_y.size):
            raise ValueError(
                f"Z ({Z_arr.shape[0]}), omega ({w_om.size}), y ({w_y.size})"
                f" must have matching sample counts."
            )
        self._Z = Z_arr
        self._w = w_om * w_y                          # combined KDE weight
        # Silverman bandwidth; clip to floor and apply anchors.
        h = silverman_bandwidth(Z_arr)
        if self.config.bandwidth_anchor is not None:
            anchor = np.asarray(self.config.bandwidth_anchor,
                                dtype=np.float64).reshape(-1)
            if anchor.size != D:
                raise ValueError(f"bandwidth_anchor must have length {D}; got {anchor.size}")
            mask = np.isfinite(anchor)
            h = np.where(mask, anchor, h)
        h = np.maximum(h, float(self.config.bandwidth_floor))
        self._h = h

    # ----------------------------------------------------------------------
    # Properties (mirror GP interface for diagnostics)
    # ----------------------------------------------------------------------
    @property
    def Z_train(self) -> FloatArray:
        if self._Z is None:
            raise RuntimeError("KDEDensity not built; call build_from_cloud first.")
        return self._Z

    @property
    def bandwidth(self) -> FloatArray:
        if self._h is None:
            raise RuntimeError("KDEDensity not built; call build_from_cloud first.")
        return self._h

    @property
    def weights(self) -> FloatArray:
        if self._w is None:
            raise RuntimeError("KDEDensity not built; call build_from_cloud first.")
        return self._w

    # ----------------------------------------------------------------------
    # Pointwise evaluation
    # ----------------------------------------------------------------------
    def predict(self, z_query: FloatArray) -> FloatArray:
        r"""
        Evaluate ρ̂_KDE(z) at query points.

        z_query : (M, D) array.  Returns (M,) array of density values.

        Direct O(M·N) sum.  For M ~ N ~ 1000 this is ~10^6 ops; fine for
        diagnostic plotting and per-step moment evaluation.
        """
        if self._Z is None:
            raise RuntimeError("KDEDensity not built; call build_from_cloud first.")
        z = np.atleast_2d(np.asarray(z_query, dtype=np.float64))
        if z.shape[1] != D:
            raise ValueError(f"z_query must have shape (M, {D}); got {z.shape}.")
        # Outer Gaussian product over axes, contracted over training points.
        diff = z[:, None, :] - self._Z[None, :, :]                   # (M, N, D)
        # log-Gaussian per axis: -½(diff/h)² ; sum over axes; exp once.
        log_k = -0.5 * np.sum((diff / self._h[None, None, :]) ** 2, axis=-1)
        norm = 1.0 / ((2.0 * np.pi) ** (D / 2) * np.prod(self._h))
        return norm * (np.exp(log_k) * self._w[None, :]).sum(axis=1)

    # ----------------------------------------------------------------------
    # Analytic moment integrals (mirror Observables.kkt_moments / nuclear_moments)
    # ----------------------------------------------------------------------
    def compute_moment_values(self) -> Dict[str, float]:
        r"""
        Closed-form integrals of polynomial-in-z observables against
        the KDE.  Mirrors `GPDensity.compute_moment_values` for the
        moments that have polynomial integrands.

        Returns
        -------
        normalization    : ∫ ρ̂_KDE dz  =  Σ_i (ω_i y_i)
        trace            : ∫ (c_00 + c_11) ρ̂_KDE dz
                         =  Σ_i (ω_i y_i) · ((r_0²+p_0²+r_1²+p_1²)_i + 2h_r²+2h_p²) / (2ℏ) − Σw
            (the +h² terms account for the KDE smoothing of x²)
        energy           : ∫ H(R) · trace(c) ρ̂_KDE dz
            This is more involved because H is non-polynomial in R.
            We use the cloud Riemann approximation
                ∫ H ρ̂_KDE dz ≈ Σ_i (ω_i y_i) H(R_i)
            (zeroth-order in h_R).  For Silverman h_R ~ 0.4 σ_R the
            error is ½ h_R² H''(R̄) ~ O(0.1) of the energy scale for the
            Tully dual model; small enough for diagnostic comparison
            but not a precision instrument.  Acceptable here because
            the GP also approximates H non-trivially — we're checking
            whether GP and KDE agree, not whether KDE is right in
            absolute terms.
        """
        if self._Z is None:
            raise RuntimeError("KDEDensity not built; call build_from_cloud first.")
        Z = self._Z
        w = self._w
        h = self._h
        hbar = self.hbar

        # Normalization: ∫ ρ̂_KDE dz = Σ w_i (each kernel integrates to 1).
        norm_int = float(np.sum(w))

        # Trace observable: c_00 + c_11 = ((r0²+p0²+r1²+p1²) - 2ℏ) / (2ℏ).
        # ⟨r0²⟩_KDE per training point = r0_i² + h_r0²  (Gaussian second moment).
        h_r0, h_r1, h_p0, h_p1 = h[2], h[3], h[4], h[5]
        sq_per_pt = (
            (Z[:, 2] ** 2 + h_r0 ** 2)
            + (Z[:, 3] ** 2 + h_r1 ** 2)
            + (Z[:, 4] ** 2 + h_p0 ** 2)
            + (Z[:, 5] ** 2 + h_p1 ** 2)
        )
        trace_int = float(np.sum(w * (sq_per_pt - 2.0 * hbar))) / (2.0 * hbar)

        # Energy observable: ∫ H(R)·trace(c) ρ̂_KDE dz
        # For Tully H is a 2×2 matrix-valued function of R, but the
        # scalar energy density Tr(H · c) at a single training point is
        # `dynamics.energy(z_i)`.  Riemann-approximate the integral.
        E_per_pt = np.asarray(self.dynamics.energy(Z), dtype=np.float64).reshape(-1)
        energy_int = float(np.sum(w * E_per_pt))

        return {
            "normalization": norm_int,
            "trace":         trace_int,
            "energy":        energy_int,
        }


# =============================================================================
# Convenience: build the KDE that matches a given GP's training cloud
# =============================================================================

def build_kde_from_gp(gp,
                      omega: FloatArray,
                      y: Optional[FloatArray] = None,
                      bandwidth_anchor: Optional[FloatArray] = None,
                      config: Optional[KDEDensityConfig] = None) -> KDEDensity:
    """
    Build a KDE surrogate from the same training cloud the GP is using.

    Bandwidth strategy by default
    -----------------------------
    For focused-mode GPs (those carrying a LabelInformation pin with
    apply_kkt=False), this function sets the bandwidth on the **pinned
    mapping axes** to (effectively) zero.  Rationale: focused labels
    are supported on a measure-zero submanifold in (r, p) — they
    don't actually sample a 4D distribution on the mapping axes,
    they sample a 2D submanifold (two circles).  Any nonzero h on
    the mapping axes adds a fictitious +Σh² to the Casimir moment
    (r_α² + p_α² becomes (r_α² + p_α²) + h_r² + h_p², which differs
    from the on-circle value 2ℏ(1+γ) or 2ℏγ).  Setting h_mapping ≈ 0
    on pinned axes keeps the KDE's analytic trace integral equal to
    the per-trajectory Casimir value, which is what the cloud Riemann
    sum reports.

    For seo_signed-mode GPs (no pin or apply_kkt=True), all six
    bandwidths use Silverman's rule — appropriate because seo_signed
    actually samples the full 6D envelope.

    Override via `bandwidth_anchor` if you want to force specific
    per-axis bandwidths (NaN entries fall back to the default above).
    """
    Z_train = gp._Z_train.detach().cpu().numpy() if hasattr(gp._Z_train, "detach") else np.asarray(gp._Z_train)
    if y is None:
        y_train = gp._y_train.detach().cpu().numpy() if hasattr(gp._y_train, "detach") else np.asarray(gp._y_train)
    else:
        y_train = np.asarray(y, dtype=np.float64).reshape(-1)

    # Auto-set bandwidth anchors from the LabelInformation pin when no
    # explicit anchor is supplied.
    auto_anchor = None
    if bandwidth_anchor is None:
        pin_mask = getattr(gp, "_pin_mask", None)
        apply_kkt = bool(getattr(gp, "_pin_apply_kkt", True))
        if pin_mask is not None and not apply_kkt and bool(np.any(pin_mask)):
            # Focused-mode pin: zero out the bandwidth on pinned axes.
            # Use a small finite value rather than literal 0 because
            # the Gaussian normalisation divides by ∏h_d — exact zero
            # would NaN.  1e-30 is small enough to be numerically zero
            # for the Casimir correction (h² ~ 1e-60 ≪ machine eps in
            # any double-precision arithmetic).
            auto_anchor = np.where(pin_mask, 1.0e-30, np.nan)

    final_anchor = bandwidth_anchor if bandwidth_anchor is not None else auto_anchor

    if config is None:
        cfg = KDEDensityConfig(bandwidth_anchor=final_anchor,
                               bandwidth_floor=1.0e-30 if auto_anchor is not None
                                                       else 1.0e-3)
    else:
        cfg = config

    kde = KDEDensity(cfg, dynamics=getattr(gp, "dynamics", None))
    kde.build_from_cloud(Z_train, omega, y_train)
    return kde