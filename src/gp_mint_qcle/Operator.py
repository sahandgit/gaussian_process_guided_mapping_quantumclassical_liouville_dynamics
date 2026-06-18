from __future__ import annotations

"""
Operator.py
===========

Analytic-Y-derivative pulled-back QCLE midpoint operator.

Mathematical setup
------------------
At a support point Z (the post-step cloud), the midpoint-corrected
density is

    ρ_m(Δt; Z) = ρ_sur(X(Z)) + Δt · Q[ρ_sur](Z),

with the backward-flow maps  X(Z) = Φ_{-Δt}(Z),  Y(Z) = Φ_{-Δt/2}(Z)
and the intrinsic midpoint operator at Y

    iL'_m(Y) ρ = -(ℏ/8) Σ_{α,β} (∂h̄^{αβ}/∂R)(Y_R)
                  · [ ∂³ρ/(∂Y_P ∂Y_{r_β} ∂Y_{r_α})
                    + ∂³ρ/(∂Y_P ∂Y_{p_β} ∂Y_{p_α}) ].

Sign convention follows Nassimi, Bonella & Kapral, J. Chem. Phys.
133, 134115 (2010), Eq. (8) and Eq. (10).  The mapping-basis QCLE
reads

    ∂_t ρ_m  =  {H_m, ρ_m}_{X,x}
                − (ℏ/8) (∂h̄/∂R) (∂²_{rr} + ∂²_{pp}) ∂_P ρ_m ,

so the QCLE-minus-PBME rate is the second term WITH ITS NEGATIVE
LEADING SIGN.  We therefore define

    Q[ρ](Z) ≡ (∂_t ρ)_QCLE − (∂_t ρ)_PBME  =  [iL'_m ρ](Y(Z)) ,

which carries the −ℏ/8 prefactor and matches the sign of the code's
pref = -hbar/8 throughout the module.

Sign of the discrete update
---------------------------
The discrete midpoint update is  ρ_m(Δt; Z) = ρ_sur(X(Z)) + Δt · Q  (PLUS Δt·Q,
NOT minus).  Derivation: the QCLE field equation rewrites as

    ∂_t ρ = -u_PBME · ∇ρ + Q,

so along a PBME trajectory z_PBME(t),

    d/dt ρ(z_PBME(t), t)
        = ∂_t ρ + u_PBME · ∇ρ
        = (-u_PBME · ∇ρ + Q) + u_PBME · ∇ρ
        = +Q,                              ← labels grow with +Q under PBME

and midpoint integration of this trajectory ODE gives the + sign on Δt·Q.
The Dynamics.py label integrator is wired to this convention: the Q returned
by this module is inserted directly into d(w_i y_i)/dt = +Q_i.

The QCLE correction at the post-step support point Z is the value of
this operator applied to the surrogate density and evaluated at the
backward midpoint Y(Z):

    Q[ρ_sur](Z) := [iL'_m ρ_sur](Y(Z)).

The derivatives are Y-derivatives of ρ_sur evaluated at the point Y(Z).

Why this is rewritten away from autodiff-on-composition
-------------------------------------------------------
The previous implementation defined  g(Z) := ρ_sur(Y(Z))  and computed

    ∂³g/(∂Z_iP ∂Z_{ir_β} ∂Z_{ir_α})

via three nested ``jax.jacfwd`` on the composed function g.  By the chain
rule

    ∂³g/∂Z_i∂Z_j∂Z_k = Σ_{abc} ρ_{,abc}(Y) · J_{a,i} J_{b,j} J_{c,k}
                     + Σ_{ab}  ρ_{,ab}(Y)  · [H,J terms]
                     + Σ_a     ρ_{,a}(Y)   · T_{a,ijk}

where J = ∂Y/∂Z, H = ∂²Y/∂Z∂Z, T = ∂³Y/∂Z∂Z∂Z.  The Z-derivatives of g
equal the Y-derivatives of ρ_sur only when J = I, H = 0, T = 0 — i.e.,
only at Δt = 0.  For finite Δt, J = I + O(Δt), H = O(Δt), T = O(Δt),
so the previous path treated  Σ dh^{αβ} · ∂³g/∂Z_{iP} ∂Z_{ir_β} ∂Z_{ir_α}
as if it were  Σ dh^{αβ} · ∂³ρ/∂Y_P ∂Y_{r_β} ∂Y_{r_α}, missing chain-rule
corrections of relative size O(Δt).  These corrections are below the
midpoint-scheme truncation order asymptotically, but at finite Δt and
near regions of rapid spatial variation in ρ̂ (e.g. avoided crossings)
they can be the leading source of error in the diagnostic Q time series.

The new path bypasses the chain rule entirely.  The surrogate

    ρ_sur(Y) = Σ_i α_i k(Y, Z_i),     k(Y, Z_i) = σ_f² exp(-½ Σ_d ((Y_d-Z_{i,d})/ℓ_d)²)

has analytic third derivatives in Y, with single-kernel components

    ∂³k_i/(∂Y_a ∂Y_b ∂Y_c)
        = [-v_a v_b v_c + δ_{ab} λ_a v_c + δ_{ac} λ_a v_b + δ_{bc} λ_b v_a] · k_i

where v_d^{(i)} = (Y_d - Z_{i,d})/ℓ_d², λ_d = 1/ℓ_d².  For the QCLE
indices (iP, ir_β, ir_α) and (iP, ip_β, ip_α) the off-block deltas
δ_{P, r_*}, δ_{P, p_*} vanish, so

    ∂³k_i/(∂Y_P ∂Y_{r_β} ∂Y_{r_α})
        = [-v_P v_{r_β} v_{r_α} + δ_{αβ} λ_{r_β} v_P] k_i,

and analogously for p.  Contracting with dh = ∂h̄/∂R(Y_R) and using
Tr(dh) = 0 (h̄ is traceless) collapses the diagonal-correction term
neatly:

    Σ_{α,β} dh^{αβ} ∂³ρ/(∂Y_P ∂Y_{r_β} ∂Y_{r_α})
        = Σ_i α_i k_i v_P^{(i)} · [-v_r^{(i)T} dh v_r^{(i)} + dh^{00}(λ_{r_0} - λ_{r_1})].

The analytic path therefore reduces to an O(N) contraction per support
point — cheaper than constructing the full (D,D,D) third-derivative tensor.
The result is an exact Y-derivative evaluation at Y(Z), with no chain-
rule approximation.

This module exports
-------------------
``compute_Q(Z, gp, dt, dynamics)``
    Returns ``(Q, Y, dbarh_dR)`` where ``Q[n]`` is the scalar QCLE
    correction, ``Y[n]`` is the backward half-step midpoint, and
    ``dbarh_dR[n]`` is the (2, 2) electronic-coupling derivative at
    ``Y[n, R]``, for each row of ``Z``.

The function works for both ``GPDensity`` (vanilla single-kernel
surrogate) and ``GPDensityDiff`` (baseline + correction surrogate),
selecting the appropriate ρ_sur structure automatically.

A back-compatible ``QCLECorrection`` class is retained so existing
callers that import ``QCLECorrection.build`` keep working.

Legacy autodiff path
--------------------
# NOTE: Legacy autodiff functions (_legacy_Q_at_*, compute_Q_legacy) were
# removed and can be found in tests/test_operator_consistency.py.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .Mint import D, PBMEMIntDynamics
from .Monodromy import (
    _get_jax_step_fn,
    _I_R, _I_P, _I_R0, _I_R1, _I_P0, _I_P1,
    _I_R_MAP, _I_P_MAP,
)


FloatArray = NDArray[np.float64]


# =============================================================================
# Output container (kept for back-compat with diagnostics that read .Y)
# =============================================================================

@dataclass
class CorrectionData:
    """
    Diagnostic output of ``QCLECorrection.build``.

    Attributes
    ----------
    Y         : (N, D)    backward half-step midpoints  Φ_{-dt/2}(Z_n).
    dbarh_dR  : (N, 2, 2) ∂h̄^{αβ}/∂R evaluated at Y_R.
    Q         : (N,)      scalar QCLE correction at each support point.

    The pre-rewrite version exposed A (N, D), B (N, D, D), and C (N, D, D, D)
    tensors that callers contracted manually with kernel derivatives.  Those
    fields are gone — Q is now produced inside this module by autodiff.
    """
    Y:        FloatArray
    dbarh_dR: FloatArray
    Q:        FloatArray



# =============================================================================
# Tully-model h̄ derivative in JAX (for use at Y_R inside the autodiff trace)
# =============================================================================

def _build_dh_dR_jax(dynamics: PBMEMIntDynamics):
    """
    Return a JAX-traceable function  R -> ∂h̄^{αβ}/∂R(R)  of shape (2, 2).

    Mirrors PBMEMIntDynamics._frozen_R_objects's ``dh`` output and is
    consistent with the JAX MInt step function's diabatic potential.
    """
    kind = dynamics.model.params.kind
    pA   = float(dynamics.model.params.A)
    pB   = float(dynamics.model.params.B)
    pC   = float(dynamics.model.params.C)
    pD   = float(dynamics.model.params.D)

    def dh_dR(R):
        # Compute dV/dR analytically per Tully kind, then trace-traceless.
        if kind == "dual":
            dV11 = 0.0 * R                                    # zero with R's dtype
            dV22 = 2.0 * pA * pB * R * jnp.exp(-pB * R * R)
            dV12 = -2.0 * pC * pD * R * jnp.exp(-pD * R * R)
        elif kind == "simple":
            dV11 = pA * pB * jnp.exp(-pB * jnp.abs(R))
            dV22 = -dV11
            dV12 = -2.0 * pC * pD * R * jnp.exp(-pD * R * R)
        elif kind == "extended":
            dV11 = 0.0 * R
            dV22 = 0.0 * R
            dV12 = pC * pD * jnp.exp(-pD * jnp.abs(R))
        else:
            raise ValueError(f"Unknown Tully kind: {kind!r}")

        dV0 = 0.5 * (dV11 + dV22)
        return jnp.array([[dV11 - dV0, dV12      ],
                          [dV12,        dV22 - dV0]])

    return dh_dR


_DH_FN_CACHE: dict = {}

def _get_dh_fn(dynamics: PBMEMIntDynamics):
    key = (dynamics.model.params.kind,
           dynamics.model.params.A, dynamics.model.params.B,
           dynamics.model.params.C, dynamics.model.params.D,
           dynamics.model.params.E0)
    if key not in _DH_FN_CACHE:
        _DH_FN_CACHE[key] = _build_dh_dR_jax(dynamics)
    return _DH_FN_CACHE[key]


# =============================================================================
# Index-selection helper: pick out the Z-derivative entries that iL'_m needs
# =============================================================================
#
# The intrinsic operator iL'_m is
#
#   -(ℏ/8) Σ_{α,β} ∂h̄^{αβ}/∂R(Y_R)
#          · [ T[iP, ir_β, ir_α] + T[iP, ip_β, ip_α] ],
#
# where T = ∂³g/∂Z∂Z∂Z is the rank-3 derivative of g(Z) = ρ_sur(Y(Z)).
# We can write this as a single double-contraction
#
#   Σ_{αβ} dh[α, β] · M[α, β]
#
# with M[α, β] = T[iP, ir_β, ir_α] + T[iP, ip_β, ip_α].
#
# Note ∂³g is fully symmetric in its three slots (Schwarz's theorem),
# and ∂h̄/∂R is symmetric in (α, β), so the (α, β) double sum is
# well-defined and symmetric overall.

# Mapping coordinate r_α and p_α indices in the Z-vector (R, P, r0, r1, p0, p1):
_R_AXES = jnp.array(_I_R_MAP, dtype=jnp.int32)   # (2,) = [_I_R0, _I_R1] = [2, 3]
_P_AXES = jnp.array(_I_P_MAP, dtype=jnp.int32)   # (2,) = [_I_P0, _I_P1] = [4, 5]

# Mapping axes (r0, r1, p0, p1) and the (R, P) nuclear sub-block.  Used by the
# anisotropic-metric Q kernel: the (R, P) exponent and the P-score are computed
# from the full 2x2 metric M_rp; the mapping axes stay diagonal.
_MAP_AXES = jnp.array([_I_R0, _I_R1, _I_P0, _I_P1], dtype=jnp.int32)  # (4,)
_RP_AXES  = jnp.array([_I_R, _I_P], dtype=jnp.int32)                  # (2,)



# =============================================================================
# Analytic Y-derivative kernel of Q  (production path)
# =============================================================================
#
# These kernels evaluate
#
#     Q(Z) = -(ℏ/8) Σ_{α,β} dh^{αβ}(Y_R) ·
#            [ ∂³ρ/(∂Y_P ∂Y_{r_β} ∂Y_{r_α})
#            + ∂³ρ/(∂Y_P ∂Y_{p_β} ∂Y_{p_α}) ](Y(Z))
#
# (note the −ℏ/8 — matches pref in the code and the QCLE field
# equation; see top-of-file docstring), using the analytic ARD-RBF
# third-derivative formula in Y at the point Y = Y(Z).  No chain rule
# through Y(Z) — the derivatives are taken with respect to the
# kernel's first argument, evaluated at Y(Z).
#
# Derivation (fully written out so the formula in code is self-contained):
#
#   k(Y, Z_i) = σ_f² exp(-½ Σ_d ((Y_d - Z_{i,d})/ℓ_d)²)
#   v_d^{(i)} = (Y_d - Z_{i,d}) / ℓ_d²
#   λ_d      = 1 / ℓ_d²
#
#   ∂_{abc} k_i = [-v_a v_b v_c
#                  + δ_{ab} λ_a v_c
#                  + δ_{ac} λ_a v_b
#                  + δ_{bc} λ_b v_a] · k_i
#
# For (a, b, c) = (iP, ir_β, ir_α) all the off-block deltas vanish
# (δ_{P, r_*} = 0), leaving
#
#   ∂_{P, r_β, r_α} k_i = [-v_P v_{r_β} v_{r_α} + δ_{αβ} λ_{r_α} v_P] · k_i
#                       = v_P · [-v_{r_β} v_{r_α} + δ_{αβ} λ_{r_α}] · k_i.
#
# Contracting with dh^{αβ} (symmetric, traceless):
#
#   Σ_{αβ} dh^{αβ} ∂³ρ/(∂Y_P ∂Y_{r_β} ∂Y_{r_α})
#       = Σ_i α_i k_i v_P · [ -v_r^T dh v_r + Σ_α dh^{αα} λ_{r_α} ]
#       = Σ_i α_i k_i v_P · [ -v_r^T dh v_r + dh^{00}(λ_{r_0} - λ_{r_1}) ]
#
# (using Tr(dh) = 0, so dh^{11} = -dh^{00}).  Analogously for p.

def _q_kernel_inner(
    Y:           jnp.ndarray,    # (D,)   midpoint Y(Z)
    dh:          jnp.ndarray,    # (2, 2) ∂h̄^{αβ}/∂R at Y_R
    Z_centers:   jnp.ndarray,    # (N, D) GP centers
    alpha:       jnp.ndarray,    # (N,)   coefficient vector
    log_sigma_f: jnp.ndarray,    # scalar log σ_f
    log_ell:     jnp.ndarray,    # (D,)   log-physical lengthscales
    pref:        jnp.ndarray,    # scalar -ℏ/8 (negative, per Nassimi-Kapral 2010 Eq. 10)
    M_rp:        jnp.ndarray = None,  # (2,2) physical (R,P) precision; None ⇒ diagonal 1/ℓ²
) -> jnp.ndarray:
    """
    One-kernel contribution to Q at a single midpoint Y, with dh fixed.

    See module-level derivation.  Returns a scalar.

    All inputs are PHYSICAL: Z_centers and Y in raw coords, log_ell in
    log-physical units.  This matches `_extract_vanilla_params`.

    Anisotropic (R, P) metric
    -------------------------
    The QCLE operator differentiates ρ̂ once w.r.t. Y_P and twice w.r.t. the
    mapping coordinates.  The mapping block stays diagonal, so ONLY the kernel
    exponent's (R, P) part and the single P-score `v_P` depend on the nuclear
    metric.  When `M_rp` (the 2x2 physical precision M_RP = W Wᵀ) is supplied,
    the (R, P) exponent is (Δ_RP)ᵀ M_RP (Δ_RP) and v_P = [M_RP Δ_RP]_P; the
    mapping score/curvature (v_r, v_p, λ_r, λ_p) are unchanged.  `M_rp=None`
    reproduces the diagonal kernel exactly (M_rp = diag(1/ℓ_R², 1/ℓ_P²)).
    """
    inv_ls = jnp.exp(-log_ell)                 # (D,) 1/ℓ_d
    lam    = inv_ls * inv_ls                   # (D,) 1/ℓ_d² = λ_d

    diff   = Y[None, :] - Z_centers            # (N, D)  Y_d - Z_{i,d}
    v      = diff * lam[None, :]               # (N, D)  mapping score (R,P unused below)

    # (R, P) block via full metric (diagonal 1/ℓ² when not supplied).
    if M_rp is None:
        M_rp = jnp.diag(jnp.array([lam[_I_R], lam[_I_P]]))
    drp    = diff[:, _RP_AXES]                          # (N, 2)  (ΔR, ΔP)
    qf_RP  = jnp.einsum("ni,ij,nj->n", drp, M_rp, drp)  # (N,) (R,P) exponent
    u_map  = diff[:, _MAP_AXES] * inv_ls[_MAP_AXES][None, :]
    d2     = qf_RP + jnp.sum(u_map * u_map, axis=1)     # (N,)
    k      = jnp.exp(2.0 * log_sigma_f - 0.5 * d2)      # (N,) σ_f² exp(-½ d²)

    v_P    = (drp @ M_rp)[:, 1]                # (N,)  P-score = [M_RP Δ_RP]_P
    v_r    = v[:, _R_AXES]                     # (N, 2)  v at (r_0, r_1)
    v_p    = v[:, _P_AXES]                     # (N, 2)  v at (p_0, p_1)

    # quadratic forms v_r^T dh v_r and v_p^T dh v_p, per support point
    qf_r   = jnp.einsum("ni,ij,nj->n", v_r, dh, v_r)   # (N,)
    qf_p   = jnp.einsum("ni,ij,nj->n", v_p, dh, v_p)   # (N,)

    # diagonal-of-dh × λ correction (scalar, same for all i):
    #   Σ_α dh^{αα} λ_{r_α}     and     Σ_α dh^{αα} λ_{p_α}
    # Tr(dh) = 0 so this collapses to dh^{00}(λ_{r_0} - λ_{r_1}) etc.,
    # nonzero only when ARD lengthscales differ across the two mapping
    # axes within a block.
    lam_r  = lam[_R_AXES]                      # (2,)
    lam_p  = lam[_P_AXES]                      # (2,)
    dh_diag = jnp.diagonal(dh)                 # (2,)
    tr_r   = jnp.dot(dh_diag, lam_r)           # scalar
    tr_p   = jnp.dot(dh_diag, lam_p)           # scalar

    A_r    = -qf_r + tr_r                      # (N,)
    A_p    = -qf_p + tr_p                      # (N,)

    # Q contribution from this kernel
    return pref * jnp.sum(alpha * k * v_P * (A_r + A_p))


# Per-(dynamics, dt) JIT cache for the vmap'd Q_batch function.  Closing over
# step_fn and dh_fn is necessary because they are dynamics-specific Python
# functions; we cache the resulting compiled fn keyed on (id(step_fn), id(dh_fn)).
_Q_BATCH_VANILLA_CACHE: dict = {}
_Q_BATCH_DIFF_CACHE: dict = {}


def _Q_at_vanilla(z, Z_centers, alpha, log_sigma_f, log_ell_phys,
                  half_tau, pref, dh_fn, step_fn, M_rp=None):
    """Single-point Q via analytic Y-derivative — JAX-traceable.

    Returns (Q_scalar, Y_z, dh_at_Y).  Y_z and dh are returned for
    diagnostic/visualization use by the caller.
    """
    Y_z = step_fn(z, half_tau)
    dh  = dh_fn(Y_z[_I_R])
    Q   = _q_kernel_inner(Y_z, dh, Z_centers, alpha, log_sigma_f, log_ell_phys, pref, M_rp)
    return Q, Y_z, dh


def _Q_at_diff(z, Z_centers,
               alpha_b, log_sigma_f_b, log_ell_b,
               alpha_d, log_sigma_f_d, log_ell_d,
               half_tau, pref, dh_fn, step_fn, M_rp_b=None, M_rp_d=None):
    """Single-point Q for diff-GP — JAX-traceable.

    Both kernels share Z_centers (centers-swap convention): at support
    points the swap is exact; off-support it is O(|z-Z_t|³) ≈ O(dt³).

    Q is additive because ρ̂ = ρ_b + ρ_d and ∂³ is linear.

    ⚠ KNOWN APPROXIMATION — baseline third derivatives (Q_b)
    ---------------------------------------------------------
    _q_kernel_inner computes ∂³_Y [k(Y, Z_j)] @ alpha_b, i.e. third
    derivatives of the KERNEL centered at Z_t (current support centers).
    The EXACT transported-baseline third derivative at Y = Φ_{-dt/2}(Z)
    requires the monodromy chain rule:

        ∂³_z [ρ̂₀(Φ_{-t}(z))]
            = Σ chain-rule terms through J=∂Φ/∂z, H=∂²Φ/∂z², T=∂³Φ/∂z³

    None of those corrections are applied here.  The error in Q_b is
    O(dt) relative to the correct baseline rate.  Because Q enters the
    label update at O(dt), this gives an O(dt²) absolute error —
    consistent with the midpoint-scheme truncation order but NOT with the
    exact density-difference representation at finite dt.

    Trust cloud-weighted estimators (lw_*, cloud_weighted_*) over
    GP-analytic observables when the baseline dominates (long times past
    the avoided crossing, large alpha_b relative to alpha_d).
    """
    Y_z = step_fn(z, half_tau)
    dh  = dh_fn(Y_z[_I_R])
    Q_b = _q_kernel_inner(Y_z, dh, Z_centers, alpha_b, log_sigma_f_b, log_ell_b, pref, M_rp_b)
    Q_d = _q_kernel_inner(Y_z, dh, Z_centers, alpha_d, log_sigma_f_d, log_ell_d, pref, M_rp_d)
    return Q_b + Q_d, Y_z, dh
def _get_Q_batch_vanilla(step_fn, dh_fn):
    key = (id(step_fn), id(dh_fn))
    if key in _Q_BATCH_VANILLA_CACHE:
        return _Q_BATCH_VANILLA_CACHE[key]

    def Q_at(z, Z_centers, alpha, log_sigma_f, log_ell_phys, half_tau, pref, M_rp):
        return _Q_at_vanilla(z, Z_centers, alpha, log_sigma_f, log_ell_phys,
                              half_tau, pref, dh_fn, step_fn, M_rp)
    Q_batch = jax.jit(jax.vmap(Q_at, in_axes=(0, None, None, None, None, None, None, None)))
    _Q_BATCH_VANILLA_CACHE[key] = Q_batch
    return Q_batch


def _get_Q_batch_diff(step_fn, dh_fn):
    key = (id(step_fn), id(dh_fn))
    if key in _Q_BATCH_DIFF_CACHE:
        return _Q_BATCH_DIFF_CACHE[key]

    def Q_at(z, Z_centers,
            alpha_b, log_sigma_f_b, log_ell_b,
            alpha_d, log_sigma_f_d, log_ell_d,
            half_tau, pref, M_rp_b, M_rp_d):
        return _Q_at_diff(z, Z_centers,
                          alpha_b, log_sigma_f_b, log_ell_b,
                          alpha_d, log_sigma_f_d, log_ell_d,
                          half_tau, pref, dh_fn, step_fn, M_rp_b, M_rp_d)
    Q_batch = jax.jit(jax.vmap(
        Q_at,
        in_axes=(0,) + (None,) * 11,
    ))
    _Q_BATCH_DIFF_CACHE[key] = Q_batch
    return Q_batch


# =============================================================================
# L-matrix factorisation:  Q_i = Σ_j L_{ij} α_j
# =============================================================================
#
# Because _q_kernel_inner is LINEAR in α (the final reduction is
# ``pref * Σ_j α_j · k_j · v_P · (A_r + A_p)``), the per-point operator
#
#     Q_i  =  [iL'_m ρ_sur](Y(z_i))
#
# can be written as a matrix-vector product
#
#     Q_i  =  Σ_j L_{ij} α_j ,
#
# with
#
#     L_{ij}  =  −(ℏ/8) Σ_{αβ} (∂h̄^{αβ}/∂R)(Y_i,R)
#                · k(Y_i, Z_j) · v_P^{(ij)} · [A_r^{(ij)} + A_p^{(ij)}]
#
# (note the −ℏ/8 — matches pref throughout the module and the
# top-of-file Q sign convention).  v_P^{(ij)} = (Y_{i,P} − Z_{j,P})/ℓ_P²
# and A_r, A_p are the same quadratic forms _q_kernel_inner already
# computes per (i, j) pair.  Building L explicitly enables the column-
# sum-zero-projected, matrix-exponential label integrator in Dynamics.py.
#
# This implementation does NOT differentiate Q against α — instead it
# rewrites the per-pair contribution directly, so the cost is one
# ``vmap`` over support points i with a fully analytic inner kernel.
# ============================================================================


@jax.jit
def _L_row_vanilla(Y_i,        # (D,)  midpoint Y(z_i)
                    dh_i,       # (2,2) ∂h̄^{αβ}/∂R at Y_{i,R}
                    Z_centers,  # (N,D)
                    log_sigma_f,
                    log_ell,    # (D,)
                    pref,
                    M_rp=None): # (2,2) (R,P) precision; None ⇒ diagonal 1/ℓ²
    """One row L_{i, :} of the linear coupling, shape (N,).

    Anisotropic (R, P) metric handled identically to `_q_kernel_inner`:
    only the (R, P) exponent and the P-score v_P use M_rp; mapping diagonal."""
    inv_ls = jnp.exp(-log_ell)
    lam    = inv_ls * inv_ls
    diff   = Y_i[None, :] - Z_centers          # (N, D)
    v      = diff * lam[None, :]

    if M_rp is None:
        M_rp = jnp.diag(jnp.array([lam[_I_R], lam[_I_P]]))
    drp    = diff[:, _RP_AXES]                          # (N, 2)
    qf_RP  = jnp.einsum("ni,ij,nj->n", drp, M_rp, drp)
    u_map  = diff[:, _MAP_AXES] * inv_ls[_MAP_AXES][None, :]
    d2     = qf_RP + jnp.sum(u_map * u_map, axis=1)
    k      = jnp.exp(2.0 * log_sigma_f - 0.5 * d2)

    v_P    = (drp @ M_rp)[:, 1]                # (N,)  [M_RP Δ_RP]_P
    v_r    = v[:, _R_AXES]                     # (N, 2)
    v_p    = v[:, _P_AXES]

    qf_r   = jnp.einsum("ni,ij,nj->n", v_r, dh_i, v_r)
    qf_p   = jnp.einsum("ni,ij,nj->n", v_p, dh_i, v_p)

    lam_r  = lam[_R_AXES]
    lam_p  = lam[_P_AXES]
    dh_diag = jnp.diagonal(dh_i)
    tr_r   = jnp.dot(dh_diag, lam_r)
    tr_p   = jnp.dot(dh_diag, lam_p)

    A_r    = -qf_r + tr_r
    A_p    = -qf_p + tr_p
    # L_{ij} = pref · k_j · v_P_j · (A_r_j + A_p_j) — note NO sum over j
    return pref * k * v_P * (A_r + A_p)        # (N,)


def _get_L_batch_vanilla(step_fn, dh_fn):
    key = ("L", id(step_fn), id(dh_fn))
    if key in _Q_BATCH_VANILLA_CACHE:
        return _Q_BATCH_VANILLA_CACHE[key]

    def L_at(z, Z_centers, log_sigma_f, log_ell, half_tau, pref, M_rp):
        Y_z = step_fn(z, half_tau)
        dh  = dh_fn(Y_z[_I_R])
        return _L_row_vanilla(Y_z, dh, Z_centers, log_sigma_f, log_ell, pref, M_rp)

    L_batch = jax.jit(jax.vmap(
        L_at, in_axes=(0, None, None, None, None, None, None)))
    _Q_BATCH_VANILLA_CACHE[key] = L_batch
    return L_batch


def compute_L_matrix(
    Z_eval:   ArrayLike,
    gp,
    dt:       float,
    dynamics: PBMEMIntDynamics,
) -> FloatArray:
    """
    Build the (N, N) linear coupling matrix L such that

        Q_i  =  Σ_j L_{ij} α_j ,

    i.e. the per-support-point QCLE corrector is a linear function of
    the GP coefficients α through the kernel structure.  Combined with
    α = K⁻¹ b at the cloud points (b_i = w_i y_i = ρ̂(z_i) is the label-
    product variable), this gives the closed-form label-ODE generator

        ḃ = Q = L α = L K⁻¹ b   ⇒   A = L K⁻¹

    used as the linear-system generator for any matrix-exponential or
    Padé integrator built on the label ODE.  Note the sign: L itself
    already carries the −ℏ/8 prefactor that ``compute_Q`` uses (the
    ``pref`` literal at the construction site below is negative), so
    A = +L K⁻¹.  An EARLIER version of this docstring wrote A = −L K⁻¹,
    which would double-count the sign and flip the integrator.  The
    code itself was already correct; the comment was wrong.

    Requires the vanilla GP path (not the diff-GP).  For diff-GP, the
    same factorisation applies separately to gp0 and gp_delta but is
    not currently exposed here.
    """
    Z_eval = np.asarray(Z_eval, dtype=np.float64)
    if Z_eval.ndim == 1:
        Z_eval = Z_eval.reshape(1, D)
    if _is_density_diff(gp):
        raise NotImplementedError(
            "compute_L_matrix is currently implemented only for the "
            "vanilla GP path.  For diff-GP, build L separately for "
            "gp0 and gp_delta and sum (same kernel structure)."
        )
    Z_centers, alpha, sigma_f, ell_phys = _extract_vanilla_params(gp)
    M_rp = _rp_metric_of(gp, ell_phys)

    step_fn = _get_jax_step_fn(dynamics)
    dh_fn   = _get_dh_fn(dynamics)
    half_tau = jnp.asarray(-0.5 * float(dt))
    pref     = jnp.asarray(-float(dynamics.params.hbar) / 8.0)

    L_batch = _get_L_batch_vanilla(step_fn, dh_fn)
    L = L_batch(
        jnp.asarray(Z_eval),
        jnp.asarray(Z_centers),
        jnp.asarray(np.log(sigma_f)),
        jnp.asarray(np.log(ell_phys)),
        half_tau, pref,
        jnp.asarray(M_rp),
    )
    return np.asarray(L, dtype=np.float64)


# =============================================================================
# Public API
# =============================================================================

def _is_density_diff(gp) -> bool:
    """Duck-type recognition of GPDensityDiff without importing it."""
    return (hasattr(gp, "gp0") and hasattr(gp, "gp_delta")
            and hasattr(gp, "Z0") and hasattr(gp, "y0"))


def _detach_to_numpy(x) -> FloatArray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64)


def _rp_metric_of(gp, ell_phys) -> FloatArray:
    """
    Physical (R, P) precision M_rp = W Wᵀ for the QCLE kernel.

    Returns gp._rp_precision when the anisotropic metric is active
    (config.aniso_nuclear_metric), otherwise the diagonal precision
    diag(1/ℓ_R², 1/ℓ_P²) built from the physical lengthscales — which
    makes _q_kernel_inner / _L_row_vanilla reproduce the legacy diagonal
    kernel exactly.
    """
    rp = getattr(gp, "_rp_precision", None)
    if rp is not None:
        return np.asarray(rp, dtype=np.float64)
    e = np.asarray(ell_phys, dtype=np.float64)
    return np.diag([1.0 / (e[0] * e[0]), 1.0 / (e[1] * e[1])]).astype(np.float64)


def _extract_vanilla_params(gp) -> Tuple[FloatArray, FloatArray, float, FloatArray]:
    """
    Pull (Z_centers, alpha, sigma_f, ell_phys) out of a vanilla GPDensity
    as numpy arrays in physical units.  ``ell_phys`` is the user-facing
    physical lengthscale (already accounts for feature_zscore via
    gp.lengthscales).
    """
    if gp._alpha is None or gp._Z_train is None:
        raise RuntimeError("GP must be fitted before compute_Q.")
    Z_centers = (gp.raw_training_centers
                 if hasattr(gp, "raw_training_centers")
                 else _detach_to_numpy(gp._Z_train))
    Z_centers = np.asarray(Z_centers, dtype=np.float64)
    alpha     = _detach_to_numpy(gp._alpha)
    sigma_f   = float(gp.sigma_f)
    ell_phys  = np.asarray(gp.lengthscales, dtype=np.float64)
    return Z_centers, alpha, sigma_f, ell_phys


def compute_Q(
    Z_eval:   ArrayLike,
    gp,                              # GPDensity or GPDensityDiff
    dt:       float,
    dynamics: PBMEMIntDynamics,
    q_sigma_n_scale: float = 1.0,   # derivative-smoothing: re-solve α with larger σ_n
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """
    Compute the QCLE midpoint correction at each row of ``Z_eval``.

        Q[n] = -(ℏ/8) Σ_{α,β} ∂h̄^{αβ}/∂R(Y_n,R)
                      · [ ∂³ρ_sur/(∂Y_P ∂Y_{r_β} ∂Y_{r_α})
                        + ∂³ρ_sur/(∂Y_P ∂Y_{p_β} ∂Y_{p_α}) ](Y_n)

    where Y_n = Φ_{-dt/2}(Z_n) is the backward half-step midpoint.  The
    third derivatives are INTRINSIC to ρ_sur: they are derivatives of
    the kernel-expansion density with respect to Y, NOT chain-rule
    derivatives of the composition ρ_sur(Y(Z)) with respect to Z.

    The −ℏ/8 prefactor matches the QCLE field equation and the
    trajectory-rate convention used by Dynamics.py
        d(w_i y_i)/dt = +Q_i
    so the operator's sign output goes directly into the label ODE
    with no extra minus.

    Production implementation (analytic Y-derivative path)
    ------------------------------------------------------
    The third-derivative tensor is computed in closed form from the
    Gaussian kernel structure (see ``_q_kernel_inner`` and the
    module-level derivation):

        ∂³k_i/(∂Y_P ∂Y_{r_β} ∂Y_{r_α}) = v_P [-v_{r_β} v_{r_α}
                                              + δ_{αβ} λ_{r_α}] k_i

    where v_d = (Y_d - Z_{i,d})/ℓ_d² and λ_d = 1/ℓ_d².  No JAX autodiff
    through Y(Z)'s dependence on Z is performed; the only "chain rule"
    is in the analytic kernel-derivative formula, which is closed-form.
    Y(Z) is evaluated by the MInt symplectic step but its Jacobian
    dY/dZ never enters the operator value — Q is built directly from
    intrinsic Y-derivatives at Y(Z).

    The LEGACY ``compute_Q_legacy`` (and ``_legacy_Q_at_vanilla``) path
    uses ``jax.jacfwd(jax.jacfwd(jax.jacfwd(g)))`` on
    g(Z) := ρ_sur(Y(Z)), i.e. JAX-autodiff on the COMPOSITION.  This is
    NOT pointwise equivalent to the intrinsic-Y-derivative path in
    general — the two differ by chain-rule terms of relative size O(Δt)
    (see the module-level "Why this is rewritten away from
    autodiff-on-composition" section for the full chain-rule expansion).

    When DO the two paths agree?
    ----------------------------
    * In the limit Δt → 0, trivially: Y = Z, the chain-rule corrections
      collapse, and Q_intrinsic = Q_composed exactly.

    * For ``build_at_points(Z, gp=…)`` (the production path used by
      MidpointScheme.step's k_1 and k_2 stages), the pullback is
      DISABLED — internally the operator sets Y := Z (no backward
      half-step).  Z IS the evaluation point, so the chain-rule "from
      Z to Y" trivializes (∂Y/∂Z = I, higher derivatives = 0) and the
      two paths agree to machine precision at the trajectory cloud
      points where the trajectory-RK2 scheme uses Q.  This is the
      regime the production path was designed for.

    * For ``build(Z, dt, gp=…)`` (the legacy Eulerian-style path with
      internal Y = Φ_{-Δt/2}(Z) pullback), Y ≠ Z generically and the
      chain-rule corrections of size O(Δt) are nontrivial.  In this
      regime the two paths DISAGREE at finite Δt by the chain-rule
      terms; the intrinsic-Y path is mathematically correct because
      the QCLE operator is defined by Y-derivatives of ρ_sur(Y) at the
      pulled-back point, not by Z-derivatives of the composition.

    A direct regression test of agreement under both regimes is
    available in tests; see ``test_intrinsic_vs_composed_Q``.  The
    earlier blanket claim "agree to ≲ 5·10⁻¹⁶ because symplectic
    structure makes corrections vanish" was inaccurate and has been
    removed; in the trajectory-RK2 regime they DO agree to that
    precision, but for reasons of how the pullback is disabled, not
    because symplectic structure cancels generic chain-rule terms.

    Parameters
    ----------
    Z_eval   : (N, D) array of phase-space points where Q is wanted.
               Typically the post-step support cloud Z_new.
    gp       : GPDensity or GPDensityDiff (the current density surrogate
               at time t — i.e. the surrogate fit BEFORE the y_new update
               that this Q drives).
    dt       : full time step.  The half-step backward map uses tau=-dt/2.
    dynamics : PBMEMIntDynamics carrying the model and integrator parameters.

    Returns
    -------
    Q        : (N,)        scalar correction values.
    Y        : (N, D)      backward half-step midpoints  Φ_{-dt/2}(Z_eval).
    dbarh_dR : (N, 2, 2)   ∂h̄^{αβ}/∂R evaluated at Y_R.
    """
    Z_eval = np.asarray(Z_eval, dtype=np.float64)
    if Z_eval.ndim == 1:
        Z_eval = Z_eval.reshape(1, D)
    if Z_eval.shape[1] != D:
        raise ValueError(f"Z_eval must have shape (N, {D}); got {Z_eval.shape}")

    step_fn = _get_jax_step_fn(dynamics)
    dh_fn   = _get_dh_fn(dynamics)

    half_tau = jnp.asarray(-0.5 * float(dt))
    pref     = jnp.asarray(-float(dynamics.params.hbar) / 8.0)

    if _is_density_diff(gp):
        # Centers-swap: use the CURRENT support cloud (gp_delta._Z_train)
        # for both kernels.  At the call site this is the OLD support
        # because the GP has not yet been refit with Z_new.
        Z_centers, _, sigma_f_d, ell_d = _extract_vanilla_params(gp.gp_delta)
        alpha_d   = _detach_to_numpy(gp.gp_delta._alpha)
        alpha_b   = _detach_to_numpy(gp.gp0._alpha)
        sigma_f_b = float(gp.gp0.sigma_f)
        ell_b     = np.asarray(gp.gp0.lengthscales, dtype=np.float64)
        M_rp_b    = _rp_metric_of(gp.gp0, ell_b)
        M_rp_d    = _rp_metric_of(gp.gp_delta, ell_d)

        Q_batch = _get_Q_batch_diff(step_fn, dh_fn)
        Q_j, Y_j, dh_j = Q_batch(
            jnp.asarray(Z_eval),
            jnp.asarray(Z_centers),
            jnp.asarray(alpha_b),
            jnp.asarray(np.log(sigma_f_b)),
            jnp.asarray(np.log(ell_b)),
            jnp.asarray(alpha_d),
            jnp.asarray(np.log(sigma_f_d)),
            jnp.asarray(np.log(ell_d)),
            half_tau, pref,
            jnp.asarray(M_rp_b), jnp.asarray(M_rp_d),
        )
    else:
        Z_centers, alpha, sigma_f, ell_phys = _extract_vanilla_params(gp)
        # Optionally re-solve with larger σ_n for smoother third derivatives.
        if q_sigma_n_scale > 1.0 + 1e-9 and hasattr(gp, "compute_smooth_alpha_for_q"):
            alpha = gp.compute_smooth_alpha_for_q(q_sigma_n_scale)
        M_rp = _rp_metric_of(gp, ell_phys)
        Q_batch = _get_Q_batch_vanilla(step_fn, dh_fn)
        Q_j, Y_j, dh_j = Q_batch(
            jnp.asarray(Z_eval),
            jnp.asarray(Z_centers),
            jnp.asarray(alpha),
            jnp.asarray(np.log(sigma_f)),
            jnp.asarray(np.log(ell_phys)),
            half_tau, pref,
            jnp.asarray(M_rp),
        )

    return (np.asarray(Q_j),
            np.asarray(Y_j),
            np.asarray(dh_j))


def compute_Q_at_points(
    Z_eval:   ArrayLike,
    gp,                              # GPDensity or GPDensityDiff
    dynamics: PBMEMIntDynamics,
    q_sigma_n_scale: float = 1.0,
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """
    No-pullback variant of ``compute_Q``: evaluate the intrinsic QCLE
    midpoint operator [iL'_m ρ_sur] directly at the supplied points,
    with NO internal backward half-step.

        Q_intrinsic[n] = [iL'_m ρ_sur](Z_eval[n])
                       = -(ℏ/8) Σ_{α,β} (∂h̄^{αβ}/∂R)(Z_eval[n, R])
                         · [ ∂³ρ_sur/(∂z_P ∂z_{r_β} ∂z_{r_α})
                           + ∂³ρ_sur/(∂z_P ∂z_{p_β} ∂z_{p_α}) ](Z_eval[n]).

    When to use this entry point vs. ``compute_Q``
    ----------------------------------------------
    * ``compute_Q(Z, gp, dt, dynamics)`` is for the *Eulerian post-step
      field-equation* discretization
          ρ_m(Δt; Z) = ρ_sur(X(Z)) + Δt · Q[ρ_sur](Z),
                       Q[ρ_sur](Z) := [iL'_m ρ_sur](Y(Z)),  Y = Φ_{-Δt/2}.
      Here Z is a POST-STEP grid point and the internal Y-pullback is
      part of the operator's definition.

    * ``compute_Q_at_points(Z, gp, dynamics)`` is for the *trajectory-
      level* explicit-midpoint (Heun) integration of the label ODE
          d(w_i y_i)/dt = +[iL'_m ρ̂(t)](z_i(t))
      where you want the rate AT the trajectory location itself.  In
      that setting an internal -Δt/2 backward step would misplace the
      evaluation by half a step (k₁ at z^{n-1/2} instead of z^n, k₂ at
      z^n instead of z^{n+1/2}).

    Calling ``compute_Q_at_points(Z, gp, dynamics)`` is numerically
    equivalent to ``compute_Q(Z, gp, dt=0.0, dynamics)`` — the half-step
    backward map collapses to identity at dt=0 — but the name documents
    intent for the trajectory-Heun call site.

    Parameters
    ----------
    Z_eval   : (N, D) array of phase-space points where the intrinsic
               operator value is wanted.  For Heun stages of the
               midpoint scheme these are the trajectory locations
               z_i^n (k₁) or z_i^{n+1/2} (k₂) directly.
    gp       : GPDensity or GPDensityDiff (the current density surrogate).
    dynamics : PBMEMIntDynamics carrying the model and integrator
               parameters (used only for ∂h̄/∂R at the evaluation point).

    Returns
    -------
    Q        : (N,)        scalar intrinsic operator values.
    Y        : (N, D)      identity copy of Z_eval (kept for API
                           symmetry with compute_Q's return tuple;
                           Y[n] == Z_eval[n] exactly).
    dbarh_dR : (N, 2, 2)   ∂h̄^{αβ}/∂R evaluated at Z_eval[n, R].
    """
    return compute_Q(Z_eval, gp=gp, dt=0.0, dynamics=dynamics,
                     q_sigma_n_scale=q_sigma_n_scale)



# =============================================================================
# Back-compatible class wrapper
# =============================================================================

class QCLECorrection:
    """
    Thin wrapper preserving the historical ``QCLECorrection`` import
    path while delegating the actual Q computation to ``compute_Q``.

    The previous ``QCLECorrection`` returned A, B, C tensors and required
    the caller to perform the kernel-derivative contraction in Dynamics.py.
    That coupled the operator math (chain rule through Y) to the surrogate
    representation (kernel-derivative formulas), making both error-prone.
    The new ``build`` returns only the post-contraction Q together with
    diagnostic geometry (Y, ∂h̄/∂R(Y_R)).

    .. note::
       The returned ``CorrectionData`` no longer carries A, B, C fields.
       The only in-tree caller (``Dynamics.MidpointScheme``) has been
       updated to consume ``data.Q`` directly.  External diagnostic code
       that needed A/B/C tensors should switch to the autodiff Y geometry
       from ``Monodromy.MonodromyTools.midpoint_geometry`` (which still
       returns J, H, T) plus ``GPDerivatives.rho_derivative_bundle``
       and reproduce the contraction manually if desired.
    """

    def __init__(
        self,
        dynamics:  Optional[PBMEMIntDynamics] = None,
        eps_jac:   float = 1.0e-7,
        eps_hess:  float = 1.0e-5,
        eps_third: float = 1.0e-4,
        q_sigma_n_scale: float = 1.0,
    ) -> None:
        self.dynamics = dynamics if dynamics is not None else PBMEMIntDynamics()
        self.q_sigma_n_scale = float(q_sigma_n_scale)

    def build(self, Z_eval: ArrayLike, dt: float, gp=None) -> CorrectionData:
        """
        Compute Q at the support points ``Z_eval``.

        Parameters
        ----------
        Z_eval : (N, D) phase-space points (typically the post-step cloud).
        dt     : full time step.
        gp     : GPDensity or GPDensityDiff.  REQUIRED — the previous
                 implementation built a geometry-only object that did not
                 need the GP, but the autodiff path needs ρ_sur(Y(Z)) to
                 compute Q.

        Returns
        -------
        CorrectionData with fields (Y, dbarh_dR, Q).
        """
        if gp is None:
            raise TypeError(
                "QCLECorrection.build now requires the GP surrogate as a "
                "keyword argument: `op.build(Z, dt=dt, gp=state.gp)`.  "
                "The autodiff path needs ρ_sur(Y(Z)) to compute Q."
            )
        Q, Y, dh = compute_Q(Z_eval, gp=gp, dt=float(dt), dynamics=self.dynamics)
        return CorrectionData(Y=Y, dbarh_dR=dh, Q=Q)

    def build_at_points(self, Z_eval: ArrayLike, gp=None) -> CorrectionData:
        """
        Compute the intrinsic operator value [iL'_m ρ_sur] AT the supplied
        points, with no internal backward half-step pullback.

        Use this from trajectory-level explicit-midpoint (Heun) integration
        of the label ODE  d(w·y)/dt = +Q  where you want the rate at the
        actual trajectory locations z_i^n (k₁) and z_i^{n+1/2} (k₂).

        Equivalent to ``self.build(Z_eval, dt=0.0, gp=gp)`` but the name
        documents intent at the call site.

        Parameters
        ----------
        Z_eval : (N, D) phase-space points where the intrinsic operator
                 value is wanted (NOT post-step grid points).
        gp     : GPDensity or GPDensityDiff.  REQUIRED.

        Returns
        -------
        CorrectionData with fields (Y, dbarh_dR, Q).  Note that Y == Z_eval
        exactly because there is no pullback.
        """
        if gp is None:
            raise TypeError(
                "QCLECorrection.build_at_points requires the GP surrogate as a "
                "keyword argument: `op.build_at_points(Z, gp=state.gp)`."
            )
        Q, Y, dh = compute_Q_at_points(Z_eval, gp=gp, dynamics=self.dynamics,
                                        q_sigma_n_scale=self.q_sigma_n_scale)
        return CorrectionData(Y=Y, dbarh_dR=dh, Q=Q)
