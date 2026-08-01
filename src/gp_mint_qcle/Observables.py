from __future__ import annotations

"""
Observables.py
==============

Physics observables computed against the GP surrogate density ρ̂.

All observables that reduce to polynomials in z have closed-form expressions
under the ARD-RBF kernel's tensor-product factorization:

    ⟨φ⟩ = ∫ φ(z) ρ̂(z) dz  =  Σ_i α_i ∫ φ(z) k(z, Z_i) dz,

where each 1D factor of k gives a Gaussian integral.

Convention (physical MMST symbols, γ = 1/2 SEO)
------------------------------------------------
The sampler targets the SEO Wigner density

    W_λ(r, p)  =  q(r, p) · w_λ(r, p),
    q(r, p)    =  (1/πℏ)^F exp(-Σ_μ (r_μ² + p_μ²) / ℏ),
    w_λ(r, p)  =  (2/ℏ)(r_λ² + p_λ²) - 1,

multiplied by a classical Gaussian in (R, P).

The bare weights w_α are useful diagnostics, but they are NOT physical
populations.  The physical MMST/Wigner symbol of |α><β| is

    c_{αβ}(r, p)
      = (1/2ℏ)[ r_α r_β + p_α p_β
                + i(r_α p_β - r_β p_α) - ℏ δ_{αβ} ].

Therefore the physical diabatic populations are

    P_α = <c_{αα}>
        = (1/2ℏ) <r_α² + p_α² - ℏ>,

and the physical diabatic coherence is

    ρ^el_{01} = <c_{01}>
              = (1/2ℏ) <r_0 r_1 + p_0 p_1
                         + i(r_0 p_1 - r_1 p_0)>.

The GP surrogate should enforce KKT constraints with

    <1>_{ρ̂}         = 1,      (normalization)
    <c_00 + c_11>   = 1,      (trace)
    <H>_{ρ̂}         = E_0,    (energy)

so that the reported populations/coherences are physical electronic
matrix elements in the mapping basis.

Adiabatic populations
---------------------
The adiabatic basis rotation U(R) diagonalizes h̄(R); computing
<P^{ad}_α> requires Gauss-Hermite quadrature in R and a per-node
diabatic→adiabatic rotation.  Provided as `adiabatic_populations(...)`.
"""

from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from .Mint import D, PBMEMIntDynamics
from .GP_Density import GPDensity, _as_numpy


FloatArray = NDArray[np.float64]

I_R, I_P = 0, 1
I_R0, I_R1 = 2, 3
I_P0, I_P1 = 4, 5

_MOMENT_NAMES = ("normalization", "trace", "energy")


# =============================================================================
# Common GP ingredients
# =============================================================================

def _gp_ingredients(gp):
    """
    Bundle the pieces used by every analytic moment.

    Dispatches on whether `gp` is a single GPDensity or a GPDensityDiff:

    * Single GPDensity → returns one tuple as before.
    * GPDensityDiff     → returns a LIST of tuples, one per α-component
                          (baseline transported + correction).  Callers
                          that sum `aG * f(Z)` just sum the two tuples'
                          contributions.

    This keeps the moment-math compatible with both surrogate types.

    Returns
    -------
    alpha : (N,)
    Z_tr  : (N, D)
    ell   : (D,)
    G     : scalar    = σ_f^2 · Π_d √(2π) ℓ_d
    aG    : (N,)      = alpha * G
    """
    # Density-diff: return the two ingredient tuples so callers can sum.
    if _is_density_diff(gp):
        return _gp_ingredients_diff(gp)

    if gp._alpha is None or gp._Z_train is None:
        raise RuntimeError("GP must be fitted.")
    alpha = _as_numpy(gp._alpha)
    Z_tr  = gp.raw_training_centers if hasattr(gp, "raw_training_centers") else _as_numpy(gp._Z_train)
    ell   = gp.lengthscales
    G     = (gp.sigma_f ** 2) * float(np.prod(np.sqrt(2.0 * np.pi) * ell))
    aG    = alpha * G
    return alpha, Z_tr, ell, G, aG


def _is_density_diff(gp) -> bool:
    """Recognize GPDensityDiff without a hard import dependency."""
    return (hasattr(gp, "gp0") and hasattr(gp, "gp_delta")
            and hasattr(gp, "Z0") and hasattr(gp, "y0"))


def _gp_ingredients_diff(gp) -> list:
    """
    For a density-difference surrogate, return a list of two ingredient
    tuples:
        [ (alpha_base, Z_support_t, ell_base, G_base, aG_base),
          (alpha_delta, Z_support_t, ell_delta, G_delta, aG_delta) ]

    Both rows use the same Z centers (the current support cloud) but
    different alpha/ell/sigma_f.  The baseline is evaluated via its
    frozen labels and the CURRENT support centers, because support
    points transport along the classical PBME flow — so a kernel sum
    with centers = Z_t and "alpha" = baseline GP's alpha gives the
    pulled-back baseline density by construction.

    IMPORTANT: the baseline's effective "alpha" at time t is NOT the
    baseline GP's α₀ evaluated at frozen Z₀; it's the α₀ whose kernel
    expansion on the CURRENT cloud Z_t reproduces the transported
    baseline density.  For single-step ARD-RBF kernels this is the
    same α₀ -- the kernel expansion k(z*, Z_t) @ α₀ IS the evaluation
    of rho_hat_0 at points Phi_{-t}^0(z*), since Z_t = Phi_t(Z₀).
    See GPDensityDiff._evaluate_baseline_transported.
    """
    if not gp._initial_fit_done:
        raise RuntimeError("GPDensityDiff must be fitted before moment calls.")

    # Baseline tuple — use baseline's (alpha, sigma_f, ell) but CURRENT centers
    gp0 = gp.gp0
    alpha_b = _as_numpy(gp0._alpha)
    ell_b   = gp0.lengthscales
    G_b     = (gp0.sigma_f ** 2) * float(np.prod(np.sqrt(2.0 * np.pi) * ell_b))
    aG_b    = alpha_b * G_b
    Z_tr_t  = _as_numpy(gp.gp_delta._Z_train)     # current support centers
    baseline = (alpha_b, Z_tr_t, ell_b, G_b, aG_b)

    # Correction tuple — standard GPDensity ingredients
    gpd = gp.gp_delta
    alpha_d = _as_numpy(gpd._alpha)
    ell_d   = gpd.lengthscales
    G_d     = (gpd.sigma_f ** 2) * float(np.prod(np.sqrt(2.0 * np.pi) * ell_d))
    aG_d    = alpha_d * G_d
    Z_tr_d  = gpd.raw_training_centers if hasattr(gpd, "raw_training_centers") else _as_numpy(gpd._Z_train)
    correction = (alpha_d, Z_tr_d, ell_d, G_d, aG_d)

    return [baseline, correction]


def _safe_norm_from_gp(gp: GPDensity) -> float:
    """Density normalization used to convert raw integrals to normalized expectations."""
    if getattr(gp, "_is_product", False):
        from .ProductMoments import product_norm_raw
        norm = product_norm_raw(gp)
        return norm if abs(norm) > 1.0e-15 else 1.0
    norm = float(gp.compute_moment_values()["normalization"])
    return norm if abs(norm) > 1.0e-15 else 1.0


# =============================================================================
# Normalization, trace, energy (the KKT constraints)
# =============================================================================

def kkt_moments(gp: GPDensity) -> Dict[str, float]:
    """
    Density-normalized physical moments under the current GP, plus their
    unnormalized counterparts for diagnostics.

    Returns
    -------
    normalization        : ∫ ρ̂_sn = 1  (the SELF-NORMALIZED density's norm,
                           = 1 by construction — consistent with how trace
                           and energy below are reported)
    trace                : (∫ (c_00+c_11) ρ̂) / (∫ ρ̂)    (normalized)
    energy               : (∫ H ρ̂) / (∫ ρ̂)              (normalized)
    normalization_raw    : ∫ ρ̂  (= A_norm·α — the RAW kernel integral;
                           on focused-manifold support this is a kernel-
                           smearing artifact ≈1.6-4, NOT a physical norm)
    trace_raw            : ∫ (c_00+c_11) ρ̂              (= A_trace·α)
    energy_raw           : ∫ H ρ̂                        (= A_energy·α)

    Why normalization is reported as the self-normalized value (=1)
    --------------------------------------------------------------
    Every physical quantity the pipeline consumes is self-normalized:
    observables come from the cloud Riemann sum (trace = Σω·b = 1 by IS
    construction) and the QCLE source Q is LOCAL and LINEAR in α, so the
    overall scale of ρ̂ cancels.  The density actually represented for
    moment purposes is therefore ρ̂_sn = ρ̂ / ∫ρ̂dz, whose norm is exactly
    1 — the same density whose trace and energy are reported on the two
    lines above.  Reporting `normalization = 1` is thus the consistent
    reading, not a cosmetic override: it is the norm of the self-
    normalized density the trace/energy already describe.

    The raw kernel integral ∫ρ̂dz is preserved as `normalization_raw`
    for diagnostics.  On focused-manifold support it sits at ≈1.6 and
    drifts upward as the cloud spreads; that drift is a property of the
    Gaussian kernel smearing a measure-zero manifold, and is independent
    of fit quality (α interpolates the labels exactly under apply_kkt=
    False, so fit_r²≈1 regardless of where the raw integral sits).
    """
    if getattr(gp, "_is_product", False):
        # Product surrogate rho_hat = g*mu: the vanilla path below would
        # (via __getattr__ delegation) integrate the inner mu WITHOUT the
        # profile g — silently wrong by orders of magnitude.  Route to the
        # closed-form product engine instead (2026-07-10).
        from .ProductMoments import product_kkt_moments
        return product_kkt_moments(gp)
    raw = {k: float(v) for k, v in gp.compute_moment_values().items()}
    norm = raw["normalization"] if abs(raw["normalization"]) > 1.0e-15 else 1.0
    return {
        # Self-normalized density's norm: 1 by construction, consistent
        # with the self-normalized trace/energy reported below.
        "normalization": raw["normalization"] / norm,
        "trace": raw["trace"] / norm,
        "energy": raw["energy"] / norm,
        # Raw kernel integral, kept for diagnostics (the old
        # `normalization` field).  Watch this — not `normalization` —
        # if you want to see the kernel-smearing integral drift.
        "normalization_raw": raw["normalization"],
        "trace_raw": raw["trace"],
        "energy_raw": raw["energy"],
    }


# =============================================================================
# Nuclear moments
# =============================================================================

def nuclear_moments(gp) -> Dict[str, float]:
    r"""
    ⟨R⟩, ⟨R²⟩, Var(R),  ⟨P⟩, ⟨P²⟩, Var(P).

    Closed form:
        ⟨R⟩   = Σ_i α_i G · Z_{i, R}
        ⟨R²⟩  = Σ_i α_i G · (Z_{i, R}² + ℓ_R²)
    and similarly for P.

    Split-aware: if `gp` is a GPDensityDiff, we sum the baseline-transported
    and correction contributions before dividing by the total normalization.
    """
    if getattr(gp, "_is_product", False):
        from .ProductMoments import product_nuclear_moments
        return product_nuclear_moments(gp)
    ingredients = _gp_ingredients(gp)
    if isinstance(ingredients, tuple):
        ingredients_list = [ingredients]
    else:
        ingredients_list = ingredients

    R_raw = 0.0; P_raw = 0.0
    R_sq_raw = 0.0; P_sq_raw = 0.0
    for (_, Z_tr, ell, _, aG) in ingredients_list:
        R_raw    += float(np.sum(aG * Z_tr[:, I_R]))
        P_raw    += float(np.sum(aG * Z_tr[:, I_P]))
        R_sq_raw += float(np.sum(aG * (Z_tr[:, I_R] ** 2 + ell[I_R] ** 2)))
        P_sq_raw += float(np.sum(aG * (Z_tr[:, I_P] ** 2 + ell[I_P] ** 2)))

    norm = _safe_norm_from_gp(gp)
    R_mean = R_raw / norm
    P_mean = P_raw / norm
    R_sq = R_sq_raw / norm
    P_sq = P_sq_raw / norm
    return {
        "R_mean": R_mean, "P_mean": P_mean,
        "R_sq":   R_sq,   "P_sq":   P_sq,
        "R_var":  R_sq - R_mean ** 2,
        "P_var":  P_sq - P_mean ** 2,
    }


# =============================================================================
# Mapping moments (r_α, p_α quadratic/cross)
# =============================================================================

def _quadratic_mapping_moments(gp) -> Dict[str, float]:
    r"""
    All ⟨r_α r_β⟩, ⟨p_α p_β⟩, ⟨r_α p_β⟩ moments.
    α, β ∈ {0, 1}.

    Also returns `mapping_radius_sq` = ⟨r_0² + p_0² + r_1² + p_1²⟩, the
    mapping-sector radius squared under ρ̂.  This is a sharp physics check:

    * The MInt propagator conserves r_0² + p_0² + r_1² + p_1² along every
      single trajectory (exact algebraic invariant of the mapping rotation).
    * Therefore its expectation under ANY propagated density — PBME or
      corrected QCLE — must equal its initial value for all time.
    * For the signed-SEO initial state on λ=0, the exact initial value is
      ⟨r²+p²⟩ = 2·N_s·ℏ = 4ℏ  (N_s = 2 states).

    Any deviation of ⟨r²+p²⟩ from 4ℏ is a direct, unambiguous signal that
    the surrogate has lost fidelity — it is not a physics signal and not a
    convention issue.  This is particularly useful as a trust indicator
    under signed-SEO sampling, where ESS collapse can silently drive all
    density-weighted expectations into the noise floor without changing
    the reported R² on training points.

    For density-difference surrogates, each moment is the SUM of the
    baseline-transported moment and the correction moment, divided by
    the TOTAL normalization (likewise a sum).  This preserves the
    identity that a moment of (baseline + correction) equals baseline-
    moment + correction-moment, while the normalization in the
    denominator is the TOTAL integral of rho_hat.
    """
    if getattr(gp, "_is_product", False):
        from .ProductMoments import product_quadratic_mapping_moments
        return product_quadratic_mapping_moments(gp)
    ingredients = _gp_ingredients(gp)

    # Normalize to list-of-tuples form so the arithmetic is uniform
    if isinstance(ingredients, tuple):
        ingredients_list = [ingredients]
    else:
        ingredients_list = ingredients

    # Accumulators (raw, unnormalized)
    sq_vals    = {I_R0: 0.0, I_R1: 0.0, I_P0: 0.0, I_P1: 0.0}
    cross_vals = {}  # keyed by sorted (d1, d2)
    total_int  = 0.0

    def _add_from_tuple(alpha, Z_tr, ell, G, aG):
        nonlocal total_int
        total_int += float(np.sum(aG))
        for d in (I_R0, I_R1, I_P0, I_P1):
            sq_vals[d] += float(np.sum(aG * (Z_tr[:, d] ** 2 + ell[d] ** 2)))
        pairs = [(I_R0, I_R1), (I_P0, I_P1),
                 (I_R0, I_P0), (I_R1, I_P1),
                 (I_R0, I_P1), (I_R1, I_P0)]
        for (d1, d2) in pairs:
            key = (min(d1, d2), max(d1, d2))
            cross_vals[key] = cross_vals.get(key, 0.0) \
                + float(np.sum(aG * Z_tr[:, d1] * Z_tr[:, d2]))

    for tup in ingredients_list:
        _add_from_tuple(*tup)

    norm = _safe_norm_from_gp(gp)  # handles both single-GP and diff-GP

    def _val(raw: float) -> float:
        return raw / norm

    def _cross(d1, d2) -> float:
        key = (min(d1, d2), max(d1, d2))
        return cross_vals[key] / norm

    r0_sq = _val(sq_vals[I_R0]); r1_sq = _val(sq_vals[I_R1])
    p0_sq = _val(sq_vals[I_P0]); p1_sq = _val(sq_vals[I_P1])
    return {
        "r0_sq":  r0_sq, "r1_sq":  r1_sq,
        "p0_sq":  p0_sq, "p1_sq":  p1_sq,
        "mapping_radius_sq": r0_sq + r1_sq + p0_sq + p1_sq,
        "r0_r1":  _cross(I_R0, I_R1),
        "p0_p1":  _cross(I_P0, I_P1),
        "r0_p0":  _cross(I_R0, I_P0),
        "r1_p1":  _cross(I_R1, I_P1),
        "r0_p1":  _cross(I_R0, I_P1),
        "r1_p0":  _cross(I_R1, I_P0),
    }


def _electronic_symbol_moments(
    gp: GPDensity,
    hbar: Optional[float] = None,
) -> Dict[str, float]:
    r"""
    Physical MMST/Wigner symbols c_{αβ} for the electronic density matrix.

        c_{αβ}(r,p)
          = (1/2ℏ)[ r_α r_β + p_α p_β
                    + i(r_α p_β - r_β p_α) - ℏ δ_{αβ} ].

    Returns the physical diabatic populations/coherences integrated against ρ̂.
    """
    hbar = 1.0 if hbar is None else float(hbar)
    qm = _quadratic_mapping_moments(gp)

    rho00 = (qm["r0_sq"] + qm["p0_sq"] - hbar) / (2.0 * hbar)
    rho11 = (qm["r1_sq"] + qm["p1_sq"] - hbar) / (2.0 * hbar)
    rho01_re = (qm["r0_r1"] + qm["p0_p1"]) / (2.0 * hbar)
    rho01_im = (qm["r1_p0"] - qm["r0_p1"]) / (2.0 * hbar)

    return {
        "rho00": rho00,
        "rho11": rho11,
        "rho01_re": rho01_re,
        "rho01_im": rho01_im,
        "trace": rho00 + rho11,
    }


# =============================================================================
# Mapping weights  w_α  (bare signed SEO estimator)
# =============================================================================

def mapping_weights(gp: GPDensity,
                    hbar: Optional[float] = None) -> Dict[str, float]:
    r"""
    ⟨w_α⟩ = ⟨(2/ℏ)(r_α² + p_α²) - 1⟩ for α = 0, 1, and their difference.

    For a signed-SEO sampled initial state λ under the raw SEO target, the
    mapping moment is ⟨r_λ² + p_λ²⟩ = 3ℏ (active state) and
    ⟨r_{1-λ}² + p_{1-λ}²⟩ = ℏ (inactive state), so the bare weights are

        ⟨w_λ⟩_{W_λ}     = (2/ℏ)·3ℏ - 1 = 5,
        ⟨w_{1-λ}⟩_{W_λ} = (2/ℏ)·ℏ  - 1 = 1.

    These are bare mapping weights, not physical populations.  The
    physical MMST/Wigner population is P_α = (⟨r_α² + p_α²⟩ - ℏ)/(2ℏ),
    which gives P_λ = 1 and P_{1-λ} = 0 at t = 0 as expected.
    """
    hbar = 1.0 if hbar is None else float(hbar)
    qm = _quadratic_mapping_moments(gp)

    w0 = (2.0 / hbar) * (qm["r0_sq"] + qm["p0_sq"]) - 1.0
    w1 = (2.0 / hbar) * (qm["r1_sq"] + qm["p1_sq"]) - 1.0
    return {"w0": w0, "w1": w1, "w_diff": w0 - w1, "w_sum": w0 + w1}


# =============================================================================
# Diabatic populations (physical MMST symbols: P_α = <c_{αα}>)
# =============================================================================

def diabatic_populations_from_cloud(
    Z: FloatArray,
    omega: FloatArray,
    rho: FloatArray,
    hbar: Optional[float] = None,
) -> Dict[str, float]:
    r"""
    Cloud Riemann-sum estimator of the diabatic populations.

    The cloud is treated as a value-on-points discretization of the density,
    not as an importance sample of a probability distribution.  Each support
    point z_i carries a geometric measure ω_i (frozen at t=0 from the initial
    sampling) and a density value ρ(z_i, t).  Any phase-space integral is
    then a Riemann sum:

        ⟨A⟩(t) = ∫ A(z) ρ(z, t) dz  ≈  Σ_i ω_i A(z_i(t)) ρ(z_i, t).

    For the diabatic populations, A = c_{αα}, so

        P_α(t) = Σ_i ω_i c_{αα}(z_i(t)) ρ(z_i, t).

    There is NO division by Σ ω_i ρ_i.  The trace identity
    ∫ (c_{00} + c_{11}) ρ dz = 1 is enforced by the KKT moment projection on
    the GP that produces ρ — it is NOT a normalization to be re-imposed at
    the cloud level.  Re-imposing it via self-normalization would inject
    spurious time-dependence whenever Σ ρ_i drifts (which it does under the
    midpoint scheme as soon as Q kicks in).

    For PBME, ρ(z_i, t) = ρ_0(z_i^0) is frozen along trajectories (Liouville
    + frozen labels), so this estimator reduces to a constant × symbol-sum:

        P_α^PBME(t) = Σ_i ω_i ρ_0(z_i^0) c_{αα}(z_i(t)),

    which is conserved exactly under MInt for the trace combination
    c_{00} + c_{11} because the mapping rotation is a Casimir of |r|² + |p|².

    Parameters
    ----------
    Z     : (N, 6) phase-space coordinates at current time t.
    omega : (N,)   frozen geometric measure ω_i = 1/(N q(z_i^0)) from the
                   initial sampling.  Sign-definite (positive) and never
                   updated after t=0.
    rho   : (N,)   density values at the support points at the current time:
                   ρ_0(z_i^0) for PBME, y_i(t) for midpoint.
    hbar  : float, optional (defaults to 1.0).
    """
    hb = 1.0 if hbar is None else float(hbar)
    Z     = np.asarray(Z,     dtype=np.float64).reshape(-1, 6)
    omega = np.asarray(omega, dtype=np.float64).reshape(-1)
    rho   = np.asarray(rho,   dtype=np.float64).reshape(-1)
    N = Z.shape[0]
    if omega.shape[0] != N or rho.shape[0] != N:
        raise ValueError(
            f"Z, omega, rho must all have length {N}; "
            f"got {omega.shape[0]} and {rho.shape[0]}."
        )
    # Coordinate layout: (R, P, r_0, r_1, p_0, p_1)
    r0 = Z[:, 2]; r1 = Z[:, 3]
    p0 = Z[:, 4]; p1 = Z[:, 5]
    c00 = (r0**2 + p0**2 - hb) / (2.0 * hb)
    c11 = (r1**2 + p1**2 - hb) / (2.0 * hb)

    or_ = omega * rho                                   # (N,)
    P0 = float(np.dot(or_, c00))
    P1 = float(np.dot(or_, c11))
    return {"P0": P0, "P1": P1,
            "P_sum": P0 + P1,
            "P_diff": P0 - P1}


def diabatic_populations_from_labels(
    Z: FloatArray,
    y: FloatArray,
    omega: FloatArray,
    hbar: Optional[float] = None,
) -> Dict[str, float]:
    r"""
    Cloud Riemann-sum estimator using the QCLE-corrected labels y_i(t) as the
    density values at the support points.

    Identical mathematical form to :func:`diabatic_populations_from_cloud`:

        P_α(t) = Σ_i ω_i c_{αα}(z_i(t)) y_i(t).

    The only thing that distinguishes this entry point is what is being passed
    as ρ.  Under PBME, ρ = ρ_0(z_i^0) (frozen initial density).  Under the
    midpoint scheme,

        y_i(t) = ρ_0(z_i^0) - Σ_{steps} Δt · Q(z_i(t')),

    which is the propagated density value at z_i carrying the QCLE correction.
    Use this entry point when y has been updated by the midpoint operator;
    use :func:`diabatic_populations_from_cloud` when the density values are
    just the frozen initial labels.

    Parameters
    ----------
    Z     : (N, 6)  current phase-space coordinates z_i(t).
    y     : (N,)    current density values ρ(z_i, t).
    omega : (N,)    frozen geometric measure ω_i = 1/(N q(z_i^0)).
    hbar  : float, optional.

    Notes
    -----
    There is intentionally no normalization by Σ ω_i y_i.  The trace identity
    is enforced by the KKT projection on the GP that produces ρ; reimposing
    it here would mask any drift in Σ ω_i y_i (which IS itself the diagnostic
    for whether the QCLE correction respects normalization at the cloud level).
    """
    hb = 1.0 if hbar is None else float(hbar)
    Z     = np.asarray(Z,     dtype=np.float64).reshape(-1, 6)
    y     = np.asarray(y,     dtype=np.float64).reshape(-1)
    omega = np.asarray(omega, dtype=np.float64).reshape(-1)
    N = Z.shape[0]
    if y.shape[0] != N or omega.shape[0] != N:
        raise ValueError(
            f"Z, y, omega must all have length {N}; "
            f"got {y.shape[0]} and {omega.shape[0]}."
        )

    r0 = Z[:, 2]; r1 = Z[:, 3]
    p0 = Z[:, 4]; p1 = Z[:, 5]
    c00 = (r0**2 + p0**2 - hb) / (2.0 * hb)
    c11 = (r1**2 + p1**2 - hb) / (2.0 * hb)

    oy = omega * y                                      # (N,)
    P0 = float(np.dot(oy, c00))
    P1 = float(np.dot(oy, c11))
    return {"P0": P0, "P1": P1,
            "P_sum": P0 + P1,
            "P_diff": P0 - P1}


def diabatic_populations_from_gp(gp: GPDensity,
                                 hbar: Optional[float] = None) -> Dict[str, float]:
    r"""
    GP kernel-integral diabatic populations (DIAGNOSTIC ONLY).

    Uses analytic kernel-integral identities on the surrogate's α
    coefficients.  This path is **fragile**: when the cloud spreads and
    the surrogate can no longer faithfully represent ρ(z) off-support,
    the integrals can swing wildly (seen in production runs: P₀ reaching
    1.48, P₁ reaching -0.48 while the cloud-weighted estimator stayed at
    P₀ ≈ 0.94, P₁ ≈ 0.02).

    Kept for comparison with the cloud estimator, not as the primary
    population readout.
    """
    em = _electronic_symbol_moments(gp, hbar=hbar)
    return {"P0": em["rho00"], "P1": em["rho11"],
            "P_sum": em["trace"],
            "P_diff": em["rho00"] - em["rho11"]}


def diabatic_populations(gp: GPDensity,
                         hbar: Optional[float] = None,
                         Z: Optional[FloatArray] = None,
                         omega: Optional[FloatArray] = None,
                         y: Optional[FloatArray] = None,
                         ) -> Dict[str, float]:
    r"""
    Physical diabatic populations P_α = ⟨c_{αα}⟩, with cloud-as-discretization
    semantics (no IS reweighting).

    Dispatch hierarchy:

    1. Cloud Riemann sum with live density values:
       if Z, y, and omega are all supplied.
       Used for the midpoint scheme (and reduces to the PBME estimator
       when y is the frozen initial density).
       Formula:  P_α = Σ_i ω_i y_i c_{αα}(z_i(t)).

    2. GP kernel-integral (analytic, KKT-constrained):
       if cloud arguments are not supplied.
       Formula:  P_α = ∫ c_{αα}(z) ρ̂(z) dz / ∫ ρ̂(z) dz, where the
       denominator is fixed to 1 by the normalization KKT constraint.
       Includes the QCLE correction implicitly through the updated α.
    """
    if Z is not None and y is not None and omega is not None:
        return diabatic_populations_from_labels(Z, y, omega, hbar=hbar)
    return diabatic_populations_from_gp(gp, hbar=hbar)


# =============================================================================
# Adiabatic populations via Gauss-Hermite quadrature in R
# =============================================================================

def _diabatic_to_adiabatic(dynamics: PBMEMIntDynamics, R: FloatArray):
    """
    For each R value, diagonalize h̄(R) analytically as a real-symmetric 2×2,
    returning a GLOBALLY SMOOTH adiabatic frame U(R).

    Previous implementation
    -----------------------
    Used np.linalg.eigh(h) and fixed signs via sign(U[:, 0, 0]).  At avoided
    crossings — where the adiabatic rotation angle passes through π/4 and
    U[:, 0, 0] crosses zero — that sign convention flips discretely from one
    R-node to the next, introducing a spurious O(1) discontinuity in the
    reported adiabatic populations exactly where the nonadiabatic physics
    lives.

    Analytic 2×2 replacement
    ------------------------
    For h = [[h00, h01], [h01, h11]], let
        Δ = (h11 - h00) / 2,
        θ = (1/2) atan2(h01, Δ).
    Then the eigenvectors are
        U = [[cos θ, -sin θ], [sin θ, cos θ]],
    with eigenvalues (h00+h11)/2 ∓ sqrt(Δ² + h01²).
    `atan2` is continuous everywhere except at h01 = Δ = 0 (degenerate), so
    this frame is smooth in R as long as the avoided crossing has a nonzero
    off-diagonal coupling — which is exactly the assumption of the Tully
    dual-crossing model (h01 = C·exp(-D R²) > 0 everywhere).
    """
    R = np.atleast_1d(np.asarray(R, dtype=np.float64))
    _, h, _, _ = dynamics._frozen_R_objects(R)  # h: (M, 2, 2)

    h00 = h[:, 0, 0]
    h01 = h[:, 0, 1]
    h11 = h[:, 1, 1]
    half_diff = 0.5 * (h11 - h00)

    # Continuous rotation angle.  atan2 is smooth wherever (h01, half_diff)
    # does not pass through the origin; that would require simultaneously
    # vanishing coupling and gap, i.e. a true conical intersection.
    theta = 0.5 * np.arctan2(h01, half_diff)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    U = np.empty((R.shape[0], 2, 2), dtype=np.float64)
    U[:, 0, 0] = cos_t
    U[:, 0, 1] = -sin_t
    U[:, 1, 0] = sin_t
    U[:, 1, 1] = cos_t

    # Matching eigenvalues (ascending, i.e. E[..., 0] is the lower branch).
    half_sum = 0.5 * (h00 + h11)
    disc = np.sqrt(half_diff * half_diff + h01 * h01)
    eigvals = np.stack([half_sum - disc, half_sum + disc], axis=-1)
    return U, eigvals


def adiabatic_populations(
    gp, dynamics: PBMEMIntDynamics,
    n_gh: int = 16, hbar: Optional[float] = None,
) -> Dict[str, float]:
    r"""
    Adiabatic diagonal populations

        ρ^{ad}_{αα}(R) = Σ_{ab} U_{aα}(R) U_{bα}(R) ρ^{el}_{ab}(R),

    with ρ^{el}_{ab}(R) = ⟨ diabatic Wigner symbol Ô_{ab} · δ(R - R') ⟩_{ρ̂}.

    For (diagonal a=b) we use the physical MMST symbol
        c_{aa} = (1/2ℏ)(r_a² + p_a² - ℏ).
    For (off-diagonal a≠b) we use
        Re c_{01} = (1/2ℏ)(r_0 r_1 + p_0 p_1).

    The R integral is done by Gauss-Hermite quadrature, using the fact that
    the GP kernel factorizes in R with lengthscale ℓ_R.

    Split-aware: for a GPDensityDiff, raw numerators are computed per
    ingredient tuple (baseline-transported + correction) and summed
    before dividing by the total normalization.
    """
    if getattr(gp, "_is_product", False):
        from .ProductMoments import product_adiabatic_populations
        return product_adiabatic_populations(gp, dynamics, n_gh=n_gh,
                                             hbar=hbar)
    hbar = 1.0 if hbar is None else float(hbar)
    ingredients = _gp_ingredients(gp)
    if isinstance(ingredients, tuple):
        ingredients_list = [ingredients]
    else:
        ingredients_list = ingredients

    # Accumulate raw (unnormalized) adiabatic sums across all ingredient tuples
    Pad_0_raw = 0.0
    Pad_1_raw = 0.0

    x_gh, w_gh = np.polynomial.hermite.hermgauss(n_gh)   # (n_gh,)
    gh_weights = w_gh / np.sqrt(np.pi)                  # (n_gh,)
    sqrt2 = np.sqrt(2.0)

    for (alpha_vec, Z_tr, ell, _G, _aG) in ingredients_list:
        N = Z_tr.shape[0]
        ell_R = ell[I_R]

        R_nodes = Z_tr[:, I_R:I_R+1] + sqrt2 * ell_R * x_gh[None, :]
        R_flat = R_nodes.reshape(-1)
        U, _ = _diabatic_to_adiabatic(dynamics, R_flat)       # (N*n_gh, 2, 2)

        M2_r0 = (Z_tr[:, I_R0] ** 2 + ell[I_R0] ** 2)
        M2_r1 = (Z_tr[:, I_R1] ** 2 + ell[I_R1] ** 2)
        M2_p0 = (Z_tr[:, I_P0] ** 2 + ell[I_P0] ** 2)
        M2_p1 = (Z_tr[:, I_P1] ** 2 + ell[I_P1] ** 2)

        e00 = (M2_r0 + M2_p0 - hbar) / (2.0 * hbar)
        e11 = (M2_r1 + M2_p1 - hbar) / (2.0 * hbar)
        e01 = (Z_tr[:, I_R0] * Z_tr[:, I_R1] + Z_tr[:, I_P0] * Z_tr[:, I_P1]) / (2.0 * hbar)

        U_mat = U.reshape(N, n_gh, 2, 2)
        U0 = U_mat[:, :, 0, :]
        U1 = U_mat[:, :, 1, :]
        rho_ad_alpha = (U0 * U0 * e00[:, None, None]
                        + 2.0 * U0 * U1 * e01[:, None, None]
                        + U1 * U1 * e11[:, None, None])

        per_support_per_alpha = np.sum(
            rho_ad_alpha * gh_weights[None, :, None], axis=1
        )                                                      # (N, 2)

        # _aG = alpha * G for each training point i — the correct prefactor.
        Pad = _aG[:, None] * per_support_per_alpha           # (N, 2)
        Pad_0_raw += float(np.sum(Pad[:, 0]))
        Pad_1_raw += float(np.sum(Pad[:, 1]))

    norm = _safe_norm_from_gp(gp)
    Pad_0 = Pad_0_raw / norm
    Pad_1 = Pad_1_raw / norm

    return {
        "Pad_0": Pad_0, "Pad_1": Pad_1,
        "Pad_sum":  Pad_0 + Pad_1,
        "Pad_diff": Pad_0 - Pad_1,
    }


# =============================================================================
# Diabatic coherences  ρ^el_{01}
# =============================================================================

def diabatic_coherences(gp: GPDensity,
                        hbar: Optional[float] = None) -> Dict[str, float]:
    r"""
    Diabatic electronic coherence between states 0 and 1.

        Re ρ^el_{01} = (1/2ℏ) ⟨r_0 r_1 + p_0 p_1⟩
        Im ρ^el_{01} = (1/2ℏ) ⟨r_0 p_1 - p_0 r_1⟩
        |ρ^el_{01}|  = sqrt(Re² + Im²)

    This uses the physical MMST/Wigner symbol c_{01}, consistent with the
    diagonal populations c_{00}, c_{11}.
    """
    hbar = 1.0 if hbar is None else float(hbar)
    qm = _quadratic_mapping_moments(gp)
    re = (qm["r0_r1"] + qm["p0_p1"]) / (2.0 * hbar)
    im = (qm["r1_p0"] - qm["r0_p1"]) / (2.0 * hbar)
    return {"coh_re": re, "coh_im": im, "coh_abs": float(np.hypot(re, im))}


# =============================================================================
# Local energy and ensemble energy (along support-point trajectories)
# =============================================================================

def support_point_energies(dynamics: PBMEMIntDynamics,
                           Z: FloatArray,
                           rho: Optional[FloatArray] = None,
                           omega: Optional[FloatArray] = None) -> Dict[str, float]:
    r"""
    Energy statistics on the support-point cloud.

    Two distinct quantities, both useful but for different purposes:

    1. ``E_traj``  — equiprobable trajectory mean (1/N) Σ H(z_i(t)).
       Under PBME each MInt trajectory conserves H exactly, so this is
       conserved to floating-point precision per step regardless of labels,
       weights, or GP state.  This is the integrator sanity check.  It is
       NOT the physical ⟨H⟩.

    2. ``E_density`` — density-weighted Riemann sum  ⟨H⟩(t) = Σ_i ω_i H(z_i(t)) ρ(z_i, t).
       This is the physical energy expectation of the propagated density.
       Computed only when both ``omega`` and ``rho`` are supplied.  No
       self-normalization (the density's normalization is enforced by the
       KKT projection on the GP).

    ``E_std``, ``E_min``, ``E_max`` are unweighted trajectory-level statistics
    of the per-point Hamiltonian values, useful as MInt conservation checks.
    """
    E = np.asarray(dynamics.energy(Z), dtype=np.float64)
    E_traj = float(np.mean(E))

    if (omega is not None) and (rho is not None):
        omega = np.asarray(omega, dtype=np.float64).reshape(-1)
        rho_  = np.asarray(rho,   dtype=np.float64).reshape(-1)
        if omega.shape[0] != E.shape[0] or rho_.shape[0] != E.shape[0]:
            raise ValueError("omega, rho, Z must all have the same length.")
        wrho = omega * rho_
        norm = float(np.sum(wrho))
        E_density = float(np.dot(wrho, E))
        E_density_sn = E_density / (norm if abs(norm) > 1.0e-15 else 1.0)
    else:
        norm = float("nan")
        E_density = float("nan")
        E_density_sn = float("nan")

    return {
        "E_traj":       E_traj,       # equiprobable mean — PBME conservation check
        "E_density":    E_density,    # raw density-weighted Σ ω_i ρ_i H_i
        "E_density_sn": E_density_sn, # normalized physical ⟨H⟩
        "E_density_norm": norm,
        "E_std":        float(np.std(E)),
        "E_min":        float(np.min(E)),
        "E_max":        float(np.max(E)),
    }


# =============================================================================
# Correction magnitude statistics (midpoint scheme only)
# =============================================================================

def correction_statistics(Q: Optional[FloatArray],
                          y: FloatArray,
                          dt: float,
                          omega: Optional[FloatArray] = None,
                          denominator_rtol: float = np.sqrt(np.finfo(float).eps)) -> Dict[str, float]:
    """
    Aggregate the per-point correction Q_i = (i𝓛' ρ_n)(Y_i) into scalar
    diagnostics for comparison plots.

    Returned keys deliberately include both the raw-Q and dt·Q statistics so
    Dynamics.py can log either the intrinsic pulled-back operator magnitude or
    the actually applied one-step update after any clipping.

    If Q is None (PBME baseline), all values are zero.

    Parameters
    ----------
    Q     : per-point QCLE correction values, shape (N,), or None for PBME.
    y     : live label vector, shape (N,).
    dt    : time step.
    omega : optional IS geometric measure weights ω_i = 1/(N·q(z_i)).
            When provided, density-weighted diagnostics use ω_i·y_i as
            the cloud estimator weight instead of y_i alone, giving the
            proper IS-corrected integral ∫ Q(z) ρ(z) dz ≈ Σ_i ω_i y_i Q_i.
    """
    if Q is None:
        return {
            "q_rms": 0.0,
            "q_max": 0.0,
            "q_abs_sum": 0.0,
            "dtq_rms": 0.0,
            "dtq_max": 0.0,
            "dq_over_y_rms": 0.0,
            "dq_over_y_max": 0.0,
            # density-weighted diagnostics (zero when there is no correction)
            "q_y_weighted_mean": 0.0,
            "q_y_weighted_rms":  0.0,
            "q_sum_yc":          0.0,
            "q_weight_denominator": 0.0,
            "q_weight_denominator_threshold": 0.0,
            "q_weighted_mean_defined": 0.0,
        }

    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    Q_arr = np.asarray(Q, dtype=np.float64).reshape(-1)
    y_scale = np.abs(y_arr) + 1.0e-30
    y_rms = float(np.sqrt(np.mean(y_arr ** 2)) + 1.0e-30)
    dQ = float(dt) * Q_arr

    dq_over_y_pointwise = np.abs(dQ) / y_scale
    dq_over_y_rms = float(np.sqrt(np.mean(dQ ** 2)) / y_rms)

    # ── density-weighted diagnostics ─────────────────────────────────────────
    # When omega (IS geometric measure) is provided, use ω_i·y_i as the
    # cloud-estimator weight so that Σ_i ω_i y_i Q_i approximates ∫Q·ρ dz.
    # Without omega, fall back to y_i weights (original behavior).
    if omega is not None:
        omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
        w_arr = omega_arr * y_arr          # IS-weighted labels
        w_abs_arr = omega_arr * np.abs(y_arr)
    else:
        w_arr     = y_arr
        w_abs_arr = np.abs(y_arr)

    # Signed sum: reproduces  ∫ Q(z) ρ(z) dz  ≈ Σ_i w_i Q_i
    q_sum_yc = float(np.dot(w_arr, Q_arr))

    # Signed-weighted mean:  (Σ_i w_i Q_i) / (Σ_i w_i)
    w_sum = float(np.sum(w_arr))
    # A scale-aware guard is essential for signed SEO clouds.  Testing only
    # against an absolute 1e-30 threshold declares a catastrophically
    # cancelled denominator valid.  tau_N scales with total absolute mass,
    # and the valid/invalid flag is saved so plots never connect values across
    # an undefined normalized observable.
    w_sum_threshold = max(1.0e-15,
                          float(denominator_rtol) * float(np.sum(np.abs(w_arr))))
    weighted_mean_defined = bool(np.isfinite(w_sum) and abs(w_sum) > w_sum_threshold)
    if weighted_mean_defined:
        q_y_weighted_mean = q_sum_yc / w_sum
    else:
        q_y_weighted_mean = float("nan")

    # |w|-weighted RMS:  √( Σ_i |w_i| Q_i² / Σ_i |w_i| )
    w_abs_sum = float(np.sum(w_abs_arr))
    if w_abs_sum > 0.0:
        q_y_weighted_rms = float(
            np.sqrt(np.dot(w_abs_arr, Q_arr ** 2) / w_abs_sum)
        )
    else:
        q_y_weighted_rms = float("nan")

    return {
        "q_rms":    float(np.sqrt(np.mean(Q_arr ** 2))),
        "q_max":    float(np.max(np.abs(Q_arr))),
        "q_abs_sum": float(np.sum(np.abs(Q_arr))),
        "dtq_rms":  float(np.sqrt(np.mean(dQ ** 2))),
        "dtq_max":  float(np.max(np.abs(dQ))),
        "dq_over_y_rms": dq_over_y_rms,
        "dq_over_y_max": float(np.max(dq_over_y_pointwise)),
        # density-weighted diagnostics
        "q_y_weighted_mean": q_y_weighted_mean,
        "q_y_weighted_rms":  q_y_weighted_rms,
        "q_sum_yc":          q_sum_yc,
        "q_weight_denominator": w_sum,
        "q_weight_denominator_threshold": w_sum_threshold,
        "q_weighted_mean_defined": float(weighted_mean_defined),
    }




# =============================================================================
# Signed-weight / cancellation diagnostics on the carried label vector y
# =============================================================================

def signed_weight_diagnostics(y: FloatArray) -> Dict[str, float]:
    """Diagnostics of the signed carried label vector y_i used in the fit."""
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(y_arr.size)
    if n == 0:
        return {
            "weight_sum": float("nan"),
            "abs_weight_sum": float("nan"),
            "ess": float("nan"),
            "ess_frac": float("nan"),
            "abs_ess": float("nan"),
            "abs_ess_frac": float("nan"),
            "cancel_ratio": float("nan"),
            "neg_frac": float("nan"),
            "pos_frac": float("nan"),
        }

    S = float(np.sum(y_arr))
    S_abs = float(np.sum(np.abs(y_arr)))
    S2 = float(np.sum(y_arr * y_arr))
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
        "neg_frac": float(np.mean(y_arr < 0.0)),
        "pos_frac": float(np.mean(y_arr > 0.0)),
    }


def cloud_estimator_diagnostics(omega: Optional[FloatArray],
                                y:     FloatArray) -> Dict[str, float]:
    r"""
    Diagnostics for the cloud Riemann-sum estimator c_i = ω_i y_i.

    Captures the variance regime that actually drives the noise in
    ⟨A⟩(t) = Σ_i c_i A(z_i(t)):

      sum_c        = Σ_i c_i  (≈ 1 by IS construction at t=0; drifts under midpoint)
      abs_sum_c    = Σ_i |c_i|
      ess_c        = (Σ c_i)² / Σ c_i² — IS-style ESS for the signed estimator
      ess_c_frac   = ess_c / N
      abs_ess_c    = (Σ |c_i|)² / Σ c_i² — Kong's abs-ESS (variance-only ESS)
      cancel_c     = |Σ c_i| / Σ |c_i| — sign-cancellation ratio
      max_abs_c    = max_i |c_i| — single-point dominance check
      max_abs_c_frac = max_i |c_i| / Σ_i |c_i|  (1 means one point dominates)

    A "healthy" estimator has ess_c_frac close to 1 and max_abs_c_frac small.
    A heavy-tailed proposal pushes ess_c_frac → 0 (a few large c_i dominate).
    """
    y_ = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(y_.size)
    if omega is None or n == 0:
        return {k: float("nan") for k in (
            "sum_c", "abs_sum_c", "ess_c", "ess_c_frac",
            "abs_ess_c", "abs_ess_c_frac", "cancel_c",
            "max_abs_c", "max_abs_c_frac",
        )}
    omega_ = np.asarray(omega, dtype=np.float64).reshape(-1)
    c = omega_ * y_                                              # (N,)
    abs_c = np.abs(c)
    S    = float(np.sum(c))
    S_abs= float(np.sum(abs_c))
    S2   = float(np.sum(c * c))
    if S2 > 0.0 and np.isfinite(S2):
        ess     = (S * S) / S2
        abs_ess = (S_abs * S_abs) / S2
    else:
        ess = abs_ess = float("nan")
    max_abs_c = float(np.max(abs_c))
    return {
        "sum_c":          S,
        "abs_sum_c":      S_abs,
        "ess_c":          float(ess),
        "ess_c_frac":     float(ess / n)     if np.isfinite(ess)     else float("nan"),
        "abs_ess_c":      float(abs_ess),
        "abs_ess_c_frac": float(abs_ess / n) if np.isfinite(abs_ess) else float("nan"),
        "cancel_c":       float(abs(S) / S_abs) if S_abs > 0 else float("nan"),
        "max_abs_c":      max_abs_c,
        "max_abs_c_frac": float(max_abs_c / S_abs) if S_abs > 0 else float("nan"),
    }

# =============================================================================
# Absolute integrals (|ρ̂| and support-point phase-space volume proxies)
# =============================================================================

def density_abs_integral_mc(
    gp: GPDensity, n_mc: int = 20_000, seed: Optional[int] = None,
    proposal_std_multiplier: float = 2.0,
) -> Dict[str, float]:
    r"""
    Monte-Carlo estimate of ∫ |ρ̂(z)| dz and ∫ [ρ̂(z)]₊ dz, ∫ [ρ̂(z)]₋ dz.

    ρ̂ can be signed (the sampler is signed-SEO), so these integrals give the
    'positive mass' and 'negative mass' separately.  Their sum is the L¹ norm
    of ρ̂, which is always ≥ 1 = ⟨1⟩ = (positive mass - negative mass).

    Proposal: isotropic Gaussian centered at the support-point mean with
    scale = (std of Z_tr) * proposal_std_multiplier + ℓ_d.
    """
    rng = np.random.default_rng(seed)

    # For the diff-GP, both ingredient tuples share the same current support
    # cloud (Z_tr is the time-t support).  Just use the first tuple for the
    # proposal geometry.
    ingredients = _gp_ingredients(gp)
    if isinstance(ingredients, list):
        _, Z_tr, ell, _, _ = ingredients[1]   # correction tuple; same Z, native ell
    else:
        _, Z_tr, ell, _, _ = ingredients

    mu = Z_tr.mean(axis=0)
    sd = Z_tr.std(axis=0) * proposal_std_multiplier + ell

    z  = rng.normal(size=(n_mc, D)) * sd + mu
    # log-proposal
    logq = (-0.5 * np.sum(((z - mu) / sd) ** 2, axis=1)
            - np.sum(np.log(sd)) - 0.5 * D * np.log(2.0 * np.pi))
    q = np.exp(logq)

    rho = np.empty(n_mc, dtype=np.float64)
    B = 2000
    for i in range(0, n_mc, B):
        rho[i:i+B] = gp.predict(z[i:i+B])

    w = rho / q
    pos = np.mean(np.clip(w, 0.0, None))
    neg = np.mean(np.clip(-w, 0.0, None))
    return {
        "positive_mass":  float(pos),
        "negative_mass":  float(neg),
        "L1_norm":        float(pos + neg),
        "signed_norm":    float(pos - neg),   # should be ≈ ⟨1⟩ = 1
        "n_mc":           int(n_mc),
    }


# =============================================================================
# Unified snapshot builder
# =============================================================================

def compute_all(
    gp: GPDensity,
    Z: FloatArray,
    Q: Optional[FloatArray],
    dt: float,
    y: FloatArray,
    dynamics: PBMEMIntDynamics,
    omega: Optional[FloatArray] = None,
    include_abs_integral: bool = False,
    n_mc_abs: int = 20_000,
) -> Dict[str, float]:
    r"""
    Build the full observable dictionary for one simulation state.

    Cloud-as-discretization semantics
    ---------------------------------
    Every cloud-side observable is a Riemann sum

        ⟨A⟩(t) = Σ_i ω_i A(z_i(t)) y_i(t),

    where ω_i is the FROZEN geometric measure of support point i (set at
    t=0 from the initial sampler) and y_i(t) is the density value at z_i.
    No self-normalization, no weight ratios, no IS reweighting.

    Estimator namespaces
    --------------------
    cloud_*  cloud Riemann sum  Σ_i ω_i y_i A(z_i(t)).  Tracks the QCLE
             correction through y_i(t) for the midpoint scheme.  Reduces
             to the PBME estimator when y is frozen at ρ_0.
    gpi_*    GP analytic kernel-integral against the KKT-constrained surrogate.
             Includes the QCLE correction through α, and is normalized by
             construction (the projection enforces ∫ρ̂ = 1).
    """
    out: Dict[str, float] = {}

    # KKT moments.  The row-indexed transported product profile has no global
    # off-cloud integral; in that regime the raw cloud estimators remain the
    # authoritative observables and analytic GP values are explicitly NaN.
    try:
        km = kkt_moments(gp)
        for k, v in km.items(): out[f"km_{k}"] = v
    except NotImplementedError:
        for k in ("normalization", "trace", "energy", "normalization_raw",
                  "trace_raw", "energy_raw"):
            out[f"km_{k}"] = float("nan")

    # Nuclear moments
    try:
        nm = nuclear_moments(gp)
        for k, v in nm.items(): out[f"nm_{k}"] = v
    except NotImplementedError:
        for k in ("R_mean", "P_mean", "R_sq", "P_sq", "R_var", "P_var"):
            out[f"nm_{k}"] = float("nan")

    # Quadratic mapping moments
    try:
        qm = _quadratic_mapping_moments(gp)
        for k, v in qm.items(): out[f"qm_{k}"] = v
    except NotImplementedError:
        for k in ("r0_sq", "r1_sq", "p0_sq", "p1_sq", "mapping_radius_sq",
                  "r0_r1", "p0_p1", "r0_p0", "r1_p1", "r0_p1", "r1_p0"):
            out[f"qm_{k}"] = float("nan")

    # Mapping weights
    try:
        mw = mapping_weights(gp, hbar=dynamics.params.hbar)
        for k, v in mw.items(): out[f"mw_{k}"] = v
    except NotImplementedError:
        for k in ("w0", "w1", "w_sum", "w_diff"):
            out[f"mw_{k}"] = float("nan")

    # Physical diabatic populations.
    #
    # cloud_*  — cloud Riemann sum  Σ_i ω_i y_i c_{αα}(z_i(t)).
    #            For PBME y is frozen → conservation tied to MInt Casimir.
    #            For midpoint y carries the QCLE correction.
    if omega is not None:
        try:
            cl = diabatic_populations_from_labels(
                Z, y, omega, hbar=dynamics.params.hbar)
            for k, v in cl.items(): out[f"cloud_{k}"] = v
        except Exception:
            for k in ("P0", "P1", "P_sum", "P_diff"):
                out[f"cloud_{k}"] = float("nan")

    # gpi_*  — GP analytic kernel-integral, KKT-constrained surrogate.
    try:
        dp_gpi = diabatic_populations_from_gp(gp, hbar=dynamics.params.hbar)
        for k, v in dp_gpi.items(): out[f"gpi_{k}"] = v
        # dp_* — primary density-based alias (identical to gpi_*; named for clarity)
        for k, v in dp_gpi.items(): out[f"dp_{k}"] = v
    except NotImplementedError:
        for prefix in ("gpi", "dp"):
            for k in ("P0", "P1", "P_sum", "P_diff"):
                out[f"{prefix}_{k}"] = float("nan")

    # Adiabatic populations (GH-quadrature in R against the GP surrogate)
    try:
        ap = adiabatic_populations(gp, dynamics, n_gh=16,
                                   hbar=dynamics.params.hbar)
        for k, v in ap.items(): out[f"ap_{k}"] = v
    except Exception:
        for k in ("Pad_0", "Pad_1", "Pad_sum", "Pad_diff"):
            out[f"ap_{k}"] = float("nan")

    # Coherences (GP analytic; cloud-side coherence at the support points
    # is the same Riemann sum and can be added here if desired)
    try:
        dc = diabatic_coherences(gp, hbar=dynamics.params.hbar)
        for k, v in dc.items(): out[f"dc_{k}"] = v
    except NotImplementedError:
        for k in ("coh_re", "coh_im", "rho00", "rho11", "trace"):
            out[f"dc_{k}"] = float("nan")

    # Energy diagnostics on the support points.
    # spe_E_traj is the per-trajectory equiprobable mean (PBME conservation
    #   check — should be flat to machine precision under MInt).
    # spe_E_density is the cloud Riemann sum Σ ω_i y_i H(z_i)  (physical ⟨H⟩).
    spe = support_point_energies(dynamics, Z, rho=y, omega=omega)
    for k, v in spe.items(): out[f"spe_{k}"] = v

    # Signed-cancellation diagnostics on the carried density values
    sw = signed_weight_diagnostics(y)
    for k, v in sw.items(): out[f"sw_{k}"] = v

    # Cloud Riemann-sum estimator diagnostics: ESS and max-share of c_i = ω_i y_i.
    # These are the right diagnostics for the variance of <A>(t); the sw_*
    # diagnostics only see the labels y, not the ω·y combination that actually
    # enters the estimator.
    ce = cloud_estimator_diagnostics(omega, y)
    for k, v in ce.items(): out[f"ce_{k}"] = v

    # Correction statistics
    cs = correction_statistics(Q, y=y, dt=dt, omega=omega)
    for k, v in cs.items(): out[f"cs_{k}"] = v

    # ------------------------------------------------------------------
    # cloud_norm and self-normalised coherences
    # ------------------------------------------------------------------
    # cloud_norm = Σ ω_i y_i ≈ ∫ρ̂ dz (should be ≈ 1).
    # When the midpoint correction drives Σ ω_i y_i away from 1, the raw
    # cloud_weighted_* sums scale proportionally — divide to recover the
    # properly normalised expectation.
    if omega is not None:
        omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
        y_arr     = np.asarray(y,     dtype=np.float64).reshape(-1)
        cloud_norm_val = float(np.dot(omega_arr, y_arr))
        out["cloud_norm"] = cloud_norm_val
        _D = cloud_norm_val if abs(cloud_norm_val) > 1e-15 else 1.0
        # Coherences are computed in _weighted_support_diagnostics via
        # cloud_weighted_coh_re/im — expose lw_* aliases here so plots
        # can find them from compute_all output.
        for _ck in ("coh_re", "coh_im"):
            _raw = out.get(f"cloud_weighted_{_ck}", float("nan"))
            out[f"lw_{_ck}"] = (_raw / _D) if np.isfinite(_raw) else float("nan")
    else:
        out["cloud_norm"] = float("nan")
        for _ck in ("coh_re", "coh_im"):
            out[f"lw_{_ck}"] = float("nan")

    # Density-change rate diagnostics: magnitude of the QCLE Q operator
    if Q is not None:
        Q_arr = np.asarray(Q, dtype=np.float64).reshape(-1)
        out["drho_q_rms"]     = float(np.sqrt(np.mean(Q_arr ** 2)))
        out["drho_q_max_abs"] = float(np.max(np.abs(Q_arr)))
        if omega is not None:
            omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
            out["drho_q_cloud_int"] = float(np.dot(omega_arr, Q_arr))
    else:
        out["drho_q_rms"] = 0.0
        out["drho_q_max_abs"] = 0.0
        out["drho_q_cloud_int"] = 0.0

    # Optional expensive MC L¹ norm
    if include_abs_integral:
        ai = density_abs_integral_mc(gp, n_mc=n_mc_abs)
        for k, v in ai.items(): out[f"ai_{k}"] = v

    return out


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    from .Mint import pack_z
    from .Sampling import (
        GaussianWavePacketParams, MappingInitParams, MMSTSampler,
    )
    from .GP_Density import GPDensityConfig

    rng = np.random.default_rng(0)
    cp = GaussianWavePacketParams(R0=-5.0, P0=25.0, sigma_R=1.0)
    mp = MappingInitParams(nstates=2, init_state=0)
    s = MMSTSampler(cp, mp).sample_seo_signed(n_samples=800, rng=rng)
    Z = pack_z(s.R, s.P, s.r, s.p); y = s.target_density
    dyn = PBMEMIntDynamics(); E0 = float(np.mean(dyn.energy(Z)))

    gp = GPDensity(GPDensityConfig(n_opt_steps=80, fix_sigma_n=True,
                                   init_log_sigma_n=-11.5,
                                   log_sn_floor=-14.0), dynamics=dyn)
    gp.fit(Z_train=Z, y_train=y,
           moment_targets={"normalization":1.0,"trace":1.0,"energy":E0})

    obs = compute_all(gp, Z, Q=None, dt=0.5, y=y, dynamics=dyn,
                      include_abs_integral=True, n_mc_abs=5000)
    for k, v in sorted(obs.items()):
        print(f"  {k:30s} = {v:+.4e}")
