from __future__ import annotations

"""
GPDerivatives.py
================

Analytic derivatives of an ARD-RBF GP density surrogate with respect to its
phase-space argument.

The density is
    ρ(z) = Σ_i α_i k(z, Z_i),
    k(z, z') = σ_f^2 Π_d exp(-½ (z_d - z'_d)^2 / ℓ_d^2).

Let  v_d^{(i)} = (z_d - Z_{i,d}) / ℓ_d^2,   λ_d = 1/ℓ_d^2.

Then the analytic derivatives of a single kernel column are
    ∂_a     k_i = -v_a^{(i)} k_i
    ∂_{ab}   k_i = (v_a^{(i)} v_b^{(i)} - δ_{ab} λ_a) k_i
    ∂_{abc}  k_i = [-v_a v_b v_c
                    + δ_{ab} λ_a v_c
                    + δ_{ac} λ_a v_b
                    + δ_{bc} λ_b v_a] k_i

and the GP-level derivatives are the α-weighted sums:
    ∂^k ρ(Y) = Σ_i α_i ∂^k k(Y, Z_i).
"""

from typing import Tuple

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from .GP_Density import GPDensity, _ard_gram, _as_tensor
from .Mint import D

FloatArray = NDArray[np.float64]


def _prepare(gp: GPDensity, Y: ArrayLike) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool
]:
    """
    Common preparation for all derivative orders.

    Returns
    -------
    K      : (M, N) kernel  k(Y_m, Z_i)
    V      : (M, N, D)      v_d^{(i)}(Y_m) = (Y_{m,d} - Z_{i,d}) / ℓ_d^2
    lam    : (D,)           1/ℓ_d^2
    alpha  : (N,)           GP coefficients
    W      : (M, N)         α_i * k(Y_m, Z_i), ready for einsum
    single : bool           True if Y was passed as 1-D
    """
    if gp._alpha is None or gp._Z_train is None or gp._Z_train_norm is None:
        raise RuntimeError("GP must be fitted before taking derivatives.")

    Y_np = np.asarray(Y, dtype=np.float64)
    single = (Y_np.ndim == 1)
    if single:
        Y_np = Y_np.reshape(1, D)

    Y_t_norm = _as_tensor(gp._normalize_features_np(Y_np))
    Z_t_norm = gp._Z_train_norm
    Z_t_raw = gp._Z_train
    ell_raw = _as_tensor(gp.lengthscales)
    lam = 1.0 / (ell_raw ** 2)
    alpha = gp._alpha

    with torch.no_grad():
        K = _ard_gram(Y_t_norm, Z_t_norm, gp.log_sigma_f, gp.log_lengthscales)
        diff_raw = _as_tensor(Y_np)[:, None, :] - Z_t_raw[None, :, :]
        V = diff_raw * lam[None, None, :]
        W = K * alpha[None, :]

    return K, V, lam, alpha, W, single


def rho_value(gp: GPDensity, Y: ArrayLike) -> FloatArray:
    r"""ρ(Y). Returns shape () for single query, (M,) for batch."""
    K, _, _, alpha, _, single = _prepare(gp, Y)
    with torch.no_grad():
        val = K @ alpha
    out = val.detach().cpu().numpy().astype(np.float64)
    return float(out[0]) if single else out


def rho_gradient(gp: GPDensity, Y: ArrayLike) -> FloatArray:
    r"""
    ∂_a ρ(Y) = Σ_i α_i (-v_a^{(i)}) k(Y, Z_i).

    Returns shape (D,) for single query, (M, D) for batch.
    """
    _, V, _, _, W, single = _prepare(gp, Y)
    with torch.no_grad():
        grad = -torch.einsum("mi,mia->ma", W, V)
    out = grad.detach().cpu().numpy().astype(np.float64)
    return out[0] if single else out


def rho_hessian(gp: GPDensity, Y: ArrayLike) -> FloatArray:
    r"""
    ∂_{ab} ρ(Y) = Σ_i α_i [v_a v_b - δ_{ab} λ_a] k(Y, Z_i).

    Returns shape (D, D) for single query, (M, D, D) for batch.
    """
    _, V, lam, _, W, single = _prepare(gp, Y)
    with torch.no_grad():
        T_vv = torch.einsum("mi,mia,mib->mab", W, V, V)
        S = torch.einsum("mi->m", W)
        diag_correction = torch.zeros_like(T_vv)
        diag_correction[:, torch.arange(D), torch.arange(D)] = lam[None, :] * S[:, None]
        H = T_vv - diag_correction
    out = H.detach().cpu().numpy().astype(np.float64)
    return out[0] if single else out


def rho_third_derivative(gp: GPDensity, Y: ArrayLike) -> FloatArray:
    r"""
    ∂_{abc} ρ(Y) = Σ_i α_i [-v_a v_b v_c
                            + δ_{ab} λ_a v_c
                            + δ_{ac} λ_a v_b
                            + δ_{bc} λ_b v_a] k(Y, Z_i).

    The result is fully symmetric in (a, b, c). Returns shape (D, D, D) for
    single query, (M, D, D, D) for batch.
    """
    _, V, lam, _, W, single = _prepare(gp, Y)
    with torch.no_grad():
        T1 = -torch.einsum("mi,mia,mib,mic->mabc", W, V, V, V)
        S1 = torch.einsum("mi,mid->md", W, V)
        I = torch.eye(D, dtype=T1.dtype)
        T2 = torch.einsum("ab,a,mc->mabc", I, lam, S1)
        T3 = torch.einsum("ac,a,mb->mabc", I, lam, S1)
        T4 = torch.einsum("bc,b,ma->mabc", I, lam, S1)
        T = T1 + T2 + T3 + T4
    out = T.detach().cpu().numpy().astype(np.float64)
    return out[0] if single else out


def _is_density_diff(gp) -> bool:
    """Duck-type check for GPDensityDiff (avoid hard import cycle)."""
    return (hasattr(gp, "gp0") and hasattr(gp, "gp_delta")
            and hasattr(gp, "Z0") and hasattr(gp, "y0"))


def _prepare_baseline_transported(gp_diff, Y: ArrayLike):
    """
    Prepare the same tensors as `_prepare` but for the APPROXIMATE
    baseline-transported part of a GPDensityDiff surrogate:

        ρ₀(Φ_{-t}(Y)) ≈ k_raw(Y, Z_t) @ α₀

    where k_raw is evaluated in physical (raw) coordinates using gp0's
    physical lengthscales (gp0.lengthscales, which are in raw space).

    The equality k_raw(Y, Z_t) @ α₀ = ρ₀(Φ_{-t}(Y)) is exact only at
    support points Z_t where Φ_{-t}(Z_t) = Z_0 by construction.  For
    off-support queries (e.g. midpoints Y ≠ Z_t), MInt is a non-Euclidean
    symplectic map, so the kernel is translation-invariant but not
    flow-invariant — the equality becomes an approximation whose error is
    O(|Y - Z_t|³) in the kernel's Taylor expansion around each Z_t.
    This approximation is used because exactly evaluating ρ₀(Φ_{-t}(Y))
    would require pushing Y back through the backward flow at every call,
    which is expensive.  Accept this as-is for production runs with small dt.

    Feature-zscore fix (H1): gp0.log_lengthscales are stored in NORMALIZED
    feature space, while Z_t and Y are in raw physical coordinates.  We must
    use gp0.lengthscales (which multiplies norm-space ell by _feature_std to
    return physical units) for the diff scaling, and we build a synthetic
    log_sigma_f_eff / log_lengthscales_eff pair that drives _ard_gram in raw
    space with the correct physical metric.  This mirrors the approach in the
    vanilla `_prepare`, which computes diff_raw × lam_raw.
    """
    gp0 = gp_diff.gp0
    gpd = gp_diff.gp_delta
    if gp0._alpha is None or gpd._Z_train is None:
        raise RuntimeError("GPDensityDiff must be fitted before taking derivatives.")

    Y_np = np.asarray(Y, dtype=np.float64)
    single = (Y_np.ndim == 1)
    if single:
        Y_np = Y_np.reshape(1, D)

    # gp0.lengthscales returns PHYSICAL lengthscales: ell_raw = ell_norm * std
    # when feature_zscore=True, or just ell_norm when feature_zscore=False.
    # We use physical units throughout so there is no normalizer mismatch.
    ell_raw = _as_tensor(gp0.lengthscales)          # (D,) physical units
    lam = 1.0 / (ell_raw ** 2)
    # log_lengthscales_raw = log(ell_raw) for use inside _ard_gram
    log_ell_raw = torch.log(ell_raw)

    Z_t_raw = gpd._Z_train                          # (N, D) current support, raw
    Y_t_raw = _as_tensor(Y_np)                      # (M, D) raw

    alpha = gp0._alpha

    with torch.no_grad():
        # _ard_gram expects both arguments in the same coordinate space; here
        # both Y_t_raw and Z_t_raw are physical, and log_ell_raw is also physical.
        K = _ard_gram(Y_t_raw, Z_t_raw, gp0.log_sigma_f, log_ell_raw)
        diff_raw = Y_t_raw[:, None, :] - Z_t_raw[None, :, :]
        V = diff_raw * lam[None, None, :]
        W = K * alpha[None, :]

    return K, V, lam, alpha, W, single


def _rho_derivative_bundle_from_WVlam(W, V, lam):
    """
    Core math used by rho_derivative_bundle, factored out so it can be
    reused on the baseline-transported tensors.
    """
    with torch.no_grad():
        grad = -torch.einsum("mi,mia->ma", W, V)

        T_vv = torch.einsum("mi,mia,mib->mab", W, V, V)
        S = torch.einsum("mi->m", W)
        diag_correction = torch.zeros_like(T_vv)
        diag_correction[:, torch.arange(D), torch.arange(D)] = lam[None, :] * S[:, None]
        hess = T_vv - diag_correction

        T1 = -torch.einsum("mi,mia,mib,mic->mabc", W, V, V, V)
        S1 = torch.einsum("mi,mid->md", W, V)
        I = torch.eye(D, dtype=T1.dtype)
        T2 = torch.einsum("ab,a,mc->mabc", I, lam, S1)
        T3 = torch.einsum("ac,a,mb->mabc", I, lam, S1)
        T4 = torch.einsum("bc,b,ma->mabc", I, lam, S1)
        third = T1 + T2 + T3 + T4

    return grad, hess, third


def rho_derivative_bundle(gp, Y: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""
    Return the analytic first, second, and third derivatives of the GP
    surrogate at the same query batch with a single kernel preparation.

    For a GPDensityDiff, returns the SUM of the baseline-transported
    derivatives and the correction derivatives (both are linear in α).

    Returns
    -------
    grad : (D,) or (M, D)
        ρ_{,a}(Y)
    hess : (D, D) or (M, D, D)
        ρ_{,ab}(Y)
    third : (D, D, D) or (M, D, D, D)
        ρ_{,abc}(Y)
    """
    # Query batching is algebraically identical because prediction and every
    # derivative are pointwise in Y.  A full production evaluation with
    # M=N=2000 otherwise materializes V with shape (M,N,6), plus large einsum
    # intermediates, and can exhaust a 16-GB workstation even though the
    # returned derivative arrays are small.  Batching changes only the peak
    # memory footprint; kernels, coefficients, dtype, and contractions are
    # unchanged.
    Y_array = np.asarray(Y, dtype=np.float64)
    if Y_array.ndim == 2 and Y_array.shape[0] > 1:
        if _is_density_diff(gp):
            n_train = int(gp.gp_delta._Z_train.shape[0])
        else:
            n_train = int(gp._Z_train.shape[0])
        max_query_support_pairs = 750_000
        if Y_array.shape[0] * n_train > max_query_support_pairs:
            batch_size = max(1, max_query_support_pairs // max(n_train, 1))
            chunks = [
                rho_derivative_bundle(gp, Y_array[start:start + batch_size])
                for start in range(0, Y_array.shape[0], batch_size)
            ]
            return tuple(
                np.concatenate([chunk[index] for chunk in chunks], axis=0)
                for index in range(3)
            )

    if _is_density_diff(gp):
        # Baseline-transported contribution
        _, V_b, lam_b, _, W_b, single = _prepare_baseline_transported(gp, Y)
        grad_b, hess_b, third_b = _rho_derivative_bundle_from_WVlam(W_b, V_b, lam_b)

        # Correction contribution (uses the δ-GP as a vanilla GPDensity)
        _, V_d, lam_d, _, W_d, _ = _prepare(gp.gp_delta, Y)
        grad_d, hess_d, third_d = _rho_derivative_bundle_from_WVlam(W_d, V_d, lam_d)

        grad_np  = (grad_b  + grad_d ).detach().cpu().numpy().astype(np.float64)
        hess_np  = (hess_b  + hess_d ).detach().cpu().numpy().astype(np.float64)
        third_np = (third_b + third_d).detach().cpu().numpy().astype(np.float64)
        if single:
            return grad_np[0], hess_np[0], third_np[0]
        return grad_np, hess_np, third_np

    # Vanilla GPDensity path (unchanged)
    _, V, lam, _, W, single = _prepare(gp, Y)
    grad, hess, third = _rho_derivative_bundle_from_WVlam(W, V, lam)
    grad_np = grad.detach().cpu().numpy().astype(np.float64)
    hess_np = hess.detach().cpu().numpy().astype(np.float64)
    third_np = third.detach().cpu().numpy().astype(np.float64)
    if single:
        return grad_np[0], hess_np[0], third_np[0]
    return grad_np, hess_np, third_np


# =============================================================================
# Finite-difference verification utilities
# =============================================================================

def _fd_gradient_from_value(func, x: ArrayLike, h: float = 1.0e-6) -> FloatArray:
    """
    Central finite-difference gradient of a scalar function f: R^D -> R.

    Returns
    -------
    grad : (D,)
        [f(x + h e_a) - f(x - h e_a)] / (2 h)
    """
    x = np.asarray(x, dtype=np.float64).reshape(D)
    grad = np.zeros(D, dtype=np.float64)
    for a in range(D):
        ea = np.zeros(D, dtype=np.float64)
        ea[a] = float(h)
        grad[a] = (float(func(x + ea)) - float(func(x - ea))) / (2.0 * h)
    return grad


def _fd_hessian_from_value(func, x: ArrayLike, h: float = 1.0e-4) -> FloatArray:
    """
    Central finite-difference Hessian of a scalar function f: R^D -> R.

    Diagonal entries:
        f_{,aa}(x) ≈ [f(x+h e_a) - 2 f(x) + f(x-h e_a)] / h^2

    Off-diagonal entries:
        f_{,ab}(x) ≈ [f(x+h e_a+h e_b) - f(x+h e_a-h e_b)
                      - f(x-h e_a+h e_b) + f(x-h e_a-h e_b)] / (4 h^2)
    """
    x = np.asarray(x, dtype=np.float64).reshape(D)
    H = np.zeros((D, D), dtype=np.float64)
    fx = float(func(x))

    for a in range(D):
        ea = np.zeros(D, dtype=np.float64)
        ea[a] = float(h)
        H[a, a] = (float(func(x + ea)) - 2.0 * fx + float(func(x - ea))) / (h * h)

        for b in range(a + 1, D):
            eb = np.zeros(D, dtype=np.float64)
            eb[b] = float(h)
            val = (
                float(func(x + ea + eb))
                - float(func(x + ea - eb))
                - float(func(x - ea + eb))
                + float(func(x - ea - eb))
            ) / (4.0 * h * h)
            H[a, b] = val
            H[b, a] = val

    return H


def _fd_third_from_value(
    func,
    x: ArrayLike,
    h_hess: float = 2.0e-4,
    h_third: float = 2.0e-3,
) -> FloatArray:
    """
    Finite-difference third derivative tensor of a scalar function f: R^D -> R.

    Strategy
    --------
    Differentiate the Hessian with respect to each coordinate:
        f_{,abc}(x) ≈ [H_{ab}(x + h e_c) - H_{ab}(x - h e_c)] / (2 h)

    where H_{ab} itself is obtained from central finite differences of f.

    Because the exact third derivative is fully symmetric, the returned tensor
    is symmetrized over index permutations to reduce finite-difference noise.
    """
    x = np.asarray(x, dtype=np.float64).reshape(D)
    T = np.zeros((D, D, D), dtype=np.float64)

    for c in range(D):
        ec = np.zeros(D, dtype=np.float64)
        ec[c] = float(h_third)
        H_plus  = _fd_hessian_from_value(func, x + ec, h=h_hess)
        H_minus = _fd_hessian_from_value(func, x - ec, h=h_hess)
        T[:, :, c] = (H_plus - H_minus) / (2.0 * h_third)

    # Full symmetrization over the three derivative indices.
    T_sym = (
        T
        + np.transpose(T, (0, 2, 1))
        + np.transpose(T, (1, 0, 2))
        + np.transpose(T, (1, 2, 0))
        + np.transpose(T, (2, 0, 1))
        + np.transpose(T, (2, 1, 0))
    ) / 6.0
    return T_sym


def test_gp_derivatives_against_finite_differences(
    feature_zscore: bool = False,
    n_train: int = 80,
    seed: int = 0,
    h_grad: float = 1.0e-6,
    h_hess: float = 1.0e-4,
    h_third: float = 2.0e-3,
    grad_tol: float = 1.0e-6,
    hess_tol: float = 1.0e-4,
    third_tol: float = 5.0e-3,
) -> dict:
    """
    Verify the analytic GP derivatives against finite differences of ρ itself.

    The test fits a small GP surrogate to SEO-signed MMST samples, evaluates
    ρ, ∇ρ, ∇²ρ, and the third derivative tensor at one off-support query point,
    and compares the analytic formulas to central finite differences of the
    scalar value function z -> ρ(z).

    Parameters
    ----------
    feature_zscore : bool
        If True, also verify the chain rule through internal feature z-scoring.
    n_train : int
        Number of support points used to fit the test GP.
    seed : int
        RNG seed.
    h_grad, h_hess, h_third : float
        Finite-difference step sizes.
    grad_tol, hess_tol, third_tol : float
        Absolute error tolerances.

    Returns
    -------
    dict
        Summary containing the maximum absolute differences.
    """
    from .Sampling import GaussianWavePacketParams, MappingInitParams, MMSTSampler
    from .Mint import PBMEMIntParams, PBMEMIntDynamics, pack_z
    from .Models import TullyModel, TullyParams
    from .GP_Density import GPDensity, GPDensityConfig

    rng = np.random.default_rng(seed)

    model = TullyModel(TullyParams.defaults("dual"))
    dynamics = PBMEMIntDynamics(
        model=model,
        params=PBMEMIntParams(mass=2000.0, hbar=1.0),
    )

    sampler = MMSTSampler(
        GaussianWavePacketParams(R0=[-8.0], P0=[30.0], sigma_R=[1.0], hbar=1.0),
        MappingInitParams(nstates=2, init_state=0, hbar=1.0, gamma=0.5),
    )
    s = sampler.sample_seo_signed(n_samples=int(n_train), rng=rng)
    Z0 = pack_z(s.R, s.P, s.r, s.p)
    y0 = s.target_density

    cfg = GPDensityConfig(
        n_opt_steps=0,
        fix_sigma_n=True,
        init_log_sigma_n=-4.0,
        reinit_lengthscales=True,
        feature_zscore=bool(feature_zscore),
        recompute_feature_zscore=False,
        interpolate_targets=False,
        constraints_enabled=False,
        # normalize_targets was removed from GPDensityConfig (RNS removed).
    )
    gp = GPDensity(cfg, dynamics=dynamics)
    gp.fit(
        Z_train=Z0,
        y_train=y0,
        moment_targets={},
        optimize=False,
        apply_constraints=False,
    )

    # Off-support query point so the test does not sit exactly on a center.
    x = np.mean(Z0, axis=0) + np.array([0.13, -0.07, 0.11, -0.09, 0.05, -0.04], dtype=np.float64)

    value_func = lambda z: float(rho_value(gp, z))

    grad_an = np.asarray(rho_gradient(gp, x), dtype=np.float64)
    hess_an = np.asarray(rho_hessian(gp, x), dtype=np.float64)
    third_an = np.asarray(rho_third_derivative(gp, x), dtype=np.float64)

    grad_fd = _fd_gradient_from_value(value_func, x, h=h_grad)
    hess_fd = _fd_hessian_from_value(value_func, x, h=h_hess)
    third_fd = _fd_third_from_value(value_func, x, h_hess=2.0 * h_hess, h_third=h_third)

    grad_err = np.abs(grad_an - grad_fd)
    hess_err = np.abs(hess_an - hess_fd)
    third_err = np.abs(third_an - third_fd)

    max_grad_err = float(np.max(grad_err))
    max_hess_err = float(np.max(hess_err))
    max_third_err = float(np.max(third_err))

    # Symmetry checks for the analytic tensors.
    hess_sym = float(np.max(np.abs(hess_an - hess_an.T)))
    third_sym = float(np.max(np.abs(
        third_an
        - np.transpose(third_an, (0, 2, 1))
    )))
    third_sym = max(
        third_sym,
        float(np.max(np.abs(third_an - np.transpose(third_an, (1, 0, 2))))),
        float(np.max(np.abs(third_an - np.transpose(third_an, (1, 2, 0))))),
        float(np.max(np.abs(third_an - np.transpose(third_an, (2, 0, 1))))),
        float(np.max(np.abs(third_an - np.transpose(third_an, (2, 1, 0))))),
    )

    print("[GP derivative FD test]")
    print(f"  feature_zscore = {feature_zscore}")
    print(f"  n_train        = {n_train}")
    print(f"  query x        = {x}")
    print(f"  max |grad_an - grad_fd|   = {max_grad_err:.6e}")
    print(f"  max |hess_an - hess_fd|   = {max_hess_err:.6e}")
    print(f"  max |third_an - third_fd| = {max_third_err:.6e}")
    print(f"  max Hessian antisymmetry  = {hess_sym:.6e}")
    print(f"  max Third antisymmetry    = {third_sym:.6e}")

    if max_grad_err > grad_tol:
        raise AssertionError(
            f"GP gradient FD test failed: max error {max_grad_err:.3e} > tol {grad_tol:.3e}"
        )
    if max_hess_err > hess_tol:
        raise AssertionError(
            f"GP Hessian FD test failed: max error {max_hess_err:.3e} > tol {hess_tol:.3e}"
        )
    if max_third_err > third_tol:
        raise AssertionError(
            f"GP third-derivative FD test failed: max error {max_third_err:.3e} > tol {third_tol:.3e}"
        )
    if hess_sym > 1.0e-12:
        raise AssertionError(
            f"Analytic Hessian lost symmetry: max antisymmetry {hess_sym:.3e}"
        )
    if third_sym > 1.0e-12:
        raise AssertionError(
            f"Analytic third derivative lost symmetry: max antisymmetry {third_sym:.3e}"
        )

    return {
        "feature_zscore": bool(feature_zscore),
        "n_train": int(n_train),
        "max_abs_grad_error": max_grad_err,
        "max_abs_hess_error": max_hess_err,
        "max_abs_third_error": max_third_err,
        "max_hessian_antisymmetry": hess_sym,
        "max_third_antisymmetry": third_sym,
    }


if __name__ == "__main__":
    out0 = test_gp_derivatives_against_finite_differences(feature_zscore=False)
    out1 = test_gp_derivatives_against_finite_differences(feature_zscore=True)
    print("\n[summary]")
    print(out0)
    print(out1)
