from __future__ import annotations

"""Closed-form moments for ``rho_hat(z) = g_SEO(x) * mu_GP(z)``.

The mapping profile is a Gaussian times a quadratic polynomial and the
modulation GP uses ARD Gaussian kernels.  Their product is another Gaussian,
so all mapping and nuclear polynomial moments are analytic.  Only the
R-dependent Tully potential/eigenvectors use deterministic Gauss--Hermite
quadrature.  These routines intentionally reject the row-indexed transported
profile: it is defined only on aligned support rows and therefore has no
unambiguous global integral.
"""

from typing import Dict, Iterable

import numpy as np


I_R, I_P = 0, 1
MAP_DIMS = (2, 3, 4, 5)


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _require_static(gp) -> None:
    if getattr(gp, "_footpoints", None) is not None:
        raise NotImplementedError(
            "Global analytic moments are undefined for the row-indexed "
            "transported SEO profile. Use the saved raw cloud Riemann-sum "
            "observables for product_transported runs."
        )


def _context(gp) -> dict:
    _require_static(gp)
    inner = gp._inner
    if not getattr(inner, "_initial_fit_done", False):
        raise RuntimeError("Product GP must be fitted before moment evaluation.")
    alpha = _numpy(inner._alpha).reshape(-1)
    centers = (inner.raw_training_centers if hasattr(inner, "raw_training_centers")
               else _numpy(inner._Z_train))
    centers = _numpy(centers)
    ell = _numpy(inner.lengthscales).reshape(-1)
    hbar = float(gp._hbar)
    active = int(gp._init_state)
    nstates = int(gp._nstates)

    mu = np.zeros((centers.shape[0], 4), dtype=np.float64)
    var = np.zeros_like(mu)
    map_factor = np.ones(centers.shape[0], dtype=np.float64)
    for j, d in enumerate(MAP_DIMS):
        den = hbar + 2.0 * ell[d] ** 2
        var[:, j] = hbar * ell[d] ** 2 / den
        mu[:, j] = centers[:, d] * hbar / den
        map_factor *= (np.sqrt(2.0 * np.pi * var[:, j])
                       * np.exp(-centers[:, d] ** 2 / den))

    bath_factor = (np.sqrt(2.0 * np.pi) * ell[I_R]
                   * np.sqrt(2.0 * np.pi) * ell[I_P])
    profile_A = (np.pi * hbar) ** (-nstates)
    prefactor = alpha * float(inner.sigma_f) ** 2 * bath_factor * profile_A * map_factor
    return {"gp": gp, "inner": inner, "alpha": alpha, "centers": centers,
            "ell": ell, "hbar": hbar, "active": active,
            "mu": mu, "var": var, "prefactor": prefactor}


def _normal_moment(mu: np.ndarray, var: np.ndarray, power: int) -> np.ndarray:
    if power == 0:
        return np.ones_like(mu)
    if power == 1:
        return mu
    if power == 2:
        return mu * mu + var
    if power == 3:
        return mu ** 3 + 3.0 * mu * var
    if power == 4:
        return mu ** 4 + 6.0 * mu * mu * var + 3.0 * var * var
    raise ValueError(f"Only moments through fourth order are required; got {power}.")


def _poly(ctx: dict, powers: Iterable[int]) -> np.ndarray:
    out = np.ones(ctx["mu"].shape[0], dtype=np.float64)
    for j, p in enumerate(tuple(powers)):
        out *= _normal_moment(ctx["mu"][:, j], ctx["var"][:, j], int(p))
    return out


def _q_poly(ctx: dict, powers=(0, 0, 0, 0)) -> np.ndarray:
    powers = list(powers)
    base = -_poly(ctx, powers)
    hbar = ctx["hbar"]
    for j in (ctx["active"], 2 + ctx["active"]):
        raised = list(powers); raised[j] += 2
        base += (2.0 / hbar) * _poly(ctx, raised)
    return base


def _map_raw(ctx: dict, powers=(0, 0, 0, 0)) -> float:
    return float(np.dot(ctx["prefactor"], _q_poly(ctx, powers)))


def product_norm_raw(gp) -> float:
    ctx = _context(gp)
    return _map_raw(ctx)


def _normalized(raw: float, norm: float) -> float:
    return raw / norm if abs(norm) > 1.0e-15 else float("nan")


def product_nuclear_moments(gp) -> Dict[str, float]:
    ctx = _context(gp)
    c, ell, a = ctx["centers"], ctx["ell"], ctx["prefactor"] * _q_poly(ctx)
    norm = float(np.sum(a))
    R_raw = float(np.dot(a, c[:, I_R])); P_raw = float(np.dot(a, c[:, I_P]))
    R2_raw = float(np.dot(a, c[:, I_R] ** 2 + ell[I_R] ** 2))
    P2_raw = float(np.dot(a, c[:, I_P] ** 2 + ell[I_P] ** 2))
    R = _normalized(R_raw, norm); P = _normalized(P_raw, norm)
    R2 = _normalized(R2_raw, norm); P2 = _normalized(P2_raw, norm)
    return {"R_mean": R, "P_mean": P, "R_sq": R2, "P_sq": P2,
            "R_var": R2 - R * R, "P_var": P2 - P * P}


def product_quadratic_mapping_moments(gp) -> Dict[str, float]:
    ctx = _context(gp); norm = _map_raw(ctx)
    def m2(j):
        p = [0, 0, 0, 0]; p[j] = 2
        return _normalized(_map_raw(ctx, p), norm)
    def cross(j, k):
        p = [0, 0, 0, 0]; p[j] += 1; p[k] += 1
        return _normalized(_map_raw(ctx, p), norm)
    r0, r1, p0, p1 = m2(0), m2(1), m2(2), m2(3)
    return {"r0_sq": r0, "r1_sq": r1, "p0_sq": p0, "p1_sq": p1,
            "mapping_radius_sq": r0 + r1 + p0 + p1,
            "r0_r1": cross(0, 1), "p0_p1": cross(2, 3),
            "r0_p0": cross(0, 2), "r1_p1": cross(1, 3),
            "r0_p1": cross(0, 3), "r1_p0": cross(1, 2)}


def _mapping_electronic_raw(ctx: dict) -> dict[str, np.ndarray]:
    h = ctx["hbar"]
    q0 = _q_poly(ctx)
    def qp(powers): return _q_poly(ctx, powers)
    r0 = qp((2, 0, 0, 0)); r1 = qp((0, 2, 0, 0))
    p0 = qp((0, 0, 2, 0)); p1 = qp((0, 0, 0, 2))
    c00 = (r0 + p0 - h * q0) / (2.0 * h)
    c11 = (r1 + p1 - h * q0) / (2.0 * h)
    c01 = (qp((1, 1, 0, 0)) + qp((0, 0, 1, 1))) / (2.0 * h)
    return {"q0": q0, "c00": c00, "c11": c11, "c01": c01,
            "r0": r0, "r1": r1, "p0": p0, "p1": p1,
            "r01": qp((1, 1, 0, 0)), "p01": qp((0, 0, 1, 1))}


def product_kkt_moments(gp, n_gh: int = 24) -> Dict[str, float]:
    ctx = _context(gp); e = _mapping_electronic_raw(ctx)
    pref, c, ell, h = ctx["prefactor"], ctx["centers"], ctx["ell"], ctx["hbar"]
    norm = float(np.dot(pref, e["q0"]))
    trace_raw = float(np.dot(pref, e["c00"] + e["c11"]))

    dyn = getattr(ctx["inner"], "dynamics", None)
    if dyn is None:
        energy_raw = float("nan")
    else:
        x, w = np.polynomial.hermite.hermgauss(int(n_gh)); w = w / np.sqrt(np.pi)
        R = c[:, I_R, None] + np.sqrt(2.0) * ell[I_R] * x[None, :]
        V0, hmat, _, _ = dyn._frozen_R_objects(R.reshape(-1))
        V0 = np.asarray(V0).reshape(R.shape)
        hmat = np.asarray(hmat).reshape(R.shape + (2, 2))
        map_rrpp = np.empty((c.shape[0], 2, 2), dtype=np.float64)
        map_rrpp[:, 0, 0] = e["r0"] + e["p0"]
        map_rrpp[:, 1, 1] = e["r1"] + e["p1"]
        map_rrpp[:, 0, 1] = map_rrpp[:, 1, 0] = e["r01"] + e["p01"]
        potential = V0 * e["q0"][:, None]
        electronic = 0.5 / h * np.einsum("nqab,nab->nq", hmat, map_rrpp)
        R_expect = np.sum(w[None, :] * (potential + electronic), axis=1)
        kinetic = e["q0"] * (c[:, I_P] ** 2 + ell[I_P] ** 2) / (2.0 * dyn.params.mass)
        energy_raw = float(np.dot(pref, kinetic + R_expect))
    return {"normalization": 1.0 if abs(norm) > 1.0e-15 else float("nan"),
            "trace": _normalized(trace_raw, norm),
            "energy": _normalized(energy_raw, norm),
            "normalization_raw": norm, "trace_raw": trace_raw,
            "energy_raw": energy_raw}


def product_adiabatic_populations(gp, dynamics, n_gh: int = 24,
                                  hbar=None) -> Dict[str, float]:
    ctx = _context(gp); e = _mapping_electronic_raw(ctx)
    pref, c, ell = ctx["prefactor"], ctx["centers"], ctx["ell"]
    norm = float(np.dot(pref, e["q0"]))
    x, w = np.polynomial.hermite.hermgauss(int(n_gh)); w = w / np.sqrt(np.pi)
    R = c[:, I_R, None] + np.sqrt(2.0) * ell[I_R] * x[None, :]
    # Local import avoids a module cycle at import time.
    from .Observables import _diabatic_to_adiabatic
    U, _ = _diabatic_to_adiabatic(dynamics, R.reshape(-1))
    U = U.reshape(R.shape + (2, 2))
    vals = []
    for ad in (0, 1):
        local = (U[:, :, 0, ad] ** 2 * e["c00"][:, None]
                 + U[:, :, 1, ad] ** 2 * e["c11"][:, None]
                 + 2.0 * U[:, :, 0, ad] * U[:, :, 1, ad] * e["c01"][:, None])
        vals.append(_normalized(float(np.dot(pref, np.sum(w[None, :] * local, axis=1))), norm))
    return {"Pad_0": vals[0], "Pad_1": vals[1],
            "Pad_sum": vals[0] + vals[1], "Pad_diff": vals[0] - vals[1]}
