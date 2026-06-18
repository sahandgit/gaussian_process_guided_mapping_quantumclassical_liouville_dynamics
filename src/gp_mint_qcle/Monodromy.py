from __future__ import annotations

"""
Monodromy.py
============

Monodromy matrices and pullback-geometry tensors for the MInt propagator.

Midpoint-map geometry (primary path)
--------------------------------------
The geometric tensors

    J_{a,i}(Z)   = ∂Y_a / ∂Z_i
    H_{a,ij}(Z)  = ∂²Y_a / (∂Z_i ∂Z_j)
    T_{a,ijk}(Z) = ∂³Y_a / (∂Z_i ∂Z_j ∂Z_k)

of the backward half-step map  Y(Z) = Φ^0_{-Δt/2}(Z)  are computed by
**JAX forward-mode automatic differentiation** (`jax.jacfwd`).  This
eliminates finite-difference truncation error and noise from all three
derivative orders; results are exact up to floating-point rounding.

If JAX is not installed the module falls back to central finite differences
via `midpoint_required_tensors` and emits a one-time RuntimeWarning.

JAX setup
---------
`jax_enable_x64` is activated at import time so all JAX arrays use float64,
consistent with the rest of the pipeline.

Derivative strategy
-------------------
J_{a,i}     — full Jacobian          (N, D, D)     via vmap(jacfwd(f))
H_{a,ij}    — full Hessian           (N, D, D, D)  via vmap(jacfwd(jacfwd(f)))
T_{a,iP,j,k}— third-derivative slice (N, D, D, D)  via vmap(jacfwd(jacfwd(g)))
              where g(z) = J(z)[:, iP] grabs the iP-column of J.

Because every required triple in _QCLE_TRIPLES has _I_P as its lowest
index, only the iP-fixed slice of the rank-4 third-derivative tensor is
needed, avoiding the full O(D^4) computation.

Coordinate convention  z = (R, P, r₀, r₁, p₀, p₁)
---------------------------------------------------
    _I_R  = 0   nuclear coordinate
    _I_P  = 1   nuclear momentum
    _I_R0 = 2   mapping coordinate r₀
    _I_R1 = 3   mapping coordinate r₁
    _I_P0 = 4   mapping momentum   p₀
    _I_P1 = 5   mapping momentum   p₁
"""

import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Optional, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .Mint import D, PBMEMIntDynamics, pack_z, unpack_z
from .Models import TullyModel, TullyParams

FloatArray = NDArray[np.float64]
StageName  = Literal["z0", "z_open", "z_core", "z1"]


# =============================================================================
# JAX availability
# =============================================================================

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from functools import partial as _partial
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


# =============================================================================
# Coordinate indices
# =============================================================================

_I_R  = 0
_I_P  = 1
_I_R0 = 2
_I_R1 = 3
_I_P0 = 4
_I_P1 = 5

_I_R_MAP: tuple[int, int] = (_I_R0, _I_R1)
_I_P_MAP: tuple[int, int] = (_I_P0, _I_P1)


# =============================================================================
# QCLE-required tensor index sets  (Operator.py must not redeclare these)
# =============================================================================

_QCLE_COLUMNS: tuple[int, ...] = (_I_P, _I_R0, _I_R1, _I_P0, _I_P1)

_QCLE_PAIRS: tuple[tuple[int, int], ...] = (
    (_I_P, _I_R0), (_I_P, _I_R1),
    (_I_R0, _I_R0), (_I_R0, _I_R1), (_I_R1, _I_R1),
    (_I_P, _I_P0), (_I_P, _I_P1),
    (_I_P0, _I_P0), (_I_P0, _I_P1), (_I_P1, _I_P1),
)

_QCLE_TRIPLES: tuple[tuple[int, int, int], ...] = (
    (_I_P, _I_R0, _I_R0), (_I_P, _I_R0, _I_R1), (_I_P, _I_R1, _I_R1),
    (_I_P, _I_P0, _I_P0), (_I_P, _I_P0, _I_P1), (_I_P, _I_P1, _I_P1),
)


# =============================================================================
# Internal helpers
# =============================================================================

def _as_z_batch(z: ArrayLike) -> tuple[FloatArray, bool]:
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != D:
            raise ValueError(f"Expected z.shape == ({D},), got {arr.shape}")
        return arr.reshape(1, D), True
    if arr.ndim == 2:
        if arr.shape[1] != D:
            raise ValueError(f"Expected z.shape == (N,{D}), got {arr.shape}")
        return arr, False
    raise ValueError(f"z must be rank-1 or rank-2, got ndim={arr.ndim}")


def _as_matrix_batch(M: ArrayLike) -> tuple[FloatArray, bool]:
    arr = np.asarray(M, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape != (D, D):
            raise ValueError(f"Expected M.shape == ({D},{D}), got {arr.shape}")
        return arr.reshape(1, D, D), True
    if arr.ndim == 3:
        if arr.shape[1:] != (D, D):
            raise ValueError(f"Expected M.shape == (N,{D},{D}), got {arr.shape}")
        return arr, False
    raise ValueError(f"M must be rank-2 or rank-3, got ndim={arr.ndim}")


def _return_single_or_batch(arr: FloatArray, single: bool) -> FloatArray:
    return arr[0] if single else arr


def _block_slices() -> dict[str, slice]:
    return {"R": slice(0, 1), "P": slice(1, 2), "r": slice(2, 4), "p": slice(4, 6)}


def identity_batch(n: int) -> FloatArray:
    return np.broadcast_to(np.eye(D, dtype=np.float64), (n, D, D)).copy()


def _pair_key(i: int, j: int) -> tuple[int, int]:
    return tuple(sorted((int(i), int(j))))


def _triple_key(i: int, j: int, k: int) -> tuple[int, int, int]:
    return tuple(sorted((int(i), int(j), int(k))))


# =============================================================================
# JAX-traceable MInt step
# =============================================================================

def _build_jax_step_fn(dynamics: PBMEMIntDynamics):
    """
    Return a JAX-traceable single-point MInt step function for the given dynamics.

    Signature:  mint_step(z, tau) -> z'
        z   : (D,) JAX array   current phase-space point
        tau : float             step size  (use tau = -dt/2 for backward half-step)

    The function re-implements PBMEMIntDynamics.step in pure JAX so that
    jax.jacfwd can trace through all operations:
      - the stable sinc integrals  Ic / Is
      - the eigensolver  jnp.linalg.eigh  (exact gradient away from degeneracy)
      - all matrix rotations and the exact momentum kick

    Diabatic potentials V(R) and dV/dR(R) are supplied analytically for all
    three Tully kinds.  For the "dual" model (default pipeline) all functions
    are C^∞ smooth, guaranteeing exact derivatives to any order.

    The returned function should be cached and reused between calls; see
    _get_jax_step_fn for the per-dynamics-instance cache.
    """
    mass = float(dynamics.params.mass)
    hbar = float(dynamics.params.hbar)
    kind = dynamics.model.params.kind
    pA   = float(dynamics.model.params.A)
    pB   = float(dynamics.model.params.B)
    pC   = float(dynamics.model.params.C)
    pD   = float(dynamics.model.params.D)
    pE0  = float(dynamics.model.params.E0)

    def _h_and_dh(R):
        """
        Traceless electronic Hamiltonian h and its R-derivative dh at scalar R.

        Returns (h, dh, dV0) where h = V - V0·I, dh = dV - dV0·I, and dV0
        is the diagonal-mean part of dV (needed for the nuclear force).
        """
        if kind == "dual":
            V11  = 0.0
            V22  = -pA * jnp.exp(-pB * R * R) + pE0
            V12  = pC  * jnp.exp(-pD * R * R)
            dV11 = 0.0
            dV22 = 2.0 * pA * pB * R * jnp.exp(-pB * R * R)
            dV12 = -2.0 * pC * pD * R * jnp.exp(-pD * R * R)
        elif kind == "simple":
            V11  = jnp.where(R >= 0.0, pA * (1.0 - jnp.exp(-pB * R)),
                                       -pA * (1.0 - jnp.exp(pB * R)))
            V22  = -V11
            V12  = pC * jnp.exp(-pD * R * R)
            dV11 = pA * pB * jnp.exp(-pB * jnp.abs(R))
            dV22 = -dV11
            dV12 = -2.0 * pC * pD * R * jnp.exp(-pD * R * R)
        elif kind == "extended":
            V11  = pA
            V22  = -pA
            V12  = pC * jnp.where(R >= 0.0, 2.0 - jnp.exp(-pD * R),
                                             jnp.exp(pD * R))
            dV11 = 0.0
            dV22 = 0.0
            dV12 = pC * pD * jnp.exp(-pD * jnp.abs(R))
        else:
            raise ValueError(f"Unknown Tully kind: {kind!r}")

        V0  = 0.5 * (V11 + V22)
        dV0 = 0.5 * (dV11 + dV22)
        h   = jnp.array([[V11 - V0,  V12     ],
                          [V12,       V22 - V0]])
        dh  = jnp.array([[dV11 - dV0, dV12    ],
                          [dV12,       dV22 - dV0]])
        return h, dh, dV0

    def mint_step(z, tau):
        """
        One MInt step for a single point z of shape (D,).

        Three-stage map:
          1.  Half nuclear drift   R_half = R + (τ/2) P/M
          2.  Frozen-R mapping rotation + exact momentum kick at R_half
          3.  Closing half drift   R1 = R_half + (τ/2) P1/M
        """
        R = z[0];  P = z[1]
        r = z[2:4];  p = z[4:6]

        # --- 1. half nuclear drift ---
        R_half = R + 0.5 * tau * P / mass

        # --- frozen-R electronic objects at R_half ---
        h, dh, dV0 = _h_and_dh(R_half)

        # --- diagonalise h = U diag(E) U^T ---
        E, U = jnp.linalg.eigh(h)           # E: (2,), U: (2,2)

        # --- rotate mapping variables into adiabatic basis ---
        r_e = U.T @ r
        p_e = U.T @ p

        # --- adiabatic-basis rotation by angle θ_a = E_a τ / ℏ ---
        theta = E * tau / hbar
        c = jnp.cos(theta);  s = jnp.sin(theta)
        r_e1 = c * r_e + s * p_e
        p_e1 = c * p_e - s * r_e

        # --- rotate back to diabatic basis ---
        r1 = U @ r_e1
        p1 = U @ p_e1

        # --- exact momentum integral at frozen R ---
        A_mat = U.T @ dh @ U
        C_mat = jnp.outer(r_e, r_e) + jnp.outer(p_e, p_e)
        S_mat = jnp.outer(p_e, r_e) - jnp.outer(r_e, p_e)

        dw = (E[:, None] - E[None, :]) / hbar
        x  = 0.5 * dw * tau
        # jnp.sinc is the normalised sinc:  sinc(y) = sin(πy) / (πy)
        # sinc(x/π) = sin(x) / x  — matches the pipeline's _stable_int convention
        Ic = tau * jnp.cos(x) * jnp.sinc(x / jnp.pi)
        Is = tau * jnp.sin(x) * jnp.sinc(x / jnp.pi)

        I_exact = jnp.sum(A_mat * (C_mat * Ic + S_mat * Is))
        P1 = P - dV0 * tau - 0.5 / hbar * I_exact

        # --- 3. closing half drift ---
        R1 = R_half + 0.5 * tau * P1 / mass

        return jnp.stack([R1, P1, r1[0], r1[1], p1[0], p1[1]])

    return mint_step


# Module-level cache: same JAX function object is reused across calls,
# preserving JAX's JIT compilation cache keyed on abstract input shapes.
_JAX_STEP_FN_CACHE: Dict = {}


def _get_jax_step_fn(dynamics: PBMEMIntDynamics):
    key = (
        dynamics.model.params.kind,
        dynamics.model.params.A, dynamics.model.params.B,
        dynamics.model.params.C, dynamics.model.params.D,
        dynamics.model.params.E0,
        dynamics.params.mass, dynamics.params.hbar,
    )
    if key not in _JAX_STEP_FN_CACHE:
        _JAX_STEP_FN_CACHE[key] = _build_jax_step_fn(dynamics)
    return _JAX_STEP_FN_CACHE[key]


# =============================================================================
# MonodromyTools
# =============================================================================

@dataclass
class MonodromyTools:
    """
    Monodromy matrices and midpoint-map geometry for PBMEMIntDynamics.

    Primary entry point for the QCLE operator
    ------------------------------------------
    midpoint_geometry(Z, dt) → (Y, J_cols, H_pairs, T_triples)
        Computes the geometric tensors of Y(Z) = Φ^0_{-dt/2}(Z) using JAX
        forward-mode autodiff (exact) or FD fallback if JAX is unavailable.
    """

    dynamics: PBMEMIntDynamics

    # =========================================================================
    # JAX autodiff midpoint geometry  (primary path)
    # =========================================================================

    def _midpoint_geometry_jax(
        self,
        Z: FloatArray,
        dt: float,
    ) -> tuple[
        FloatArray,
        Dict[int, FloatArray],
        Dict[tuple[int, int], FloatArray],
        Dict[tuple[int, int, int], FloatArray],
    ]:
        """
        Exact geometric tensors of Y(Z) = Φ^0_{-dt/2}(Z) via JAX autodiff.

        Axis conventions (same as midpoint_required_tensors)
        -----------------------------------------------------
        J_cols[i][n, a]      = J_{a,i}(Z_n)    = ∂Y_a/∂Z_i
        H_pairs[(i,j)][n, a] = H_{a,ij}(Z_n)   = ∂²Y_a/∂Z_i∂Z_j
        T_triples[t][n, a]   = T_{a,t[1],t[2]}  = ∂³Y_a/∂Z_{iP}∂Z_j∂Z_k

        jax.jacfwd axis ordering: jacfwd(f)(z) has shape (out_D, in_D)
        with [a, i] = ∂f_a/∂z_i.  Each additional jacfwd appends a new
        input index as the last dimension.
        """
        zb, single = _as_z_batch(Z)
        Z_jax = jnp.asarray(zb)                     # (N, D) float64

        step_fn  = _get_jax_step_fn(self.dynamics)
        tau      = -0.5 * float(dt)
        f        = _partial(step_fn, tau=tau)        # f : R^D → R^D

        # --- Y = Φ^0_{-dt/2}(Z) ---
        Y_jax = jax.jit(jax.vmap(f))(Z_jax)         # (N, D)

        # --- J_{a,i} : jac_f(z)[a, i] = ∂Y_a/∂Z_i ---
        jac_f = jax.jacfwd(f)
        J_all = jax.jit(jax.vmap(jac_f))(Z_jax)     # (N, D, D)

        # --- H_{a,ij} : hess_f(z)[a, i, j] = ∂²Y_a/∂Z_i∂Z_j ---
        hess_f = jax.jacfwd(jac_f)
        H_all  = jax.jit(jax.vmap(hess_f))(Z_jax)   # (N, D, D, D)

        # --- T_{a, iP, j, k} via iP-column slice ---
        # g(z) = J(z)[:, iP]  shape (D,)  →  ∂Y_a/∂Z_{iP} for each a
        # jacfwd(g)(z)         shape (D,D) →  [a,j] = ∂²Y_a/∂Z_{iP}∂Z_j
        # jacfwd(jacfwd(g))(z) shape (D,D,D)→ [a,j,k] = ∂³Y_a/∂Z_{iP}∂Z_j∂Z_k
        g_fn    = lambda z_: jac_f(z_)[:, _I_P]
        T_iP_fn = jax.jacfwd(jax.jacfwd(g_fn))
        T_iP    = jax.jit(jax.vmap(T_iP_fn))(Z_jax) # (N, D, D, D)

        # --- convert to NumPy and build output dicts ---
        Y    = np.asarray(Y_jax)
        J_np = np.asarray(J_all)    # [n, a, i]
        H_np = np.asarray(H_all)    # [n, a, i, j]
        T_np = np.asarray(T_iP)     # [n, a, j, k] — T_{a, iP, j, k}

        J_cols: Dict[int, FloatArray] = {
            i: J_np[:, :, i] for i in _QCLE_COLUMNS
        }

        H_pairs: Dict[tuple[int, int], FloatArray] = {
            _pair_key(i, j): H_np[:, :, i, j]
            for i, j in _QCLE_PAIRS
        }

        # All _QCLE_TRIPLES are sorted with _I_P first:
        # triple = (_I_P, j, k)  →  T_np[:, :, j, k]
        T_triples: Dict[tuple[int, int, int], FloatArray] = {
            triple: T_np[:, :, triple[1], triple[2]]
            for triple in _QCLE_TRIPLES
        }

        if single:
            Y         = Y[0]
            J_cols    = {k: v[0] for k, v in J_cols.items()}
            H_pairs   = {k: v[0] for k, v in H_pairs.items()}
            T_triples = {k: v[0] for k, v in T_triples.items()}

        return Y, J_cols, H_pairs, T_triples

    # =========================================================================
    # Public QCLE geometry entry point
    # =========================================================================

    def midpoint_geometry(
        self,
        Z: ArrayLike,
        dt: float,
        eps_jac:   float = 1.0e-7,
        eps_hess:  float = 1.0e-5,
        eps_third: float = 1.0e-4,
    ) -> tuple[
        FloatArray,
        Dict[int, FloatArray],
        Dict[tuple[int, int], FloatArray],
        Dict[tuple[int, int, int], FloatArray],
    ]:
        """
        Compute (Y, J_cols, H_pairs, T_triples) for Y(Z) = Φ^0_{-dt/2}(Z).

        Uses JAX forward-mode autodiff when JAX is available (exact, no FD
        truncation error).  Falls back to central finite differences with
        a RuntimeWarning if JAX is not installed.

        The eps_* arguments are accepted for API compatibility but are only
        used in the FD fallback; the autodiff path ignores them.

        Returns
        -------
        Y         : (N, D)
        J_cols    : {i -> (N, D)}       J_{a,i}(Z_n) = ∂Y_a/∂Z_i
        H_pairs   : {(i,j) -> (N, D)}   H_{a,ij}(Z_n)
        T_triples : {(i,j,k) -> (N,D)}  T_{a,ijk}(Z_n)
        """
        if _JAX_AVAILABLE:
            return self._midpoint_geometry_jax(
                np.asarray(Z, dtype=np.float64), dt
            )

        warnings.warn(
            "JAX is not installed — falling back to central finite differences "
            "for midpoint geometry.  Install JAX for exact autodiff derivatives.",
            RuntimeWarning, stacklevel=2,
        )
        return self.midpoint_required_tensors(
            Z, dt=dt,
            columns=_QCLE_COLUMNS,
            pairs=_QCLE_PAIRS,
            triples=_QCLE_TRIPLES,
            eps_jac=eps_jac,
            eps_hess=eps_hess,
            eps_third=eps_third,
        )

    # =========================================================================
    # Low-level FD engine  (fallback / custom analyses)
    # =========================================================================

    def midpoint_required_tensors(
        self,
        z_target:  ArrayLike,
        dt:        float,
        columns:   Iterable[int],
        pairs:     Iterable[tuple[int, int]],
        triples:   Iterable[tuple[int, int, int]],
        eps_jac:   float = 1.0e-7,
        eps_hess:  float = 1.0e-5,
        eps_third: float = 1.0e-4,
    ) -> tuple[
        FloatArray,
        Dict[int, FloatArray],
        Dict[tuple[int, int], FloatArray],
        Dict[tuple[int, int, int], FloatArray],
    ]:
        """
        Central-finite-difference fallback for midpoint geometry tensors.

        Computes the caller-specified subsets of J, H, T for
        Y(Z) = Φ^0_{-dt/2}(Z).  Prefer `midpoint_geometry` for the exact
        autodiff path.
        """
        zb, single = _as_z_batch(z_target)

        col_keys    = tuple(sorted({int(c) for c in columns}))
        pair_keys   = tuple(sorted({_pair_key(*p) for p in pairs}))
        triple_keys = tuple(sorted({_triple_key(*t) for t in triples}))

        cache: Dict[tuple[float, ...], FloatArray] = {}

        def shifted(delta: FloatArray) -> FloatArray:
            key = tuple(float(x) for x in delta)
            out = cache.get(key)
            if out is None:
                out = np.asarray(
                    self.midpoint_map(zb + delta[None, :], dt), dtype=np.float64
                )
                cache[key] = out
            return out

        def unit(axis: int, step: float) -> FloatArray:
            e = np.zeros(D, dtype=np.float64);  e[int(axis)] = float(step)
            return e

        Y = shifted(np.zeros(D, dtype=np.float64))

        J_cols: Dict[int, FloatArray] = {}
        for i in col_keys:
            e = unit(i, eps_jac)
            J_cols[i] = (shifted(e) - shifted(-e)) / (2.0 * eps_jac)

        H_pairs: Dict[tuple[int, int], FloatArray] = {}
        for key in pair_keys:
            i, j = key
            if i == j:
                e = unit(i, eps_hess)
                H_pairs[key] = (shifted(e) - 2.0 * Y + shifted(-e)) / eps_hess ** 2
            else:
                ei = unit(i, eps_hess);  ej = unit(j, eps_hess)
                H_pairs[key] = (
                    shifted( ei + ej) - shifted( ei - ej)
                    - shifted(-ei + ej) + shifted(-ei - ej)
                ) / (4.0 * eps_hess ** 2)

        T_triples: Dict[tuple[int, int, int], FloatArray] = {}
        for key in triple_keys:
            counts = Counter(key)
            if len(counts) == 1:
                i = key[0];  e = unit(i, eps_third)
                T_triples[key] = (
                    shifted(2.0*e) - 2.0*shifted(e)
                    + 2.0*shifted(-e) - shifted(-2.0*e)
                ) / (2.0 * eps_third ** 3)
            elif len(counts) == 2:
                repeated = next(a for a, cnt in counts.items() if cnt == 2)
                distinct  = next(a for a, cnt in counts.items() if cnt == 1)
                er = unit(repeated, eps_third);  ed = unit(distinct, eps_third)
                d2p = (shifted( ed+er) - 2.0*shifted( ed) + shifted( ed-er)) / eps_third**2
                d2m = (shifted(-ed+er) - 2.0*shifted(-ed) + shifted(-ed-er)) / eps_third**2
                T_triples[key] = (d2p - d2m) / (2.0 * eps_third)
            else:
                i, j, k = key
                ei = unit(i, eps_third);  ej = unit(j, eps_third);  ek = unit(k, eps_third)
                T_triples[key] = (
                      shifted(+ei+ej+ek) - shifted(+ei+ej-ek)
                    - shifted(+ei-ej+ek) + shifted(+ei-ej-ek)
                    - shifted(-ei+ej+ek) + shifted(-ei+ej-ek)
                    + shifted(-ei-ej+ek) - shifted(-ei-ej-ek)
                ) / (8.0 * eps_third ** 3)

        if single:
            Y         = Y[0]
            J_cols    = {k: v[0] for k, v in J_cols.items()}
            H_pairs   = {k: v[0] for k, v in H_pairs.items()}
            T_triples = {k: v[0] for k, v in T_triples.items()}

        return Y, J_cols, H_pairs, T_triples

    # =========================================================================
    # Split-step stage maps  (diagnostics)
    # =========================================================================

    def half_drift_map(self, z: ArrayLike, dt: float, sign: int = +1) -> FloatArray:
        zb, single = _as_z_batch(z)
        tau = 0.5 * float(sign) * float(dt) / self.dynamics.params.mass
        z1 = zb.copy();  z1[:, 0] = zb[:, 0] + tau * zb[:, 1]
        return _return_single_or_batch(z1, single)

    def half_drift_jacobian(self, dt: float, sign: int = +1,
                            n_batch: int = 1) -> FloatArray:
        tau = 0.5 * float(sign) * float(dt) / self.dynamics.params.mass
        J = identity_batch(n_batch);  J[:, 0, 1] = tau
        return J[0] if n_batch == 1 else J

    def core_map(self, z_open: ArrayLike, dt: float) -> FloatArray:
        zb, single = _as_z_batch(z_open)
        hbar = self.dynamics.params.hbar
        R = zb[:, 0];  P = zb[:, 1]
        r = zb[:, 2:4];  p = zb[:, 4:6]
        V0, h, dV0, dh = self.dynamics._frozen_R_objects(R)
        E, U = np.linalg.eigh(h);  U_T = np.swapaxes(U, 1, 2)
        r_e = np.einsum("nij,nj->ni", U_T, r)
        p_e = np.einsum("nij,nj->ni", U_T, p)
        theta = E * dt / hbar;  c = np.cos(theta);  s = np.sin(theta)
        r_e1 = c*r_e + s*p_e;  p_e1 = c*p_e - s*r_e
        r1 = np.einsum("nij,nj->ni", U, r_e1)
        p1 = np.einsum("nij,nj->ni", U, p_e1)
        A = np.einsum("nij,njk,nkl->nil", U_T, dh, U)
        C = r_e[:,:,None]*r_e[:,None,:] + p_e[:,:,None]*p_e[:,None,:]
        S = p_e[:,:,None]*r_e[:,None,:] - r_e[:,:,None]*p_e[:,None,:]
        dw = (E[:,:,None] - E[:,None,:]) / hbar
        Ic = self.dynamics._stable_int_cos(dw, dt)
        Is = self.dynamics._stable_int_sin(dw, dt)
        I_exact = np.sum(A * (C*Ic + S*Is), axis=(1, 2))
        P1 = P - dV0*dt - 0.5/hbar * I_exact
        z1 = zb.copy();  z1[:,1]=P1;  z1[:,2:4]=r1;  z1[:,4:6]=p1
        return _return_single_or_batch(z1, single)

    def core_jacobian(self, z_open: ArrayLike, dt: float,
                      eps: float = 1e-7) -> FloatArray:
        zb, single = _as_z_batch(z_open)
        def jac_single(zs: FloatArray) -> FloatArray:
            J = np.zeros((D, D), dtype=np.float64)
            for j in range(D):
                dz = np.zeros(D, dtype=np.float64);  dz[j] = eps
                J[:, j] = (np.asarray(self.core_map(zs+dz, dt), dtype=np.float64)
                           - np.asarray(self.core_map(zs-dz, dt), dtype=np.float64)
                           ) / (2.0 * eps)
            return J
        Jb = np.stack([jac_single(zs) for zs in zb], axis=0)
        return _return_single_or_batch(Jb, single)

    def split_step_states(self, z0: ArrayLike,
                          dt: float) -> Dict[StageName, FloatArray]:
        z_open = self.half_drift_map(z0, dt, sign=+1)
        z_core = self.core_map(z_open, dt)
        z1     = self.half_drift_map(z_core, dt, sign=+1)
        return {
            "z0":     np.asarray(z0,     dtype=np.float64),
            "z_open": np.asarray(z_open, dtype=np.float64),
            "z_core": np.asarray(z_core, dtype=np.float64),
            "z1":     np.asarray(z1,     dtype=np.float64),
        }

    def split_step_jacobians(self, z0: ArrayLike, dt: float,
                             eps: float = 1e-7) -> Dict[str, FloatArray]:
        z0b, single = _as_z_batch(z0);  n = z0b.shape[0]
        z_open  = self.half_drift_map(z0b, dt, sign=+1)
        J_open  = self.half_drift_jacobian(dt=dt, sign=+1, n_batch=n)
        J_core  = self.core_jacobian(z_open, dt=dt, eps=eps)
        J_close = self.half_drift_jacobian(dt=dt, sign=+1, n_batch=n)
        return {
            "J_open":  J_open[0]  if single else J_open,
            "J_core":  J_core,
            "J_close": J_close[0] if single else J_close,
        }

    # =========================================================================
    # Backward half-step map
    # =========================================================================

    def midpoint_map(self, z_target: ArrayLike, dt: float) -> FloatArray:
        """Y(Z) = Φ^0_{-dt/2}(Z) — the full backward MInt half-step."""
        zb, single = _as_z_batch(z_target)
        Y = np.asarray(self.dynamics.step(zb, -0.5*float(dt)), dtype=np.float64)
        return _return_single_or_batch(Y, single)

    # =========================================================================
    # Full-step monodromy utilities
    # =========================================================================

    def one_step_forward_monodromy(self, z0: ArrayLike, dt: float,
                                   eps: float = 1e-7) -> FloatArray:
        z0b, single = _as_z_batch(z0);  n = z0b.shape[0]
        fac     = self.split_step_jacobians(z0b, dt=dt, eps=eps)
        J_open  = fac["J_open"];  J_core = fac["J_core"];  J_close = fac["J_close"]
        if J_open.ndim  == 2:  J_open  = np.broadcast_to(J_open,  (n, D, D))
        if J_close.ndim == 2:  J_close = np.broadcast_to(J_close, (n, D, D))
        M = np.einsum("nij,njk,nkl->nil", J_close, J_core, J_open)
        return _return_single_or_batch(M, single)

    def one_step_backward_monodromy(self, z0: ArrayLike, dt: float,
                                    eps: float = 1e-7) -> FloatArray:
        return np.linalg.inv(np.asarray(
            self.one_step_forward_monodromy(z0, dt=dt, eps=eps), dtype=np.float64
        ))

    def stage_forward_monodromies(self, z0: ArrayLike, dt: float,
                                  eps: float = 1e-7,
                                  ) -> Dict[Tuple[StageName, StageName], FloatArray]:
        z0b, single = _as_z_batch(z0);  n = z0b.shape[0]
        fac     = self.split_step_jacobians(z0b, dt=dt, eps=eps)
        J_open  = fac["J_open"];  J_core = fac["J_core"];  J_close = fac["J_close"]
        if J_open.ndim  == 2:  J_open  = np.broadcast_to(J_open,  (n, D, D))
        if J_close.ndim == 2:  J_close = np.broadcast_to(J_close, (n, D, D))
        I = identity_batch(n)
        out = {
            ("z0",     "z0"):     I,
            ("z0",     "z_open"): J_open,
            ("z0",     "z_core"): np.einsum("nij,njk->nik", J_core, J_open),
            ("z0",     "z1"):     np.einsum("nij,njk,nkl->nil", J_close, J_core, J_open),
            ("z_open", "z_core"): J_core,
            ("z_open", "z1"):     np.einsum("nij,njk->nik", J_close, J_core),
            ("z_core", "z1"):     J_close,
        }
        if single: out = {k: v[0] for k, v in out.items()}
        return out

    def stage_backward_monodromies(self, z0: ArrayLike, dt: float,
                                   eps: float = 1e-7,
                                   ) -> Dict[Tuple[StageName, StageName], FloatArray]:
        fwd = self.stage_forward_monodromies(z0, dt=dt, eps=eps)
        return {
            (b, a): np.linalg.inv(np.asarray(M, dtype=np.float64))
            for (a, b), M in fwd.items() if a != b
        }

    def trajectory_forward_monodromies(
        self, z0: ArrayLike, dt: float, n_steps: int, eps: float = 1e-7,
    ) -> tuple[FloatArray, FloatArray]:
        traj = self.dynamics.propagate(z0, dt=dt, n_steps=n_steps)
        if traj.ndim == 2:
            M = np.zeros((n_steps+1, D, D), dtype=np.float64)
            M[0] = np.eye(D, dtype=np.float64)
            for n in range(n_steps):
                M[n+1] = self.dynamics.compute_step_jacobian(traj[n], dt=dt, eps=eps) @ M[n]
            return traj, M
        N = traj.shape[1]
        M = np.zeros((n_steps+1, N, D, D), dtype=np.float64)
        M[0] = np.broadcast_to(np.eye(D, dtype=np.float64), (N, D, D))
        for n in range(n_steps):
            Jn = self.dynamics.compute_step_jacobian(traj[n], dt=dt, eps=eps)
            M[n+1] = np.einsum("nij,njk->nik", Jn, M[n])
        return traj, M

    def trajectory_backward_monodromies(
        self, z0: ArrayLike, dt: float, n_steps: int, eps: float = 1e-7,
    ) -> tuple[FloatArray, FloatArray]:
        traj, M_fwd = self.trajectory_forward_monodromies(z0, dt=dt, n_steps=n_steps, eps=eps)
        return traj, np.linalg.inv(M_fwd)

    # =========================================================================
    # Block extraction and symplectic diagnostics
    # =========================================================================

    def extract_block(self, M: ArrayLike, row: str, col: str) -> FloatArray:
        sl = _block_slices()
        if row not in sl or col not in sl:
            raise ValueError(f"row/col must be among {tuple(sl.keys())}")
        Mb, single = _as_matrix_batch(M)
        return _return_single_or_batch(Mb[:, sl[row], sl[col]], single)

    def extract_all_blocks(self, M: ArrayLike) -> Dict[str, FloatArray]:
        sl = _block_slices()
        return {f"{r}{c}": self.extract_block(M, r, c) for r in sl for c in sl}

    def omega_matrix(self) -> FloatArray:
        return self.dynamics.omega_matrix()

    def symplectic_residual(self, M: ArrayLike) -> FloatArray:
        Mb, single = _as_matrix_batch(M)
        Omega = self.omega_matrix()
        res = (
            np.einsum("nij,jk,nkl->nil", np.swapaxes(Mb, 1, 2), Omega, Mb)
            - Omega[None, :, :]
        )
        return _return_single_or_batch(res, single)

    def determinant(self, M: ArrayLike) -> FloatArray:
        Mb, single = _as_matrix_batch(M)
        det = np.linalg.det(Mb)
        return float(det[0]) if single else det


# =============================================================================
# NumPy/JAX step consistency check
# =============================================================================

def check_mint_jax_consistency(
    dynamics: PBMEMIntDynamics,
    dt:        float = 0.5,
    n_samples: int   = 64,
    atol:      float = 1e-12,
    rtol:      float = 1e-10,
    seed:      int   = 0,
) -> dict:
    """
    Verify PBMEMIntDynamics.step (NumPy) == _build_jax_step_fn (JAX) on a
    batch of random phase-space points.  Raises AssertionError on mismatch.

    Designed to be called once at simulation startup; O(n_samples · D) cost.
    """
    if not _JAX_AVAILABLE:
        return {"available": False, "ok": None}

    rng = np.random.default_rng(seed)
    R = rng.uniform(-8.0, 8.0, size=n_samples)
    P = rng.uniform(-40.0, 40.0, size=n_samples)
    r = rng.normal(size=(n_samples, 2))
    p = rng.normal(size=(n_samples, 2))
    Z = pack_z(R, P, r, p)

    Z_np  = np.asarray(dynamics.step(Z, dt), dtype=np.float64)

    step_fn = _get_jax_step_fn(dynamics)
    f       = _partial(step_fn, tau=float(dt))
    Z_jax   = np.asarray(jax.jit(jax.vmap(f))(jnp.asarray(Z)))

    # The final propagated Z is invariant under column-sign flips of U
    # (since U and U^T bracket every mapping-variable operation), so
    # sign differences between np.linalg.eigh and jnp.linalg.eigh do NOT
    # cause a numeric mismatch in Z.  The only failure mode is eigenvalue
    # SWAPPING at near-degenerate R, which would change the rotation
    # angles theta = E * dt / hbar.  np.linalg.eigh and jnp.linalg.eigh
    # both guarantee ascending order, so swaps cannot happen as long as
    # the two solvers agree on the dominant eigenvector when eigenvalues
    # differ by < machine epsilon.  For all three Tully models the gap
    # |E_1 - E_0| >= 2|V12| > 0 everywhere, so this check is exact.
    # If a future model has a true degeneracy, this test will correctly
    # fire and the gap column of the AssertionError will show which R
    # triggered it.

    diff    = np.abs(Z_np - Z_jax)
    tol     = atol + rtol * np.abs(Z_np)
    max_abs = float(diff.max())
    max_rel = float((diff / (np.abs(Z_np) + atol)).max())

    if not np.all(diff <= tol):
        k = int(np.argmax(diff - tol))
        i, j = np.unravel_index(k, diff.shape)
        raise AssertionError(
            f"MInt NumPy/JAX mismatch at (sample={i}, dim={j}): "
            f"np={Z_np[i,j]:+.16e}  jax={Z_jax[i,j]:+.16e}  "
            f"|diff|={diff[i,j]:.3e}  tol={tol[i,j]:.3e}\n"
            f"kind={dynamics.model.params.kind!r}, dt={dt}"
        )

    return {"available": True, "ok": True, "n_samples": n_samples,
            "max_abs_diff": max_abs, "max_rel_diff": max_rel}


# =============================================================================
# Human-readable monodromy / symplectic tests
# =============================================================================

def _format_matrix(M: ArrayLike, precision: int = 6) -> str:
    arr = np.asarray(M, dtype=np.float64)
    return np.array2string(arr, precision=precision, suppress_small=False)


def midpoint_full_jacobian(
    tools: MonodromyTools,
    z_target: ArrayLike,
    dt: float,
    eps: float = 1.0e-7,
) -> FloatArray:
    r"""
    Full Jacobian of the backward half-step map

        Y(Z) = \Phi^0_{-dt/2}(Z).

    Returns
    -------
    J_mid : (D, D) or (N, D, D)
        J_mid[a, i] = \partial Y_a / \partial Z_i.
    """
    zb, single = _as_z_batch(z_target)

    if _JAX_AVAILABLE:
        Z_jax = jnp.asarray(zb)
        step_fn = _get_jax_step_fn(tools.dynamics)
        tau = -0.5 * float(dt)
        f = _partial(step_fn, tau=tau)
        jac_f = jax.jacfwd(f)
        J_all = np.asarray(jax.jit(jax.vmap(jac_f))(Z_jax), dtype=np.float64)
        return J_all[0] if single else J_all

    # finite-difference fallback
    def jac_single(zs: FloatArray) -> FloatArray:
        J = np.zeros((D, D), dtype=np.float64)
        for i in range(D):
            dz = np.zeros(D, dtype=np.float64)
            dz[i] = eps
            fp = np.asarray(tools.midpoint_map(zs + dz, dt), dtype=np.float64)
            fm = np.asarray(tools.midpoint_map(zs - dz, dt), dtype=np.float64)
            J[:, i] = (fp - fm) / (2.0 * eps)
        return J

    Jb = np.stack([jac_single(zs) for zs in zb], axis=0)
    return Jb[0] if single else Jb


def test_one_step_monodromy_matrix(
    dynamics: PBMEMIntDynamics,
    z0: Optional[ArrayLike] = None,
    dt: float = 0.5,
    eps: float = 1.0e-7,
    precision: int = 6,
    sym_tol: Optional[float] = 1.0e-6,
    det_tol: Optional[float] = 1.0e-6,
) -> dict:
    r"""
    Print the full one-step forward monodromy matrix and its symplectic diagnostics.

    This is the matrix

        M^+(z_0; dt) = \partial \Phi^0_{dt}(z_0) / \partial z_0.

    The test reports
        - the propagated point z_1,
        - the monodromy matrix M,
        - the symplectic residual matrix M^T Ω M - Ω,
        - ||M^T Ω M - Ω||_F,
        - det(M).
    """
    tools = MonodromyTools(dynamics)
    if z0 is None:
        z0 = pack_z(
            R=-10.0,
            P=30.0,
            r=np.array([np.sqrt(2.0), 0.0], dtype=np.float64),
            p=np.array([np.sqrt(2.0), 0.0], dtype=np.float64),
        )

    z0_arr = np.asarray(z0, dtype=np.float64)
    z1 = np.asarray(dynamics.step(z0_arr, dt), dtype=np.float64)
    M = np.asarray(tools.one_step_forward_monodromy(z0_arr, dt=dt, eps=eps), dtype=np.float64)
    Omega = tools.omega_matrix()
    resid = np.asarray(tools.symplectic_residual(M), dtype=np.float64)
    resid_fro = float(np.linalg.norm(resid, ord="fro"))
    detM = float(np.linalg.det(M))

    print("[one-step forward monodromy test]")
    print(f"dt = {dt:.6f}, eps = {eps:.2e}")
    print("z0 =", _format_matrix(z0_arr, precision=precision))
    print("z1 =", _format_matrix(z1, precision=precision))
    print("Omega =")
    print(_format_matrix(Omega, precision=precision))
    print("M = dPhi_dt/dz0 =")
    print(_format_matrix(M, precision=precision))
    print("M^T Omega M - Omega =")
    print(_format_matrix(resid, precision=precision))
    print(f"||M^T Omega M - Omega||_F = {resid_fro:.6e}")
    print(f"det(M) = {detM:.12e}   |det(M)-1| = {abs(detM - 1.0):.6e}")

    if sym_tol is not None and resid_fro > float(sym_tol):
        raise AssertionError(
            f"Forward monodromy is not sufficiently symplectic: "
            f"||M^TΩM-Ω||_F={resid_fro:.3e} > {float(sym_tol):.3e}"
        )
    if det_tol is not None and abs(detM - 1.0) > float(det_tol):
        raise AssertionError(
            f"Forward monodromy determinant deviates too much from 1: "
            f"|det(M)-1|={abs(detM - 1.0):.3e} > {float(det_tol):.3e}"
        )

    return {
        "z0": z0_arr,
        "z1": z1,
        "M": M,
        "Omega": Omega,
        "symplectic_residual": resid,
        "symplectic_fro": resid_fro,
        "determinant": detM,
    }


def test_midpoint_jacobian_symplecticity(
    dynamics: PBMEMIntDynamics,
    z_target: Optional[ArrayLike] = None,
    dt: float = 0.5,
    eps: float = 1.0e-7,
    precision: int = 6,
    sym_tol: Optional[float] = 1.0e-6,
    det_tol: Optional[float] = 1.0e-6,
) -> dict:
    r"""
    Print the full Jacobian of the backward half-step map and test its symplecticity.

    The map is

        Y(Z) = \Phi^0_{-dt/2}(Z),

    and the test reports the matrix

        J_mid(Z) = \partial Y(Z) / \partial Z,

    together with the symplectic residual and determinant.
    """
    tools = MonodromyTools(dynamics)
    if z_target is None:
        z_target = pack_z(
            R=-9.5,
            P=28.0,
            r=np.array([1.10, -0.35], dtype=np.float64),
            p=np.array([0.65, 0.20], dtype=np.float64),
        )

    Z = np.asarray(z_target, dtype=np.float64)
    Y = np.asarray(tools.midpoint_map(Z, dt), dtype=np.float64)
    J_mid = np.asarray(midpoint_full_jacobian(tools, Z, dt=dt, eps=eps), dtype=np.float64)
    Omega = tools.omega_matrix()
    resid = np.asarray(tools.symplectic_residual(J_mid), dtype=np.float64)
    resid_fro = float(np.linalg.norm(resid, ord="fro"))
    detJ = float(np.linalg.det(J_mid))

    print("[backward half-step midpoint Jacobian test]")
    print(f"dt = {dt:.6f}, eps = {eps:.2e}, jax_available = {_JAX_AVAILABLE}")
    print("Z =", _format_matrix(Z, precision=precision))
    print("Y(Z) =", _format_matrix(Y, precision=precision))
    print("Omega =")
    print(_format_matrix(Omega, precision=precision))
    print("J_mid = dPhi_{-dt/2}/dZ =")
    print(_format_matrix(J_mid, precision=precision))
    print("J_mid^T Omega J_mid - Omega =")
    print(_format_matrix(resid, precision=precision))
    print(f"||J_mid^T Omega J_mid - Omega||_F = {resid_fro:.6e}")
    print(f"det(J_mid) = {detJ:.12e}   |det(J_mid)-1| = {abs(detJ - 1.0):.6e}")

    if sym_tol is not None and resid_fro > float(sym_tol):
        raise AssertionError(
            f"Backward half-step Jacobian is not sufficiently symplectic: "
            f"||J^TΩJ-Ω||_F={resid_fro:.3e} > {float(sym_tol):.3e}"
        )
    if det_tol is not None and abs(detJ - 1.0) > float(det_tol):
        raise AssertionError(
            f"Backward half-step determinant deviates too much from 1: "
            f"|det(J)-1|={abs(detJ - 1.0):.3e} > {float(det_tol):.3e}"
        )

    return {
        "Z": Z,
        "Y": Y,
        "J_mid": J_mid,
        "Omega": Omega,
        "symplectic_residual": resid,
        "symplectic_fro": resid_fro,
        "determinant": detJ,
    }


def test_monodromy_over_short_trajectory(
    dynamics: PBMEMIntDynamics,
    z0: Optional[ArrayLike] = None,
    dt: float = 0.5,
    n_steps: int = 10,
    eps: float = 1.0e-7,
    sym_tol: Optional[float] = 1.0e-5,
    det_tol: Optional[float] = 1.0e-5,
) -> dict:
    r"""
    Check the full-step forward monodromy along a short trajectory.

    Reports the maximum symplectic residual norm and maximum |det(M_n)-1|
    for the cumulative monodromy

        M_n = \partial z_n / \partial z_0.
    """
    tools = MonodromyTools(dynamics)
    if z0 is None:
        z0 = pack_z(
            R=-10.0,
            P=30.0,
            r=np.array([np.sqrt(2.0), 0.0], dtype=np.float64),
            p=np.array([np.sqrt(2.0), 0.0], dtype=np.float64),
        )

    traj, M_traj = tools.trajectory_forward_monodromies(z0, dt=dt, n_steps=n_steps, eps=eps)
    Omega = tools.omega_matrix()
    if M_traj.ndim == 3:
        resid = np.einsum("nij,jk,nkl->nil", np.swapaxes(M_traj, 1, 2), Omega, M_traj) - Omega[None, :, :]
        resid_fro = np.linalg.norm(resid.reshape(n_steps + 1, -1), axis=1)
        dets = np.linalg.det(M_traj)
    else:
        raise ValueError("test_monodromy_over_short_trajectory expects a single initial point.")

    print("[short-trajectory monodromy test]")
    print(f"dt = {dt:.6f}, n_steps = {n_steps}, eps = {eps:.2e}")
    print(f"max_n ||M_n^T Omega M_n - Omega||_F = {float(np.max(resid_fro)):.6e}")
    print(f"max_n |det(M_n)-1| = {float(np.max(np.abs(dets - 1.0))):.6e}")

    if sym_tol is not None and float(np.max(resid_fro)) > float(sym_tol):
        raise AssertionError(
            f"Trajectory monodromy loses symplecticity beyond tolerance: "
            f"max ||M_n^TΩM_n-Ω||_F={float(np.max(resid_fro)):.3e} > {float(sym_tol):.3e}"
        )
    if det_tol is not None and float(np.max(np.abs(dets - 1.0))) > float(det_tol):
        raise AssertionError(
            f"Trajectory monodromy determinant drifts too much: "
            f"max |det(M_n)-1|={float(np.max(np.abs(dets - 1.0))):.3e} > {float(det_tol):.3e}"
        )

    return {
        "trajectory": traj,
        "monodromies": M_traj,
        "symplectic_fro_by_step": resid_fro,
        "determinant_by_step": dets,
    }


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=False)

    dyn = PBMEMIntDynamics(model=TullyModel(TullyParams.defaults("dual")))

    print("[run] MInt np/jax consistency:", check_mint_jax_consistency(dyn))
    print()
    test_one_step_monodromy_matrix(dyn, dt=0.5, eps=1.0e-7, precision=6)
    print()
    test_midpoint_jacobian_symplecticity(dyn, dt=0.5, eps=1.0e-7, precision=6)
    print()
    test_monodromy_over_short_trajectory(dyn, dt=0.5, n_steps=10, eps=1.0e-7)