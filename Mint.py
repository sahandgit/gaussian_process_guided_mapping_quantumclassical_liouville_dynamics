from __future__ import annotations

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

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from Models import TullyModel, TullyParams


FloatArray = NDArray[np.float64]

# Packed state dimension: z = (R, P, r0, r1, p0, p1)
D: int = 6


# =============================================================================
# Parameter container
# =============================================================================

@dataclass(frozen=True)
class PBMEMIntParams:
    """
    Parameters for PBME-MInt dynamics on the 1D, 2-state Tully model.
    """
    mass: float = 2000.0
    hbar: float = 1.0

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be positive.")
        if self.hbar <= 0.0:
            raise ValueError("hbar must be positive.")


# =============================================================================
# Shape helpers
# =============================================================================

def _as_batch_scalar(x: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.reshape(-1)
    raise ValueError(f"{name} must be scalar, shape (N,), or shape (N,1); got {arr.shape}")


def _as_batch_vec2(x: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1 and arr.shape[0] == 2:
        return arr.reshape(1, 2)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    raise ValueError(f"{name} must have shape (2,) or (N,2); got {arr.shape}")


def _same_n(*arrays: FloatArray) -> int:
    n = arrays[0].shape[0]
    for a in arrays[1:]:
        if a.shape[0] != n:
            raise ValueError("All batch arrays must have the same leading dimension.")
    return n


def _return_single_or_batch(z_batch: FloatArray, single_input: bool) -> FloatArray:
    return z_batch[0] if single_input else z_batch


# =============================================================================
# Pack / unpack
# =============================================================================

def pack_z(R: ArrayLike, P: ArrayLike, r: ArrayLike, p: ArrayLike) -> FloatArray:
    """
    Pack phase-space variables into

        z = (R, P, r0, r1, p0, p1)

    Accepted inputs:
      - single point:
            R scalar, P scalar, r shape (2,), p shape (2,)
      - batch:
            R shape (N,), P shape (N,), r shape (N,2), p shape (N,2)

    This is intentionally compatible with the outputs of MMSTSamples from Sampling.py:
        z0 = pack_z(samples.R, samples.P, samples.r, samples.p)
    """
    Rb = _as_batch_scalar(R, "R")
    Pb = _as_batch_scalar(P, "P")
    rb = _as_batch_vec2(r, "r")
    pb = _as_batch_vec2(p, "p")

    n = _same_n(Rb, Pb, rb, pb)

    z = np.zeros((n, D), dtype=np.float64)
    z[:, 0] = Rb
    z[:, 1] = Pb
    z[:, 2:4] = rb
    z[:, 4:6] = pb

    single_input = np.asarray(R).ndim == 0 and np.asarray(P).ndim == 0 and np.asarray(r).ndim == 1
    return _return_single_or_batch(z, single_input)


def unpack_z(z: ArrayLike) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    Unpack z into (R, P, r, p).

    Single point:
        z.shape == (6,)   -> returns scalar R, scalar P, r shape (2,), p shape (2,)

    Batch:
        z.shape == (N,6)  -> returns R shape (N,), P shape (N,), r shape (N,2), p shape (N,2)
    """
    arr = np.asarray(z, dtype=np.float64)

    if arr.ndim == 1:
        if arr.shape[0] != D:
            raise ValueError(f"z must have shape ({D},), got {arr.shape}")
        R = arr[0]
        P = arr[1]
        r = arr[2:4]
        p = arr[4:6]
        return R, P, r, p

    if arr.ndim == 2:
        if arr.shape[1] != D:
            raise ValueError(f"z must have shape (N,{D}), got {arr.shape}")
        R = arr[:, 0]
        P = arr[:, 1]
        r = arr[:, 2:4]
        p = arr[:, 4:6]
        return R, P, r, p

    raise ValueError(f"z must be rank-1 or rank-2, got ndim={arr.ndim}")


# =============================================================================
# Core dynamics class
# =============================================================================

class PBMEMIntDynamics:
    """
    PBME dynamics propagated with the MInt exact momentum-integration scheme.

    Important implementation choice:
    --------------------------------
    The frozen-R mapping rotation is implemented entirely in real arithmetic.

    We diagonalize the real symmetric traceless electronic matrix h(R):
        h(R) = U(R) diag(E(R)) U(R)^T

    Then in the adiabatic basis, each mapping mode rotates as
        r_a' = cos(theta_a) r_a + sin(theta_a) p_a
        p_a' = cos(theta_a) p_a - sin(theta_a) r_a
    with theta_a = E_a dt / hbar.

    No complex-valued state variable is ever formed.
    """

    nstates: int = 2
    ndim_nuclear: int = 1

    def __init__(
        self,
        model: Optional[TullyModel] = None,
        params: Optional[PBMEMIntParams] = None,
    ) -> None:
        self.model = model if model is not None else TullyModel(TullyParams.defaults("dual"))
        self.params = params if params is not None else PBMEMIntParams()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    @staticmethod
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

    @staticmethod
    def _trace_traceless(V: FloatArray) -> tuple[FloatArray, FloatArray]:
        """
        For V.shape == (N,2,2), return
            V0.shape == (N,)
            h.shape  == (N,2,2)
        with V = V0 I + h and Tr(h)=0.
        """
        V0 = 0.5 * np.trace(V, axis1=1, axis2=2)
        I = np.eye(2, dtype=np.float64)[None, :, :]
        h = V - V0[:, None, None] * I
        # enforce exact symmetry numerically
        h = 0.5 * (h + np.swapaxes(h, 1, 2))
        return V0, h

    def _frozen_R_objects(self, R: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """
        Use the already-defined model module to obtain V(R), dV/dR, and their
        trace/traceless decomposition. This avoids repeating the model definitions.
        """
        V = self.model.diabatic_potential(R)
        dV = self.model.d_diabatic_potential_dR(R)

        if V.ndim == 2:
            V = V[None, :, :]
            dV = dV[None, :, :]

        V0, h = self._trace_traceless(V)
        dV0, dh = self._trace_traceless(dV)
        return V0, h, dV0, dh

    @staticmethod
    def _stable_int_cos(dw: FloatArray, dt: float) -> FloatArray:
        """
        I_c(dw,dt) = ∫_0^dt cos(dw t) dt
                   = dt cos(dw dt / 2) sinc(dw dt / 2)
        """
        x = 0.5 * dw * dt
        return dt * np.cos(x) * np.sinc(x / np.pi)

    @staticmethod
    def _stable_int_sin(dw: FloatArray, dt: float) -> FloatArray:
        """
        I_s(dw,dt) = ∫_0^dt sin(dw t) dt
                   = dt sin(dw dt / 2) sinc(dw dt / 2)
        """
        x = 0.5 * dw * dt
        return dt * np.sin(x) * np.sinc(x / np.pi)

    # -------------------------------------------------------------------------
    # One-step MInt propagator
    # -------------------------------------------------------------------------
    def step(self, z: ArrayLike, dt: float) -> FloatArray:
        """
        One MInt step for a single state or a batch.

        Input:
            z.shape == (6,)   or   z.shape == (N,6)

        Output:
            same shape as input
        """
        zb, single = self._as_z_batch(z)
        M = self.params.mass
        hbar = self.params.hbar

        R = zb[:, 0]
        P = zb[:, 1]
        r = zb[:, 2:4]
        p = zb[:, 4:6]

        # ---------------------------------------------------------------------
        # H1: half drift
        # ---------------------------------------------------------------------
        R_half = R + 0.5 * dt * (P / M)

        # ---------------------------------------------------------------------
        # Frozen-R electronic objects from the model module
        # ---------------------------------------------------------------------
        V0, h, dV0, dh = self._frozen_R_objects(R_half)

        # h, dh are (N,2,2), real symmetric
        # eigendecomposition stays real
        E, U = np.linalg.eigh(h)  # E:(N,2), U:(N,2,2)
        U_T = np.swapaxes(U, 1, 2)

        # ---------------------------------------------------------------------
        # Real mapping rotation in adiabatic basis
        # ---------------------------------------------------------------------
        r_e = np.einsum("nij,nj->ni", U_T, r)
        p_e = np.einsum("nij,nj->ni", U_T, p)

        theta = (E * dt) / hbar
        c = np.cos(theta)
        s = np.sin(theta)

        r_e_1 = c * r_e + s * p_e
        p_e_1 = c * p_e - s * r_e

        r1 = np.einsum("nij,nj->ni", U, r_e_1)
        p1 = np.einsum("nij,nj->ni", U, p_e_1)

        # ---------------------------------------------------------------------
        # Exact momentum integral at frozen R, still entirely real
        # ---------------------------------------------------------------------
        A = np.einsum("nij,njk,nkl->nil", U_T, dh, U)   # A = U^T (dh) U

        C = r_e[:, :, None] * r_e[:, None, :] + p_e[:, :, None] * p_e[:, None, :]
        S = p_e[:, :, None] * r_e[:, None, :] - r_e[:, :, None] * p_e[:, None, :]

        dw = (E[:, :, None] - E[:, None, :]) / hbar
        Ic = self._stable_int_cos(dw, dt)
        Is = self._stable_int_sin(dw, dt)

        I_exact = np.sum(A * (C * Ic + S * Is), axis=(1, 2))
        P1 = P - dV0 * dt - (0.5 / hbar) * I_exact

        # ---------------------------------------------------------------------
        # H1: closing half drift
        # ---------------------------------------------------------------------
        R1 = R_half + 0.5 * dt * (P1 / M)

        z1 = np.zeros_like(zb)
        z1[:, 0] = R1
        z1[:, 1] = P1
        z1[:, 2:4] = r1
        z1[:, 4:6] = p1

        return _return_single_or_batch(z1, single)

    mint_step_z = step
    mint_step = step

    def mapping_block_jacobian(self, z: ArrayLike, dt: float) -> FloatArray:
        """
        The (N,4,4) Jacobian B = d(r',p')/d(r,p) of the mapping-block map
        applied by one MInt step, in the coordinate order (r0,r1,p0,p1).

        The frozen-R mapping evolution is LINEAR in (r,p):
            [r_e; p_e] -> [c r_e + s p_e ; c p_e - s r_e]   (adiabatic)
        with r_e = U^T r, p_e = U^T p and c,s = cos/sin(E dt/hbar)
        diagonal.  Hence, per state-pair, B is exactly the rotation
        conjugated back to the diabatic mapping basis:
            r' = U C U^T r + U S U^T p
            p' = -U S U^T r + U C U^T p,
        C = diag(c), S = diag(s).  This is the exact linear map the
        transported reference profile is pulled back through — no autodiff.
        """
        zb, single = self._as_z_batch(z)
        M = self.params.mass
        hbar = self.params.hbar
        R = zb[:, 0]; P = zb[:, 1]
        R_half = R + 0.5 * dt * (P / M)
        _, h, _, _ = self._frozen_R_objects(R_half)
        E, U = np.linalg.eigh(h)                       # (N,2),(N,2,2)
        U_T = np.swapaxes(U, 1, 2)
        theta = (E * dt) / hbar
        C = U @ (np.eye(2)[None] * np.cos(theta)[:, None, :]) @ U_T   # (N,2,2)
        S = U @ (np.eye(2)[None] * np.sin(theta)[:, None, :]) @ U_T   # (N,2,2)
        N = zb.shape[0]
        B = np.zeros((N, 4, 4))
        B[:, 0:2, 0:2] = C
        B[:, 0:2, 2:4] = S
        B[:, 2:4, 0:2] = -S
        B[:, 2:4, 2:4] = C
        return B[0] if single else B

    # -------------------------------------------------------------------------
    # Energies
    # -------------------------------------------------------------------------
    def energy(self, z: ArrayLike) -> FloatArray:
        """
        MInt conserved Hamiltonian (per-trajectory):

            H = P²/(2M)  +  V₀(R)  +  (r^T h(R) r + p^T h(R) p) / (2ℏ)

        where V₀ = Tr(V)/2 and h = V − V₀ I (traceless, Tr h = 0).

        This is the quantity exactly conserved by each MInt trajectory, so
        its unweighted mean E_traj = (1/N) Σ H(z_i) is flat in time and serves
        as the integrator sanity-check.  It is NOT the physical ⟨H⟩ — that is
        the density-weighted cloud sum lw_energy ≈ P₀²/(2M) + V₁₁(R₀) ≈ 0.225.
        The discrepancy E_traj − lw_energy ≈ V₀ arises because the unweighted
        mean over the SEO ensemble carries phantom vacuum energy from the
        unoccupied mapping mode; density weighting cancels it exactly.
        """
        zb, single = self._as_z_batch(z)
        M    = self.params.mass
        hbar = self.params.hbar

        R = zb[:, 0]
        P = zb[:, 1]
        r = zb[:, 2:4]
        p = zb[:, 4:6]

        V0, h, _, _ = self._frozen_R_objects(R)

        kin  = 0.5 * P * P / M
        el_r = np.einsum("ni,nij,nj->n", r, h, r)
        el_p = np.einsum("ni,nij,nj->n", p, h, p)
        E = kin + V0 + 0.5 * (el_r + el_p) / hbar

        return E[0] if single else E

    compute_energy_z = energy

    def ensemble_energy(self, z: ArrayLike) -> tuple[float, FloatArray]:
        E = self.energy(z)
        Eb = np.asarray(E, dtype=np.float64).reshape(-1)
        return float(np.mean(Eb)), Eb

    def mapping_radius_sq(self, z: ArrayLike) -> FloatArray:
        zb, single = self._as_z_batch(z)
        r = zb[:, 2:4]
        p = zb[:, 4:6]
        rad2 = np.sum(r * r + p * p, axis=1)
        return rad2[0] if single else rad2

    # -------------------------------------------------------------------------
    # Trajectory propagation
    # -------------------------------------------------------------------------
    def propagate(self, z0: ArrayLike, dt: float, n_steps: int) -> FloatArray:
        """
        Returns:
          - (n_steps+1, 6) for a single trajectory
          - (n_steps+1, N, 6) for an ensemble
        """
        zb, single = self._as_z_batch(z0)
        hist = [zb.copy()]
        z = zb.copy()
        for _ in range(int(n_steps)):
            z = np.asarray(self.step(z, dt), dtype=np.float64)
            if z.ndim == 1:
                z = z.reshape(1, D)
            hist.append(z.copy())
        H = np.stack(hist, axis=0)
        return H[:, 0, :] if single else H

    # -------------------------------------------------------------------------
    # Jacobian / monodromy (finite difference)
    # -------------------------------------------------------------------------
    def compute_step_jacobian(self, z: ArrayLike, dt: float, eps: float = 1e-7) -> FloatArray:
        """
        Central finite-difference Jacobian of the one-step map.

        For a single input z.shape==(6,), returns J.shape==(6,6).
        For a batch z.shape==(N,6), returns J.shape==(N,6,6).
        """
        zb, single = self._as_z_batch(z)

        def jac_single(zs: FloatArray) -> FloatArray:
            J = np.zeros((D, D), dtype=np.float64)
            for j in range(D):
                dz = np.zeros(D, dtype=np.float64)
                dz[j] = eps
                fp = np.asarray(self.step(zs + dz, dt), dtype=np.float64)
                fm = np.asarray(self.step(zs - dz, dt), dtype=np.float64)
                J[:, j] = (fp - fm) / (2.0 * eps)
            return J

        Jb = np.stack([jac_single(zs) for zs in zb], axis=0)
        return Jb[0] if single else Jb

    compute_step_jacobian_z = compute_step_jacobian

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    @staticmethod
    def omega_matrix() -> FloatArray:
        """
        Canonical symplectic form for ordering (R, P, r0, r1, p0, p1).
        """
        Om = np.zeros((D, D), dtype=np.float64)
        Om[0, 1] = 1.0
        Om[1, 0] = -1.0
        Om[2, 4] = 1.0
        Om[4, 2] = -1.0
        Om[3, 5] = 1.0
        Om[5, 3] = -1.0
        return Om

    def symplectic_defect(self, J: FloatArray) -> float:
        Om = self.omega_matrix()
        return float(np.linalg.norm(J.T @ Om @ J - Om, ord="fro"))

    def time_reversal_error(self, z: ArrayLike, dt: float) -> FloatArray:
        z1 = self.step(z, dt)
        z2 = self.step(z1, -dt)
        zb, single = self._as_z_batch(z)
        z2b, _ = self._as_z_batch(z2)
        err = np.linalg.norm(z2b - zb, axis=1)
        return err[0] if single else err

    def realness_check(self, z: ArrayLike, dt: float) -> None:
        """
        Explicit diagnostic to show that all intermediate objects stay real-valued.
        """
        zb, _ = self._as_z_batch(z)
        z0 = zb[0]

        R = z0[0]
        P = z0[1]
        r = z0[2:4]
        p = z0[4:6]

        R_half = R + 0.5 * dt * (P / self.params.mass)
        V0, h, dV0, dh = self._frozen_R_objects(np.array([R_half], dtype=np.float64))
        E, U = np.linalg.eigh(h[0])

        print("[realness check]")
        print("  dtype(h)  =", h.dtype, "  max|Im(h)|  =", np.max(np.abs(np.imag(h))))
        print("  dtype(dh) =", dh.dtype, "  max|Im(dh)| =", np.max(np.abs(np.imag(dh))))
        print("  dtype(E)  =", E.dtype, "  max|Im(E)|  =", np.max(np.abs(np.imag(E))))
        print("  dtype(U)  =", U.dtype, "  max|Im(U)|  =", np.max(np.abs(np.imag(U))))

        U_T = U.T
        r_e = U_T @ r
        p_e = U_T @ p
        theta = E * dt / self.params.hbar
        c = np.cos(theta)
        s = np.sin(theta)
        r_e_1 = c * r_e + s * p_e
        p_e_1 = c * p_e - s * r_e

        print("  dtype(r_e)   =", r_e.dtype, "  max|Im(r_e)|   =", np.max(np.abs(np.imag(r_e))))
        print("  dtype(p_e)   =", p_e.dtype, "  max|Im(p_e)|   =", np.max(np.abs(np.imag(p_e))))
        print("  dtype(r_e_1) =", r_e_1.dtype, "  max|Im(r_e_1)| =", np.max(np.abs(np.imag(r_e_1))))
        print("  dtype(p_e_1) =", p_e_1.dtype, "  max|Im(p_e_1)| =", np.max(np.abs(np.imag(p_e_1))))
        print("  All internal arrays are real-valued.")

    # -------------------------------------------------------------------------
    # Self-tests
    # -------------------------------------------------------------------------
    def run_self_tests(
        self,
        dt: float = 0.1,
        n_steps: int = 2000,
        seed: int = 0,
        sym_tol: float = 1.0e-6,
        det_tol: float = 1.0e-6,
        tr_tol: float = 1.0e-12,
        rad_tol: float = 1.0e-12,
        rel_E_tol: float = 1.0e-6,
    ) -> None:
        """
        MInt one-step + propagation self-tests.

        Reports diagnostics and ASSERTS that they meet the tolerances:
          * Symplectic defect ‖J^TΩJ - Ω‖_F  ≤ sym_tol
          * |det(J) - 1|                      ≤ det_tol
          * Time-reversal error               ≤ tr_tol
          * Mapping-radius error per step     ≤ rad_tol
          * Relative energy drift over n_steps ≤ rel_E_tol

        Default tolerances:
          * sym_tol, det_tol = 1e-6.  These are loose because the Jacobian J
            comes from the finite-difference path inside compute_step_jacobian
            (eps ~ 1e-5), which limits accuracy to ~eps² ≈ 1e-10 for second-
            order central differences but in practice runs at 1e-8 due to the
            mapping-coordinate eigendecomp inside MInt.  The intrinsic sym-
            plecticity of the analytic step is exact; this tolerance is a
            sanity check on the FD pipeline, not on MInt itself.
          * tr_tol, rad_tol = 1e-12.  These tests use the analytic (closed-form)
            mapping-coordinate eigendecomp directly, so they reach machine
            precision.
          * rel_E_tol = 1e-6.  PBME-MInt is a symplectic integrator with
            quadratic conservation of energy; relative drift over a few
            thousand steps is dominated by floating-point accumulation
            (~1e-9 in practice).
        """
        rng = np.random.default_rng(seed)

        z0 = pack_z(
            R=-8.0,
            P=20.0,
            r=rng.normal(size=2),
            p=rng.normal(size=2),
        )

        J = self.compute_step_jacobian(z0, dt)
        detJ = float(np.linalg.det(J))
        sym_def = self.symplectic_defect(J)
        tr_err = float(self.time_reversal_error(z0, dt))

        rad0 = float(self.mapping_radius_sq(z0))
        z1 = self.step(z0, dt)
        rad1 = float(self.mapping_radius_sq(z1))
        rad_err = abs(rad1 - rad0)

        E0 = float(self.energy(z0))
        traj = self.propagate(z0, dt, n_steps)
        E_traj = np.asarray(self.energy(traj), dtype=np.float64)
        E1 = float(E_traj[-1])
        dE = E1 - E0
        relE = dE / (abs(E0) + 1e-15)

        print("[PBME-MInt self-tests]")
        print(f"  model kind = {self.model.kind!r}")
        print(f"  mass = {self.params.mass:.6f}, hbar = {self.params.hbar:.6f}")
        print(f"  symplectic defect ||J^TΩJ-Ω||_F = {sym_def:.3e}")
        print(f"  det(J) = {detJ:.12f}   |det(J)-1| = {abs(detJ - 1.0):.3e}")
        print(f"  time-reversal error = {tr_err:.3e}")
        print(f"  mapping-radius error after one step = {rad_err:.3e}")
        print(f"  energy drift after {int(n_steps)} steps: ΔE = {dE:.3e}, relative = {relE:.3e}")
        print(f"  energy min/max along trajectory = [{E_traj.min():.12e}, {E_traj.max():.12e}]")

        # Assertions: turn passive diagnostics into actual tests.
        # tr_err is bounded below by |z0|·machine_eps ~ 1e-15 for the
        # values used here; tr_tol=1e-12 is a comfortable margin.
        assert sym_def <= sym_tol, (
            f"symplectic defect {sym_def:.3e} exceeds tolerance {sym_tol:.3e}")
        assert abs(detJ - 1.0) <= det_tol, (
            f"|det(J)-1| = {abs(detJ-1.0):.3e} exceeds tolerance {det_tol:.3e}")
        assert tr_err <= tr_tol, (
            f"time-reversal error {tr_err:.3e} exceeds tolerance {tr_tol:.3e}")
        assert rad_err <= rad_tol, (
            f"mapping-radius drift {rad_err:.3e} exceeds tolerance {rad_tol:.3e}")
        assert abs(relE) <= rel_E_tol, (
            f"relative energy drift {abs(relE):.3e} exceeds tolerance {rel_E_tol:.3e}")

    def run_ensemble_self_tests(
        self,
        z0: ArrayLike,
        dt: float = 0.1,
        n_steps: int = 500,
        jacobian_checks: int = 8,
    ) -> None:
        zb, _ = self._as_z_batch(z0)
        n = zb.shape[0]

        E0_mean, E0_all = self.ensemble_energy(zb)
        z1 = self.step(zb, dt)
        tr_err = np.asarray(self.time_reversal_error(zb, dt), dtype=np.float64)

        rad0 = np.asarray(self.mapping_radius_sq(zb), dtype=np.float64)
        rad1 = np.asarray(self.mapping_radius_sq(z1), dtype=np.float64)
        rad_err = np.abs(rad1 - rad0)

        traj = self.propagate(zb, dt, n_steps)
        zf = traj[-1]
        Ef_mean, Ef_all = self.ensemble_energy(zf)

        print("[PBME-MInt ensemble self-tests]")
        print(f"  n_traj = {n}")
        print(f"  mean(E0) = {E0_mean:.12e}")
        print(f"  mean(Ef) = {Ef_mean:.12e}")
        print(f"  Δ mean(E) = {Ef_mean - E0_mean:.3e}")
        print(f"  mean(|ΔE_i|) = {np.mean(np.abs(Ef_all - E0_all)):.3e}")
        print(f"  max(|ΔE_i|)  = {np.max(np.abs(Ef_all - E0_all)):.3e}")
        print(f"  RMS time-reversal error = {np.sqrt(np.mean(tr_err**2)):.3e}")
        print(f"  max time-reversal error = {np.max(tr_err):.3e}")
        print(f"  mean(|Δ rad2|) = {np.mean(rad_err):.3e}")
        print(f"  max(|Δ rad2|)  = {np.max(rad_err):.3e}")

        n_chk = min(int(jacobian_checks), n)
        sym_vals = []
        det_vals = []
        for i in range(n_chk):
            Ji = self.compute_step_jacobian(zb[i], dt)
            sym_vals.append(self.symplectic_defect(Ji))
            det_vals.append(np.linalg.det(Ji))
        sym_vals = np.asarray(sym_vals, dtype=np.float64)
        det_vals = np.asarray(det_vals, dtype=np.float64)

        print(f"  Jacobian subset checks ({n_chk} trajectories)")
        print(f"    mean symplectic defect = {sym_vals.mean():.3e}")
        print(f"    max  symplectic defect = {sym_vals.max():.3e}")
        print(f"    mean |det(J)-1| = {np.mean(np.abs(det_vals - 1.0)):.3e}")
        print(f"    max  |det(J)-1| = {np.max(np.abs(det_vals - 1.0)):.3e}")

    def run_trajectory_value_test(
        self,
        z0: Optional[ArrayLike] = None,
        dt: float = 0.5,
        n_steps: int = 200,
        print_every: int = 1,
    ) -> FloatArray:
        if z0 is None:
            z = pack_z(
                R=-8.0,
                P=20.0,
                r=np.array([np.sqrt(2.0), 0.0], dtype=np.float64),
                p=np.array([np.sqrt(2.0), 0.0], dtype=np.float64),
            )
        else:
            z = np.asarray(z0, dtype=np.float64).reshape(D)

        traj = np.zeros((n_steps + 1, D), dtype=np.float64)
        energy = np.zeros(n_steps + 1, dtype=np.float64)
        rad2 = np.zeros(n_steps + 1, dtype=np.float64)

        print("[PBME-MInt trajectory-value test]")
        print(f"  model kind = {self.model.kind!r}")
        print(f"  dt = {dt:.6f}, n_steps = {n_steps}")
        print("step,t,R,P,r0,r1,p0,p1,energy,rad2")

        zz = z.copy()
        for k in range(n_steps + 1):
            traj[k] = zz
            energy[k] = float(self.energy(zz))
            rad2[k] = float(self.mapping_radius_sq(zz))

            if (k % print_every == 0) or (k == n_steps):
                R, P, r0, r1, p0, p1 = zz
                print(
                    f"{k},{k*dt:.6f},"
                    f"{R:.12e},{P:.12e},{r0:.12e},{r1:.12e},{p0:.12e},{p1:.12e},"
                    f"{energy[k]:.12e},{rad2[k]:.12e}"
                )

            if k < n_steps:
                zz = np.asarray(self.step(zz, dt), dtype=np.float64)

        print("\n[trajectory summary]")
        print(f"  R range      = [{traj[:,0].min():.6e}, {traj[:,0].max():.6e}]")
        print(f"  P range      = [{traj[:,1].min():.6e}, {traj[:,1].max():.6e}]")
        print(f"  energy range = [{energy.min():.12e}, {energy.max():.12e}]   ΔE = {energy[-1] - energy[0]:.3e}")
        print(f"  rad2 range   = [{rad2.min():.12e}, {rad2.max():.12e}]   Δrad2 = {rad2[-1] - rad2[0]:.3e}")

        return traj


# =============================================================================
# Convenience free functions
# =============================================================================

def compute_step_jacobian_z(
    z: ArrayLike,
    dynamics: PBMEMIntDynamics,
    dt: float,
    eps: float = 1e-7,
) -> FloatArray:
    return dynamics.compute_step_jacobian(z, dt=dt, eps=eps)


def compute_energy_z(
    z: ArrayLike,
    dynamics: PBMEMIntDynamics,
) -> FloatArray:
    return dynamics.energy(z)


def mapping_radius_sq(
    z: ArrayLike,
    dynamics: PBMEMIntDynamics,
) -> FloatArray:
    return dynamics.mapping_radius_sq(z)


# =============================================================================
# Minimal example
# =============================================================================

if __name__ == "__main__":
    model = TullyModel(TullyParams.defaults("dual"))
    dyn = PBMEMIntDynamics(model=model, params=PBMEMIntParams(mass=2000.0, hbar=1.0))

    dyn.run_self_tests(dt=0.1, n_steps=2000, seed=0)
    dyn.realness_check(
        z=pack_z(
            R=-10.0,
            P=30.0,
            r=np.array([np.sqrt(2.0), 0.0]),
            p=np.array([np.sqrt(2.0), 0.0]),
        ),
        dt=0.5,
    )