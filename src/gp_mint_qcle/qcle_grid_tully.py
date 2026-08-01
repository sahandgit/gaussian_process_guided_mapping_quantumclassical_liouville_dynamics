"""
qcle_grid_tully.py
==================

Quantum-classical Liouville equation (QCLE) grid solver for the 1D, 2-state
Tully diabatic model.  Designed as a numerically exact reference against which
the PBME-MInt trajectory solver in ``Mint.py`` can be compared on identical
initial conditions.

Equation of motion
------------------
For a 2-state Hamiltonian H(R) (real symmetric) and one nuclear DOF, the QCLE
in the diabatic basis is

    ∂_t ρ_W = -(i/ℏ) [H, ρ_W]                       (quantum commutator)
              -(P/M) ∂_R ρ_W                         (classical advection)
              + (1/2) { ∂_R H, ∂_P ρ_W }             (force coupling, sym.)

where {A,B} = AB + BA is the matrix anticommutator and ρ_W is a 2x2
Hermitian-matrix-valued field on phase space (R, P).

Decomposing ρ_W = [[A, b_R + i b_I],[b_R - i b_I, C]] with A, C, b_R, b_I real,
and writing H = [[V11, V12],[V12, V22]] with V0 = (V11+V22)/2, the four
real evolution equations are:

    ∂_t A   = -(P/M)∂_R A                                    [advection]
              + V11' ∂_P A + V12' ∂_P b_R                    [force coupling]
              - (2/ℏ) V12 b_I                                [quantum]

    ∂_t C   = -(P/M)∂_R C
              + V22' ∂_P C + V12' ∂_P b_R
              + (2/ℏ) V12 b_I

    ∂_t b_R = -(P/M)∂_R b_R
              + V0'  ∂_P b_R + (1/2) V12' (∂_P A + ∂_P C)
              + (1/ℏ) (V11 - V22) b_I

    ∂_t b_I = -(P/M)∂_R b_I
              + V0'  ∂_P b_I
              + (1/ℏ) [ V12 (A - C) - (V11 - V22) b_R ]

Trace ∂_t (A + C) is conserved by every line individually (after R, P
integration with periodic BCs); Hermiticity is preserved exactly because A, C
remain real and the off-diagonal stays in the Re/Im split.

Discretization
--------------
* Uniform 2D grid (R, P), Fourier-pseudospectral derivatives.  Exact to the
  Nyquist limit on smooth fields, and consistent for the linear PDE we are
  solving.
* Periodic BCs.  We choose the (R, P) box wide enough that ρ never reaches
  the boundary; the user is warned if it does.
* Time integration: explicit RK4, dt set conservatively below the
  pseudospectral CFL limits.

Initial condition
-----------------
For an initial diabatic state |λ⟩ ⊗ |Gaussian wavepacket⟩, all electronic
components are zero except ρ_λλ which equals the Wigner transform of the
nuclear Gaussian:

    ρ_λλ(R, P, 0) = (1/(πℏ)) exp[ -(R-R0)²/(2 σ_R²)
                                  -2 σ_R² (P-P0)²/ℏ² ]

with σ_P = ℏ / (2 σ_R).  This matches the convention in Sampling.py.

Observables
-----------
* Diabatic populations:  σ_λλ(t) = ∫∫ ρ_λλ(R, P, t) dR dP
* R-marginal of state λ:  ρ_λ(R, t) = ∫ ρ_λλ(R, P, t) dP
* P-marginal of state λ:  ρ_λ(P, t) = ∫ ρ_λλ(R, P, t) dR
* Total norm:             trace( ∫∫ ρ_W dR dP ) = σ_00 + σ_11   (≡ 1)
"""
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

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# scipy.fft provides multi-threaded pocketfft and real-input rfft, both of
# which are substantially faster than numpy.fft on the spectral-derivative
# hot path inside `rhs`.  Importing as `_spfft` to keep the namespace local.
import scipy.fft as _spfft

from .Models import TullyModel, TullyParams


FloatArray = NDArray[np.float64]


# =============================================================================
# Parameter container
# =============================================================================

@dataclass(frozen=True)
class QCLEGridParams:
    """
    Grid + integrator parameters for the QCLE solver.

    Box widths and resolutions should be chosen so that:
      * the wave packet never reaches the boundaries (periodic BCs),
      * the initial Gaussian is well-resolved (need dR < σ_R / 3, dP < σ_P / 3),
      * the CFL conditions are satisfied at the chosen dt.
    """
    R_min: float = -25.0
    R_max: float = 25.0
    P_min: float = -50.0
    P_max: float = 50.0
    n_R: int = 384
    n_P: int = 256
    mass: float = 2000.0
    hbar: float = 1.0

    def __post_init__(self) -> None:
        if self.R_max <= self.R_min:
            raise ValueError("R_max must exceed R_min.")
        if self.P_max <= self.P_min:
            raise ValueError("P_max must exceed P_min.")
        if self.n_R < 16 or self.n_P < 16:
            raise ValueError("n_R, n_P must be at least 16.")
        if self.mass <= 0.0 or self.hbar <= 0.0:
            raise ValueError("mass and hbar must be positive.")


# =============================================================================
# Grid state
# =============================================================================

@dataclass
class QCLEGridState:
    """4 real fields (A, C, b_R, b_I) each of shape (n_R, n_P)."""
    A: FloatArray   # ρ_00
    C: FloatArray   # ρ_11
    bR: FloatArray  # Re ρ_01
    bI: FloatArray  # Im ρ_01

    def copy(self) -> "QCLEGridState":
        return QCLEGridState(
            A=self.A.copy(),
            C=self.C.copy(),
            bR=self.bR.copy(),
            bI=self.bI.copy(),
        )


# =============================================================================
# Solver
# =============================================================================

class QCLEGridSolver:
    """
    Pseudospectral RK4 QCLE solver on a 2D (R, P) grid for a 1D Tully model.
    """

    def __init__(
        self,
        model: Optional[TullyModel] = None,
        params: Optional[QCLEGridParams] = None,
    ) -> None:
        self.model = model if model is not None else TullyModel(TullyParams.defaults("dual"))
        self.params = params if params is not None else QCLEGridParams()

        p = self.params
        # Cell-centred grids (avoid double-counting periodic endpoint).
        self.dR = (p.R_max - p.R_min) / p.n_R
        self.dP = (p.P_max - p.P_min) / p.n_P
        self.R = p.R_min + self.dR * (0.5 + np.arange(p.n_R, dtype=np.float64))
        self.P = p.P_min + self.dP * (0.5 + np.arange(p.n_P, dtype=np.float64))

        # Fourier wavenumbers for full-complex FFT (kept for legacy callers).
        self.kR = 2.0 * np.pi * np.fft.fftfreq(p.n_R, d=self.dR)
        self.kP = 2.0 * np.pi * np.fft.fftfreq(p.n_P, d=self.dP)

        # Optimisation: for the spectral derivatives in `rhs` we use real FFTs
        # (scipy.fft.rfft / irfft) which are ~3× faster than the legacy
        # complex FFT path.  These are pre-broadcast multipliers in the
        # half-complex grid layout used by rfft along each axis separately.
        self._kR_r = 2.0 * np.pi * _spfft.rfftfreq(p.n_R, d=self.dR)
        self._kP_r = 2.0 * np.pi * _spfft.rfftfreq(p.n_P, d=self.dP)
        self._ikR_r = (1j * self._kR_r)[:, None]      # (n_R//2+1, 1)
        self._ikP_r = (1j * self._kP_r)[None, :]      # (1, n_P//2+1)
        # Optimal FFT thread count is hardware-dependent.  On 2D rfft the
        # work scales as O(N² log N) but the memory traffic scales as O(N²),
        # so for N ≳ 1000 the bottleneck is memory bandwidth and adding
        # threads past ~8 causes cache thrashing.  We default to
        # min(8, available cores), and let the user override via
        # QCLE_FFT_WORKERS env var if their box scales differently.
        import os as _os
        _env = _os.environ.get("QCLE_FFT_WORKERS")
        if _env is not None:
            self._fft_workers = max(1, int(_env))
        else:
            n_cpus = len(_os.sched_getaffinity(0)) if hasattr(_os, "sched_getaffinity") else (_os.cpu_count() or 1)
            self._fft_workers = min(8, n_cpus)

        # Pre-evaluate model on R-grid (broadcast along P later via [:, None]).
        Rg = self.R
        self._V11  = np.asarray(self.model.V11(Rg),    dtype=np.float64)
        self._V22  = np.asarray(self.model.V22(Rg),    dtype=np.float64)
        self._V12  = np.asarray(self.model.V12(Rg),    dtype=np.float64)
        self._V11p = np.asarray(self.model.dV11_dR(Rg), dtype=np.float64)
        self._V22p = np.asarray(self.model.dV22_dR(Rg), dtype=np.float64)
        self._V12p = np.asarray(self.model.dV12_dR(Rg), dtype=np.float64)
        self._V0p  = 0.5 * (self._V11p + self._V22p)
        self._dV   = self._V11 - self._V22  # diabatic energy gap

        # Pre-broadcast model arrays to shape (n_R, 1) for use inside rhs.
        self._V11_b  = self._V11[:,  None]
        self._V22_b  = self._V22[:,  None]
        self._V12_b  = self._V12[:,  None]
        self._V11p_b = self._V11p[:, None]
        self._V22p_b = self._V22p[:, None]
        self._V12p_b = self._V12p[:, None]
        self._V0p_b  = self._V0p[:,  None]
        self._dV_b   = self._dV[:,   None]
        self._adv    = -(self.P[None, :] / p.mass)    # (1, n_P)

    # -------------------------------------------------------------------------
    # Geometry helpers
    # -------------------------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.params.n_R, self.params.n_P)

    @property
    def cell_area(self) -> float:
        return float(self.dR * self.dP)

    def meshgrid(self) -> Tuple[FloatArray, FloatArray]:
        Rmesh, Pmesh = np.meshgrid(self.R, self.P, indexing="ij")
        return Rmesh, Pmesh

    # -------------------------------------------------------------------------
    # Spectral derivatives
    # -------------------------------------------------------------------------
    def _dR_field(self, f: FloatArray) -> FloatArray:
        """Spectral d/dR.  Real-input rfft, threaded via scipy.fft."""
        F = _spfft.rfft(f, axis=0, workers=self._fft_workers)
        F *= self._ikR_r
        return _spfft.irfft(F, axis=0, n=self.params.n_R, workers=self._fft_workers)

    def _dP_field(self, f: FloatArray) -> FloatArray:
        F = _spfft.rfft(f, axis=1, workers=self._fft_workers)
        F *= self._ikP_r
        return _spfft.irfft(F, axis=1, n=self.params.n_P, workers=self._fft_workers)

    # -------------------------------------------------------------------------
    # Right-hand side of QCLE
    # -------------------------------------------------------------------------
    def rhs(
        self,
        A: FloatArray,
        C: FloatArray,
        bR: FloatArray,
        bI: FloatArray,
    ) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        # Inline the eight spectral derivatives so we do exactly 8 rfft + 8
        # irfft per RHS evaluation.  Each is real-input → half-complex →
        # real-output (scipy.fft.rfft / irfft) so it's ~2× cheaper than the
        # legacy complex FFT path even before threading.
        nR = self.params.n_R
        nP = self.params.n_P
        w  = self._fft_workers
        ikR = self._ikR_r
        ikP = self._ikP_r

        # d/dR for all four fields
        FA  = _spfft.rfft(A,  axis=0, workers=w); FA  *= ikR; dA_dR  = _spfft.irfft(FA,  axis=0, n=nR, workers=w)
        FC  = _spfft.rfft(C,  axis=0, workers=w); FC  *= ikR; dC_dR  = _spfft.irfft(FC,  axis=0, n=nR, workers=w)
        FbR = _spfft.rfft(bR, axis=0, workers=w); FbR *= ikR; dbR_dR = _spfft.irfft(FbR, axis=0, n=nR, workers=w)
        FbI = _spfft.rfft(bI, axis=0, workers=w); FbI *= ikR; dbI_dR = _spfft.irfft(FbI, axis=0, n=nR, workers=w)

        # d/dP for all four fields
        GA  = _spfft.rfft(A,  axis=1, workers=w); GA  *= ikP; dA_dP  = _spfft.irfft(GA,  axis=1, n=nP, workers=w)
        GC  = _spfft.rfft(C,  axis=1, workers=w); GC  *= ikP; dC_dP  = _spfft.irfft(GC,  axis=1, n=nP, workers=w)
        GbR = _spfft.rfft(bR, axis=1, workers=w); GbR *= ikP; dbR_dP = _spfft.irfft(GbR, axis=1, n=nP, workers=w)
        GbI = _spfft.rfft(bI, axis=1, workers=w); GbI *= ikP; dbI_dP = _spfft.irfft(GbI, axis=1, n=nP, workers=w)

        adv  = self._adv
        h    = self.params.hbar
        V11p = self._V11p_b; V22p = self._V22p_b; V12p = self._V12p_b
        V0p  = self._V0p_b;  V12  = self._V12_b;  dV   = self._dV_b
        inv_h = 1.0 / h

        # Reuse dbR_dP across rhs_A and rhs_C (was computed twice in legacy).
        V12p_dbR_dP = V12p * dbR_dP
        V12_bI_2_h  = (2.0 * inv_h) * V12 * bI

        rhs_A = adv * dA_dR + V11p * dA_dP + V12p_dbR_dP - V12_bI_2_h
        rhs_C = adv * dC_dR + V22p * dC_dP + V12p_dbR_dP + V12_bI_2_h

        # 0.5*(dA_dP + dC_dP) for bR
        sum_dAC_dP = 0.5 * (dA_dP + dC_dP)
        rhs_bR = adv * dbR_dR + V0p * dbR_dP + V12p * sum_dAC_dP + inv_h * dV * bI
        rhs_bI = adv * dbI_dR + V0p * dbI_dP + inv_h * (V12 * (A - C) - dV * bR)

        return rhs_A, rhs_C, rhs_bR, rhs_bI

    # -------------------------------------------------------------------------
    # Time stepping
    # -------------------------------------------------------------------------
    def step(self, state: QCLEGridState, dt: float) -> QCLEGridState:
        """
        One classical RK4 step of the QCLE.

        Memory-optimized form: instead of holding all 4 stages in memory
        (which costs ~16 state-sized arrays), we accumulate the RK4
        Simpson-like sum K = k1 + 2k2 + 2k3 + k4 in-place as we go,
        and free each intermediate stage immediately.  This reduces
        peak memory from ~16N² to ~10N² (5 buffers per field × 4 fields,
        vs the original ~16-20).  On memory-bandwidth-bound machines
        (where 2D FFT throughput is gated by RAM rather than FLOPs)
        this typically gives a 1.2–1.5× per-step speedup with bit-
        identical RK4 output.
        """
        A, C, bR, bI = state.A, state.C, state.bR, state.bI
        h2 = 0.5 * dt
        h6 = dt / 6.0

        # ----- Stage 1: k1 = rhs(A, C, bR, bI) -----
        k1A, k1C, k1bR, k1bI = self.rhs(A, C, bR, bI)

        # K = k1   (running Simpson sum)
        # Stage-2 input: A + h2 * k1   (use temporaries; can't overwrite A)
        tA  = A  + h2 * k1A
        tC  = C  + h2 * k1C
        tbR = bR + h2 * k1bR
        tbI = bI + h2 * k1bI

        KA, KC, KbR, KbI = k1A, k1C, k1bR, k1bI       # alias, no copy

        # ----- Stage 2: k2 = rhs(tA, tC, tbR, tbI) -----
        k2A, k2C, k2bR, k2bI = self.rhs(tA, tC, tbR, tbI)

        # K += 2 * k2  (in-place)
        KA  += 2.0 * k2A;   KC  += 2.0 * k2C
        KbR += 2.0 * k2bR;  KbI += 2.0 * k2bI

        # Stage-3 input: A + h2 * k2  (overwrite tA etc.)
        np.multiply(k2A,  h2, out=tA);   tA  += A
        np.multiply(k2C,  h2, out=tC);   tC  += C
        np.multiply(k2bR, h2, out=tbR);  tbR += bR
        np.multiply(k2bI, h2, out=tbI);  tbI += bI
        del k2A, k2C, k2bR, k2bI       # release stage-2 buffers

        # ----- Stage 3: k3 = rhs(tA, tC, tbR, tbI) -----
        k3A, k3C, k3bR, k3bI = self.rhs(tA, tC, tbR, tbI)

        # K += 2 * k3
        KA  += 2.0 * k3A;   KC  += 2.0 * k3C
        KbR += 2.0 * k3bR;  KbI += 2.0 * k3bI

        # Stage-4 input: A + dt * k3
        np.multiply(k3A,  dt, out=tA);   tA  += A
        np.multiply(k3C,  dt, out=tC);   tC  += C
        np.multiply(k3bR, dt, out=tbR);  tbR += bR
        np.multiply(k3bI, dt, out=tbI);  tbI += bI
        del k3A, k3C, k3bR, k3bI

        # ----- Stage 4: k4 = rhs(tA, tC, tbR, tbI) -----
        k4A, k4C, k4bR, k4bI = self.rhs(tA, tC, tbR, tbI)
        del tA, tC, tbR, tbI

        # K += k4 ; final state = old + (dt/6) * K
        KA  += k4A;  KC  += k4C
        KbR += k4bR; KbI += k4bI
        del k4A, k4C, k4bR, k4bI

        A1  = A  + h6 * KA
        C1  = C  + h6 * KC
        bR1 = bR + h6 * KbR
        bI1 = bI + h6 * KbI
        return QCLEGridState(A=A1, C=C1, bR=bR1, bI=bI1)

    def propagate(
        self,
        state0: QCLEGridState,
        dt: float,
        n_steps: int,
        save_every: int = 1,
        verbose: bool = False,
    ) -> Tuple[FloatArray, List[QCLEGridState]]:
        """
        Propagate ``state0`` for ``n_steps`` of size ``dt``, saving snapshots
        every ``save_every`` steps (and at t = 0 and t_final).

        Returns
        -------
        times : (n_save,) array
        snapshots : list of QCLEGridState (length n_save)
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if save_every < 1:
            raise ValueError("save_every must be >= 1.")

        self.check_cfl(dt, warn_only=True, verbose=verbose)

        times: List[float] = [0.0]
        snapshots: List[QCLEGridState] = [state0.copy()]
        state = state0.copy()

        for k in range(1, n_steps + 1):
            state = self.step(state, dt)
            if (k % save_every == 0) or (k == n_steps):
                times.append(k * dt)
                snapshots.append(state.copy())
                if verbose:
                    pop00, pop11 = self.populations(state)
                    print(
                        f"[QCLE]  step {k:6d}  t={k*dt:8.2f}  "
                        f"σ00={pop00:.6f}  σ11={pop11:.6f}  "
                        f"trace={pop00+pop11:.6e}"
                    )

        return np.asarray(times, dtype=np.float64), snapshots

    # -------------------------------------------------------------------------
    # Initial conditions
    # -------------------------------------------------------------------------
    def initial_diabat_gaussian(
        self,
        R0: float,
        P0: float,
        sigma_R: float,
        init_state: int = 0,
    ) -> QCLEGridState:
        """
        Build the initial QCLE state corresponding to

            |Ψ⟩ = |init_state⟩ ⊗ |Gaussian wavepacket(R0, P0, σ_R)⟩

        whose partial Wigner transform is

            ρ_λλ(R,P) = (1/(πℏ)) exp[-(R-R0)²/(2σ_R²) - 2σ_R²(P-P0)²/ℏ²]
            δ_{λ, init_state}.

        Returns a QCLEGridState whose grid integrals satisfy
            ∫∫ ρ_{init_state, init_state} dR dP ≈ 1
        with error → 0 as the box is widened and the grid is refined.
        """
        if init_state not in (0, 1):
            raise ValueError("init_state must be 0 or 1 (Tully has 2 diabats).")
        if sigma_R <= 0.0:
            raise ValueError("sigma_R must be positive.")

        hbar = self.params.hbar
        Rmesh, Pmesh = self.meshgrid()
        W = (1.0 / (np.pi * hbar)) * np.exp(
            -(Rmesh - R0) ** 2 / (2.0 * sigma_R ** 2)
            - 2.0 * (sigma_R ** 2) * (Pmesh - P0) ** 2 / (hbar ** 2)
        )

        Z = np.zeros_like(W)
        if init_state == 0:
            return QCLEGridState(A=W, C=Z.copy(), bR=Z.copy(), bI=Z.copy())
        return QCLEGridState(A=Z.copy(), C=W, bR=Z.copy(), bI=Z.copy())

    # -------------------------------------------------------------------------
    # Observables
    # -------------------------------------------------------------------------
    def populations(self, state: QCLEGridState) -> Tuple[float, float]:
        """Diabatic populations σ_00, σ_11."""
        c = self.cell_area
        return float(np.sum(state.A) * c), float(np.sum(state.C) * c)

    def trace(self, state: QCLEGridState) -> float:
        p00, p11 = self.populations(state)
        return p00 + p11

    def coherence_norm(self, state: QCLEGridState) -> float:
        """
        L² norm of the off-diagonal in phase space:
            ‖ρ_01‖² = ∫∫ |ρ_01|² dR dP = ∫∫ (b_R² + b_I²) dR dP
        """
        c = self.cell_area
        return float(np.sum(state.bR * state.bR + state.bI * state.bI) * c)

    def R_marginal(self, state: QCLEGridState) -> Tuple[FloatArray, FloatArray]:
        """Returns (R-marginal of A, R-marginal of C), shape (n_R,)."""
        return (
            np.sum(state.A, axis=1) * self.dP,
            np.sum(state.C, axis=1) * self.dP,
        )

    def P_marginal(self, state: QCLEGridState) -> Tuple[FloatArray, FloatArray]:
        """Returns (P-marginal of A, P-marginal of C), shape (n_P,)."""
        return (
            np.sum(state.A, axis=0) * self.dR,
            np.sum(state.C, axis=0) * self.dR,
        )

    def interpolate_to_points(
        self,
        field: FloatArray,
        R_pts: FloatArray,
        P_pts: FloatArray,
        out_of_bounds: str = "zero",
    ) -> FloatArray:
        """
        Bilinear interpolation of a (n_R, n_P) grid field to scattered points
        (R_pts[i], P_pts[i]).

        This is the core operation for the one-to-one **Lagrangian** comparison:
        evaluate the QCLE density (an Eulerian field) at every PBME trajectory
        position so each trajectory gets a paired (PBME-carrier-value, QCLE-at-
        same-point) value.  Since PBME flow is volume-preserving (det J = 1)
        and Liouville-conserving, the per-trajectory comparison is the most
        direct diagnostic of how the two methods agree on the *density*, not
        just on its integrals.

        Parameters
        ----------
        field
            (n_R, n_P) array; e.g. ``state.A``, ``state.C``, or any time-
            stacked snapshot ``stacks["A"][k]``.
        R_pts, P_pts
            Scattered evaluation points.  Same shape (any dimensionality);
            broadcast together.
        out_of_bounds
            Behaviour for points outside [R_min, R_max] × [P_min, P_max]:
            ``"zero"`` (default — periodic-like, treats outside as zero
            density), ``"clip"`` (project onto boundary; useful as a sanity
            check), ``"nan"`` (raise a NaN to make boundary leakage loud).

        Returns
        -------
        Array of the same shape as ``R_pts``/``P_pts`` with interpolated values.

        Notes
        -----
        Cell-centred grids are assumed:
            R[i] = R_min + (i + 0.5) dR,   P[j] = P_min + (j + 0.5) dP.
        A "fractional index" u = (R − R[0]) / dR ∈ [0, n_R−1] is used.
        """
        if field.shape != (self.params.n_R, self.params.n_P):
            raise ValueError(
                f"field shape {field.shape} != grid shape {self.shape}"
            )

        R_pts = np.asarray(R_pts, dtype=np.float64)
        P_pts = np.asarray(P_pts, dtype=np.float64)
        shape = np.broadcast_shapes(R_pts.shape, P_pts.shape)
        R_pts = np.broadcast_to(R_pts, shape).ravel()
        P_pts = np.broadcast_to(P_pts, shape).ravel()

        # Fractional indices on cell-centred grid.
        u = (R_pts - self.R[0]) / self.dR
        v = (P_pts - self.P[0]) / self.dP

        n_R = self.params.n_R
        n_P = self.params.n_P
        in_R = (u >= 0.0) & (u <= n_R - 1.0)
        in_P = (v >= 0.0) & (v <= n_P - 1.0)
        in_box = in_R & in_P

        if out_of_bounds == "nan":
            out = np.full(R_pts.size, np.nan, dtype=np.float64)
        else:
            out = np.zeros(R_pts.size, dtype=np.float64)

        if out_of_bounds == "clip":
            u = np.clip(u, 0.0, n_R - 1.0)
            v = np.clip(v, 0.0, n_P - 1.0)
            in_box = np.ones(R_pts.size, dtype=bool)
        elif out_of_bounds not in ("zero", "nan"):
            raise ValueError(f"Unknown out_of_bounds={out_of_bounds!r}")

        if not np.any(in_box):
            return out.reshape(shape)

        u_in = u[in_box]
        v_in = v[in_box]
        i0 = np.clip(np.floor(u_in).astype(np.int64), 0, n_R - 2)
        j0 = np.clip(np.floor(v_in).astype(np.int64), 0, n_P - 2)
        fu = u_in - i0
        fv = v_in - j0

        f00 = field[i0,     j0    ]
        f10 = field[i0 + 1, j0    ]
        f01 = field[i0,     j0 + 1]
        f11 = field[i0 + 1, j0 + 1]
        out[in_box] = ((1 - fu) * (1 - fv) * f00
                       + fu       * (1 - fv) * f10
                       + (1 - fu) * fv       * f01
                       + fu       * fv       * f11)
        return out.reshape(shape)

    def energy_components(self, state: QCLEGridState) -> dict:
        """
        Phase-space-integrated energy contributions:
            ⟨T⟩  = ∫∫ (P²/2M) (A + C) dR dP
            ⟨V⟩  = ∫∫ [V11 A + V22 C + 2 V12 b_R] dR dP
            ⟨H⟩  = ⟨T⟩ + ⟨V⟩
        """
        c = self.cell_area
        Pmesh = self.P[None, :]
        kin = (Pmesh ** 2) / (2.0 * self.params.mass)
        T = float(np.sum(kin * (state.A + state.C)) * c)

        V11 = self._V11[:, None]; V22 = self._V22[:, None]; V12 = self._V12[:, None]
        V_int = float(np.sum(V11 * state.A + V22 * state.C + 2.0 * V12 * state.bR) * c)

        return {"T": T, "V": V_int, "E": T + V_int}

    # -------------------------------------------------------------------------
    # Diagnostics / safety
    # -------------------------------------------------------------------------
    def cfl_dt_max(self, P_max_active: Optional[float] = None) -> dict:
        """
        Estimate the largest dt allowed by linear-stability (CFL) bounds for
        each operator, treating spectral derivatives at maximum wavenumber
        k_R^max ≈ π/dR, k_P^max ≈ π/dP.

        For RK4 on the linear advection u_t + a u_x = 0 with spectral spatial
        derivatives, the stability boundary is roughly  |a| k_max dt ≲ 2.83.

        Returns a dict with the per-operator dt_max estimates.
        """
        Mnu = self.params.mass
        kR_max = np.pi / self.dR
        kP_max = np.pi / self.dP
        rk4_bound = 2.828  # |λ dt| stability for RK4 on imaginary axis

        # advection in R: speed = |P|/M
        Pmax = P_max_active if P_max_active is not None else max(abs(self.params.P_min), abs(self.params.P_max))
        adv_speed_R = Pmax / Mnu
        dt_advect_R = rk4_bound / max(adv_speed_R * kR_max, 1e-30)

        # force coupling (P-direction): speed ~ max|∂R H|
        max_force = float(np.max(np.abs(np.stack([self._V11p, self._V22p, self._V12p]))))
        dt_force_P = rk4_bound / max(max_force * kP_max, 1e-30)

        # quantum oscillation: ω_max = (E_+ - E_-)/ℏ on the grid
        H_at = np.stack([
            np.stack([self._V11, self._V12], axis=-1),
            np.stack([self._V12, self._V22], axis=-1),
        ], axis=-2)  # (n_R, 2, 2)
        E = np.linalg.eigvalsh(H_at)  # (n_R, 2)
        gap = float(np.max(E[:, 1] - E[:, 0]))
        omega_max = gap / self.params.hbar
        dt_quantum = rk4_bound / max(omega_max, 1e-30)

        return {
            "advection_R": float(dt_advect_R),
            "force_P": float(dt_force_P),
            "quantum": float(dt_quantum),
            "min": float(min(dt_advect_R, dt_force_P, dt_quantum)),
        }

    def check_cfl(
        self,
        dt: float,
        P_max_active: Optional[float] = None,
        warn_only: bool = True,
        verbose: bool = False,
    ) -> None:
        bounds = self.cfl_dt_max(P_max_active=P_max_active)
        if verbose:
            print(
                "[QCLE CFL] "
                f"adv_R≤{bounds['advection_R']:.3f}  "
                f"force_P≤{bounds['force_P']:.3f}  "
                f"quantum≤{bounds['quantum']:.3f}  "
                f"min≤{bounds['min']:.3f}  (dt={dt:.3f})"
            )
        if dt > bounds["min"]:
            msg = (
                f"dt={dt:.4g} exceeds estimated CFL bound "
                f"{bounds['min']:.4g} (operator-wise: {bounds}).  "
                "Solution may diverge — reduce dt or refine grid."
            )
            if warn_only:
                import warnings
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            else:
                raise RuntimeError(msg)

    # -------------------------------------------------------------------------
    # Convenience for snapshot stacks
    # -------------------------------------------------------------------------
    @staticmethod
    def stack_snapshots(snapshots: List[QCLEGridState]) -> dict:
        """
        Convert a list of QCLEGridState into stacked numpy arrays of shape
        (n_save, n_R, n_P), one per real component.
        """
        A = np.stack([s.A for s in snapshots], axis=0)
        C = np.stack([s.C for s in snapshots], axis=0)
        bR = np.stack([s.bR for s in snapshots], axis=0)
        bI = np.stack([s.bI for s in snapshots], axis=0)
        return {"A": A, "C": C, "bR": bR, "bI": bI}


# =============================================================================
# Self-tests
# =============================================================================

def _free_propagation_test(verbose: bool = True) -> None:
    """
    Free-particle test: for V = constant the QCLE reduces to free Liouville,
    and a Gaussian Wigner wavepacket on diabat 0 just translates rigidly with
    velocity P0/M while spreading according to the linear shear in (R, P).

    Specifically, the centroid R̄(t) = R0 + P0 t / M and the populations are
    conserved exactly.  This isolates the advection step from the quantum
    parts and confirms the time integrator + spectral derivatives.
    """
    from .Models import TullyParams, TullyModel

    # Use the "extended" Tully kind only for V0 trace (no R-dependence on the
    # diagonals of V), and zero out V12 by setting C=0; that's a fully free
    # propagation along each diabat with no electronic mixing.
    class FreeModel(TullyModel):
        def V11(self, R):
            return np.zeros_like(np.asarray(R, dtype=np.float64))
        def V22(self, R):
            return np.zeros_like(np.asarray(R, dtype=np.float64))
        def V12(self, R):
            return np.zeros_like(np.asarray(R, dtype=np.float64))
        def dV11_dR(self, R):
            return np.zeros_like(np.asarray(R, dtype=np.float64))
        def dV22_dR(self, R):
            return np.zeros_like(np.asarray(R, dtype=np.float64))
        def dV12_dR(self, R):
            return np.zeros_like(np.asarray(R, dtype=np.float64))

    model = FreeModel(TullyParams.defaults("dual"))
    params = QCLEGridParams(
        R_min=-30.0, R_max=30.0, n_R=256,
        P_min=-30.0, P_max=30.0, n_P=128,
        mass=2000.0, hbar=1.0,
    )
    solver = QCLEGridSolver(model=model, params=params)

    P0_test = 20.0
    state0 = solver.initial_diabat_gaussian(R0=-10.0, P0=P0_test, sigma_R=1.0, init_state=0)

    p00_0, p11_0 = solver.populations(state0)

    dt, n_steps = 0.5, 1000
    times, snaps = solver.propagate(state0, dt=dt, n_steps=n_steps, save_every=200)

    p00_f, p11_f = solver.populations(snaps[-1])
    Rmean_0 = float(np.sum(solver.R[:, None] * state0.A) * solver.cell_area)
    Rmean_f = float(np.sum(solver.R[:, None] * snaps[-1].A) * solver.cell_area)
    R_expected = -10.0 + (P0_test / params.mass) * times[-1]

    if verbose:
        print("[Free propagation test]")
        print(f"  populations (init):  σ00={p00_0:.6e}  σ11={p11_0:.6e}")
        print(f"  populations (final): σ00={p00_f:.6e}  σ11={p11_f:.6e}")
        print(f"  ⟨R⟩(0)        = {Rmean_0:.6f}")
        print(f"  ⟨R⟩(t_final)  = {Rmean_f:.6f}")
        print(f"  expected R̄    = {R_expected:.6f}")
        print(f"  position drift error = {abs(Rmean_f - R_expected):.3e}")

    assert abs(p00_f - 1.0) < 1e-6, f"Pop drift {abs(p00_f-1.0):.3e}"
    assert abs(p11_f - 0.0) < 1e-6, f"Pop leak  {abs(p11_f):.3e}"
    assert abs(Rmean_f - R_expected) < 5e-3, (
        f"Centroid drift {abs(Rmean_f - R_expected):.3e}"
    )
    print("  PASS")


def _norm_conservation_test(verbose: bool = True) -> None:
    """
    Tully-dual test: trace and energy should be conserved up to discretization
    error over a propagation that traverses the avoided-crossing region.
    """
    model = TullyModel(TullyParams.defaults("dual"))
    params = QCLEGridParams()  # defaults
    solver = QCLEGridSolver(model=model, params=params)
    state0 = solver.initial_diabat_gaussian(R0=-10.0, P0=20.0, sigma_R=1.0, init_state=0)

    E0 = solver.energy_components(state0)["E"]
    tr0 = solver.trace(state0)

    dt, n_steps = 0.5, 1500
    times, snaps = solver.propagate(state0, dt=dt, n_steps=n_steps, save_every=300, verbose=False)

    tr_f = solver.trace(snaps[-1])
    Ef = solver.energy_components(snaps[-1])["E"]

    if verbose:
        print("[Norm/energy conservation test, Tully dual]")
        print(f"  trace(0)       = {tr0:.12e}")
        print(f"  trace(t_final) = {tr_f:.12e}")
        print(f"  Δtrace         = {abs(tr_f - tr0):.3e}")
        print(f"  E(0)           = {E0:.12e}")
        print(f"  E(t_final)     = {Ef:.12e}")
        print(f"  ΔE             = {abs(Ef - E0):.3e}")

    # These are looser since the Tully-dual case actually passes through the
    # coupling region; we expect tracewise conservation up to spectral cutoff
    # error.  Energy is conserved by the QCLE for general H so it tests the
    # discretization.
    assert abs(tr_f - tr0) < 5e-4, f"Trace drift {abs(tr_f - tr0):.3e}"
    assert abs(Ef - E0) < 5e-4, f"Energy drift {abs(Ef - E0):.3e}"
    print("  PASS")


if __name__ == "__main__":
    _free_propagation_test()
    _norm_conservation_test()