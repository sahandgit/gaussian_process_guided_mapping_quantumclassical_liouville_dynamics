from __future__ import annotations

# --- UTF-8 console safety (see run.py) ---
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
# -----------------------------------------

r"""
conservative_excess.py
======================

Conservative / weak-form discretization of the mapping-QCLE excess term --
the second "longer" recommendation of the review (report sections 5, 6.2,
15/Ch.5 Q2, 15/Ch.7 Q2).

The problem
-----------
In continuum form the excess term is a pure bath-momentum divergence,

    Q[rho] = - d/dP  J_P[rho],

    J_P[rho] = (hbar/8) sum_{lam,lam'} dhbar_{lam lam'}/dR
               ( d^2/dr_lam' dr_lam + d^2/dp_lam' dp_lam ) rho ,

so under decaying boundary conditions  \int Q dP = 0  exactly: the excess term
moves density in momentum but neither creates nor destroys it.

The production scheme collocates Q pointwise on a moving cloud.  That
discretization does NOT inherit the continuum null space, which is why the
campaign observes raw normalization drift.  Renormalizing after the step is
not a fix -- it masks an inconsistent source.

What this module provides
-------------------------
A finite-volume discretization in the bath-momentum direction for which the
discrete normalization functional is an EXACT left null vector of the discrete
generator, to machine precision, independent of the flux values:

    1^T A = 0     =>     d/dt ( sum_k rho_k dP ) = 0 .

The construction is the standard telescoping-flux argument: with cell averages
rho_k and face fluxes J_{k+1/2},

    d rho_k / dt = - ( J_{k+1/2} - J_{k-1/2} ) / dP ,

summing over k cancels every interior face and leaves only the two boundary
faces.  Zero-flux (or periodic) boundaries therefore give exact discrete
conservation regardless of how the fluxes are computed -- including from a GP
surrogate.

Scope
-----
This is a one-dimensional-in-P prototype demonstrating the conservation
property and providing the operator assembly.  It is not yet coupled to the
moving-cloud transport in ``Dynamics.py``; adopting it is a follow-on step.
Use it to (a) demonstrate in the thesis that the drift is a property of the
discretization rather than of the physics, and (b) benchmark against the
collocated source.
"""

import argparse
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

FloatArray = np.ndarray


# ===========================================================================
# Finite-volume assembly in the bath-momentum direction
# ===========================================================================

@dataclass
class MomentumGrid:
    """Uniform cell-centred grid in P with cell width dP."""
    P_min: float
    P_max: float
    n_P: int

    @property
    def dP(self) -> float:
        return (self.P_max - self.P_min) / self.n_P

    @property
    def centres(self) -> FloatArray:
        return self.P_min + self.dP * (0.5 + np.arange(self.n_P))

    @property
    def faces(self) -> FloatArray:
        return self.P_min + self.dP * np.arange(self.n_P + 1)


def divergence_matrix(grid: MomentumGrid, boundary: str = "zero-flux"
                      ) -> FloatArray:
    r"""
    Discrete  -d/dP  acting on FACE fluxes and returning CELL rates.

    Returns ``Dv`` of shape (n_P, n_P+1) with

        (Dv @ J)_k = -( J_{k+1/2} - J_{k-1/2} ) / dP .

    ``boundary``:
      * ``"zero-flux"`` -- the two outermost faces are forced to zero, so the
        column sums vanish and conservation is exact.
      * ``"periodic"``  -- the first and last face are identified.
    """
    n = int(grid.n_P)
    dP = grid.dP
    Dv = np.zeros((n, n + 1))
    for k in range(n):
        Dv[k, k + 1] -= 1.0 / dP
        Dv[k, k] += 1.0 / dP
    if boundary == "zero-flux":
        Dv[:, 0] = 0.0
        Dv[:, -1] = 0.0
    elif boundary == "periodic":
        # identify face 0 with face n: fold the last column onto the first
        Dv[:, 0] += Dv[:, -1]
        Dv = Dv[:, :-1]
        Dv = np.hstack([Dv, Dv[:, :1]])
    else:
        raise ValueError(f"unknown boundary {boundary!r}")
    return Dv


def face_interpolation_matrix(grid: MomentumGrid,
                              boundary: str = "zero-flux") -> FloatArray:
    """
    Cell-to-face averaging, shape (n_P+1, n_P).  Interior faces take the mean
    of their two neighbours; boundary faces are zero under ``zero-flux``.
    """
    n = int(grid.n_P)
    Af = np.zeros((n + 1, n))
    for f in range(1, n):
        Af[f, f - 1] = 0.5
        Af[f, f] = 0.5
    if boundary == "periodic":
        Af[0, -1] = Af[0, 0] = 0.5
        Af[n, -1] = Af[n, 0] = 0.5
    return Af


def conservative_generator(grid: MomentumGrid,
                           flux_operator: FloatArray,
                           boundary: str = "zero-flux") -> FloatArray:
    r"""
    Assemble  A = Dv @ F  where ``flux_operator`` F maps cell values to FACE
    fluxes, shape (n_P+1, n_P).

    The resulting cell-rate operator satisfies ``1^T A = 0`` exactly under
    zero-flux or periodic boundaries, so the discrete normalization
    ``sum_k rho_k dP`` is conserved to machine precision.
    """
    Dv = divergence_matrix(grid, boundary)
    F = np.asarray(flux_operator, float)
    if F.shape[0] != grid.n_P + 1:
        raise ValueError(f"flux operator must have {grid.n_P+1} rows, got {F.shape}")
    return Dv @ F


def conservation_residual(A: FloatArray) -> float:
    """max_j |sum_i A_ij| -- zero means exact discrete conservation."""
    return float(np.max(np.abs(np.sum(np.asarray(A, float), axis=0))))


# ===========================================================================
# Weak (Galerkin) form -- transfers one derivative onto the test space
# ===========================================================================

def weak_form_matrix(grid: MomentumGrid, test_basis: FloatArray,
                     flux_operator: FloatArray,
                     boundary: str = "zero-flux") -> FloatArray:
    r"""
    Galerkin excess operator.

    Starting from  <psi, Q[rho]> = -<psi, dJ/dP>  and integrating by parts,

        <psi, Q[rho]> = <dpsi/dP, J[rho]>  -  [ psi J ]_boundary ,

    so a test function with vanishing boundary trace never sees the boundary
    term, and a CONSTANT test function gives identically zero -- which is the
    conservation statement in weak form.

    ``test_basis`` has shape (n_test, n_P): rows are test functions sampled at
    cell centres.  The first row should be the constant function if the caller
    wants the conservation property to be visible in the assembled matrix.
    """
    Psi = np.asarray(test_basis, float)
    n = int(grid.n_P)
    if Psi.shape[1] != n:
        raise ValueError(f"test basis must have {n} columns")
    # d psi / dP by centred differences with the chosen boundary rule
    dPsi = np.zeros_like(Psi)
    dPsi[:, 1:-1] = (Psi[:, 2:] - Psi[:, :-2]) / (2.0 * grid.dP)
    if boundary == "periodic":
        dPsi[:, 0] = (Psi[:, 1] - Psi[:, -1]) / (2.0 * grid.dP)
        dPsi[:, -1] = (Psi[:, 0] - Psi[:, -2]) / (2.0 * grid.dP)
    # cell-centred flux from the face operator
    Af = face_interpolation_matrix(grid, boundary)
    F_cell = np.linalg.pinv(Af) @ np.asarray(flux_operator, float)
    return (dPsi * grid.dP) @ F_cell


# ===========================================================================
# Reference flux from the analytic SEO representation
# ===========================================================================

def seo_flux_operator(grid: MomentumGrid, x_probe: FloatArray,
                      dhbar_dR: FloatArray, hbar: float = 1.0,
                      boundary: str = "zero-flux") -> FloatArray:
    r"""
    Build a face-flux operator for a density represented in the exact SEO
    basis, using ``seo_coefficient_gp`` for the analytic mapping derivatives.

    The returned matrix maps the four SEO coefficient values at cell centres
    (flattened) to face fluxes.  This is a demonstration of how a
    projection-preserving representation and a conservative discretization
    compose: the mapping derivatives are exact, and the P-divergence is
    telescoping.
    """
    from seo_coefficient_gp import seo_mapping_laplacian
    D = seo_mapping_laplacian(np.asarray(x_probe, float).reshape(1, 4), hbar)
    dh = np.asarray(dhbar_dR, float).reshape(2, 2)
    # scalar weight per basis function a
    w = (hbar / 8.0) * np.einsum("ij,aij->a", dh, D[0])          # (4,)
    n = int(grid.n_P)
    Af = face_interpolation_matrix(grid, boundary)                # (n+1, n)
    # block-diagonal over the four coefficient fields, weighted by w
    F = np.zeros((n + 1, 4 * n))
    for a in range(4):
        F[:, a * n:(a + 1) * n] = w[a] * Af
    return F


# ===========================================================================
# Self-test
# ===========================================================================

def run_self_test() -> None:
    rng = np.random.default_rng(0)
    grid = MomentumGrid(P_min=-10.0, P_max=10.0, n_P=64)

    # 1. Divergence matrix telescopes: column sums vanish (zero-flux).
    Dv = divergence_matrix(grid, "zero-flux")
    assert Dv.shape == (64, 65)
    assert np.max(np.abs(np.sum(Dv, axis=0))) < 1e-12

    # 2. Conservation for an ARBITRARY flux operator -- the key property.
    F = rng.standard_normal((65, 64))
    A = conservative_generator(grid, F, "zero-flux")
    assert A.shape == (64, 64)
    res = conservation_residual(A)
    assert res < 1e-10, res

    # 3. Conservation holds for a second random flux (not a lucky draw).
    A2 = conservative_generator(grid, rng.standard_normal((65, 64)))
    assert conservation_residual(A2) < 1e-10

    # 4. Time evolution preserves total mass to machine precision.
    rho = np.exp(-0.5 * (grid.centres / 2.0) ** 2)
    m0 = float(np.sum(rho) * grid.dP)
    r = rho.copy()
    for _ in range(200):
        r = r + 1e-3 * (A @ r)
    m1 = float(np.sum(r) * grid.dP)
    assert abs(m1 - m0) < 1e-10 * max(1.0, abs(m0)), (m0, m1)

    # 5. Periodic boundaries also conserve.
    Ap = conservative_generator(grid, rng.standard_normal((65, 64)), "periodic")
    assert conservation_residual(Ap) < 1e-10

    # 6. Weak form: a constant test function gives identically zero, which is
    #    the conservation statement in Galerkin form.
    Psi = np.vstack([np.ones(64), grid.centres / 10.0,
                     np.sin(grid.centres / 3.0)])
    W = weak_form_matrix(grid, Psi, rng.standard_normal((65, 64)))
    assert np.max(np.abs(W[0])) < 1e-10, np.max(np.abs(W[0]))
    assert np.max(np.abs(W[1])) > 0.0        # non-constant tests are non-trivial

    # 7. Composed with the exact SEO representation.
    try:
        x = rng.uniform(-1.0, 1.0, size=4)
        dh = np.array([[0.05, 0.02], [0.02, -0.05]])
        Fs = seo_flux_operator(grid, x, dh)
        As = conservative_generator(grid, Fs, "zero-flux")
        assert As.shape == (64, 256)
        # conservation over each coefficient block
        assert conservation_residual(As) < 1e-10
    except ImportError:                                       # pragma: no cover
        print("  (seo_coefficient_gp unavailable; skipped composed test)")

    print("[self-test] conservative_excess checks passed "
          "(telescoping divergence, exact conservation for arbitrary flux, "
          "mass preserved under evolution, periodic case, weak-form constant "
          "null space, SEO composition).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        run_self_test()
    else:
        print("Conservative excess discretization. "
              "Run with --self-test to validate.")


if __name__ == "__main__":
    main()
