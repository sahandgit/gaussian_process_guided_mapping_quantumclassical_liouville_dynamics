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
seo_coefficient_gp.py
=====================

Route B: the projection-preserving successor representation recommended by the
examiner (report sections 2.1, 15/Ch.4 Q1, 15/Ch.7 Q2).

Instead of learning a generic six-dimensional modulation and multiplying it by
a fixed mapping profile, this module represents the projected mapping density
in the EXACT finite SEO basis derived in the thesis:

    rho(R, P, x, t) = sum_a c_a(R, P, t) phi_a(x),        a = 1..4

for a two-state subsystem, where the four real basis functions are

    phi_1 = E(x) [ (2/hbar)(r0^2 + p0^2) - 1 ]          (population 0)
    phi_2 = E(x) [ (2/hbar)(r1^2 + p1^2) - 1 ]          (population 1)
    phi_3 = E(x) (2/hbar)(r0 r1 + p0 p1)                (Re coherence)
    phi_4 = E(x) (2/hbar)(r0 p1 - r1 p0)                (Im coherence)
    E(x)  = (pi hbar)^{-2} exp(-(r0^2+r1^2+p0^2+p1^2)/hbar)

matching ``ReviewerValidation.seo_basis_matrix``.

Why this fixes the two structural defects
-----------------------------------------
1. **Zero leakage by construction.**  Any field of this form lies in the image
   of the SEO projector, so the projection residual is identically zero rather
   than the 0.27--0.95 measured for the product ansatz.  ``projection_residual``
   verifies this numerically.

2. **No off-manifold derivative inference.**  The excess operator differentiates
   twice with respect to mapping variables.  Here those derivatives act on the
   ANALYTIC ``phi_a`` and are available in closed form everywhere, including
   normal to the focused shell.  The Gaussian process only ever supplies
   ``d c_a / d P`` -- a derivative along a bath direction that the support
   actually samples.  The learned domain drops from six dimensions to two.

The excess source becomes

    Q[rho](R,P,x)
      = -(hbar/8) sum_{lam,lam'} dhbar_{lam lam'}/dR
        * sum_a [ D_{lam lam'} phi_a ](x) * dc_a/dP (R,P)

with  D_{lam lam'} = d^2/dr_{lam'} dr_{lam} + d^2/dp_{lam'} dp_{lam}.

Status
------
This module is a research implementation of the successor method.  It is
self-contained (NumPy only) and unit-tested against finite differences.  It is
NOT wired into ``run.py``; adopting it is a deliberate follow-on step that
requires re-running the production campaign.
"""

import argparse
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

FloatArray = np.ndarray

# Mapping-coordinate layout used throughout: x = (r0, r1, p0, p1)
_IR = (0, 1)          # indices of r_lambda
_IP = (2, 3)          # indices of p_lambda
N_BASIS = 4


# ===========================================================================
# The exact SEO basis and its analytic mapping derivatives
# ===========================================================================

def seo_envelope(x: FloatArray, hbar: float = 1.0) -> FloatArray:
    """E(x) = (pi hbar)^{-2} exp(-|x|^2 / hbar) for a two-state subsystem."""
    x = np.asarray(x, float).reshape(-1, 4)
    return (np.pi * hbar) ** -2 * np.exp(-np.sum(x * x, axis=1) / hbar)


def _polynomials(x: FloatArray, hbar: float) -> FloatArray:
    """The four polynomial factors P_a(x), shape (N, 4)."""
    r0, r1, p0, p1 = x.T
    return np.column_stack([
        (2.0 / hbar) * (r0 * r0 + p0 * p0) - 1.0,
        (2.0 / hbar) * (r1 * r1 + p1 * p1) - 1.0,
        (2.0 / hbar) * (r0 * r1 + p0 * p1),
        (2.0 / hbar) * (r0 * p1 - r1 * p0),
    ])


def _poly_grad(x: FloatArray, hbar: float) -> FloatArray:
    """dP_a/dx_u, shape (N, 4 basis, 4 coords)."""
    r0, r1, p0, p1 = x.T
    n = x.shape[0]
    g = np.zeros((n, N_BASIS, 4))
    c = 2.0 / hbar
    # P_1 = c(r0^2+p0^2) - 1
    g[:, 0, 0] = 2.0 * c * r0
    g[:, 0, 2] = 2.0 * c * p0
    # P_2 = c(r1^2+p1^2) - 1
    g[:, 1, 1] = 2.0 * c * r1
    g[:, 1, 3] = 2.0 * c * p1
    # P_3 = c(r0 r1 + p0 p1)
    g[:, 2, 0] = c * r1
    g[:, 2, 1] = c * r0
    g[:, 2, 2] = c * p1
    g[:, 2, 3] = c * p0
    # P_4 = c(r0 p1 - r1 p0)
    g[:, 3, 0] = c * p1
    g[:, 3, 1] = -c * p0
    g[:, 3, 2] = -c * r1
    g[:, 3, 3] = c * r0
    return g


def _poly_hess(x: FloatArray, hbar: float) -> FloatArray:
    """d^2P_a/dx_u dx_v, shape (N, 4 basis, 4, 4).  Constant in x."""
    n = x.shape[0]
    h = np.zeros((n, N_BASIS, 4, 4))
    c = 2.0 / hbar
    h[:, 0, 0, 0] = 2.0 * c
    h[:, 0, 2, 2] = 2.0 * c
    h[:, 1, 1, 1] = 2.0 * c
    h[:, 1, 3, 3] = 2.0 * c
    h[:, 2, 0, 1] = h[:, 2, 1, 0] = c
    h[:, 2, 2, 3] = h[:, 2, 3, 2] = c
    h[:, 3, 0, 3] = h[:, 3, 3, 0] = c
    h[:, 3, 1, 2] = h[:, 3, 2, 1] = -c
    return h


def seo_basis(x: FloatArray, hbar: float = 1.0) -> FloatArray:
    """phi_a(x), shape (N, 4).  Matches ReviewerValidation.seo_basis_matrix."""
    x = np.asarray(x, float).reshape(-1, 4)
    return seo_envelope(x, hbar)[:, None] * _polynomials(x, hbar)


def seo_basis_hessian(x: FloatArray, hbar: float = 1.0) -> FloatArray:
    r"""
    Analytic d^2 phi_a / dx_u dx_v, shape (N, 4 basis, 4, 4).

    With phi = E * P and dE/dx_u = -(2 x_u / hbar) E,

        d2(E P)/du dv = [ (4 x_u x_v / hbar^2) - (2/hbar) delta_uv ] E P
                        - (2 x_u / hbar) E dP/dv
                        - (2 x_v / hbar) E dP/du
                        + E d2P/du dv .
    """
    x = np.asarray(x, float).reshape(-1, 4)
    E = seo_envelope(x, hbar)                      # (N,)
    P = _polynomials(x, hbar)                      # (N,4)
    dP = _poly_grad(x, hbar)                       # (N,4,4)
    d2P = _poly_hess(x, hbar)                      # (N,4,4,4)
    two_x = (2.0 / hbar) * x                       # (N,4)
    eye = np.eye(4)[None, :, :]                    # (1,4,4)

    # envelope second-derivative factor, (N,4,4)
    envf = (two_x[:, :, None] * two_x[:, None, :] - (2.0 / hbar) * eye)

    term1 = envf[:, None, :, :] * P[:, :, None, None]
    term2 = -two_x[:, None, :, None] * dP[:, :, None, :]
    term3 = -two_x[:, None, None, :] * dP[:, :, :, None]
    return E[:, None, None, None] * (term1 + term2 + term3 + d2P)


def seo_mapping_laplacian(x: FloatArray, hbar: float = 1.0) -> FloatArray:
    r"""
    D_{lam lam'} phi_a = (d^2/dr_lam' dr_lam + d^2/dp_lam' dp_lam) phi_a.

    Returns shape (N, 4 basis, 2, 2) indexed [:, a, lam, lam'].
    This is the exact mapping-derivative combination the excess operator needs,
    available everywhere -- including normal to the focused shell.
    """
    H = seo_basis_hessian(x, hbar)                 # (N,4,4,4)
    n = H.shape[0]
    out = np.zeros((n, N_BASIS, 2, 2))
    for lam in range(2):
        for lamp in range(2):
            out[:, :, lam, lamp] = (H[:, :, _IR[lamp], _IR[lam]]
                                    + H[:, :, _IP[lamp], _IP[lam]])
    return out


# ===========================================================================
# Coefficient-field surrogate over the bath plane
# ===========================================================================

def _rbf(A: FloatArray, B: FloatArray, ell: FloatArray, sf2: float) -> FloatArray:
    A = np.asarray(A, float) / ell
    B = np.asarray(B, float) / ell
    d2 = (np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2.0 * A @ B.T)
    return sf2 * np.exp(-0.5 * np.clip(d2, 0.0, None))


@dataclass
class SEOCoefficientSurrogate:
    r"""
    Independent ARD-RBF GPs for the four bath-dependent coefficient fields
    c_a(R, P), plus analytic derivatives with respect to P.

    Only bath coordinates are learned.  The mapping dependence is exact.
    """
    hbar: float = 1.0
    lengthscales: Tuple[float, float] = (1.0, 1.0)   # (R, P)
    sigma_f2: float = 1.0
    sigma_n2: float = 1e-8
    X: Optional[FloatArray] = None                   # (M, 2) bath centres
    alpha: Optional[FloatArray] = None               # (M, 4) coefficients
    _ell: FloatArray = field(default=None, repr=False)

    # -- fitting -----------------------------------------------------------
    def fit(self, X_bath: FloatArray, C: FloatArray) -> "SEOCoefficientSurrogate":
        """Fit to observed coefficient fields C (M,4) at bath points (M,2)."""
        X = np.asarray(X_bath, float).reshape(-1, 2)
        C = np.asarray(C, float).reshape(X.shape[0], N_BASIS)
        self._ell = np.asarray(self.lengthscales, float)
        K = _rbf(X, X, self._ell, self.sigma_f2)
        K[np.diag_indices_from(K)] += self.sigma_n2
        self.X = X
        self.alpha = np.linalg.solve(K, C)           # (M,4)
        return self

    @staticmethod
    def coefficients_from_density(y: FloatArray, x: FloatArray,
                                  hbar: float = 1.0) -> FloatArray:
        r"""
        Recover c_a at a bath point from density values on mapping probes.

        Least-squares projection of the sampled density onto the SEO basis:
        this is exactly the operation whose residual the leakage diagnostic
        measures for the product ansatz.
        """
        B = seo_basis(x, hbar)                       # (n_probe, 4)
        c, *_ = np.linalg.lstsq(B, np.asarray(y, float).reshape(-1), rcond=None)
        return c

    # -- prediction --------------------------------------------------------
    def _require_fit(self) -> None:
        if self.X is None or self.alpha is None:
            raise RuntimeError("SEOCoefficientSurrogate is not fitted.")

    def coefficients(self, X_bath: FloatArray) -> FloatArray:
        """c_a(R,P), shape (n, 4)."""
        self._require_fit()
        Xq = np.asarray(X_bath, float).reshape(-1, 2)
        return _rbf(Xq, self.X, self._ell, self.sigma_f2) @ self.alpha

    def dc_dP(self, X_bath: FloatArray) -> FloatArray:
        r"""
        dc_a/dP, shape (n, 4), analytic.

        For the RBF kernel, dk/dP_q = ((P_j - P_q)/ell_P^2) k.
        """
        self._require_fit()
        Xq = np.asarray(X_bath, float).reshape(-1, 2)
        K = _rbf(Xq, self.X, self._ell, self.sigma_f2)          # (n,M)
        dP = (self.X[None, :, 1] - Xq[:, None, 1]) / self._ell[1] ** 2
        return (K * dP) @ self.alpha

    def density(self, X_bath: FloatArray, x: FloatArray) -> FloatArray:
        """rho(R,P,x) = sum_a c_a(R,P) phi_a(x); broadcast over paired rows."""
        C = self.coefficients(X_bath)                            # (n,4)
        B = seo_basis(x, self.hbar)                              # (n,4)
        return np.einsum("na,na->n", C, B)

    # -- excess operator ---------------------------------------------------
    def excess_source(self, X_bath: FloatArray, x: FloatArray,
                      dhbar_dR: FloatArray) -> FloatArray:
        r"""
        Q[rho] at paired points, using analytic mapping derivatives.

        ``dhbar_dR`` is the traceless subsystem-Hamiltonian derivative
        d hbar^{lam lam'} / dR evaluated at each row's R, shape (n, 2, 2).
        """
        D = seo_mapping_laplacian(x, self.hbar)                  # (n,4,2,2)
        dC = self.dc_dP(X_bath)                                  # (n,4)
        contracted = np.einsum("naij,na->nij", D, dC)            # (n,2,2)
        dh = np.asarray(dhbar_dR, float).reshape(-1, 2, 2)
        return -(self.hbar / 8.0) * np.einsum("nij,nij->n", dh, contracted)

    # -- diagnostics -------------------------------------------------------
    def projection_residual(self, X_bath: FloatArray, x_probe: FloatArray
                            ) -> float:
        r"""
        Relative L2 residual of the represented field against the SEO span.

        Zero to machine precision by construction -- this is the quantity that
        measured 0.27--0.95 for the product ansatz.
        """
        C = self.coefficients(np.asarray(X_bath).reshape(1, 2))[0]   # (4,)
        B = seo_basis(x_probe, self.hbar)                            # (n,4)
        y = B @ C
        coef, *_ = np.linalg.lstsq(B, y, rcond=None)
        resid = y - B @ coef
        return float(np.linalg.norm(resid) / max(np.linalg.norm(y), 1e-300))


# ===========================================================================
# Self-test
# ===========================================================================

def _fd_hessian(f: Callable[[FloatArray], float], x0: FloatArray,
                h: float = 1e-4) -> FloatArray:
    n = x0.size
    H = np.zeros((n, n))
    for u in range(n):
        for v in range(n):
            eu = np.zeros(n); eu[u] = h
            ev = np.zeros(n); ev[v] = h
            H[u, v] = (f(x0 + eu + ev) - f(x0 + eu - ev)
                       - f(x0 - eu + ev) + f(x0 - eu - ev)) / (4 * h * h)
    return H


def run_self_test() -> None:
    rng = np.random.default_rng(0)
    hbar = 1.0

    # 1. Basis matches the pipeline's independent definition.
    x = rng.uniform(-1.2, 1.2, size=(9, 4))
    try:
        from ReviewerValidation import seo_basis_matrix
        assert np.allclose(seo_basis(x, hbar), seo_basis_matrix(x, hbar),
                           atol=1e-12), "basis mismatch vs ReviewerValidation"
    except ImportError:                                   # pragma: no cover
        pass

    # 2. Analytic Hessian vs central finite differences.
    H = seo_basis_hessian(x, hbar)
    for a in range(N_BASIS):
        for i in (0, 4, 8):
            f = lambda z, a=a: float(seo_basis(z.reshape(1, 4), hbar)[0, a])
            Hfd = _fd_hessian(f, x[i])
            err = np.max(np.abs(H[i, a] - Hfd))
            assert err < 5e-5, (a, i, err)

    # 3. Mapping Laplacian equals the intended derivative combination.
    D = seo_mapping_laplacian(x, hbar)
    for a in range(N_BASIS):
        for lam in range(2):
            for lamp in range(2):
                expect = (H[:, a, _IR[lamp], _IR[lam]]
                          + H[:, a, _IP[lamp], _IP[lam]])
                assert np.allclose(D[:, a, lam, lamp], expect, atol=1e-12)

    # 4. Coefficient recovery is exact for a field built from the basis.
    c_true = np.array([0.7, -0.3, 0.15, -0.05])
    xp = rng.uniform(-1.5, 1.5, size=(200, 4))
    y = seo_basis(xp, hbar) @ c_true
    c_rec = SEOCoefficientSurrogate.coefficients_from_density(y, xp, hbar)
    assert np.allclose(c_rec, c_true, atol=1e-10), c_rec

    # 5. Zero leakage by construction.
    Xb = rng.uniform(-2, 2, size=(40, 2))
    C = np.column_stack([np.sin(Xb[:, 0]), np.cos(Xb[:, 1]),
                         0.1 * Xb[:, 0], 0.05 * Xb[:, 1]])
    gp = SEOCoefficientSurrogate(lengthscales=(1.0, 1.0),
                                 sigma_n2=1e-10).fit(Xb, C)
    leak = gp.projection_residual(np.array([0.3, 0.4]), xp)
    assert leak < 1e-10, leak

    # 6. dc/dP against finite differences.
    q = np.array([[0.25, -0.4]])
    an = gp.dc_dP(q)[0]
    eps = 1e-6
    fd = (gp.coefficients(q + np.array([[0.0, eps]]))[0]
          - gp.coefficients(q - np.array([[0.0, eps]]))[0]) / (2 * eps)
    assert np.max(np.abs(an - fd)) < 1e-6, (an, fd)

    # 7. Excess source is finite and linear in dhbar/dR.
    n = 5
    Xq = rng.uniform(-1, 1, size=(n, 2))
    xq = rng.uniform(-1, 1, size=(n, 4))
    dh = rng.uniform(-0.1, 0.1, size=(n, 2, 2))
    dh = 0.5 * (dh + np.transpose(dh, (0, 2, 1)))     # symmetric, traceless-ish
    q1 = gp.excess_source(Xq, xq, dh)
    q2 = gp.excess_source(Xq, xq, 2.0 * dh)
    assert np.all(np.isfinite(q1))
    assert np.allclose(q2, 2.0 * q1, atol=1e-12)

    print("[self-test] seo_coefficient_gp checks passed "
          "(basis, Hessian vs FD, Laplacian, recovery, zero leakage, "
          "dc/dP vs FD, source linearity).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        run_self_test()
    else:
        print("Route B representation. Run with --self-test to validate.")


if __name__ == "__main__":
    main()
