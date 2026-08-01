from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


TullyKind = Literal["simple", "dual", "extended"]


@dataclass(frozen=True)
class TullyParams:
    """
    Parameter container for the 1D two-state Tully models.

    All quantities are assumed to be in atomic units.
    """
    kind: TullyKind = "dual"
    A: float = 0.10
    B: float = 0.28
    C: float = 0.015
    D: float = 0.06
    E0: float = 0.05

    @staticmethod
    def defaults(kind: TullyKind = "dual") -> "TullyParams":
        if kind == "simple":    # single avoided crossing
            return TullyParams(kind="simple", A=0.01, B=1.6,  C=0.005, D=1.0,  E0=0.0)
        if kind == "dual":      # dual avoided crossing
            return TullyParams(kind="dual",   A=0.10, B=0.28, C=0.015, D=0.06, E0=0.05)
        if kind == "extended":  # extended-coupling reflection model
            return TullyParams(kind="extended", A=0.01, B=1.0, C=0.005, D=0.10, E0=0.0)
        raise ValueError("Unknown Tully kind.")


class TullyModel:
    """
    Vectorized 1D, 2-state Tully diabatic model.

    The diabatic Hamiltonian is
        H(R) = [[V11(R), V12(R)],
                [V12(R), V22(R)]]

    Input:
        R : scalar or array-like of nuclear coordinates

    Output conventions:
        - scalar input -> arrays with leading shape ()
        - vector input -> arrays with leading shape (N,)
    """

    nstates: int = 2
    ndim_nuclear: int = 1

    def __init__(self, params: TullyParams | None = None) -> None:
        self.params = params if params is not None else TullyParams.defaults("dual")

    @property
    def kind(self) -> TullyKind:
        return self.params.kind

    def __repr__(self) -> str:
        p = self.params
        return (
            f"TullyModel(kind={p.kind!r}, A={p.A}, B={p.B}, "
            f"C={p.C}, D={p.D}, E0={p.E0})"
        )

    # ------------------------------------------------------------------
    # internal utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _asarray_1d(R: ArrayLike) -> NDArray[np.float64]:
        arr = np.asarray(R, dtype=np.float64)
        return arr

    # ------------------------------------------------------------------
    # diabatic potentials
    # ------------------------------------------------------------------
    def V11(self, R: ArrayLike) -> NDArray[np.float64]:
        R = self._asarray_1d(R)
        p = self.params

        if p.kind == "simple":
            # Standard piecewise form:
            # V11 = A (1 - exp(-B R)), R >= 0
            #     = -A (1 - exp(+B R)), R < 0
            return np.where(
                R >= 0.0,
                p.A * (1.0 - np.exp(-p.B * R)),
                -p.A * (1.0 - np.exp(p.B * R)),
            )

        if p.kind == "dual":
            return np.zeros_like(R)

        if p.kind == "extended":
            return np.full_like(R, p.A)

        raise ValueError(f"Unsupported Tully kind: {p.kind}")

    def V22(self, R: ArrayLike) -> NDArray[np.float64]:
        R = self._asarray_1d(R)
        p = self.params

        if p.kind == "simple":
            return -self.V11(R)

        if p.kind == "dual":
            return -p.A * np.exp(-p.B * R**2) + p.E0

        if p.kind == "extended":
            return -np.full_like(R, p.A)

        raise ValueError(f"Unsupported Tully kind: {p.kind}")

    def V12(self, R: ArrayLike) -> NDArray[np.float64]:
        R = self._asarray_1d(R)
        p = self.params

        if p.kind in ("simple", "dual"):
            return p.C * np.exp(-p.D * R**2)

        if p.kind == "extended":
            return p.C * np.where(R >= 0.0, 2.0 - np.exp(-p.D * R), np.exp(p.D * R))

        raise ValueError(f"Unsupported Tully kind: {p.kind}")

    # ------------------------------------------------------------------
    # first derivatives
    # ------------------------------------------------------------------
    def dV11_dR(self, R: ArrayLike) -> NDArray[np.float64]:
        R = self._asarray_1d(R)
        p = self.params

        if p.kind == "simple":
            return p.A * p.B * np.exp(-p.B * np.abs(R))

        if p.kind == "dual":
            return np.zeros_like(R)

        if p.kind == "extended":
            return np.zeros_like(R)

        raise ValueError(f"Unsupported Tully kind: {p.kind}")

    def dV22_dR(self, R: ArrayLike) -> NDArray[np.float64]:
        R = self._asarray_1d(R)
        p = self.params

        if p.kind == "simple":
            return -self.dV11_dR(R)

        if p.kind == "dual":
            return 2.0 * p.A * p.B * R * np.exp(-p.B * R**2)

        if p.kind == "extended":
            return np.zeros_like(R)

        raise ValueError(f"Unsupported Tully kind: {p.kind}")

    def dV12_dR(self, R: ArrayLike) -> NDArray[np.float64]:
        R = self._asarray_1d(R)
        p = self.params

        if p.kind in ("simple", "dual"):
            return -2.0 * p.C * p.D * R * np.exp(-p.D * R**2)

        if p.kind == "extended":
            return p.C * p.D * np.exp(-p.D * np.abs(R))

        raise ValueError(f"Unsupported Tully kind: {p.kind}")

    # ------------------------------------------------------------------
    # assembled matrix objects
    # ------------------------------------------------------------------
    def diabatic_potential(self, R: ArrayLike) -> NDArray[np.float64]:
        """
        Return the 2x2 diabatic potential matrix for each R.

        Shape:
            scalar R   -> (2, 2)
            vector R   -> (..., 2, 2)
        """
        R = self._asarray_1d(R)
        v11 = self.V11(R)
        v22 = self.V22(R)
        v12 = self.V12(R)

        H = np.zeros(R.shape + (2, 2), dtype=np.float64)
        H[..., 0, 0] = v11
        H[..., 1, 1] = v22
        H[..., 0, 1] = v12
        H[..., 1, 0] = v12
        return H

    def d_diabatic_potential_dR(self, R: ArrayLike) -> NDArray[np.float64]:
        """
        Return dH/dR for each R.

        Shape:
            scalar R   -> (2, 2)
            vector R   -> (..., 2, 2)
        """
        R = self._asarray_1d(R)
        dv11 = self.dV11_dR(R)
        dv22 = self.dV22_dR(R)
        dv12 = self.dV12_dR(R)

        dH = np.zeros(R.shape + (2, 2), dtype=np.float64)
        dH[..., 0, 0] = dv11
        dH[..., 1, 1] = dv22
        dH[..., 0, 1] = dv12
        dH[..., 1, 0] = dv12
        return dH

    # ------------------------------------------------------------------
    # adiabatic quantities
    # ------------------------------------------------------------------
    def adiabatic_energies(self, R: ArrayLike) -> NDArray[np.float64]:
        """
        Eigenvalues of the diabatic Hamiltonian, sorted ascending.

        Shape:
            scalar R   -> (2,)
            vector R   -> (..., 2)
        """
        H = self.diabatic_potential(R)
        return np.linalg.eigvalsh(H)

    def adiabatic_states(self, R: ArrayLike) -> NDArray[np.float64]:
        """
        Orthogonal eigenvector matrix U(R), column-wise eigenvectors.

        Shape:
            scalar R   -> (2, 2)
            vector R   -> (..., 2, 2)
        """
        H = self.diabatic_potential(R)
        _, U = np.linalg.eigh(H)
        return U

    def adiabatic_gap(self, R: ArrayLike) -> NDArray[np.float64]:
        E = self.adiabatic_energies(R)
        return E[..., 1] - E[..., 0]

    # ------------------------------------------------------------------
    # convenience interface
    # ------------------------------------------------------------------
    def evaluate(self, R: ArrayLike) -> dict[str, NDArray[np.float64]]:
        """
        Collect the main quantities needed downstream.
        """
        return {
            "R": self._asarray_1d(R),
            "H_d": self.diabatic_potential(R),
            "dH_dR": self.d_diabatic_potential_dR(R),
            "E_ad": self.adiabatic_energies(R),
            "U_ad": self.adiabatic_states(R),
            "gap": self.adiabatic_gap(R),
        }
    
if __name__ == "__main__":
    model = TullyModel()  # default: dual avoided crossing
    R = np.linspace(-10.0, 10.0, 500)

    out = model.evaluate(R)

    print(model)
    print("H_d shape:", out["H_d"].shape)     # (500, 2, 2)
    print("dH_dR shape:", out["dH_dR"].shape) # (500, 2, 2)
    print("E_ad shape:", out["E_ad"].shape)   # (500, 2)
    print("gap min/max:", out["gap"].min(), out["gap"].max())