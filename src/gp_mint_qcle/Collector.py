from __future__ import annotations

"""
Collector.py
============

Collects per-step diagnostics and periodic (Z, y, GP) snapshots for a
dynamics run, and serializes them to a single .npz + sidecar .json.

Design
------
*   One StepDiagnostics per time step, a flat dict of observables.
*   One Snapshot per snapshot_every steps, with enough state to rebuild
    the GP surrogate for slicing or post-hoc analysis.
*   Collector.as_arrays() returns a dict of numpy arrays for easy plotting.
*   Collector.save(path_no_ext) writes {path}.npz + {path}.json.
*   Collector.load(path_no_ext) restores the full structure.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import json
import os

import numpy as np
from numpy.typing import NDArray

from .Mint import D


FloatArray = NDArray[np.float64]


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class StepDiagnostics:
    """
    One step's observables.  The `values` dict holds the flat observable
    suite produced by Observables.compute_all(...).  The scalar fields
    below track step-level metadata that's independent of physics.
    """
    step_index: int
    t:          float
    wall_time:  float                      # seconds for this step
    sigma_f:    float
    sigma_n:    float
    lengthscales: FloatArray
    fit_rms_on_support: float
    values: Dict[str, float] = field(default_factory=dict)


@dataclass
class Snapshot:
    """
    Full GP state at a given step, enough to reconstruct ρ̂ at any point.

    Density-difference extension
    ----------------------------
    When the pipeline uses a GPDensityDiff surrogate, the Snapshot carries
    BOTH the baseline state (alpha_base, y0, Z0, sigma_f_base, ...) and
    the correction state (alpha_delta, delta, sigma_f_delta, ...).  The
    scalar `is_density_diff` flag distinguishes the two regimes so
    loaders know which fields to expect.

    For legacy GPDensity snapshots, only the top-level (alpha, sigma_f,
    lengthscales, y) fields are populated and is_density_diff=False.

    CRITICAL — alpha semantics differ by regime
    -------------------------------------------
    * is_density_diff=False (vanilla GPDensity):
        alpha  = the FULL density coefficient vector α such that
                 ρ̂(z) = Σ_i alpha_i k(z, Z_i).
    * is_density_diff=True (GPDensityDiff):
        alpha  = the CORRECTION coefficient vector α_δ of the δ-GP only.
        alpha_base = the BASELINE coefficient vector α₀ (frozen after t=0).
        Full density: ρ̂(z) = Σ_i alpha_base_i k_0(z, Z_i)
                             + Σ_i alpha_i     k_δ(z, Z_i).

    Any downstream code that interprets snap.alpha as "the coefficient
    for the full density" will silently compute HALF the density when
    is_density_diff=True.  Always branch on is_density_diff and combine
    alpha_base + alpha where necessary.

    For midpoint runs, ``y`` is the effective label vector actually fitted by
    the GP, i.e. ``correction_weight * raw_initial_y``.  The optional ``weight``
    field stores the correction weights when present.
    """
    step_index: int
    t:          float
    Z:          FloatArray                 # (N, D)
    y:          FloatArray                 # (N,) effective fitted density labels
    alpha:      FloatArray                 # (N,) — see docstring for semantics
    sigma_f:    float
    sigma_n:    float
    lengthscales: FloatArray               # (D,)
    feature_mean: Optional[FloatArray] = None
    feature_std: Optional[FloatArray] = None
    feature_zscore: bool = False
    proposal_density: Optional[FloatArray] = None
    target_density: Optional[FloatArray] = None
    weight: Optional[FloatArray] = None

    # Density-difference extension (all optional, all ignored unless
    # is_density_diff=True).
    is_density_diff: bool = False
    alpha_base:      Optional[FloatArray] = None    # (N,)   baseline α₀
    Z0:              Optional[FloatArray] = None    # (N, D) baseline support positions
    y0:              Optional[FloatArray] = None    # (N,)   initial labels
    sigma_f_base:    Optional[float]      = None
    sigma_n_base:    Optional[float]      = None
    lengthscales_base: Optional[FloatArray] = None  # (D,)
    delta:           Optional[FloatArray] = None    # (N,) = effective y - y0

    def __post_init__(self) -> None:
        if self.is_density_diff and self.alpha_base is None:
            raise ValueError(
                "Snapshot.is_density_diff=True but alpha_base is None.  "
                "A density-diff snapshot must carry both alpha (correction) "
                "and alpha_base (baseline).  Check the snapshot-building path "
                "in Dynamics._snapshot()."
            )


# =============================================================================
# Collector
# =============================================================================

class Collector:
    """
    Accumulates diagnostics over a run.  Agnostic to the scheme name.
    """

    def __init__(self, scheme_name: str):
        self.scheme_name: str = scheme_name
        self.history: List[StepDiagnostics] = []
        self.snapshots: Dict[int, Snapshot] = {}

    # -------------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------------
    def record_diagnostics(self, diag: StepDiagnostics) -> None:
        self.history.append(diag)

    def record_snapshot(self, snap: Snapshot) -> None:
        self.snapshots[snap.step_index] = snap

    # -------------------------------------------------------------------------
    # Dict-of-arrays view (for plotting / analysis)
    # -------------------------------------------------------------------------
    def as_arrays(self) -> Dict[str, FloatArray]:
        H = self.history
        if not H:
            return {}

        out: Dict[str, FloatArray] = {
            "step_index": np.array([d.step_index for d in H]),
            "t":          np.array([d.t for d in H]),
            "wall_time":  np.array([d.wall_time for d in H]),
            "sigma_f":    np.array([d.sigma_f for d in H]),
            "sigma_n":    np.array([d.sigma_n for d in H]),
            "lengthscales": np.stack([d.lengthscales for d in H], axis=0),
            "fit_rms_on_support":
                np.array([d.fit_rms_on_support for d in H]),
        }

        # Observable keys: union across all diag.values dicts, filled with
        # NaN where missing.
        keys = sorted({k for d in H for k in d.values.keys()})
        for k in keys:
            out[k] = np.array([d.values.get(k, float("nan")) for d in H],
                              dtype=np.float64)
        return out

    # -------------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------------
    def save(self, path_no_ext: str) -> str:
        """
        Save to `{path_no_ext}.npz`.  Returns the .npz path.  Also writes
        {path_no_ext}.json with the scheme name and snapshot step indices.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path_no_ext)),
                    exist_ok=True)

        arrays = self.as_arrays()

        # Snapshots -> flattened keys snap_{step:06d}_{field}
        for step, snap in self.snapshots.items():
            pref = f"snap_{step:06d}_"
            arrays[pref + "Z"]            = snap.Z
            arrays[pref + "y"]            = snap.y
            arrays[pref + "alpha"]        = snap.alpha
            arrays[pref + "sigma_f"]      = np.array([snap.sigma_f])
            arrays[pref + "sigma_n"]      = np.array([snap.sigma_n])
            arrays[pref + "lengthscales"] = snap.lengthscales
            arrays[pref + "t"]            = np.array([snap.t])
            arrays[pref + "feature_zscore"] = np.array([1 if snap.feature_zscore else 0], dtype=np.int64)
            if snap.feature_mean is not None:
                arrays[pref + "feature_mean"] = snap.feature_mean
            if snap.feature_std is not None:
                arrays[pref + "feature_std"] = snap.feature_std
            if snap.proposal_density is not None:
                arrays[pref + "proposal_density"] = snap.proposal_density
            if snap.target_density is not None:
                arrays[pref + "target_density"] = snap.target_density
            if snap.weight is not None:
                arrays[pref + "weight"] = snap.weight

            # Density-difference extras
            arrays[pref + "is_density_diff"] = np.array(
                [1 if snap.is_density_diff else 0], dtype=np.int64)
            if snap.alpha_base is not None:
                arrays[pref + "alpha_base"] = snap.alpha_base
            if snap.Z0 is not None:
                arrays[pref + "Z0"] = snap.Z0
            if snap.y0 is not None:
                arrays[pref + "y0"] = snap.y0
            if snap.sigma_f_base is not None:
                arrays[pref + "sigma_f_base"] = np.array([snap.sigma_f_base])
            if snap.sigma_n_base is not None:
                arrays[pref + "sigma_n_base"] = np.array([snap.sigma_n_base])
            if snap.lengthscales_base is not None:
                arrays[pref + "lengthscales_base"] = snap.lengthscales_base
            if snap.delta is not None:
                arrays[pref + "delta"] = snap.delta

        npz_path = path_no_ext + ".npz"
        np.savez_compressed(npz_path, **arrays)

        meta = {
            "scheme_name":      self.scheme_name,
            "n_steps":          len(self.history),
            "snapshot_steps":   sorted(self.snapshots.keys()),
            "observable_keys":  sorted({k for d in self.history
                                        for k in d.values.keys()}),
        }
        with open(path_no_ext + ".json", "w") as f:
            json.dump(meta, f, indent=2)
        return npz_path

    @staticmethod
    def load(path_no_ext: str) -> Dict[str, Any]:
        """
        Load a saved run.  Returns dict with keys:
            'meta'       : metadata dict
            'arrays'     : full array dict (time-series)
            'snapshots'  : {step_index: Snapshot}
        """
        with open(path_no_ext + ".json", "r") as f:
            meta = json.load(f)
        data = dict(np.load(path_no_ext + ".npz"))

        snapshots: Dict[int, Snapshot] = {}
        for step in meta["snapshot_steps"]:
            pref = f"snap_{step:06d}_"
            snap = Snapshot(
                step_index=step,
                t=float(data[pref + "t"][0]),
                Z=data[pref + "Z"],
                y=data[pref + "y"],
                alpha=data[pref + "alpha"],
                sigma_f=float(data[pref + "sigma_f"][0]),
                sigma_n=float(data[pref + "sigma_n"][0]),
                lengthscales=data[pref + "lengthscales"],
                feature_mean=data[pref + "feature_mean"] if pref + "feature_mean" in data else None,
                feature_std=data[pref + "feature_std"] if pref + "feature_std" in data else None,
                feature_zscore=bool(int(data[pref + "feature_zscore"][0])) if pref + "feature_zscore" in data else False,
                proposal_density=data[pref + "proposal_density"] if pref + "proposal_density" in data else None,
                target_density=data[pref + "target_density"] if pref + "target_density" in data else None,
                weight=data[pref + "weight"] if pref + "weight" in data else None,
                # Density-difference extras (all optional; default to None on legacy NPZ)
                is_density_diff=bool(int(data[pref + "is_density_diff"][0])) if pref + "is_density_diff" in data else False,
                alpha_base=data[pref + "alpha_base"] if pref + "alpha_base" in data else None,
                Z0=data[pref + "Z0"] if pref + "Z0" in data else None,
                y0=data[pref + "y0"] if pref + "y0" in data else None,
                sigma_f_base=float(data[pref + "sigma_f_base"][0]) if pref + "sigma_f_base" in data else None,
                sigma_n_base=float(data[pref + "sigma_n_base"][0]) if pref + "sigma_n_base" in data else None,
                lengthscales_base=data[pref + "lengthscales_base"] if pref + "lengthscales_base" in data else None,
                delta=data[pref + "delta"] if pref + "delta" in data else None,
            )
            snapshots[step] = snap

        # Filter out snapshot keys from arrays
        arrays = {k: v for k, v in data.items() if not k.startswith("snap_")}
        return {"meta": meta, "arrays": arrays, "snapshots": snapshots}