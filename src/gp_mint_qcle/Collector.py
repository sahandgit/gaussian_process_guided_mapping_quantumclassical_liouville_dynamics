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
from typing import Any, Dict, Iterable, List, Optional

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
    geometric_measure: Optional[FloatArray] = None

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

    # Product-surrogate metadata.  Without these fields, post-processing
    # silently reconstructed the inner modulation GP ``mu`` as though it were
    # the physical density ``rho = g*mu``.
    is_product: bool = False
    product_hbar: Optional[float] = None
    product_init_state: Optional[int] = None
    product_nstates: Optional[int] = None
    product_g_floor_rel: Optional[float] = None
    product_transported: bool = False

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

    def __init__(self, scheme_name: str,
                 run_metadata: Optional[Dict[str, Any]] = None):
        self.scheme_name: str = scheme_name
        self.run_metadata: Dict[str, Any] = dict(run_metadata or {})
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
            if snap.geometric_measure is not None:
                arrays[pref + "geometric_measure"] = snap.geometric_measure

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

            arrays[pref + "is_product"] = np.array(
                [1 if snap.is_product else 0], dtype=np.int64)
            arrays[pref + "product_transported"] = np.array(
                [1 if snap.product_transported else 0], dtype=np.int64)
            if snap.product_hbar is not None:
                arrays[pref + "product_hbar"] = np.array([snap.product_hbar])
            if snap.product_init_state is not None:
                arrays[pref + "product_init_state"] = np.array(
                    [snap.product_init_state], dtype=np.int64)
            if snap.product_nstates is not None:
                arrays[pref + "product_nstates"] = np.array(
                    [snap.product_nstates], dtype=np.int64)
            if snap.product_g_floor_rel is not None:
                arrays[pref + "product_g_floor_rel"] = np.array(
                    [snap.product_g_floor_rel])

        npz_path = path_no_ext + ".npz"
        np.savez_compressed(npz_path, **arrays)

        meta = {
            "scheme_name":      self.scheme_name,
            "n_steps":          len(self.history),
            "snapshot_steps":   sorted(self.snapshots.keys()),
            "observable_keys":  sorted({k for d in self.history
                                        for k in d.values.keys()}),
            "run_metadata":     self.run_metadata,
        }
        with open(path_no_ext + ".json", "w") as f:
            json.dump(meta, f, indent=2)
        return npz_path

    # -------------------------------------------------------------------------
    # Lightweight metadata peek (no array I/O at all)
    # -------------------------------------------------------------------------
    @staticmethod
    def peek_snapshot_steps(path_no_ext: str) -> List[int]:
        """
        Return the available snapshot step indices from the JSON sidecar
        WITHOUT opening the (possibly several-hundred-MB) .npz.

        Figure code uses this to decide *which* snapshots it actually needs
        (e.g. a strided subset) before paying for any array decompression.
        """
        with open(path_no_ext + ".json", "r") as f:
            meta = json.load(f)
        return [int(s) for s in meta.get("snapshot_steps", [])]

    @staticmethod
    def _read_snapshot(z: Any, keys: set, step: int) -> Snapshot:
        """
        Reconstruct a single Snapshot from an *open* ``NpzFile`` ``z``,
        reading only that snapshot's members.  Each ``z[member]`` access
        decompresses exactly one array on demand, so the other (thousands of)
        snapshots are never materialised.
        """
        pref = f"snap_{step:06d}_"

        def has(name: str) -> bool:
            return (pref + name) in keys

        def get(name: str):
            return z[pref + name] if (pref + name) in keys else None

        return Snapshot(
            step_index=step,
            t=float(z[pref + "t"][0]),
            Z=z[pref + "Z"],
            y=z[pref + "y"],
            alpha=z[pref + "alpha"],
            sigma_f=float(z[pref + "sigma_f"][0]),
            sigma_n=float(z[pref + "sigma_n"][0]),
            lengthscales=z[pref + "lengthscales"],
            feature_mean=get("feature_mean"),
            feature_std=get("feature_std"),
            feature_zscore=bool(int(z[pref + "feature_zscore"][0])) if has("feature_zscore") else False,
            proposal_density=get("proposal_density"),
            target_density=get("target_density"),
            weight=get("weight"),
            geometric_measure=get("geometric_measure"),
            # Density-difference extras (all optional; default to None on legacy NPZ)
            is_density_diff=bool(int(z[pref + "is_density_diff"][0])) if has("is_density_diff") else False,
            alpha_base=get("alpha_base"),
            Z0=get("Z0"),
            y0=get("y0"),
            sigma_f_base=float(z[pref + "sigma_f_base"][0]) if has("sigma_f_base") else None,
            sigma_n_base=float(z[pref + "sigma_n_base"][0]) if has("sigma_n_base") else None,
            lengthscales_base=get("lengthscales_base"),
            delta=get("delta"),
            is_product=bool(int(z[pref + "is_product"][0])) if has("is_product") else False,
            product_hbar=float(z[pref + "product_hbar"][0]) if has("product_hbar") else None,
            product_init_state=int(z[pref + "product_init_state"][0]) if has("product_init_state") else None,
            product_nstates=int(z[pref + "product_nstates"][0]) if has("product_nstates") else None,
            product_g_floor_rel=float(z[pref + "product_g_floor_rel"][0]) if has("product_g_floor_rel") else None,
            product_transported=bool(int(z[pref + "product_transported"][0])) if has("product_transported") else False,
        )

    @staticmethod
    def load(path_no_ext: str,
             arrays_only: bool = False,
             snapshot_steps: Optional[Iterable[int]] = None) -> Dict[str, Any]:
        """
        Load a saved run.  Returns dict with keys:
            'meta'       : metadata dict
            'arrays'     : time-series array dict
            'snapshots'  : {step_index: Snapshot}

        Memory-safe selective loading
        -----------------------------
        The on-disk .npz stores every periodic snapshot (positions, labels,
        coefficients, ...) as a separate member.  A long run with a small
        ``snapshot_every`` can hold *thousands* of snapshots totalling
        hundreds of MB.  The legacy implementation did
        ``dict(np.load(path))``, which eagerly DECOMPRESSES AND MATERIALISES
        every member into RAM at once — the direct cause of ``MemoryError``
        during figure generation, where the time-series plots need none of
        the snapshots.

        Two opt-in controls avoid that:

        * ``arrays_only=True``     → load only the time-series arrays and
                                     return ``snapshots={}``.  Use for any
                                     figure that plots observables vs. time.
        * ``snapshot_steps=[...]`` → load only those snapshots (e.g. the
                                     strided subset actually rendered as
                                     density-marginal panels).  Steps not
                                     present on disk are silently ignored.

        With neither argument the behaviour matches the legacy "load
        everything" contract, but it is still implemented with per-member
        lazy reads (the open ``NpzFile`` decompresses one array at a time)
        rather than one giant ``dict(...)`` allocation.
        """
        with open(path_no_ext + ".json", "r") as f:
            meta = json.load(f)

        snapshots: Dict[int, Snapshot] = {}
        with np.load(path_no_ext + ".npz") as z:
            keys = set(z.files)
            # Time-series arrays: every member that is NOT a snapshot field.
            # These are O(n_steps) 1-D series — small and always safe to load.
            arrays = {k: z[k] for k in keys if not k.startswith("snap_")}

            if not arrays_only:
                avail = [int(s) for s in meta.get("snapshot_steps", [])]
                if snapshot_steps is None:
                    steps = avail
                else:
                    want = {int(s) for s in snapshot_steps}
                    steps = [s for s in avail if s in want]
                for step in steps:
                    snapshots[step] = Collector._read_snapshot(z, keys, step)

        return {"meta": meta, "arrays": arrays, "snapshots": snapshots}
