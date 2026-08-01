from __future__ import annotations

"""Reproducibility helpers shared by production and reviewer-validation runs.

The original pipeline saved numerical arrays but not enough information to
reproduce them.  This module provides JSON-safe run manifests, deterministic
cloud fingerprints, environment capture, and per-figure metadata sidecars.
"""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Optional

import json
import os
import platform
import subprocess
import sys

import numpy as np


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def array_fingerprint(array: Any) -> str:
    """Content hash used to prove that compared methods share a cloud."""
    a = np.ascontiguousarray(np.asarray(array))
    h = sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(repr(a.shape).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def environment_metadata() -> dict[str, Any]:
    versions: dict[str, Optional[str]] = {}
    for name in ("numpy", "scipy", "matplotlib", "torch", "jax"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            text=True, timeout=2,
        ).strip()
    except Exception:
        commit = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": versions,
        "git_commit": commit,
    }


def build_run_metadata(*, config: Any, state: Any,
                       dynamics: Any = None,
                       extra: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Build the metadata saved beside every PBME/MIDPOINT data file."""
    Z = np.asarray(getattr(state, "Z"))
    gp = getattr(state, "gp", None)
    gp_cfg = getattr(gp, "config", None)
    if gp_cfg is None and hasattr(gp, "_inner"):
        gp_cfg = getattr(gp._inner, "config", None)
    dyn = dynamics if dynamics is not None else getattr(gp, "dynamics", None)
    model = getattr(dyn, "model", None)
    model_params = getattr(model, "params", None)
    dyn_params = getattr(dyn, "params", None)
    meta = {
        "schema_version": 2,
        "configuration": _jsonable(config),
        "support": {
            "n_trajectories": int(Z.shape[0]),
            "dimension": int(Z.shape[1]),
            "initial_cloud_sha256": array_fingerprint(Z),
            "sampling_mode": getattr(state, "sampling_mode", None),
            "sampling_diagnostics": _jsonable(
                getattr(state, "sampling_diagnostics", None)),
        },
        "surrogate": {
            "class": type(gp).__name__ if gp is not None else None,
            "is_product": bool(getattr(gp, "_is_product", False)),
            "configuration": _jsonable(gp_cfg),
            "profile_floor_relative": getattr(gp, "_g_floor_rel", None),
        },
        "physics": {
            "dynamics_parameters": _jsonable(dyn_params),
            "model": type(model).__name__ if model is not None else None,
            "model_parameters": _jsonable(model_params),
            "moment_targets": _jsonable(getattr(state, "moment_targets", None)),
        },
        "normalization_policy": {
            "cloud_observables": "raw Riemann sums are stored; self-normalized aliases are labeled *_sn or lw_*",
            "gp_observables": "raw integrals are stored as *_raw; normalized values divide by the raw GP norm",
        },
        "environment": environment_metadata(),
    }
    if extra:
        meta["extra"] = _jsonable(extra)
    return meta


def write_json(path: os.PathLike[str] | str, payload: Any) -> str:
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True,
                  allow_nan=True)
        handle.write("\n")
    return path


def write_figure_metadata(figure_path: os.PathLike[str] | str, *,
                          title: str, data_sources: list[str],
                          run_metadata: Optional[Mapping[str, Any]] = None,
                          scale_policy: str = "as plotted",
                          normalization: str = "see axis labels",
                          deviations: Optional[str] = None) -> str:
    """Write a machine-readable caption/configuration sidecar for a figure."""
    fig_path = Path(figure_path)
    payload = {
        "figure": fig_path.name,
        "title": title,
        "data_sources": data_sources,
        "scale_policy": scale_policy,
        "normalization": normalization,
        "deviations_from_run_configuration": deviations,
        "run_metadata": _jsonable(run_metadata),
    }
    return write_json(str(fig_path) + ".meta.json", payload)
