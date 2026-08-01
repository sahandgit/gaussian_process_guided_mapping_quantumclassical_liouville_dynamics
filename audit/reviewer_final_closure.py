"""
Resumable final-evidence orchestration for the MSc reviewer closure.

This file deliberately contains no scientific propagator, GP, KDE, TDSE,
grid-QCLE, or excess-operator implementation.  It calls the existing modules,
records every resolved input and output, and builds reader-facing tables from
machine-readable results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO = Path(__file__).resolve().parent
DEFAULT_FIXED = REPO / "reviewer_closure_20260723_194254"
DEFAULT_REPLICATION = REPO / "reviewer_closure_20260726_174927"
OBSERVABLES = (
    "P0", "P1", "trace", "energy", "R_mean", "P_mean", "R_var", "P_var",
)
DTYPE = "float64"
DEVICE = "cpu"
ETA_LEVELS = (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
T_QUANTILES = (0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999)
JOB_RECORD_LOCK = threading.Lock()

# Declared absolute-plus-relative numerical resolvability rule used by every
# three-level order diagnostic.  Both successive differences must exceed this
# floor before an order is interpreted.
NUMERICAL_NOISE_ABS_TOL = 1.0e-12
NUMERICAL_NOISE_REL_TOL = 1.0e-12
MAX_INTERPRETABLE_ORDER = 6.0


def numerical_noise_threshold(values: Sequence[float]) -> float:
    """Return tau_abs + tau_rel * max_k |u_k| for the supplied levels."""
    scale = max((abs(float(value)) for value in values), default=0.0)
    return NUMERICAL_NOISE_ABS_TOL + NUMERICAL_NOISE_REL_TOL * scale


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]],
              fields: Optional[Sequence[str]] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        ordered: List[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    ordered.append(key)
                    seen.add(key)
        fields = ordered
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (json.dumps(value, separators=(",", ":"), default=json_default)
                      if isinstance(value, (list, tuple, dict)) else value)
                for key, value in row.items()
            })
    tmp.replace(path)
    return path


def copy_case_compatible(source: Path, destination: Path) -> Path:
    """Copy ``source`` unless both spellings resolve to one filesystem path.

    Windows filesystems are normally case-insensitive, so names that differ
    only by case (for example ``table_data_crosswalk.csv`` and
    ``TABLE_DATA_CROSSWALK.csv``) identify the same file.  Calling
    ``shutil.copy2`` in that situation raises ``WinError 32``.  On a
    case-sensitive filesystem the two paths remain distinct and the requested
    compatibility copy is created normally.
    """
    source = Path(source)
    destination = Path(destination)
    same_filesystem_path = (
        os.path.normcase(os.path.abspath(source))
        == os.path.normcase(os.path.abspath(destination))
    )
    if same_filesystem_path:
        return source
    shutil.copy2(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise RuntimeError("compatibility copy is not byte-identical")
    return destination


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def package_versions() -> Dict[str, Optional[str]]:
    names = ("numpy", "scipy", "torch", "jax", "matplotlib", "pandas")
    out: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def source_control_record() -> Dict[str, Any]:
    probe = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        return {
            "system": "git",
            "repository_present": False,
            "repository_root": "NOT IDENTIFIABLE",
            "commit": "NOT IDENTIFIABLE",
            "reason": "the copied workspace has no .git metadata",
        }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    return {
        "system": "git",
        "repository_present": True,
        "repository_root": probe.stdout.strip(),
        "commit": (
            commit.stdout.strip() if commit.returncode == 0 else "NOT IDENTIFIABLE"
        ),
        "working_tree_dirty": (
            bool(status.stdout.strip()) if status.returncode == 0
            else "NOT IDENTIFIABLE"
        ),
    }


def environment_record(argv: Sequence[str]) -> Dict[str, Any]:
    return {
        "timestamp_utc": utcnow(),
        "command": subprocess.list2cmdline([sys.executable, *argv]),
        "argv": list(argv),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "packages": package_versions(),
        "platform": platform.platform(),
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "dtype": DTYPE,
        "device": DEVICE,
        "cpu_count": os.cpu_count(),
        "source_control": source_control_record(),
    }


def command_history(out: Path, argv: Sequence[str]) -> None:
    path = out / "commands" / "executed_commands.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{utcnow()}\t{subprocess.list2cmdline([sys.executable, *argv])}\n")


def write_reproducibility_docs(out: Path) -> None:
    env = environment_record(sys.argv)
    environment_dir = out / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    (environment_dir / "python_version.txt").write_text(
        sys.version + "\n", encoding="utf-8"
    )
    (environment_dir / "platform.txt").write_text(
        json.dumps(env["operating_system"], indent=2) + "\n",
        encoding="utf-8",
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    if freeze.returncode != 0:
        raise RuntimeError(
            "pip freeze failed while recording the audit environment: "
            + freeze.stderr.strip()
        )
    (environment_dir / "pip_freeze.txt").write_text(
        freeze.stdout, encoding="utf-8"
    )
    requirements = [
        f"python=={platform.python_version()}",
        *[
            f"{name}=={version}" if version else f"{name}==DATA ABSENT"
            for name, version in env["packages"].items()
        ],
        "",
        "Declared project requirements:",
    ]
    declared = REPO / "requirements.txt"
    if declared.exists():
        requirements.extend(declared.read_text(encoding="utf-8").splitlines())
    (out / "requirements_audit.txt").write_text(
        "\n".join(requirements) + "\n", encoding="utf-8"
    )
    readme = """# Final reviewer closure evidence

This directory is generated by `reviewer_final_closure.py` from the repository
root. Paths in the commands below are relative to that root.
Raw simulations outside this directory are read-only. Reused calculations are
accepted only after their manifest, PBME/MIDPOINT pairing, physical endpoint,
and finite declared observables are verified.

Run with Python 3.10 or newer:

```powershell
python reviewer_final_closure.py --mode plan --out .\\final_reviewer_closure `
  --production-dir-P0-20 .\\results\\P0_20 `
  --production-dir-P0-100 .\\results\\P0_100

python reviewer_final_closure.py --mode execute --out .\\final_reviewer_closure `
  --P0 20 100 --manufactured-l2 1e-6 0.01 0.05 `
  --manufactured-N 300 600 1200 2400 `
  --manufactured-seeds 123 124 125 --dynamics-seeds 11 29 47 73 `
  --dt-levels 0.5 0.25 0.125 --support-levels 500 1000 2000 `
  --parallel-workers 3 --parallel-dynamics-workers 2 --resume

python reviewer_final_closure.py --mode analyze --out .\\final_reviewer_closure
python reviewer_final_closure.py --mode verify --out .\\final_reviewer_closure
```

After the thesis and response compile, run package with the five document
arguments printed by `--help`. The versioned GitHub release is a public release
record, not a DOI or institutional persistent identifier. No DOI is invented
by this audit.

Machine-readable values retain full precision. LaTeX tables are generated
from declared CSV files, and `TABLE_DATA_CROSSWALK.csv` records each link
(with a byte-identical lowercase compatibility copy).
Independent support clouds are never described as deterministic convergence.
Parallel subprocesses receive an explicit CPU-thread budget; this affects
scheduling only and is recorded in the job JSONL.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")


def l2_tag(value: float) -> str:
    return f"{float(value):.12g}".replace("-", "m").replace(".", "p").replace("+", "")


def equal_number(a: Any, b: Any, tol: float = 1e-12) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    except (TypeError, ValueError):
        return a == b


def reference_modes_complete(
    rows: Sequence[Mapping[str, Any]], P0_values: Sequence[float]
) -> bool:
    """Return whether every requested momentum has time and grid evidence.

    CSV writers may serialize an integral momentum as either ``20`` or
    ``20.0``.  Treat the momentum as numeric while keeping the refinement mode
    as an exact categorical value.
    """
    return all(
        any(
            equal_number(row.get("P0"), P0)
            and row.get("refinement_mode") == mode
            for row in rows
        )
        for P0 in P0_values
        for mode in ("time", "grid")
    )


def manifest_args(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return None
    try:
        return read_json(path).get("cli_arguments", {})
    except Exception:
        return None


def validate_run(run_dir: Path, expected: Mapping[str, Any]) -> Tuple[bool, str]:
    args = manifest_args(run_dir)
    if args is None:
        return False, "run_manifest.json absent or unreadable"
    aliases = {"N": "n_train"}
    conflicts = []
    for key, expected_value in expected.items():
        actual = args.get(aliases.get(key, key))
        if not equal_number(actual, expected_value):
            conflicts.append(f"{key}={actual!r}, expected {expected_value!r}")
    if conflicts:
        return False, "configuration conflict: " + "; ".join(conflicts)
    for method in ("pbme", "midpoint"):
        npz = run_dir / f"{method}.npz"
        meta = run_dir / f"{method}.json"
        if not npz.exists() or not meta.exists():
            return False, f"{method} output absent"
        try:
            with np.load(npz) as data:
                if "t" not in data.files:
                    return False, f"{method}: time array absent"
                t = np.asarray(data["t"], float)
                if t.size < 2 or not np.all(np.isfinite(t)):
                    return False, f"{method}: invalid time array"
                t_final = float(args.get("t_final_resolved", args.get("t_final", np.nan)))
                if not np.isfinite(t_final) or not math.isclose(
                    float(t[-1]), t_final, rel_tol=1e-10, abs_tol=1e-10
                ):
                    return False, f"{method}: endpoint {t[-1]} != {t_final}"
            # Undefined diagnostics may be represented by NaN (for example a
            # PBME-only "delta alpha" field).  They remain visible in the
            # numerical-stability audit but do not make a completed physical
            # time history unusable.  Completion is therefore tested on the
            # declared observables, not every optional diagnostic column.
            from Compare_gp_se_qcle import load_collector_run
            physical = load_collector_run(str(npz))
            for key in ("t", *OBSERVABLES):
                arr = np.asarray(physical[key], float)
                if arr.size < 2 or not np.all(np.isfinite(arr)):
                    return False, f"{method}: nonfinite/incomplete physical series {key}"
        except Exception as exc:
            return False, f"{method}: unreadable output ({exc})"
    return True, "complete, finite, endpoint verified"


def all_candidate_run_dirs(out: Path) -> List[Path]:
    roots = [out / "dynamics", DEFAULT_FIXED, DEFAULT_REPLICATION]
    candidates: List[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(path.parent for path in root.rglob("run_manifest.json"))
    return sorted(set(candidates), key=lambda p: str(p).lower())


def resolve_run(out: Path, expected: Mapping[str, Any]) -> Tuple[Optional[Path], str]:
    reasons: List[str] = []
    valid: List[Path] = []
    for run_dir in all_candidate_run_dirs(out):
        args = manifest_args(run_dir)
        if args is None:
            continue
        if not all(equal_number(args.get("n_train" if k == "N" else k), v)
                   for k, v in expected.items()):
            continue
        ok, why = validate_run(run_dir, expected)
        if ok:
            valid.append(run_dir)
        else:
            reasons.append(f"{run_dir}: {why}")
    if valid:
        # Prefer new closure outputs, then the four-seed campaign, then fixed.
        def rank(path: Path) -> Tuple[int, str]:
            text = str(path)
            return (
                0 if str(out / "dynamics") in text else
                1 if str(DEFAULT_REPLICATION) in text else 2,
                text,
            )
        chosen = sorted(valid, key=rank)[0]
        return chosen, "reused verified run"
    return None, " | ".join(reasons) if reasons else "no matching run discovered"


def resolve_method_run(out: Path, expected: Mapping[str, Any],
                       method: str) -> Tuple[Optional[Path], str]:
    """Resolve one completed method from an otherwise interrupted paired run."""
    for run_dir in all_candidate_run_dirs(out):
        args = manifest_args(run_dir)
        if args is None or not all(
            equal_number(args.get("n_train" if key == "N" else key), value)
            for key, value in expected.items()
        ):
            continue
        npz = run_dir / f"{method}.npz"
        meta = run_dir / f"{method}.json"
        if not npz.exists() or not meta.exists():
            continue
        try:
            from Compare_gp_se_qcle import load_collector_run
            data = load_collector_run(str(npz))
            t = np.asarray(data["t"], float)
            t_final = float(args.get("t_final_resolved", args.get("t_final")))
            if (
                t.size >= 2 and np.all(np.isfinite(t))
                and math.isclose(float(t[-1]), t_final, rel_tol=1e-10, abs_tol=1e-10)
                and all(np.all(np.isfinite(np.asarray(data[key], float)))
                        for key in OBSERVABLES)
            ):
                return run_dir, "single-method artifact finite and endpoint verified"
        except Exception:
            continue
    return None, f"no valid {method} artifact discovered"


def manufactured_path(out: Path, l2: float, n_train: int, seed: int) -> Path:
    return out / "manufactured" / "runs" / f"l2_{l2_tag(l2)}" / f"N{n_train}_seed{seed}"


def manufactured_valid(path: Path, l2: float, n_train: int, seed: int) -> Tuple[bool, str]:
    metric = path / "manufactured_operator_metrics.json"
    if not metric.exists():
        return False, "metrics absent"
    try:
        data = read_json(metric)
        checks = (
            equal_number(data.get("l2_regularization"), l2),
            int(data.get("n_train", -1)) == int(n_train),
            int(data.get("seed", -1)) == int(seed),
            int(data.get("n_query_off_support", -1)) == 1000,
            int(data.get("n_query_on_support", -1)) == int(n_train),
        )
        if not all(checks):
            return False, "configuration conflict"
        for query in ("on_support", "off_support"):
            for quantity in ("density", "gradient", "operator_Q"):
                for metric_name in (
                    "mae", "rmse", "linf", "relative_l1", "relative_l2", "relative_linf"
                ):
                    value = data["metrics"][query][quantity][metric_name]
                    if not np.isfinite(float(value)):
                        return False, f"nonfinite {query}/{quantity}/{metric_name}"
        return True, "complete and finite"
    except Exception as exc:
        return False, f"unreadable metrics ({exc})"


@dataclass
class Job:
    kind: str
    job_id: str
    status: str
    output_dir: str
    parameters: Dict[str, Any]
    source: Optional[str] = None
    reason: str = ""
    command: Optional[List[str]] = None


def dynamics_command(out_dir: Path, *, P0: float, seed: int, n_train: int,
                     dt: float) -> List[str]:
    from reviewer_closure_campaign import (
        collision_time, run_py_cmd, snapshot_every_for,
    )
    t_c = collision_time(2000.0, -15.0, P0)
    return run_py_cmd(
        sys.executable, REPO / "run.py", out_dir,
        P0=P0, n_train=n_train, dt=dt, t_final=2.0 * t_c, seed=seed,
        snapshot_every=snapshot_every_for(t_c, dt),
        density_mode="full", sampling_mode="focused", surrogate="product",
        l2_regularization=0.05, R0=-15.0, sigma_R=1.0, mass=2000.0,
        hbar=1.0, abs_target=False, refit_hyper_policy="breathing",
    )


def build_plan(args: argparse.Namespace) -> Dict[str, Any]:
    out = args.out.resolve()
    jobs: List[Job] = []
    for l2 in args.manufactured_l2:
        for n_train in args.manufactured_N:
            for seed in args.manufactured_seeds:
                target = manufactured_path(out, l2, n_train, seed)
                ok, why = manufactured_valid(target, l2, n_train, seed)
                command = [
                    sys.executable, str(REPO / "ReviewerValidation.py"), "manufactured",
                    "--out", str(target), "--n-train", str(n_train),
                    "--n-query", "1000", "--seed", str(seed), "--l2", repr(float(l2)),
                ]
                jobs.append(Job(
                    "manufactured", f"manufactured_l2{l2:g}_N{n_train}_seed{seed}",
                    "REUSE" if ok else "MISSING", str(target),
                    {"l2": l2, "N": n_train, "seed": seed, "n_query": 1000},
                    str(target) if ok else None, why, command,
                ))

    for P0 in args.P0:
        t_final = 2.0 * 2000.0 * 15.0 / abs(P0)
        for seed in args.dynamics_seeds:
            for dt in args.dt_levels:
                expected = {
                    "P0": P0, "seed": seed, "N": 1000, "dt": dt,
                    "t_final_resolved": t_final, "l2_regularization": 0.05,
                }
                source, why = resolve_run(out, expected)
                target = out / "dynamics" / "timestep" / f"P0{P0:g}" / f"seed{seed}_dt{dt:g}"
                jobs.append(Job(
                    "timestep", f"timestep_P0{P0:g}_seed{seed}_dt{dt:g}",
                    "REUSE" if source else "MISSING", str(target),
                    dict(expected), str(source) if source else None, why,
                    dynamics_command(target, P0=P0, seed=seed, n_train=1000, dt=dt),
                ))

    for P0 in args.P0:
        t_final = 2.0 * 2000.0 * 15.0 / abs(P0)
        for seed in (11, 29, 47):
            for n_train in args.support_levels:
                expected = {
                    "P0": P0, "seed": seed, "N": n_train, "dt": 0.25,
                    "t_final_resolved": t_final, "l2_regularization": 0.05,
                }
                source, why = resolve_run(out, expected)
                target = out / "dynamics" / "support" / f"P0{P0:g}" / f"seed{seed}_N{n_train}"
                command = dynamics_command(
                    target, P0=P0, seed=seed, n_train=n_train, dt=0.25
                )
                parameters = dict(expected)
                if source is None:
                    pbme_source, pbme_reason = resolve_method_run(
                        out, expected, "pbme"
                    )
                    if pbme_source is not None:
                        parameters["reuse_pbme_source"] = str(pbme_source)
                        parameters["reuse_pbme_reason"] = pbme_reason
                        command.extend(["--run_methods", "midpoint"])
                jobs.append(Job(
                    "support", f"support_P0{P0:g}_seed{seed}_N{n_train}",
                    "REUSE" if source else "MISSING", str(target),
                    parameters, str(source) if source else None, why,
                    command,
                ))

    for method in ("tdse", "qcle"):
        for mode in ("time", "grid"):
            for P0 in args.P0:
                target = out / f"reference_{'grid_qcle' if method == 'qcle' else 'tdse'}" / f"{method}_{mode}_P0{P0:g}.json"
                ok = False
                why = "result absent"
                if target.exists():
                    try:
                        data = read_json(target)
                        expected_cfg = reference_configuration(method, P0)
                        recorded_cfg = data.get("resolved_configuration", {})
                        config_matches = (
                            set(expected_cfg).issubset(recorded_cfg)
                            and all(
                                equal_number(recorded_cfg.get(key), value)
                                for key, value in expected_cfg.items()
                            )
                        )
                        ok = (
                            data.get("status") == "COMPLETE"
                            and data.get("method") == method
                            and data.get("refine_mode") == mode
                            and len(data.get("levels", [])) == 3
                            and config_matches
                            and (
                                mode != "time"
                                or target.with_suffix(".npz").exists()
                            )
                        )
                        why = (
                            "complete configuration-matched three-level result"
                            if ok else "invalid or configuration-conflicting result"
                        )
                    except Exception as exc:
                        why = f"unreadable result ({exc})"
                jobs.append(Job(
                    "reference", f"{method}_{mode}_P0{P0:g}",
                    "REUSE" if ok else "MISSING", str(target),
                    {"method": method, "mode": mode, "P0": P0},
                    str(target) if ok else None, why, None,
                ))

    records = [asdict(job) for job in jobs]
    counts: Dict[str, Dict[str, int]] = {}
    for job in jobs:
        counts.setdefault(job.kind, {"expected": 0, "reuse": 0, "missing": 0})
        counts[job.kind]["expected"] += 1
        counts[job.kind]["reuse" if job.status == "REUSE" else "missing"] += 1
    production_audit: List[Dict[str, Any]] = []
    for declared_P0, directory in (
        (20.0, args.production_dir_P0_20),
        (100.0, args.production_dir_P0_100),
    ):
        directory = Path(directory).resolve()
        manifest = directory / "run_manifest.json"
        if not manifest.exists():
            production_audit.append({
                "declared_P0": declared_P0, "directory": str(directory),
                "status": "DATA ABSENT", "manifest": str(manifest),
            })
            continue
        data = read_json(manifest)
        recorded = data.get("cli_arguments", {}).get("P0")
        production_audit.append({
            "declared_P0": declared_P0, "recorded_P0": recorded,
            "directory": str(directory), "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "status": (
                "VERIFIED"
                if equal_number(recorded, declared_P0)
                else "CONFIGURATION CONFLICT"
            ),
            "use_policy": (
                "eligible only if verified"
                if equal_number(recorded, declared_P0)
                else "excluded from momentum-specific evidence"
            ),
        })
    return {
        "created_utc": utcnow(),
        "environment": environment_record(sys.argv),
        "resolved_arguments": vars(args),
        "input_hashes": {
            str(path): sha256_file(path)
            for path in (
                REPO / "reviewer_final_closure.py",
                REPO / "ReviewerValidation.py",
                REPO / "GP_Density.py",
                REPO / "run.py",
                REPO / "thesis_closure.py",
                REPO / "Compare_gp_se_qcle.py",
                REPO / "qcle_grid_tully.py",
            )
            if path.exists()
        },
        "counts": counts,
        "production_directory_audit": production_audit,
        "jobs": records,
        "notes": [
            "REUSE means the manifest, paired PBME/MIDPOINT files, finite arrays, and endpoint were verified.",
            "Support clouds are independent across N; no deterministic support order will be calculated.",
            "The permanent public archive identifier remains an external publication action.",
        ],
    }


def append_job_record(out: Path, record: Mapping[str, Any]) -> None:
    path = out / "commands" / "job_records.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Manufactured fits may be scheduled concurrently.  Serialize each JSONL
    # append so the command history remains independently parseable.
    with JOB_RECORD_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, default=json_default, allow_nan=False) + "\n"
            )


def run_subprocess_job(
    out: Path,
    job: Mapping[str, Any],
    threads_per_worker: Optional[int] = None,
) -> bool:
    command = [str(x) for x in job["command"]]
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log = out / "commands" / "logs" / f"{job['job_id']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    record: Dict[str, Any] = {
        "job_id": job["job_id"],
        "kind": job["kind"],
        "parameters": job["parameters"],
        "command": command,
        "start_utc": utcnow(),
        "status": "RUNNING",
        "code_file": (
            "ReviewerValidation.py::manufactured_test"
            if job["kind"] == "manufactured"
            else "run.py::main"
        ),
        "dtype": DTYPE,
        "device": DEVICE,
        "retries": 0,
    }
    pbme_source_text = job["parameters"].get("reuse_pbme_source")
    if pbme_source_text:
        pbme_source = Path(pbme_source_text)
        for filename in ("pbme.npz", "pbme.json"):
            source_file = pbme_source / filename
            if not source_file.exists():
                raise FileNotFoundError(
                    f"recovery source artifact absent: {source_file}"
                )
            destination = output_dir / filename
            shutil.copy2(source_file, destination)
        source_manifest = read_json(pbme_source / "run_manifest.json")
        record["reused_pbme"] = {
            "source_directory": str(pbme_source),
            "pbme_npz_sha256": sha256_file(output_dir / "pbme.npz"),
            "pbme_json_sha256": sha256_file(output_dir / "pbme.json"),
            "paired_initial_cloud_sha256": source_manifest.get(
                "paired_initial_cloud_sha256"
            ),
            "copy_policy": "byte-identical copy into new recovery output; raw source unchanged",
        }
    append_job_record(out, record)
    try:
        worker_env = os.environ.copy()
        if threads_per_worker is not None:
            thread_text = str(max(1, int(threads_per_worker)))
            for variable in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            ):
                worker_env[variable] = thread_text
            record["parallel_thread_budget"] = {
                "threads_per_worker": int(thread_text),
                "environment_variables": {
                    variable: worker_env[variable]
                    for variable in (
                        "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    )
                },
            }
        with log.open("w", encoding="utf-8", errors="replace") as handle:
            handle.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
            handle.flush()
            proc = subprocess.run(
                command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT,
                check=False, env=worker_env,
            )
        record["return_code"] = int(proc.returncode)
        record["status"] = "COMPLETE" if proc.returncode == 0 else "FAILED"
        if proc.returncode != 0:
            record["exception"] = f"subprocess exit code {proc.returncode}"
        elif pbme_source_text:
            recovered_manifest = read_json(output_dir / "run_manifest.json")
            old_hash = record["reused_pbme"]["paired_initial_cloud_sha256"]
            new_hash = recovered_manifest.get("paired_initial_cloud_sha256")
            if old_hash != new_hash:
                record["status"] = "FAILED"
                record["exception"] = (
                    "recovered MIDPOINT initial-cloud hash does not match "
                    "the reused PBME artifact"
                )
    except Exception as exc:
        record["status"] = "FAILED"
        record["exception"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    record["end_utc"] = utcnow()
    record["runtime_seconds"] = time.perf_counter() - started
    output_hashes = {}
    for path in output_dir.rglob("*"):
        if path.is_file():
            output_hashes[str(path)] = sha256_file(path)
    record["output_hashes"] = output_hashes
    append_job_record(out, record)
    return record["status"] == "COMPLETE"


def reference_configuration(method: str, P0: float) -> Dict[str, Any]:
    t_final = 2.0 * 2000.0 * 15.0 / abs(P0)
    # TDSE uses the examiner-specified 0.2/0.1/0.05 ladder.  Grid QCLE has a
    # much looser solver-computed RK4 CFL limit (about 4.66 a.u. on the planned
    # finest grid), so 2.0/1.0/0.5 provides a stable, explicit three-level
    # temporal test across the full scattering window without pretending that
    # the TDSE ladder is a QCLE stability requirement.
    base_dt = 0.2 if method == "tdse" else 2.0
    cfg: Dict[str, Any] = {
        "P0": float(P0), "R0": -15.0, "sigma_R": 1.0,
        "dt": base_dt, "n_steps": int(round(t_final / base_dt)),
        "t_final": t_final,
        # TDSE edge probability is nonnegative and supports a 1e-6 gate.
        # Grid-QCLE uses signed Wigner densities, so domain adequacy is tested
        # on absolute physical marginals with a declared 0.1% threshold.
        "edge_mass_tolerance": 1.0e-6 if method == "tdse" else 1.0e-3,
        "negative_momentum_tolerance": 1.0e-6,
    }
    if method == "tdse":
        cfg.update({"n_grid": 2048})
    else:
        # At low momentum the box includes the negative-P reflected branch.
        # At P0=100 the established scattering regime is overwhelmingly
        # forward and a positive box resolves sigma_P=0.5 without reproducing
        # the old invalid configuration whose [-35,35] domain did not even
        # contain the incoming packet.  Absolute edge-mass diagnostics remain
        # mandatory and force a domain revision if this assumption is unsafe.
        low_momentum = P0 <= 20.0
        cfg.update({
            "edge_mass_diagnostic": "absolute_physical_marginals_v2",
            # P0=20 sequence: 48x64, 96x128, 192x256.
            # P0=100 sequence: 48x32, 96x64, 192x128.  The review
            # specification explicitly permits a different verified sequence
            # when the 192x128 -> 768x512 example is incompatible with memory
            # and full-scattering runtime.  The finest grid is also held fixed
            # for the separate time refinement.  Adequacy is decided from the
            # displayed successive differences and edge masses, not assumed.
            "n_R": 48, "n_P": 64 if low_momentum else 32,
            "R_min": -30.0, "R_max": 30.0,
            "P_min": -35.0 if low_momentum else 80.0,
            "P_max": 35.0 if low_momentum else 120.0,
        })
    return cfg


def run_reference_job(out: Path, job: Mapping[str, Any]) -> bool:
    from thesis_closure import qcle_three_level, tdse_three_level

    params = job["parameters"]
    method, mode, P0 = params["method"], params["mode"], float(params["P0"])
    cfg = reference_configuration(method, P0)
    target = Path(job["output_dir"])
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    record: Dict[str, Any] = {
        "job_id": job["job_id"], "kind": "reference",
        "parameters": {**cfg, "method": method, "mode": mode},
        "command": [
            "in-process", "thesis_closure.py",
            f"{method}_three_level", f"refine={mode}",
        ],
        "start_utc": utcnow(), "status": "RUNNING",
        "code_file": f"thesis_closure.py::{method}_three_level",
        "dtype": DTYPE, "device": DEVICE, "retries": 0,
    }
    append_job_record(out, record)
    try:
        series_out = target.with_suffix(".npz") if mode == "time" else None
        if method == "tdse":
            result = tdse_three_level(refine=mode, **{
                key: cfg[key] for key in (
                    "P0", "R0", "sigma_R", "dt", "n_steps", "n_grid"
                )
            }, series_out=series_out)
        else:
            result = qcle_three_level(refine=mode, **{
                key: cfg[key] for key in (
                    "P0", "R0", "sigma_R", "dt", "n_steps", "n_R", "n_P",
                    "R_min", "R_max", "P_min", "P_max",
                )
            }, series_out=series_out)
        result["P0"] = P0
        result["resolved_configuration"] = cfg
        if series_out is not None and series_out.exists():
            result["finest_series_sha256"] = sha256_file(series_out)
        write_json(target, result)
        record["status"] = "COMPLETE"
        record["output_hashes"] = {str(target): sha256_file(target)}
    except Exception as exc:
        record["status"] = "FAILED"
        record["exception"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    record["end_utc"] = utcnow()
    record["runtime_seconds"] = time.perf_counter() - started
    append_job_record(out, record)
    return record["status"] == "COMPLETE"


def execute(args: argparse.Namespace, plan: Mapping[str, Any]) -> int:
    out = args.out.resolve()
    lock = out / "EXECUTION.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(
            f"[execute] refused: another execution lock exists at {lock}. "
            "Inspect the recorded PID before removing a stale lock.",
            file=sys.stderr,
        )
        return 2
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "created_utc": utcnow()}))
    failures = 0
    selected = set(args.execute_kinds)
    selected_reference_methods = set(args.execute_reference_methods)

    def selected_job(job: Mapping[str, Any]) -> bool:
        if job["kind"] not in selected:
            return False
        if job["kind"] == "reference":
            return job["parameters"].get("method") in selected_reference_methods
        return True

    missing = [
        job for job in plan["jobs"]
        if job["status"] == "MISSING" and selected_job(job)
    ]
    def execute_parallel(
        jobs: Sequence[Mapping[str, Any]], workers: int, label: str
    ) -> None:
        nonlocal failures
        if not jobs:
            return
        bounded_workers = min(max(1, int(workers)), len(jobs))
        threads_per_worker = max(
            1, int(os.cpu_count() or 1) // bounded_workers
        )
        print(
            f"[execute] scheduling {len(jobs)} {label} job(s) with "
            f"{bounded_workers} bounded workers and "
            f"{threads_per_worker} numerical thread(s) per worker",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=bounded_workers) as pool:
            futures = {
                pool.submit(
                    run_subprocess_job, out, job, threads_per_worker
                ): job
                for job in jobs
            }
            for index, future in enumerate(as_completed(futures), 1):
                job = futures[future]
                try:
                    ok = bool(future.result())
                except Exception as exc:
                    ok = False
                    append_job_record(out, {
                        "job_id": job["job_id"],
                        "kind": job["kind"],
                        "parameters": job["parameters"],
                        "status": "FAILED",
                        "exception": (
                            "orchestration future failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        "end_utc": utcnow(),
                    })
                if not ok:
                    failures += 1
                    print(
                        f"[execute {index}/{len(jobs)}] FAILED: "
                        f"{job['job_id']}",
                        flush=True,
                    )
                else:
                    print(
                        f"[execute {index}/{len(jobs)}] complete: "
                        f"{job['job_id']}",
                        flush=True,
                    )

    def execute_sequential(jobs: Sequence[Mapping[str, Any]]) -> None:
        nonlocal failures
        for index, job in enumerate(jobs, 1):
            print(
                f"[execute {index}/{len(jobs)}] {job['job_id']} "
                f"({job['kind']})",
                flush=True,
            )
            ok = (
                run_reference_job(out, job)
                if job["kind"] == "reference"
                else run_subprocess_job(out, job)
            )
            if not ok:
                failures += 1
                print(f"[execute] FAILED: {job['job_id']}", flush=True)
            else:
                print(f"[execute] complete: {job['job_id']}", flush=True)

    try:
        manufactured = [job for job in missing if job["kind"] == "manufactured"]
        dynamics = [
            job for job in missing if job["kind"] in ("timestep", "support")
        ]
        references = [job for job in missing if job["kind"] == "reference"]

        if manufactured and args.parallel_workers > 1:
            execute_parallel(
                manufactured, args.parallel_workers, "manufactured-fit"
            )
        else:
            execute_sequential(manufactured)

        if dynamics and args.parallel_dynamics_workers > 1:
            execute_parallel(
                dynamics, args.parallel_dynamics_workers, "dynamics"
            )
        else:
            execute_sequential(dynamics)

        # Reference calculations run in-process and remain sequential so their
        # solver diagnostics and memory use cannot interfere with one another.
        execute_sequential(references)
        refreshed = build_plan(args)
        write_json(out / "FINAL_RUN_MANIFEST.json", refreshed)
        remaining = sum(
            1 for job in refreshed["jobs"]
            if job["status"] == "MISSING" and selected_job(job)
        )
        print(
            f"[execute] subprocess failures={failures}; "
            f"remaining invalid/missing jobs={remaining}",
            flush=True,
        )
        return 0 if failures == 0 and remaining == 0 else 2
    finally:
        if lock.exists():
            lock.unlink()


def t_interval(values: Sequence[float], confidence: float = 0.95
               ) -> Tuple[float, float, float, float, float]:
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return math.nan, math.nan, math.nan, math.nan, math.nan
    mean = float(np.mean(arr))
    if n < 2:
        return mean, math.nan, math.nan, math.nan, math.nan
    sd = float(np.std(arr, ddof=1))
    se = sd / math.sqrt(n)
    try:
        from scipy.stats import t as student_t
        critical = float(student_t.ppf(0.5 + confidence / 2.0, n - 1))
    except Exception:
        critical = 1.96
    return mean, sd, se, mean - critical * se, mean + critical * se


def analyze_manufactured(out: Path) -> Dict[str, Path]:
    complete: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    for path in sorted((out / "manufactured" / "runs").rglob(
        "manufactured_operator_metrics.json"
    )):
        data = read_json(path)
        sources.append({
            "source": str(path),
            "sha256": sha256_file(path),
            "training_cloud_sha256": data["training_cloud_sha256"],
            "off_support_query_sha256": data["metrics"]["off_support"]["query_sha256"],
        })
        for query_type in ("on_support", "off_support"):
            row: Dict[str, Any] = {
                "l2_regularization": data["l2_regularization"],
                "N": data["n_train"], "seed": data["seed"],
                "query_type": query_type,
                "query_count": data["metrics"][query_type]["query_count"],
                "training_cloud_sha256": data["training_cloud_sha256"],
                "query_sha256": data["metrics"][query_type]["query_sha256"],
                "cholesky_jitter": data["cholesky_jitter"],
                "cholesky_adaptive_jitter": data["cholesky_adaptive_jitter"],
                "cholesky_attempts": data["cholesky_attempts"],
                "minimum_eigenvalue_estimate": data["minimum_eigenvalue_estimate"],
                "sigma_n": data["sigma_n"], "sigma_f": data["sigma_f"],
                "lengthscales": data["lengthscales"],
                "input_scaling": data["input_scaling"],
                "hyperparameter_policy": data["hyperparameter_policy"],
                "dtype": data["dtype"], "source_file": str(path),
            }
            for quantity in ("density", "gradient", "operator_Q"):
                metric = data["metrics"][query_type][quantity]
                prefix = "Q" if quantity == "operator_Q" else quantity
                for name in (
                    "relative_l1", "relative_l2", "relative_linf",
                    "mae", "rmse", "linf",
                ):
                    row[f"{prefix}_{name}"] = metric[name]
                row[f"{prefix}_denominator_floor"] = metric["denominator_floor"]
                row[f"{prefix}_denominator_floor_used"] = metric["denominator_floor_used"]
            complete.append(row)

    complete_path = write_csv(
        out / "manufactured" / "manufactured_complete.csv", complete
    )
    summary: List[Dict[str, Any]] = []
    metric_columns = [
        key for key in complete[0]
        if any(key.endswith(suffix) for suffix in (
            "_relative_l1", "_relative_l2", "_relative_linf",
            "_mae", "_rmse", "_linf",
        ))
    ] if complete else []
    baseline_rows = {
        (int(row["N"]), int(row["seed"]), str(row["query_type"])): row
        for row in complete
        if equal_number(row["l2_regularization"], 1e-6)
    }
    groups: Dict[Tuple[float, int, str], List[Dict[str, Any]]] = {}
    for row in complete:
        groups.setdefault(
            (float(row["l2_regularization"]), int(row["N"]), str(row["query_type"])),
            [],
        ).append(row)
    for (l2, n_train, query), rows in sorted(groups.items()):
        for metric in metric_columns:
            values = [float(row[metric]) for row in rows]
            mean, sd, se, lo, hi = t_interval(values)
            paired_differences: List[float] = []
            paired_clouds_verified = True
            for row in rows:
                control = baseline_rows.get(
                    (int(row["N"]), int(row["seed"]), str(row["query_type"]))
                )
                if control is None:
                    raise RuntimeError(
                        "Manufactured regularization comparison is missing its "
                        f"l2=1e-6 control for N={row['N']}, seed={row['seed']}, "
                        f"query={row['query_type']}"
                    )
                paired_clouds_verified &= (
                    row["training_cloud_sha256"]
                    == control["training_cloud_sha256"]
                    and row["query_sha256"] == control["query_sha256"]
                )
                paired_differences.append(
                    float(row[metric]) - float(control[metric])
                )
            dmean, dsd, dse, dlo, dhi = t_interval(paired_differences)
            summary.append({
                "l2_regularization": l2, "N": n_train, "query_type": query,
                "metric": metric, "n_seeds": len(values),
                "seeds": [int(row["seed"]) for row in rows],
                "mean": mean, "sample_sd": sd, "standard_error": se,
                "ci95_low": lo, "ci95_high": hi,
                "confidence_interval_method": "two-sided Student t, df=n-1",
                "baseline_l2_regularization": 1e-6,
                "mean_paired_difference_from_baseline": dmean,
                "paired_difference_sample_sd": dsd,
                "paired_difference_standard_error": dse,
                "paired_difference_ci95_low": dlo,
                "paired_difference_ci95_high": dhi,
                "paired_training_and_query_clouds_verified": (
                    paired_clouds_verified
                ),
                "source_csv": str(complete_path),
            })
    summary_path = write_csv(
        out / "manufactured" / "manufactured_summary.csv", summary
    )

    refinement_rows: List[Dict[str, Any]] = []
    support_levels = sorted({int(row["N"]) for row in complete})
    for l2 in sorted({float(row["l2_regularization"]) for row in complete}):
        for query_type in ("on_support", "off_support"):
            for quantity in ("density", "gradient", "Q"):
                for metric_name in (
                    "relative_l1", "relative_l2", "relative_linf"
                ):
                    metric = f"{quantity}_{metric_name}"
                    means: List[float] = []
                    for level in support_levels:
                        values = [
                            float(row[metric]) for row in complete
                            if equal_number(row["l2_regularization"], l2)
                            and int(row["N"]) == level
                            and row["query_type"] == query_type
                        ]
                        means.append(float(np.mean(values)))
                    changes = [
                        100.0 * (b - a) / abs(a) if a != 0.0 else math.nan
                        for a, b in zip(means, means[1:])
                    ]
                    monotone = all(b < a for a, b in zip(means, means[1:]))
                    endpoint_improved = means[-1] < means[0]
                    refinement_rows.append({
                        "l2_regularization": l2,
                        "query_type": query_type,
                        "quantity": quantity,
                        "metric": metric_name,
                        **{
                            f"N{level}_seed_mean": value
                            for level, value in zip(support_levels, means)
                        },
                        **{
                            f"percent_change_N{left}_to_N{right}": value
                            for left, right, value in zip(
                                support_levels, support_levels[1:], changes
                            )
                        },
                        "Nmax_better_than_Nmin": endpoint_improved,
                        "monotone_decrease": monotone,
                        "refinement_verdict": (
                            "MONOTONE_DECREASE_OBSERVED"
                            if monotone else "NO_MONOTONE_DECREASE_OBSERVED"
                        ),
                        "criterion": (
                            "descriptive independent-cloud enlargement check: "
                            "all four seed-mean errors decrease strictly with N; "
                            "clouds are nonnested, so this is not a deterministic "
                            "support-convergence test and no absolute accuracy "
                            "threshold was declared"
                        ),
                    })
    refinement_path = write_csv(
        out / "manufactured" / "manufactured_refinement_verdicts.csv",
        refinement_rows,
    )

    base = {
        (int(row["N"]), int(row["seed"]), str(row["query_type"])): row
        for row in complete if equal_number(row["l2_regularization"], 1e-6)
    }
    comparisons: List[Dict[str, Any]] = []
    for row in complete:
        if equal_number(row["l2_regularization"], 1e-6):
            continue
        key = (int(row["N"]), int(row["seed"]), str(row["query_type"]))
        control = base.get(key)
        if control is None:
            continue
        for metric in metric_columns:
            comparisons.append({
                "l2_regularization": row["l2_regularization"],
                "baseline_l2": 1e-6, "N": row["N"], "seed": row["seed"],
                "query_type": row["query_type"], "metric": metric,
                "value": row[metric], "baseline_value": control[metric],
                "paired_difference": float(row[metric]) - float(control[metric]),
                "paired_percent_change": (
                    100.0 * (float(row[metric]) - float(control[metric]))
                    / abs(float(control[metric]))
                    if float(control[metric]) != 0.0 else math.nan
                ),
                "paired_training_cloud": (
                    row["training_cloud_sha256"] == control["training_cloud_sha256"]
                ),
                "paired_query_cloud": row["query_sha256"] == control["query_sha256"],
                "source_csv": str(complete_path),
            })
    comparison_path = write_csv(
        out / "manufactured" / "manufactured_policy_comparison.csv", comparisons
    )
    sampling_geometry_path = write_json(
        out / "manufactured" / "manufactured_sampling_geometry.json",
        {
            "dimension": 6,
            "coordinate_order": ["R", "P", "r0", "r1", "p0", "p1"],
            "training_and_query_geometry": (
                "fully dimensional independent Gaussian points"
            ),
            "independent_coordinate_distributions": {
                "R": {"family": "Normal", "mean": 0.0, "standard_deviation": 1.2},
                "P": {"family": "Normal", "mean": 8.0, "standard_deviation": 0.7},
                "r0": {"family": "Normal", "mean": 0.0, "standard_deviation": math.sqrt(0.5)},
                "r1": {"family": "Normal", "mean": 0.0, "standard_deviation": math.sqrt(0.5)},
                "p0": {"family": "Normal", "mean": 0.0, "standard_deviation": math.sqrt(0.5)},
                "p1": {"family": "Normal", "mean": 0.0, "standard_deviation": math.sqrt(0.5)},
            },
            "mapping_coordinates_independent": True,
            "focused_mapping_shell": False,
            "scope_limitation": (
                "This fully dimensional test is more informative about ambient "
                "derivatives than the focused production cloud. It does not "
                "reproduce focused-MMST normal-derivative nonidentifiability, "
                "and its 2--3 percent operator errors are not quantitative "
                "estimates of production off-manifold error."
            ),
            "implementation": "ReviewerValidation.py::manufactured_test",
            "implementation_sha256": sha256_file(REPO / "ReviewerValidation.py"),
        },
    )
    write_json(out / "manufactured" / "manufactured_manifest.json", {
        "created_utc": utcnow(), "expected_complete_rows": 72,
        "actual_complete_rows": len(complete),
        "paired_design": "same training and off-support clouds at fixed N and seed",
        "on_support_contract": "all N unique training points",
        "off_support_contract": "1000 independent query points",
        "sampling_geometry_contract": read_json(sampling_geometry_path),
        "denominator_floor": 1e-30,
        "sources": sources,
        "outputs": {
            str(path): sha256_file(path)
            for path in (
                complete_path, summary_path, comparison_path, refinement_path,
                sampling_geometry_path,
            )
        },
    })
    return {
        "complete": complete_path, "summary": summary_path,
        "comparison": comparison_path, "refinement": refinement_path,
        "sampling_geometry": sampling_geometry_path,
    }


def analyze_mint_controls(out: Path) -> Dict[str, Path]:
    """Regenerate deterministic implementation controls used by the MInt tests."""
    from Mint import PBMEMIntDynamics

    dynamics = PBMEMIntDynamics()
    z0 = np.array([-1.3, 18.0, 0.9, -0.4, 0.3, 0.6], dtype=float)
    dt = 0.5
    n_steps = 200
    fd_epsilon = 1.0e-7
    jacobian = np.asarray(
        dynamics.compute_step_jacobian(z0, dt, eps=fd_epsilon), float
    )
    trajectory = np.asarray(dynamics.propagate(z0, dt, n_steps), float)
    radii = np.asarray(dynamics.mapping_radius_sq(trajectory), float)
    energies = np.asarray(dynamics.energy(trajectory), float)
    diagnostics = (
        (
            "one-step symplectic residual",
            float(dynamics.symplectic_defect(jacobian)),
            "Frobenius norm of J^T Omega J - Omega",
            1.0e-6,
        ),
        (
            "one-step round-trip residual",
            float(dynamics.time_reversal_error(z0, dt)),
            "Euclidean norm of Psi_-dt(Psi_dt(z0)) - z0",
            1.0e-10,
        ),
        (
            "mapping-radius drift",
            float(np.max(np.abs(radii - radii[0]))),
            "maximum absolute drift over 200 steps",
            1.0e-11,
        ),
        (
            "trajectory-energy drift",
            float(abs(energies[-1] - energies[0])),
            "absolute PBME trajectory-energy drift after 200 steps",
            1.0e-7,
        ),
    )
    rows = [
        {
            "diagnostic": name,
            "value": value,
            "aggregation": aggregation,
            "tolerance": tolerance,
            "status": "PASS" if value < tolerance else "FAIL",
            "dt": dt,
            "n_steps": n_steps,
            "finite_difference_epsilon": fd_epsilon,
            "initial_state": z0.tolist(),
            "implementation": "Mint.py::PBMEMIntDynamics",
        }
        for name, value, aggregation, tolerance in diagnostics
    ]
    csv_path = write_csv(
        out / "implementation_controls" / "mint_implementation_controls.csv",
        rows,
    )
    manifest_path = write_json(
        out / "implementation_controls" / "mint_implementation_controls_manifest.json",
        {
            "created_utc": utcnow(),
            "purpose": "deterministic numerical implementation control; not a production simulation",
            "coordinate_order": ["R", "P", "r0", "r1", "p0", "p1"],
            "initial_state": z0.tolist(),
            "dt": dt,
            "n_steps": n_steps,
            "finite_difference_epsilon": fd_epsilon,
            "Mint.py_sha256": sha256_file(REPO / "Mint.py"),
            "test_math_expressions.py_sha256": sha256_file(
                REPO / "test_math_expressions.py"
            ),
            "output": {str(csv_path): sha256_file(csv_path)},
        },
    )
    return {"controls": csv_path, "manifest": manifest_path}


def snapshot_prefix(files: Iterable[str], which: str) -> str:
    steps = sorted({
        int(name.split("_")[1])
        for name in files
        if name.startswith("snap_") and name.endswith("_Z")
    })
    if not steps:
        raise KeyError("no stored cloud snapshots")
    step = steps[0] if which == "initial" else steps[-1]
    return f"snap_{step:06d}"


def quantile_fields(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, float)
    out = {"minimum": float(np.min(arr)), "maximum": float(np.max(arr))}
    for q in T_QUANTILES:
        out[f"q{q:g}"] = float(np.quantile(arr, q))
    return out


def safe_ratio(value: float, base: float) -> float:
    return float(value - base)


def analyze_tail(out: Path, args: argparse.Namespace) -> Dict[str, Path]:
    from Dynamics import _support_mapping_observables
    from Mint import PBMEMIntDynamics, PBMEMIntParams
    from Models import TullyModel, TullyParams

    dynamics = PBMEMIntDynamics(
        TullyModel(TullyParams.defaults("dual")), PBMEMIntParams()
    )
    distributions: List[Dict[str, Any]] = []
    sweep: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []

    for P0 in args.P0:
        for seed in args.dynamics_seeds:
            expected = {
                "P0": P0, "seed": seed, "N": 1000, "dt": 0.25,
                "t_final_resolved": 2.0 * 2000.0 * 15.0 / abs(P0),
                "l2_regularization": 0.05,
            }
            run_dir, reason = resolve_run(out, expected)
            if run_dir is None:
                raise RuntimeError(
                    f"tail analysis requires P0={P0}, seed={seed}: {reason}"
                )
            for method in ("pbme", "midpoint"):
                path = run_dir / f"{method}.npz"
                with np.load(path) as data:
                    initial = snapshot_prefix(data.files, "initial")
                    final = snapshot_prefix(data.files, "final")
                    required = (
                        f"{initial}_Z", f"{initial}_y", f"{initial}_weight",
                        f"{initial}_target_density",
                        f"{initial}_proposal_density",
                        f"{initial}_geometric_measure", f"{final}_Z",
                        f"{final}_y", f"{final}_weight",
                        f"{final}_geometric_measure", f"{final}_alpha",
                    )
                    missing = [key for key in required if key not in data.files]
                    if missing:
                        raise RuntimeError(f"{path}: missing tail fields {missing}")
                    Z0 = np.asarray(data[f"{initial}_Z"], float)
                    Z = np.asarray(data[f"{final}_Z"], float)
                    saved_initial_weight = np.asarray(
                        data[f"{initial}_weight"], float
                    )
                    saved_live_weight = np.asarray(
                        data[f"{final}_weight"], float
                    )
                    proposal_density = np.asarray(
                        data[f"{initial}_proposal_density"], float
                    )
                    y0 = np.asarray(data[f"{initial}_target_density"], float)
                    y_eff = np.asarray(data[f"{final}_y"], float)
                    omega = np.asarray(data[f"{final}_geometric_measure"], float)
                    alpha = np.asarray(data[f"{final}_alpha"], float)
                    # Snapshot.y is the exact effective label fitted by the
                    # surrogate.  Dividing the final effective label by the
                    # separately saved initial target density reconstructs
                    # y_i(t)/y_i^0 without assuming that Snapshot.weight has
                    # the same semantic role for PBME and MIDPOINT.
                    ratio = np.divide(
                        y_eff, y0, out=np.full_like(y_eff, np.nan),
                        where=np.abs(y0) > np.finfo(float).tiny,
                    )
                    # The saved correction multiplier is the documented
                    # y_i(t)/y_i^0 diagnostic.  y_eff is the propagated density
                    # label actually used by cloud Riemann sums.
                    if not all(np.all(np.isfinite(a)) for a in (
                        Z0, Z, saved_initial_weight, saved_live_weight,
                        proposal_density, ratio, y0, y_eff, omega, alpha,
                    )):
                        raise RuntimeError(f"{path}: nonfinite tail snapshot")
                    abs_y0 = np.abs(y0)
                    q = quantile_fields(abs_y0)
                    distributions.append({
                        "P0": P0, "method": method.upper(), "seed": seed,
                        "N": len(y0), **q,
                        "initial_label_sha256": sha256_array(y0),
                        "initial_cloud_sha256": sha256_array(Z0),
                        "proposal_density_sha256": sha256_array(proposal_density),
                        "geometric_measure_sha256": sha256_array(omega),
                        "saved_initial_weight_sha256": sha256_array(
                            saved_initial_weight
                        ),
                        "saved_live_weight_sha256": sha256_array(
                            saved_live_weight
                        ),
                        "source_file": str(path),
                        "initial_snapshot": initial, "final_snapshot": final,
                    })

                    mapping = _support_mapping_observables(Z, hbar=1.0)
                    energy_point = np.asarray(dynamics.energy(Z), float)
                    physical_weight = omega * y_eff
                    abs_physical_initial = np.abs(omega * y0)
                    abs_mass_denom = float(np.sum(abs_physical_initial))
                    base_values: Dict[str, float] = {}
                    for eta in ETA_LEVELS:
                        threshold = float(eta * np.max(abs_y0))
                        mask = abs_y0 > threshold
                        excluded = ~mask
                        retained_ratio = np.abs(ratio[mask])
                        contribution = physical_weight * mask
                        denom_sq = float(np.sum(contribution * contribution))
                        signed_ess = (
                            float(np.sum(contribution) ** 2 / denom_sq)
                            if denom_sq > 0 else math.nan
                        )
                        absolute_ess = (
                            float(np.sum(np.abs(contribution)) ** 2 / denom_sq)
                            if denom_sq > 0 else math.nan
                        )
                        cloud_P0 = float(np.dot(contribution, mapping["P0"]))
                        cloud_P1 = float(np.dot(contribution, mapping["P1"]))
                        raw_norm = float(np.sum(contribution))
                        raw_energy = float(np.dot(contribution, energy_point))
                        # Match Dynamics._weighted_support_diagnostics exactly:
                        # cloud_* are raw Riemann sums and lw_* are divided by
                        # the live raw cloud mass only when it is not near zero.
                        live_denominator = (
                            raw_norm if abs(raw_norm) > 1.0e-15 else 1.0
                        )
                        live_P0 = float(cloud_P0 / live_denominator)
                        live_P1 = float(cloud_P1 / live_denominator)
                        if eta == 0.0:
                            base_values = {
                                "live_P0": live_P0, "live_P1": live_P1,
                                "cloud_P0": cloud_P0, "cloud_P1": cloud_P1,
                                "raw_norm": raw_norm, "raw_energy": raw_energy,
                            }
                        ratio_q = (
                            quantile_fields(retained_ratio)
                            if retained_ratio.size else {
                                "minimum": math.nan, "maximum": math.nan,
                                **{f"q{x:g}": math.nan for x in T_QUANTILES},
                            }
                        )
                        row = {
                            "P0": P0, "method": method.upper(), "seed": seed,
                            "eta": eta, "threshold": threshold,
                            "included_count": int(np.sum(mask)),
                            "excluded_count": int(np.sum(excluded)),
                            "included_fraction": float(np.mean(mask)),
                            "excluded_fraction": float(np.mean(excluded)),
                            "excluded_absolute_physical_mass_fraction": (
                                float(np.sum(abs_physical_initial[excluded]) / abs_mass_denom)
                                if abs_mass_denom > 0 else math.nan
                            ),
                            "excluded_signed_mass": float(np.sum((omega * y0)[excluded])),
                            "minimum_retained_abs_y0": (
                                float(np.min(abs_y0[mask])) if np.any(mask) else math.nan
                            ),
                            "ratio_abs_max": ratio_q["maximum"],
                            **{
                                f"ratio_abs_q{qv:g}": ratio_q[f"q{qv:g}"]
                                for qv in T_QUANTILES
                            },
                            "signed_ESS": signed_ess, "absolute_ESS": absolute_ess,
                            "live_weight_P0": live_P0, "live_weight_P1": live_P1,
                            "cloud_weighted_P0": cloud_P0,
                            "cloud_weighted_P1": cloud_P1,
                            "raw_normalization": raw_norm,
                            "raw_energy": raw_energy,
                            "live_weight_denominator": live_denominator,
                            "near_zero_signed_denominator": bool(
                                abs(raw_norm) <= 1.0e-15
                            ),
                            "point_inclusion_mask_sha256": sha256_array(mask),
                            "initial_label_sha256": sha256_array(y0),
                            "correction_multiplier_sha256": sha256_array(ratio),
                            "proposal_density_sha256": sha256_array(
                                proposal_density
                            ),
                            "geometric_measure_sha256": sha256_array(omega),
                            "saved_live_weight_sha256": sha256_array(
                                saved_live_weight
                            ),
                            "cloud_coefficient_sha256": sha256_array(alpha),
                            "source_file": str(path),
                            "initial_snapshot": initial, "final_snapshot": final,
                        }
                        for key, base_key in (
                            ("live_weight_P0", "live_P0"),
                            ("live_weight_P1", "live_P1"),
                            ("cloud_weighted_P0", "cloud_P0"),
                            ("cloud_weighted_P1", "cloud_P1"),
                            ("raw_normalization", "raw_norm"),
                            ("raw_energy", "raw_energy"),
                        ):
                            row[f"{key}_change_from_eta0"] = safe_ratio(
                                float(row[key]), float(base_values.get(base_key, row[key]))
                            )
                        sweep.append(row)
                sources.append({
                    "source": str(path), "sha256": sha256_file(path),
                    "run_manifest": str(run_dir / "run_manifest.json"),
                    "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
                })

    distribution_path = write_csv(
        out / "tail_sensitivity" / "y0_distribution.csv", distributions
    )
    sweep_path = write_csv(
        out / "tail_sensitivity" / "threshold_sweep.csv", sweep
    )
    summaries: List[Dict[str, Any]] = []
    cases: Dict[Tuple[float, str, int], List[Dict[str, Any]]] = {}
    for row in sweep:
        cases.setdefault(
            (float(row["P0"]), str(row["method"]), int(row["seed"])), []
        ).append(row)
    for (P0, method, seed), rows in sorted(cases.items()):
        negligible_nontrivial = [
            row for row in rows
            if int(row["excluded_count"]) > 0
            and float(row["excluded_absolute_physical_mass_fraction"]) <= 1e-6
        ]
        changes = [
            abs(float(row["raw_normalization_change_from_eta0"]))
            for row in negligible_nontrivial
        ] + [
            abs(float(row["cloud_weighted_P0_change_from_eta0"]))
            for row in negligible_nontrivial
        ] + [
            abs(float(row["cloud_weighted_P1_change_from_eta0"]))
            for row in negligible_nontrivial
        ]
        base = next(row for row in rows if float(row["eta"]) == 0.0)
        scale = max(
            1.0,
            abs(float(base["raw_normalization"])),
            abs(float(base["cloud_weighted_P0"])),
            abs(float(base["cloud_weighted_P1"])),
        )
        material = bool(changes and max(changes) > 0.01 * scale)
        stable = [row for row in rows if all(
            abs(float(row[key])) <= 0.01 * scale
            for key in (
                "raw_normalization_change_from_eta0",
                "cloud_weighted_P0_change_from_eta0",
                "cloud_weighted_P1_change_from_eta0",
            )
        )]
        first_excluding = next(
            (row for row in sorted(rows, key=lambda item: float(item["eta"]))
             if int(row["excluded_count"]) > 0),
            None,
        )
        if material:
            verdict = "TAIL_SENSITIVE"
        elif not negligible_nontrivial:
            # With focused sampling omega_i*y_i^0 is uniform here, so deleting
            # even one of N=1000 points removes 10^-3 of the initial absolute
            # physical mass.  A no-deletion plateau is tautological and must
            # not be presented as an empirical stability result.
            verdict = "NO_NONTRIVIAL_NEGLIGIBLE_MASS_THRESHOLD"
        else:
            verdict = "STABLE_WITHIN_DECLARED_RULE"
        summaries.append({
            "P0": P0, "method": method, "seed": seed,
            "negligible_mass_rule": "excluded absolute physical mass fraction <= 1e-6",
            "material_change_rule": "absolute change > 1% of max(1, eta=0 principal scales)",
            "tail_sensitive": material,
            "verdict": verdict,
            "nontrivial_negligible_threshold_count": len(
                negligible_nontrivial
            ),
            "maximum_change_at_negligible_mass": max(changes) if changes else math.nan,
            "stable_eta_min": min((r["eta"] for r in stable), default=math.nan),
            "stable_eta_max": max((r["eta"] for r in stable), default=math.nan),
            "first_excluding_eta": (
                first_excluding["eta"] if first_excluding else math.nan
            ),
            "first_excluding_absolute_mass_fraction": (
                first_excluding["excluded_absolute_physical_mass_fraction"]
                if first_excluding else math.nan
            ),
            "eta0_ratio_abs_max": base["ratio_abs_max"],
            "eta0_signed_ESS": base["signed_ESS"],
            "eta0_absolute_ESS": base["absolute_ESS"],
            "eta0_raw_normalization": base["raw_normalization"],
            "source_csv": str(sweep_path),
        })
    summary_path = write_csv(
        out / "tail_sensitivity" / "tail_summary.csv", summaries
    )
    write_json(out / "tail_sensitivity" / "tail_manifest.json", {
        "created_utc": utcnow(),
        "estimator_contract": {
            "ratio": (
                "saved final effective density label divided by the separately "
                "saved initial target density, y_i(t)/y_i^0"
            ),
            "cloud_measure": "saved frozen omega_i times saved effective label y_i(t)",
            "threshold": "post-processing only; propagated dynamics unchanged",
            "live_weight_population": (
                "cloud_weighted population divided by raw normalization when "
                "|raw normalization| > 1e-15; otherwise production fallback "
                "denominator 1.0"
            ),
            "cloud_weighted_population": "sum mask_i omega_i y_i(t) mapping_population_i",
            "raw_normalization": "sum mask_i omega_i y_i(t), not self-normalized",
            "raw_energy": "sum mask_i omega_i y_i(t) H_i, not self-normalized",
        },
        "eta_levels": list(ETA_LEVELS),
        "sources": sources,
        "outputs": {
            str(path): sha256_file(path)
            for path in (distribution_path, sweep_path, summary_path)
        },
    })
    return {
        "distribution": distribution_path, "sweep": sweep_path,
        "summary": summary_path,
    }


def load_method_series(run_dir: Path, method: str) -> Dict[str, np.ndarray]:
    from Compare_gp_se_qcle import load_collector_run
    result = load_collector_run(str(run_dir / f"{method.lower()}.npz"))
    return {key: np.asarray(value, float) for key, value in result.items()}


def common_time_l2(t_ref: np.ndarray, a: np.ndarray,
                   t_other: np.ndarray, b: np.ndarray) -> float:
    t_ref = np.asarray(t_ref, float)
    a = np.asarray(a, float)
    t_other = np.asarray(t_other, float)
    b = np.asarray(b, float)
    lo = max(float(t_ref[0]), float(t_other[0]))
    hi = min(float(t_ref[-1]), float(t_other[-1]))
    common = t_ref[(t_ref >= lo) & (t_ref <= hi)]
    if common.size < 2 or hi <= lo:
        return math.nan
    ai = np.interp(common, t_ref, a)
    bi = np.interp(common, t_other, b)
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(np.sqrt(max(trap((ai - bi) ** 2, common) / (hi - lo), 0.0)))


def analyze_timestep(out: Path, args: argparse.Namespace) -> Dict[str, Path]:
    runs: Dict[Tuple[float, int, float], Path] = {}
    sources: List[Dict[str, Any]] = []
    for P0 in args.P0:
        for seed in args.dynamics_seeds:
            for dt in args.dt_levels:
                expected = {
                    "P0": P0, "seed": seed, "N": 1000, "dt": dt,
                    "t_final_resolved": 2.0 * 2000.0 * 15.0 / abs(P0),
                    "l2_regularization": 0.05,
                }
                run_dir, reason = resolve_run(out, expected)
                if run_dir is None:
                    raise RuntimeError(
                        f"missing required timestep run P0={P0}, seed={seed}, "
                        f"dt={dt}: {reason}"
                    )
                runs[(float(P0), int(seed), float(dt))] = run_dir
                manifest = run_dir / "run_manifest.json"
                sources.append({
                    "P0": P0, "seed": seed, "dt": dt,
                    "run_directory": str(run_dir),
                    "manifest": str(manifest),
                    "manifest_sha256": sha256_file(manifest),
                })

    cache: Dict[Tuple[float, int, float, str], Dict[str, np.ndarray]] = {}
    for key, run_dir in runs.items():
        for method in ("pbme", "midpoint"):
            cache[(*key, method)] = load_method_series(run_dir, method)

    dts = sorted((float(x) for x in args.dt_levels), reverse=True)
    if len(dts) != 3:
        raise ValueError("time-step analysis requires exactly three dt levels")
    dt1, dt2, dt3 = dts

    # Retain the raw finest-level cross-seed observable spread only as a
    # descriptive cloud-variability diagnostic.  It is not an uncertainty
    # estimate for the paired refinement differences and is never used as an
    # order gate.
    raw_seed_spreads: Dict[Tuple[float, str, str, int], float] = {}
    pooled_raw_spreads: Dict[Tuple[float, str, str], float] = {}
    for P0 in map(float, args.P0):
        for method in ("pbme", "midpoint"):
            for observable in OBSERVABLES:
                pair_values: List[float] = []
                per_seed: Dict[int, List[float]] = {
                    int(seed): [] for seed in args.dynamics_seeds
                }
                for i, seed_a in enumerate(args.dynamics_seeds):
                    a = cache[(P0, int(seed_a), dt3, method)]
                    for seed_b in args.dynamics_seeds[i + 1:]:
                        b = cache[(P0, int(seed_b), dt3, method)]
                        value = common_time_l2(
                            a["t"], a[observable], b["t"], b[observable]
                        )
                        pair_values.append(value)
                        per_seed[int(seed_a)].append(value)
                        per_seed[int(seed_b)].append(value)
                pooled = float(np.sqrt(np.mean(np.square(pair_values))))
                pooled_raw_spreads[(P0, method, observable)] = pooled
                for seed, values in per_seed.items():
                    raw_seed_spreads[(P0, method, observable, seed)] = float(
                        np.sqrt(np.mean(np.square(values)))
                    )

    # The physical gate is evaluated on the three endpoint states for the
    # same method, momentum and seed.  The reported population estimators must
    # lie in [0,1], their norm/trace must be unity within the declared
    # tolerance, energy must be finite, and signed central second moments must
    # be nonnegative.  A case-level failure prevents interpreting any
    # non-noise-limited observable order for that seed.
    physical_by_case: Dict[Tuple[float, str, int], Tuple[bool, str]] = {}
    physical_tol = 1.0e-10
    norm_tol = 1.0e-8
    for P0 in map(float, args.P0):
        for method in ("pbme", "midpoint"):
            for seed in map(int, args.dynamics_seeds):
                failures: List[str] = []
                for dt in (dt1, dt2, dt3):
                    item = cache[(P0, seed, dt, method)]
                    endpoints = {
                        observable: float(item[observable][-1])
                        for observable in OBSERVABLES
                    }
                    if not all(np.isfinite(value) for value in endpoints.values()):
                        failures.append(f"dt={dt:g}: nonfinite endpoint")
                        continue
                    for population in ("P0", "P1"):
                        value = endpoints[population]
                        if not (-physical_tol <= value <= 1.0 + physical_tol):
                            failures.append(
                                f"dt={dt:g}: {population}={value:.8g} outside [0,1]"
                            )
                    if abs(endpoints["trace"] - 1.0) > norm_tol:
                        failures.append(
                            f"dt={dt:g}: norm/trace={endpoints['trace']:.8g} "
                            f"differs from unity by more than {norm_tol:g}"
                        )
                    if not np.isfinite(endpoints["energy"]):
                        failures.append(f"dt={dt:g}: nonfinite energy")
                    for moment in ("R_var", "P_var"):
                        value = endpoints[moment]
                        if value < -physical_tol:
                            failures.append(
                                f"dt={dt:g}: {moment}={value:.8g} is negative"
                            )
                physical_by_case[(P0, method, seed)] = (
                    not failures,
                    "passed endpoint population, norm, energy and signed-central-moment checks"
                    if not failures else "; ".join(failures),
                )

    rows: List[Dict[str, Any]] = []
    for P0 in map(float, args.P0):
        for method in ("pbme", "midpoint"):
            for seed in map(int, args.dynamics_seeds):
                series = [
                    cache[(P0, seed, dt, method)] for dt in (dt1, dt2, dt3)
                ]
                for observable in OBSERVABLES:
                    t_common = series[0]["t"]
                    hi = min(float(item["t"][-1]) for item in series)
                    t_common = t_common[t_common <= hi]
                    if t_common.size < 2:
                        verdict, reason = "MISSING_RUN", "no common saved times"
                        values = [math.nan] * 3
                        D12 = D23 = tau = p = math.nan
                    else:
                        aligned = [
                            np.interp(t_common, item["t"], item[observable])
                            for item in series
                        ]
                        values = [float(item[observable][-1]) for item in series]
                        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
                        duration = float(t_common[-1] - t_common[0])
                        D12 = float(np.sqrt(max(
                            trap((aligned[0] - aligned[1]) ** 2, t_common) / duration,
                            0.0,
                        )))
                        D23 = float(np.sqrt(max(
                            trap((aligned[1] - aligned[2]) ** 2, t_common) / duration,
                            0.0,
                        )))
                        rms_scales = [
                            float(np.sqrt(max(
                                trap(item * item, t_common) / duration, 0.0
                            ))) for item in aligned
                        ]
                        tau = numerical_noise_threshold(rms_scales)
                        raw_seed_spread = raw_seed_spreads[
                            (P0, method, observable, seed)
                        ]
                        pooled_raw = pooled_raw_spreads[(P0, method, observable)]
                        physically_admissible, physical_reason = physical_by_case[
                            (P0, method, seed)
                        ]
                        if not all(np.isfinite(x) for x in (
                            *values, D12, D23, tau, raw_seed_spread, pooled_raw
                        )):
                            verdict, reason, p = (
                                "NONFINITE_RUN", "nonfinite aligned value", math.nan
                            )
                        elif D12 <= tau or D23 <= tau:
                            verdict, reason, p = (
                                "REJECT_NUMERICAL_NOISE",
                                "roundoff- or saturation-limited; order not interpreted",
                                math.nan,
                            )
                        elif not physically_admissible:
                            verdict, reason, p = (
                                "REJECT_PHYSICAL_INADMISSIBILITY",
                                "one or more levels physically inadmissible; temporal order not interpreted",
                                math.nan,
                            )
                        else:
                            p = float(np.log2(D12 / D23))
                            if p > 0:
                                verdict, reason = (
                                    "COMPUTED_POSITIVE",
                                    "three finite and physically admissible levels; paired finer difference contracts",
                                )
                            else:
                                verdict, reason = (
                                    "COMPUTED_ZERO_OR_NEGATIVE",
                                    "three finite and physically admissible levels but paired finer difference does not decrease",
                                )
                    raw_seed_spread = raw_seed_spreads.get(
                        (P0, method, observable, seed), math.nan
                    )
                    pooled_raw = pooled_raw_spreads.get(
                        (P0, method, observable), math.nan
                    )
                    physically_admissible, physical_reason = physical_by_case.get(
                        (P0, method, seed), (False, "case unavailable")
                    )
                    run_dirs = [runs[(P0, seed, dt)] for dt in (dt1, dt2, dt3)]
                    rows.append({
                        "method": method.upper(), "P0": P0, "seed": seed,
                        "observable": observable,
                        "dt1": dt1, "dt2": dt2, "dt3": dt3,
                        "value1": values[0], "value2": values[1], "value3": values[2],
                        "D12": D12, "D23": D23,
                        "D12_over_D23": (
                            D12 / D23 if np.isfinite(D23) and D23 > 0.0 else math.nan
                        ),
                        "paired_difference_D12_minus_D23": D12 - D23,
                        "paired_contraction": bool(D23 < D12),
                        "raw_observable_seed_spread": raw_seed_spread,
                        "pooled_raw_observable_seed_spread": pooled_raw,
                        "raw_seed_spread_role": (
                            "descriptive cloud-to-cloud variability only; not an order or uncertainty gate"
                        ),
                        "roundoff_threshold": tau,
                        "physically_admissible_case": physically_admissible,
                        "physical_admissibility_reason": physical_reason,
                        "p_observed": p, "verdict": verdict, "reason": reason,
                        "alignment": "linear interpolation to common coarse saved times; no extrapolation",
                        "run1_manifest": str(run_dirs[0] / "run_manifest.json"),
                        "run2_manifest": str(run_dirs[1] / "run_manifest.json"),
                        "run3_manifest": str(run_dirs[2] / "run_manifest.json"),
                    })

    run_path = write_csv(out / "timestep" / "timestep_run_by_run.csv", rows)
    summary: List[Dict[str, Any]] = []
    paired_summary: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, float, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["method"]), float(row["P0"]), str(row["observable"])), []
        ).append(row)
    for (method, P0, observable), items in sorted(groups.items()):
        for metric in (
            "value1", "value2", "value3", "D12", "D23",
            "raw_observable_seed_spread",
        ):
            values = [float(item[metric]) for item in items]
            mean, sd, se, lo, hi = t_interval(values)
            summary.append({
                "method": method, "P0": P0, "observable": observable,
                "metric": metric, "n_seeds": len(values),
                "mean": mean, "sample_sd": sd, "standard_error": se,
                "ci95_low": lo, "ci95_high": hi,
                "confidence_interval_method": "two-sided Student t, df=3",
                "source_csv": str(run_path),
            })
        deltas = [float(item["paired_difference_D12_minus_D23"]) for item in items]
        ratios = [float(item["D12_over_D23"]) for item in items]
        dmean, dsd, dse, dlo, dhi = t_interval(deltas)
        noise_limited = [
            item for item in items if item["verdict"] == "REJECT_NUMERICAL_NOISE"
        ]
        physically_bad = [
            item for item in items
            if item["verdict"] == "REJECT_PHYSICAL_INADMISSIBILITY"
        ]
        finite_bad = [item for item in items if item["verdict"] == "NONFINITE_RUN"]
        if noise_limited:
            final_verdict = "REJECT_NUMERICAL_NOISE"
            interpretation = "one or more seeds are numerical-floor limited; temporal order not interpreted"
        elif finite_bad:
            final_verdict = "REJECT_NONFINITE"
            interpretation = "one or more seeds are nonfinite; temporal order not interpreted"
        elif physically_bad:
            final_verdict = "REJECT_PHYSICAL_INADMISSIBILITY"
            interpretation = "one or more levels physically inadmissible; temporal order not interpreted"
        else:
            n_contract = sum(bool(item["paired_contraction"]) for item in items)
            if n_contract == len(items):
                final_verdict = "PAIRED_CONTRACTION_ALL_SEEDS"
                interpretation = (
                    "paired differences contract in all four seeds; interval is descriptive"
                )
            elif n_contract > 0:
                final_verdict = "MIXED_PAIRED_CONTRACTION"
                interpretation = (
                    "paired contraction is not reproducible across all four seeds; no order identified"
                )
            else:
                final_verdict = "NO_PAIRED_CONTRACTION"
                interpretation = "no paired seed contracts; no order identified"
        paired_summary.append({
            "method": method, "P0": P0, "observable": observable,
            "n_seeds": len(items),
            "mean_value1": float(np.mean([float(item["value1"]) for item in items])),
            "mean_value2": float(np.mean([float(item["value2"]) for item in items])),
            "mean_value3": float(np.mean([float(item["value3"]) for item in items])),
            "mean_D12": float(np.mean([float(item["D12"]) for item in items])),
            "sample_sd_D12": float(np.std([float(item["D12"]) for item in items], ddof=1)),
            "mean_D23": float(np.mean([float(item["D23"]) for item in items])),
            "sample_sd_D23": float(np.std([float(item["D23"]) for item in items], ddof=1)),
            **{
                f"seed_{int(item['seed'])}_D12_over_D23": float(item["D12_over_D23"])
                for item in sorted(items, key=lambda item: int(item["seed"]))
            },
            "n_paired_contractions": sum(
                bool(item["paired_contraction"]) for item in items
            ),
            "mean_paired_difference_D12_minus_D23": dmean,
            "paired_difference_sample_sd": dsd,
            "paired_difference_standard_error": dse,
            "paired_difference_ci95_low": dlo,
            "paired_difference_ci95_high": dhi,
            "confidence_interval_method": "two-sided paired Student t on D12-D23, df=3; descriptive",
            "all_cases_physically_admissible": all(
                bool(item["physically_admissible_case"]) for item in items
            ),
            "final_verdict": final_verdict,
            "interpretation": interpretation,
            "source_csv": str(run_path),
        })
    summary_path = write_csv(out / "timestep" / "timestep_summary.csv", summary)
    paired_path = write_csv(
        out / "timestep" / "timestep_paired_summary.csv", paired_summary
    )
    paired_by_seed_path = write_csv(
        out / "timestep" / "timestep_paired_differences_by_seed.csv", rows
    )
    write_json(out / "timestep" / "timestep_manifest.json", {
        "created_utc": utcnow(), "methods": ["PBME", "MIDPOINT"],
        "P0": list(args.P0), "seeds": list(args.dynamics_seeds),
        "dt_levels": dts, "N": 1000,
        "alignment": "common coarse saved times, linear interpolation, no extrapolation",
        "decision_hierarchy": [
            "numerical floor", "finite output", "physical admissibility",
            "paired within-seed contraction and descriptive paired uncertainty",
        ],
        "physical_admissibility": (
            "endpoint populations in [0,1] within 1e-10; norm/trace within "
            "1e-8 of unity; finite energy; signed central second moments >= -1e-10"
        ),
        "order_guard": (
            "D12 and D23 must exceed the scale-aware numerical floor; raw "
            "cross-seed observable spread is descriptive only and is not an order gate"
        ),
        "sources": sources,
        "outputs": {
            str(path): sha256_file(path)
            for path in (
                run_path, summary_path, paired_path, paired_by_seed_path
            )
        },
    })
    return {
        "run_by_run": run_path, "summary": summary_path,
        "paired_summary": paired_path,
        "paired_by_seed": paired_by_seed_path,
    }


def analyze_support_and_replication(out: Path, args: argparse.Namespace
                                    ) -> Dict[str, Path]:
    support_rows: List[Dict[str, Any]] = []
    source_records: List[Dict[str, Any]] = []
    for P0 in map(float, args.P0):
        for seed in (11, 29, 47):
            for N in map(int, args.support_levels):
                expected = {
                    "P0": P0, "seed": seed, "N": N, "dt": 0.25,
                    "t_final_resolved": 2.0 * 2000.0 * 15.0 / abs(P0),
                    "l2_regularization": 0.05,
                }
                run_dir, reason = resolve_run(out, expected)
                if run_dir is None:
                    raise RuntimeError(
                        f"missing support run P0={P0}, seed={seed}, N={N}: {reason}"
                    )
                source_records.append({
                    "campaign": "independent_support", "P0": P0,
                    "seed": seed, "N": N, "run_directory": str(run_dir),
                    "manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
                })
                for method in ("pbme", "midpoint"):
                    data = load_method_series(run_dir, method)
                    for observable in OBSERVABLES:
                        support_rows.append({
                            "method": method.upper(), "P0": P0, "seed": seed,
                            "N": N, "dt": 0.25, "observable": observable,
                            "endpoint_value": float(data[observable][-1]),
                            "completed": True,
                            "cloud_relationship": "independently sampled; not nested",
                            "source_npz": str(run_dir / f"{method}.npz"),
                            "source_manifest": str(run_dir / "run_manifest.json"),
                        })
    support_run_path = write_csv(
        out / "support" / "independent_cloud_run_by_run.csv", support_rows
    )
    support_summary: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, float, str, int], List[Dict[str, Any]]] = {}
    for row in support_rows:
        grouped.setdefault((
            str(row["method"]), float(row["P0"]), str(row["observable"]), int(row["N"])
        ), []).append(row)
    means: Dict[Tuple[str, float, str, int], Tuple[float, float]] = {}
    for key, items in grouped.items():
        values = [float(item["endpoint_value"]) for item in items]
        mean, sd, se, lo, hi = t_interval(values)
        means[key] = (mean, sd)
        support_summary.append({
            "method": key[0], "P0": key[1], "observable": key[2], "N": key[3],
            "n_seeds": len(values), "seeds": [int(item["seed"]) for item in items],
            "mean": mean, "sample_sd": sd, "standard_error": se,
            "ci95_low": lo, "ci95_high": hi,
            "confidence_interval_method": "two-sided Student t, df=2",
            "cloud_relationship": "independently sampled; not nested",
            "source_csv": str(support_run_path),
        })
    for row in support_summary:
        key = (row["method"], row["P0"], row["observable"], row["N"])
        ref = means.get((row["method"], row["P0"], row["observable"], 2000))
        if ref:
            difference = float(row["mean"] - ref[0])
            combined_sd = float(np.sqrt(row["sample_sd"] ** 2 + ref[1] ** 2))
            row["difference_from_N2000_mean"] = difference
            row["combined_seed_sd_scale"] = combined_sd
            row["change_exceeds_seed_variability"] = (
                abs(difference) > combined_sd if np.isfinite(combined_sd) else False
            )
        else:
            row["difference_from_N2000_mean"] = math.nan
            row["combined_seed_sd_scale"] = math.nan
            row["change_exceeds_seed_variability"] = False
    support_summary_path = write_csv(
        out / "support" / "independent_cloud_summary.csv", support_summary
    )

    replication_rows: List[Dict[str, Any]] = []
    for P0 in map(float, args.P0):
        for seed in map(int, args.dynamics_seeds):
            expected = {
                "P0": P0, "seed": seed, "N": 1000, "dt": 0.25,
                "t_final_resolved": 2.0 * 2000.0 * 15.0 / abs(P0),
                "l2_regularization": 0.05,
            }
            run_dir, reason = resolve_run(out, expected)
            if run_dir is None:
                raise RuntimeError(
                    f"missing replication run P0={P0}, seed={seed}: {reason}"
                )
            source_records.append({
                "campaign": "replication", "P0": P0, "seed": seed, "N": 1000,
                "run_directory": str(run_dir),
                "manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
            })
            for method in ("pbme", "midpoint"):
                data = load_method_series(run_dir, method)
                for observable in OBSERVABLES:
                    replication_rows.append({
                        "record_type": "seed_value", "method": method.upper(),
                        "P0": P0, "seed": seed, "observable": observable,
                        "endpoint_value": float(data[observable][-1]),
                        "source_npz": str(run_dir / f"{method}.npz"),
                        "source_manifest": str(run_dir / "run_manifest.json"),
                    })
    replication_seed_path = write_csv(
        out / "replication" / "four_seed_values.csv", replication_rows
    )
    replication_summary: List[Dict[str, Any]] = []
    rep_groups: Dict[Tuple[str, float, str], List[Dict[str, Any]]] = {}
    for row in replication_rows:
        rep_groups.setdefault(
            (str(row["method"]), float(row["P0"]), str(row["observable"])), []
        ).append(row)
    for (method, P0, observable), items in sorted(rep_groups.items()):
        values = [float(item["endpoint_value"]) for item in items]
        mean, sd, se, lo, hi = t_interval(values)
        spread = float(max(values) - min(values))
        replication_summary.append({
            "method": method, "P0": P0, "observable": observable,
            "n_independent_seeds": len(values),
            "seeds": [int(item["seed"]) for item in items],
            "values": values, "mean": mean, "sample_sd": sd,
            "standard_error": se, "ci95_low": lo, "ci95_high": hi,
            "confidence_interval_method": "two-sided Student t, df=3",
            "maximum_spread": spread,
            "coefficient_of_variation": (
                sd / abs(mean) if np.isfinite(sd) and abs(mean) > 1e-15 else math.nan
            ),
            "limitation": "four independent seeds provide a small-sample sensitivity interval, not strong uncertainty calibration",
            "source_csv": str(replication_seed_path),
        })
    replication_summary_path = write_csv(
        out / "replication" / "four_seed_summary.csv", replication_summary
    )
    write_json(out / "support" / "support_replication_manifest.json", {
        "created_utc": utcnow(),
        "support_design": "independent clouds across N; no pointwise support order",
        "support_seeds": [11, 29, 47],
        "replication_seeds": list(args.dynamics_seeds),
        "sources": source_records,
        "outputs": {
            str(path): sha256_file(path)
            for path in (
                support_run_path, support_summary_path,
                replication_seed_path, replication_summary_path,
            )
        },
    })
    return {
        "support_run": support_run_path,
        "support_summary": support_summary_path,
        "replication_values": replication_seed_path,
        "replication_summary": replication_summary_path,
    }


def analyze_references(out: Path, args: argparse.Namespace) -> Dict[str, Path]:
    from Compare_gp_se_qcle import tdse_grid_metadata

    outputs: Dict[str, Path] = {}
    for method in ("tdse", "qcle"):
        rows: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []
        base = out / (
            "reference_tdse" if method == "tdse" else "reference_grid_qcle"
        )
        for mode in ("time", "grid"):
            for P0 in map(float, args.P0):
                path = base / f"{method}_{mode}_P0{P0:g}.json"
                if not path.exists():
                    raise RuntimeError(f"required reference result absent: {path}")
                data = read_json(path)
                levels = data.get("levels", [])
                if len(levels) != 3:
                    raise RuntimeError(f"{path}: expected three levels")
                sources.append({
                    "source": str(path), "sha256": sha256_file(path),
                    "series_file": data.get("finest_series_file"),
                    "series_sha256": data.get("finest_series_sha256"),
                })
                for observable in OBSERVABLES:
                    metric = data.get("observables", {}).get(observable)
                    if metric is None:
                        raise RuntimeError(f"{path}: observable {observable} absent")
                    # Apply the declared interpretation hierarchy before an
                    # order is promoted: numerical-noise floor, monotone
                    # contraction, then the non-asymptotic rapid-contraction
                    # guard.  Raw levels and differences remain visible.
                    order_reason = metric["reason"]
                    try:
                        p_value = float(metric["p_obs"])
                    except (TypeError, ValueError):
                        p_value = None
                    level_values = [
                        float(metric["coarse"]),
                        float(metric["fine"]),
                        float(metric["finer"]),
                    ]
                    delta12 = float(metric["abs_diff_coarse_fine"])
                    delta23 = float(metric["abs_diff_fine_finer"])
                    noise_threshold = numerical_noise_threshold(level_values)
                    if min(delta12, delta23) <= noise_threshold:
                        order_reason = "roundoff_or_saturation_limited"
                    elif p_value is not None and p_value <= 0.0:
                        order_reason = "nonmonotone_difference_retained"
                    elif (
                        order_reason == "ok"
                        and p_value is not None
                        and p_value > MAX_INTERPRETABLE_ORDER
                    ):
                        order_reason = "rapid_contraction_not_asymptotic"
                    interpreted_order: Any = (
                        metric["p_obs"]
                        if order_reason == "ok"
                        else "NOT COMPUTED"
                    )
                    row: Dict[str, Any] = {
                        "method": method.upper(), "P0": P0,
                        "refinement_mode": mode, "observable": observable,
                        "value1": metric["coarse"], "value2": metric["fine"],
                        "value3": metric["finer"],
                        "delta12": metric["abs_diff_coarse_fine"],
                        "delta23": metric["abs_diff_fine_finer"],
                        # Preserve the raw ratio-derived number for auditability,
                        # but expose an interpreted order only after every guard.
                        "raw_ratio_order": metric["p_obs"],
                        "p_observed": interpreted_order,
                        "order_reason": order_reason,
                        "numerical_noise_abs_tolerance": NUMERICAL_NOISE_ABS_TOL,
                        "numerical_noise_rel_tolerance": NUMERICAL_NOISE_REL_TOL,
                        "numerical_noise_threshold": noise_threshold,
                        "declared_edge_mass_tolerance": data.get(
                            "resolved_configuration", {}
                        ).get("edge_mass_tolerance"),
                        "declared_negative_momentum_tolerance": data.get(
                            "resolved_configuration", {}
                        ).get("negative_momentum_tolerance"),
                        "source_file": str(path),
                    }
                    for index, level in enumerate(levels, 1):
                        meta = dict(level["metadata"])
                        if method == "tdse":
                            configuration = data.get("resolved_configuration", {})
                            derived = tdse_grid_metadata(
                                float(configuration["R0"]),
                                float(configuration["P0"]),
                                float(configuration["sigma_R"]),
                                float(level["dt"]),
                                int(level["n_steps"]),
                                n_grid_min=int(level["n_grid"]),
                            )
                            # This is exact deterministic metadata recovery,
                            # not a numerical-result substitution. Existing
                            # run metadata remains authoritative when present.
                            meta = {**derived, **meta}
                        row[f"level{index}_label"] = level["label"]
                        row[f"level{index}_dt"] = level["dt"]
                        row[f"level{index}_n_steps"] = level["n_steps"]
                        row[f"level{index}_t_final"] = meta.get("t_final")
                        for key in (
                            "R_min", "R_max", "P_min", "P_max", "dR", "dP",
                            "n_grid_actual", "n_R", "n_P", "boundary_rule",
                            "absorber_policy", "split_operator_composition",
                            "fft_convention", "derivative_method",
                            "time_integrator", "edge_mass_5pct",
                            "maximum_edge_mass_5pct",
                            "negative_momentum_probability",
                            "maximum_negative_momentum_probability",
                            "edge_R_mass_5pct", "edge_P_mass_5pct",
                            "maximum_edge_R_mass_5pct",
                            "maximum_edge_P_mass_5pct",
                            "edge_phase_space_R_mass_5pct",
                            "edge_phase_space_P_mass_5pct",
                            "maximum_edge_phase_space_R_mass_5pct",
                            "maximum_edge_phase_space_P_mass_5pct",
                            "edge_mass_convention", "cfl_dt_max", "cfl_ratio",
                        ):
                            row[f"level{index}_{key}"] = meta.get(key)
                    rows.append(row)
        filename = "tdse_three_level.csv" if method == "tdse" else "qcle_three_level.csv"
        csv_path = write_csv(base / filename, rows)
        manifest_name = "tdse_manifest.json" if method == "tdse" else "qcle_manifest.json"
        write_json(base / manifest_name, {
            "created_utc": utcnow(), "method": method,
            "refinement_modes": ["time", "grid"], "P0": list(args.P0),
            "observables": list(OBSERVABLES),
            "order_rule": "log2(delta12/delta23) with scale-aware roundoff guard; rejected and negative rows retained",
            "sources": sources,
            "output": {"path": str(csv_path), "sha256": sha256_file(csv_path)},
        })
        outputs[method] = csv_path
    return outputs


def error_metrics(reference: np.ndarray, estimate: np.ndarray,
                  axes: Sequence[np.ndarray]) -> Dict[str, float]:
    ref = np.asarray(reference, float)
    est = np.asarray(estimate, float)
    if ref.shape != est.shape:
        raise ValueError(f"density shape mismatch {ref.shape} != {est.shape}")
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    def integrate(value: np.ndarray) -> float:
        result = np.asarray(value, float)
        for axis_values in reversed(axes):
            result = trap(result, axis_values, axis=-1)
        return float(result)

    diff = est - ref
    denom1 = integrate(np.abs(ref))
    denom2 = math.sqrt(max(integrate(ref * ref), 0.0))
    denom_inf = float(np.max(np.abs(ref)))
    floor = 1e-30
    return {
        "raw_mass_reference": integrate(ref),
        "raw_mass_estimate": integrate(est),
        "E1": integrate(np.abs(diff)) / max(denom1, floor),
        "E2": math.sqrt(max(integrate(diff * diff), 0.0)) / max(denom2, floor),
        "Einf": float(np.max(np.abs(diff))) / max(denom_inf, floor),
    }


def normalized_density(value: np.ndarray, axes: Sequence[np.ndarray]) -> np.ndarray:
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    mass: Any = np.asarray(value, float)
    for axis_values in reversed(axes):
        mass = trap(mass, axis_values, axis=-1)
    mass = float(mass)
    return np.asarray(value, float) / mass if abs(mass) > 1e-30 else np.full_like(value, np.nan)


def production_snapshot(run_dir: Path, method: str
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    path = run_dir / f"{method}.npz"
    with np.load(path) as data:
        prefix = snapshot_prefix(data.files, "final")
        Z = np.asarray(data[f"{prefix}_Z"], float)
        y = np.asarray(data[f"{prefix}_y"], float)
        omega = np.asarray(data[f"{prefix}_geometric_measure"], float)
    return Z, y, omega, prefix


def gaussian_kde_1d(grid: np.ndarray, points: np.ndarray,
                    weights: np.ndarray, bandwidth: float) -> np.ndarray:
    norm = bandwidth * math.sqrt(2.0 * math.pi)
    kernel = np.exp(-0.5 * ((grid[:, None] - points[None, :]) / bandwidth) ** 2) / norm
    return kernel @ weights


def gaussian_kde_2d(R: np.ndarray, P: np.ndarray, Z: np.ndarray,
                    weights: np.ndarray, hR: float, hP: float) -> np.ndarray:
    normR = hR * math.sqrt(2.0 * math.pi)
    normP = hP * math.sqrt(2.0 * math.pi)
    KR = np.exp(-0.5 * ((R[:, None] - Z[None, :, 0]) / hR) ** 2) / normR
    KP = np.exp(-0.5 * ((P[:, None] - Z[None, :, 1]) / hP) ** 2) / normP
    return (KR * weights[None, :]) @ KP.T


def analyze_physical_comparison(out: Path, args: argparse.Namespace
                                ) -> Dict[str, Path]:
    from Dynamics import _support_mapping_observables

    observable_rows: List[Dict[str, Any]] = []
    density_rows: List[Dict[str, Any]] = []
    source_records: List[Dict[str, Any]] = []
    for P0 in map(float, args.P0):
        references: Dict[str, Dict[str, np.ndarray]] = {}
        reference_paths: Dict[str, Path] = {}
        for reference in ("tdse", "qcle"):
            base = out / (
                "reference_tdse" if reference == "tdse" else "reference_grid_qcle"
            )
            path = base / f"{reference}_time_P0{P0:g}.npz"
            if not path.exists():
                raise RuntimeError(f"reference time series absent: {path}")
            with np.load(path, allow_pickle=False) as data:
                references[reference] = {
                    key: np.asarray(data[key])
                    for key in data.files
                }
            reference_paths[reference] = path
        for seed in map(int, args.dynamics_seeds):
            expected = {
                "P0": P0, "seed": seed, "N": 1000, "dt": 0.25,
                "t_final_resolved": 2.0 * 2000.0 * 15.0 / abs(P0),
                "l2_regularization": 0.05,
            }
            run_dir, reason = resolve_run(out, expected)
            if run_dir is None:
                raise RuntimeError(f"paired comparison run absent: {reason}")
            manifest = read_json(run_dir / "run_manifest.json")
            if not manifest.get("paired_initial_cloud"):
                raise RuntimeError(f"{run_dir}: paired initial cloud is not certified")
            support_hash = manifest.get("paired_initial_cloud_sha256")
            for method in ("pbme", "midpoint"):
                series = load_method_series(run_dir, method)
                for reference, ref in references.items():
                    for observable in OBSERVABLES:
                        t = np.asarray(series["t"], float)
                        t_ref = np.asarray(ref["t"], float)
                        lo, hi = max(t[0], t_ref[0]), min(t[-1], t_ref[-1])
                        common = t[(t >= lo) & (t <= hi)]
                        u = np.interp(common, t, series[observable])
                        v = np.interp(common, t_ref, np.asarray(ref[observable], float))
                        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
                        duration = float(common[-1] - common[0])
                        rms = float(np.sqrt(max(
                            trap((u - v) ** 2, common) / duration, 0.0
                        )))
                        linf = float(np.max(np.abs(u - v)))
                        observable_rows.append({
                            "method": method.upper(), "reference": reference.upper(),
                            "P0": P0, "seed": seed, "observable": observable,
                            "E_RMS": rms, "E_infinity": linf,
                            "common_t_min": float(common[0]),
                            "common_t_max": float(common[-1]),
                            "alignment": "production saved times within reference interval; linear interpolation; no extrapolation",
                            "paired_initial_cloud_sha256": support_hash,
                            "production_npz": str(run_dir / f"{method}.npz"),
                            "reference_npz": str(reference_paths[reference]),
                        })

                Z, y, omega, prefix = production_snapshot(run_dir, method)
                mapping = _support_mapping_observables(Z, 1.0)
                physical_weight = omega * y * mapping["trace"]
                N = len(Z)
                if "snap_R" in references["tdse"] and "snap_psi" in references["tdse"]:
                    R = np.asarray(references["tdse"]["snap_R"], float)
                    psi = np.asarray(references["tdse"]["snap_psi"])[-1]
                    ref_rho = np.sum(np.abs(psi) ** 2, axis=0)
                    hR = max(1.06 * float(np.std(Z[:, 0])) * N ** (-1 / 5), 1e-6)
                    est_rho = gaussian_kde_1d(R, Z[:, 0], physical_weight, hR)
                    raw = error_metrics(ref_rho, est_rho, [R])
                    shape = error_metrics(
                        normalized_density(ref_rho, [R]),
                        normalized_density(est_rho, [R]), [R],
                    )
                    density_rows.append({
                        "method": method.upper(), "reference": "TDSE",
                        "P0": P0, "seed": seed, "snapshot": prefix,
                        "quantity": "nuclear R marginal",
                        "bandwidth_R": hR, "bandwidth_P": None,
                        **{f"raw_{k}": v for k, v in raw.items()},
                        **{f"shape_{k}": v for k, v in shape.items()},
                        "paired_initial_cloud_sha256": support_hash,
                        "production_npz": str(run_dir / f"{method}.npz"),
                        "reference_npz": str(reference_paths["tdse"]),
                    })
                qref = references["qcle"]
                if all(key in qref for key in ("snap_R_axis", "snap_P_axis", "snap_A", "snap_C")):
                    R = np.asarray(qref["snap_R_axis"], float)
                    P = np.asarray(qref["snap_P_axis"], float)
                    ref_rho = np.asarray(qref["snap_A"] + qref["snap_C"], float)
                    hR = max(1.06 * float(np.std(Z[:, 0])) * N ** (-1 / 6), 1e-6)
                    hP = max(1.06 * float(np.std(Z[:, 1])) * N ** (-1 / 6), 1e-6)
                    est_rho = gaussian_kde_2d(R, P, Z, physical_weight, hR, hP)
                    raw = error_metrics(ref_rho, est_rho, [R, P])
                    shape = error_metrics(
                        normalized_density(ref_rho, [R, P]),
                        normalized_density(est_rho, [R, P]), [R, P],
                    )
                    density_rows.append({
                        "method": method.upper(), "reference": "QCLE",
                        "P0": P0, "seed": seed, "snapshot": prefix,
                        "quantity": "mapping-integrated R-P density",
                        "bandwidth_R": hR, "bandwidth_P": hP,
                        **{f"raw_{k}": v for k, v in raw.items()},
                        **{f"shape_{k}": v for k, v in shape.items()},
                        "paired_initial_cloud_sha256": support_hash,
                        "production_npz": str(run_dir / f"{method}.npz"),
                        "reference_npz": str(reference_paths["qcle"]),
                    })
            source_records.append({
                "P0": P0, "seed": seed, "run_directory": str(run_dir),
                "paired_initial_cloud_sha256": support_hash,
                "manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
            })

    observable_path = write_csv(
        out / "physical_comparison" / "observable_errors_by_seed.csv",
        observable_rows,
    )
    density_path = write_csv(
        out / "physical_comparison" / "density_errors_by_seed.csv",
        density_rows,
    )
    summaries: List[Dict[str, Any]] = []
    pairs: Dict[
        Tuple[str, str, float, str, str],
        Dict[int, Dict[str, float]],
    ] = {}
    for row in observable_rows:
        key = (
            "observable", str(row["reference"]), float(row["P0"]),
            str(row["observable"]), "E_RMS",
        )
        pairs.setdefault(key, {}).setdefault(int(row["seed"]), {})[
            str(row["method"])
        ] = float(row["E_RMS"])
        key_inf = (
            "observable", str(row["reference"]), float(row["P0"]),
            str(row["observable"]), "E_infinity",
        )
        pairs.setdefault(key_inf, {}).setdefault(int(row["seed"]), {})[
            str(row["method"])
        ] = float(row["E_infinity"])
    for row in density_rows:
        for metric in (
            "raw_E1", "raw_E2", "raw_Einf",
            "shape_E1", "shape_E2", "shape_Einf",
        ):
            key = (
                "density", str(row["reference"]), float(row["P0"]),
                str(row["quantity"]), metric,
            )
            pairs.setdefault(key, {}).setdefault(int(row["seed"]), {})[
                str(row["method"])
            ] = float(row[metric])
    for (
        comparison_kind, reference, P0, observable, metric
    ), by_seed in sorted(pairs.items()):
        deltas = []
        seeds = []
        for seed, methods in sorted(by_seed.items()):
            if "MIDPOINT" in methods and "PBME" in methods:
                deltas.append(methods["MIDPOINT"] - methods["PBME"])
                seeds.append(seed)
        mean, sd, se, lo, hi = t_interval(deltas)
        if np.isfinite(hi) and hi < 0:
            verdict = "MIDPOINT_ERROR_SMALLER_BEFORE_SCIENTIFIC_GATES"
        elif np.isfinite(lo) and lo > 0:
            verdict = "MIDPOINT_ERROR_LARGER"
        else:
            verdict = "NO_RESOLVED_DIFFERENCE"
        summaries.append({
            "comparison_kind": comparison_kind,
            "reference": reference, "P0": P0, "observable": observable,
            "metric": metric, "n_paired_seeds": len(deltas), "seeds": seeds,
            "paired_differences_midpoint_minus_pbme": deltas,
            "mean_paired_difference": mean, "sample_sd": sd,
            "standard_error": se, "ci95_low": lo, "ci95_high": hi,
            "confidence_interval_method": "two-sided paired Student t, df=3",
            "verdict_before_scientific_gates": verdict,
            "source_csv": str(
                observable_path
                if comparison_kind == "observable" else density_path
            ),
        })
    summary_path = write_csv(
        out / "physical_comparison" / "paired_improvement_summary.csv",
        summaries,
    )
    write_json(out / "physical_comparison" / "comparison_manifest.json", {
        "created_utc": utcnow(),
        "pairing_contract": "identical stored initial support hash within each production run; same N, dt, endpoint, estimator and reference",
        "density_contract": "raw-amplitude and normalized shape errors reported separately; TDSE uses nuclear R marginal; QCLE uses mapping-integrated R-P density",
        "sources": source_records,
        "outputs": {
            str(path): sha256_file(path)
            for path in (observable_path, density_path, summary_path)
        },
    })
    return {
        "observable": observable_path, "density": density_path,
        "summary": summary_path,
    }


def tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    # Escape characters from the original string exactly once.  Sequential
    # str.replace calls would re-escape braces introduced by
    # ``\textbackslash{}`` and produce malformed path cells.
    return "".join(replacements.get(character, character) for character in text)


def tex_number(value: Any) -> str:
    if value is None or value == "":
        return r"\textsc{Not Computed}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    status = str(value).strip().upper()
    if status in ("TRUE", "FALSE"):
        return "yes" if status == "TRUE" else "no"
    status_tex = {
        "NOT COMPUTED": r"\textsc{Not Computed}",
        "DATA ABSENT": r"\textsc{Data Absent}",
        "RUN INCOMPLETE": r"\textsc{Run Incomplete}",
        "NOT IDENTIFIABLE": r"\textsc{Not Identifiable}",
        "NO_MONOTONE_DECREASE_OBSERVED": "no monotone decrease",
        "MONOTONE_DECREASE_OBSERVED": "monotone decrease",
        "REJECT_NUMERICAL_NOISE": "noise-floor reject",
        "REJECT_PHYSICAL_INADMISSIBILITY": "physical-admissibility reject",
        "COMPUTED_ZERO_OR_NEGATIVE": "zero/negative",
        "COMPUTED_POSITIVE": "positive",
        "PAIRED_CONTRACTION_ALL_SEEDS": "all four paired seeds contract",
        "MIXED_PAIRED_CONTRACTION": "mixed paired contraction",
        "NO_PAIRED_CONTRACTION": "no paired contraction",
        "NO_RESOLVED_DIFFERENCE": "no resolved difference",
        "MIDPOINT_ERROR_LARGER": "MIDPOINT larger",
        "MAPPING-INTEGRATED R-P DENSITY": r"$R$--$P$ density",
        "P0": r"$\rho_{11}^{\mathrm{SN}}$",
        "P1": r"$\rho_{22}^{\mathrm{SN}}$",
    }
    if status in status_tex:
        return status_tex[status]
    if status.startswith("INSUFFICIENT EVIDENCE"):
        return "insufficient evidence"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return tex_escape(value)
    if not np.isfinite(number):
        return r"\textsc{Not Computed}"
    if number == 0:
        return "0"
    if abs(number) < 1e-3 or abs(number) >= 1e4:
        return f"{number:.6e}"
    return f"{number:.7g}"


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def longtable_from_csv(csv_path: Path, tex_path: Path,
                       columns: Sequence[Tuple[str, str]],
                       caption: str, label: str,
                       note: str) -> Path:
    rows = read_csv_rows(csv_path)
    n_columns = len(columns)
    colspec = "@{}" + "r" * n_columns + "@{}"
    landscape = n_columns > 8
    font = r"\tiny" if n_columns > 10 else r"\scriptsize"
    tabcolsep = "0.25pt" if n_columns > 10 else "1.0pt"
    lines = [
        "% Auto-generated by reviewer_final_closure.py.",
        f"% Source CSV: {csv_path}",
        *([r"\begin{landscape}"] if landscape else []),
        r"\begingroup",
        font,
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
        rf"\begin{{longtable}}{{{colspec}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        " & ".join(header for _, header in columns) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(header for _, header in columns) + r" \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(
            " & ".join(tex_number(row.get(key)) for key, _ in columns) + r" \\"
        )
    lines.extend([
        r"\bottomrule",
        rf"\multicolumn{{{n_columns}}}{{p{{0.97\linewidth}}}}{{\footnotesize "
        + tex_escape(note) + r"}\\",
        r"\end{longtable}",
        r"\endgroup",
        *([r"\end{landscape}"] if landscape else []),
        "",
    ])
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return tex_path


def fitted_landscape_table_from_csv(
    csv_path: Path,
    tex_path: Path,
    columns: Sequence[Tuple[str, str]],
    caption: str,
    label: str,
    note: str,
) -> Path:
    """Write a short, wide table at a guaranteed landscape page width.

    This is used for the eight-row reference-settings crosswalk.  A regular
    ``longtable`` cannot be width-fitted and allowed the domain ladders to
    project into the margins; a page-sized tabular keeps all exact settings
    together and legible without deleting a reviewer-requested field.
    """
    rows = read_csv_rows(csv_path)
    n_columns = len(columns)
    colspec = "@{}" + "r" * n_columns + "@{}"
    lines = [
        "% Auto-generated by reviewer_final_closure.py.",
        f"% Source CSV: {csv_path}",
        r"\begin{landscape}",
        r"\begin{table}[p]",
        r"\centering",
        rf"\caption{{{caption}}}\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(header for _, header in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(tex_number(row.get(key)) for key, _ in columns) + r" \\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\par\vspace{3pt}",
        r"\begin{minipage}{0.97\linewidth}\footnotesize "
        + tex_escape(note) + r"\end{minipage}",
        r"\end{table}",
        r"\end{landscape}",
        "",
    ])
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return tex_path


def preserve_existing_evidence(out: Path) -> Dict[str, Path]:
    source_map = {
        "raw_conservation.csv": REPO / "reviewer_data_audit" / "tables" / "raw_conservation.csv",
        "projection_leakage.csv": REPO / "reviewer_data_audit" / "tables" / "seo_projection_leakage_all_snapshots.csv",
        "kde_gp_baseline.csv": REPO / "thesis_revision_evidence" / "tables" / "kde_gp_four_seed_three_snapshot.csv",
        "numerical_stability_audit.csv": REPO / "reviewer_data_audit" / "numerical_stability_audit.csv",
        "figure_disposition.csv": REPO / "thesis_revision_evidence" / "FIGURE_DISPOSITION.csv",
    }
    preserved = out / "preserved_evidence"
    preserved.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Path] = {}
    provenance: List[Dict[str, Any]] = []
    for name, source in source_map.items():
        if not source.exists():
            raise FileNotFoundError(f"preserved evidence source absent: {source}")
        target = preserved / name
        shutil.copy2(source, target)
        result[name] = target
        provenance.append({
            "target": str(target), "target_sha256": sha256_file(target),
            "source": str(source), "source_sha256": sha256_file(source),
            "copy_is_byte_identical": sha256_file(target) == sha256_file(source),
        })
    write_json(preserved / "preserved_evidence_manifest.json", {
        "created_utc": utcnow(),
        "rule": "existing computations preserved byte-for-byte; not rerun",
        "files": provenance,
    })
    return result


def build_reader_facing_closure_outputs(
    out: Path, paths: Dict[str, Path]
) -> None:
    """Create compact, thesis-ready views of already verified calculations.

    This function performs no propagation and no fitting.  It only checks and
    reshapes the complete per-seed outputs so that absolute method errors,
    initial-label distributions, and reference settings can be read directly
    without reconstructing them from aggregate summaries.
    """

    density_rows = read_csv_rows(paths["physical_density"])
    density_groups: Dict[Tuple[str, str, str], Dict[str, Dict[str, str]]] = {}
    for row in density_rows:
        key = (row["reference"], row["P0"], row["seed"])
        density_groups.setdefault(key, {})[row["method"]] = row
    density_pairs: List[Dict[str, Any]] = []
    for (reference, P0, seed), methods in sorted(
        density_groups.items(),
        key=lambda item: (float(item[0][1]), item[0][0], int(item[0][2])),
    ):
        if set(methods) != {"PBME", "MIDPOINT"}:
            raise RuntimeError(
                f"incomplete density method pair for {(reference, P0, seed)}"
            )
        pbme, midpoint = methods["PBME"], methods["MIDPOINT"]
        if pbme["paired_initial_cloud_sha256"] != midpoint["paired_initial_cloud_sha256"]:
            raise RuntimeError(
                f"unpaired density support for {(reference, P0, seed)}"
            )
        density_pairs.append({
            "reference": reference,
            "P0": P0,
            "seed": seed,
            "quantity": pbme["quantity"],
            "PBME_raw_E1": pbme["raw_E1"],
            "PBME_raw_E2": pbme["raw_E2"],
            "MIDPOINT_raw_E1": midpoint["raw_E1"],
            "MIDPOINT_raw_E2": midpoint["raw_E2"],
            "PBME_shape_E1": pbme["shape_E1"],
            "PBME_shape_E2": pbme["shape_E2"],
            "MIDPOINT_shape_E1": midpoint["shape_E1"],
            "MIDPOINT_shape_E2": midpoint["shape_E2"],
            "paired_initial_cloud_sha256": pbme["paired_initial_cloud_sha256"],
        })
    paths["physical_density_per_seed"] = write_csv(
        out / "physical_comparison" / "density_errors_method_pair_by_seed.csv",
        density_pairs,
    )

    observable_rows = [
        row for row in read_csv_rows(paths["physical_observable"])
        if row["observable"] in {"P0", "P1"}
    ]
    observable_groups: Dict[
        Tuple[str, str, str, str], Dict[str, Dict[str, str]]
    ] = {}
    for row in observable_rows:
        key = (row["reference"], row["P0"], row["seed"], row["observable"])
        observable_groups.setdefault(key, {})[row["method"]] = row
    observable_pairs: List[Dict[str, Any]] = []
    for (reference, P0, seed, observable), methods in sorted(
        observable_groups.items(),
        key=lambda item: (
            float(item[0][1]), item[0][0], int(item[0][2]), item[0][3]
        ),
    ):
        if set(methods) != {"PBME", "MIDPOINT"}:
            raise RuntimeError(
                "incomplete observable method pair for "
                f"{(reference, P0, seed, observable)}"
            )
        pbme, midpoint = methods["PBME"], methods["MIDPOINT"]
        if pbme["paired_initial_cloud_sha256"] != midpoint["paired_initial_cloud_sha256"]:
            raise RuntimeError(
                "unpaired observable support for "
                f"{(reference, P0, seed, observable)}"
            )
        observable_pairs.append({
            "reference": reference,
            "P0": P0,
            "seed": seed,
            "observable": observable,
            "PBME_E_RMS": pbme["E_RMS"],
            "MIDPOINT_E_RMS": midpoint["E_RMS"],
            "MIDPOINT_minus_PBME_E_RMS": (
                float(midpoint["E_RMS"]) - float(pbme["E_RMS"])
            ),
            "PBME_E_infinity": pbme["E_infinity"],
            "MIDPOINT_E_infinity": midpoint["E_infinity"],
            "MIDPOINT_minus_PBME_E_infinity": (
                float(midpoint["E_infinity"]) - float(pbme["E_infinity"])
            ),
            "paired_initial_cloud_sha256": pbme["paired_initial_cloud_sha256"],
        })
    paths["physical_observable_per_seed"] = write_csv(
        out / "physical_comparison" / "observable_errors_method_pair_by_seed.csv",
        observable_pairs,
    )

    distribution_rows = read_csv_rows(paths["tail_distribution"])
    distribution_groups: Dict[Tuple[str, str], Dict[str, Dict[str, str]]] = {}
    for row in distribution_rows:
        distribution_groups.setdefault(
            (row["P0"], row["seed"]), {}
        )[row["method"]] = row
    paired_distributions: List[Dict[str, Any]] = []
    compare_fields = (
        "N", "minimum", "maximum", "q0.001", "q0.01", "q0.05",
        "q0.1", "q0.25", "q0.5", "q0.75", "q0.9", "q0.95",
        "q0.99", "q0.999", "initial_label_sha256",
        "initial_cloud_sha256",
    )
    for (P0, seed), methods in sorted(
        distribution_groups.items(), key=lambda item: (float(item[0][0]), int(item[0][1]))
    ):
        if set(methods) != {"PBME", "MIDPOINT"}:
            raise RuntimeError(f"incomplete initial-label pair for {(P0, seed)}")
        pbme, midpoint = methods["PBME"], methods["MIDPOINT"]
        if any(pbme[field] != midpoint[field] for field in compare_fields):
            raise RuntimeError(f"initial-label distribution is not paired for {(P0, seed)}")
        paired_distributions.append({
            "P0": P0,
            "seed": seed,
            **{field: pbme[field] for field in compare_fields},
        })
    paths["tail_distribution_paired"] = write_csv(
        out / "tail_sensitivity" / "y0_distribution_paired.csv",
        paired_distributions,
    )

    reference_settings: List[Dict[str, Any]] = []
    for method, key in (("TDSE", "tdse"), ("grid QCLE", "qcle")):
        rows = read_csv_rows(paths[key])
        by_case: Dict[Tuple[str, str], Dict[str, str]] = {}
        for row in rows:
            by_case.setdefault((row["P0"], row["refinement_mode"]), row)
        for (P0, mode), row in sorted(
            by_case.items(), key=lambda item: (float(item[0][0]), item[0][1])
        ):
            dt_ladder = "/".join(
                f"{float(row[f'level{level}_dt']):g}" for level in (1, 2, 3)
            )
            if method == "TDSE":
                grid_ladder = "/".join(
                    str(int(float(row[f"level{level}_n_grid_actual"])))
                    for level in (1, 2, 3)
                )
                edge = max(
                    float(row[f"level{level}_maximum_edge_mass_5pct"])
                    for level in (1, 2, 3)
                )
                cfl = ""
            else:
                grid_ladder = "/".join(
                    f"{int(float(row[f'level{level}_n_R']))}x"
                    f"{int(float(row[f'level{level}_n_P']))}"
                    for level in (1, 2, 3)
                )
                edge = max(
                    float(row[f"level3_maximum_edge_R_mass_5pct"]),
                    float(row[f"level3_maximum_edge_P_mass_5pct"]),
                )
                cfl = row["level3_cfl_ratio"]

            def domain(axis: str) -> str:
                return "/".join(
                    "["
                    + f"{float(row[f'level{level}_{axis}_min']):g},"
                    + f"{float(row[f'level{level}_{axis}_max']):g}]"
                    for level in (1, 2, 3)
                )

            reference_settings.append({
                "method": method,
                "P0": P0,
                "mode": mode,
                "t_final": row["level3_t_final"],
                "dt_ladder": dt_ladder,
                "grid_ladder": grid_ladder,
                "R_domain_ladder": domain("R"),
                "P_domain_ladder": "---" if method == "TDSE" else domain("P"),
                "finest_physical_edge_fraction": edge,
                "finest_CFL_ratio": cfl or "---",
                "interpretation": (
                    "boundary adequacy/repeatability; no independent grid order"
                    if method == "TDSE" and float(P0) == 100.0 and mode == "grid"
                    else "three-level refinement"
                ),
            })
    paths["reference_settings"] = write_csv(
        out / "reference_settings_by_method_and_momentum.csv",
        reference_settings,
    )


def generate_tables(out: Path, paths: Mapping[str, Path],
                    preserved: Mapping[str, Path],
                    plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    crosswalk: List[Dict[str, Any]] = []

    inventory_rows = [
        {"campaign": kind, **counts}
        for kind, counts in sorted(plan["counts"].items())
    ]
    # Replication is an examiner-facing campaign even though its N=1000,
    # dt=0.25 configurations are reused from the time/support job pool.
    # Count paired run cells rather than double-counting PBME and MIDPOINT as
    # independent stochastic replicates.
    replication_values = read_csv_rows(paths["replication_values"])
    replication_methods: Dict[Tuple[str, str], set] = {}
    for row in replication_values:
        replication_methods.setdefault(
            (row["P0"], row["seed"]), set()
        ).add(row["method"])
    replication_verified = sum(
        methods == {"PBME", "MIDPOINT"}
        for methods in replication_methods.values()
    )
    resolved_arguments = plan.get("resolved_arguments", {})
    replication_expected = (
        len(resolved_arguments.get("P0", []))
        * len(resolved_arguments.get("dynamics_seeds", []))
    )
    inventory_rows.append({
        "campaign": "replication",
        "expected": replication_expected,
        "reuse": replication_verified,
        "missing": replication_expected - replication_verified,
    })
    inventory_csv = write_csv(
        out / "validation_inventory.csv",
        inventory_rows,
    )
    inventory_tex = longtable_from_csv(
        inventory_csv, tables / "ValidationInventory.tex",
        (("campaign", "Campaign"), ("expected", "Expected"),
         ("reuse", "Verified"), ("missing", "Missing")),
        "Final validation inventory after manifest and finite-endpoint verification.",
        "tab:final-validation-inventory",
        "Run completion is distinct from scientific acceptance. Source: validation_inventory.csv.",
    )
    crosswalk.append({
        "table": str(inventory_tex), "source_csv": str(inventory_csv),
        "source_csv_sha256": sha256_file(inventory_csv),
    })

    manufactured_csv = paths["manufactured_complete"]
    quantity_specs = (
        ("density", "Density"), ("gradient", "Gradient"), ("Q", r"$Q[\rho]$"),
    )
    manufactured_parts: List[str] = []
    for prefix, display in quantity_specs:
        temp = tables / f"_manufactured_{prefix}.tex"
        longtable_from_csv(
            manufactured_csv, temp,
            (
                ("l2_regularization", r"$\ell_2$"), ("N", r"$N$"),
                ("seed", "Seed"), ("query_type", "Query"), ("query_count", r"$n_q$"),
                (f"{prefix}_relative_l1", r"$E_1$"),
                (f"{prefix}_relative_l2", r"$E_2$"),
                (f"{prefix}_relative_linf", r"$E_\infty$"),
                (f"{prefix}_mae", "MAE"), (f"{prefix}_rmse", "RMSE"),
                (f"{prefix}_linf", r"$L^\infty$"),
                (f"{prefix}_denominator_floor_used", "floor used?"),
                ("cholesky_adaptive_jitter", "jitter"),
                ("cholesky_attempts", "tries"),
            ),
            f"Manufactured-operator complete paired policy table: {display}.",
            f"tab:manufactured-complete-{prefix.lower()}",
            "The on-support set contains all N training points; the independent off-support set contains 1000 points. Relative and absolute errors are printed. Production and manufactured regularizations are not conflated.",
        )
        manufactured_parts.append(temp.read_text(encoding="utf-8"))
        temp.unlink()
    manufactured_tex = tables / "ManufacturedComplete.tex"
    manufactured_tex.write_text("\n".join(manufactured_parts), encoding="utf-8")
    crosswalk.append({
        "table": str(manufactured_tex), "source_csv": str(manufactured_csv),
        "source_csv_sha256": sha256_file(manufactured_csv),
    })

    table_specs = [
        (
            "manufactured_summary", tables / "ManufacturedSummary.tex",
            (("l2_regularization", r"$\ell_2$"), ("N", r"$N$"),
             ("query_type", "Query"), ("metric", "Metric"),
             ("n_seeds", r"$n$"), ("mean", "Mean"), ("sample_sd", "SD"),
             ("standard_error", "SE"), ("ci95_low", r"95\% low"),
             ("ci95_high", r"95\% high"),
             ("baseline_l2_regularization", r"baseline $\ell_2$"),
             ("mean_paired_difference_from_baseline",
              r"mean paired $\Delta$"),
             ("paired_difference_sample_sd", r"paired $\Delta$ SD"),
             ("paired_difference_ci95_low", r"paired 95\% low"),
             ("paired_difference_ci95_high", r"paired 95\% high"),
             ("paired_training_and_query_clouds_verified", "paired?")),
            "Manufactured-test means, spread, uncertainty, and paired regularization-policy comparison.",
            "tab:manufactured-summary",
            "Three independent seeds per cell; two-sided Student t intervals with two degrees of freedom. Paired differences are relative to l2=1e-6 on identical training and query clouds.",
        ),
        (
            "manufactured_refinement", tables / "ManufacturedRefinement.tex",
            (("l2_regularization", r"$\ell_2$"),
             ("query_type", "Query"), ("quantity", "Quantity"),
             ("metric", "Metric"),
             ("N300_seed_mean", r"$N=300$"),
             ("N600_seed_mean", r"$N=600$"),
             ("N1200_seed_mean", r"$N=1200$"),
             ("N2400_seed_mean", r"$N=2400$"),
             ("percent_change_N300_to_N600", r"$\%\Delta_{300,600}$"),
             ("percent_change_N600_to_N1200", r"$\%\Delta_{600,1200}$"),
             ("percent_change_N1200_to_N2400", r"$\%\Delta_{1200,2400}$"),
             ("Nmax_better_than_Nmin", r"$E_{2400}<E_{300}$?"),
             ("monotone_decrease", "Monotone?"),
             ("refinement_verdict", "Verdict")),
            "Manufactured density, gradient, and operator independent-cloud enlargement results.",
            "tab:manufactured-refinement",
            "This is a descriptive nonnested enlargement check, not a deterministic support-convergence test. 'Monotone decrease' means all four three-seed mean errors decrease strictly with N; no absolute physical-accuracy threshold was declared.",
        ),
        (
            "timestep_run", tables / "TimeStepRunByRun.tex",
            (("method", "Method"), ("P0", r"$P_{\mathrm{init}}$"), ("seed", "Seed"),
             ("observable", "Observable"), ("value1", r"$O_h(T)$"),
             ("value2", r"$O_{h/2}(T)$"), ("value3", r"$O_{h/4}(T)$"),
             ("D12", r"$D_{12}$"), ("D23", r"$D_{23}$"),
             ("D12_over_D23", r"$D_{12}/D_{23}$"),
             ("paired_difference_D12_minus_D23", r"$D_{12}-D_{23}$"),
             ("paired_contraction", "Contracts?"),
             ("roundoff_threshold", r"$\tau_{\rm noise}$"),
             ("physically_admissible_case", "Physical?"),
             ("verdict", "Verdict")),
            "Run-by-run paired three-level time-step evidence with reconstructible endpoint values.",
            "tab:timestep-run-by-run",
            "N=1000; seeds 11,29,47,73. Differences are within-seed time-normalized L2 values on common saved times without extrapolation. The decision hierarchy is numerical floor, finite output, endpoint physical admissibility, and only then paired contraction. Raw cross-seed observable spread is retained in the source CSV as a descriptive cloud-variability diagnostic and is not an order gate. Rejected orders remain visible.",
        ),
        (
            "support_summary", tables / "IndependentCloudSupport.tex",
            (("method", "Method"), ("P0", r"$P_{\mathrm{init}}$"), ("observable", "Observable"),
             ("N", r"$N$"), ("n_seeds", r"$n$"), ("mean", "Mean"),
             ("sample_sd", "SD"), ("standard_error", "SE"),
             ("ci95_low", r"95\% low"), ("ci95_high", r"95\% high"),
             ("difference_from_N2000_mean", r"$\Delta\bar O_{2000}$"),
             ("change_exceeds_seed_variability", "Exceeds SD?")),
            "Independent-cloud support enlargement.",
            "tab:independent-cloud-support",
            "Clouds are independently sampled across N and are not nested. Displayed observables are signed-mass-normalized estimators; raw conservation is reported separately. This is not deterministic support convergence and no support order is reported.",
        ),
        (
            "replication_summary", tables / "FourSeedReplication.tex",
            (("method", "Method"), ("P0", r"$P_{\mathrm{init}}$"), ("observable", "Observable"),
             ("n_independent_seeds", r"$n$"), ("mean", "Mean"),
             ("sample_sd", "SD"), ("standard_error", "SE"),
             ("ci95_low", r"95\% low"), ("ci95_high", r"95\% high"),
             ("maximum_spread", "Max spread"),
             ("coefficient_of_variation", "CV")),
            "Four-seed replication endpoint statistics.",
            "tab:four-seed-replication",
            "The independent seeds, not trajectories, define n=4. Displayed observables are signed-mass-normalized estimators; raw conservation is reported separately. These small-sample intervals are not strong uncertainty calibration.",
        ),
        (
            "tail_distribution_paired", tables / "InitialLabelDistribution.tex",
            (("P0", r"$P_{\mathrm{init}}$"), ("seed", "Seed"),
             ("N", r"$N$"), ("minimum", "Minimum"),
             ("q0.001", r"$q_{0.001}$"), ("q0.01", r"$q_{0.01}$"),
             ("q0.05", r"$q_{0.05}$"), ("q0.5", "Median"),
             ("q0.95", r"$q_{0.95}$"), ("q0.99", r"$q_{0.99}$"),
             ("q0.999", r"$q_{0.999}$"), ("maximum", "Maximum")),
            r"Per-seed distribution of the absolute initial labels $|y_i^0|$ shared by the paired PBME and MIDPOINT calculations.",
            "tab:y0-initial-distribution",
            "The inclusion set at threshold eta is M_eta={i: |y_i^0| > eta max_j |y_j^0|}. PBME and MIDPOINT use identical initial labels within each momentum/seed pair, verified by the stored label and cloud hashes.",
        ),
        (
            "tail_sweep", tables / "TailSensitivity.tex",
            (("P0", r"$P_{\mathrm{init}}$"), ("method", "Method"), ("seed", "Seed"),
             ("eta", r"$\eta$"), ("included_fraction", "Included"),
             ("excluded_absolute_physical_mass_fraction", "Excluded abs. mass"),
             ("minimum_retained_abs_y0", r"min $|y_i^0|$"),
             ("ratio_abs_max", r"max $|y_i/y_i^0|$"),
             ("signed_ESS", "signed ESS"), ("absolute_ESS", "abs. ESS"),
             ("raw_normalization", r"$N_{\rm raw}$"),
             ("raw_energy", r"$E_{\rm raw}$"),
             ("cloud_weighted_P0_change_from_eta0", r"$\Delta \rho_{11}^{\rm raw}$"),
             ("cloud_weighted_P1_change_from_eta0", r"$\Delta \rho_{22}^{\rm raw}$")),
            r"Post-processing sensitivity to the $|y_i^0|$ inclusion threshold.",
            "tab:y0-tail-sensitivity",
            "The threshold sweep does not alter propagation. The inclusion set is M_eta={i: |y_i^0| > eta max_j |y_j^0|}; Included is the retained point fraction and Excluded abs. mass is the removed fraction of sum_i |omega_i y_i^0|. Raw quantities are not self-normalized.",
        ),
        (
            "reference_settings", tables / "ReferenceSettingsByMomentum.tex",
            (("method", "Method"), ("P0", r"$P_{\mathrm{init}}$"),
             ("mode", "Mode"), ("t_final", r"$t_{\mathrm{final}}$"),
             ("dt_ladder", r"$\Delta t$ ladder"),
             ("grid_ladder", "Grid ladder"),
             ("R_domain_ladder", r"$R$ domains"),
             ("P_domain_ladder", r"$P$ domains"),
             ("finest_physical_edge_fraction", "Finest edge fraction"),
             ("finest_CFL_ratio", "Finest CFL"),
             ("interpretation", "Interpretation")),
            "Reference settings split by equation, initial momentum, and refinement mode.",
            "tab:reference-discretizations",
            "Slash-separated entries are coarse/fine/finer. For grid QCLE, the reported edge value is the larger of the finest-grid physical R and P marginal edge fractions. The sign-indefinite phase-space absolute-W ringing diagnostic is not substituted for this boundary-mass control. The dimensionless CFL entry is Delta t/Delta t_max, where Delta t_max is the minimum of the RK4 imaginary-axis bound 2.828 divided by the spectral advection, force, and electronic-frequency scales; values below one satisfy this declared linear-stability check.",
        ),
        (
            "tdse", tables / "TDSEReferenceOrders.tex",
            (("P0", r"$P_{\mathrm{init}}$"), ("refinement_mode", "Mode"),
             ("observable", "Observable"), ("value1", r"$O_1$"),
             ("value2", r"$O_2$"), ("value3", r"$O_3$"),
             ("delta12", r"$\delta_{12}$"), ("delta23", r"$\delta_{23}$"),
             ("numerical_noise_threshold", r"$\tau_{\rm noise}$"),
             ("p_observed", r"$p_{\rm obs}$"),
             ("order_reason", "Verdict")),
            "TDSE separate time and grid three-level verification.",
            "tab:tdse-reference-orders",
            "TDSE is model-exact only within the numerical controls tabulated in Appendix F. Space and time are refined separately. Both differences must exceed tau_noise=1e-12+1e-12 max_k |O_k| before an order is interpreted; rejected and nonmonotone rows remain visible.",
        ),
        (
            "qcle", tables / "QCLEReferenceOrders.tex",
            (("P0", r"$P_{\mathrm{init}}$"), ("refinement_mode", "Mode"),
             ("observable", "Observable"), ("value1", r"$O_1$"),
             ("value2", r"$O_2$"), ("value3", r"$O_3$"),
             ("delta12", r"$\delta_{12}$"), ("delta23", r"$\delta_{23}$"),
             ("numerical_noise_threshold", r"$\tau_{\rm noise}$"),
             ("p_observed", r"$p_{\rm obs}$"),
             ("order_reason", "Verdict")),
            "Grid-QCLE separate time and grid three-level verification.",
            "tab:qcle-reference-orders",
            "Grid QCLE is a numerical solution of the approximate QCLE. Time and phase-space grids are refined separately. Both differences must exceed tau_noise=1e-12+1e-12 max_k |O_k| before an order is interpreted; exact domains, steps, edge masses, and CFL ratios are tabulated in Appendix F.",
        ),
        (
            "physical_density_per_seed", tables / "DensityReferenceErrorsPerSeed.tex",
            (("reference", "Reference"), ("P0", r"$P_{\mathrm{init}}$"),
             ("seed", "Seed"), ("quantity", "Field"),
             ("PBME_raw_E1", r"PBME raw $E_1$"),
             ("PBME_raw_E2", r"PBME raw $E_2$"),
             ("MIDPOINT_raw_E1", r"MIDPOINT raw $E_1$"),
             ("MIDPOINT_raw_E2", r"MIDPOINT raw $E_2$"),
             ("PBME_shape_E1", r"PBME shape $E_1$"),
             ("PBME_shape_E2", r"PBME shape $E_2$"),
             ("MIDPOINT_shape_E1", r"MIDPOINT shape $E_1$"),
             ("MIDPOINT_shape_E2", r"MIDPOINT shape $E_2$")),
            "Absolute per-seed PBME and MIDPOINT field errors against the common physical references.",
            "tab:density-reference-errors-per-seed",
            "Raw errors retain each field integral. Shape errors divide each signed field by its own nonzero signed integral before comparison and therefore cannot be used as conservation evidence. E1 and E2 are relative L1 and L2 field errors.",
        ),
        (
            "physical_observable_per_seed", tables / "ObservableReferenceErrorsPerSeed.tex",
            (("reference", "Reference"), ("P0", r"$P_{\mathrm{init}}$"),
             ("seed", "Seed"), ("observable", "Observable"),
             ("PBME_E_RMS", r"PBME $E_{\mathrm{RMS}}$"),
             ("MIDPOINT_E_RMS", r"MIDPOINT $E_{\mathrm{RMS}}$"),
             ("MIDPOINT_minus_PBME_E_RMS", r"$\Delta E_{\mathrm{RMS}}$"),
             ("PBME_E_infinity", r"PBME $E_\infty$"),
             ("MIDPOINT_E_infinity", r"MIDPOINT $E_\infty$"),
             ("MIDPOINT_minus_PBME_E_infinity", r"$\Delta E_\infty$")),
            "Absolute per-seed population time-series errors and paired MIDPOINT-minus-PBME differences.",
            "tab:observable-reference-errors-per-seed",
            "PBME and MIDPOINT use the same initial cloud for every momentum/seed pair. Population curves are signed-mass-normalized estimators; raw normalization and trace are tested separately. Negative paired differences favor MIDPOINT for that seed and metric.",
        ),
        (
            "physical_summary", tables / "PhysicalReferenceComparison.tex",
            (("comparison_kind", "Kind"), ("reference", "Reference"),
             ("P0", r"$P_{\mathrm{init}}$"),
             ("observable", "Observable"), ("metric", "Metric"),
             ("n_paired_seeds", r"$n$"),
             ("mean_paired_difference", r"$\overline{\Delta E}$"),
             ("sample_sd", "SD"), ("standard_error", "SE"),
             ("ci95_low", r"95\% low"), ("ci95_high", r"95\% high"),
             ("verdict_before_scientific_gates", "Verdict")),
            "Paired MIDPOINT-minus-PBME errors against common physical references.",
            "tab:physical-reference-comparison",
            "Negative differences favor MIDPOINT; positive differences favor PBME. A paired interval alone does not override stability, conservation, or correction-magnitude gates.",
        ),
    ]
    for key, tex_path, columns, caption, label, note in table_specs:
        csv_path = paths[key]
        writer = (
            fitted_landscape_table_from_csv
            if key == "reference_settings"
            else longtable_from_csv
        )
        writer(csv_path, tex_path, columns, caption, label, note)
        crosswalk.append({
            "table": str(tex_path), "source_csv": str(csv_path),
            "source_csv_sha256": sha256_file(csv_path),
        })

    preserved_specs = [
        (
            "raw_conservation.csv", tables / "RawConservation.tex",
            (("method", "Method"), ("P0", r"$P_{\mathrm{init}}$"), ("seed", "Seed"),
             ("quantity", "Quantity"), ("endpoint_drift", "Endpoint drift"),
             ("maximum_absolute_drift", "Max abs."), ("rms_drift", "RMS")),
            "Raw pre-renormalization conservation audit.",
            "tab:raw-conservation",
            "Self-normalized observables are not substituted for raw normalization, trace, or energy.",
        ),
        (
            "projection_leakage.csv", tables / "ProjectionLeakage.tex",
            (("P0", r"$P_{\mathrm{init}}$"), ("propagation_seed", "Seed"), ("method", "Method"),
             ("snapshot_step", "Step"), ("physical_time", "Time"),
             ("mean_relative_l2_leakage", "Mean"),
             ("median_relative_l2_leakage", "Median"),
             ("maximum_relative_l2_leakage", "Maximum"),
             ("sample_sd_relative_l2_leakage", "SD")),
            "SEO projection-leakage diagnostics.",
            "tab:seo-projection-leakage",
            "This is diagnostic projection of the fitted surrogate, not enforced projection and not a physical-error estimate.",
        ),
        (
            "kde_gp_baseline.csv", tables / "IdenticalSupportKDEGP.tex",
            (("P0", r"$P_{\mathrm{init}}$"), ("n_cases", r"$n$"),
             ("max_E1", r"max $E_1$"), ("max_E2", r"max $E_2$"),
             ("max_Einf", r"max $E_\infty$"),
             ("threshold_E1", r"$E_1$ gate"), ("result", "Result")),
            "Identical-support PBME KDE/GP reconstruction baseline.",
            "tab:identical-support-kde-gp",
            "The same support, measure, bandwidth convention, grid, normalization, physical mass, and snapshot are required. The predeclared gate is E1 <= 0.02.",
        ),
    ]
    for name, tex_path, columns, caption, label, note in preserved_specs:
        csv_path = preserved[name]
        longtable_from_csv(csv_path, tex_path, columns, caption, label, note)
        crosswalk.append({
            "table": str(tex_path), "source_csv": str(csv_path),
            "source_csv_sha256": sha256_file(csv_path),
            "preserved": True,
        })

    for row in crosswalk:
        labels = re.findall(
            r"\\label\{([^}]+)\}",
            Path(row["table"]).read_text(encoding="utf-8"),
        )
        if not labels:
            raise RuntimeError(f"generated table has no manifest label: {row['table']}")
        row["artifact_ids"] = ";".join(labels)
    crosswalk_path = write_csv(out / "table_data_crosswalk.csv", crosswalk)
    # The release specification names the uppercase form explicitly.  Keep the
    # lowercase compatibility copy used by the local checker byte-equivalent.
    uppercase_crosswalk = out / "TABLE_DATA_CROSSWALK.csv"
    copy_case_compatible(crosswalk_path, uppercase_crosswalk)
    for row in crosswalk:
        row["crosswalk"] = str(uppercase_crosswalk)
    return crosswalk


def analyze(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    plan = build_plan(args)
    missing = [job for job in plan["jobs"] if job["status"] == "MISSING"]
    if missing:
        print(
            f"[analyze] refused: {len(missing)} required calculation(s) are "
            "missing or invalid. Run --mode execute --resume first.",
            file=sys.stderr,
        )
        write_json(out / "FINAL_RUN_MANIFEST.json", plan)
        return 2
    paths: Dict[str, Path] = {}
    manufactured = analyze_manufactured(out)
    paths["manufactured_complete"] = manufactured["complete"]
    paths["manufactured_summary"] = manufactured["summary"]
    paths["manufactured_comparison"] = manufactured["comparison"]
    paths["manufactured_refinement"] = manufactured["refinement"]
    paths["manufactured_sampling_geometry"] = manufactured["sampling_geometry"]
    mint_controls = analyze_mint_controls(out)
    paths["mint_controls"] = mint_controls["controls"]
    paths["mint_controls_manifest"] = mint_controls["manifest"]
    timestep = analyze_timestep(out, args)
    paths["timestep_run"] = timestep["run_by_run"]
    paths["timestep_summary"] = timestep["summary"]
    paths["timestep_paired_summary"] = timestep["paired_summary"]
    paths["timestep_paired_by_seed"] = timestep["paired_by_seed"]
    support = analyze_support_and_replication(out, args)
    paths["support_run"] = support["support_run"]
    paths["support_summary"] = support["support_summary"]
    paths["replication_values"] = support["replication_values"]
    paths["replication_summary"] = support["replication_summary"]
    tail = analyze_tail(out, args)
    paths["tail_distribution"] = tail["distribution"]
    paths["tail_sweep"] = tail["sweep"]
    paths["tail_summary"] = tail["summary"]
    references = analyze_references(out, args)
    paths["tdse"] = references["tdse"]
    paths["qcle"] = references["qcle"]
    physical = analyze_physical_comparison(out, args)
    paths["physical_observable"] = physical["observable"]
    paths["physical_density"] = physical["density"]
    paths["physical_summary"] = physical["summary"]
    build_reader_facing_closure_outputs(out, paths)
    preserved = preserve_existing_evidence(out)
    crosswalk = generate_tables(out, paths, preserved, plan)
    write_json(out / "analysis_manifest.json", {
        "created_utc": utcnow(),
        "outputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
        "preserved_outputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in preserved.items()
        },
        "table_crosswalk_rows": crosswalk,
    })
    print(f"[analyze] complete: {len(paths)} quantitative CSV outputs", flush=True)
    return 0


def csv_finite(path: Path, numeric_columns: Sequence[str]) -> Tuple[bool, str]:
    rows = read_csv_rows(path)
    for index, row in enumerate(rows, 2):
        for column in numeric_columns:
            try:
                value = float(row[column])
            except (KeyError, TypeError, ValueError):
                return False, f"{path}:{index}: missing/non-numeric {column}"
            if not np.isfinite(value):
                return False, f"{path}:{index}: nonfinite {column}"
    return True, f"{len(rows)} rows finite"


def verify(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    checks: List[Dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append({
            "name": name, "passed": bool(condition), "detail": detail
        })

    plan = build_plan(args)
    missing = [job for job in plan["jobs"] if job["status"] == "MISSING"]
    check("all_required_jobs_complete", not missing,
          f"{len(missing)} missing/invalid jobs")

    manufactured = out / "manufactured" / "manufactured_complete.csv"
    if manufactured.exists():
        rows = read_csv_rows(manufactured)
        check("manufactured_row_count", len(rows) == 72,
              f"{len(rows)} rows; expected 72")
        ok, detail = csv_finite(manufactured, [
            f"{prefix}_{metric}"
            for prefix in ("density", "gradient", "Q")
            for metric in (
                "relative_l1", "relative_l2", "relative_linf",
                "mae", "rmse", "linf",
            )
        ])
        check("manufactured_metrics_finite", ok, detail)
        paired_ok = True
        by_case: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
        for row in rows:
            by_case.setdefault(
                (row["N"], row["seed"], row["query_type"]), []
            ).append(row)
        for case_rows in by_case.values():
            paired_ok &= len({r["training_cloud_sha256"] for r in case_rows}) == 1
            paired_ok &= len({r["query_sha256"] for r in case_rows}) == 1
        check("manufactured_policy_pairing", paired_ok,
              "training/query hashes identical across l2 at fixed N, seed and query")
    else:
        check("manufactured_row_count", False, f"absent: {manufactured}")

    expected_counts = {
        out / "manufactured" / "manufactured_summary.csv": 432,
        out / "manufactured" / "manufactured_policy_comparison.csv": 864,
        out / "timestep" / "timestep_run_by_run.csv": 128,
        out / "timestep" / "timestep_paired_differences_by_seed.csv": 128,
        out / "timestep" / "timestep_paired_summary.csv": 32,
        out / "implementation_controls" / "mint_implementation_controls.csv": 4,
        out / "support" / "independent_cloud_run_by_run.csv": 288,
        out / "replication" / "four_seed_values.csv": 128,
        out / "tail_sensitivity" / "threshold_sweep.csv": 160,
        out / "reference_tdse" / "tdse_three_level.csv": 32,
        out / "reference_grid_qcle" / "qcle_three_level.csv": 32,
        out / "physical_comparison" / "observable_errors_by_seed.csv": 256,
        out / "physical_comparison" / "density_errors_by_seed.csv": 32,
        out / "physical_comparison" / "observable_errors_method_pair_by_seed.csv": 32,
        out / "physical_comparison" / "density_errors_method_pair_by_seed.csv": 16,
        out / "tail_sensitivity" / "y0_distribution_paired.csv": 8,
        out / "reference_settings_by_method_and_momentum.csv": 8,
    }
    for path, expected in expected_counts.items():
        if not path.exists():
            check(f"rows_{path.stem}", False, f"absent: {path}")
            continue
        actual = len(read_csv_rows(path))
        check(f"rows_{path.stem}", actual == expected,
              f"{actual} rows; expected {expected}")

    manufactured_path = out / "manufactured" / "manufactured_complete.csv"
    if manufactured_path.exists():
        rows = read_csv_rows(manufactured_path)
        actual = {
            (float(row["l2_regularization"]), int(row["N"]),
             int(row["seed"]), row["query_type"])
            for row in rows
        }
        expected = {
            (float(l2), int(N), int(seed), query)
            for l2 in args.manufactured_l2
            for N in args.manufactured_N
            for seed in args.manufactured_seeds
            for query in ("on_support", "off_support")
        }
        check(
            "manufactured_every_seed_cell_present_once",
            len(rows) == len(actual) and actual == expected,
            f"unique cells={len(actual)}; expected={len(expected)}",
        )

    timestep_path = out / "timestep" / "timestep_run_by_run.csv"
    if timestep_path.exists():
        rows = read_csv_rows(timestep_path)
        actual = {
            (row["method"], float(row["P0"]), int(row["seed"]), row["observable"])
            for row in rows
        }
        expected = {
            (method, float(P0), int(seed), observable)
            for method in ("PBME", "MIDPOINT")
            for P0 in args.P0
            for seed in args.dynamics_seeds
            for observable in OBSERVABLES
        }
        check(
            "timestep_every_method_momentum_seed_observable_present_once",
            len(rows) == len(actual) and actual == expected,
            f"unique cells={len(actual)}; expected={len(expected)}",
        )

    density_pair_path = (
        out / "physical_comparison" / "density_errors_method_pair_by_seed.csv"
    )
    if density_pair_path.exists():
        rows = read_csv_rows(density_pair_path)
        ok, detail = csv_finite(density_pair_path, [
            f"{method}_{kind}_E{norm}"
            for method in ("PBME", "MIDPOINT")
            for kind in ("raw", "shape")
            for norm in ("1", "2")
        ])
        expected = {
            (reference, float(P0), int(seed))
            for reference in ("TDSE", "QCLE")
            for P0 in args.P0
            for seed in args.dynamics_seeds
        }
        actual = {
            (row["reference"], float(row["P0"]), int(row["seed"]))
            for row in rows
        }
        check(
            "density_absolute_L1_L2_every_seed",
            ok and len(rows) == len(actual) and actual == expected,
            detail + f"; unique cells={len(actual)}",
        )

    observable_pair_path = (
        out / "physical_comparison" / "observable_errors_method_pair_by_seed.csv"
    )
    if observable_pair_path.exists():
        rows = read_csv_rows(observable_pair_path)
        ok, detail = csv_finite(observable_pair_path, [
            "PBME_E_RMS", "MIDPOINT_E_RMS", "MIDPOINT_minus_PBME_E_RMS",
            "PBME_E_infinity", "MIDPOINT_E_infinity",
            "MIDPOINT_minus_PBME_E_infinity",
        ])
        expected = {
            (reference, float(P0), int(seed), observable)
            for reference in ("TDSE", "QCLE")
            for P0 in args.P0
            for seed in args.dynamics_seeds
            for observable in ("P0", "P1")
        }
        actual = {
            (row["reference"], float(row["P0"]), int(row["seed"]), row["observable"])
            for row in rows
        }
        check(
            "observable_absolute_errors_every_seed",
            ok and len(rows) == len(actual) and actual == expected,
            detail + f"; unique cells={len(actual)}",
        )

    distribution_path = out / "tail_sensitivity" / "y0_distribution_paired.csv"
    if distribution_path.exists():
        rows = read_csv_rows(distribution_path)
        quantiles = [
            "minimum", "q0.001", "q0.01", "q0.05", "q0.1", "q0.25",
            "q0.5", "q0.75", "q0.9", "q0.95", "q0.99", "q0.999",
            "maximum",
        ]
        finite, detail = csv_finite(distribution_path, quantiles)
        ordered = all(
            all(float(row[left]) <= float(row[right])
                for left, right in zip(quantiles, quantiles[1:]))
            for row in rows
        )
        check(
            "initial_label_distribution_complete_and_ordered",
            finite and ordered,
            detail + f"; ordered={ordered}",
        )

    threshold_path = out / "tail_sensitivity" / "threshold_sweep.csv"
    if threshold_path.exists() and distribution_path.exists():
        maxima = {
            (float(row["P0"]), int(row["seed"])): float(row["maximum"])
            for row in read_csv_rows(distribution_path)
        }
        rows = read_csv_rows(threshold_path)
        rule_ok = all(
            math.isclose(
                float(row["threshold"]),
                float(row["eta"]) * maxima[(float(row["P0"]), int(row["seed"]))],
                rel_tol=1e-12,
                abs_tol=1e-18,
            )
            and int(row["included_count"]) + int(row["excluded_count"]) == 1000
            for row in rows
        )
        check(
            "tail_inclusion_rule_and_affected_fraction_reconstructible",
            rule_ok,
            "threshold=eta*max|y0| and included+excluded=N for every row",
        )

    settings_path = out / "reference_settings_by_method_and_momentum.csv"
    if settings_path.exists():
        rows = read_csv_rows(settings_path)
        keyed = {
            (row["method"], float(row["P0"]), row["mode"]): row
            for row in rows
        }
        expected_keys = {
            (method, float(P0), mode)
            for method in ("TDSE", "grid QCLE")
            for P0 in args.P0
            for mode in ("time", "grid")
        }
        qcle_domains_ok = all(
            (
                "[-35,35]" in row["P_domain_ladder"]
                if float(P0) == 20.0 else "[80,120]" in row["P_domain_ladder"]
            )
            for (method, P0, _), row in keyed.items()
            if method == "grid QCLE"
        )
        tdse_high_grid = keyed.get(("TDSE", 100.0, "grid"), {})
        tdse_high_ok = (
            tdse_high_grid.get("grid_ladder") == "8192/8192/8192"
            and "no independent grid order" in tdse_high_grid.get("interpretation", "")
        )
        check(
            "reference_settings_split_and_consistent",
            set(keyed) == expected_keys and qcle_domains_ok and tdse_high_ok,
            (
                f"cells={len(keyed)}; qcle_domains_ok={qcle_domains_ok}; "
                f"high_momentum_tdse_grid_ok={tdse_high_ok}"
            ),
        )

    manufactured_summary = out / "manufactured" / "manufactured_summary.csv"
    if manufactured_summary.exists():
        rows = read_csv_rows(manufactured_summary)
        ok, detail = csv_finite(manufactured_summary, [
            "mean", "sample_sd", "standard_error", "ci95_low", "ci95_high",
            "mean_paired_difference_from_baseline",
            "paired_difference_sample_sd",
            "paired_difference_standard_error",
            "paired_difference_ci95_low",
            "paired_difference_ci95_high",
        ])
        check("manufactured_summary_policy_statistics_finite", ok, detail)
        check(
            "manufactured_summary_policy_cloud_pairing",
            all(
                row.get("paired_training_and_query_clouds_verified", "").lower()
                == "true"
                for row in rows
            ),
            "all regularization comparisons use identical training/query clouds",
        )

    timestep = out / "timestep" / "timestep_run_by_run.csv"
    if timestep.exists():
        rows = read_csv_rows(timestep)
        forbidden = {"MISSING_RUN", "NONFINITE_RUN"}
        bad = [row for row in rows if row["verdict"] in forbidden]
        check("timestep_no_missing_or_nonfinite_verdict", not bad,
              f"{len(bad)} forbidden verdicts")
        required = (
            "value1", "value2", "value3", "D12", "D23",
            "D12_over_D23", "paired_difference_D12_minus_D23",
            "raw_observable_seed_spread", "roundoff_threshold",
        )
        ok, detail = csv_finite(timestep, required)
        check("timestep_reconstruction_fields_finite", ok, detail)
        check(
            "timestep_raw_seed_spread_not_used_as_gate",
            all(
                row.get("raw_seed_spread_role", "").startswith("descriptive")
                and row.get("verdict") != "REJECT_SEED_VARIABILITY"
                for row in rows
            ),
            "raw cross-seed spread is descriptive only; legacy seed-spread veto absent",
        )

    paired_timestep = out / "timestep" / "timestep_paired_summary.csv"
    if paired_timestep.exists():
        rows = read_csv_rows(paired_timestep)
        ok, detail = csv_finite(paired_timestep, (
            "mean_D12", "sample_sd_D12", "mean_D23", "sample_sd_D23",
            "mean_paired_difference_D12_minus_D23",
            "paired_difference_sample_sd", "paired_difference_standard_error",
            "paired_difference_ci95_low", "paired_difference_ci95_high",
            "seed_11_D12_over_D23", "seed_29_D12_over_D23",
            "seed_47_D12_over_D23", "seed_73_D12_over_D23",
        ))
        check("timestep_paired_statistics_finite", ok, detail)
        check(
            "timestep_paired_hierarchy_reported",
            all(row.get("final_verdict", "") for row in rows),
            "all 32 summaries report the numerical/finite/physical/paired hierarchy",
        )

    geometry = out / "manufactured" / "manufactured_sampling_geometry.json"
    if geometry.exists():
        record = read_json(geometry)
        check(
            "manufactured_geometry_exact",
            record.get("dimension") == 6
            and record.get("focused_mapping_shell") is False
            and record.get("mapping_coordinates_independent") is True,
            "fully dimensional independent Gaussian geometry recorded; no focused shell",
        )
    else:
        check("manufactured_geometry_exact", False, f"absent: {geometry}")

    mint_controls = (
        out / "implementation_controls" / "mint_implementation_controls.csv"
    )
    if mint_controls.exists():
        rows = read_csv_rows(mint_controls)
        ok, detail = csv_finite(mint_controls, ("value", "tolerance"))
        check("mint_control_values_finite", ok, detail)
        check(
            "mint_implementation_controls_pass",
            len(rows) == 4
            and all(row.get("status") == "PASS" for row in rows)
            and all(float(row["value"]) < float(row["tolerance"]) for row in rows),
            "symplectic, round-trip, mapping-radius and energy controls pass",
        )

    support = out / "support" / "independent_cloud_run_by_run.csv"
    if support.exists():
        rows = read_csv_rows(support)
        check("support_declared_independent", all(
            "not nested" in row["cloud_relationship"] for row in rows
        ), "all support rows explicitly identify independent clouds")

    tail = out / "tail_sensitivity" / "threshold_sweep.csv"
    if tail.exists():
        rows = read_csv_rows(tail)
        check("tail_masks_hashed", all(
            len(row.get("point_inclusion_mask_sha256", "")) == 64 for row in rows
        ), "every threshold mask has SHA-256")
        check("tail_raw_not_self_normalized", all(
            row.get("raw_normalization", "") != "" and row.get("raw_energy", "") != ""
            for row in rows
        ), "raw normalization and energy present at every threshold")

    for method, name in (("tdse", "tdse"), ("qcle", "grid_qcle")):
        path = out / (
            "reference_tdse" if method == "tdse" else "reference_grid_qcle"
        ) / (f"{method}_three_level.csv")
        if path.exists():
            rows = read_csv_rows(path)
            modes = {(row["P0"], row["refinement_mode"]) for row in rows}
            check(
                f"{name}_separate_time_grid",
                reference_modes_complete(rows, args.P0),
                f"modes found: {sorted(modes)}",
            )
            ok, detail = csv_finite(path, [
                "value1", "value2", "value3", "delta12", "delta23"
            ])
            check(f"{name}_values_differences_finite", ok, detail)
            setting_fields = [
                f"level{level}_{field}"
                for level in (1, 2, 3)
                for field in (
                    ("R_min", "R_max", "dR", "n_grid_actual")
                    if method == "tdse"
                    else (
                        "R_min", "R_max", "P_min", "P_max",
                        "dR", "dP", "n_R", "n_P",
                    )
                )
            ]
            settings_ok, settings_detail = csv_finite(path, setting_fields)
            check(
                f"{name}_exact_grid_settings_finite",
                settings_ok,
                settings_detail,
            )
            edge_columns = (
                ("maximum_edge_mass_5pct",)
                if method == "tdse"
                else (
                    "maximum_edge_R_mass_5pct",
                    "maximum_edge_P_mass_5pct",
                )
            )
            edge_values: List[float] = []
            edge_ok = True
            # Coarse QCLE grids are intentionally included to display the
            # onset of spatial convergence.  Domain adequacy is asserted for
            # the finest reference level; all coarser edge diagnostics remain
            # visible in the CSV and generated table.  TDSE retains the more
            # stringent all-level check because its density is nonnegative.
            checked_levels = (1, 2, 3) if method == "tdse" else (3,)
            for row in rows:
                try:
                    tolerance = float(row["declared_edge_mass_tolerance"])
                    for level in checked_levels:
                        for edge_name in edge_columns:
                            value = float(row[f"level{level}_{edge_name}"])
                            edge_values.append(value)
                            edge_ok &= np.isfinite(value) and value <= tolerance
                except (KeyError, TypeError, ValueError):
                    edge_ok = False
            check(
                f"{name}_edge_mass_within_declared_tolerance",
                edge_ok,
                (
                    f"maximum={max(edge_values) if edge_values else 'absent'}; "
                    f"levels checked={checked_levels}; declared tolerance "
                    "recorded per row"
                ),
            )
            if method == "tdse":
                reflected_rows = [
                    row for row in rows
                    if equal_number(row.get("P0"), 100.0)
                    and row.get("refinement_mode") == "time"
                ]
                try:
                    reflected = max(
                        float(row[
                            "level3_maximum_negative_momentum_probability"
                        ])
                        for row in reflected_rows
                    )
                    reflected_tolerance = min(
                        float(row["declared_negative_momentum_tolerance"])
                        for row in reflected_rows
                    )
                    reflection_ok = (
                        np.isfinite(reflected)
                        and reflected <= reflected_tolerance
                    )
                except (KeyError, TypeError, ValueError):
                    reflected = math.nan
                    reflected_tolerance = math.nan
                    reflection_ok = False
                check(
                    "high_momentum_positive_qcle_box_admissible",
                    reflection_ok,
                    (
                        "finest TDSE maximum negative-momentum probability="
                        f"{reflected}; tolerance={reflected_tolerance}"
                    ),
                )

    crosswalk = out / "table_data_crosswalk.csv"
    if crosswalk.exists():
        rows = read_csv_rows(crosswalk)
        table_ids = [
            value
            for row in rows
            for value in row.get("artifact_ids", "").split(";")
            if value
        ]
        all_present = all(
            Path(row["table"]).exists()
            and Path(row["source_csv"]).exists()
            and sha256_file(Path(row["source_csv"])) == row["source_csv_sha256"]
            for row in rows
        ) and bool(table_ids) and len(table_ids) == len(set(table_ids))
        check("table_crosswalk_complete", all_present,
              f"{len(rows)} table files and {len(table_ids)} unique artifact IDs traced to hash-verified CSVs")
    else:
        check("table_crosswalk_complete", False, f"absent: {crosswalk}")

    figure_crosswalk = out / "figures" / "FIGURE_DATA_CROSSWALK.csv"
    figure_rows: list[dict[str, str]] = []
    if figure_crosswalk.exists():
        figure_rows = read_csv_rows(figure_crosswalk)
        figure_ids = [row.get("artifact_id", "") for row in figure_rows]
        figure_ok = (
            len(figure_rows) == 5
            and all(figure_ids)
            and len(figure_ids) == len(set(figure_ids))
        )
        for row in figure_rows:
            figure_path = REPO / row["figure"]
            sources = [REPO / value for value in row["source_csvs"].split(";")]
            source_hashes = row["source_csv_sha256"].split(";")
            figure_ok = (
                figure_ok
                and figure_path.exists()
                and sha256_file(figure_path) == row["figure_sha256"]
                and len(sources) == len(source_hashes)
                and all(
                    path.exists() and sha256_file(path) == digest
                    for path, digest in zip(sources, source_hashes)
                )
            )
        check(
            "figure_crosswalk_complete",
            figure_ok,
            f"{len(figure_rows)} figures traced to hash-verified CSVs",
        )
    else:
        check("figure_crosswalk_complete", False, f"absent: {figure_crosswalk}")

    thesis = REPO / "Thesis" / "Thesis.tex"
    if thesis.exists():
        text = thesis.read_text(encoding="utf-8", errors="replace")
        included = re.findall(
            r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text
        )
        expected = {"../" + row["figure"] for row in figure_rows}
        check(
            "thesis_verified_figures_only",
            len(included) == 5 and set(included) == expected,
            f"included={included}",
        )
    else:
        check("thesis_source_present", False, f"absent: {thesis}")

    passed = all(item["passed"] for item in checks)
    result = {
        "created_utc": utcnow(), "status": "PASSED" if passed else "FAILED",
        "checks": checks,
        "public_release_record": "final_reviewer_closure/PUBLIC_RELEASE.json",
        "persistent_identifier_note": (
            "The versioned GitHub release is a public retrieval record, not a "
            "DOI or institutional persistent identifier; no DOI is invented."
        ),
    }
    write_json(out / "verification_result.json", result)
    write_json(out / "FINAL_ACCEPTANCE.json", result)
    marker = out / "VERIFY_PASSED.json"
    if passed:
        write_json(marker, result)
    elif marker.exists():
        marker.unlink()
    for item in checks:
        print(
            f"[verify] {'PASS' if item['passed'] else 'FAIL'} "
            f"{item['name']}: {item['detail']}"
        )
    return 0 if passed else 2


def copy_file_with_parents(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def package(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    marker = out / "VERIFY_PASSED.json"
    if not marker.exists() or read_json(marker).get("status") != "PASSED":
        print(
            "[package] refused: --mode verify has not passed.",
            file=sys.stderr,
        )
        return 2
    required = {
        "thesis": args.thesis, "bibliography": args.bibliography,
        "thesis_pdf": args.thesis_pdf, "response_tex": args.response_tex,
        "response_pdf": args.response_pdf,
        "frozen_evidence_payload": (
            REPO / "reviewer_data_audit"
            / "frozen_numerical_evidence_payload.zip"
        ),
        "frozen_evidence_payload_manifest": (
            REPO / "reviewer_data_audit"
            / "frozen_numerical_evidence_payload_manifest.json"
        ),
    }
    absent = [f"{name}: {path}" for name, path in required.items()
              if path is None or not Path(path).exists()]
    if absent:
        print("[package] required file(s) absent:\n" + "\n".join(absent),
              file=sys.stderr)
        return 2

    code_snapshot = out / "code_snapshot"
    code_snapshot.mkdir(parents=True, exist_ok=True)
    for source in sorted(REPO.glob("*.py")):
        shutil.copy2(source, code_snapshot / source.name)

    release_name = f"MSC-THESIS-FINAL-CLOSURE-{datetime.now():%Y-%m-%dT%H%M%S}"
    release_root = out / "release" / release_name
    release_root.mkdir(parents=True)

    # The complete final evidence directory, excluding earlier release zips.
    evidence_target = release_root / "evidence" / "final_reviewer_closure"
    shutil.copytree(
        out, evidence_target,
        ignore=shutil.ignore_patterns("release", "*.zip"),
        dirs_exist_ok=True,
    )
    for name, source in required.items():
        suffix = Path(source).suffix
        copy_file_with_parents(
            Path(source).resolve(),
            release_root / "thesis_and_response" / f"{name}{suffix}",
        )
    for code in (
        "reviewer_final_closure.py", "final_acceptance_check.py",
        "ReviewerValidation.py", "GP_Density.py", "run.py",
        "thesis_closure.py", "Compare_gp_se_qcle.py", "qcle_grid_tully.py",
        "Dynamics.py", "Mint.py", "Operator.py",
    ):
        source = REPO / code
        if source.exists():
            copy_file_with_parents(source, release_root / "code" / code)

    release_commit = "NOT IDENTIFIABLE"
    snapshot_manifest = (
        REPO / "reviewer_data_audit" / "source_release_snapshot_manifest.json"
    )
    snapshot_repository = (
        REPO / "reviewer_data_audit" / "source_release_snapshot"
    )
    if snapshot_manifest.exists() and snapshot_repository.exists():
        snapshot_record = read_json(snapshot_manifest)
        release_commit = snapshot_record.get(
            "release_commit", "NOT IDENTIFIABLE"
        )
        shutil.copytree(
            snapshot_repository,
            release_root / "versioned_code_snapshot",
            dirs_exist_ok=True,
        )
        shutil.copy2(
            snapshot_manifest,
            release_root / "VERSIONED_CODE_SNAPSHOT_MANIFEST.json",
        )

    # Add every production/reference source named in final CSVs.  Paths are
    # deduplicated by SHA-256 and copied without modifying the raw originals.
    source_paths: Dict[str, Path] = {}
    for csv_path in out.rglob("*.csv"):
        if "release" in csv_path.parts:
            continue
        for row in read_csv_rows(csv_path):
            for key, value in row.items():
                if not value:
                    continue
                if any(token in key.lower() for token in (
                    "source", "manifest", "production_npz", "reference_npz",
                    "run1_manifest", "run2_manifest", "run3_manifest",
                )):
                    candidate = Path(value)
                    if candidate.is_file():
                        digest = sha256_file(candidate)
                        source_paths.setdefault(digest, candidate)
    source_index: List[Dict[str, Any]] = []
    for digest, source in sorted(source_paths.items()):
        target = release_root / "raw_source_files" / digest[:2] / f"{digest}_{source.name}"
        copy_file_with_parents(source, target)
        source_index.append({
            "archive_path": str(target.relative_to(release_root)),
            "original_path": str(source.resolve()), "sha256": digest,
            "size_bytes": source.stat().st_size,
        })
    write_csv(release_root / "RAW_SOURCE_CROSSWALK.csv", source_index)
    write_json(release_root / "ENVIRONMENT.json", environment_record(sys.argv))

    checksum_rows = []
    for path in sorted(release_root.rglob("*")):
        if path.is_file():
            checksum_rows.append({
                "relative_path": str(path.relative_to(release_root)),
                "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
            })
    write_csv(release_root / "SHA256SUMS.csv", checksum_rows)
    (release_root / "SHA256SUMS.txt").write_text(
        "".join(
            f"{row['sha256']}  {row['relative_path']}\n"
            for row in checksum_rows
        ),
        encoding="utf-8",
    )
    archive = out / f"{release_name}.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as zf:
        for path in sorted(release_root.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(Path(release_name) / path.relative_to(release_root)))
    archive_sha = sha256_file(archive)
    write_json(out / "archive_manifest.json", {
        "created_utc": utcnow(), "archive": str(archive),
        "archive_sha256": archive_sha, "archive_size_bytes": archive.stat().st_size,
        "files_hashed": len(checksum_rows),
        "audit_created_release_commit": release_commit,
        "originating_development_commit": "NOT IDENTIFIABLE",
        "public_release_record": "final_reviewer_closure/PUBLIC_RELEASE.json",
        "persistent_identifier_note": (
            "The versioned GitHub release is not a DOI or institutional "
            "persistent identifier."
        ),
    })
    print(f"[package] archive: {archive}")
    print(f"[package] SHA-256: {archive_sha}")
    print("[package] public release record: final_reviewer_closure/PUBLIC_RELEASE.json")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True,
                   choices=("plan", "execute", "analyze", "verify", "package"))
    p.add_argument("--out", type=Path, default=Path("final_reviewer_closure"))
    p.add_argument("--production-dir-P0-20", type=Path,
                   default=Path("results") / "P0_20")
    p.add_argument("--production-dir-P0-100", type=Path,
                   default=Path("results") / "P0_100")
    p.add_argument("--P0", type=float, nargs="+", default=[20.0, 100.0])
    p.add_argument("--manufactured-l2", type=float, nargs="+",
                   default=[1e-6, 0.01, 0.05])
    p.add_argument("--manufactured-N", type=int, nargs="+",
                   default=[300, 600, 1200, 2400])
    p.add_argument("--manufactured-seeds", type=int, nargs="+",
                   default=[123, 124, 125])
    p.add_argument("--dynamics-seeds", type=int, nargs="+",
                   default=[11, 29, 47, 73])
    p.add_argument("--dt-levels", type=float, nargs="+",
                   default=[0.5, 0.25, 0.125])
    p.add_argument("--support-levels", type=int, nargs="+",
                   default=[500, 1000, 2000])
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--parallel-workers", type=int, default=1,
        help=(
            "Bounded parallelism for independent manufactured fits only; "
            "reference calculations remain sequential."
        ),
    )
    p.add_argument(
        "--parallel-dynamics-workers", type=int, default=1,
        help=(
            "Bounded parallelism for independent support/time-step subprocesses. "
            "Each job has a distinct output directory; default 1 is sequential."
        ),
    )
    p.add_argument("--execute-kinds", nargs="+",
                   choices=("manufactured", "timestep", "support", "reference"),
                   default=["manufactured", "timestep", "support", "reference"],
                   help="Optional safe phase selection; the published command sequence uses all kinds.")
    p.add_argument(
        "--execute-reference-methods", nargs="+",
        choices=("tdse", "qcle"), default=["tdse", "qcle"],
        help=(
            "Optional reference-only phase selection so TDSE boundary and "
            "reflection diagnostics can be checked before grid-QCLE launch."
        ),
    )
    p.add_argument("--thesis", type=Path)
    p.add_argument("--bibliography", type=Path)
    p.add_argument("--thesis-pdf", type=Path)
    p.add_argument("--response-tex", type=Path)
    p.add_argument("--response-pdf", type=Path)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    command_history(args.out, sys.argv)
    write_json(args.out / "environment.json", environment_record(sys.argv))
    write_reproducibility_docs(args.out)
    if args.mode == "plan":
        plan = build_plan(args)
        write_json(args.out / "FINAL_RUN_MANIFEST.json", plan)
        print(json.dumps(plan["counts"], indent=2))
        return 0
    if args.mode == "execute":
        plan = build_plan(args)
        write_json(args.out / "FINAL_RUN_MANIFEST.json", plan)
        return execute(args, plan)
    if args.mode == "analyze":
        return analyze(args)
    if args.mode == "verify":
        return verify(args)
    if args.mode == "package":
        return package(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
