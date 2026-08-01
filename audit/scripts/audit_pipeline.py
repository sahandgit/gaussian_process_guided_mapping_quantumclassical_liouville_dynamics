#!/usr/bin/env python3
"""
Read-only forensic audit of the GP-RKHS-MInt QCLE pipeline.

All inputs are read from the pipeline root.  All outputs are written below
reviewer_data_audit/.  Raw files are never changed.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import statistics
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import PIL
except Exception:
    PIL = None


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "reviewer_data_audit"
TABLES = AUDIT / "tables"
PLOTS = AUDIT / "plots"
SCRIPTS = AUDIT / "scripts"
CANON_DT = ROOT / "reviewer_closure_20260723_194254"
CANON_REPL = ROOT / "reviewer_closure_20260726_174927"
GENERATED_ZIP = ROOT / "reviewer_data_audit_complete.zip"

OBSERVABLES = (
    "lw_P0", "lw_P1", "lw_P_sum", "lw_trace", "lw_energy",
    "nm_R_mean", "nm_P_mean", "nm_R_var", "nm_P_var",
    "raw_norm_drift", "raw_trace_drift", "raw_energy_drift",
    "cs_q_rms", "applied_cs_q_rms",
)
CORE_KEYS = {
    "step_index", "t", "sigma_f", "sigma_n", "lengthscales",
    "fit_rms_on_support", "gp_fit_r2", "lw_P0", "lw_P1", "lw_P_sum",
    "lw_trace", "lw_energy", "nm_R_mean", "nm_P_mean", "nm_R_var",
    "nm_P_var", "raw_norm_drift", "raw_trace_drift",
    "raw_energy_drift", "raw_norm_relative_drift",
    "raw_trace_relative_drift", "raw_energy_relative_drift",
    "sw_abs_ess_frac", "alpha_linf",
}
TCRIT_975 = {2: 4.302652729911275, 3: 3.182446305284263, 4: 2.7764451051977987}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def sha256(path: Path, block: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            data = fh.read(block)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        fields = ordered
    fields = list(fields)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: serialize(row.get(k, "")) for k in fields})


def read_csv_table(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def serialize(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return repr(float(value))
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, allow_nan=True)
    return value


def fmt(x: Any, sig: int = 5) -> str:
    if x in (None, ""):
        return "NOT COMPUTED"
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(v):
        return "NOT COMPUTED"
    if v == 0:
        return "0"
    if abs(v) >= 1e4 or abs(v) < 1e-3:
        return f"{v:.{sig-1}e}"
    return f"{v:.{sig}g}"


def md_table(rows: list[dict[str, Any]], fields: list[str], limit: int | None = None) -> str:
    use = rows if limit is None else rows[:limit]
    out = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in use:
        vals = []
        for field in fields:
            val = row.get(field, "")
            if isinstance(val, float):
                val = fmt(val)
            vals.append(str(val).replace("|", "\\|").replace("\n", " "))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def classify_file(path: Path) -> str:
    low = rel(path).lower()
    name = path.name.lower()
    ext = path.suffix.lower()
    if ext == ".py":
        return "test" if "test" in name or "/tests/" in low else "python_source"
    if name == "run_manifest.json":
        return "run_manifest"
    if "campaign_manifest" in name or "campaign_status" in name:
        return "campaign_manifest"
    if ext in {".npz", ".npy"}:
        if "tdse" in low:
            return "tdse_array"
        if "qcle" in low and "midpoint" not in low:
            return "grid_qcle_array"
        if name == "pbme.npz":
            return "pbme_output"
        if name == "midpoint.npz":
            return "midpoint_output"
        return "numerical_array"
    if ext in {".csv", ".tsv"}:
        return "table"
    if ext in {".log", ".out", ".err"}:
        return "text_log"
    if ext in {".png", ".pdf", ".svg", ".jpg", ".jpeg"}:
        return "figure"
    if ext == ".json":
        if "manufactured" in low:
            return "manufactured_metric"
        if "projection" in low:
            return "projection_metric"
        if "kde" in low or "baseline" in low:
            return "kde_gp_metric"
        if "reference" in low:
            return "reference_metric"
        if name.endswith(".meta.json"):
            return "figure_metadata"
        return "json_metric"
    if ext in {".md", ".tex", ".docx", ".txt"}:
        return "thesis_reviewer_document" if any(x in low for x in ("thesis", "reviewer", "examiner", "handoff")) else "document"
    if ext == ".zip":
        return "archive"
    return "other"


def source_files() -> list[Path]:
    out = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if AUDIT in path.parents or path == GENERATED_ZIP:
            continue
        out.append(path)
    return sorted(out, key=lambda p: rel(p).lower())


def build_file_inventory() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    principal_ext = {".json", ".npz", ".npy", ".csv", ".tsv", ".tex", ".md", ".py", ".zip"}
    for path in source_files():
        stat = path.stat()
        kind = classify_file(path)
        digest = ""
        if path.suffix.lower() in principal_ext and not path.name.lower().endswith(".meta.json"):
            digest = sha256(path)
            checks.append({"scope": "source", "path": rel(path), "sha256": digest, "bytes": stat.st_size})
        rows.append({
            "path": rel(path), "storage": "filesystem", "archive_container": "",
            "classification": kind, "extension": path.suffix.lower(),
            "bytes": stat.st_size, "modified_utc": iso_mtime(path), "sha256": digest,
        })
        if path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        rows.append({
                            "path": f"{rel(path)}!{info.filename}", "storage": "zip_member",
                            "archive_container": rel(path),
                            "classification": "archive_member",
                            "extension": Path(info.filename).suffix.lower(),
                            "bytes": info.file_size,
                            "modified_utc": "",
                            "sha256": "",
                            "zip_crc32": f"{info.CRC:08x}",
                        })
            except Exception as exc:
                rows.append({
                    "path": rel(path), "storage": "zip_error", "classification": "archive_error",
                    "extension": ".zip", "bytes": stat.st_size, "modified_utc": iso_mtime(path),
                    "sha256": digest, "error": repr(exc),
                })
    return rows, checks


def load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = read_json(path)
    args = data.get("cli_arguments", data if isinstance(data, dict) else {})
    return data, args


def campaign_environment(path: Path) -> tuple[str, str]:
    """Return nearest ancestor campaign-manifest path and its saved environment."""
    for parent in (path.parent, *path.parents):
        if parent == ROOT.parent:
            break
        manifest = parent / "campaign_manifest.json"
        if manifest.exists():
            try:
                env = read_json(manifest).get("environment", {})
                compact = {
                    "python_version": env.get("python_version"),
                    "packages": env.get("packages"),
                    "platform": env.get("platform"),
                    "git_commit": env.get("git_commit"),
                    "git_dirty_worktree": env.get("git_dirty_worktree"),
                }
                return rel(manifest), json.dumps(compact, sort_keys=True)
            except Exception as exc:
                return rel(manifest), f"NOT VERIFIABLE: {exc!r}"
    return "DATA ABSENT", "DATA ABSENT"


def inspect_npz_completion(path: Path, n_expected: int | None = None, t_expected: float | None = None) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False, "readable": False, "complete": False,
            "reason": "DATA ABSENT", "n_points": 0, "step_end": "", "t_end": "",
            "core_finite": False,
        }
    try:
        with np.load(path, allow_pickle=False) as z:
            t = np.asarray(z["t"], dtype=float)
            steps = np.asarray(z["step_index"], dtype=float)
            core = [k for k in ("t", "step_index", "lw_P0", "lw_P1", "lw_trace") if k in z]
            finite = all(np.all(np.isfinite(np.asarray(z[k]))) for k in core)
            monotonic = bool(t.size > 0 and np.all(np.diff(t) > 0))
            step_end = int(steps[-1]) if steps.size else -1
            t_end = float(t[-1]) if t.size else float("nan")
            complete = finite and monotonic
            reasons = []
            if n_expected is not None and step_end != int(n_expected):
                complete = False
                reasons.append(f"step_end={step_end}, expected={n_expected}")
            if t_expected is not None and not math.isclose(t_end, float(t_expected), rel_tol=1e-12, abs_tol=1e-9):
                complete = False
                reasons.append(f"t_end={t_end}, expected={t_expected}")
            if not finite:
                reasons.append("nonfinite core arrays")
            if not monotonic:
                reasons.append("time not strictly increasing")
            return {
                "exists": True, "readable": True, "complete": complete,
                "reason": "complete" if complete else "; ".join(reasons),
                "n_points": int(t.size), "step_end": step_end, "t_end": t_end,
                "core_finite": finite,
            }
    except Exception as exc:
        return {
            "exists": True, "readable": False, "complete": False,
            "reason": f"unreadable: {exc!r}", "n_points": 0, "step_end": "",
            "t_end": "", "core_finite": False,
        }


def run_inventory() -> list[dict[str, Any]]:
    rows = []
    manifests = [p for p in source_files() if p.name == "run_manifest.json"]
    config_seen: Counter[tuple[Any, ...]] = Counter()
    extracted = []
    for path in manifests:
        try:
            data, args = load_manifest(path)
        except Exception as exc:
            rows.append({"run_directory": rel(path.parent), "manifest": rel(path), "run_status": "NOT VERIFIABLE", "failure_message": repr(exc)})
            continue
        key = (args.get("P0"), args.get("seed"), args.get("n_train"), args.get("dt"), args.get("t_final_resolved"), args.get("sampling_mode"), args.get("surrogate"), args.get("density_mode"))
        config_seen[key] += 1
        extracted.append((path, data, args, key))
    for path, data, args, key in extracted:
        nsteps = args.get("n_steps")
        tfinal = args.get("t_final_resolved", args.get("t_final"))
        pb = inspect_npz_completion(path.parent / "pbme.npz", nsteps, tfinal)
        mid = inspect_npz_completion(path.parent / "midpoint.npz", nsteps, tfinal)
        if pb["complete"] and mid["complete"]:
            status = "COMPLETE"
        elif pb["exists"] or mid["exists"]:
            status = "RUN INCOMPLETE"
        else:
            status = "OUTPUTS DATA ABSENT"
        logs = sorted(path.parent.glob("*.log"))
        start = min((p.stat().st_mtime for p in logs + [path]), default=path.stat().st_mtime)
        end_files = [p for p in (path.parent / "pbme.npz", path.parent / "midpoint.npz") if p.exists()]
        end = max((p.stat().st_mtime for p in end_files), default=path.stat().st_mtime)
        campaign_manifest, environment = campaign_environment(path)
        rows.append({
            "run_directory": rel(path.parent), "manifest": rel(path),
            "P0": args.get("P0", ""), "seed": args.get("seed", ""),
            "N_n_train": args.get("n_train", ""), "dt": args.get("dt", ""),
            "dt_requested": args.get("dt_requested", ""), "final_time": tfinal or "",
            "expected_steps": nsteps or "", "completed_steps_pbme": pb["step_end"],
            "completed_steps_midpoint": mid["step_end"],
            "snapshot_cadence": args.get("snapshot_every", ""),
            "sampling_mode": args.get("sampling_mode", ""),
            "surrogate_type": args.get("surrogate", ""), "density_mode": args.get("density_mode", ""),
            "absolute_target_policy": args.get("abs_target", ""),
            "gp_regularization": args.get("l2_regularization", ""),
            "gp_noise_policy": "fixed" if args.get("fix_sigma_n") else "optimized",
            "hyperparameter_refit_policy": args.get("refit_hyper_policy", ""),
            "kernel_family": "ARD squared-exponential/RBF (from GP_Density.py)",
            "length_scales": "saved per-step in NPZ" if (path.parent / "pbme.npz").exists() else "DATA ABSENT",
            "density_profile_floor": args.get("product_g_floor_rel", ""),
            "normalization_policy": "label-weighted observables self-normalized; raw cloud drifts separately saved",
            "reference_grid_dimensions": "NOT APPLICABLE",
            "boundary_conditions": "NOT APPLICABLE",
            "software_versions": environment,
            "software_versions_source": campaign_manifest,
            "pbme_output_status": pb["reason"], "midpoint_output_status": mid["reason"],
            "run_status": status, "start_timestamp_utc_mtime_proxy": datetime.fromtimestamp(start, timezone.utc).isoformat(),
            "end_timestamp_utc_mtime_proxy": datetime.fromtimestamp(end, timezone.utc).isoformat(),
            "timestamps_note": "filesystem mtime proxy; explicit start/end timestamps not saved in run manifest",
            "failure_message": "" if status == "COMPLETE" else f"PBME: {pb['reason']}; MIDPOINT: {mid['reason']}",
            "duplicate_configuration_count": config_seen[key],
            "paired_initial_cloud": data.get("paired_initial_cloud", ""),
            "paired_initial_cloud_sha256": data.get("paired_initial_cloud_sha256", ""),
        })
    return rows


def expected_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = []
    for p0 in (20, 100):
        for seed in (11, 29):
            for dt in (0.5, 0.25, 0.125):
                specs.append(("A_time_step", p0, seed, 1000, dt, CANON_DT / f"step7_dt_P0{p0}" / f"seed{seed}_dt{dt}"))
    for p0 in (20, 100):
        for seed in (11, 29, 47):
            for n in (500, 1000, 2000):
                specs.append(("B_support", p0, seed, n, 0.25, CANON_DT / f"step8_support_P0{p0}" / f"seed{seed}_N{n}"))
    for p0 in (20, 100):
        for seed in (11, 29, 47, 73):
            specs.append(("C_replication", p0, seed, 1000, 0.25, CANON_REPL / f"step9_repl_P0{p0}" / f"seed{seed}"))
    for campaign, p0, seed, n, dt, directory in specs:
        manifest = directory / "run_manifest.json"
        discovered = manifest.exists()
        conflict = ""
        nsteps = None
        tfinal = None
        actual = {}
        if discovered:
            _, actual = load_manifest(manifest)
            nsteps = actual.get("n_steps")
            tfinal = actual.get("t_final_resolved", actual.get("t_final"))
            checks = {"P0": p0, "seed": seed, "n_train": n, "dt": dt}
            bad = []
            for k, exp in checks.items():
                got = actual.get(k)
                if isinstance(exp, float):
                    if got is None or not math.isclose(float(got), exp, rel_tol=0, abs_tol=1e-12):
                        bad.append(f"{k}={got} expected {exp}")
                elif got != exp:
                    bad.append(f"{k}={got} expected {exp}")
            conflict = "; ".join(bad)
        pb = inspect_npz_completion(directory / "pbme.npz", nsteps, tfinal)
        mid = inspect_npz_completion(directory / "midpoint.npz", nsteps, tfinal)
        if not discovered:
            status = "MISSING"
        elif conflict:
            status = "CONFIGURATION CONFLICT"
        elif pb["complete"] and mid["complete"]:
            status = "COMPLETE"
        elif pb["exists"] or mid["exists"]:
            status = "INCOMPLETE"
        else:
            status = "FAILED"
        rows.append({
            "campaign": campaign, "P0": p0, "seed": seed, "N": n, "dt": dt,
            "expected": True, "run_directory": rel(directory), "discovered": discovered,
            "pbme_complete": pb["complete"], "midpoint_complete": mid["complete"],
            "status": status, "missing_outputs": ", ".join(m for m, x in (("pbme.npz", pb), ("midpoint.npz", mid)) if not x["exists"]),
            "configuration_conflict": conflict, "pbme_detail": pb["reason"],
            "midpoint_detail": mid["reason"], "expected_steps": nsteps or "",
            "expected_final_time": tfinal or "",
        })
    return rows


FAIL_PATTERNS = re.compile(
    r"(Traceback \(most recent call last\)|out of memory|bad_alloc|paging file|"
    r"not enough memory|DLL load failed|cholesky.*fail|singular matrix|"
    r"not positive definite|process exited with code [1-9]|killed)",
    re.IGNORECASE,
)


def scan_logs() -> list[dict[str, Any]]:
    rows = []
    for path in [p for p in source_files() if p.suffix.lower() in {".log", ".out", ".err"}]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            rows.append({"source_file": rel(path), "issue": "unreadable log", "severity": "EXCLUDE", "message": repr(exc)})
            continue
        matches = list(FAIL_PATTERNS.finditer(text))
        nan_lines = [line for line in text.splitlines() if re.search(r"\b(?:nan|inf)\b", line, re.I)]
        if matches:
            first = matches[0]
            line_no = text.count("\n", 0, first.start()) + 1
            snippet = text.splitlines()[line_no - 1][:1000]
            rows.append({
                "source_file": rel(path), "issue": "failed attempt log", "severity": "EXCLUDE ATTEMPT",
                "first_affected_step": parse_step(snippet), "first_affected_time": parse_time(snippet),
                "affected_quantity": "process/run attempt", "propagation_continued": False,
                "post_issue_usable": False, "exclusion_rule": "attempt log contains a fatal exception/OOM/process failure",
                "message": snippet, "line": line_no,
            })
        if nan_lines:
            # Most loss=nan/reg=nan entries are deliberately unused diagnostic placeholders.
            active = [ln for ln in nan_lines if not re.search(r"loss=nan\s+reg=nan", ln, re.I)]
            rows.append({
                "source_file": rel(path), "issue": "nonfinite text token",
                "severity": "WARN" if active else "INFORMATIONAL PLACEHOLDER",
                "first_affected_step": parse_step(nan_lines[0]), "first_affected_time": parse_time(nan_lines[0]),
                "affected_quantity": "log diagnostic", "propagation_continued": "unknown",
                "post_issue_usable": "requires array check",
                "exclusion_rule": "exclude only if corresponding active numerical array is nonfinite",
                "message": nan_lines[0][:1000], "count_lines": len(nan_lines),
            })
    return rows


def parse_step(text: str) -> str:
    m = re.search(r"\bstep\s*[=:]?\s*(\d+)", text, re.I)
    return m.group(1) if m else ""


def parse_time(text: str) -> str:
    m = re.search(r"\bt\s*[=:]\s*([-+0-9.eE]+)", text)
    return m.group(1) if m else ""


def npz_stability() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    npz_paths = [p for p in source_files() if p.suffix.lower() == ".npz"]
    for path in npz_paths:
        try:
            with np.load(path, allow_pickle=False) as z:
                if "t" not in z or "step_index" not in z:
                    summaries.append({"source_file": rel(path), "method": path.stem, "status": "array artifact; no time history"})
                    continue
                t = np.asarray(z["t"], dtype=float)
                steps = np.asarray(z["step_index"])
                method = path.stem.lower()
                core_nonfinite = False
                for key in z.files:
                    arr = np.asarray(z[key])
                    if arr.dtype.kind not in "fc" or arr.ndim == 0:
                        continue
                    if arr.shape[0] != len(t):
                        continue
                    finite = np.isfinite(arr)
                    if np.all(finite):
                        continue
                    bad_idx = np.argwhere(~finite)
                    first_i = int(bad_idx[0][0])
                    is_core = key in CORE_KEYS
                    # Initial-step NaNs in correction-only diagnostics are explicit N/A placeholders.
                    only_initial = bool(np.all(bad_idx[:, 0] == 0))
                    sev = "EXCLUDE FROM FIRST NONFINITE" if is_core else ("INFORMATIONAL PLACEHOLDER" if only_initial or key.startswith(("fc_", "delta_alpha_", "omega_A_", "sigma1_", "sigma2_")) else "WARN")
                    core_nonfinite |= is_core
                    issues.append({
                        "source_file": rel(path), "method": method, "issue": "nonfinite numerical values",
                        "severity": sev, "first_affected_step": int(steps[first_i]) if first_i < len(steps) else first_i,
                        "first_affected_time": float(t[first_i]) if first_i < len(t) else "",
                        "affected_quantity": key, "count": int(np.size(finite) - np.count_nonzero(finite)),
                        "propagation_continued": first_i < len(t) - 1,
                        "post_issue_usable": not is_core,
                        "exclusion_rule": "active core observable nonfinite => exclude from first affected index; inactive/initial placeholder is not a run failure",
                    })
                for key in ("nm_R_var", "nm_P_var", "cloud_weighted_R_var", "cloud_weighted_P_var"):
                    if key in z:
                        arr = np.asarray(z[key], dtype=float)
                        if np.any(arr < -1e-12):
                            i = int(np.where(arr < -1e-12)[0][0])
                            issues.append({
                                "source_file": rel(path), "method": method, "issue": "negative variance",
                                "severity": "EXCLUDE QUANTITY", "first_affected_step": int(steps[i]),
                                "first_affected_time": float(t[i]), "affected_quantity": key,
                                "count": int(np.count_nonzero(arr < -1e-12)),
                                "propagation_continued": i < len(t)-1, "post_issue_usable": False,
                                "exclusion_rule": "variance below -1e-12 is invalid",
                            })
                if "sigma_n" in z:
                    arr = np.asarray(z["sigma_n"], dtype=float)
                    maxv = float(np.nanmax(arr))
                    at_ceiling = np.isclose(arr, math.e, rtol=1e-10, atol=1e-12)
                    if np.any(at_ceiling) or maxv > 1.0:
                        i = int(np.where(at_ceiling | (arr > 1.0))[0][0])
                        issues.append({
                            "source_file": rel(path), "method": method, "issue": "GP noise abnormal/at upper bound",
                            "severity": "SCIENTIFIC WARNING", "first_affected_step": int(steps[i]),
                            "first_affected_time": float(t[i]), "affected_quantity": "sigma_n",
                            "observed_max": maxv, "count": int(np.count_nonzero(at_ceiling | (arr > 1.0))),
                            "propagation_continued": i < len(t)-1, "post_issue_usable": "finite but surrogate fidelity not validated",
                            "exclusion_rule": "not automatically excluded; interpret with saved fit diagnostics and conservation",
                        })
                if "alpha_linf" in z:
                    arr = np.asarray(z["alpha_linf"], dtype=float)
                    maxv = float(np.nanmax(np.abs(arr)))
                    if maxv > 1e6:
                        i = int(np.nanargmax(np.abs(arr)))
                        issues.append({
                            "source_file": rel(path), "method": method, "issue": "coefficient explosion",
                            "severity": "SCIENTIFIC WARNING", "first_affected_step": int(steps[i]),
                            "first_affected_time": float(t[i]), "affected_quantity": "alpha_linf",
                            "observed_max": maxv, "propagation_continued": i < len(t)-1,
                            "post_issue_usable": "not without corroborating stability", "exclusion_rule": "flag >1e6; threshold is audit diagnostic, not predeclared acceptance criterion",
                        })
                if "sw_abs_ess_frac" in z:
                    arr = np.asarray(z["sw_abs_ess_frac"], dtype=float)
                    if np.any(arr < 0.05):
                        i = int(np.where(arr < 0.05)[0][0])
                        issues.append({
                            "source_file": rel(path), "method": method, "issue": "effective sample size collapse",
                            "severity": "SCIENTIFIC WARNING", "first_affected_step": int(steps[i]),
                            "first_affected_time": float(t[i]), "affected_quantity": "sw_abs_ess_frac",
                            "observed_min": float(np.nanmin(arr)), "propagation_continued": i < len(t)-1,
                            "post_issue_usable": "not without qualification", "exclusion_rule": "flag below manifest resampling threshold 0.05; resampling was disabled",
                        })
                # Raw-conservation diagnostic gates. These 1% gates were introduced
                # by the audit for visibility; they were not predeclared acceptance
                # tolerances and therefore do not convert a scientific warning into
                # an automatic run-file exclusion.
                for key, threshold in (
                    ("raw_norm_drift", 0.01),
                    ("raw_trace_drift", 0.01),
                    ("raw_energy_relative_drift", 0.01),
                ):
                    if key in z:
                        arr = np.asarray(z[key], dtype=float)
                        hit = np.isfinite(arr) & (np.abs(arr) > threshold)
                        if np.any(hit):
                            i = int(np.where(hit)[0][0])
                            issues.append({
                                "source_file": rel(path), "method": method,
                                "issue": "raw conservation drift exceeds audit visibility gate",
                                "severity": "SCIENTIFIC WARNING",
                                "first_affected_step": int(steps[i]), "first_affected_time": float(t[i]),
                                "affected_quantity": key, "observed_max": float(np.nanmax(np.abs(arr))),
                                "audit_visibility_threshold": threshold,
                                "propagation_continued": i < len(t)-1,
                                "post_issue_usable": "not as conservation validation",
                                "exclusion_rule": "1% audit gate is diagnostic, not predeclared; retain file but reject conservation-pass claim",
                            })
                for key in ("lw_P0", "lw_P1"):
                    if key in z:
                        arr = np.asarray(z[key], dtype=float)
                        hit = np.isfinite(arr) & ((arr < -1e-6) | (arr > 1.0 + 1e-6))
                        if np.any(hit):
                            i = int(np.where(hit)[0][0])
                            issues.append({
                                "source_file": rel(path), "method": method,
                                "issue": "self-normalized population outside [0,1]",
                                "severity": "SCIENTIFIC WARNING",
                                "first_affected_step": int(steps[i]), "first_affected_time": float(t[i]),
                                "affected_quantity": key,
                                "observed_min": float(np.nanmin(arr)), "observed_max": float(np.nanmax(arr)),
                                "propagation_continued": i < len(t)-1,
                                "post_issue_usable": "only as a signed-estimator diagnostic",
                                "exclusion_rule": "flag outside [0,1] by >1e-6; retain signed estimator but do not call it a physical probability",
                            })
                if "cs_q_rms" in z:
                    arr = np.asarray(z["cs_q_rms"], dtype=float)
                    hit = np.isfinite(arr) & (np.abs(arr) > 1.0)
                    if np.any(hit):
                        i = int(np.where(hit)[0][0])
                        issues.append({
                            "source_file": rel(path), "method": method,
                            "issue": "excess-correction magnitude explosion",
                            "severity": "SCIENTIFIC WARNING",
                            "first_affected_step": int(steps[i]), "first_affected_time": float(t[i]),
                            "affected_quantity": "cs_q_rms",
                            "observed_max": float(np.nanmax(np.abs(arr))),
                            "audit_visibility_threshold": 1.0,
                            "propagation_continued": i < len(t)-1,
                            "post_issue_usable": "not without stability qualification",
                            "exclusion_rule": "audit diagnostic threshold |Q| RMS>1; not a predeclared acceptance threshold",
                        })
                if "cs_q_weight_denominator" in z and "cs_q_weighted_mean_defined" in z:
                    denom = np.asarray(z["cs_q_weight_denominator"], dtype=float)
                    defined = np.asarray(z["cs_q_weighted_mean_defined"], dtype=float)
                    q = np.asarray(z["cs_q_rms"], dtype=float) if "cs_q_rms" in z else np.zeros_like(denom)
                    # Ignore initialization and inactive PBME Q=0. A nonzero Q
                    # with undefined signed mean is a genuine near-zero-denominator
                    # diagnostic, not necessarily a failed propagation.
                    hit = (np.arange(len(denom)) > 0) & (np.abs(q) > 0) & (defined < 0.5)
                    if np.any(hit):
                        i = int(np.where(hit)[0][0])
                        issues.append({
                            "source_file": rel(path), "method": method,
                            "issue": "near-zero signed denominator / weighted Q mean undefined",
                            "severity": "SCIENTIFIC WARNING",
                            "first_affected_step": int(steps[i]), "first_affected_time": float(t[i]),
                            "affected_quantity": "cs_q_weight_denominator",
                            "observed_min_abs": float(np.nanmin(np.abs(denom[hit]))),
                            "count": int(np.count_nonzero(hit)),
                            "propagation_continued": i < len(t)-1,
                            "post_issue_usable": "weighted-mean diagnostic unavailable at affected steps",
                            "exclusion_rule": "do not use undefined signed-weighted Q mean; other finite observables retained",
                        })
                for left, right in (("lw_P0", "lw_P1"),):
                    if left in z and right in z and "lw_trace" in z:
                        resid = np.asarray(z[left]) + np.asarray(z[right]) - np.asarray(z["lw_trace"])
                        mx = float(np.nanmax(np.abs(resid)))
                        if mx > 1e-10:
                            i = int(np.nanargmax(np.abs(resid)))
                            issues.append({
                                "source_file": rel(path), "method": method, "issue": "population/trace identity violation",
                                "severity": "SCIENTIFIC WARNING", "first_affected_step": int(steps[i]),
                                "first_affected_time": float(t[i]), "affected_quantity": "lw_P0+lw_P1-lw_trace",
                                "observed_max": mx, "propagation_continued": i < len(t)-1,
                                "post_issue_usable": "with disclosed residual", "exclusion_rule": "flag maximum identity residual >1e-10",
                            })
                summary = {
                    "source_file": rel(path), "method": method, "n_points": len(t),
                    "step_end": int(steps[-1]), "t_end": float(t[-1]),
                    "strictly_increasing_time": bool(np.all(np.diff(t) > 0)),
                    "core_finite": not core_nonfinite,
                }
                for key in ("sigma_n", "alpha_linf", "sw_abs_ess_frac", "gp_fit_r2", "cs_q_rms", "raw_norm_drift", "raw_trace_drift", "raw_energy_drift"):
                    if key in z:
                        arr = np.asarray(z[key], dtype=float)
                        summary[f"{key}_min"] = float(np.nanmin(arr))
                        summary[f"{key}_max"] = float(np.nanmax(arr))
                        summary[f"{key}_endpoint"] = float(arr[-1])
                summaries.append(summary)
        except Exception as exc:
            issues.append({
                "source_file": rel(path), "method": path.stem, "issue": "NPZ unreadable",
                "severity": "EXCLUDE FILE", "first_affected_step": "", "first_affected_time": "",
                "affected_quantity": "entire file", "propagation_continued": "unknown",
                "post_issue_usable": False, "exclusion_rule": "unreadable archive",
                "message": repr(exc),
            })
    return issues, summaries


def manufactured_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    paths = [p for p in source_files() if p.name == "manufactured_operator_metrics.json"]
    for path in paths:
        try:
            data = read_json(path)
        except Exception:
            continue
        campaign = path.parts[path.parts.index(ROOT.name) + 1] if ROOT.name in path.parts else path.parent.name
        canonical = "reviewer_closure_20260726_174927" in path.parts
        for support_name, support in data.get("metrics", {}).items():
            for quantity, vals in support.items():
                rows.append({
                    "source_file": rel(path), "campaign_root": campaign,
                    "canonical_latest_campaign": canonical,
                    "N": data.get("n_train"), "seed": data.get("seed"),
                    "n_query": data.get("n_query"), "query_set": support_name,
                    "quantity": quantity, "E1_relative": "NOT COMPUTED",
                    "E2_relative": vals.get("relative_l2", "NOT COMPUTED"),
                    "Einf_relative": "NOT COMPUTED",
                    "rmse_absolute": vals.get("rmse", ""),
                    "linf_absolute": vals.get("linf", ""),
                    "conditioning_metric": "DATA ABSENT",
                    "note": "JSON lacks reference arrays/denominators; stored linf is absolute, not relative Einf",
                })
    summaries = []
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    sources: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for r in rows:
        val = r["E2_relative"]
        if isinstance(val, (int, float)):
            key = (r["campaign_root"], r["canonical_latest_campaign"], r["N"], r["query_set"], r["quantity"])
            groups[key].append(float(val))
            sources[key].append(r["source_file"])
    for (campaign, canonical, n, qset, quantity), vals in sorted(groups.items()):
        k = len(vals)
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if k > 1 else float("nan")
        summaries.append({
            "campaign_root": campaign, "canonical_latest_campaign": canonical,
            "N": n, "query_set": qset, "quantity": quantity, "n_independent_seeds": k,
            "mean_E2": mean, "sample_sd_E2": sd,
            "seed_min_E2": min(vals), "seed_max_E2": max(vals),
            "E1": "NOT COMPUTED", "Einf": "NOT COMPUTED",
            "source_files": "; ".join(sources[(campaign, canonical, n, qset, quantity)]),
        })
    # Percentage change uses seed-aggregated means and only adjacent available N.
    by_metric: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_metric[(row["campaign_root"], row["query_set"], row["quantity"])].append(row)
    for group in by_metric.values():
        group.sort(key=lambda r: int(r["N"]))
        for i, row in enumerate(group):
            if i == 0:
                row["percent_change_from_previous_N"] = "NOT APPLICABLE"
            else:
                prev = group[i-1]["mean_E2"]
                row["percent_change_from_previous_N"] = 100.0 * (row["mean_E2"] - prev) / prev
    return rows, summaries


def projection_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, summaries = [], []
    paths = [p for p in source_files() if p.name == "projection_leakage.json"]
    for path in paths:
        try:
            d = read_json(path)
        except Exception:
            continue
        src = str(d.get("source", "")).replace("\\", "/")
        p0 = 100 if "P0100" in src or "P0_100" in rel(path) else (20 if "P020" in src or "P0_20" in rel(path) else "")
        method = "midpoint" if "midpoint" in src.lower() else "pbme"
        seed_match = re.search(r"seed(\d+)", src)
        propagation_seed = int(seed_match.group(1)) if seed_match else ""
        vals = [float(x["relative_l2_leakage"]) for x in d.get("per_anchor", [])]
        absvals = [float(x["absolute_rms_leakage"]) for x in d.get("per_anchor", [])]
        canonical = "step9_repl" in src and propagation_seed == 11 and p0 in (20, 100)
        for item in d.get("per_anchor", []):
            rows.append({
                "source_file": rel(path), "source_run": src, "canonical": canonical,
                "P0": p0, "propagation_seed": propagation_seed, "diagnostic_seed": d.get("seed"),
                "method": method, "snapshot_step": d.get("snapshot_step"),
                "support_index": item.get("support_index"),
                "relative_l2_leakage": item.get("relative_l2_leakage"),
                "absolute_rms_leakage": item.get("absolute_rms_leakage"),
                "basis_rank": d.get("basis_rank"), "n_bath_anchors": d.get("n_bath_anchors"),
                "n_mapping_probes": d.get("n_mapping_probes"),
            })
        if vals:
            summaries.append({
                "source_file": rel(path), "source_run": src, "canonical": canonical,
                "P0": p0, "propagation_seed": propagation_seed, "diagnostic_seed": d.get("seed"),
                "method": method, "snapshot_step": d.get("snapshot_step"),
                "n_bath_anchors": len(vals), "n_mapping_probes": d.get("n_mapping_probes"),
                "basis_rank": d.get("basis_rank"), "mean": statistics.mean(vals),
                "median": statistics.median(vals), "sample_sd": statistics.stdev(vals) if len(vals)>1 else float("nan"),
                "maximum": max(vals), "minimum": min(vals), "mean_absolute_rms": statistics.mean(absvals),
                "interpretation": "diagnostic surrogate projection residual; projection was not enforced; anchors are not independent propagation seeds",
            })
    return rows, summaries


def baseline_table() -> list[dict[str, Any]]:
    rows = []
    paths = [p for p in source_files() if p.name == "kde_gp_identical_support.json"]
    for path in paths:
        try:
            d = read_json(path)
        except Exception:
            continue
        src = str(d.get("source", "")).replace("\\", "/")
        p0 = 100 if "P0100" in src else (20 if "P020" in src else "")
        seed_match = re.search(r"seed(\d+)", src)
        seed = int(seed_match.group(1)) if seed_match else ""
        e = d.get("shape_errors", {})
        a = d.get("acceptance", {})
        rows.append({
            "source_file": rel(path), "source_run": src,
            "canonical": bool("step9_repl" in src and seed == 11 and p0 in (20, 100)),
            "method": "PBME" if "pbme" in src.lower() else "NOT IDENTIFIABLE",
            "P0": p0, "seed": seed, "snapshot_step": d.get("step"),
            "n_support": d.get("n_support"), "same_support_hash": d.get("initial_cloud_sha256"),
            "weight_policy": d.get("weight_policy"), "estimator_contract": d.get("estimator_contract"),
            "n_R": d.get("grid", {}).get("n_R"), "n_P": d.get("grid", {}).get("n_P"),
            "R_range": d.get("grid", {}).get("R_range"), "P_range": d.get("grid", {}).get("P_range"),
            "bandwidth_R": d.get("bandwidth", {}).get("R"), "bandwidth_P": d.get("bandwidth", {}).get("P"),
            "bandwidth_policy": d.get("bandwidth", {}).get("policy"),
            "target_raw_mass": d.get("raw_norms", {}).get("target_infinite_domain"),
            "gp_grid_mass": d.get("raw_norms", {}).get("gp_on_grid"),
            "kde_grid_mass": d.get("raw_norms", {}).get("kde_on_grid"),
            "E1": e.get("E1", "NOT COMPUTED"), "E2": e.get("E2", "NOT COMPUTED"),
            "Einf": e.get("Einf", "NOT COMPUTED"), "threshold_metric": a.get("metric"),
            "threshold": a.get("threshold"), "threshold_applies": a.get("applies"),
            "passed": a.get("passed"), "scale_policy": d.get("scale_policy"),
        })
    return rows


def interp_common(t_src: np.ndarray, u_src: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    if t_target[0] < t_src[0] - 1e-12 or t_target[-1] > t_src[-1] + 1e-12:
        raise ValueError("target grid would require extrapolation")
    return np.interp(t_target, t_src, u_src)


def diff_metrics(t: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = np.asarray(a) - np.asarray(b)
    span = max(float(t[-1] - t[0]), 1e-30)
    return {
        "endpoint_signed_difference": float(d[-1]),
        "endpoint_absolute_difference": float(abs(d[-1])),
        "maximum_in_time_absolute_difference": float(np.max(np.abs(d))),
        "time_integrated_L1_difference": float(np.trapezoid(np.abs(d), t) / span),
        "time_integrated_L2_difference": float(np.sqrt(np.trapezoid(d*d, t) / span)),
    }


def time_refinement_table() -> list[dict[str, Any]]:
    rows = []
    cache: dict[tuple[int, int, str, float], dict[str, np.ndarray]] = {}
    for p0 in (20, 100):
        for seed in (11, 29):
            for method in ("pbme", "midpoint"):
                for dt in (0.5, 0.25, 0.125):
                    path = CANON_DT / f"step7_dt_P0{p0}" / f"seed{seed}_dt{dt}" / f"{method}.npz"
                    if path.exists():
                        with np.load(path, allow_pickle=False) as z:
                            cache[(p0, seed, method, dt)] = {k: np.asarray(z[k], dtype=float) for k in ("t",) + OBSERVABLES if k in z}
                if not all((p0, seed, method, dt) in cache for dt in (0.5, 0.25, 0.125)):
                    continue
                finest = cache[(p0, seed, method, 0.125)]
                common_obs = set(finest) - {"t"}
                for dt in (0.5, 0.25):
                    common_obs &= set(cache[(p0, seed, method, dt)]) - {"t"}
                for obs in sorted(common_obs):
                    ref_t = finest["t"]
                    ref_u = finest[obs]
                    per_level = {}
                    for dt in (0.5, 0.25):
                        data = cache[(p0, seed, method, dt)]
                        aligned_fine = interp_common(ref_t, ref_u, data["t"])
                        met = diff_metrics(data["t"], data[obs], aligned_fine)
                        row = {
                            "P0": p0, "seed": seed, "method": method, "observable": obs,
                            "dt": dt, "reference_dt": 0.125, "time_alignment": "linear interpolation of finest onto coarser saved times; no extrapolation",
                            **met,
                            "source_file": rel(CANON_DT / f"step7_dt_P0{p0}" / f"seed{seed}_dt{dt}" / f"{method}.npz"),
                            "reference_source_file": rel(CANON_DT / f"step7_dt_P0{p0}" / f"seed{seed}_dt0.125" / f"{method}.npz"),
                        }
                        per_level[dt] = row
                        rows.append(row)
                    # Observed order from coarse-vs-medium and medium-vs-fine on coarse grid.
                    coarse = cache[(p0, seed, method, 0.5)]
                    med = cache[(p0, seed, method, 0.25)]
                    fine = cache[(p0, seed, method, 0.125)]
                    tg = coarse["t"]
                    ug0 = coarse[obs]
                    ug1 = interp_common(med["t"], med[obs], tg)
                    ug2 = interp_common(fine["t"], fine[obs], tg)
                    n01 = diff_metrics(tg, ug0, ug1)["time_integrated_L2_difference"]
                    n12 = diff_metrics(tg, ug1, ug2)["time_integrated_L2_difference"]
                    # Compare refinement signal to independent-seed variation on the finest grid.
                    other = 29 if seed == 11 else 11
                    other_data = cache.get((p0, other, method, 0.125))
                    seed_noise = float("nan")
                    if other_data and obs in other_data:
                        other_u = interp_common(other_data["t"], other_data[obs], tg)
                        this_u = interp_common(fine["t"], fine[obs], tg)
                        seed_noise = diff_metrics(tg, this_u, other_u)["time_integrated_L2_difference"]
                    if n12 <= 1e-14:
                        pval, reason = "NOT COMPUTED", "fine-level denominator <=1e-14"
                    elif math.isfinite(seed_noise) and min(n01, n12) <= seed_noise:
                        pval, reason = "NOT COMPUTED", "independent-seed variation equals/exceeds refinement signal"
                    else:
                        pval, reason = math.log(n01 / n12, 2.0), "computed from three compatible levels"
                    for dt in (0.5, 0.25):
                        per_level[dt]["coarse_medium_L2_on_coarse_grid"] = n01
                        per_level[dt]["medium_fine_L2_on_coarse_grid"] = n12
                        per_level[dt]["finest_independent_seed_L2_on_coarse_grid"] = seed_noise
                        per_level[dt]["empirical_order_p"] = pval
                        per_level[dt]["order_status"] = reason
    return rows


def initial_cloud_hash(path: Path) -> str:
    if not path.exists():
        return ""
    with np.load(path, allow_pickle=False) as z:
        if "snap_000000_Z" not in z:
            return ""
        arr = np.ascontiguousarray(z["snap_000000_Z"])
        return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def ci_stats(vals: list[float]) -> dict[str, Any]:
    n = len(vals)
    mean = statistics.mean(vals) if n else float("nan")
    sd = statistics.stdev(vals) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    tcrit = TCRIT_975.get(n, float("nan"))
    return {
        "n_independent_seeds": n, "mean": mean, "sample_sd": sd, "standard_error": se,
        "ci95_lower": mean - tcrit * se if n > 1 else float("nan"),
        "ci95_upper": mean + tcrit * se if n > 1 else float("nan"),
        "ci_method": f"two-sided Student-t, df={n-1}" if n > 1 else "NOT COMPUTED",
    }


def support_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    endpoints, summaries, nested = [], [], []
    data: dict[tuple[int, int, int, str], dict[str, float]] = {}
    hashes: dict[tuple[int, int, int, str], str] = {}
    clouds: dict[tuple[int, int, int, str], np.ndarray] = {}
    for p0 in (20, 100):
        for seed in (11, 29, 47):
            for n in (500, 1000, 2000):
                directory = CANON_DT / f"step8_support_P0{p0}" / f"seed{seed}_N{n}"
                for method in ("pbme", "midpoint"):
                    path = directory / f"{method}.npz"
                    if not path.exists():
                        endpoints.append({
                            "P0": p0, "seed": seed, "N": n, "method": method,
                            "observable": "ALL", "endpoint_value": "DATA ABSENT",
                            "status": "RUN INCOMPLETE", "source_file": rel(path),
                        })
                        continue
                    with np.load(path, allow_pickle=False) as z:
                        vals = {}
                        if "snap_000000_Z" in z:
                            clouds[(p0, seed, n, method)] = np.asarray(z["snap_000000_Z"]).copy()
                        for obs in OBSERVABLES:
                            if obs in z:
                                value = float(np.asarray(z[obs])[-1])
                                vals[obs] = value
                                endpoints.append({
                                    "P0": p0, "seed": seed, "N": n, "method": method,
                                    "observable": obs, "endpoint_value": value, "status": "EXTRACTED",
                                    "source_file": rel(path),
                                })
                        data[(p0, seed, n, method)] = vals
                    hashes[(p0, seed, n, method)] = initial_cloud_hash(path)
            # Prefix/nesting check within each seed and method.
            for method in ("pbme", "midpoint"):
                h = [hashes.get((p0, seed, n, method), "") for n in (500, 1000, 2000)]
                a500 = clouds.get((p0, seed, 500, method))
                a1000 = clouds.get((p0, seed, 1000, method))
                a2000 = clouds.get((p0, seed, 2000, method))
                eq_500_1000 = bool(a500 is not None and a1000 is not None and np.array_equal(a500, a1000[:500]))
                eq_500_2000 = bool(a500 is not None and a2000 is not None and np.array_equal(a500, a2000[:500]))
                eq_1000_2000 = bool(a1000 is not None and a2000 is not None and np.array_equal(a1000, a2000[:1000]))
                all_available = all(x is not None for x in (a500, a1000, a2000))
                nested.append({
                    "P0": p0, "seed": seed, "method": method,
                    "N500_initial_hash": h[0], "N1000_initial_hash": h[1], "N2000_initial_hash": h[2],
                    "all_three_clouds_available": all_available,
                    "N500_equals_N1000_prefix": eq_500_1000,
                    "N500_equals_N2000_prefix": eq_500_2000,
                    "N1000_equals_N2000_prefix": eq_1000_2000,
                    "nested_clouds": bool(all_available and eq_500_1000 and eq_500_2000 and eq_1000_2000),
                    "test": "exact np.array_equal comparison of saved snap_000000_Z prefixes",
                    "conclusion": ("not nested; independently sampled support; no pointwise trajectory comparison and no deterministic support-convergence claim"
                                   if all_available else "RUN INCOMPLETE; available prefix comparisons do not establish nesting"),
                })
    for p0 in (20, 100):
        for method in ("pbme", "midpoint"):
            for n in (500, 1000, 2000):
                for obs in OBSERVABLES:
                    vals = [data[(p0, seed, n, method)][obs] for seed in (11, 29, 47) if (p0, seed, n, method) in data and obs in data[(p0, seed, n, method)]]
                    if not vals:
                        continue
                    stats = ci_stats(vals)
                    ref_vals = [data[(p0, seed, 2000, method)][obs] for seed in (11, 29, 47) if (p0, seed, 2000, method) in data and obs in data[(p0, seed, 2000, method)]]
                    diff = stats["mean"] - statistics.mean(ref_vals) if ref_vals else "NOT COMPUTED"
                    summaries.append({
                        "P0": p0, "method": method, "N": n, "observable": obs, **stats,
                        "difference_from_N2000_seed_mean": diff,
                        "N2000_available_seed_count": len(ref_vals),
                        "trend_interpretation": "compare mean changes with sample SD; clouds independently sampled",
                        "source_files": "; ".join(rel(CANON_DT / f"step8_support_P0{p0}" / f"seed{s}_N{n}" / f"{method}.npz") for s in (11,29,47) if (p0,s,n,method) in data),
                    })
    return endpoints, summaries, nested


def replication_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trajectories, endpoints = [], []
    tau_grid = np.linspace(0.0, 2.0, 41)
    for p0 in (20, 100):
        tc = 30000.0 / p0
        for method in ("pbme", "midpoint"):
            series: dict[str, list[np.ndarray]] = defaultdict(list)
            for seed in (11, 29, 47, 73):
                path = CANON_REPL / f"step9_repl_P0{p0}" / f"seed{seed}" / f"{method}.npz"
                with np.load(path, allow_pickle=False) as z:
                    t = np.asarray(z["t"], dtype=float)
                    for obs in OBSERVABLES:
                        if obs in z:
                            u = np.interp(tau_grid * tc, t, np.asarray(z[obs], dtype=float))
                            series[obs].append(u)
                            endpoints.append({
                                "P0": p0, "method": method, "seed": seed, "observable": obs,
                                "endpoint_value": float(u[-1]), "source_file": rel(path),
                            })
            for obs, arrays in series.items():
                a = np.vstack(arrays)
                for j, tau in enumerate(tau_grid):
                    vals = list(map(float, a[:, j]))
                    st = ci_stats(vals)
                    mean = st["mean"]
                    trajectories.append({
                        "P0": p0, "method": method, "observable": obs,
                        "t_over_tc": float(tau), "physical_time": float(tau*tc), **st,
                        "minimum": min(vals), "maximum": max(vals), "spread": max(vals)-min(vals),
                        "coefficient_of_variation": st["sample_sd"]/abs(mean) if abs(mean)>1e-14 else "NOT COMPUTED",
                        "source_files": "; ".join(rel(CANON_REPL / f"step9_repl_P0{p0}" / f"seed{s}" / f"{method}.npz") for s in (11,29,47,73)),
                    })
    return trajectories, endpoints


def drift_behavior(arr: np.ndarray) -> str:
    d = np.diff(arr)
    tol = max(1e-14, np.nanmax(np.abs(arr)) * 1e-10)
    nz = d[np.abs(d) > tol]
    if nz.size == 0:
        return "constant within tolerance"
    if np.all(nz >= 0) or np.all(nz <= 0):
        return "monotonic"
    signs = np.sign(nz)
    changes = int(np.count_nonzero(signs[1:] != signs[:-1]))
    return f"oscillatory/nonmonotonic ({changes} derivative sign changes)"


def conservation_table(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in run_rows:
        directory = ROOT / run["run_directory"]
        for method in ("pbme", "midpoint"):
            path = directory / f"{method}.npz"
            if not path.exists():
                continue
            try:
                with np.load(path, allow_pickle=False) as z:
                    if "t" not in z:
                        continue
                    t = np.asarray(z["t"], dtype=float)
                    for quantity, key, init_key, rel_key in (
                        ("normalization", "raw_norm_drift", "raw_norm_initial", "raw_norm_relative_drift"),
                        ("trace", "raw_trace_drift", "raw_trace_initial", "raw_trace_relative_drift"),
                        ("energy", "raw_energy_drift", "raw_energy_initial", "raw_energy_relative_drift"),
                    ):
                        if key not in z:
                            rows.append({
                                "source_file": rel(path), "method": method, "quantity": quantity,
                                "status": "DATA ABSENT", "endpoint_drift": "NOT COMPUTED",
                            })
                            continue
                        arr = np.asarray(z[key], dtype=float)
                        i = int(np.nanargmax(np.abs(arr)))
                        init = float(np.asarray(z[init_key], dtype=float)[0]) if init_key in z else float("nan")
                        eps = 1e-30
                        relarr = np.asarray(z[rel_key], dtype=float) if rel_key in z else arr/max(abs(init),eps)
                        rows.append({
                            "source_file": rel(path), "run_directory": run["run_directory"],
                            "method": method, "P0": run.get("P0"), "seed": run.get("seed"),
                            "N": run.get("N_n_train"), "dt": run.get("dt"),
                            "quantity": quantity, "status": "EXTRACTED",
                            "initial_raw_value": init, "endpoint_drift": float(arr[-1]),
                            "maximum_absolute_drift": float(np.nanmax(np.abs(arr))),
                            "rms_drift": float(np.sqrt(np.nanmean(arr*arr))),
                            "time_of_max_abs_drift": float(t[i]),
                            "endpoint_relative_drift": float(relarr[-1]),
                            "maximum_absolute_relative_drift": float(np.nanmax(np.abs(relarr))),
                            "behavior": drift_behavior(arr),
                            "renormalization_applied_to_displayed_lw_observables": True,
                            "normalization_note": "raw drift is pre-self-normalization cloud estimator; lw_* observables divide by current raw cloud norm",
                        })
            except Exception:
                continue
    return rows


def all_run_endpoint_table(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract comparable saved endpoint observables from every discovered run."""
    rows: list[dict[str, Any]] = []
    for run in run_rows:
        directory = ROOT / run["run_directory"]
        for method in ("pbme", "midpoint"):
            path = directory / f"{method}.npz"
            if not path.exists():
                rows.append({
                    "run_directory": run["run_directory"], "source_file": rel(path),
                    "method": method, "P0": run.get("P0"), "seed": run.get("seed"),
                    "N": run.get("N_n_train"), "dt": run.get("dt"),
                    "observable": "ALL", "endpoint_value": "DATA ABSENT",
                    "status": "RUN INCOMPLETE",
                })
                continue
            try:
                with np.load(path, allow_pickle=False) as z:
                    if "t" not in z:
                        continue
                    for obs in OBSERVABLES:
                        if obs in z:
                            rows.append({
                                "run_directory": run["run_directory"], "source_file": rel(path),
                                "method": method, "P0": run.get("P0"), "seed": run.get("seed"),
                                "N": run.get("N_n_train"), "dt": run.get("dt"),
                                "final_time": float(np.asarray(z["t"])[-1]),
                                "observable": obs, "endpoint_value": float(np.asarray(z[obs])[-1]),
                                "status": "EXTRACTED",
                            })
                        else:
                            rows.append({
                                "run_directory": run["run_directory"], "source_file": rel(path),
                                "method": method, "P0": run.get("P0"), "seed": run.get("seed"),
                                "N": run.get("N_n_train"), "dt": run.get("dt"),
                                "observable": obs, "endpoint_value": "DATA ABSENT",
                                "status": "DATA ABSENT",
                            })
            except Exception as exc:
                rows.append({
                    "run_directory": run["run_directory"], "source_file": rel(path),
                    "method": method, "P0": run.get("P0"), "seed": run.get("seed"),
                    "N": run.get("N_n_train"), "dt": run.get("dt"),
                    "observable": "ALL", "endpoint_value": "NOT COMPUTED",
                    "status": f"NOT VERIFIABLE: {exc!r}",
                })
    return rows


def midpoint_pbme_table() -> list[dict[str, Any]]:
    rows = []
    # Use replication campaign: paired initial cloud, same N/dt/time grid.
    for p0 in (20, 100):
        for seed in (11, 29, 47, 73):
            d = CANON_REPL / f"step9_repl_P0{p0}" / f"seed{seed}"
            with np.load(d/"pbme.npz", allow_pickle=False) as pb, np.load(d/"midpoint.npz", allow_pickle=False) as mid:
                t = np.asarray(pb["t"], dtype=float)
                same = np.array_equal(t, np.asarray(mid["t"], dtype=float))
                for obs in OBSERVABLES:
                    if obs not in pb or obs not in mid:
                        continue
                    met = diff_metrics(t, np.asarray(mid[obs], dtype=float), np.asarray(pb[obs], dtype=float))
                    rows.append({
                        "P0": p0, "seed": seed, "observable": obs, "same_time_grid": same,
                        "same_initial_cloud_hash": read_json(d/"run_manifest.json").get("paired_initial_cloud_sha256"),
                        **{f"MIDPOINT_minus_PBME_{k}": v for k,v in met.items()},
                        "PBME_endpoint": float(np.asarray(pb[obs])[-1]),
                        "MIDPOINT_endpoint": float(np.asarray(mid[obs])[-1]),
                        "reference_error_difference": "NOT COMPUTED",
                        "reason": "no compatible TDSE/grid-QCLE trajectory shares production R0=-15, t_final=2tc, times and observable normalization",
                        "pbme_source": rel(d/"pbme.npz"), "midpoint_source": rel(d/"midpoint.npz"),
                    })
    return rows


def reference_table() -> list[dict[str, Any]]:
    rows = []
    paths = [p for p in source_files() if p.name == "reference_convergence.json"]
    for path in paths:
        d = read_json(path)
        cfg = d.get("configuration", {})
        for method in ("tdse", "qcle"):
            for obs, vals in d.get(method, {}).items():
                invalid = method == "qcle" and float(cfg.get("P0", -1)) == 100.0
                p0 = float(cfg.get("P0", 0.0))
                duration = float(cfg.get("dt_coarse", 0))*int(cfg.get("n_steps_coarse", 0))
                if method == "tdse":
                    # Reconstruct the actual adaptive grid selected by run_tdse,
                    # not merely the n_grid_min values stored as "tdse_grids".
                    r0, sigma_r, mass, hbar = -10.0, 1.0, 2000.0, 1.0
                    travel = abs(max(p0, 1.0)/mass) * duration
                    rlo = min(r0-6*sigma_r, r0-25.0)
                    rhi = max(r0+travel+6*sigma_r+25.0, r0+25.0)
                    lx = rhi-rlo
                    p_dyn = math.sqrt(p0*p0 + 2.0*mass*0.115)
                    sigma_p = hbar/(2*sigma_r)
                    kneed = (p_dyn+8*sigma_p)/hbar
                    requested = cfg.get("tdse_grids", [None, None])
                    actual = []
                    dr = []
                    for minimum in requested:
                        base = int(2 ** math.ceil(math.log2(2.0*lx*kneed/math.pi)))
                        ngrid = min(max(int(minimum), base), 32768)
                        actual.append(ngrid); dr.append(lx/ngrid)
                    coarse_req, fine_req = requested
                    coarse_actual, fine_actual = actual
                    coarse_spacing, fine_spacing = dr
                    domain_r, domain_p = [rlo, rhi], "FFT momentum; periodic spatial grid"
                else:
                    coarse_req, fine_req = cfg.get("qcle_grids", [None, None])
                    coarse_actual, fine_actual = coarse_req, fine_req
                    coarse_spacing = [50.0/coarse_actual[0], 70.0/coarse_actual[1]]
                    fine_spacing = [50.0/fine_actual[0], 70.0/fine_actual[1]]
                    domain_r, domain_p = [-25.0, 25.0], [-35.0, 35.0]
                rows.append({
                    "source_file": rel(path), "P0": cfg.get("P0"), "method": method.upper(),
                    "observable": obs, "coarse": vals.get("coarse"), "fine": vals.get("fine"),
                    "absolute_difference": vals.get("absolute_difference"),
                    "dt_coarse": cfg.get("dt_coarse"), "dt_fine": cfg.get("dt_fine"),
                    "coarse_requested_resolution": coarse_req,
                    "fine_requested_resolution": fine_req,
                    "coarse_actual_resolution": coarse_actual,
                    "fine_actual_resolution": fine_actual,
                    "coarse_grid_spacing": coarse_spacing, "fine_grid_spacing": fine_spacing,
                    "R_domain": domain_r, "P_domain_or_convention": domain_p,
                    "study_label": "one-step refinement (two levels), not convergence",
                    "reference_initial_R0": -10.0, "duration": duration,
                    "compatible_with_production_campaign": False,
                    "invalid_configuration": invalid,
                    "invalid_reason": "P0=100 lies outside grid-QCLE P domain [-35,35], yielding identically zero state" if invalid else "",
                    "cfl_diagnostic": "NOT COMPUTED/saved by run_qcle direct step loop" if method=="qcle" else "NOT APPLICABLE",
                    "boundary_probability_or_reflection": "DATA ABSENT",
                })
    return rows


def figure_metadata(run_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    figures = [p for p in source_files() if p.suffix.lower() in {".png", ".pdf", ".svg", ".jpg", ".jpeg"}]
    manifests = [(ROOT/r["run_directory"], r) for r in run_rows]
    yellow_hits: set[Path] = set()
    yellow_scanned: set[Path] = set()
    try:
        from PIL import Image
        target = np.array([240, 228, 66], dtype=int)  # Okabe-Ito yellow #F0E442
        for png in (p for p in figures if p.suffix.lower()==".png" and
                    ("qcle_correction" in rel(p).lower() or "flow_correction" in rel(p).lower())):
            rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)
            yellow_scanned.add(p)
            if np.any(np.max(np.abs(rgb-target), axis=2) <= 8):
                yellow_hits.add(p)
    except Exception:
        pass
    rows = []
    for path in figures:
        nearest = None
        for d, run in manifests:
            try:
                path.relative_to(d)
            except ValueError:
                continue
            if nearest is None or len(d.parts) > len(nearest[0].parts):
                nearest = (d, run)
        run = nearest[1] if nearest else {}
        low = rel(path).lower()
        if "comparison_se_qcle" in low:
            generator = "Compare_gp_se_qcle.py"
        elif "/results/p0_" in "/" + low:
            generator = "Visualization.py via run.py"
        elif "thesis_analysis" in low:
            generator = "thesis_analysis.py"
        else:
            generator = "NOT IDENTIFIABLE"
        sidecar = path.with_suffix(".meta.json")
        meta = {}
        if sidecar.exists():
            try:
                meta = read_json(sidecar)
            except Exception:
                pass
        correction_only = ("surrogate_health/qcle_correction" in low or "flow_correction" in low)
        png_twin = path if path.suffix.lower()==".png" else path.with_suffix(".png")
        if png_twin in yellow_scanned:
            yellow_status: Any = png_twin in yellow_hits
            yellow_evidence = "targeted RGB pixel scan for #F0E442 within tolerance 8"
        elif correction_only:
            yellow_status = "NOT VERIFIABLE"
            yellow_evidence = "no readable PNG twin for targeted scan"
        else:
            yellow_status = "NOT APPLICABLE"
            yellow_evidence = "not identified as a correction diagnostic"
        # Existing plotting code sets empty titles. Sidecar title is metadata and is not image-internal.
        rows.append({
            "filename": rel(path), "generating_script_function": generator,
            "source_run": run.get("run_directory", "NOT IDENTIFIABLE"),
            "method": "MIDPOINT" if "midpoint" in low else ("PBME" if "pbme" in low else "multiple/not identifiable"),
            "P0": run.get("P0", ""), "seed": run.get("seed", ""), "N": run.get("N_n_train", ""),
            "dt": run.get("dt", ""), "time_snapshot": infer_snapshot(path.name),
            "bandwidth": "see source data/NOT IDENTIFIABLE", "GP_policy": run.get("hyperparameter_refit_policy", ""),
            "regularization": run.get("gp_regularization", ""), "floor": run.get("density_profile_floor", ""),
            "threshold": "NOT IDENTIFIABLE", "density_scale": "NOT IDENTIFIABLE",
            "raw_or_normalized": infer_normalization(path.name),
            "positive_negative_density_convention": "sign-split only where filename/code identifies signed density; otherwise NOT IDENTIFIABLE",
            "reference_resolution": run.get("reference_grid_dimensions", ""),
            "internal_visible_title_violation": False,
            "title_evidence": "plotting-code audit found only set_title(\"\"); sidecar title is not rendered into image",
            "sidecar_metadata_file": rel(sidecar) if sidecar.exists() else "DATA ABSENT",
            "caption_standalone": bool(meta.get("caption")) and all(run.get(k,"") not in ("",None) for k in ("P0","seed","N_n_train","dt")),
            "caption": meta.get("caption", ""),
            "yellow_correction_curve": yellow_status,
            "yellow_curve_evidence": yellow_evidence,
            "discontinuous_line_style": False if correction_only else "NOT IDENTIFIABLE",
            "correction_only_panel": correction_only,
        })
    code_rows = []
    for source in ("Visualization.py", "Compare_gp_se_qcle.py", "thesis_analysis.py"):
        path = ROOT/source
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if re.search(r"\b(?:set_title|suptitle|plt\.title)\s*\(", line):
                empty = bool(re.search(r"(?:set_title|suptitle|plt\.title)\s*\(\s*[\"']\s*[\"']", line))
                code_rows.append({
                    "source_file": rel(path), "line": i, "code": line.strip(),
                    "visible_internal_title_violation": not empty,
                })
    return rows, code_rows


def infer_snapshot(name: str) -> str:
    m = re.search(r"(?:step|t)[_=-]?(\d+(?:\.\d+)?)", name, re.I)
    return m.group(1) if m else "NOT IDENTIFIABLE"


def infer_normalization(name: str) -> str:
    low = name.lower()
    if "raw" in low:
        return "raw"
    if "norm" in low or "population" in low or "observable" in low:
        return "normalized or self-normalized"
    return "NOT IDENTIFIABLE"


def gp_policy_table(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    representative = next((r for r in run_rows if "step9_repl_P020/seed11" in r["run_directory"]), {})
    policies = [
        ("kernel", "ARD squared-exponential/RBF", "ARD squared-exponential/RBF", "GP_Density.py:1211-1215 and kernel implementation"),
        ("surrogate architecture", "gp", representative.get("surrogate_type"), representative.get("manifest")),
        ("input standardization", False, False, representative.get("manifest")),
        ("density-label transformation", "physical labels", "product surrogate: rho_hat=g(x)mu(z), labels y/g with signed floor", "run.py:525-545; manifest"),
        ("absolute-target sampling", True, False, representative.get("manifest")),
        ("L2 regularization", 1e-6, 0.05, representative.get("manifest")),
        ("initial log sigma_n", -2.5, -10.0, representative.get("manifest")),
        ("sigma_n policy", "optimized", "optimized; clipped log sigma_n to [-8,1]", "GP_Density.py:118-140; manifest"),
        ("initial optimization", "full-batch L-BFGS/MLL", "full-batch L-BFGS/LOO-CV, n_opt_steps=250", "GP_Density.py:1415-1611; run.py focused override; manifest"),
        ("refit policy", "breathing", "adaptive (focused-mode runtime override)", "run.py:671-700; manifest"),
        ("refit optimizer", "projected L-BFGS, 25 outer steps", "adaptive trigger budget=5; refit_opt_steps=100 recorded", "GP_Density.py:1958-2215; manifest"),
        ("breathing anchor", "EWMA, MAD factor 0.4, beta 0.9", "EWMA, MAD factor 0.4, beta 0.9", representative.get("manifest")),
        ("lengthscale bounds", "log ell in [-2,2]", "log ell in [-2,2]", "GP_Density.py:137-140"),
        ("lengthscale prior", "weight 1.0, clip 1.0", "weight 0.1, clip 0.5", representative.get("manifest")),
        ("product profile floor", "1e-3 relative", "1e-3 relative", representative.get("manifest")),
        ("Cholesky jitter", "base 1e-6 with positive escalation", "base 1e-6 with positive escalation", "GP_Density.py:1215-1258"),
        ("moment constraints", True, "configured true; focused label information disables redundant KKT constraints", "run.py:671-700 and GP_Density.py pin/label contract"),
        ("coefficient normalization", "physical targets; no RNS", "physical targets; label-weighted observables self-normalized separately", "GP_Density.py:142-150; Dynamics.py raw drift definitions"),
        ("source evaluation", "QCLE Q applied to per-trajectory weights", "weight-label explicit midpoint/Heun; flow_fraction=0", "run.py:840-854; manifest"),
        ("midpoint evaluation", "intrinsic midpoint weight update", "intrinsic stage evaluation; optional pulled-back transported product not used", "run.py:448-467; manifest surrogate=product"),
        ("failure recovery", "best-state restore, optimizer reset, transactional adaptive rejection", "same implementation", "GP_Density.py:1492-1557, 1958-2215"),
        ("operator posterior variance", "not computed", "operator_variance_computed=0", "NPZ arrays"),
    ]
    return [{"policy_item": p, "code_default": d, "reviewer_campaign_actual": a, "source": s} for p,d,a,s in policies]


def contradiction_table() -> list[dict[str, Any]]:
    return [
        {
            "item": "production momentum label conflict",
            "claim_source": "results/P0_20 directory and production_contract.json key 20.0",
            "claim": "P0=20",
            "contradictory_evidence_source": "results/P0_20/run_manifest.json",
            "evidence": "cli_arguments.P0=40.0",
            "resolution": "manifest is authoritative for this run; relabel as P0=40 or quarantine the mislabeled directory",
        },
        {
            "item": "support campaign completeness",
            "claim_source": "THESIS_REVISION_HANDOFF.md:44,228; REVIEWER_RESPONSE_AND_THESIS_REVISION_GUIDE.md:161,263",
            "claim": "18/18 support runs complete/exist",
            "contradictory_evidence_source": "reviewer_closure_20260723_194254/step8_support_P0*/seed*_N*/",
            "evidence": "12/18 paired complete; six directories lack midpoint.npz",
            "resolution": "report 12 complete and six RUN INCOMPLETE",
        },
        {
            "item": "manufactured refinement trend",
            "claim_source": "THESIS_REVISION_HANDOFF.md:57-74; REVIEWER_RESPONSE_AND_THESIS_REVISION_GUIDE.md:111-115,185-195",
            "claim": "all quantities increase monotonically and increase exceeds seed spread",
            "contradictory_evidence_source": "reviewer_closure_20260726_174927/step5_manufactured/N*_seed*/manufactured_operator_metrics.json",
            "evidence": "all-seed on-support Q mean E2 is 0.0188632, 0.0305848, 0.0218893 for N=300,600,1200; seed SD is large at N=300/600",
            "resolution": "conclude no reproducible decreasing refinement trend; do not claim monotonic increase from a single seed",
        },
        {
            "item": "P0=100 baseline metric availability",
            "claim_source": "THESIS_REVISION_HANDOFF.md:118-123",
            "claim": "P0=100 E2 and Einf unavailable",
            "contradictory_evidence_source": "reviewer_closure_20260726_174927/step11_baseline/step9_repl_P0100_seed11_pbme/kde_gp_identical_support.json",
            "evidence": "E2=0.00036430185717972746 and Einf=0.0006639177058668476 are stored",
            "resolution": "report the stored values with exact provenance",
        },
        {
            "item": "trace conservation interpretation",
            "claim_source": "self-normalized lw_trace arrays/figures",
            "claim": "trace remains near one",
            "contradictory_evidence_source": "step9 MIDPOINT raw_norm_drift and raw_trace_drift arrays",
            "evidence": "near-unit self-normalized trace coexists with raw drifts up to 8.69e20",
            "resolution": "label lw curves self-normalized and use raw_* arrays for conservation",
        },
        {
            "item": "P0=100 grid-QCLE reference",
            "claim_source": "step12_reference/P0100/reference_convergence.json",
            "claim": "coarse/fine QCLE endpoints agree at zero",
            "contradictory_evidence_source": "ReviewerValidation.py reference configuration and qcle_grid_tully.py initial Gaussian",
            "evidence": "P domain [-35,35] excludes P0=100; initial grid density underflows to zero",
            "resolution": "mark the P0=100 QCLE reference invalid, not converged/agreed",
        },
        {
            "item": "TDSE spatial refinement at P0=100",
            "claim_source": "reference_convergence.json tdse_grids=[2048,4096]",
            "claim": "spatial and temporal grids are both refined",
            "contradictory_evidence_source": "Compare_gp_se_qcle.py:532-580 adaptive grid formula",
            "evidence": "actual adaptive grid is 4096 points at both P0=100 levels; only dt changes",
            "resolution": "describe P0=100 TDSE as a temporal one-step refinement with unchanged actual spatial grid",
        },
    ]


def checksums_for_outputs() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(AUDIT.rglob("*")):
        if path.is_file() and path.name != "checksums_sha256.csv":
            rows.append({"scope": "generated", "path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    return rows


def create_plots(manufactured_summary: list[dict[str, Any]], projection_summary: list[dict[str, Any]], conservation: list[dict[str, Any]]) -> None:
    if plt is None:
        return
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    # Manufactured Q E2.
    rows = [r for r in manufactured_summary if r.get("canonical_latest_campaign") and r["query_set"]=="on_support" and r["quantity"]=="operator_Q"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = [r["N"] for r in rows]
    y = [r["mean_E2"] for r in rows]
    sd = [r["sample_sd_E2"] if math.isfinite(float(r["sample_sd_E2"])) else 0 for r in rows]
    ax.errorbar(x, y, yerr=sd, marker="o", color="#0072B2", capsize=3, label="seed mean ± sample SD")
    ax.set_xscale("log", base=2); ax.set_xlabel("Training support N"); ax.set_ylabel("Relative L2 error E2")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(PLOTS/"manufactured_operator_E2.png", dpi=220)
    fig.savefig(PLOTS/"manufactured_operator_E2.pdf")
    plt.close(fig)
    # Canonical leakage.
    rows = sorted([r for r in projection_summary if r.get("canonical")], key=lambda r: r["P0"])
    if rows:
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        xp = np.arange(len(rows))
        ax.bar(xp, [r["mean"] for r in rows], color="#56B4E9", label="mean")
        ax.scatter(xp, [r["median"] for r in rows], color="#000000", marker="_", s=130, label="median")
        ax.scatter(xp, [r["maximum"] for r in rows], color="#D55E00", marker="D", s=24, label="maximum")
        ax.set_xticks(xp, [f"P0={r['P0']}" for r in rows]); ax.set_ylabel("Relative L2 projection residual")
        ax.legend(frameon=False); fig.tight_layout()
        fig.savefig(PLOTS/"seo_projection_leakage.png", dpi=220)
        fig.savefig(PLOTS/"seo_projection_leakage.pdf")
        plt.close(fig)
    # Replication campaign raw norm max drift (log scale).
    rows = [r for r in conservation if r.get("run_directory","").startswith("reviewer_closure_20260726_174927/step9_repl_") and r["quantity"]=="normalization"]
    if rows:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for j, method in enumerate(("pbme","midpoint")):
            sub = [r for r in rows if r["method"]==method]
            xp = np.arange(len(sub)) + (-0.18 if method=="pbme" else 0.18)
            vals = [max(float(r["maximum_absolute_drift"]), 1e-30) for r in sub]
            ax.bar(xp, vals, width=.34, label=method.upper(), color="#0072B2" if method=="pbme" else "#D55E00")
        labels = [f"{int(r['P0'])}/{int(r['seed'])}" for r in rows if r["method"]=="pbme"]
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        ax.set_yscale("log"); ax.set_ylabel("Maximum |raw normalization drift|")
        ax.set_xlabel("P0 / seed"); ax.legend(frameon=False); fig.tight_layout()
        fig.savefig(PLOTS/"raw_normalization_drift_replication.png", dpi=220)
        fig.savefig(PLOTS/"raw_normalization_drift_replication.pdf")
        plt.close(fig)
    # Four-seed population trajectories: mean and descriptive t interval.
    replication_rows = read_csv_table(TABLES/"seed_replication_trajectories.csv")
    if replication_rows:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=False)
        for ax, p0 in zip(axes, ("20", "100")):
            for method, color, line_style in (
                ("pbme", "#0072B2", "-"),
                ("midpoint", "#D55E00", "--"),
            ):
                sub = sorted(
                    [
                        r for r in replication_rows
                        if r.get("P0") == p0
                        and r.get("method") == method
                        and r.get("observable") == "lw_P0"
                    ],
                    key=lambda r: float(r["t_over_tc"]),
                )
                tau = np.asarray([float(r["t_over_tc"]) for r in sub])
                mean = np.asarray([float(r["mean"]) for r in sub])
                lo = np.asarray([float(r["ci95_lower"]) for r in sub])
                hi = np.asarray([float(r["ci95_upper"]) for r in sub])
                ax.plot(
                    tau, mean, color=color, linestyle=line_style,
                    linewidth=1.7, label=method.upper(),
                )
                ax.fill_between(tau, lo, hi, color=color, alpha=0.14)
            ax.set_xlabel(r"$t/t_c$")
            ax.set_ylabel(r"Self-normalized $P_0$ population")
            ax.text(
                0.03, 0.95, f"$P_0={p0}$", transform=ax.transAxes,
                va="top", ha="left",
            )
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(PLOTS/"seed_replication_lw_P0.png", dpi=220)
        fig.savefig(PLOTS/"seed_replication_lw_P0.pdf")
        plt.close(fig)
    # Three saved leakage snapshots are discrete comparisons, not a trend.
    leakage_rows = read_csv_table(
        TABLES/"seo_projection_leakage_seed_summary.csv"
    )
    if leakage_rows:
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
        for ax, p0 in zip(axes, ("20", "100")):
            tau_values = (0.0, 1.0, 2.0)
            x = np.arange(len(tau_values), dtype=float)
            for method, color, offset, hatch in (
                ("pbme", "#0072B2", -0.18, ""),
                ("midpoint", "#D55E00", 0.18, "//"),
            ):
                sub = {
                    float(r["t_over_tc"]): r for r in leakage_rows
                    if r.get("P0") == p0 and r.get("method") == method
                }
                means = np.asarray(
                    [float(sub[t]["mean_of_seed_mean_leakage"]) for t in tau_values]
                )
                se = np.asarray(
                    [float(sub[t]["standard_error_across_seed_means"]) for t in tau_values]
                )
                ax.bar(
                    x + offset, means, width=0.34, color=color, alpha=0.75,
                    edgecolor="#222222", linewidth=0.6, hatch=hatch,
                    label=method.upper(),
                )
                ax.errorbar(
                    x + offset, means, yerr=TCRIT_975[3] * se, fmt="none",
                    ecolor="#222222", elinewidth=0.8, capsize=2,
                )
            ax.set_xticks(x, ["0", "1", "2"])
            ax.text(
                0.03, 0.96, f"$P_0={p0}$", transform=ax.transAxes,
                va="top", ha="left",
            )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, frameon=False, ncol=2, loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
        )
        fig.supxlabel(r"Saved snapshot $t/t_c$", y=0.04)
        fig.supylabel("Relative L2 projection residual", x=0.015)
        fig.subplots_adjust(left=0.10, right=0.99, bottom=0.20, top=0.82, wspace=0.18)
        fig.savefig(PLOTS/"seo_projection_leakage_replication_snapshots.png", dpi=220)
        fig.savefig(PLOTS/"seo_projection_leakage_replication_snapshots.pdf")
        plt.close(fig)


def write_run_inventory_md(expected: list[dict[str, Any]], run_rows: list[dict[str, Any]]) -> None:
    counts = Counter(r["status"] for r in expected)
    text = f"""# Run inventory

Generated by `scripts/audit_pipeline.py`. Completion requires a readable manifest,
both method arrays, a final saved step equal to `n_steps`, a final time equal to
the manifest target, a strictly increasing time grid, and finite core arrays.

## Expected campaign summary

- Expected configuration groups: {len(expected)}
- Complete: {counts['COMPLETE']}
- Incomplete: {counts['INCOMPLETE']}
- Failed (no usable method output): {counts['FAILED']}
- Missing configuration groups: {counts['MISSING']}
- Configuration conflicts: {counts['CONFIGURATION CONFLICT']}

## Expected status matrix

{md_table(expected, ['campaign','P0','seed','N','dt','pbme_complete','midpoint_complete','status','missing_outputs'])}

## All discovered run manifests

The machine-readable inventory contains {len(run_rows)} discovered run manifests.
Duplicate counts mean matching numerical/configuration keys, not necessarily
byte-identical results.

See `run_inventory.csv` for all fields and exact source paths.
"""
    (AUDIT/"run_inventory.md").write_text(text, encoding="utf-8")


def build_report(
    file_rows: list[dict[str, Any]], run_rows: list[dict[str, Any]], expected: list[dict[str, Any]],
    stability: list[dict[str, Any]], manufactured: list[dict[str, Any]], manufactured_summary: list[dict[str, Any]],
    projection_summary: list[dict[str, Any]], baseline: list[dict[str, Any]], time_ref: list[dict[str, Any]],
    support_summary: list[dict[str, Any]], replication: list[dict[str, Any]], conservation: list[dict[str, Any]],
    comparison: list[dict[str, Any]], references: list[dict[str, Any]], figures: list[dict[str, Any]],
    gp_policy: list[dict[str, Any]],
) -> str:
    counts = Counter(r["status"] for r in expected)
    dispersion_rows = read_csv_table(
        TABLES / "seed_replication_method_dispersion_ratios.csv"
    )
    projection_seed_rows = read_csv_table(
        TABLES / "seo_projection_leakage_seed_summary.csv"
    )
    baseline_snapshot_rows = read_csv_table(
        TABLES / "kde_gp_identical_support_all_snapshots.csv"
    )
    population_rows = read_csv_table(TABLES / "population_physicality_audit.csv")
    health_rows = read_csv_table(TABLES / "replication_gp_health.csv")
    physical = [r for r in file_rows if r["storage"]=="filesystem"]
    kinds = Counter(r["classification"] for r in physical)
    fatal_logs = [r for r in stability if r.get("issue")=="failed attempt log"]
    critical_npz = [r for r in stability if str(r.get("severity","")).startswith("EXCLUDE")]
    canonical_mfg = [r for r in manufactured_summary if r.get("canonical_latest_campaign") and r["query_set"]=="on_support" and r["quantity"]=="operator_Q"]
    canonical_proj = [r for r in projection_summary if r.get("canonical")]
    canonical_base = [r for r in baseline if r.get("canonical")]
    repl_mid_norm = [r for r in conservation if r.get("run_directory","").startswith("reviewer_closure_20260726_174927/step9_repl_") and r["method"]=="midpoint" and r["quantity"]=="normalization"]
    repl_pb_norm = [r for r in conservation if r.get("run_directory","").startswith("reviewer_closure_20260726_174927/step9_repl_") and r["method"]=="pbme" and r["quantity"]=="normalization"]
    max_mid_norm = max((abs(float(r["maximum_absolute_drift"])) for r in repl_mid_norm), default=float("nan"))
    max_pb_norm = max((abs(float(r["maximum_absolute_drift"])) for r in repl_pb_norm), default=float("nan"))
    mfg_table = [{
        "N": r["N"], "n": r["n_independent_seeds"], "mean_E2": fmt(r["mean_E2"]),
        "sample_SD": fmt(r["sample_sd_E2"]), "% change": fmt(r.get("percent_change_from_previous_N"))
    } for r in canonical_mfg]
    proj_table = [{"P0": r["P0"], "mean": fmt(r["mean"]), "median": fmt(r["median"]), "SD": fmt(r["sample_sd"]), "max": fmt(r["maximum"])} for r in canonical_proj]
    base_table = [{"P0": r["P0"], "seed": r["seed"], "E1": fmt(r["E1"]), "E2": fmt(r["E2"]), "Einf": fmt(r["Einf"]), "pass": r["passed"]} for r in canonical_base]
    ref_p100_zero = any(r["invalid_configuration"] for r in references)
    order_computed = [r for r in time_ref if isinstance(r.get("empirical_order_p"), (int,float)) and r["dt"]==0.5]
    support_n2000_mid = [r for r in support_summary if r["method"]=="midpoint" and r["N"]==2000]
    fig_missing = sum(1 for r in figures if not r["caption_standalone"])
    yellow_count = sum(r.get("yellow_correction_curve") is True for r in figures)
    correction_only_count = sum(r.get("correction_only_panel") is True for r in figures)
    time_pop = [{
        "P0": r["P0"], "seed": r["seed"], "method": r["method"],
        "dt": r["dt"], "|endpoint-finest|": fmt(r["endpoint_absolute_difference"]),
        "max|difference|": fmt(r["maximum_in_time_absolute_difference"]),
        "L2": fmt(r["time_integrated_L2_difference"]),
        "p": fmt(r["empirical_order_p"]) if r["empirical_order_p"] != "NOT COMPUTED" else "NOT COMPUTED",
    } for r in time_ref if r["observable"]=="lw_P0" and r["dt"]==0.5]
    support_pop = [{
        "P0": r["P0"], "method": r["method"], "N": r["N"],
        "n": r["n_independent_seeds"], "mean": fmt(r["mean"]),
        "SD": fmt(r["sample_sd"]), "SE": fmt(r["standard_error"]),
        "95% CI": f"[{fmt(r['ci95_lower'])}, {fmt(r['ci95_upper'])}]" if r["n_independent_seeds"]>1 else "NOT COMPUTED",
        "mean-N2000": fmt(r["difference_from_N2000_seed_mean"]),
    } for r in support_summary if r["observable"]=="lw_P0"]
    repl_pop = [{
        "P0": r["P0"], "method": r["method"], "n": r["n_independent_seeds"],
        "mean": fmt(r["mean"]), "SD": fmt(r["sample_sd"]), "SE": fmt(r["standard_error"]),
        "95% CI": f"[{fmt(r['ci95_lower'])}, {fmt(r['ci95_upper'])}]",
        "spread": fmt(r["spread"]),
    } for r in replication if r["observable"]=="lw_P0" and math.isclose(float(r["t_over_tc"]), 2.0)]
    conservation_repl = [{
        "P0": int(float(r["P0"])), "seed": int(float(r["seed"])), "method": r["method"],
        "endpoint raw dN": fmt(r["endpoint_drift"]), "max |raw dN|": fmt(r["maximum_absolute_drift"]),
        "RMS": fmt(r["rms_drift"]), "behavior": r["behavior"],
    } for r in conservation if r.get("run_directory","").startswith("reviewer_closure_20260726_174927/step9_repl_") and r["quantity"]=="normalization"]
    comparison_pop = [{
        "P0": r["P0"], "seed": r["seed"],
        "PBME endpoint": fmt(r["PBME_endpoint"]), "MIDPOINT endpoint": fmt(r["MIDPOINT_endpoint"]),
        "MID-PB endpoint": fmt(r["MIDPOINT_minus_PBME_endpoint_signed_difference"]),
        "reference error delta": r["reference_error_difference"],
    } for r in comparison if r["observable"]=="lw_P0"]
    reference_trace = [{
        "P0": int(float(r["P0"])), "method": r["method"], "coarse": fmt(r["coarse"]),
        "fine": fmt(r["fine"]), "|difference|": fmt(r["absolute_difference"]),
        "valid": not r["invalid_configuration"], "source": r["source_file"],
    } for r in references if r["observable"]=="trace" and "reviewer_closure_20260726_174927" in r["source_file"]]
    dispersion_pop = [{
        "P0": r["P0"],
        "PBME endpoint spread": fmt(float(r["PBME_endpoint_spread"])),
        "MIDPOINT endpoint spread": fmt(float(r["MIDPOINT_endpoint_spread"])),
        "endpoint spread ratio": r["MIDPOINT_to_PBME_endpoint_spread_ratio"],
        "PBME pairwise L2": fmt(float(r["PBME_mean_pairwise_time_L2"])),
        "MIDPOINT pairwise L2": fmt(float(r["MIDPOINT_mean_pairwise_time_L2"])),
        "pairwise L2 ratio": r["MIDPOINT_to_PBME_pairwise_L2_ratio"],
    } for r in dispersion_rows if r.get("observable") == "lw_P0"]
    projection_time = [{
        "P0": r["P0"], "method": r["method"], "t/tc": r["t_over_tc"],
        "n seeds": r["n_independent_propagation_seeds"],
        "mean leakage": fmt(float(r["mean_of_seed_mean_leakage"])),
        "seed SD": fmt(float(r["sample_sd_across_seed_means"])),
        "95% CI": (
            f"[{fmt(float(r['ci95_lower']))}, {fmt(float(r['ci95_upper']))}]"
        ),
        "max anchor": fmt(float(r["maximum_anchor_leakage"])),
    } for r in projection_seed_rows]
    baseline_snapshot_summary = []
    for p0 in ("20", "100"):
        rows = [
            r for r in baseline_snapshot_rows
            if r.get("P0") == p0 and r.get("method") == "pbme"
        ]
        baseline_snapshot_summary.append({
            "P0": p0,
            "cases": len(rows),
            "maximum E1": fmt(max(float(r["E1"]) for r in rows)) if rows else "DATA ABSENT",
            "maximum E2": fmt(max(float(r["E2"]) for r in rows)) if rows else "DATA ABSENT",
            "maximum Einf": fmt(max(float(r["Einf"]) for r in rows)) if rows else "DATA ABSENT",
            "passes / applicable": (
                f"{sum(r.get('passed_declared_threshold','').lower() == 'true' for r in rows)}/{len(rows)}"
                if rows else "DATA ABSENT"
            ),
        })
    physicality_summary = []
    for p0 in ("20", "100"):
        for method in ("pbme", "midpoint"):
            rows = [
                r for r in population_rows
                if r.get("P0") == p0 and r.get("method") == method
            ]
            affected = {
                r["seed"] for r in rows
                if int(float(r["n_outside_0_1_tolerance"])) > 0
            }
            physicality_summary.append({
                "P0": p0, "method": method,
                "affected seeds": f"{len(affected)}/4",
                "out-of-range saved values": sum(
                    int(float(r["n_outside_0_1_tolerance"])) for r in rows
                ),
                "minimum": fmt(min(float(r["minimum"]) for r in rows)),
                "maximum": fmt(max(float(r["maximum"]) for r in rows)),
            })
    health_midpoint = [{
        "P0": r["P0"], "seed": r["seed"],
        "max |alpha|": fmt(float(r["alpha_linf_max_abs"])),
        "min ESS fraction": fmt(float(r["sw_abs_ess_fraction_min"])),
        "max |Q| RMS": fmt(float(r["cs_q_rms_max_abs"])),
        "max |raw dN|": fmt(float(r["raw_norm_max_abs_drift"])),
        "max |relative dE|": fmt(float(r["raw_energy_max_abs_relative_drift"])),
    } for r in health_rows if r.get("method") == "midpoint"]
    report = f"""# Pipeline Data Audit and Thesis Evidence

**Audit root:** `{ROOT}`  
**Generated:** {datetime.now(timezone.utc).isoformat()}  
**Scope:** recursive read-only evidence extraction; every generated artifact is under `reviewer_data_audit/`.

## 1. Executive conclusion

The campaign is not a successful validation of systematic MIDPOINT improvement.
All {len(expected)} expected configuration directories were discovered, but only
{counts['COMPLETE']} are complete paired PBME/MIDPOINT runs. Six support-refinement
configurations are incomplete because `midpoint.npz` is absent. There are
{counts['FAILED']} final configuration groups with neither usable method output and
{counts['MISSING']} missing expected groups.

The strongest negative result is raw conservation. In the four-seed production-like
replication campaign, MIDPOINT maximum absolute raw-normalization drift reaches
{fmt(max_mid_norm)}, whereas PBME reaches {fmt(max_pb_norm)}. The high-momentum
MIDPOINT arrays remain finite and run to the declared final time, but raw
normalization/trace and energy can grow by many orders of magnitude. The near-unit
`lw_trace` is self-normalized and must not be presented as raw conservation.

Manufactured-operator testing is complete only for the stored relative L2 metric
E2. Seed-aggregated on-training-support operator E2 is non-monotonic with N and does
not demonstrate refinement. SEO projection residuals are substantial and much
larger at P0=100. The expanded identical-support PBME KDE/2D-projected-GP baseline
passes its predeclared 2% E1 threshold in all 24 replication snapshots
(two momenta, four seeds, and three times). No compatible TDSE or grid-QCLE
trajectory exists for a valid production-condition error comparison, so
`E_MIDPOINT,ref - E_PBME,ref` is **NOT COMPUTED** and improvement is not demonstrated.

## 2. Directory and data provenance

The physical filesystem inventory contains {len(physical)} files before generated
audit outputs. It includes {kinds['run_manifest']} run manifests,
{kinds['pbme_output']} PBME NPZ outputs, {kinds['midpoint_output']} MIDPOINT NPZ
outputs, {kinds['figure']} figures, and {kinds['text_log']} logs. ZIP members are
listed separately in `file_inventory.csv` and are not counted as independent
simulations when they duplicate filesystem artifacts.

Every number in this document is traceable through `metric_provenance.csv`, the
source columns of the quantitative CSVs, and `checksums_sha256.csv`. Filesystem
modification times are only proxies when manifests omit explicit start/end times.

## 3. Exact campaign inventory

{md_table([
    {"campaign": c, "expected": sum(r["campaign"]==c for r in expected),
     "complete": sum(r["campaign"]==c and r["status"]=="COMPLETE" for r in expected),
     "incomplete": sum(r["campaign"]==c and r["status"]=="INCOMPLETE" for r in expected),
     "failed": sum(r["campaign"]==c and r["status"]=="FAILED" for r in expected),
     "missing": sum(r["campaign"]==c and r["status"]=="MISSING" for r in expected)}
    for c in ("A_time_step","B_support","C_replication")
], ['campaign','expected','complete','incomplete','failed','missing'])}

The six incomplete configurations are:

{md_table([r for r in expected if r["status"]!="COMPLETE"], ['campaign','P0','seed','N','dt','status','missing_outputs','run_directory'])}

Each of these six is classified **RUN INCOMPLETE**: PBME data were saved, but the
paired MIDPOINT data are absent.

Completion here means that the simulation was run to the saved final step, data
were saved, core arrays are finite, and the time grid is complete. It does not mean
that a scientific validation passed. Fatal attempt logs ({len(fatal_logs)}) are
retained in `numerical_stability_audit.csv`; successful retries do not erase them.

## 4. Numerical-stability audit

The audit found {len(stability)} logged/array stability findings, including
{len(fatal_logs)} fatal attempt logs and {len(critical_npz)} file/quantity exclusion
findings. Initial-step or inactive-channel NaNs are explicitly labelled
informational placeholders; they are not silently treated as propagation failure.

High-momentum MIDPOINT is the decisive instability: correction norms, raw norm,
raw trace, and raw energy explode in every P0=100 replication seed. Low-momentum
MIDPOINT is also not conservative: raw normalization drifts from O(10^-1) through
O(1), depending on seed. The arrays are finite, so “run completed” and “validation
passed” must remain separate statements.

Exact first affected indices, times, quantities, continuation status, usability,
and exclusion rules are in `numerical_stability_audit.csv`. GP noise frequently
hits the configured upper bound exp(1); this is a surrogate-health warning, not a
physical-error estimate.

## 5. Manufactured-operator results

The analytic test uses a Gaussian nuclear density multiplied by the two-state SEO
profile, its analytic gradient, and analytic Q. The “off_support” set is an
independent out-of-training sample from the same distribution, not a deliberately
distant geometric shell. Each N/seed generates its own query set. Configuration:
N in {{300,600,1200,2400}}, seeds {{123,124,125}} where completed, 1000 queries,
frozen hyperparameters, no moment constraints, and L2 regularization 1e-6.

Canonical on-support operator results:

{md_table(mfg_table, ['N','n','mean_E2','sample_SD','% change'])}

Stored JSONs contain RMSE, absolute Linf, and relative L2. They do not contain the
arrays or reference denominators needed for relative E1 or relative Einf, so those
metrics are **NOT COMPUTED**. Conditioning metrics are **DATA ABSENT**. PASS/FAIL:
**COMPLETE — FAILED** as a refinement validation, because error does not decrease
reproducibly with N across seeds. Conditioning as a cause remains a hypothesis.

## 6. SEO projection leakage

The diagnostic projects the fitted surrogate at 20 bath anchors and 400 mapping
probes onto a rank-4 real SEO image basis (active diagonal, other diagonal,
symmetric off-diagonal, antisymmetric off-diagonal, each with the SEO envelope).
The relative residual is `||y-Bc||2 / max(||y||2,1e-30)`. It is a diagnostic
least-squares projection; it is not enforced.

The saved replication snapshots permit an expanded analysis at t/tc=0, 1, and 2
for PBME and MIDPOINT. The uncertainty below is across four independent
propagation seeds; the 20 anchors within each run are diagnostic probes and are
not treated as independent replicates.

{md_table(projection_time, ['P0','method','t/tc','n seeds','mean leakage','seed SD','95% CI','max anchor'])}

The common t=0 result is expected because paired PBME and MIDPOINT runs share the
same fitted initial surrogate. At P0=100 both propagated representations have
mean leakage near 0.96 at t/tc=1 and 2. At P0=20, final MIDPOINT leakage is larger
and more seed-variable than final PBME leakage. These data distinguish fitted
surrogate leakage by propagated method and time, but they do not isolate a causal
increment from PBME propagation, and projection was never enforced.

## 7. Identical-support KDE/GP baseline

The expanded comparison uses each PBME source, exactly its saved support and frozen
geometric measure, Scott/Silverman d=2 bandwidths, a common 120x120 R-P grid, an
analytic common raw-mass constraint, and the same snapshot. It covers four seeds
and t/tc=0, 1, and 2:

{md_table(baseline_snapshot_summary, ['P0','cases','maximum E1','maximum E2','maximum Einf','passes / applicable'])}

The predeclared criterion E1 <= 0.02 applies to PBME and all 24 PBME cases pass.
MIDPOINT reconstruction metrics are also saved, but the recorded PBME acceptance
criterion is not retrospectively applied to them. These are reconstruction-
agreement metrics, not physical errors and not evidence that MIDPOINT improves
PBME.

## 8. Time-step analysis

All 12 expected time-step configurations have three saved levels. The audit aligns
the finest trajectory onto coarser saved times by linear interpolation and never
extrapolates. Endpoint, maximum-in-time, time-normalized integral L1/L2 differences,
and finest-level differences are in `time_step_refinement_metrics.csv`.

An empirical order is reported only when all three levels exist, the fine-level
difference exceeds 1e-14, and the refinement signal exceeds the independent-seed
difference. {len(order_computed)} observable/method/momentum/seed combinations pass
that audit gate. The other orders are **NOT COMPUTED** with a row-specific reason.
Even computed p values are numerical self-consistency indicators, not evidence of
accuracy against a physical reference.

Selected final-population (`lw_P0`, self-normalized) coarse-versus-finest results:

{md_table(time_pop, ['P0','seed','method','dt','|endpoint-finest|','max|difference|','L2','p'])}

## 9. Support-size analysis

The N=500,1000,2000 clouds are independently sampled; no recorded prefix/nesting
policy exists. Therefore the tables compare endpoint distributions across
independent seeds and do not make pointwise trajectory or deterministic support
convergence claims. Means, sample SD, standard error, and two-sided Student-t 95%
intervals are in `support_size_summary.csv`.

MIDPOINT lacks N=2000 for four configurations and also lacks two N=1000
outputs, leaving only {max((r['n_independent_seeds'] for r in support_n2000_mid), default=0)}
usable seeds in affected summaries. Changes with N cannot be cleanly separated
from seed variation, and a stable support trend is not established.

Selected endpoint `lw_P0` summaries:

{md_table(support_pop, ['P0','method','N','n','mean','SD','SE','95% CI','mean-N2000'])}

## 10. Independent-seed replication

For each method, momentum, and observable, four independent seeds are interpolated
to 41 common values of t/tc in [0,2]. `seed_replication_trajectories.csv` reports
mean, sample SD, standard error, Student-t 95% CI (df=3), min, max, spread, and CV
where meaningful. The independent sample size is four—not the trajectory count.
These intervals are descriptive and do not constitute strong uncertainty
calibration.

Endpoint (`t/tc=2`) `lw_P0` replication summaries:

{md_table(repl_pop, ['P0','method','n','mean','SD','SE','95% CI','spread'])}

For a direct threshold-free reliability comparison, all six independent seed
pairs were compared over the same 41-point t/tc grid. The L2 quantity below is
the square root of the time-average of the squared pairwise difference. A ratio
greater than one means MIDPOINT is more seed-sensitive than PBME:

{md_table(dispersion_pop, ['P0','PBME endpoint spread','MIDPOINT endpoint spread','endpoint spread ratio','PBME pairwise L2','MIDPOINT pairwise L2','pairwise L2 ratio'])}

For `lw_P0`, the MIDPOINT/PBME mean pairwise-L2 dispersion ratio is approximately
66.5 at P0=20 and 133 at P0=100. These ratios do not depend on a post-hoc pass
threshold. Four seeds remain too few for strong uncertainty calibration, but they
are sufficient to show that the observed corrected trajectories are substantially
more seed-sensitive than the paired PBME trajectories.

Population physicality was checked independently of normalized population-sum
identity:

{md_table(physicality_summary, ['P0','method','affected seeds','out-of-range saved values','minimum','maximum'])}

Small PBME excursions are at the roughly 0.002 level. MIDPOINT excursions reach
large negative and greater-than-one values, especially at P0=100. Signed-estimator
values are retained, but they cannot be described as physical probabilities when
outside [0,1].

## 11. Raw conservation

`raw_conservation.csv` uses only `raw_norm_drift`, `raw_trace_drift`, and
`raw_energy_drift`, defined in the implementation as current pre-self-normalization
cloud estimators minus their step-0 values. It reports endpoint, maximum, RMS,
relative drift, time of maximum, and monotonic/oscillatory behavior.

Self-normalized `lw_*` curves are labelled self-normalized. In particular, a
near-unit `lw_trace` can coexist with catastrophic raw mass/trace drift, as it does
for high-momentum MIDPOINT. Raw conservation therefore fails for the corrected
method under the available production-like campaign.

Four-seed raw-normalization results:

{md_table(conservation_repl, ['P0','seed','method','endpoint raw dN','max |raw dN|','RMS','behavior'])}

Saved surrogate-health and correction diagnostics for the same MIDPOINT runs are:

{md_table(health_midpoint, ['P0','seed','max |alpha|','min ESS fraction','max |Q| RMS','max |raw dN|','max |relative dE|'])}

No adaptive-refit failure flag is set in these rows; numerical completion therefore
does not imply that the resulting surrogate coefficients, correction, or raw
conservation remained controlled.

## 12. TDSE and grid-QCLE numerical controls

TDSE uses a second-order V/2-T-V/2 split operator on a periodic FFT grid, a
dynamically sized spatial box, diabatic Gaussian initial packet, and no absorber.
Grid QCLE uses cell-centred periodic R-P grids, Fourier-pseudospectral derivatives,
and classical RK4. A CFL estimator exists in code, but the reference JSON does not
save its value. The reference artifacts store endpoints only.

Only two levels were run: TDSE 2048/dt=0.2 versus 4096/dt=0.1, and grid QCLE
192x128/dt=0.2 versus 384x256/dt=0.1. These are one-step refinement differences,
not convergence studies or observed-order estimates.

The TDSE values 2048/4096 in the JSON are minimum requested grids. Reconstructing
the adaptive grid logic shows that P0=100 used 4096 points at both levels, so that
case refines time step but not the actual spatial resolution. Exact domains,
actual grid sizes, and spacings are in `reference_refinement.csv`.

The reference run uses R0=-10 and t_final=40, while production campaigns use R0=-15
and t_final=2tc. Moreover, the P0=100 grid-QCLE reference uses P in [-35,35], so the
P0=100 initial packet lies outside the grid and every saved QCLE endpoint is zero
({ref_p100_zero}). That is invalid configuration, not numerical agreement. TDSE is
model-exact only within its numerical controls; grid QCLE solves the approximate
QCLE equation.

Stored canonical trace endpoints:

{md_table(reference_trace, ['P0','method','coarse','fine','|difference|','valid','source'])}

## 13. PBME-versus-MIDPOINT error comparison

Paired PBME/MIDPOINT differences at common times are extracted in
`midpoint_vs_pbme_paired_differences.csv`. A physical error difference
`E_MIDPOINT,ref - E_PBME,ref` is **NOT COMPUTED** because no saved reference shares
the production initial position, full-scattering time grid, and normalization.
The correction is non-negligible in unstable regimes, but its effects are neither
conservative nor reproducible across seeds. Therefore systematic MIDPOINT
improvement is **not demonstrated**.

Paired endpoint `lw_P0` values (these are method differences, not reference-error
differences):

{md_table(comparison_pop, ['P0','seed','PBME endpoint','MIDPOINT endpoint','MID-PB endpoint','reference error delta'])}

## 14. GP policy reconstructed from code

{md_table(gp_policy, ['policy_item','code_default','reviewer_campaign_actual','source'])}

Production manifests are authoritative per run. A crucial runtime override changes
focused-mode requests from breathing to adaptive refits. Operator posterior variance
is explicitly not computed; LOO/training residuals and R2 are surrogate-fit
diagnostics, not physical-error estimates.

## 15. Figure and caption audit

`figure_metadata.csv` inventories {len(figures)} figure files. Current plotting
code contains only empty internal-title calls, so no code-level visible title
violation is established. {fig_missing} figure rows lack enough sidecar/run metadata
to be classified as stand-alone captions. A targeted RGB scan of correction PNGs
found {yellow_count} yellow (`#F0E442`) correction figures. Current correction code
uses solid blue/magenta/green curves, while source comments document that a prior
yellow broken-line figure was replaced. {correction_only_count} figure files are
identified by path/code as correction-only diagnostics and therefore require a
baseline/full-result companion or explicit caption qualification. Details are in
`figure_caption_audit.md`.

## 16. Examiner checklist

| Examiner item | Status | Evidence |
|---|---|---|
| Manufactured Q fidelity | COMPLETE — FAILED | E2 exists; no reproducible refinement; E1/Einf absent |
| SEO projection leakage | COMPLETE — FAILED | four-seed, three-snapshot residuals are large; projection not enforced |
| Time-step refinement | COMPLETE — FAILED | three levels exist, but physical accuracy/stability not established |
| Support-size refinement | PARTIAL — DATA EXIST, ANALYSIS INCOMPLETE | six MIDPOINT outputs absent; clouds independent |
| Independent-seed replication | COMPLETE — FAILED | four seeds expose strong MIDPOINT instability/variation |
| Raw conservation | COMPLETE — FAILED | MIDPOINT raw drift catastrophic; normalized trace is not raw |
| Common-support KDE/GP | COMPLETE — PASSED | all 24 PBME replication snapshots pass E1<=0.02 |
| TDSE refinement | PARTIAL — DATA EXIST, ANALYSIS INCOMPLETE | two levels/endpoints only; incompatible production setup |
| Grid-QCLE refinement | COMPLETE — FAILED | two levels only; P0=100 grid invalid |
| Exact production configuration | COMPLETE — PASSED | manifests and code reconciled; runtime override documented |
| Figure metadata | PARTIAL — DATA EXIST, ANALYSIS INCOMPLETE | full inventory; many captions not stand-alone |
| Numerical stability and failed runs | COMPLETE — FAILED | failures and finite-but-unstable runs exposed |
| MIDPOINT improves PBME | OPEN — DATA ABSENT | no compatible physical reference; prerequisites fail |

## 17. Thesis-ready conclusions

1. PBME common-support KDE versus projected-GP reconstruction passes the declared
   E1<=0.02 shape-agreement test for the two canonical seed-11 snapshots.
2. Manufactured Q relative-L2 results do not show reproducible refinement with
   support size; relative E1 and Einf were not saved and cannot be reconstructed.
3. SEO-image leakage is material and especially large at P0=100 across four seeds
   and three saved snapshots; it remains a diagnostic rather than an enforced
   projection or causal decomposition.
4. The MIDPOINT campaign completes many propagations but fails raw-conservation and
   cross-seed stability requirements. Unit normalized trace does not contradict
   this because it is self-normalized.
5. No systematic physical improvement over PBME is demonstrated.

## 18. Missing or unidentifiable evidence

- Six expected MIDPOINT support outputs.
- Manufactured reference/prediction arrays needed for relative E1 and Einf.
- Manufactured conditioning experiments and common query sets across N.
- A causal decomposition of SEO leakage into PBME propagation, correction-only,
  and enforced-projection effects; no enforced projection was run.
- A production-compatible TDSE/grid-QCLE reference trajectory and density arrays.
- Reference boundary-probability/absorber metrics and saved QCLE CFL values.
- Three or more usable refinement levels for TDSE and grid QCLE order estimation.
- Image/caption metadata sufficient to identify every threshold, density scale,
  sign convention, and reference resolution.
- A validated tolerance suite for raw conservation and population physicality.

## 19. Recommended thesis changes

- Replace claims of MIDPOINT improvement, conservation, convergence, or validation
  with the qualified findings above.
- Label all `lw_*` curves as self-normalized and report raw drift beside them.
- Correct campaign completeness from 18/18 support pairs to 12/18 paired complete.
- Describe two-level reference checks as one-step refinement.
- State that support clouds are independently sampled.
- Remove/qualify the claim that manufactured Q error increases monotonically with
  N; the all-seed mean is non-monotonic.
- Correct the P0=20 production-label conflict: `results/P0_20/run_manifest.json`
  records P0=40.
- Do not use the P0=100 grid-QCLE zero output as a reference.

## 20. Exact provenance for tables and conclusions

Every quantitative CSV includes exact source files. `metric_provenance.csv` maps
metric families to source arrays/JSON/code and extraction methods.
`checksums_sha256.csv` provides source and generated-file checksums. The analysis
is reproducible with `python reviewer_data_audit/scripts/audit_pipeline.py`.

## Scientific contradictions found

1. `results/P0_20` and production-contract key “20.0” point to a manifest whose
   actual P0 is 40.0.
2. Existing handoff prose states 18/18 support pairs completed; only 12/18 have
   both method arrays.
3. Existing handoff prose characterizes manufactured error as monotonically
   increasing using one seed; all completed seeds give a non-monotonic mean with
   substantial seed variation.
4. Existing handoff prose says P0=100 baseline E2/Einf are unavailable, but the
   canonical JSON stores both.
5. Normalized trace appears conserved while raw MIDPOINT norm/trace is unstable;
   these are different normalization conventions, not equivalent evidence.
6. P0=100 grid-QCLE endpoints are zero because the configured P domain excludes
   the initial momentum, not because the coarse and fine solvers agree.
"""
    return report


def write_examiner_response() -> None:
    text = """# Examiner Response Evidence

The forensic audit closes the bookkeeping and common-support reconstruction items,
but it does not support a claim that the MIDPOINT correction improves PBME.

| Concern | Status | Response supported by data |
|---|---|---|
| Manufactured excess operator | COMPLETE — FAILED | Stored E2 does not decrease reproducibly with N; E1/Einf are NOT COMPUTED. |
| SEO leakage | COMPLETE — FAILED | Four-seed results at t/tc=0,1,2 show material leakage, especially at P0=100; projection was not enforced. |
| Time-step refinement | COMPLETE — FAILED | Three levels exist; several order estimates are omitted due to seed variation and raw instability. |
| Support refinement | PARTIAL — DATA EXIST, ANALYSIS INCOMPLETE | 12/18 paired runs complete; supports are independently sampled. |
| Four-seed replication | COMPLETE — FAILED | For lw_P0, MIDPOINT/PBME pairwise seed-dispersion ratios are 66.5 (P0=20) and 133 (P0=100). |
| Raw conservation | COMPLETE — FAILED | Raw drift fails; displayed unit trace is self-normalized. |
| KDE/GP same support | COMPLETE — PASSED | All 24 PBME replication snapshots pass E1<=0.02. |
| TDSE/grid-QCLE controls | COMPLETE — FAILED | Two levels only; production-incompatible; P0=100 QCLE grid invalid. |
| Production configuration | COMPLETE — PASSED | Manifest/code policy reconstructed, including adaptive runtime override. |
| Improvement over PBME | OPEN — DATA ABSENT | No compatible reference errors; stability prerequisites fail. |

Exact values and paths are in `PIPELINE_DATA_AUDIT_AND_THESIS_EVIDENCE.md` and the
CSV tables.
"""
    (AUDIT/"EXAMINER_RESPONSE_EVIDENCE.md").write_text(text, encoding="utf-8")


def write_missing() -> None:
    text = """# Missing Data and Analyses

The following are not recoverable from the current artifacts and were not inferred:

- MIDPOINT arrays for six expected support configurations.
- Manufactured per-query truth/prediction arrays needed for relative E1 and Einf.
- A common manufactured query set across N and direct conditioning experiments.
- A causal SEO-leakage decomposition or an enforced-projection comparison; saved
  snapshots support four-seed method/time diagnostics only.
- Production-compatible TDSE/grid-QCLE trajectories and densities.
- Saved TDSE boundary probability/reflection metrics, absorber diagnostics, and
  grid-QCLE CFL values.
- Three-level TDSE/grid-QCLE refinement.
- Predeclared raw-conservation and population-physicality tolerances.
- Complete stand-alone caption metadata for every figure.
- Any physical-error calibration from GP posterior variance, LOO residuals,
  training residuals, or R2.

Rows use `DATA ABSENT`, `NOT COMPUTED`, `RUN INCOMPLETE`, or `NOT IDENTIFIABLE`
throughout the machine-readable tables.
"""
    (AUDIT/"MISSING_DATA_AND_ANALYSES.md").write_text(text, encoding="utf-8")


def write_figure_audit(code_rows: list[dict[str, Any]], figures: list[dict[str, Any]]) -> None:
    violations = [r for r in code_rows if r["visible_internal_title_violation"]]
    text = f"""# Figure and Caption Audit

- Figure files inventoried: {len(figures)}
- Plot-title code calls inspected: {len(code_rows)}
- Nonempty internal-title violations found in current code: {len(violations)}
- Stand-alone caption metadata failures: {sum(not r['caption_standalone'] for r in figures)}
- Correction-only figure files identified: {sum(r.get('correction_only_panel') is True for r in figures)}
- Yellow correction figures found by targeted PNG RGB scan: {sum(r.get('yellow_correction_curve') is True for r in figures)}

Current method colors are PBME blue `#0072B2` and MIDPOINT orange `#D55E00`.
Yellow `#F0E442` exists in a palette but is not assigned to these two methods.
The current correction functions use solid blue/magenta/green curves. Source
comments explicitly state that a prior yellow broken-line correction-only figure
was replaced. Targeted pixel scanning found no current correction PNG using that
yellow within RGB tolerance 8.

Correction-only files under `surrogate_health/qcle_correction/` and
`flow_correction/` are identified in `figure_metadata.csv`. The current correction
code uses solid lines. Outside those panels, line styles are not perfectly uniform:
the central map uses solid PBME/MIDPOINT lines while some panel-specific code may
apply scheme-specific styles.

Code title calls:

{md_table(code_rows, ['source_file','line','code','visible_internal_title_violation']) if code_rows else 'No title calls found.'}

See `figure_metadata.csv` for the complete per-file inventory.
"""
    (AUDIT/"figure_caption_audit.md").write_text(text, encoding="utf-8")


def write_latex(expected: list[dict[str, Any]], manufactured_summary: list[dict[str, Any]], projection_summary: list[dict[str, Any]], baseline: list[dict[str, Any]], conservation: list[dict[str, Any]]) -> None:
    mfg = [r for r in manufactured_summary if r.get("canonical_latest_campaign") and r["query_set"]=="on_support" and r["quantity"]=="operator_Q"]
    proj = [r for r in projection_summary if r.get("canonical")]
    base = [r for r in baseline if r.get("canonical")]
    repl = [r for r in conservation if r.get("run_directory","").startswith("reviewer_closure_20260726_174927/step9_repl_") and r["quantity"]=="normalization"]
    time_rows = [
        r for r in read_csv_table(TABLES/"time_step_refinement_metrics.csv")
        if r.get("observable") == "lw_P0" and r.get("dt") == "0.5"
    ]
    support_rows = [
        r for r in read_csv_table(TABLES/"support_size_summary.csv")
        if r.get("observable") == "lw_P0"
    ]
    dispersion_rows = [
        r for r in read_csv_table(TABLES/"seed_replication_method_dispersion_ratios.csv")
        if r.get("observable") == "lw_P0"
    ]
    projection_seed_rows = [
        r for r in read_csv_table(TABLES/"seo_projection_leakage_seed_summary.csv")
        if math.isclose(float(r.get("t_over_tc", "nan")), 2.0)
    ]
    baseline_snapshot_rows = read_csv_table(
        TABLES/"kde_gp_identical_support_all_snapshots.csv"
    )
    reference_rows = [
        r for r in read_csv_table(TABLES/"reference_refinement.csv")
        if r.get("observable") == "trace"
        and "reviewer_closure_20260726_174927" in r.get("source_file", "")
    ]
    comparison_rows = [
        r for r in read_csv_table(TABLES/"midpoint_vs_pbme_paired_differences.csv")
        if r.get("observable") == "lw_P0"
    ]

    def tex_num(value: Any) -> str:
        if value in ("", None, "NOT COMPUTED", "DATA ABSENT", "NOT IDENTIFIABLE"):
            return r"\NA"
        try:
            number = float(value)
            return fmt(number) if math.isfinite(number) else r"\NA"
        except Exception:
            return str(value).replace("_", r"\_")
    lines = [
        r"\providecommand{\NA}{\textsc{Not computed}}",
        r"\providecommand{\DA}{\textsc{Data absent}}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Expected campaign completion. A configuration is complete only when both PBME and MIDPOINT arrays reach the manifest final step and time with finite core arrays.}",
        r"\label{tab:campaign-completion}",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Campaign & Expected & Complete & Incomplete & Missing \\",
        r"\hline",
    ]
    for c in ("A_time_step","B_support","C_replication"):
        lines.append(f"{c.replace('_', r'\_')} & {sum(r['campaign']==c for r in expected)} & {sum(r['campaign']==c and r['status']=='COMPLETE' for r in expected)} & {sum(r['campaign']==c and r['status']=='INCOMPLETE' for r in expected)} & {sum(r['campaign']==c and r['status']=='MISSING' for r in expected)} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    lines += [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Manufactured-operator on-support relative $L^2$ error $E_2$. Uncertainty is the sample standard deviation across independent seeds; $E_1$ and relative $E_\infty$ are not computable from saved summaries. Manufactured-test regularization is $10^{-6}$, distinct from production $0.05$.}",
        r"\label{tab:manufactured-q}", r"\begin{tabular}{rrrrll}", r"\hline",
        r"$N$ & seeds & mean $E_2$ & SD & $E_1$ & $E_\infty$ \\", r"\hline",
    ]
    for r in mfg:
        lines.append(f"{r['N']} & {r['n_independent_seeds']} & {fmt(r['mean_E2'])} & {fmt(r['sample_sd_E2'])} & \\NA & \\NA \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    lines += [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{SEO projection leakage at the final PBME snapshot. SD is across 20 bath anchors, not independent propagation seeds.}",
        r"\label{tab:seo-leakage}", r"\begin{tabular}{rrrrrr}", r"\hline",
        r"$P_0$ & propagation seeds & anchors & mean & median & maximum \\", r"\hline",
    ]
    for r in proj:
        lines.append(f"{r['P0']} & 1 & {r['n_bath_anchors']} & {fmt(r['mean'])} & {fmt(r['median'])} & {fmt(r['maximum'])} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    lines += [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Identical-support PBME KDE versus projected-GP shape errors. The predeclared acceptance threshold is $E_1\le 0.02$.}",
        r"\label{tab:kde-gp}", r"\begin{tabular}{rrrrrl}", r"\hline",
        r"$P_0$ & seed & $E_1$ & $E_2$ & $E_\infty$ & result \\", r"\hline",
    ]
    for r in base:
        lines.append(f"{r['P0']} & {r['seed']} & {fmt(r['E1'])} & {fmt(r['E2'])} & {fmt(r['Einf'])} & {'PASS' if r['passed'] else 'FAIL'} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    lines += [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Four-seed campaign raw-normalization drift. Values are pre-self-normalization cloud estimators; $n=1$ per row and no per-row uncertainty is implied.}",
        r"\label{tab:raw-norm-drift}", r"\begin{tabular}{lrrrrr}", r"\hline",
        r"Method & $P_0$ & seed & endpoint drift & max.\ absolute drift & $N$ \\", r"\hline",
    ]
    for r in repl:
        lines.append(f"{r['method'].upper()} & {int(float(r['P0']))} & {int(float(r['seed']))} & {fmt(r['endpoint_drift'])} & {fmt(r['maximum_absolute_drift'])} & {int(float(r['N']))} \\\\")
    lines += [
        r"\hline", r"\end{tabular}",
        r"\begin{minipage}{0.96\linewidth}\footnotesize Notes: production reviewer campaigns use L2 regularization 0.05, adaptive refits, focused sampling, product surrogate, and a relative product-profile floor of $10^{-3}$. Manufactured tests use L2 regularization $10^{-6}$ and frozen hyperparameters. ``Not computed'' never denotes zero.", r"\end{minipage}",
        r"\end{table}", "",
    ]
    lines += [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Three-level timestep self-consistency for the self-normalized population $P_0$. Rows compare $\Delta t=0.5$ with the finest saved level $\Delta t=0.125$. Trajectories are aligned by linear interpolation without extrapolation. An order is shown only when the refinement signal passes the stored seed-variation and small-denominator gates.}",
        r"\label{tab:timestep-refinement}", r"\begin{tabular}{rrlrrrr}", r"\hline",
        r"$P_0$ & seed & method & endpoint diff. & max. diff. & time-$L^2$ & order $p$ \\", r"\hline",
    ]
    for r in time_rows:
        lines.append(
            f"{r['P0']} & {r['seed']} & {r['method'].upper()} & "
            f"{tex_num(r['endpoint_absolute_difference'])} & "
            f"{tex_num(r['maximum_in_time_absolute_difference'])} & "
            f"{tex_num(r['time_integrated_L2_difference'])} & "
            f"{tex_num(r['empirical_order_p'])} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    lines += [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Support-size study for endpoint self-normalized population $P_0$. Supports were independently sampled, so these are distributional summaries, not deterministic support convergence. Intervals are two-sided Student-$t$ 95\% intervals using independent seeds.}",
        r"\label{tab:support-size}", r"\begin{tabular}{rlrrrrr}", r"\hline",
        r"$P_0$ & method & $N$ & seeds & mean & SD & 95\% CI \\", r"\hline",
    ]
    for r in support_rows:
        ci = (
            f"[{tex_num(r.get('ci95_lower'))}, {tex_num(r.get('ci95_upper'))}]"
            if int(float(r["n_independent_seeds"])) > 1 else r"\NA"
        )
        lines.append(
            f"{r['P0']} & {r['method'].upper()} & {r['N']} & "
            f"{r['n_independent_seeds']} & {tex_num(r['mean'])} & "
            f"{tex_num(r['sample_sd'])} & {ci} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    lines += [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Independent-seed reproducibility for the self-normalized population $P_0$. The trajectory metric is the mean over all six seed-pair time-normalized $L^2$ distances on 41 common $t/t_c$ values. Ratios greater than one denote greater MIDPOINT seed sensitivity than PBME; no pass threshold is imposed.}",
        r"\label{tab:seed-reliability}", r"\begin{tabular}{rrrrrr}", r"\hline",
        r"$P_0$ & PBME spread & MIDPOINT spread & spread ratio & PBME pair $L^2$ & MIDPOINT/PBME $L^2$ ratio \\", r"\hline",
    ]
    for r in dispersion_rows:
        lines.append(
            f"{r['P0']} & {tex_num(r['PBME_endpoint_spread'])} & "
            f"{tex_num(r['MIDPOINT_endpoint_spread'])} & "
            f"{tex_num(r['MIDPOINT_to_PBME_endpoint_spread_ratio'])} & "
            f"{tex_num(r['PBME_mean_pairwise_time_L2'])} & "
            f"{tex_num(r['MIDPOINT_to_PBME_pairwise_L2_ratio'])} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    lines += [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{SEO-image projection leakage at the final saved snapshot $t/t_c=2$. Means and uncertainty summarize four independent propagation seeds. The per-run diagnostic uses 20 bath anchors and 400 mapping probes; projection is diagnostic and was not enforced.}",
        r"\label{tab:seo-leakage-replication}", r"\begin{tabular}{rlrrrr}", r"\hline",
        r"$P_0$ & method & seeds & mean & seed SD & 95\% CI \\", r"\hline",
    ]
    for r in projection_seed_rows:
        lines.append(
            f"{r['P0']} & {r['method'].upper()} & "
            f"{r['n_independent_propagation_seeds']} & "
            f"{tex_num(r['mean_of_seed_mean_leakage'])} & "
            f"{tex_num(r['sample_sd_across_seed_means'])} & "
            f"[{tex_num(r['ci95_lower'])}, {tex_num(r['ci95_upper'])}] \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    lines += [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Expanded identical-support PBME KDE versus projected-GP validation across four seeds and three snapshots. The table reports the maximum error over 12 cases per momentum. The declared criterion is $E_1\leq0.02$.}",
        r"\label{tab:kde-gp-expanded}", r"\begin{tabular}{rrrrrl}", r"\hline",
        r"$P_0$ & cases & max $E_1$ & max $E_2$ & max $E_\infty$ & result \\", r"\hline",
    ]
    for p0 in ("20", "100"):
        rows = [
            r for r in baseline_snapshot_rows
            if r.get("P0") == p0 and r.get("method") == "pbme"
        ]
        passed = sum(
            r.get("passed_declared_threshold", "").lower() == "true" for r in rows
        )
        lines.append(
            f"{p0} & {len(rows)} & "
            f"{tex_num(max(float(r['E1']) for r in rows))} & "
            f"{tex_num(max(float(r['E2']) for r in rows))} & "
            f"{tex_num(max(float(r['Einf']) for r in rows))} & "
            f"{'PASS' if rows and passed == len(rows) else 'FAIL'} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    lines += [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Stored two-level reference trace endpoints. These are one-step refinement differences, not convergence or observed-order estimates. Validity also requires production-compatible initial conditions and domains.}",
        r"\label{tab:reference-controls}", r"\begin{tabular}{rlrrrl}", r"\hline",
        r"$P_0$ & solver & coarse & fine & abs. diff. & configuration \\", r"\hline",
    ]
    for r in reference_rows:
        valid = "INVALID" if r.get("invalid_configuration", "").lower() == "true" else "INCOMPATIBLE"
        lines.append(
            f"{r['P0']} & {r['method'].replace('_', r'\_')} & "
            f"{tex_num(r['coarse'])} & {tex_num(r['fine'])} & "
            f"{tex_num(r['absolute_difference'])} & {valid} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    lines += [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Paired PBME--MIDPOINT endpoint differences for self-normalized population $P_0$. Each row uses the same initial cloud and time grid. The physical-reference error difference is not computed because no compatible reference trajectory exists.}",
        r"\label{tab:midpoint-pbme}", r"\begin{tabular}{rrrrrl}", r"\hline",
        r"$P_0$ & seed & PBME & MIDPOINT & MIDPOINT$-$PBME & reference-error difference \\", r"\hline",
    ]
    for r in comparison_rows:
        lines.append(
            f"{r['P0']} & {r['seed']} & {tex_num(r['PBME_endpoint'])} & "
            f"{tex_num(r['MIDPOINT_endpoint'])} & "
            f"{tex_num(r['MIDPOINT_minus_PBME_endpoint_signed_difference'])} & "
            r"\NA \\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (AUDIT/"THESIS_EVIDENCE_TABLES.tex").write_text("\n".join(lines), encoding="utf-8")


def provenance_rows() -> list[dict[str, Any]]:
    items = [
        ("file inventory", "file_inventory.csv", "filesystem traversal and ZIP central directories", "os.stat; SHA-256; ZIP CRC"),
        ("run inventory", "run_inventory.csv", "all run_manifest.json plus pbme.npz/midpoint.npz", "manifest extraction; final step/time/core finite checks"),
        ("campaign status", "run_status_matrix.csv", "canonical campaign roots", "expected Cartesian products verified against manifests and arrays"),
        ("stability", "numerical_stability_audit.csv", "all logs and NPZ time histories", "regex fatal scan; finite/variance/bound/ESS/identity checks"),
        ("manufactured operator", "manufactured_operator_metrics.csv", "**/manufactured_operator_metrics.json", "stored RMSE, absolute Linf, relative L2; no inferred E1/Einf"),
        ("SEO leakage", "seo_projection_leakage_summary.csv", "**/projection_leakage.json", "mean/median/sample SD/max over saved anchor residuals"),
        ("KDE/GP baseline", "kde_gp_identical_support.csv", "**/kde_gp_identical_support.json", "stored common-support E1/E2/Einf and declared threshold"),
        ("time step", "time_step_refinement_metrics.csv", "step7_dt_* method NPZ", "linear no-extrapolation alignment; endpoint/max/integrated norms; guarded p"),
        ("support size", "support_size_summary.csv", "step8_support_* method NPZ", "endpoint seed mean/SD/SE/Student-t CI; independent clouds"),
        ("replication", "seed_replication_trajectories.csv", "step9_repl_* method NPZ", "41 common t/tc points; n=4 seed statistics"),
        ("replication pairwise reliability", "seed_replication_pairwise_distances.csv; seed_replication_reliability_summary.csv; seed_replication_method_dispersion_ratios.csv", "all four step9 replication seeds for PBME and MIDPOINT", "all six seed pairs on 41 common t/tc points; time-averaged L1/L2, maximum differences, and threshold-free method ratios"),
        ("population physicality", "population_physicality_audit.csv", "step9 lw_P0 and lw_P1 arrays", "per-run min/max and first value outside [0,1] by more than 1e-6"),
        ("replication GP health", "replication_gp_health.csv", "step9 GP, ESS, Q, optimizer, denominator, and raw-drift arrays", "per-run extrema, onset times, bound hits, and configured/audit visibility gates"),
        ("SEO leakage replication snapshots", "seo_projection_leakage_all_snapshots.csv; seo_projection_leakage_seed_summary.csv", "step9 snapshots at t/tc=0,1,2 for four seeds and both methods", "rank-4 diagnostic projection; 20 anchors x 400 common-seed probes; Student-t summary across propagation seeds"),
        ("KDE/GP replication snapshots", "kde_gp_identical_support_all_snapshots.csv", "step9 snapshots at t/tc=0,1,2 for four seeds and both methods", "same support/measure/bandwidth/120x120 grid; PBME-only declared E1 threshold"),
        ("raw conservation", "raw_conservation.csv", "raw_*_drift arrays in every readable method NPZ", "endpoint/max/RMS/relative/behavior"),
        ("all-run endpoints", "all_run_observable_endpoints.csv", "every discovered method NPZ", "saved final values for common observables; missing values explicit"),
        ("method comparison", "midpoint_vs_pbme_paired_differences.csv", "paired step9 method NPZ", "same-time paired difference; no physical reference error inferred"),
        ("references", "reference_refinement.csv", "reference_convergence.json and solver code", "stored two-level endpoint differences; configuration validity audit"),
        ("figures", "figure_metadata.csv", "all figure files, sidecars, nearest manifest, plotting code", "metadata traceability; no pixel inference"),
        ("audit chart map", "chart_map.csv", "generated audit plots and their quantitative CSVs", "visual question, chart family, supported claim, palette, and source table"),
        ("GP policy", "gp_policy.csv", "GP_Density.py, run.py, manifests, NPZ", "code defaults versus authoritative per-run overrides"),
        ("contradictions", "scientific_contradictions.csv", "code, manifests, logs, arrays, existing handoff documents", "direct source-to-source reconciliation"),
    ]
    return [{"metric_family": a, "output_table": b, "source": c, "method": d, "normalization_or_caveat": "see table and main report"} for a,b,c,d in items]


def write_readme() -> None:
    text = f"""# Reviewer Data Audit

This directory is generated read-only from:

`{ROOT}`

## Reproduce

From the pipeline root, in a clean Python process:

```powershell
python reviewer_data_audit/scripts/complete_remaining_analysis.py
python reviewer_data_audit/scripts/audit_pipeline.py
python reviewer_data_audit/scripts/verify_audit.py
python reviewer_data_audit/scripts/package_audit.py
```

The first command derives four-seed and saved-snapshot diagnostics under
`reviewer_data_audit/derived_validations`. The audit generator deletes and
recreates only `reviewer_data_audit/tables` and `reviewer_data_audit/plots`; it
never changes raw simulation artifacts. The ZIP is
written to `{GENERATED_ZIP}` because an archive cannot contain itself.

Python requirements are recorded in `requirements_audit.txt`. Full-precision
numbers are preserved in CSV. Reader-facing Markdown/LaTeX uses scientific
rounding. Every table contains source paths or is mapped through
`metric_provenance.csv`.

## Primary artifacts

- `PIPELINE_DATA_AUDIT_AND_THESIS_EVIDENCE.md`
- `THESIS_EVIDENCE_TABLES.tex`
- `EXAMINER_RESPONSE_EVIDENCE.md`
- `MISSING_DATA_AND_ANALYSES.md`
- `run_inventory.csv`
- `numerical_stability_audit.csv`
- `figure_metadata.csv`
- `metric_provenance.csv`
- `checksums_sha256.csv`
- `tables/all_run_observable_endpoints.csv`
- `tables/seed_replication_reliability_summary.csv`
- `tables/seed_replication_method_dispersion_ratios.csv`
- `tables/seo_projection_leakage_seed_summary.csv`
- `tables/kde_gp_identical_support_all_snapshots.csv`
- `tables/scientific_contradictions.csv`
"""
    (AUDIT/"README.md").write_text(text, encoding="utf-8")


def clean_generated() -> None:
    for d in (TABLES, PLOTS):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    for name in (
        "file_inventory.csv", "run_inventory.csv", "run_inventory.md",
        "numerical_stability_audit.csv", "figure_metadata.csv",
        "metric_provenance.csv", "checksums_sha256.csv",
        "PIPELINE_DATA_AUDIT_AND_THESIS_EVIDENCE.md",
        "THESIS_EVIDENCE_TABLES.tex", "EXAMINER_RESPONSE_EVIDENCE.md",
        "MISSING_DATA_AND_ANALYSES.md", "figure_caption_audit.md",
        "README.md", "requirements_audit.txt", "audit_summary.json",
        "verification_result.json",
    ):
        p = AUDIT/name
        if p.exists():
            p.unlink()


def main() -> None:
    started = time.time()
    AUDIT.mkdir(exist_ok=True)
    SCRIPTS.mkdir(exist_ok=True)
    clean_generated()
    # Recreate additional four-seed and saved-snapshot tables.  Heavy snapshot
    # diagnostics are cached only below reviewer_data_audit; collect-only mode
    # reads those derived JSONs and never modifies simulation artifacts.
    from complete_remaining_analysis import seed_reliability, snapshot_validations
    seed_reliability()
    snapshot_validations(collect_only=True)
    file_rows, source_checks = build_file_inventory()
    run_rows = run_inventory()
    expected = expected_runs()
    log_issues = scan_logs()
    npz_issues, npz_summary = npz_stability()
    stability = log_issues + npz_issues
    manufactured, manufactured_summary = manufactured_tables()
    projection, projection_summary = projection_tables()
    baseline = baseline_table()
    time_ref = time_refinement_table()
    support_endpoints, support_summary, nested = support_tables()
    replication, replication_endpoints = replication_tables()
    conservation = conservation_table(run_rows)
    all_endpoints = all_run_endpoint_table(run_rows)
    comparison = midpoint_pbme_table()
    references = reference_table()
    figures, title_code = figure_metadata(run_rows)
    gp_policy = gp_policy_table(run_rows)
    contradictions = contradiction_table()
    provenance = provenance_rows()

    # Primary and major quantitative tables.
    write_csv(AUDIT/"file_inventory.csv", file_rows)
    write_csv(AUDIT/"run_inventory.csv", run_rows)
    write_csv(TABLES/"run_status_matrix.csv", expected)
    write_csv(AUDIT/"numerical_stability_audit.csv", stability)
    write_csv(TABLES/"npz_stability_summary.csv", npz_summary)
    write_csv(TABLES/"manufactured_operator_metrics.csv", manufactured)
    write_csv(TABLES/"manufactured_operator_summary.csv", manufactured_summary)
    write_csv(TABLES/"seo_projection_leakage_per_anchor.csv", projection)
    write_csv(TABLES/"seo_projection_leakage_summary.csv", projection_summary)
    write_csv(TABLES/"kde_gp_identical_support.csv", baseline)
    write_csv(TABLES/"time_step_refinement_metrics.csv", time_ref)
    write_csv(TABLES/"support_size_endpoints.csv", support_endpoints)
    write_csv(TABLES/"support_size_summary.csv", support_summary)
    write_csv(TABLES/"support_cloud_nesting_audit.csv", nested)
    write_csv(TABLES/"seed_replication_trajectories.csv", replication)
    write_csv(TABLES/"seed_replication_endpoints.csv", replication_endpoints)
    write_csv(TABLES/"raw_conservation.csv", conservation)
    write_csv(TABLES/"all_run_observable_endpoints.csv", all_endpoints)
    write_csv(TABLES/"midpoint_vs_pbme_paired_differences.csv", comparison)
    write_csv(TABLES/"reference_refinement.csv", references)
    write_csv(AUDIT/"figure_metadata.csv", figures)
    write_csv(TABLES/"figure_title_code_audit.csv", title_code)
    write_csv(TABLES/"gp_policy.csv", gp_policy)
    write_csv(TABLES/"scientific_contradictions.csv", contradictions)
    write_csv(AUDIT/"metric_provenance.csv", provenance)
    write_run_inventory_md(expected, run_rows)
    write_figure_audit(title_code, figures)
    create_plots(manufactured_summary, projection_summary, conservation)
    write_csv(TABLES/"chart_map.csv", [
        {
            "plot": "plots/manufactured_operator_E2.png",
            "analytical_question": "Does manufactured-Q relative L2 error decrease with support size?",
            "chart_family": "uncertainty/comparison",
            "chart_type": "point and sample-SD interval",
            "supported_claim": "seed-aggregated E2 is non-monotonic",
            "source_table": "tables/manufactured_operator_summary.csv",
            "palette_policy": "single-root blue",
        },
        {
            "plot": "plots/seo_projection_leakage_replication_snapshots.png",
            "analytical_question": "How does diagnostic SEO leakage differ by method, momentum, and saved snapshot?",
            "chart_family": "comparison/uncertainty",
            "chart_type": "grouped bars with Student-t intervals across four seeds",
            "supported_claim": "leakage is large at P0=100 for both methods and seed-variable at P0=20",
            "source_table": "tables/seo_projection_leakage_seed_summary.csv",
            "palette_policy": "hard two-root cap; blue PBME, orange hatched MIDPOINT",
        },
        {
            "plot": "plots/seed_replication_lw_P0.png",
            "analytical_question": "Are population trajectories reproducible across four independent seeds?",
            "chart_family": "trend/uncertainty",
            "chart_type": "mean line with descriptive Student-t interval",
            "supported_claim": "MIDPOINT cross-seed dispersion is much larger than PBME",
            "source_table": "tables/seed_replication_trajectories.csv",
            "palette_policy": "hard two-root cap; line style also distinguishes methods",
        },
        {
            "plot": "plots/raw_normalization_drift_replication.png",
            "analytical_question": "How large is maximum raw normalization drift by run?",
            "chart_family": "comparison",
            "chart_type": "grouped log-scale bars",
            "supported_claim": "MIDPOINT raw drift greatly exceeds PBME in the four-seed campaign",
            "source_table": "tables/raw_conservation.csv",
            "palette_policy": "hard two-root cap; blue PBME, orange MIDPOINT",
        },
    ])
    report = build_report(
        file_rows, run_rows, expected, stability, manufactured, manufactured_summary,
        projection_summary, baseline, time_ref, support_summary, replication,
        conservation, comparison, references, figures, gp_policy,
    )
    (AUDIT/"PIPELINE_DATA_AUDIT_AND_THESIS_EVIDENCE.md").write_text(report, encoding="utf-8")
    write_examiner_response()
    write_missing()
    write_latex(expected, manufactured_summary, projection_summary, baseline, conservation)
    write_readme()
    req = [
        f"python=={platform.python_version()}",
        f"numpy=={np.__version__}",
        f"matplotlib=={matplotlib.__version__}" if plt is not None else "matplotlib==DATA ABSENT",
        f"Pillow=={PIL.__version__}" if PIL is not None else "Pillow==DATA ABSENT (yellow-curve scan skipped)",
        *[
            f"{package}=={importlib.metadata.version(package)}"
            for package in ("scipy", "torch", "jax")
        ],
        "Standard library: csv, hashlib, json, pathlib, statistics, zipfile",
        f"platform={platform.platform()}",
    ]
    (AUDIT/"requirements_audit.txt").write_text("\n".join(req)+"\n", encoding="utf-8")
    counts = Counter(r["status"] for r in expected)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_physical_files": sum(r["storage"]=="filesystem" for r in file_rows),
        "source_zip_members": sum(r["storage"]=="zip_member" for r in file_rows),
        "expected_configuration_groups": len(expected),
        "complete": counts["COMPLETE"], "incomplete": counts["INCOMPLETE"],
        "failed": counts["FAILED"], "missing": counts["MISSING"],
        "configuration_conflicts": counts["CONFIGURATION CONFLICT"],
        "failed_attempt_logs": sum(r.get("issue")=="failed attempt log" for r in stability),
        "elapsed_seconds_before_checksums": time.time()-started,
    }
    (AUDIT/"audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    generated_checks = checksums_for_outputs()
    write_csv(AUDIT/"checksums_sha256.csv", source_checks + generated_checks)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
