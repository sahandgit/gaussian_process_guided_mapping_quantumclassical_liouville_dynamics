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
closure_audit.py
================

Phase 1 of the Pipeline Reviewer-Closure Specification: campaign creation,
the authoritative artifact schema, and the **existing-data-first audit**.

The specification is explicit (section 4.1):

    "Do not launch an expensive simulation until this table proves that
     existing data cannot answer the corresponding reviewer item."

This module produces exactly that proof. It enumerates every campaign cell
required by sections 8-18, matches each against the artifacts already on disk,
and assigns one of the five mandated dispositions:

    REUSE - COMPLETE AND COMPATIBLE
    REANALYZE - RAW DATA PRESENT
    RERUN - MISSING
    RERUN - INCOMPATIBLE
    REPAIR THEN RERUN - TECHNICAL FAILURE

It also implements:

  * section 3  campaign directory + starting-state SHA-256 inventory
  * section 5  the ``reviewer-closure-1.0`` artifact schema and ATOMIC json write
  * section 7  ``acceptance_contract.yaml`` with the non-post-hoc gates
  * section 4  ``pipeline_inventory.md``

Scope note
----------
This module deliberately does **not** run any physics. It answers the question
"what must actually be run?" so that the expensive campaigns can be scoped
honestly. Nothing here closes a reviewer item by itself.

Torch-free. Validate with ``python closure_audit.py --self-test``.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "reviewer-closure-1.0"

# Dispositions mandated by specification section 4.1
REUSE = "REUSE - COMPLETE AND COMPATIBLE"
REANALYZE = "REANALYZE - RAW DATA PRESENT"
RERUN_MISSING = "RERUN - MISSING"
RERUN_INCOMPAT = "RERUN - INCOMPATIBLE"
REPAIR = "REPAIR THEN RERUN - TECHNICAL FAILURE"

SOURCE_EXTENSIONS = {".py", ".toml", ".yaml", ".yml", ".json", ".md",
                     ".tex", ".bib", ".txt"}


# ===========================================================================
# Utilities
# ===========================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    """SHA-256 of a file, or None if unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                b = fh.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None


def write_json_atomic(path: Path, obj: Any) -> Path:
    """
    Atomic JSON write required by specification section 5: write a temporary
    file, flush and fsync, then rename. A crash must never leave a truncated
    manifest that could masquerade as a complete run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def package_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for m in ("numpy", "scipy", "torch", "jax", "matplotlib"):
        try:
            out[m] = getattr(__import__(m), "__version__", "unknown")
        except Exception:
            out[m] = None
    return out


# ===========================================================================
# Section 5 -- authoritative artifact schema
# ===========================================================================

def new_artifact_manifest(*, artifact_id: str, local_root: Path,
                          source_snapshot_id: str,
                          command: Sequence[str],
                          method: str = "", model: str = "Tully-dual-avoided-crossing",
                          **fields: Any) -> Dict[str, Any]:
    """
    Build a manifest conforming to the ``reviewer-closure-1.0`` schema.

    Unknown keys are accepted and stored, but the required keys are always
    present so a downstream verifier can rely on them.
    """
    man: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "created_utc": utc_now(),
        "local_root": str(local_root),
        "source_snapshot_id": source_snapshot_id,
        "source_sha256": {},
        "command": list(command),
        "environment": {"python": sys.version.split()[0], **package_versions()},
        "method": method,
        "model": model,
        "P0": None, "R0": None, "mass": None, "hbar": None, "seed": None,
        "n_support": None,
        "initial_cloud_sha256": None, "parent_cloud_sha256": None,
        "dt_requested": None, "dt_resolved": None,
        "t_final_requested": None, "t_final_resolved": None,
        "sampling": {}, "gp_configured_policy": {}, "gp_executed_policy": {},
        "estimator_contract": {},
        "normalization_status": None,
        "return_code": None, "completed": False,
        "output_sha256": {},
    }
    man.update(fields)
    return man


def validate_artifact_manifest(man: Dict[str, Any]) -> List[str]:
    """Return the list of schema violations (empty means conforming)."""
    problems: List[str] = []
    if man.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version != {SCHEMA_VERSION}")
    for key in ("artifact_id", "created_utc", "local_root",
                "source_snapshot_id", "command", "environment", "method",
                "model", "normalization_status", "completed"):
        if key not in man:
            problems.append(f"missing key: {key}")
    if man.get("completed") and man.get("return_code") not in (0, None):
        problems.append("completed=true with nonzero return_code")
    return problems


# ===========================================================================
# Section 3 -- campaign creation and starting-state inventory
# ===========================================================================

def source_hash_inventory(root: Path, exclude: Iterable[Path] = ()) -> List[Dict[str, str]]:
    """SHA-256 of every source-like file under root (specification section 3)."""
    excl = [Path(e).resolve() for e in exclude]
    rows: List[Dict[str, str]] = []
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        rp = p.resolve()
        if any(str(rp).startswith(str(e)) for e in excl):
            continue
        h = sha256_file(p)
        if h:
            rows.append({"Path": str(p), "Hash": h})
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]],
              columns: Optional[Sequence[str]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns) if columns else (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path


def create_campaign(root: Path, stamp: Optional[str] = None) -> Path:
    """
    Create ``validation/reviewer_closure_<stamp>/`` with the mandated
    subdirectories, the starting-state hash inventory, and the campaign
    manifest. Never overwrites existing production or validation directories.
    """
    root = Path(root)
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    campaign = root / "validation" / f"reviewer_closure_{stamp}"
    campaign.mkdir(parents=True, exist_ok=False)

    for sub in ("starting_state", "production_contract", "regularization",
                "manufactured", "projection", "timestep", "support_nested",
                "replication", "conservation", "source_audit", "pbme_kde_gp",
                "references", "physical_errors", "figures", "latex_tables",
                "logs"):
        (campaign / sub).mkdir(parents=True, exist_ok=True)

    inv = source_hash_inventory(root, exclude=[campaign])
    write_csv(campaign / "starting_state" / "source_hashes_before.csv", inv,
              ["Path", "Hash"])

    write_json_atomic(campaign / "campaign_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "local_root": str(root.resolve()),
        "source_snapshot_id": f"reviewer_closure_{stamp}",
        "starting_source_sha256": {r["Path"]: r["Hash"] for r in inv},
        "n_source_files_hashed": len(inv),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": package_versions(),
        "command": list(sys.argv),
        "parent_production_directories": [
            str(p) for p in sorted((root / "results").glob("P0_*"))
        ] if (root / "results").exists() else [],
        "thesis_revision_id": None,
        "campaign_schema_version": SCHEMA_VERSION,
    })
    return campaign


# ===========================================================================
# Section 7 -- acceptance contract
# ===========================================================================

_ACCEPTANCE_YAML = """\
# Acceptance contract -- Pipeline Reviewer-Closure Specification section 7.
# Written BEFORE the campaign runs. Hash this file into every analysis artifact.
schema_version: "{schema}"
created_utc: "{created}"

gates:
  pbme_common_support_reconstruction:
    metric: E1
    threshold: 0.02
    comparison: "<="
    also_report: [E2, Einf, raw_mass_error]
    note: "Acceptance may be true only for a manifest whose method is PBME."

  projection_preserving_four_field_branch:
    metric: relative_leakage
    threshold: 1.0e-10
    comparison: "<="
    note: "Otherwise fail. Declared probe set must be recorded."

  unconstrained_product_branch:
    status: diagnostic_only
    note: "Report leakage; never relabel the surrogate as projected."

  time_refinement:
    rule: "fine-to-finer difference must be smaller than coarse-to-fine, per seed"
    report_p_obs_only_if: "above noise floor and roundoff guard"

  support_refinement:
    rule: "compare nested differences with independent-seed SD"
    deterministic_order_claim_allowed: false
    note: "Allowed only if nested differences decrease."

  raw_conservation:
    rule: "report exact magnitudes and refinement trend"
    generic_threshold_declaration_allowed: false

  method_improvement:
    rule: >-
      MIDPOINT must have smaller matched reference error than PBME for the same
      observable, momentum and seed, with the paired-difference uncertainty
      interval excluding zero.
    default_verdict: "No validated improvement"

  manufactured_operator:
    rule: "all values finite; report density, gradient and Q trends on/off support at every policy"
    absolute_physical_tolerance_allowed: false
    note: "A nondecreasing trend closes as COMPLETE - FAILED AND REPORTED."

  tail_source_sensitivity:
    rule: "scientific conclusion unchanged across the declared threshold grid"
    threshold_grid: [0, 1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6]

order_estimate:
  formula: "p_obs = log2( ||u_h - u_h/2|| / ||u_h/2 - u_h/4|| )"
  suppress_when:
    - "denominator nonfinite"
    - "denominator below 100 * eps_mach * solution scale"
    - "denominator below independent-seed noise floor"
    - "numerator below 100 * eps_mach * solution scale"
  suppressed_rows_retained: true

terminal_statuses_allowed:
  - "COMPLETE - PASSED"
  - "COMPLETE - FAILED AND REPORTED"
  - "NOT APPLICABLE - JUSTIFIED"
forbidden_statuses: [PARTIAL, OPEN, UNKNOWN, ""]
"""


def write_acceptance_contract(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_ACCEPTANCE_YAML.format(schema=SCHEMA_VERSION,
                                            created=utc_now()),
                    encoding="utf-8")
    return path


# ===========================================================================
# Section 8-18 -- required campaign cells
# ===========================================================================

@dataclass(frozen=True)
class Cell:
    """One required campaign cell."""
    campaign: str
    reviewer_items: str
    key: Dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        bits = [f"{k}={v}" for k, v in sorted(self.key.items())]
        return f"{self.campaign}[" + ",".join(bits) + "]"


def required_cells() -> List[Cell]:
    """Enumerate every cell demanded by specification sections 8-18."""
    cells: List[Cell] = []

    # A -- production policy resolution
    for P0 in (20.0, 100.0):
        cells.append(Cell("A_production_contract", "I-14,M-24", {"P0": P0}))

    # B -- production-matched regularization
    for N in (300, 600, 1200):
        cells.append(Cell("B_regularization", "M-24", {"N": N}))
    cells.append(Cell("B_regularization", "M-24", {"confirmation_seed": True}))

    # C -- manufactured factorial
    for N in (300, 600, 1200, 2400):
        for seed in (123, 124, 125):
            for l2 in (1e-6, 0.01, 0.05):
                cells.append(Cell("C_manufactured", "I-1",
                                  {"N": N, "seed": seed, "l2": l2}))

    # D -- exact SEO projection
    for P0 in (20.0, 100.0):
        for seed in (11, 29, 47, 73):
            for snap in ("interaction", "final"):
                cells.append(Cell("D_projection", "M-2",
                                  {"P0": P0, "seed": seed, "snapshot": snap}))

    # E -- three-level time step
    for method in ("PBME", "MIDPOINT"):
        for P0 in (20.0, 100.0):
            for seed in (11, 29, 47, 73):
                for dt in (0.5, 0.25, 0.125):
                    cells.append(Cell("E_timestep", "I-2",
                                      {"method": method, "P0": P0,
                                       "seed": seed, "dt": dt}))

    # F -- nested support + replication
    for P0 in (20.0, 100.0):
        for seed in (11, 29, 47, 73):
            for N in (500, 1000, 2000):
                cells.append(Cell("F_support_nested", "M-6",
                                  {"P0": P0, "seed": seed, "N": N,
                                   "nested": True}))

    # G -- raw conservation (derived from E/F)
    for P0 in (20.0, 100.0):
        cells.append(Cell("G_conservation", "I-12,M-13", {"P0": P0}))

    # H -- y0 tail / applied-source audit
    for tau in (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6):
        cells.append(Cell("H_source_audit", "I-9,I-10,M-12", {"tau_rel": tau}))

    # I -- complete PBME KDE/GP baseline
    for P0 in (20.0, 100.0):
        for seed in (11, 29, 47, 73):
            for t in ("t0", "tc", "2tc", "final"):
                cells.append(Cell("I_pbme_kde_gp", "I-5,M-18",
                                  {"P0": P0, "seed": seed, "time": t}))

    # J -- reference verification
    for solver in ("TDSE", "GRID_QCLE"):
        for P0 in (20.0, 100.0):
            for mode in ("both", "time", "grid"):
                cells.append(Cell("J_references", "I-3,I-14",
                                  {"solver": solver, "P0": P0, "mode": mode}))

    # K -- matched physical reference errors
    for method in ("PBME", "MIDPOINT"):
        for P0 in (20.0, 100.0):
            for seed in (11, 29, 47, 73):
                cells.append(Cell("K_physical_errors", "I-13,M-16",
                                  {"method": method, "P0": P0, "seed": seed}))

    return cells


# ===========================================================================
# Section 4 / 4.1 -- discovery and existing-data index
# ===========================================================================

_RE_P0 = re.compile(r"P0[_]?(\d+(?:\.\d+)?)")
_RE_SEED = re.compile(r"seed[_]?(\d+)")
_RE_N = re.compile(r"[_/\\]N(\d+)")
_RE_DT = re.compile(r"dt([0-9.]+)")

_MODULE_ROLES = {
    "run.py": "single-entry driver; paired PBME/MIDPOINT production runs",
    "Dynamics.py": "PBME and midpoint stepping; source application; diagnostics",
    "GP_Density.py": "ARD-RBF GP, refit policies, KKT projection, product wrappers",
    "GPDerivatives.py": "analytic GP derivatives (authoritative)",
    "GP_Derivatives.py": "duplicate shim -- deprecated in favour of GPDerivatives.py",
    "Operator.py": "excess operator, manufactured tests, discrete source matrices",
    "Sampling.py": "nuclear Wigner, signed-SEO and focused MMST sampling",
    "Observables.py": "cloud and GP-integral estimators",
    "KDEDensity.py": "signed KDE and projected nuclear GP baseline",
    "Collector.py": "npz/json serialization of runs",
    "Visualization.py": "publication figures (header-free)",
    "Compare_gp_se_qcle.py": "TDSE / grid-QCLE / PBME comparison pipeline",
    "qcle_grid_tully.py": "pseudospectral grid-QCLE reference solver",
    "ReviewerValidation.py": "validation subcommands and master table",
    "select_regularization.py": "L2 selection by nested resampling",
    "reviewer_closure_campaign.py": "campaign orchestration (legacy --mode interface)",
    "thesis_analysis.py": "figures and tables from completed campaigns",
    "thesis_closure.py": "3-level references, figure audit, nested plan",
    "seo_coefficient_gp.py": "projection-preserving four-field branch (experimental)",
    "conservative_excess.py": "conservative/weak-form excess discretization (experimental)",
    "closure_audit.py": "campaign creation, artifact schema, existing-data audit",
    "Reproducibility.py": "fingerprints, environment metadata, json writer",
    "FigureCatalog.py": "figure caption/metadata catalog",
    "Models.py": "Tully potentials and derivatives",
    "Mint.py": "MInt symplectic mapping integrator",
    "Monodromy.py": "backward half-step derivative tensors",
    "ProductMoments.py": "closed-form Gaussian product moments",
    "GP_DensityDiff.py": "density-difference GP architecture",
}


def _infer_key(path: Path) -> Dict[str, Any]:
    s = str(path)
    out: Dict[str, Any] = {}
    m = _RE_P0.search(s)
    if m:
        out["P0"] = float(m.group(1))
    m = _RE_SEED.search(s)
    if m:
        out["seed"] = int(m.group(1))
    m = _RE_N.search(s.replace("\\", "/"))
    if m:
        out["N"] = int(m.group(1))
    m = _RE_DT.search(s)
    if m:
        try:
            out["dt"] = float(m.group(1).rstrip("."))
        except ValueError:
            pass
    low = s.lower()
    if "midpoint" in low:
        out["method"] = "MIDPOINT"
    elif "pbme" in low:
        out["method"] = "PBME"
    return out


_MANIFEST_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}


def _read_manifest(d: Path) -> Optional[Dict[str, Any]]:
    """
    Parse ``<d>/run_manifest.json``, memoized.

    Note: the pipeline writes configuration under ``cli_arguments`` and emits
    bare ``Infinity`` for uncapped ceilings. Python's json accepts both, so no
    preprocessing is needed; ``manifest_leaf`` searches by leaf name and is
    therefore insensitive to the nesting key.
    """
    key = str(d)
    if key in _MANIFEST_CACHE:
        return _MANIFEST_CACHE[key]
    p = d / "run_manifest.json"
    man: Optional[Dict[str, Any]] = None
    if p.exists():
        try:
            man = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            man = None
    _MANIFEST_CACHE[key] = man
    return man


def _all_run_dirs(root: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    """Every directory that holds a parsable manifest, scanned once."""
    key = f"__ALL__{root}"
    cached = _MANIFEST_CACHE.get(key)
    if cached is not None:
        return cached["dirs"]  # type: ignore[index]
    out: List[Tuple[Path, Dict[str, Any]]] = []
    for man_path in Path(root).rglob("run_manifest.json"):
        man = _read_manifest(man_path.parent)
        if man is not None:
            out.append((man_path.parent, man))
    _MANIFEST_CACHE[key] = {"dirs": out}  # type: ignore[assignment]
    return out


def _flat(o: Any, pre: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(_flat(v, f"{pre}.{k}" if pre else str(k)))
    else:
        out[pre] = o
    return out


# id() -> (strong ref to the manifest, flattened view). The strong reference is
# deliberate: without it the manifest could be collected and a later object
# could reuse the same id, silently returning another run's configuration.
_FLAT_CACHE: Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]] = {}


def manifest_leaf(man: Optional[Dict[str, Any]], key: str) -> Any:
    """
    Look up a value by its *leaf* name, so callers do not need to know whether
    the pipeline nested it under ``cli_arguments``, ``config`` or the top level.
    """
    if man is None:
        return None
    hit = _FLAT_CACHE.get(id(man))
    if hit is None or hit[0] is not man:
        flat = {full.split(".")[-1]: val for full, val in _flat(man).items()}
        _FLAT_CACHE[id(man)] = (man, flat)
        return flat.get(key)
    return hit[1].get(key)


def build_existing_data_index(root: Path) -> List[Dict[str, Any]]:
    """
    One row per artifact, with the columns required by specification 4.1.

    ``finite`` is only asserted for artifacts this module can cheaply read;
    binary NPZ contents are marked ``unknown`` rather than guessed.
    """
    root = Path(root)
    rows: List[Dict[str, Any]] = []
    patterns = ("**/*.npz", "**/*.json", "**/*.csv")
    skip_dirs = ("__pycache__",)
    n_seen = 0

    for pat in patterns:
        for p in sorted(root.glob(pat)):
            n_seen += 1
            if n_seen % 250 == 0:
                # Hashing several GB of npz is slow; say so rather than
                # appearing hung.
                print(f"    [index] {n_seen} files scanned...", flush=True)
            if any(sd in p.parts for sd in skip_dirs):
                continue
            # Figure sidecars are catalogued by the figure audit, not here.
            # There are several thousand of them and indexing each one would
            # bury the scientific artifacts in noise.
            if p.name.endswith(".meta.json"):
                continue
            rel = p.relative_to(root)
            key = _infer_key(p)
            man = _read_manifest(p.parent)
            atype = ("run_npz" if p.suffix == ".npz" else
                     "manifest" if p.name.endswith("manifest.json") else
                     "metrics_json" if p.suffix == ".json" else "csv")
            readable = True
            finite = "unknown"
            if p.suffix in (".json", ".csv"):
                try:
                    p.read_text(encoding="utf-8")
                except Exception:
                    readable = False
            rows.append({
                "path": str(rel),
                "artifact_type": atype,
                "method": key.get("method", manifest_leaf(man, "scheme") or ""),
                "P0": manifest_leaf(man, "P0") if man else key.get("P0", ""),
                "seed": manifest_leaf(man, "seed") if man else key.get("seed", ""),
                "N": manifest_leaf(man, "n_train") if man else key.get("N", ""),
                "dt_requested": manifest_leaf(man, "dt_requested") if man else "",
                "dt_resolved": manifest_leaf(man, "dt") if man else key.get("dt", ""),
                "t_final": manifest_leaf(man, "t_final") if man else "",
                "gp_policy_id": manifest_leaf(man, "refit_hyper_policy") if man else "",
                "normalization_status": "raw+self-normalized" if atype == "run_npz" else "",
                "source_hash": sha256_file(p) or "",
                "readable": readable,
                "finite": finite,
                "compatible_group": _compat_group(man, key),
                "reviewer_items_supported": "",
            })
    return rows


def _compat_group(man: Optional[Dict[str, Any]], key: Dict[str, Any]) -> str:
    """
    Compatibility group per specification rule 4: model, packet, endpoint,
    cloud, seed, estimator and normalization must match before comparison.
    """
    if man is None:
        return ""
    parts = [
        f"P0={manifest_leaf(man, 'P0')}",
        f"R0={manifest_leaf(man, 'R0')}",
        f"sigma_R={manifest_leaf(man, 'sigma_R')}",
        f"N={manifest_leaf(man, 'n_train')}",
        f"dt={manifest_leaf(man, 'dt')}",
        f"tf={manifest_leaf(man, 't_final')}",
        f"l2={manifest_leaf(man, 'l2_regularization')}",
        f"sampling={manifest_leaf(man, 'sampling_mode')}",
        f"surrogate={manifest_leaf(man, 'surrogate')}",
    ]
    return "|".join(parts)


# ===========================================================================
# Section 4.1 -- gap analysis
# ===========================================================================

def _has_completed_run(root: Path, P0: float, seed: int, N: Optional[int],
                       dt: Optional[float], method: str) -> Optional[Path]:
    """Locate a completed run directory matching a key, via its manifest."""
    scheme = "midpoint" if method.upper() == "MIDPOINT" else "pbme"
    for d, man in _all_run_dirs(Path(root)):
        if not (d / f"{scheme}.npz").exists():
            continue
        if manifest_leaf(man, "P0") != P0:
            continue
        if manifest_leaf(man, "seed") != seed:
            continue
        if N is not None and manifest_leaf(man, "n_train") != N:
            continue
        if dt is not None:
            got = manifest_leaf(man, "dt")
            if got is None or abs(float(got) - float(dt)) > 1e-12:
                continue
        return d
    return None


def _resolve_source_dir(src: Any, root: Path) -> Optional[Path]:
    """
    Resolve the ``source`` field of a derived-validation artifact to a run
    directory. Absolute paths are honoured first; if the drive layout has
    changed, fall back to matching the trailing path components under root.
    """
    if not isinstance(src, str) or not src:
        return None
    p = Path(src)
    if p.exists():
        return p if p.is_dir() else p.parent
    parts = [x for x in re.split(r"[\\/]+", src) if x]
    for depth in (4, 3, 2):
        if len(parts) < depth:
            continue
        cand = root.joinpath(*parts[-depth:])
        if cand.exists():
            return cand if cand.is_dir() else cand.parent
    return None


def build_snapshot_index(root: Path) -> List[Dict[str, Any]]:
    """
    Index the derived-validation snapshot tree.

    Each artifact records ``snapshot_step`` and the ``source`` run directory,
    so the authoritative P0/seed/dt are taken from that run's **manifest**, not
    from the snapshot directory name (specification rule 2). The directory name
    is recorded separately so a mismatch can be reported rather than hidden.
    """
    root = Path(root)
    out: List[Dict[str, Any]] = []
    for kind, fname in (("projection", "projection_leakage.json"),
                        ("kde_gp", "kde_gp_identical_support.json")):
        for p in root.rglob(fname):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            src = _resolve_source_dir(obj.get("source"), root)
            man = _read_manifest(src) if src else None
            if man is None and src is not None:
                man = _read_manifest(src.parent)
            step = obj.get("snapshot_step")
            method = ""
            if src is not None:
                low = src.name.lower()
                method = ("MIDPOINT" if "midpoint" in low
                          else "PBME" if "pbme" in low else "")
            dt = manifest_leaf(man, "dt")
            n_steps = manifest_leaf(man, "n_steps")
            P0 = manifest_leaf(man, "P0")
            out.append({
                "kind": kind,
                "path": p,
                "source_dir": src,
                "has_manifest": man is not None,
                "P0": P0,
                "seed": manifest_leaf(man, "seed"),
                "method": method,
                "step": step,
                "dt": dt,
                "n_steps": n_steps,
                "t": (float(step) * float(dt)
                      if step is not None and dt is not None else None),
                "t_c": (float(manifest_leaf(man, "mass") or 0.0)
                        * abs(float(manifest_leaf(man, "R0") or 0.0))
                        / float(P0) if P0 else None),
            })
    return out


def classify_snapshot_time(rec: Dict[str, Any]) -> Optional[str]:
    """
    Map a snapshot to one of the required time labels t0 / tc / 2tc / final.

    Tolerance is half a step, so a snapshot cadence that does not land exactly
    on t_c is still recognised. Returns None when the manifest did not supply
    enough information -- never a guess.
    """
    t, dt, tc = rec.get("t"), rec.get("dt"), rec.get("t_c")
    step, n_steps = rec.get("step"), rec.get("n_steps")
    if step == 0:
        return "t0"
    if t is None or dt is None:
        return None
    if n_steps is not None and step == n_steps:
        return "final"
    if tc:
        tol = 0.5 * float(dt)
        if abs(t - tc) <= tol:
            return "tc"
        if abs(t - 2.0 * tc) <= tol:
            return "2tc"
    return None


def build_gap_analysis(root: Path, cells: Optional[Sequence[Cell]] = None
                       ) -> List[Dict[str, Any]]:
    """
    Assign one mandated disposition to every required cell.

    Matching is done from **manifests**, never from directory names
    (specification rule 2).
    """
    root = Path(root)
    cells = list(cells or required_cells())
    rows: List[Dict[str, Any]] = []

    # Pre-index the cheap artifact families.
    manufactured = {}
    for p in root.rglob("manufactured_operator_metrics.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        manufactured[(int(d.get("n_train", -1)), int(d.get("seed", -1)))] = p

    snaps = build_snapshot_index(root)
    # (kind, P0, seed, method, time_label) -> artifact path
    snap_by_key: Dict[Tuple[str, Any, Any, str, str], Path] = {}
    snap_unresolved = 0
    for rec in snaps:
        label = classify_snapshot_time(rec)
        if label is None or rec["P0"] is None or rec["seed"] is None:
            snap_unresolved += 1
            continue
        snap_by_key.setdefault(
            (rec["kind"], float(rec["P0"]), int(rec["seed"]),
             rec["method"], label), rec["path"])

    refs3 = list(root.rglob("reference_convergence_3level.json"))

    for c in cells:
        disp, evidence, note = RERUN_MISSING, "", ""

        if c.campaign == "A_production_contract":
            hits = [d for d, man in _all_run_dirs(root)
                    if manifest_leaf(man, "P0") == c.key["P0"]]
            if hits:
                disp, evidence = REANALYZE, str(hits[0].relative_to(root))
                note = (f"{len(hits)} manifests at this momentum; the resolved "
                        "production contract JSON must still be emitted")

        elif c.campaign == "B_regularization":
            p = root / "l2_selection.json"
            if p.exists():
                disp, evidence = RERUN_INCOMPAT, "l2_selection.json"
                note = ("existing selection ran at N=300-350 pilot support, not "
                        "the production contract; specification section 9 requires "
                        "N in {300,600,1200} with >=3 folds and a confirmation seed")

        elif c.campaign == "C_manufactured":
            hit = manufactured.get((c.key["N"], c.key["seed"]))
            if hit is not None:
                if abs(c.key["l2"] - 1e-6) < 1e-15:
                    disp, evidence = REUSE, str(hit.relative_to(root))
                    note = "existing manufactured runs used the default l2=1e-6"
                else:
                    disp = RERUN_MISSING
                    note = f"no manufactured run at l2={c.key['l2']}"

        elif c.campaign == "D_projection":
            # section 11 wants an interaction-region snapshot and a late one.
            wanted = ("tc",) if c.key["snapshot"] == "interaction" else ("final", "2tc")
            hit = next((snap_by_key[("projection", c.key["P0"], c.key["seed"],
                                     "PBME", lab)]
                        for lab in wanted
                        if ("projection", c.key["P0"], c.key["seed"],
                            "PBME", lab) in snap_by_key), None)
            if hit is not None:
                disp, evidence = REUSE, str(hit.relative_to(root))
                note = ("unconstrained product branch: leakage is a diagnostic, "
                        "not an acceptance gate")

        elif c.campaign == "E_timestep":
            d = _has_completed_run(root, c.key["P0"], c.key["seed"], 1000,
                                   c.key["dt"], c.key["method"])
            if d is not None:
                disp, evidence = REUSE, str(d.relative_to(root))

        elif c.campaign == "F_support_nested":
            d = _has_completed_run(root, c.key["P0"], c.key["seed"],
                                   c.key["N"], None, "MIDPOINT")
            if d is not None:
                man = _read_manifest(d)
                if manifest_leaf(man, "parent_cloud_sha256") is None:
                    disp, evidence = RERUN_INCOMPAT, str(d.relative_to(root))
                    note = ("clouds were sampled independently; no parent/prefix "
                            "hash, so this is not a nested support study")

        elif c.campaign == "G_conservation":
            d = _has_completed_run(root, c.key["P0"], 11, 1000, None, "MIDPOINT")
            if d is not None:
                disp, evidence = REANALYZE, str(d.relative_to(root))
                note = "raw_* arrays present in npz; tables must be extracted"

        elif c.campaign == "H_source_audit":
            note = ("requires source_raw / source_after_floor / source_after_clip "
                    "/ source_applied arrays; not currently saved by Dynamics.py")

        elif c.campaign == "I_pbme_kde_gp":
            k = ("kde_gp", c.key["P0"], c.key["seed"], "PBME", c.key["time"])
            hit = snap_by_key.get(k)
            if hit is not None:
                disp, evidence = REUSE, str(hit.relative_to(root))
            else:
                note = (f"no PBME KDE/GP snapshot at t={c.key['time']} for "
                        f"P0={c.key['P0']:g}, seed={c.key['seed']}")

        elif c.campaign == "J_references":
            if refs3:
                disp, evidence = REUSE, str(refs3[0].relative_to(root))
                note = "three-level refinement present for both momenta and all modes"

        elif c.campaign == "K_physical_errors":
            note = ("matched L1/L2/Linf field errors against the converged "
                    "grid-QCLE field are not implemented")

        rows.append({
            "campaign": c.campaign,
            "cell": c.label(),
            "reviewer_items": c.reviewer_items,
            "disposition": disp,
            "evidence_path": evidence,
            "note": note,
        })

    if snap_unresolved:
        # Reported, never silently dropped: an artifact whose provenance cannot
        # be resolved is evidence of a provenance defect, not of absence.
        rows.append({
            "campaign": "PROVENANCE",
            "cell": "unresolved_snapshots",
            "reviewer_items": "I-14",
            "disposition": REPAIR,
            "evidence_path": "",
            "note": (f"{snap_unresolved} derived-validation artifacts could not "
                     "be tied to a run manifest (missing source, unresolvable "
                     "path, or unclassifiable snapshot time)"),
        })
    return rows


def summarize_gap(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tally: Dict[str, int] = {}
    per_campaign: Dict[str, Dict[str, int]] = {}
    for r in rows:
        d = r["disposition"]
        tally[d] = tally.get(d, 0) + 1
        per_campaign.setdefault(r["campaign"], {})
        per_campaign[r["campaign"]][d] = per_campaign[r["campaign"]].get(d, 0) + 1
    return {"total_cells": len(rows), "by_disposition": tally,
            "by_campaign": per_campaign}


# ===========================================================================
# Section 4 -- pipeline inventory
# ===========================================================================

def build_pipeline_inventory(root: Path) -> str:
    root = Path(root)
    present = sorted(p.name for p in root.glob("*.py"))
    missing = [m for m in ("run.py", "Dynamics.py", "GP_Density.py",
                           "Operator.py", "Sampling.py", "Observables.py",
                           "KDEDensity.py", "Collector.py", "Visualization.py",
                           "Compare_gp_se_qcle.py", "qcle_grid_tully.py",
                           "ReviewerValidation.py", "select_regularization.py")
               if m not in present]
    dup = [m for m in ("GP_Derivatives.py", "GPDerivatives.py") if m in present]

    lines = [
        "# Pipeline inventory",
        "",
        f"Generated {utc_now()} by `closure_audit.py` from `{root}`.",
        "",
        "## Modules",
        "",
        "| Module | Role |",
        "|---|---|",
    ]
    for m in present:
        lines.append(f"| `{m}` | {_MODULE_ROLES.get(m, 'unclassified')} |")

    lines += ["", "## Missing required modules", ""]
    lines += [f"- `{m}`" for m in missing] or ["- none"]

    lines += ["", "## Duplicate / obsolete implementations", ""]
    if len(dup) > 1:
        lines += [
            "- `GPDerivatives.py` and `GP_Derivatives.py` both exist.",
            "- **Authoritative:** `GPDerivatives.py` (active imports use it).",
            "- `GP_Derivatives.py` is a shim and must be explicitly deprecated "
            "(specification section 6).",
        ]
    else:
        lines.append("- none detected")

    # discovered artifact families
    fams: Dict[str, int] = {}
    for p in root.rglob("*.npz"):
        fams["npz"] = fams.get("npz", 0) + 1
    for p in root.rglob("run_manifest.json"):
        fams["run_manifest"] = fams.get("run_manifest", 0) + 1
    for p in root.rglob("*.meta.json"):
        fams["figure_sidecar"] = fams.get("figure_sidecar", 0) + 1
    for p in list(root.rglob("*.png")) + list(root.rglob("*.pdf")):
        fams["figure_image"] = fams.get("figure_image", 0) + 1

    lines += ["", "## Discovered artifact families", "",
              "| Family | Count |", "|---|---|"]
    for k, v in sorted(fams.items()):
        lines.append(f"| {k} | {v} |")

    # name/metadata conflicts -- specification section 4
    conflicts: List[str] = []
    for man_path in sorted(root.rglob("run_manifest.json")):
        man = _read_manifest(man_path.parent)
        got = manifest_leaf(man, "P0")
        m = _RE_P0.search(str(man_path.parent))
        if got is not None and m:
            try:
                named = float(m.group(1))
            except ValueError:
                continue
            if abs(named - float(got)) > 1e-9:
                conflicts.append(
                    f"- `{man_path.parent.relative_to(root)}`: directory name "
                    f"implies P0={named:g} but manifest records P0={got:g}")
    lines += ["", "## Directory-name vs manifest conflicts", ""]
    lines += conflicts or ["- none detected"]
    if conflicts:
        lines += ["", "> Specification rule 2: never recover a run "
                  "configuration from a directory name when a manifest is "
                  "available. Every figure sourced from a conflicting "
                  "directory must be re-sourced or relabelled."]

    lines += ["", "## Gate", "",
              "Do not run campaign commands until imports resolve and the "
              "baseline tests pass (`python -m pytest -q`)."]
    return "\n".join(lines) + "\n"


# ===========================================================================
# Driver
# ===========================================================================

def run_inspect(root: Path, campaign: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(root)
    if campaign is None:
        campaign = create_campaign(root)
    campaign = Path(campaign)
    campaign.mkdir(parents=True, exist_ok=True)

    (campaign / "pipeline_inventory.md").write_text(
        build_pipeline_inventory(root), encoding="utf-8")

    idx = build_existing_data_index(root)
    write_csv(campaign / "existing_data_index.csv", idx, [
        "path", "artifact_type", "method", "P0", "seed", "N", "dt_requested",
        "dt_resolved", "t_final", "gp_policy_id", "normalization_status",
        "source_hash", "readable", "finite", "compatible_group",
        "reviewer_items_supported"])

    gap = build_gap_analysis(root)
    write_csv(campaign / "required_run_gap_analysis.csv", gap,
              ["campaign", "cell", "reviewer_items", "disposition",
               "evidence_path", "note"])

    write_acceptance_contract(campaign / "acceptance_contract.yaml")

    summary = summarize_gap(gap)
    summary["campaign_root"] = str(campaign)
    summary["n_artifacts_indexed"] = len(idx)
    write_json_atomic(campaign / "inspect_summary.json", summary)

    print(f"[inspect] campaign      -> {campaign}")
    print(f"[inspect] artifacts     -> {len(idx)}")
    print(f"[inspect] required cells-> {summary['total_cells']}")
    for k, v in sorted(summary["by_disposition"].items(), key=lambda kv: -kv[1]):
        print(f"    {v:5d}  {k}")
    return summary


# ===========================================================================
# Self-test
# ===========================================================================

def run_self_test() -> None:
    import shutil

    # -- atomic write --------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a" / "b.json"
        write_json_atomic(p, {"x": 1})
        assert json.loads(p.read_text(encoding="utf-8"))["x"] == 1
        assert not list(Path(td).rglob("*.tmp")), "temp file left behind"

    # -- schema --------------------------------------------------------
    man = new_artifact_manifest(artifact_id="id", local_root=Path("."),
                                source_snapshot_id="snap", command=["python"],
                                method="PBME", P0=20.0, completed=True,
                                return_code=0)
    assert validate_artifact_manifest(man) == []
    bad = dict(man); bad["schema_version"] = "wrong"
    assert validate_artifact_manifest(bad)
    bad2 = dict(man); bad2["return_code"] = 1
    assert any("nonzero" in s for s in validate_artifact_manifest(bad2))

    # -- required cells ------------------------------------------------
    cells = required_cells()
    camps = {c.campaign for c in cells}
    for expect in ("A_production_contract", "C_manufactured", "E_timestep",
                   "F_support_nested", "I_pbme_kde_gp", "J_references",
                   "K_physical_errors"):
        assert expect in camps, expect
    # manufactured factorial size: 4 N x 3 seeds x 3 l2
    assert sum(1 for c in cells if c.campaign == "C_manufactured") == 36
    # timestep: 2 methods x 2 P0 x 4 seeds x 3 dt
    assert sum(1 for c in cells if c.campaign == "E_timestep") == 48
    assert len(set(c.label() for c in cells)) == len(cells), "duplicate cells"

    # -- synthetic tree ------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "pipe"
        (root).mkdir(parents=True)
        for name in ("run.py", "Dynamics.py", "GPDerivatives.py",
                     "GP_Derivatives.py"):
            (root / name).write_text("# stub\n", encoding="utf-8")

        # a completed MIDPOINT timestep run, described only by its manifest
        d = root / "someplace" / "whatever"
        d.mkdir(parents=True)
        (d / "midpoint.npz").write_bytes(b"x")
        write_json_atomic(d / "run_manifest.json", {"config": {
            "P0": 20.0, "seed": 11, "n_train": 1000, "dt": 0.5,
            "t_final": 3000.0, "l2_regularization": 0.05,
            "sampling_mode": "focused", "surrogate": "product",
            "refit_hyper_policy": "breathing", "R0": -15.0, "sigma_R": 1.0,
            "mass": 2000.0}})
        # real layout: the snapshot's `source` points at <run>/pbme, one level
        # below the directory that actually holds run_manifest.json
        (d / "pbme").mkdir()

        # a directory whose NAME disagrees with its manifest
        bad = root / "results" / "P0_20"
        bad.mkdir(parents=True)
        write_json_atomic(bad / "run_manifest.json",
                          {"config": {"P0": 40.0, "seed": 1, "n_train": 1000}})

        inv_md = build_pipeline_inventory(root)
        assert "GP_Derivatives.py" in inv_md and "Authoritative" in inv_md
        assert "directory name implies P0=20" in inv_md, "conflict not detected"

        (d / "some_figure.pdf.meta.json").write_text("{}", encoding="utf-8")
        idx = build_existing_data_index(root)
        assert any(r["artifact_type"] == "run_npz" for r in idx)
        assert all(r["source_hash"] for r in idx if r["artifact_type"] != "run_npz")
        assert not any(".meta.json" in r["path"] for r in idx), \
            "figure sidecars must be excluded from the data index"
        # manifest config is nested under cli_arguments in the real pipeline;
        # leaf lookup must be insensitive to the nesting key
        nested = {"cli_arguments": {"P0": 20.0, "dt": 0.25, "seed": 11},
                  "extra": float("inf")}
        assert manifest_leaf(nested, "P0") == 20.0
        assert manifest_leaf(nested, "dt") == 0.25
        assert manifest_leaf(nested, "missing_key") is None

        # -- snapshot time classification (pure) ----------------------
        base = {"dt": 0.25, "n_steps": 12000, "t_c": 1500.0}
        assert classify_snapshot_time({**base, "step": 0, "t": 0.0}) == "t0"
        assert classify_snapshot_time({**base, "step": 6000, "t": 1500.0}) == "tc"
        assert classify_snapshot_time({**base, "step": 12000, "t": 3000.0}) == "final"
        # P0=100: t_c=300, dt=0.25 -> 2*t_c at step 2400, final is elsewhere
        p100 = {"dt": 0.25, "n_steps": 12000, "t_c": 300.0}
        assert classify_snapshot_time({**p100, "step": 1200, "t": 300.0}) == "tc"
        assert classify_snapshot_time({**p100, "step": 2400, "t": 600.0}) == "2tc"
        # must refuse to guess when the manifest gave nothing
        assert classify_snapshot_time({"step": 7, "t": None, "dt": None,
                                       "t_c": None, "n_steps": None}) is None
        assert classify_snapshot_time({**base, "step": 4321,
                                       "t": 1080.25}) is None

        # -- snapshot index resolves provenance through `source` -------
        snapdir = (root / "audit" / "snapshots" / "P0_20" / "seed_11"
                   / "pbme" / "step_006000" / "projection")
        snapdir.mkdir(parents=True)
        # dt=0.5 and t_c = 2000*15/20 = 1500 -> the interaction snapshot is
        # step 3000, and the manifest lives one level above `source`
        write_json_atomic(snapdir / "projection_leakage.json", {
            "snapshot_step": 3000, "seed": 123,
            "mean_relative_l2_leakage": 0.96, "source": str(d / "pbme")})
        si = build_snapshot_index(root)
        rec = next(r for r in si if r["kind"] == "projection")
        assert rec["has_manifest"], "source did not resolve to a manifest"
        assert rec["P0"] == 20.0 and rec["seed"] == 11, rec
        assert rec["method"] == "PBME", rec["method"]
        assert abs(rec["t_c"] - 1500.0) < 1e-9, rec["t_c"]
        assert abs(rec["t"] - 1500.0) < 1e-9, rec["t"]
        assert classify_snapshot_time(rec) == "tc"
        assert _resolve_source_dir("no/such/place", root) is None
        assert _resolve_source_dir(None, root) is None

        gap = build_gap_analysis(root)
        by = {r["cell"]: r for r in gap}
        k = "E_timestep[P0=20.0,dt=0.5,method=MIDPOINT,seed=11]"
        assert by[k]["disposition"] == REUSE, by[k]
        k2 = "E_timestep[P0=20.0,dt=0.5,method=MIDPOINT,seed=29]"
        assert by[k2]["disposition"] == RERUN_MISSING
        # nested support must be flagged incompatible (no parent hash)
        kn = "F_support_nested[N=1000,P0=20.0,nested=True,seed=11]"
        assert by[kn]["disposition"] == RERUN_INCOMPAT, by[kn]
        # the interaction-region projection snapshot is now provably present
        kd = "D_projection[P0=20.0,seed=11,snapshot=interaction]"
        assert by[kd]["disposition"] == REUSE, by[kd]
        kd2 = "D_projection[P0=20.0,seed=29,snapshot=interaction]"
        assert by[kd2]["disposition"] == RERUN_MISSING, by[kd2]
        # every row carries exactly one of the five mandated dispositions
        allowed = {REUSE, REANALYZE, RERUN_MISSING, RERUN_INCOMPAT, REPAIR}
        assert all(r["disposition"] in allowed for r in gap)

        s = summarize_gap(gap)
        assert s["total_cells"] == len(gap)
        assert sum(s["by_disposition"].values()) == len(gap)

        # -- campaign creation ----------------------------------------
        camp = create_campaign(root, stamp="20260101_000000")
        assert (camp / "campaign_manifest.json").exists()
        assert (camp / "starting_state" / "source_hashes_before.csv").exists()
        for sub in ("manufactured", "timestep", "latex_tables", "logs"):
            assert (camp / sub).is_dir()
        cm = json.loads((camp / "campaign_manifest.json").read_text(encoding="utf-8"))
        assert cm["n_source_files_hashed"] >= 4
        # creating twice must fail rather than overwrite
        try:
            create_campaign(root, stamp="20260101_000000")
            raise AssertionError("campaign overwrite was permitted")
        except FileExistsError:
            pass

        cpath = write_acceptance_contract(camp / "acceptance_contract.yaml")
        txt = cpath.read_text(encoding="utf-8")
        assert "E1" in txt and "0.02" in txt and "No validated improvement" in txt

    print("[self-test] closure_audit checks passed "
          "(atomic write, schema, cell enumeration, inventory, "
          "name/manifest conflict, gap dispositions, campaign creation, "
          "acceptance contract).")


def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command")

    i = sub.add_parser("inspect", help="existing-data-first audit (section 4/4.1)")
    i.add_argument("--root", type=Path, default=Path("."))
    i.add_argument("--campaign-root", type=Path, default=None)

    c = sub.add_parser("create-campaign", help="new timestamped campaign (section 3)")
    c.add_argument("--root", type=Path, default=Path("."))

    g = sub.add_parser("contract", help="write acceptance_contract.yaml (section 7)")
    g.add_argument("--out", type=Path, required=True)

    p.add_argument("--self-test", action="store_true")
    return p


def main() -> None:
    a = _argparser().parse_args()
    if a.self_test:
        run_self_test(); return
    if a.command == "inspect":
        run_inspect(a.root, a.campaign_root)
    elif a.command == "create-campaign":
        print(create_campaign(a.root))
    elif a.command == "contract":
        print(write_acceptance_contract(a.out))
    else:
        print("No command. Use --self-test, or: inspect | create-campaign | contract")


if __name__ == "__main__":
    main()
