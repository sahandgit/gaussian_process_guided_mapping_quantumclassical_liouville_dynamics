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
thesis_closure.py
=================

Closes the two remaining reviewer items that are addressable by code, and
provides the machinery for the third (nested support) so a future campaign can
satisfy it properly.

  A. Three-level reference convergence  (review sections 2/Q2, 6, 15/Ch.2 Q2)
     The shipped ``ReviewerValidation reference`` performs a TWO-level
     refinement, and two-level agreement is not convergence.  This module runs
     three levels, separates time refinement from grid refinement as the
     specification requires, estimates the observed order, and records the
     grid metadata (domain, spacing, k_max, boundary rule, boundary-strip
     occupancy) that the reviewer asked for.

  B. Figure and caption compliance audit  (review section 7)
     Scans the figure ``.meta.json`` sidecars and reports, per figure, which
     of the required caption fields are missing.  Emits a machine-readable
     report plus a LaTeX table.

  C. Nested support subsets  (review section 6.1 "Support")
     The executed Step-8 clouds were sampled independently at each N, so the
     refinement difference mixes support size with sampling noise.  This
     module supplies the deterministic nested-subset generator and a manifest
     writer so a future run can define N=500 as a strict subset of N=1000 of
     N=2000.  It does NOT retroactively make the existing data nested -- that
     limitation must be stated in the thesis.

Everything here is torch-free.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from reviewer_closure_campaign import nested_subset_indices, observed_order

NOT_COMPUTED = "NOT COMPUTED"


# ===========================================================================
# A. Three-level reference convergence
# ===========================================================================

_REF_OBSERVABLES = (
    "P0", "P1", "trace", "energy", "R_mean", "P_mean", "R_var", "P_var",
)


@dataclass
class ReferenceLevel:
    """One refinement level of a reference calculation."""
    label: str
    dt: float
    n_steps: int
    n_grid: int
    endpoints: Dict[str, float]
    metadata: Dict[str, Any]


def _endpoints(result: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in _REF_OBSERVABLES:
        v = result.get(k)
        if v is None:
            continue
        a = np.asarray(v, float).reshape(-1)
        if a.size:
            out[k] = float(a[-1])
    return out


def _order_from_three(a: float, b: float, c: float) -> Tuple[Optional[float], str]:
    """Observed order from three endpoint values (coarse, mid, fine)."""
    return observed_order(np.array([a]), np.array([b]), np.array([c]))


def tdse_three_level(P0: float, R0: float = -10.0, sigma_R: float = 1.0,
                     dt: float = 0.2, n_steps: int = 200,
                     n_grid: int = 2048,
                     refine: str = "both",
                     series_out: Optional[Path] = None) -> Dict[str, Any]:
    r"""
    Three-level TDSE refinement.

    ``refine``:
      * ``"both"``  -- (dt, N), (dt/2, 2N), (dt/4, 4N)
      * ``"time"``  -- refine dt at the finest grid (isolates temporal error)
      * ``"grid"``  -- refine N at the finest dt (isolates spatial error)
    """
    from Compare_gp_se_qcle import run_tdse

    if refine == "both":
        cfgs = [("coarse", dt, n_grid), ("fine", dt / 2, 2 * n_grid),
                ("finer", dt / 4, 4 * n_grid)]
    elif refine == "time":
        g = 4 * n_grid
        cfgs = [("coarse", dt, g), ("fine", dt / 2, g), ("finer", dt / 4, g)]
    elif refine == "grid":
        h = dt / 4
        cfgs = [("coarse", h, n_grid), ("fine", h, 2 * n_grid),
                ("finer", h, 4 * n_grid)]
    else:
        raise ValueError(f"unknown refine mode {refine!r}")

    levels: List[ReferenceLevel] = []
    T = n_steps * dt
    for label, h, g in cfgs:
        ns = int(round(T / h))
        save_every = (
            max(1, int(round(0.25 / h)))
            if series_out is not None and label == "finer"
            else max(1, ns)
        )
        res = run_tdse(
            R0, P0, sigma_R, h, ns, n_grid_min=g,
            save_every=save_every,
            t_snapshots=([T] if series_out is not None and label == "finer" else None),
            verbose=False,
        )
        if series_out is not None and label == "finer":
            series_out = Path(series_out)
            series_out.parent.mkdir(parents=True, exist_ok=True)
            keep = {
                key: np.asarray(value)
                for key, value in res.items()
                if key in (
                    "t", "P0", "P1", "trace", "energy", "R_mean", "P_mean",
                    "R_var", "P_var", "edge_mass_5pct",
                    "negative_momentum_probability", "snap_R", "snap_dR",
                    "snap_k", "snap_Lx", "snap_hbar", "snap_psi", "snap_t",
                )
            }
            np.savez_compressed(series_out, **keep)
        meta: Dict[str, Any] = {"n_grid_requested": g, "dt": h, "n_steps": ns,
                                "t_final": T}
        # Grid metadata the reviewer asked for, where the solver exposes it.
        for key, name in (("snap_R", "R_grid"), ("dR", "dR"), ("k_max", "k_max")):
            if key in res:
                arr = np.asarray(res[key])
                if key == "snap_R" and arr.size > 1:
                    meta["R_min"] = float(arr.min())
                    meta["R_max"] = float(arr.max())
                    meta["dR"] = float(arr[1] - arr[0])
                    meta["k_max"] = float(np.pi / (arr[1] - arr[0]))
                    meta["n_grid_actual"] = int(arr.size)
                else:
                    meta[name] = float(np.asarray(res[key]).reshape(-1)[0])
        meta.setdefault("boundary_rule", "periodic (split-operator FFT)")
        if "edge_mass_5pct" in res:
            edge = np.asarray(res["edge_mass_5pct"], float).reshape(-1)
            meta["edge_mass_5pct"] = float(edge[-1])
            meta["maximum_edge_mass_5pct"] = float(np.max(edge))
        if "negative_momentum_probability" in res:
            reflected = np.asarray(
                res["negative_momentum_probability"], float
            ).reshape(-1)
            meta["negative_momentum_probability"] = float(reflected[-1])
            meta["maximum_negative_momentum_probability"] = float(
                np.max(reflected)
            )
        meta["absorber_policy"] = "none"
        meta["split_operator_composition"] = "symmetric V/2-T-V/2"
        meta["fft_convention"] = "periodic numpy FFT; P=hbar*2*pi*fftfreq"
        meta["edge_mass_convention"] = (
            "probability in the two outer 5% spatial bands"
        )
        levels.append(ReferenceLevel(label, h, ns, g, _endpoints(res), meta))

    assembled = _assemble_reference("tdse", levels, refine)
    if series_out is not None:
        assembled["finest_series_file"] = str(series_out)
    return assembled


def qcle_three_level(P0: float, R0: float = -10.0, sigma_R: float = 1.0,
                     dt: float = 0.2, n_steps: int = 200,
                     n_R: int = 192, n_P: int = 128,
                     R_min: float = -25.0, R_max: float = 25.0,
                     P_min: float = -35.0, P_max: float = 35.0,
                     refine: str = "both",
                     series_out: Optional[Path] = None) -> Dict[str, Any]:
    """Three-level grid-QCLE refinement, same ``refine`` semantics."""
    from Compare_gp_se_qcle import run_qcle
    from qcle_grid_tully import QCLEGridParams

    if refine == "both":
        cfgs = [("coarse", dt, n_R, n_P), ("fine", dt / 2, 2 * n_R, 2 * n_P),
                ("finer", dt / 4, 4 * n_R, 4 * n_P)]
    elif refine == "time":
        cfgs = [("coarse", dt, 4 * n_R, 4 * n_P),
                ("fine", dt / 2, 4 * n_R, 4 * n_P),
                ("finer", dt / 4, 4 * n_R, 4 * n_P)]
    elif refine == "grid":
        h = dt / 4
        cfgs = [("coarse", h, n_R, n_P), ("fine", h, 2 * n_R, 2 * n_P),
                ("finer", h, 4 * n_R, 4 * n_P)]
    else:
        raise ValueError(f"unknown refine mode {refine!r}")

    levels: List[ReferenceLevel] = []
    T = n_steps * dt
    for label, h, nr, npp in cfgs:
        ns = int(round(T / h))
        params = QCLEGridParams(R_min=R_min, R_max=R_max, P_min=P_min,
                                P_max=P_max, n_R=nr, n_P=npp,
                                mass=2000.0, hbar=1.0)
        save_every = (
            max(1, int(round(0.25 / h)))
            if series_out is not None and label == "finer"
            else max(1, ns)
        )
        res = run_qcle(
            R0, P0, sigma_R, h, ns, qcle_params=params,
            save_every=save_every,
            t_snapshots=([T] if series_out is not None and label == "finer" else None),
            verbose=False,
        )
        if series_out is not None and label == "finer":
            series_out = Path(series_out)
            series_out.parent.mkdir(parents=True, exist_ok=True)
            keep = {
                key: np.asarray(value)
                for key, value in res.items()
                if key in (
                    "t", "P0", "P1", "trace", "energy", "R_mean", "P_mean",
                    "R_var", "P_var", "edge_R_mass_5pct",
                    "edge_P_mass_5pct", "edge_phase_space_R_mass_5pct",
                    "edge_phase_space_P_mass_5pct", "cfl_dt_max", "snap_Rg", "snap_dR",
                    "snap_dP", "snap_R_axis", "snap_P_axis",
                )
            }
            if res.get("snap_states"):
                final_state = res["snap_states"][-1]
                keep.update({
                    "snap_A": np.asarray(final_state.A),
                    "snap_C": np.asarray(final_state.C),
                    "snap_bR": np.asarray(final_state.bR),
                    "snap_bI": np.asarray(final_state.bI),
                    "snap_t": np.asarray(res.get("snap_t", [])),
                })
            np.savez_compressed(series_out, **keep)
        meta = {"n_R": nr, "n_P": npp, "dt": h, "n_steps": ns, "t_final": T,
                "R_min": R_min, "R_max": R_max, "P_min": P_min, "P_max": P_max,
                "dR": (R_max - R_min) / nr, "dP": (P_max - P_min) / npp,
                "boundary_rule": "periodic (pseudospectral rfft)"}
        if "edge_R_mass_5pct" in res:
            edge_R = np.asarray(res["edge_R_mass_5pct"], float).reshape(-1)
            meta["edge_R_mass_5pct"] = float(edge_R[-1])
            meta["maximum_edge_R_mass_5pct"] = float(np.max(edge_R))
        if "edge_P_mass_5pct" in res:
            edge_P = np.asarray(res["edge_P_mass_5pct"], float).reshape(-1)
            meta["edge_P_mass_5pct"] = float(edge_P[-1])
            meta["maximum_edge_P_mass_5pct"] = float(np.max(edge_P))
        for axis in ("R", "P"):
            key = f"edge_phase_space_{axis}_mass_5pct"
            if key in res:
                edge_phase = np.asarray(res[key], float).reshape(-1)
                meta[key] = float(edge_phase[-1])
                meta[f"maximum_{key}"] = float(np.max(edge_phase))
        if "cfl_dt_max" in res:
            cfl_max = float(np.asarray(res["cfl_dt_max"], float).reshape(-1)[0])
            meta["cfl_dt_max"] = cfl_max
            meta["cfl_ratio"] = float(h / cfl_max) if cfl_max > 0 else None
        meta["derivative_method"] = "Fourier pseudospectral"
        meta["time_integrator"] = "classical RK4"
        meta["edge_mass_convention"] = (
            "absolute physical R/P marginal mass in the outer 5% bands "
            "divided by total absolute marginal mass; phase-space integral "
            "of abs(W) retained separately as a numerical-ringing diagnostic"
        )
        levels.append(ReferenceLevel(label, h, ns, nr * npp,
                                     _endpoints(res), meta))

    assembled = _assemble_reference("qcle", levels, refine)
    if series_out is not None:
        assembled["finest_series_file"] = str(series_out)
    return assembled


def _assemble_reference(method: str, levels: Sequence[ReferenceLevel],
                        refine: str) -> Dict[str, Any]:
    """Differences and observed order across three levels."""
    out: Dict[str, Any] = {
        "method": method, "refine_mode": refine,
        "levels": [{"label": l.label, "dt": l.dt, "n_steps": l.n_steps,
                    "n_grid": l.n_grid, "endpoints": l.endpoints,
                    "metadata": l.metadata} for l in levels],
        "observables": {},
    }
    if len(levels) < 3:
        out["status"] = f"{NOT_COMPUTED}: fewer than three levels"
        return out
    a, b, c = levels[0], levels[1], levels[2]
    for key in _REF_OBSERVABLES:
        if not all(key in l.endpoints for l in (a, b, c)):
            continue
        va, vb, vc = a.endpoints[key], b.endpoints[key], c.endpoints[key]
        d1, d2 = abs(va - vb), abs(vb - vc)
        p, why = _order_from_three(va, vb, vc)
        out["observables"][key] = {
            "coarse": va, "fine": vb, "finer": vc,
            "abs_diff_coarse_fine": d1, "abs_diff_fine_finer": d2,
            "ratio": (d1 / d2) if d2 > 0 else None,
            "p_obs": (float(p) if p is not None else NOT_COMPUTED),
            "reason": why,
        }
    out["status"] = "COMPLETE"
    return out


def collision_time(P0: float, R0: float, mass: float = 2000.0) -> float:
    """t_c = M|R0|/|P0| -- the time at which the packet reaches the crossing."""
    return mass * abs(R0) / abs(P0)


def run_reference_study(out_dir: Path, P0_list: Sequence[float] = (20.0, 100.0),
                        dt: float = 0.2, n_steps: int = 200,
                        modes: Sequence[str] = ("both",),
                        R0: float = -10.0, mass: float = 2000.0,
                        methods: Sequence[str] = ("tdse", "qcle"),
                        ) -> Dict[str, Any]:
    """
    Run the three-level study for each momentum and refinement mode.

    A refinement study only has power where the propagator is not already
    exact. In the flat asymptotic region of the Tully model the split-operator
    FFT propagator is exact for free motion (the kinetic step is exact in
    momentum space and the potential is locally constant), so refining dt or
    the grid changes the endpoints by nothing at all. Each block therefore
    records whether the requested window actually reaches the avoided crossing,
    and the caller is warned when it does not.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fns = {"tdse": tdse_three_level, "qcle": qcle_three_level}
    all_results: Dict[str, Any] = {}
    t_final = dt * n_steps
    for P0 in P0_list:
        t_c = collision_time(P0, R0, mass)
        reaches = t_final >= t_c
        if not reaches:
            print(f"[reference] WARNING P0={P0:g}: t_final={t_final:g} < "
                  f"t_c={t_c:g}. The packet never reaches the crossing; the "
                  f"propagator is exact for free motion there and the "
                  f"refinement test will have no power.", flush=True)
        for mode in modes:
            for name in methods:
                key = f"{name}_P0{P0:g}_{mode}"
                print(f"[reference] {key} (t_final={t_final:g}, "
                      f"t_c={t_c:g}) ...", flush=True)
                try:
                    res = fns[name](P0=P0, R0=R0, dt=dt, n_steps=n_steps,
                                    refine=mode)
                except Exception as exc:
                    res = {"status": f"FAILED: {type(exc).__name__}: {exc}"}
                    print(f"  FAILED: {exc}")
                if isinstance(res, dict):
                    res["R0"] = R0
                    res["mass"] = mass
                    res["t_final"] = t_final
                    res["collision_time"] = t_c
                    res["window_reaches_crossing"] = reaches
                all_results[key] = res
    p = out_dir / "reference_convergence_3level.json"
    p.write_text(json.dumps(all_results, indent=2, default=str),
                 encoding="utf-8")
    print(f"[reference] wrote {p}")
    return all_results


# ===========================================================================
# B. Figure and caption compliance audit
# ===========================================================================

# The pipeline's actual sidecar schema (FigureCatalog):
#   figure, title, normalization, scale_policy, data_sources,
#   run_metadata, deviations_from_run_configuration
# Sidecar-level fields that must be present and non-empty.
SIDECAR_FIELDS = ("figure", "normalization", "scale_policy", "run_metadata")

# Provenance the reviewer requires each caption to state or inherit
# (report section 7). These live inside ``run_metadata`` when it is populated.
PROVENANCE_FIELDS = (
    "P0", "n_train", "seed", "dt", "sampling_mode", "surrogate",
    "l2_regularization", "refit_hyper_policy", "product_g_floor_rel",
    "init_log_sigma_n",
)

# Retained for backwards compatibility with earlier callers.
REQUIRED_CAPTION_FIELDS = SIDECAR_FIELDS + PROVENANCE_FIELDS
OPTIONAL_CAPTION_FIELDS = ("bandwidth", "threshold", "adaptive_activations")


def _meta_has(meta: Dict[str, Any], field: str) -> bool:
    """True if any leaf key matches, case-insensitively and loosely."""
    target = field.lower().replace("_", "")

    def walk(o: Any) -> bool:
        if isinstance(o, dict):
            for k, v in o.items():
                if str(k).lower().replace("_", "") == target:
                    return True
                if walk(v):
                    return True
        elif isinstance(o, list):
            return any(walk(v) for v in o)
        return False

    return walk(meta)


def _is_populated(v: Any) -> bool:
    """A field counts as present only if it is non-null and non-empty."""
    if v is None:
        return False
    if isinstance(v, (str, list, dict, tuple)):
        return len(v) > 0
    return True


def audit_figures(roots: Sequence[Path]) -> List[Dict[str, Any]]:
    """
    One row per FIGURE (not per file), against the pipeline's real sidecar
    schema.

    Two distinct failure modes are separated, because they need different
    fixes:

    * ``sidecar_missing``   -- required top-level keys absent or empty.
    * ``provenance_missing`` -- ``run_metadata`` is null/empty, so N, seed,
      dt, GP policy, floor, and regularization are not recorded for that
      figure.  This is the substantive failure under review section 7.

    PDF/PNG pairs of the same figure are counted once.
    """
    rows: List[Dict[str, Any]] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        seen: set = set()
        for fig in sorted(list(root.rglob("*.pdf")) + list(root.rglob("*.png"))):
            stem = fig.with_suffix("")            # dedupe pdf/png pairs
            if stem in seen:
                continue
            seen.add(stem)

            side = Path(str(fig) + ".meta.json")
            if not side.exists():
                for cand in (Path(str(stem) + ".pdf.meta.json"),
                             Path(str(stem) + ".png.meta.json")):
                    if cand.exists():
                        side = cand
                        break
            rel = str(stem.relative_to(root))
            if not side.exists():
                rows.append({"figure": rel, "root": root.name,
                             "has_sidecar": False,
                             "sidecar_missing": list(SIDECAR_FIELDS),
                             "provenance_missing": list(PROVENANCE_FIELDS),
                             "n_missing": len(SIDECAR_FIELDS)
                                          + len(PROVENANCE_FIELDS),
                             "status": "FAIL: no .meta.json sidecar"})
                continue
            try:
                meta = json.loads(side.read_text(encoding="utf-8"))
            except Exception as exc:
                rows.append({"figure": rel, "root": root.name,
                             "has_sidecar": True,
                             "sidecar_missing": list(SIDECAR_FIELDS),
                             "provenance_missing": list(PROVENANCE_FIELDS),
                             "n_missing": len(SIDECAR_FIELDS)
                                          + len(PROVENANCE_FIELDS),
                             "status": f"FAIL: unreadable ({exc})"})
                continue

            sm = [f for f in SIDECAR_FIELDS if not _is_populated(meta.get(f))]
            run_meta = meta.get("run_metadata")
            if _is_populated(run_meta):
                pm = [f for f in PROVENANCE_FIELDS if not _meta_has(run_meta, f)]
            else:
                pm = list(PROVENANCE_FIELDS)      # nothing recorded at all
            n_missing = len(sm) + len(pm)
            rows.append({
                "figure": rel, "root": root.name, "has_sidecar": True,
                "sidecar_missing": sm, "provenance_missing": pm,
                "n_missing": n_missing,
                "status": "PASS" if n_missing == 0 else
                          ("FAIL: run_metadata empty" if not _is_populated(run_meta)
                           else "FAIL: missing fields"),
            })
    return rows


def figure_audit_report(rows: Sequence[Dict[str, Any]], out_dir: Path
                        ) -> Dict[str, Any]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    # Build the full summary FIRST, then write it once.  (A previous version
    # wrote the JSON before adding these fields, so they never reached disk.)
    tally: Dict[str, int] = {}
    for r in rows:
        for f in (list(r.get("sidecar_missing", []))
                  + list(r.get("provenance_missing", []))
                  + list(r.get("missing", []))):
            tally[f] = tally.get(f, 0) + 1

    summary = {
        "n_figures": n,
        "n_pass": n_pass,
        "n_fail": n - n_pass,
        "n_no_sidecar": sum(1 for r in rows if not r.get("has_sidecar", True)),
        "n_empty_run_metadata": sum(
            1 for r in rows if "run_metadata empty" in str(r.get("status", ""))),
        "missing_field_counts": tally,
        "rows": list(rows),
    }
    (out_dir / "figure_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["% Auto-generated by thesis_closure.py",
             r"\begin{table}[p]", r"\centering", r"\small",
             r"\caption{Figure metadata audit: number of production figures "
             r"missing each required caption field.}",
             r"\label{tab:figure-audit}",
             r"\begin{tabular}{lr}", r"\toprule",
             r"Required field & Figures missing it \\", r"\midrule"]
    for f, c in sorted(tally.items(), key=lambda kv: -kv[1]):
        lines.append(f"{f.replace('_', chr(92) + '_')} & {c} \\\\")
    lines += [r"\midrule",
              rf"Total figures & {n} \\",
              rf"Compliant & {n_pass} \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (out_dir / "figure_audit.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"[figure-audit] {n_pass}/{n} figures compliant -> "
          f"{out_dir / 'figure_audit.json'}")
    return summary


# ===========================================================================
# C. Nested support subsets (for future campaigns)
# ===========================================================================

def write_nested_support_plan(out: Path, n_max: int = 2000,
                              levels: Sequence[int] = (500, 1000, 2000),
                              seeds: Sequence[int] = (11, 29, 47)
                              ) -> Dict[str, Any]:
    r"""
    Emit a deterministic nested-subset plan.

    For each seed one permutation of ``range(n_max)`` is drawn and the levels
    are prefixes of it, so N=500 is a strict subset of N=1000 of N=2000.  This
    is what review section 6.1 requires and what the executed campaign did NOT
    do.

    Consuming this plan requires ``run.py`` to accept an externally supplied
    support-index file; that hook does not exist yet.  The plan is therefore a
    specification for the next campaign, not a retroactive fix.
    """
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    plan: Dict[str, Any] = {"n_max": int(n_max), "levels": list(levels),
                            "seeds": list(seeds),
                            "note": "Levels are prefixes of one permutation "
                                    "per seed, so smaller levels are strict "
                                    "subsets of larger ones.",
                            "per_seed": {}}
    for seed in seeds:
        sub = nested_subset_indices(n_max, levels, seed)
        # verify the nesting property explicitly before writing it out
        ordered = sorted(sub)
        for small, large in zip(ordered, ordered[1:]):
            assert set(sub[small]).issubset(set(sub[large])), (seed, small, large)
        plan["per_seed"][str(seed)] = {str(k): [int(i) for i in v]
                                       for k, v in sub.items()}
    p = out / "nested_support_plan.json"
    p.write_text(json.dumps(plan), encoding="utf-8")
    print(f"[nested-support] wrote {p} "
          f"({len(seeds)} seeds x {len(levels)} levels, verified nested)")
    return plan


def executed_support_is_nested(root: Path) -> Dict[str, Any]:
    """
    Report whether the executed Step-8 clouds were nested.

    The campaign driver sampled each N independently, so the answer is No.
    This function states that explicitly for the thesis rather than leaving it
    to be inferred, and records the evidence it could and could not check.
    """
    from thesis_analysis import discover_step8, load_manifest, manifest_value
    runs = discover_step8(Path(root))
    hashes: Dict[Tuple[float, int, int], Any] = {}
    for key, d in runs.items():
        man = load_manifest(d)
        hashes[key] = manifest_value(man, "initial_cloud_sha256")
    return {
        "root": str(root),
        "n_runs": len(runs),
        "nested": False,
        "basis": ("The campaign driver invoked run.py once per (P0, seed, N) "
                  "with independent sampling; no support-index file was "
                  "supplied, so smaller clouds are not subsets of larger "
                  "ones."),
        "consequence": ("Support-refinement differences combine support size "
                        "with sampling noise and must be reported against the "
                        "independent-seed standard deviation. Do not claim a "
                        "deterministic support convergence order."),
        "initial_cloud_hashes": {f"P0{k[0]:g}_seed{k[1]}_N{k[2]}": v
                                 for k, v in sorted(hashes.items())},
    }


# ===========================================================================
# Self-test
# ===========================================================================

def run_self_test() -> None:
    import tempfile

    # --- A: assembly logic on synthetic exact 2nd-order endpoints ---------
    lv = [ReferenceLevel("coarse", 0.2, 100, 2048, {"P0": 4.0}, {}),
          ReferenceLevel("fine", 0.1, 200, 4096, {"P0": 1.0}, {}),
          ReferenceLevel("finer", 0.05, 400, 8192, {"P0": 0.25}, {})]
    res = _assemble_reference("tdse", lv, "both")
    assert res["status"] == "COMPLETE"
    p = res["observables"]["P0"]["p_obs"]
    assert abs(float(p) - 2.0) < 1e-9, p
    assert res["observables"]["P0"]["ratio"] == 4.0

    two = _assemble_reference("tdse", lv[:2], "both")
    assert NOT_COMPUTED in str(two["status"])

    # --- B: figure audit --------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "root"
        (root / "figs").mkdir(parents=True)

        # fully compliant: sidecar fields present AND run_metadata populated
        good = root / "figs" / "a.png"
        good.write_bytes(b"x")
        Path(str(good) + ".meta.json").write_text(json.dumps({
            **{k: "v" for k in SIDECAR_FIELDS},
            "run_metadata": {k: 1 for k in PROVENANCE_FIELDS},
        }), encoding="utf-8")

        # the real-world failure: sidecar present but run_metadata null
        bad = root / "figs" / "b.png"
        bad.write_bytes(b"x")
        Path(str(bad) + ".meta.json").write_text(json.dumps({
            "figure": "b.png", "normalization": "stated",
            "scale_policy": "shared", "run_metadata": None,
        }), encoding="utf-8")

        none = root / "figs" / "c.png"
        none.write_bytes(b"x")

        rows = audit_figures([root])
        by = {Path(r["figure"]).name: r for r in rows}
        assert by["a"]["status"] == "PASS", by["a"]
        assert "run_metadata empty" in by["b"]["status"], by["b"]
        assert by["b"]["provenance_missing"] == list(PROVENANCE_FIELDS)
        assert by["c"]["has_sidecar"] is False
        summ = figure_audit_report(rows, Path(td) / "out")
        assert summ["n_figures"] == 3 and summ["n_pass"] == 1
        assert summ["n_empty_run_metadata"] == 1
        assert summ["n_no_sidecar"] == 1

        # pdf/png pair of the same figure counts once
        (root / "figs" / "a.pdf").write_bytes(b"x")
        assert len(audit_figures([root])) == 3

        # --- C: nested plan ----------------------------------------------
        plan = write_nested_support_plan(Path(td) / "plan", n_max=200,
                                         levels=(50, 100, 200), seeds=(11, 29))
        for seed, lv_map in plan["per_seed"].items():
            s50, s100, s200 = (set(lv_map["50"]), set(lv_map["100"]),
                               set(lv_map["200"]))
            assert s50 < s100 < s200, seed
            assert len(s50) == 50 and len(s100) == 100 and len(s200) == 200

    print("[self-test] thesis_closure checks passed "
          "(3-level assembly + order, figure audit, nested-subset plan).")


# ===========================================================================
# CLI
# ===========================================================================

def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=False)

    r = sub.add_parser("reference", help="three-level TDSE/QCLE refinement")
    r.add_argument("--out", type=Path, default=Path("thesis_closure_out"))
    r.add_argument("--P0", type=float, nargs="+", default=[20.0, 100.0])
    r.add_argument("--dt", type=float, default=0.2)
    r.add_argument("--n-steps", type=int, default=200)
    r.add_argument("--modes", nargs="+", default=["both"],
                   choices=["both", "time", "grid"])
    # R0 must match the production packet or the reference is not a valid
    # comparison partner. The -10.0 default reproduces the earlier study.
    r.add_argument("--R0", type=float, default=-10.0)
    r.add_argument("--mass", type=float, default=2000.0)
    r.add_argument("--methods", nargs="+", default=["tdse", "qcle"],
                   choices=["tdse", "qcle"],
                   help="restrict to one solver; grid QCLE is far more "
                        "expensive than TDSE at deep refinement")

    a = sub.add_parser("figure-audit", help="caption/metadata compliance")
    a.add_argument("--roots", type=Path, nargs="+", required=True)
    a.add_argument("--out", type=Path, default=Path("thesis_closure_out"))

    n = sub.add_parser("nested-plan", help="emit a nested support plan")
    n.add_argument("--out", type=Path, default=Path("thesis_closure_out"))
    n.add_argument("--n-max", type=int, default=2000)
    n.add_argument("--levels", type=str, default="500,1000,2000")
    n.add_argument("--seeds", type=str, default="11,29,47")

    c = sub.add_parser("check-nesting", help="report nesting of executed runs")
    c.add_argument("--root", type=Path, required=True)
    c.add_argument("--out", type=Path, default=Path("thesis_closure_out"))

    p.add_argument("--self-test", action="store_true")
    return p


def main() -> None:
    args = _argparser().parse_args()
    if args.self_test:
        run_self_test(); return
    if args.command == "reference":
        run_reference_study(args.out, args.P0, args.dt, args.n_steps,
                            args.modes, R0=args.R0, mass=args.mass,
                            methods=args.methods)
    elif args.command == "figure-audit":
        figure_audit_report(audit_figures(args.roots), args.out)
    elif args.command == "nested-plan":
        write_nested_support_plan(
            args.out, args.n_max,
            tuple(int(x) for x in args.levels.split(",") if x.strip()),
            tuple(int(x) for x in args.seeds.split(",") if x.strip()))
    elif args.command == "check-nesting":
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        rep = executed_support_is_nested(args.root)
        (out / "support_nesting_report.json").write_text(
            json.dumps(rep, indent=2), encoding="utf-8")
        print(f"[check-nesting] nested={rep['nested']} "
              f"({rep['n_runs']} runs) -> {out / 'support_nesting_report.json'}")
    else:
        print("No command given. Use --self-test, or one of: "
              "reference | figure-audit | nested-plan | check-nesting")


if __name__ == "__main__":
    main()
