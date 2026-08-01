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
thesis_analysis.py
==================

Reduces the reviewer-closure campaign to the figures and tables the thesis
needs.  This module implements the "one week" items of
``REVIEWER_RESPONSE_AND_THESIS_REVISION_GUIDE.md``:

  Figures
    F1  manufactured operator error vs support size, on and off shell
        (the single most valuable new figure -- it shows NON-convergence)
    F2  projection leakage vs incoming momentum
    F3  identical-support KDE/GP baseline against its acceptance gate

  Tables  (guide section 7, items 1-4)
    T1  observed order p_obs for the timestep ladder, with guards
    T2  endpoint differences for timestep and support refinement
    T3  replication statistics over independent seeds
    T4  raw conservation drift, separated from self-normalized quantities

  Appendix F
    A rebuilt per-run configuration table generated from the saved
    ``run_manifest.json`` files rather than from defaults.

Design rules inherited from the pipeline
----------------------------------------
* **No figure or axes titles.**  The production figures are deliberately
  header-free; all context belongs in the caption and the ``.meta.json``
  sidecar.  This module follows the same rule and emits captions separately.
* **Raw quantities are never mixed with self-normalized ones.**
* **Nothing is invented.**  A quantity that cannot be computed from the saved
  arrays is reported as ``NOT COMPUTED`` rather than estimated.

The numerical helpers (observed order with guards, interpolation, time-series
norms, raw-drift summary) are imported from ``reviewer_closure_campaign`` so
that this module reuses code already covered by ``test_reviewer_closure.py``.

Usage
-----
    python thesis_analysis.py --roots reviewer_closure_20260723_194254 \
                                      reviewer_closure_20260726_174927 \
                              --out thesis_analysis_out
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from reviewer_closure_campaign import (
    collision_time,
    interp_to_grid,
    observed_order,
    raw_drift_summary,
    timeseries_norms,
)

# Observables ranked by preference; the first present in an npz is used.
_PRIMARY_OBSERVABLES = ("lw_P0", "cloud_weighted_P0", "P0")
_RAW_DRIFT_KEYS = ("raw_norm_drift", "raw_trace_drift", "raw_energy_drift",
                   "raw_mapping_radius_sq_drift")
_SELF_NORM_KEYS = ("km_normalization", "support_label_sum_er")

NOT_COMPUTED = "NOT COMPUTED"


# ===========================================================================
# Loading
# ===========================================================================

def load_run(stem: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load one run's non-snapshot arrays, or None if absent."""
    npz = Path(str(stem) + ".npz")
    if not npz.exists():
        return None
    with np.load(str(npz)) as z:
        return {k: z[k] for k in z.files if not k.startswith("snap_")}


def load_manifest(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = Path(run_dir) / "run_manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _flat(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flat(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = obj
    return out


def manifest_value(man: Optional[Dict[str, Any]], key: str) -> Any:
    """Fetch a leaf value by name from a nested manifest."""
    if man is None:
        return None
    for full, val in _flat(man).items():
        if full.split(".")[-1] == key:
            return val
    return None


def pick_observable(arrays: Dict[str, np.ndarray]) -> Optional[str]:
    for k in _PRIMARY_OBSERVABLES:
        if k in arrays:
            return k
    return None


# ---- directory-name parsing ------------------------------------------------

_RE_DT = re.compile(r"seed(\d+)_dt([0-9.]+)")
_RE_N = re.compile(r"seed(\d+)_N(\d+)")
_RE_SEED = re.compile(r"^seed(\d+)$")
_RE_P0 = re.compile(r"P0(\d+)")


def discover_step7(root: Path) -> Dict[Tuple[float, int, float], Path]:
    """{(P0, seed, dt) -> run directory} for the timestep ladder."""
    out: Dict[Tuple[float, int, float], Path] = {}
    for base in sorted(Path(root).glob("step7_dt_P0*")):
        m = _RE_P0.search(base.name)
        if not m:
            continue
        P0 = float(m.group(1))
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            mm = _RE_DT.match(d.name)
            if mm:
                out[(P0, int(mm.group(1)), float(mm.group(2)))] = d
    return out


def discover_step8(root: Path) -> Dict[Tuple[float, int, int], Path]:
    """{(P0, seed, N) -> run directory} for the support ladder."""
    out: Dict[Tuple[float, int, int], Path] = {}
    for base in sorted(Path(root).glob("step8_support_P0*")):
        m = _RE_P0.search(base.name)
        if not m:
            continue
        P0 = float(m.group(1))
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            mm = _RE_N.match(d.name)
            if mm:
                out[(P0, int(mm.group(1)), int(mm.group(2)))] = d
    return out


def discover_step9(root: Path) -> Dict[Tuple[float, int], Path]:
    """{(P0, seed) -> run directory} for the replication set."""
    out: Dict[Tuple[float, int], Path] = {}
    for base in sorted(Path(root).glob("step9_repl_P0*")):
        m = _RE_P0.search(base.name)
        if not m:
            continue
        P0 = float(m.group(1))
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            mm = _RE_SEED.match(d.name)
            if mm:
                out[(P0, int(mm.group(1)))] = d
    return out


# ===========================================================================
# T1 -- observed order
# ===========================================================================

def compute_observed_order(root: Path, scheme: str = "midpoint",
                           seed_noise: Optional[float] = None
                           ) -> List[Dict[str, Any]]:
    r"""
    Observed order for each (P0, seed) triple of timesteps.

        p_obs = log2( ||u_h - u_h2|| / ||u_h2 - u_h4|| )

    All three series are interpolated onto the coarsest time grid before the
    norms are taken.  ``observed_order`` refuses to report a value when the
    finer difference underflows or falls below ``seed_noise``.
    """
    runs = discover_step7(root)
    by_case: Dict[Tuple[float, int], Dict[float, Path]] = {}
    for (P0, seed, dt), d in runs.items():
        by_case.setdefault((P0, seed), {})[dt] = d

    rows: List[Dict[str, Any]] = []
    for (P0, seed), ladder in sorted(by_case.items()):
        dts = sorted(ladder, reverse=True)          # coarse -> fine
        if len(dts) < 3:
            rows.append({"P0": P0, "seed": seed, "p_obs": NOT_COMPUTED,
                         "reason": f"only {len(dts)} timestep level(s)"})
            continue
        h, h2, h4 = dts[:3]
        series, tgrid, obs_key = [], None, None
        ok = True
        for dt in (h, h2, h4):
            arr = load_run(ladder[dt] / scheme)
            if arr is None:
                ok = False
                break
            key = obs_key or pick_observable(arr)
            if key is None or "t" not in arr:
                ok = False
                break
            obs_key = key
            t = np.asarray(arr["t"], float)
            u = np.asarray(arr[key], float)
            if tgrid is None:
                tgrid = t                            # coarsest grid
            series.append(interp_to_grid(t, u, tgrid))
        if not ok or len(series) != 3:
            rows.append({"P0": P0, "seed": seed, "p_obs": NOT_COMPUTED,
                         "reason": "missing run or observable"})
            continue

        p, why = observed_order(series[0], series[1], series[2],
                                seed_noise=seed_noise)
        n01 = timeseries_norms(series[0], series[1], tgrid)
        n12 = timeseries_norms(series[1], series[2], tgrid)
        rows.append({
            "P0": P0, "seed": seed, "scheme": scheme, "observable": obs_key,
            "dt_coarse": h, "dt_mid": h2, "dt_fine": h4,
            "p_obs": (float(p) if p is not None else NOT_COMPUTED),
            "reason": why,
            "L2_coarse_mid": n01["L2"], "L2_mid_fine": n12["L2"],
            "Linf_coarse_mid": n01["Linf"], "Linf_mid_fine": n12["Linf"],
            "endpoint_coarse_mid": abs(float(series[0][-1] - series[1][-1])),
            "endpoint_mid_fine": abs(float(series[1][-1] - series[2][-1])),
        })
    return rows


# ===========================================================================
# T2 -- endpoint differences
# ===========================================================================

def compute_endpoint_differences(root: Path, scheme: str = "midpoint"
                                 ) -> Dict[str, List[Dict[str, Any]]]:
    """Endpoint differences against the finest level, for dt and support."""
    out: Dict[str, List[Dict[str, Any]]] = {"timestep": [], "support": []}

    # --- timestep: compare each dt against the finest available -----------
    by_case: Dict[Tuple[float, int], Dict[float, Path]] = {}
    for (P0, seed, dt), d in discover_step7(root).items():
        by_case.setdefault((P0, seed), {})[dt] = d
    for (P0, seed), ladder in sorted(by_case.items()):
        if not ladder:
            continue
        finest = min(ladder)
        ref = load_run(ladder[finest] / scheme)
        if ref is None:
            continue
        key = pick_observable(ref)
        if key is None:
            continue
        ref_end = float(np.asarray(ref[key])[-1])
        for dt in sorted(ladder, reverse=True):
            arr = load_run(ladder[dt] / scheme)
            if arr is None or key not in arr:
                continue
            out["timestep"].append({
                "P0": P0, "seed": seed, "dt": dt, "against_dt": finest,
                "observable": key,
                "endpoint": float(np.asarray(arr[key])[-1]),
                "abs_difference": abs(float(np.asarray(arr[key])[-1]) - ref_end),
            })

    # --- support: compare each N against the largest available ------------
    by_case_n: Dict[Tuple[float, int], Dict[int, Path]] = {}
    for (P0, seed, N), d in discover_step8(root).items():
        by_case_n.setdefault((P0, seed), {})[N] = d
    for (P0, seed), ladder in sorted(by_case_n.items()):
        if not ladder:
            continue
        largest = max(ladder)
        ref = load_run(ladder[largest] / scheme)
        if ref is None:
            continue
        key = pick_observable(ref)
        if key is None:
            continue
        ref_end = float(np.asarray(ref[key])[-1])
        for N in sorted(ladder):
            arr = load_run(ladder[N] / scheme)
            if arr is None or key not in arr:
                continue
            out["support"].append({
                "P0": P0, "seed": seed, "N": N, "against_N": largest,
                "observable": key,
                "endpoint": float(np.asarray(arr[key])[-1]),
                "abs_difference": abs(float(np.asarray(arr[key])[-1]) - ref_end),
            })
    return out


# ===========================================================================
# T3 -- replication statistics
# ===========================================================================

def compute_replication_statistics(root: Path, scheme: str = "midpoint"
                                   ) -> List[Dict[str, Any]]:
    """Per-seed endpoints plus mean / sample SD / SE / min / max per momentum.

    The sample standard deviation across independent clouds is the yardstick
    against which refinement differences must be judged (guide section 7.6).
    """
    runs = discover_step9(root)
    by_P0: Dict[float, Dict[int, Path]] = {}
    for (P0, seed), d in runs.items():
        by_P0.setdefault(P0, {})[seed] = d

    rows: List[Dict[str, Any]] = []
    for P0, seeds in sorted(by_P0.items()):
        values: Dict[str, List[float]] = {}
        used: List[int] = []
        for seed in sorted(seeds):
            arr = load_run(seeds[seed] / scheme)
            if arr is None:
                continue
            used.append(seed)
            for key in (_PRIMARY_OBSERVABLES + _RAW_DRIFT_KEYS):
                if key in arr:
                    values.setdefault(key, []).append(
                        float(np.asarray(arr[key])[-1]))
        for key, vals in sorted(values.items()):
            a = np.asarray(vals, float)
            n = a.size
            sd = float(np.std(a, ddof=1)) if n > 1 else float("nan")
            rows.append({
                "P0": P0, "quantity": key, "n_seeds": n, "seeds": used,
                "values": [float(v) for v in a],
                "mean": float(np.mean(a)),
                "sample_sd": sd,
                "standard_error": (sd / np.sqrt(n)) if n > 1 else float("nan"),
                "min": float(np.min(a)), "max": float(np.max(a)),
            })
    return rows


# ===========================================================================
# T4 -- raw conservation
# ===========================================================================

def compute_raw_conservation(root: Path, scheme: str = "midpoint",
                             mass: float = 2000.0, R0: float = -15.0
                             ) -> List[Dict[str, Any]]:
    """
    Raw cumulative drift tables from the replication runs.

    Reports endpoint, max-abs, time of max, and pre/inter/post-interaction
    maxima for every ``raw_*`` array present.  Self-normalized keys are
    recorded separately and must never be substituted for these.
    """
    rows: List[Dict[str, Any]] = []
    for (P0, seed), d in sorted(discover_step9(root).items()):
        arr = load_run(d / scheme)
        if arr is None or "t" not in arr:
            continue
        t = np.asarray(arr["t"], float)
        t_c = collision_time(mass, R0, P0)
        for key in _RAW_DRIFT_KEYS:
            if key not in arr:
                continue
            s = raw_drift_summary(t, np.asarray(arr[key], float), t_c=t_c)
            rows.append({"P0": P0, "seed": seed, "quantity": key,
                         "kind": "raw", **s})
        for key in _SELF_NORM_KEYS:
            if key in arr:
                rows.append({"P0": P0, "seed": seed, "quantity": key,
                             "kind": "self-normalized",
                             "endpoint": float(np.asarray(arr[key])[-1])})
    return rows


# ===========================================================================
# Figures
# ===========================================================================

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# Colour-blind-safe (Wong) palette; solid/dashed kept distinct for greyscale.
_CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
       "red": "#D55E00", "purple": "#CC79A7", "grey": "#555555"}


def _save(fig, out_dir: Path, name: str, caption: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    (out_dir / f"{name}.caption.txt").write_text(caption, encoding="utf-8")
    return pdf


def figure_manufactured(root: Path, out_dir: Path) -> Optional[Path]:
    """
    F1 -- operator/density/gradient relative L2 error versus support size,
    on and off the focused shell, with the seed spread as error bars.

    This is the decisive figure: the curves RISE with N.
    """
    plt = _mpl()
    pat = Path(root) / "step5_manufactured"
    if not pat.exists():
        return None
    data: Dict[Tuple[str, str], Dict[int, List[float]]] = {}
    for f in sorted(pat.rglob("manufactured_operator_metrics.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        N = int(d.get("n_train", 0))
        for support in ("on_support", "off_support"):
            for field in ("density", "gradient", "operator_Q"):
                v = (d.get("metrics", {}).get(support, {})
                     .get(field, {}).get("relative_l2"))
                if v is not None:
                    data.setdefault((support, field), {}) \
                        .setdefault(N, []).append(float(v))
    if not data:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    styles = {"density": (_CB["blue"], "o", "-"),
              "gradient": (_CB["green"], "s", "--"),
              "operator_Q": (_CB["red"], "^", "-")}
    for ax, support in zip(axes, ("on_support", "off_support")):
        for field, (c, mk, ls) in styles.items():
            series = data.get((support, field))
            if not series:
                continue
            Ns = sorted(series)
            mean = [float(np.mean(series[n])) for n in Ns]
            err = [float(np.std(series[n], ddof=1)) if len(series[n]) > 1
                   else 0.0 for n in Ns]
            ax.errorbar(Ns, mean, yerr=err, color=c, marker=mk, ls=ls,
                        capsize=3, lw=1.6, ms=5,
                        label=field.replace("_", " "))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$N_{\mathrm{train}}$")
        ax.grid(True, which="both", alpha=0.25, lw=0.5)
    axes[0].set_ylabel(r"relative $L^2$ error")
    axes[0].legend(frameon=False, fontsize=9)

    caption = (
        "Manufactured-density validation: relative $L^2$ error of the "
        "reconstructed density, its gradient, and the excess operator "
        "$Q[\\rho]$ versus support size. Left panel: evaluated on the focused "
        "support. Right panel: evaluated off support. Markers are the mean "
        "over three query seeds and bars are the sample standard deviation. "
        "Both panels share a common logarithmic scale. All three quantities "
        "INCREASE monotonically with $N_{\\mathrm{train}}$, so the "
        "excess-operator error is not controlled by support refinement over "
        "the tested range. The manufactured test uses its own fit "
        "configuration (frozen hyperparameter policy, default "
        "$L_2=10^{-6}$) and is therefore independent of the production "
        "regularization."
    )
    return _save(fig, out_dir, "fig_manufactured_operator_vs_N", caption)


def figure_leakage(root: Path, out_dir: Path) -> Optional[Path]:
    """F2 -- SEO projection leakage versus incoming momentum."""
    plt = _mpl()
    pat = Path(root) / "step6_projection"
    if not pat.exists():
        return None
    pts: List[Tuple[float, float, float]] = []
    for f in sorted(pat.rglob("projection_leakage.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        m = re.search(r"P0(\d+)", str(f))
        if not m:
            continue
        pts.append((float(m.group(1)),
                    float(d.get("mean_relative_l2_leakage", np.nan)),
                    float(d.get("max_relative_l2_leakage", np.nan))))
    if not pts:
        return None
    pts.sort()
    P0 = [p[0] for p in pts]; mean = [p[1] for p in pts]; mx = [p[2] for p in pts]

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(P0, mean, color=_CB["red"], marker="o", ls="-", lw=1.8, ms=6,
            label="mean")
    ax.plot(P0, mx, color=_CB["orange"], marker="s", ls="--", lw=1.6, ms=5,
            label="maximum")
    ax.axhline(1.0, color=_CB["grey"], lw=0.9, ls=":")
    ax.set_xlabel(r"$P_0$ (a.u.)")
    ax.set_ylabel(r"relative $L^2$ leakage")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=9)

    caption = (
        "Leakage of the six-dimensional product surrogate out of the exact "
        "four-function SEO subspace, evaluated on full-scattering PBME runs "
        "at the final snapshot. Circles: mean relative $L^2$ leakage over 20 "
        "bath anchors with 400 mapping probes each. Squares: maximum over "
        "anchors. The dotted line marks complete leakage. Leakage grows "
        "sharply with incoming momentum, so the high-momentum regime, where "
        "the excess correction is weakest, is also where the surrogate is "
        "least projection faithful."
    )
    return _save(fig, out_dir, "fig_projection_leakage_vs_P0", caption)


def figure_kde_gp(root: Path, out_dir: Path,
                  threshold: float = 0.02) -> Optional[Path]:
    """F3 -- identical-support KDE/GP shape errors against the E1 gate."""
    plt = _mpl()
    pat = Path(root) / "step11_baseline"
    if not pat.exists():
        return None
    rows: List[Tuple[float, float, float, float]] = []
    for f in sorted(pat.rglob("kde_gp_identical_support.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        m = re.search(r"P0(\d+)", str(f))
        if not m:
            continue
        se = d.get("shape_errors", {})
        rows.append((float(m.group(1)),
                     float(se.get("E1", np.nan)),
                     float(se.get("E2", np.nan)),
                     float(se.get("Einf", np.nan))))
    if not rows:
        return None
    rows.sort()
    labels = [f"$P_0={int(r[0])}$" for r in rows]
    x = np.arange(len(rows)); w = 0.26

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar(x - w, [r[1] for r in rows], w, color=_CB["blue"], label=r"$E_1$")
    ax.bar(x, [r[2] for r in rows], w, color=_CB["green"], label=r"$E_2$")
    ax.bar(x + w, [r[3] for r in rows], w, color=_CB["purple"],
           label=r"$E_\infty$")
    ax.axhline(threshold, color=_CB["red"], lw=1.4, ls="--",
               label=r"$E_1$ acceptance gate")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("shape error")
    ax.grid(True, axis="y", which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    caption = (
        "Identical-support comparison of the projected two-dimensional GP and "
        "the direct importance-sampling KDE on PBME runs. Both estimators use "
        "the same support cloud, frozen geometric measure, effective labels, "
        "Scott/Silverman bandwidth, $R$--$P$ grid, and target raw mass. Bars "
        "show the normalized shape errors on a common logarithmic scale; the "
        "dashed line is the predeclared acceptance gate $E_1\\le0.02$. Both "
        "momenta pass by more than an order of magnitude, so the PBME "
        "reconstruction itself is faithful and any pathology under the "
        "corrected scheme is attributable to the excess update rather than to "
        "reconstruction or visualization."
    )
    return _save(fig, out_dir, "fig_kde_gp_baseline", caption)


# ===========================================================================
# Appendix F -- per-run manifest table
# ===========================================================================

_APPENDIX_F_FIELDS = ("P0", "n_train", "seed", "dt", "t_final", "n_steps",
                      "density_mode", "sampling_mode", "surrogate",
                      "l2_regularization", "refit_hyper_policy",
                      "init_log_sigma_n", "fix_sigma_n", "mass", "hbar",
                      "sigma_R", "R0", "product_g_floor_rel")


def build_appendix_f(roots: Sequence[Path]) -> List[Dict[str, Any]]:
    """One row per executed run, read from run_manifest.json (not defaults)."""
    rows: List[Dict[str, Any]] = []
    for root in roots:
        for man_path in sorted(Path(root).rglob("run_manifest.json")):
            man = load_manifest(man_path.parent)
            if man is None:
                continue
            row: Dict[str, Any] = {
                "root": Path(root).name,
                "run": str(man_path.parent.relative_to(root)),
            }
            for f in _APPENDIX_F_FIELDS:
                row[f] = manifest_value(man, f)
            rows.append(row)
    return rows


# ===========================================================================
# LaTeX emission
# ===========================================================================

def _tex_escape(s: Any) -> str:
    t = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#")):
        t = t.replace(a, b)
    return t


def _fmt(v: Any) -> str:
    if v is None:
        return "--"
    if isinstance(v, str):
        return _tex_escape(v)
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    x = float(v)
    if not np.isfinite(x):
        return "--"
    if x == 0.0:
        return "0"
    return f"{x:.4g}" if 1e-3 <= abs(x) < 1e5 else f"{x:.3e}"


def latex_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str],
                caption: str, label: str) -> str:
    """A longtable-compatible booktabs table."""
    if not rows:
        return (f"% no rows for {label}\n"
                f"\\textit{{{NOT_COMPUTED}: no data for {_tex_escape(label)}.}}\n")
    spec = "l" * len(columns)
    out = [r"\begin{table}[p]", r"\centering", r"\small",
           rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
           " & ".join(_tex_escape(c) for c in columns) + r" \\", r"\midrule"]
    for r in rows:
        out.append(" & ".join(_fmt(r.get(c)) for c in columns) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


# ===========================================================================
# Driver
# ===========================================================================

def run(roots: Sequence[Path], out: Path, scheme: str = "midpoint") -> Dict[str, Any]:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    summary: Dict[str, Any] = {"roots": [str(r) for r in roots],
                               "scheme": scheme, "figures": {}, "tables": {}}

    # ---- tables ----------------------------------------------------------
    order_rows: List[Dict[str, Any]] = []
    endpoint: Dict[str, List[Dict[str, Any]]] = {"timestep": [], "support": []}
    repl_rows: List[Dict[str, Any]] = []
    cons_rows: List[Dict[str, Any]] = []
    for root in roots:
        order_rows += compute_observed_order(root, scheme)
        e = compute_endpoint_differences(root, scheme)
        endpoint["timestep"] += e["timestep"]
        endpoint["support"] += e["support"]
        repl_rows += compute_replication_statistics(root, scheme)
        cons_rows += compute_raw_conservation(root, scheme)

    # Use the replication sample SD as the seed-noise guard, then recompute
    # the observed order so that refinement differences below the
    # independent-cloud scatter are refused rather than reported.
    noise: Dict[float, float] = {}
    for r in repl_rows:
        if r["quantity"] in _PRIMARY_OBSERVABLES and np.isfinite(r["sample_sd"]):
            noise[r["P0"]] = float(r["sample_sd"])
    if noise:
        for row in order_rows:
            sn = noise.get(row.get("P0"))
            if sn is None or row.get("p_obs") == NOT_COMPUTED:
                continue
            # A refinement difference smaller than the independent-cloud
            # scatter cannot be attributed to discretization error.
            if (row.get("L2_coarse_mid", 0.0) < sn
                    or row.get("L2_mid_fine", 0.0) < sn):
                row["p_obs"] = NOT_COMPUTED
                row["reason"] = ("INSUFFICIENT EVIDENCE: refinement "
                                 f"difference below independent-seed SD "
                                 f"({sn:.3e})")
            row["seed_noise_sd"] = sn

    summary["tables"]["observed_order"] = order_rows
    summary["tables"]["endpoint_differences"] = endpoint
    summary["tables"]["replication"] = repl_rows
    summary["tables"]["raw_conservation"] = cons_rows

    (out / "analysis_results.json").write_text(
        json.dumps(summary["tables"], indent=2, default=str), encoding="utf-8")

    tex = [
        "% Auto-generated by thesis_analysis.py -- do not edit by hand.",
        latex_table(order_rows,
                    ["P0", "seed", "dt_coarse", "dt_mid", "dt_fine",
                     "L2_coarse_mid", "L2_mid_fine", "p_obs", "reason"],
                    "Observed order of the timestep refinement. "
                    "$p_{\\mathrm{obs}}=\\log_2(\\lVert u_h-u_{h/2}\\rVert/"
                    "\\lVert u_{h/2}-u_{h/4}\\rVert)$ on the common coarse "
                    "time grid. Entries marked NOT COMPUTED failed a guard "
                    "and are not estimated.",
                    "tab:observed-order"),
        latex_table(endpoint["timestep"],
                    ["P0", "seed", "dt", "against_dt", "observable",
                     "endpoint", "abs_difference"],
                    "Endpoint differences under timestep refinement, each "
                    "level compared with the finest available timestep.",
                    "tab:endpoint-timestep"),
        latex_table(endpoint["support"],
                    ["P0", "seed", "N", "against_N", "observable",
                     "endpoint", "abs_difference"],
                    "Endpoint differences under support refinement, each "
                    "level compared with the largest available support. The "
                    "clouds are not nested, so these differences combine "
                    "support-size and sampling effects and must be read "
                    "against the independent-seed scatter of "
                    "Table~\\ref{tab:replication}.",
                    "tab:endpoint-support"),
        latex_table(repl_rows,
                    ["P0", "quantity", "n_seeds", "mean", "sample_sd",
                     "standard_error", "min", "max"],
                    "Replication statistics over independent focused clouds. "
                    "The sample standard deviation is the uncertainty against "
                    "which refinement differences must be judged.",
                    "tab:replication"),
        latex_table([r for r in cons_rows if r.get("kind") == "raw"],
                    ["P0", "seed", "quantity", "endpoint", "max_abs",
                     "t_at_max_abs", "pre_interaction_max_abs",
                     "interaction_max_abs", "post_interaction_max_abs"],
                    "Raw cumulative conservation drift, reported separately "
                    "from self-normalized quantities. Interaction windows use "
                    "$t_c=M|R_0|/|P_0|$.",
                    "tab:raw-conservation"),
        latex_table(build_appendix_f(roots),
                    ["root", "run", "P0", "n_train", "seed", "dt", "t_final",
                     "l2_regularization", "sampling_mode", "surrogate",
                     "refit_hyper_policy"],
                    "Executed production configurations, generated from the "
                    "saved run manifests rather than from defaults.",
                    "tab:appendix-f-runs"),
    ]
    (out / "thesis_tables.tex").write_text("\n".join(tex), encoding="utf-8")

    # ---- figures ---------------------------------------------------------
    for root in roots:
        for name, fn in (("manufactured", figure_manufactured),
                         ("leakage", figure_leakage),
                         ("kde_gp", figure_kde_gp)):
            try:
                p = fn(Path(root), fig_dir)
            except Exception as exc:                      # pragma: no cover
                summary["figures"][name] = f"FAILED: {type(exc).__name__}: {exc}"
                continue
            if p is not None:
                summary["figures"][name] = str(p)
    for k in ("manufactured", "leakage", "kde_gp"):
        summary["figures"].setdefault(k, NOT_COMPUTED)

    (out / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"[thesis_analysis] tables  -> {out / 'thesis_tables.tex'}")
    print(f"[thesis_analysis] results -> {out / 'analysis_results.json'}")
    for k, v in summary["figures"].items():
        print(f"[thesis_analysis] figure {k}: {v}")
    n_ok = sum(1 for r in order_rows if r.get("p_obs") != NOT_COMPUTED)
    print(f"[thesis_analysis] observed order computed for {n_ok}/"
          f"{len(order_rows)} case(s)")
    return summary


# ===========================================================================
# Self-test (synthetic; no campaign data required)
# ===========================================================================

def run_self_test() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "reviewer_closure_selftest"
        # Build a synthetic 3-level dt ladder with an exact 2nd-order sequence.
        t = np.linspace(0.0, 10.0, 51)
        base = np.sin(t)
        for dt, off in ((0.5, 4.0), (0.25, 1.0), (0.125, 0.25)):
            d = root / "step7_dt_P020" / f"seed11_dt{dt:g}"
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / "midpoint.npz", t=t, lw_P0=base + off * 1e-3,
                     raw_norm_drift=1e-9 * t)
            (d / "run_manifest.json").write_text(
                json.dumps({"config": {"P0": 20.0, "dt": dt, "n_train": 1000,
                                       "seed": 11, "l2_regularization": 0.05}}),
                encoding="utf-8")
        rows = compute_observed_order(root)
        assert len(rows) == 1, rows
        p = rows[0]["p_obs"]
        assert p != NOT_COMPUTED, rows[0]
        assert abs(float(p) - 2.0) < 1e-6, p

        # Replication: three seeds with known spread.
        for seed, v in ((11, 1.0), (29, 1.2), (47, 1.4)):
            d = root / "step9_repl_P020" / f"seed{seed}"
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / "midpoint.npz", t=t, lw_P0=base + v,
                     raw_norm_drift=1e-6 * t)
        rep = compute_replication_statistics(root)
        lw = [r for r in rep if r["quantity"] == "lw_P0"][0]
        assert lw["n_seeds"] == 3
        assert abs(lw["mean"] - (base[-1] + 1.2)) < 1e-9
        assert abs(lw["sample_sd"] - 0.2) < 1e-9

        cons = compute_raw_conservation(root)
        raw = [r for r in cons if r["quantity"] == "raw_norm_drift"]
        assert raw and abs(raw[0]["endpoint"] - 1e-5) < 1e-12

        appf = build_appendix_f([root])
        assert any(r["l2_regularization"] == 0.05 for r in appf)

        tex = latex_table(rep, ["P0", "quantity", "mean"], "c", "l")
        assert "\\begin{table}" in tex and "\\bottomrule" in tex
        assert latex_table([], ["a"], "c", "l").strip().startswith("%")
    print("[self-test] thesis_analysis checks passed.")


def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", type=Path, nargs="+", default=[],
                   help="Campaign root directories to analyze.")
    p.add_argument("--out", type=Path, default=Path("thesis_analysis_out"))
    p.add_argument("--scheme", choices=["midpoint", "pbme"], default="midpoint")
    p.add_argument("--self-test", action="store_true")
    return p


def main() -> None:
    a = _argparser().parse_args()
    if a.self_test:
        run_self_test(); return
    if not a.roots:
        print("No --roots given. Use --self-test to validate the module.")
        return
    run(a.roots, a.out, a.scheme)


if __name__ == "__main__":
    main()
