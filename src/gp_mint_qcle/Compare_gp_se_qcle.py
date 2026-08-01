#!/usr/bin/env python3
"""
compare_gp_se_qcle.py
=====================

Four-method comparison on the 1D Tully Dual Avoided Crossing model:

    SE         : exact TDSE on a 1D nuclear grid (split-operator).
    QCLE       : pseudospectral RK4 on a 2D (R, P) phase-space grid.
    PBME       : trajectory-based (loaded from pbme.npz produced by run.py).
    midpoint   : GP-RKHS midpoint integrator (loaded from midpoint.npz).

Inputs
------
A finished GP run directory containing
    ``midpoint.npz`` / ``midpoint.json``  (GP-RKHS-MInt run)
    ``pbme.npz``   / ``pbme.json``        (companion PBME run; optional)

The initial conditions for SE and QCLE are taken from CLI flags (or
defaults that match run.py: R0 = -15, P0 = 40, sigma_R = 1.0).  SE and
QCLE are integrated on the same resolved physical time grid as the GP run.
When a GP run is present, the comparison driver infers dt and T directly from
the saved Collector time array; otherwise it resolves them from --t_final /
--scattering-cycles / --dt just as run.py does.

Observables (same schema across all four methods)
-------------------------------------------------
    P0(t), P1(t)         diabatic populations
    Re(rho_01)(t)        coherence (real)
    Im(rho_01)(t)        coherence (imag)
    |rho_01|(t)          coherence magnitude
    <R>(t), <P>(t)       nuclear means
    Var(R)(t), Var(P)(t) nuclear variances
    Tr rho(t) = P0+P1    trace conservation
    <H>(t)               energy conservation

Outputs
-------
PDF + PNG panels written to <gp_dir>/comparison_se_qcle/:
    panel_populations.{pdf,png}     P0(t), P1(t)         (1x2)
    panel_coherence.{pdf,png}       Re, Im, |coh|        (1x3)
    panel_nuclear.{pdf,png}         <R>, <P>             (1x2)
    panel_variance.{pdf,png}        Var(R), Var(P)       (1x2)
    panel_conservation.{pdf,png}    trace, energy drift  (1x2)
    panel_summary.{pdf,png}         2x3 mega-panel of the above

Usage
-----
    python compare_gp_se_qcle.py runs/focused_with_kde

    python compare_gp_se_qcle.py runs/focused_with_kde --talk

    python compare_gp_se_qcle.py runs/focused_with_kde --no-qcle  # quick
"""
from __future__ import annotations

# --- UTF-8 console safety: prevent UnicodeEncodeError on Windows cp1252 ---
# Banners/diagnostics below print non-ASCII physics notation (α, ρ̂, Δ, →, ħ).
# Reconfigure the console streams to UTF-8 so direct execution of this module
# does not abort under Windows' default cp1252 encoding.  No-op where unsupported.
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
# --------------------------------------------------------------------------

import os, sys, time, json, argparse
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .Models import TullyModel, TullyParams
from .KDEDensity import ProjectedNuclearGP
from .qcle_grid_tully import QCLEGridSolver, QCLEGridParams, QCLEGridState


# ============================================================================
# Publication-quality figure styling
# ----------------------------------------------------------------------------
# Same style block as the other corrected drivers in the pipeline.
# Switch FIG_MODE to "talk" before generating slide figures.
# ============================================================================
# ============================================================================
# JCP / UofT thesis figure constants
# ============================================================================
_W1  = 3.375   # JCP single column [in]
_W15 = 5.0     # JCP 1.5 column
_W2  = 6.75    # JCP double column / UofT full-width

_LABEL_FONT  = 9.0
_TICK_FONT   = 8.0
_LEGEND_FONT = 8.0
_TITLE_FONT  = 9.0

import warnings as _warnings
_warnings.filterwarnings("ignore", message=r".*timestamp seems very low.*",
                         category=UserWarning)


def qcle_boundary_masses(
    rho_total: np.ndarray,
    dR: float,
    dP: float,
    fraction: float = 0.05,
) -> Dict[str, float]:
    """Return robust marginal and phase-space QCLE boundary diagnostics.

    A Wigner density is signed, but its physical position and momentum
    marginals are the quantities whose boundary occupancy determines whether
    the computational box truncates the scattering packet.  We integrate the
    signed Wigner density to each marginal first and then take absolute values
    so small numerical negative lobes cannot cancel across a boundary band.
    The stricter integral of ``abs(W)`` over phase-space boundary strips is
    retained separately as a numerical-ringing diagnostic; it is not a
    probability and therefore is not used as the domain-adequacy gate.
    """
    rho = np.asarray(rho_total, dtype=float)
    if rho.ndim != 2 or min(rho.shape) < 1:
        raise ValueError("rho_total must be a non-empty two-dimensional array")
    if not (np.isfinite(dR) and dR > 0 and np.isfinite(dP) and dP > 0):
        raise ValueError("dR and dP must be finite and positive")
    if not (0.0 < fraction <= 0.5):
        raise ValueError("fraction must lie in (0, 0.5]")

    n_R, n_P = rho.shape
    n_edge_R = max(1, int(np.ceil(fraction * n_R)))
    n_edge_P = max(1, int(np.ceil(fraction * n_P)))
    marginal_R = np.sum(rho, axis=1) * dP
    marginal_P = np.sum(rho, axis=0) * dR

    def absolute_edge_fraction(values: np.ndarray, n_edge: int) -> float:
        magnitude = np.abs(values)
        total = float(np.sum(magnitude))
        edge = float(np.sum(magnitude[:n_edge]) + np.sum(magnitude[-n_edge:]))
        return edge / max(total, 1e-30)

    rho_abs = np.abs(rho)
    phase_total = float(np.sum(rho_abs) * dR * dP)
    phase_R = float(
        (np.sum(rho_abs[:n_edge_R, :]) + np.sum(rho_abs[-n_edge_R:, :]))
        * dR * dP
    ) / max(phase_total, 1e-30)
    phase_P = float(
        (np.sum(rho_abs[:, :n_edge_P]) + np.sum(rho_abs[:, -n_edge_P:]))
        * dR * dP
    ) / max(phase_total, 1e-30)
    return {
        "marginal_R": absolute_edge_fraction(marginal_R, n_edge_R),
        "marginal_P": absolute_edge_fraction(marginal_P, n_edge_P),
        "phase_space_R": phase_R,
        "phase_space_P": phase_P,
    }

# Okabe–Ito colour-blind-safe qualitative palette.  Referenced by
# _apply_rcparams() to build the Matplotlib prop_cycle.  Defined at module
# scope so it exists when _apply_rcparams() runs (its absence was the cause
# of `NameError: name '_PALETTE' is not defined`).
_PALETTE = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "skyblue":    "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
}


# Per-method plot styling (colour / linestyle / linewidth), keyed by the
# canonical method labels.  Referenced by the module-level _curve() helper and
# every line-comparison panel, so they MUST live at module scope (their absence
# was the cause of `NameError: name 'METHOD_COLOR' is not defined`).  Colours
# match the local map previously used only inside _plot_phasespace_with_marginals.
METHOD_COLOR = {
    "SE":       "#333333",   # split-operator TDSE (exact reference)
    "QCLE":     "#009E73",   # grid QCLE (exact quantum-classical reference)
    "PBME":     "#0072B2",   # GP-PBME
    "midpoint": "#D55E00",   # GP midpoint (QCLE-corrected)
}
METHOD_LS = {
    "SE":       "-",
    "QCLE":     "-",
    "PBME":     "-",
    "midpoint": "-",
}
METHOD_LW = {
    "SE":       2.0,
    "QCLE":     1.8,
    "PBME":     1.6,
    "midpoint": 1.9,
}


def _apply_rcparams(mode: str = "journal") -> None:
    base = {
        "axes.linewidth":        0.9,
        "axes.axisbelow":        True,
        "xtick.direction":       "in",   "ytick.direction":       "in",
        "xtick.top":             True,   "ytick.right":           True,
        "xtick.minor.visible":   True,   "ytick.minor.visible":   True,
        "xtick.major.width":     0.9,    "ytick.major.width":     0.9,
        "xtick.minor.width":     0.6,    "ytick.minor.width":     0.6,
        "xtick.major.size":      4.0,    "ytick.major.size":      4.0,
        "xtick.minor.size":      2.2,    "ytick.minor.size":      2.2,
        "legend.frameon":        False,
        "legend.handlelength":   1.8,    "legend.handletextpad":  0.5,
        "legend.columnspacing":  1.2,    "legend.labelspacing":   0.3,
        "lines.linewidth":       1.5,    "lines.markersize":      4.5,
        "lines.markeredgewidth": 0.0,
        "errorbar.capsize":      2.5,
        "mathtext.fontset":      "cm",   "mathtext.default":      "regular",
        "savefig.dpi":           300,    "savefig.bbox":          "tight",
        "savefig.pad_inches":    0.05,
        "figure.dpi":            120,    "figure.facecolor":      "white",
        "axes.prop_cycle": matplotlib.cycler(color=[
            _PALETTE["black"],  _PALETTE["blue"],    _PALETTE["vermillion"],
            _PALETTE["green"],  _PALETTE["orange"],  _PALETTE["purple"],
            _PALETTE["skyblue"], _PALETTE["yellow"]]),
    }
    if mode == "talk":
        base.update({
            "font.family":       "sans-serif",
            "font.sans-serif":   ["Helvetica", "Arial", "Liberation Sans",
                                  "DejaVu Sans"],
            "font.size":         15.0,  "axes.titlesize":    16.0,
            "axes.labelsize":    16.0,  "xtick.labelsize":   13.5,
            "ytick.labelsize":   13.5,  "legend.fontsize":   13.5,
            "figure.titlesize":  17.0,  "lines.linewidth":    2.1,
            "axes.linewidth":     1.1,  "xtick.major.width":  1.1,
            "ytick.major.width":  1.1,  "xtick.major.size":   5.0,
            "ytick.major.size":   5.0,
        })
    else:
        base.update({
            "font.family":       "serif",
            "font.serif":        ["Times New Roman", "Nimbus Roman",
                                  "Liberation Serif", "DejaVu Serif",
                                  "STIXGeneral"],
            "font.size":         11.0,  "axes.titlesize":    12.0,
            "axes.labelsize":    12.0,  "xtick.labelsize":   10.5,
            "ytick.labelsize":   10.5,  "legend.fontsize":   10.5,
            "figure.titlesize":  13.0,
        })
    plt.rcParams.update(base)


def _curve(method: str, **extra) -> dict:
    """Stable ax.plot kwargs across every panel."""
    kw = dict(color=METHOD_COLOR.get(method, "#777777"),
              linestyle=METHOD_LS.get(method, "-"),
              lw=METHOD_LW.get(method, 1.6), label=method)
    kw.update(extra)
    return kw


def _save_pub(fig, stem: str) -> None:
    """Save PDF (vector, font-embedded) + 300-dpi PNG companion."""
    os.makedirs(os.path.dirname(os.path.abspath(stem)) or ".", exist_ok=True)
    kw = dict(dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem + ".pdf", **kw)
    fig.savefig(stem + ".png", **kw)
    try:
        from .Reproducibility import write_figure_metadata
        title = os.path.basename(stem).replace("_", " ") or "Figure"
        write_figure_metadata(
            stem + ".pdf", title=title, data_sources=[],
            scale_policy="shared across compared methods where color/intensity encodes magnitude",
            normalization="stated by axis and legend labels",
        )
    except Exception as exc:
        _warnings.warn(f"Could not write figure metadata sidecar: {exc}")


def _setup_ax(ax, xlabel: str, ylabel: str) -> None:
    """JCP tick / spine / minor-tick styling applied to *ax*."""
    import matplotlib.ticker as _mtic
    ax.set_xlabel(xlabel, fontsize=_LABEL_FONT, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=_LABEL_FONT, labelpad=3)
    ax.tick_params(axis="both", which="major",
                   labelsize=_TICK_FONT, direction="in", length=3.5, width=0.75)
    ax.tick_params(axis="both", which="minor",
                   direction="in", length=1.8, width=0.50)
    if ax.get_xscale() == "linear":
        ax.xaxis.set_minor_locator(_mtic.AutoMinorLocator())
    if ax.get_yscale() == "linear":
        ax.yaxis.set_minor_locator(_mtic.AutoMinorLocator())
    for sp in ax.spines.values():
        sp.set_linewidth(0.75)


def _plot_one_cmp(
    runs: Dict[str, Dict[str, np.ndarray]],
    key: str,
    ylabel: str,
    savepath: str,
    *,
    transform=None,
    hline: Optional[float] = None,
    yscale: str = "linear",
    ylim: Optional[Tuple[float, float]] = None,
    legend_loc: str = "best",
) -> None:
    """Write one JCP-column PDF+PNG for a single observable vs time.

    Parameters
    ----------
    runs       : ``{method_name: obs_dict}``
    key        : key in obs_dict (e.g. ``"P0"``)
    ylabel     : y-axis label string (LaTeX ok)
    savepath   : output path stem (no extension)
    transform  : optional callable ``(t, y, method_name) -> y_new``
    hline      : draw a horizontal reference line at this y-value
    yscale     : ``"linear"`` or ``"log"``
    ylim       : explicit y-limits
    legend_loc : matplotlib legend location string
    """
    fig, ax = plt.subplots(figsize=(_W15, 2.4))
    plotted = False
    for name, r in runs.items():
        if key not in r:
            continue
        t = np.asarray(r["t"], dtype=float)
        y = np.asarray(r[key], dtype=float)
        if transform is not None:
            y = transform(t, y, name)
        n = min(len(t), len(y))
        if n < 2:
            continue
        ax.plot(t[:n], y[:n], **_curve(name))
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    if hline is not None:
        ax.axhline(hline, color="0.45", lw=0.75, ls=":")
    ax.set_yscale(yscale)
    if ylim:
        ax.set_ylim(*ylim)
    _setup_ax(ax, r"$t$ [a.u.]", ylabel)
    ax.legend(fontsize=_LEGEND_FONT, loc=legend_loc, frameon=True,
              framealpha=0.85, edgecolor="0.75")
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    _save_pub(fig, savepath)
    plt.close(fig)


def _qcle_params_for_p0(
    P0: float, R0: float, sigma_R: float,
    n_steps: int, dt: float, mass: float = 2000.0, hbar: float = 1.0,
) -> "QCLEGridParams":
    """Return auto-sized QCLEGridParams appropriate for momentum P0.

    Rules
    -----
    * R-box: covers wavepacket travel plus 30 a.u. margin.
    * P-box: centred on P0 with ±max(8σ_P, 0.5|P0|+15) padding.
    * Grid resolution: dR < 0.15 a.u., dP < 0.15 a.u. or
      σ_P/5 (whichever is stricter).
    * n_R, n_P rounded up to next power of two (needed for FFT efficiency).
    """
    sigma_P = hbar / (2.0 * sigma_R)
    T_total = n_steps * dt

    # R-box
    R_travel = abs(P0) / mass * T_total
    R_pad    = 30.0
    R_min    = R0 - R_pad
    R_max    = R0 + R_travel + R_pad

    # P-box — must cover P0 ± spread, plus room for upper-state kick
    from .Models import TullyParams
    p   = TullyParams.defaults("dual")
    dV  = getattr(p, "A", 0.1) + getattr(p, "C", 0.015)   # rough upper bound
    P_kick = np.sqrt(max(P0**2 + 2.0 * mass * dV, 0.0)) - abs(P0)
    P_spread = max(8.0 * sigma_P, 0.4 * abs(P0) + 10.0) + abs(P_kick) + 10.0
    P_centre = P0
    P_min = P_centre - P_spread
    P_max = P_centre + P_spread

    # Resolution targets
    dR_target = min(0.12, sigma_R / 8.0)
    dP_target = min(0.12, sigma_P / 5.0)

    def _pow2(n: int) -> int:
        return int(2 ** np.ceil(np.log2(max(n, 16))))

    n_R = _pow2(int(np.ceil((R_max - R_min) / dR_target)))
    n_P = _pow2(int(np.ceil((P_max - P_min) / dP_target)))

    # Cap to avoid OOM on very fast packets
    n_R = min(n_R, 4096)
    n_P = min(n_P, 2048)

    return QCLEGridParams(
        R_min=float(R_min), R_max=float(R_max),
        P_min=float(P_min), P_max=float(P_max),
        n_R=n_R, n_P=n_P,
        mass=mass, hbar=hbar,
    )


# Common constants matching run.py / the GP pipeline.
MASS_DEFAULT = 2000.0
HBAR_DEFAULT = 1.0


def _positive_abs_p0(P0: float) -> float:
    pabs = abs(float(P0))
    if not np.isfinite(pabs) or pabs <= 0.0:
        raise ValueError("P0 must be non-zero for scattering time-grid controls.")
    return pabs


def _collision_time(R0: float, P0: float, mass: float) -> float:
    """Return the incoming classical collision time M|R0|/|P0|."""
    return float(mass) * abs(float(R0)) / _positive_abs_p0(P0)


def _resolve_reference_grid_from_args(args, P0: float) -> Tuple[float, int, float]:
    """Resolve dt, n_steps, T when no GP Collector time grid is available."""
    pabs = _positive_abs_p0(P0)
    dt_nominal = float(args.dt)
    if dt_nominal <= 0.0 or not np.isfinite(dt_nominal):
        raise ValueError("dt must be positive.")
    dt_eff = dt_nominal
    if bool(getattr(args, "auto_dt", False)):
        pref = _positive_abs_p0(getattr(args, "auto_dt_ref_p0", 40.0))
        scaled = float(args.auto_dt_ref) * (pabs / pref) ** float(args.auto_dt_power)
        dt_eff = min(dt_nominal, scaled)
        dt_eff = max(float(args.dt_min), min(float(args.dt_max), dt_eff))
    tc = _collision_time(args.R0, P0, args.mass)
    if getattr(args, "t_final", None) is not None:
        T = float(args.t_final)
    elif getattr(args, "scattering_cycles", None) is not None:
        T = float(args.scattering_cycles) * tc
    else:
        # Preserve legacy endpoint if auto_dt lowers dt: --n_steps * requested --dt.
        T = float(args.n_steps) * dt_nominal
    if T <= 0.0 or not np.isfinite(T):
        raise ValueError("resolved final time must be positive.")
    n_steps = max(1, int(np.ceil(T / dt_eff - 1.0e-12)))
    dt_exact = T / float(n_steps)
    return float(dt_exact), int(n_steps), float(T)


def _infer_time_grid_from_runs(runs: Dict[str, Dict[str, np.ndarray]],
                               args, P0: float) -> Tuple[float, int, float, str]:
    """Prefer the saved GP trajectory time grid, otherwise use CLI controls."""
    if not bool(getattr(args, "ignore_gp_time_grid", False)):
        for key in ("midpoint", "PBME"):
            if key in runs and "t" in runs[key]:
                t = np.asarray(runs[key]["t"], dtype=np.float64).reshape(-1)
                t = t[np.isfinite(t)]
                if t.size >= 2:
                    diffs = np.diff(t)
                    diffs = diffs[diffs > 1.0e-14]
                    if diffs.size:
                        dt = float(np.median(diffs))
                        T = float(t[-1])
                        n_steps = int(round(T / dt))
                        if n_steps > 0:
                            dt = T / float(n_steps)
                            return dt, n_steps, T, f"Collector time grid from {key}.npz"
    dt, n_steps, T = _resolve_reference_grid_from_args(args, P0)
    return dt, n_steps, T, "CLI-resolved time grid"


def _parse_density_times(spec: str, *, R0: float, P0: float, mass: float,
                         T_final: float, include_final: bool = False) -> List[float]:
    """Parse density snapshot times. 'auto' means 0, t_c, 2 t_c."""
    text = (spec or "").strip().lower()
    tc = _collision_time(R0, P0, mass)
    if text in ("", "none", "off"):
        vals: List[float] = []
    elif text in ("auto", "collision", "scattering"):
        vals = [0.0, tc, 2.0 * tc]
    elif text in ("final", "t_final", "end"):
        vals = [T_final]
    else:
        vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if include_final:
        vals.append(float(T_final))
    out: List[float] = []
    seen = set()
    for v in vals:
        if not np.isfinite(v):
            continue
        vv = min(max(0.0, float(v)), float(T_final))
        if vv != float(v):
            print(f"[compare] WARNING: requested density-time t={float(v):g} a.u. is "
                  f"outside the run's saved range [0, {float(T_final):g}] a.u. and was "
                  f"dropped (clamped to t={vv:g}). The trajectory does not contain "
                  f"that time — extend n_steps/dt in run.py to reach it.")
        key = round(vv, 10)
        if key in seen:
            continue
        out.append(vv); seen.add(key)
    return sorted(out)


def _nearest_time_index(run: Dict[str, np.ndarray], t_tgt: float) -> int:
    t = np.asarray(run.get("t", []), dtype=np.float64).reshape(-1)
    if t.size == 0:
        return 0
    return int(np.argmin(np.abs(t - t_tgt)))


def _finite_quantile_bounds(values: np.ndarray, qlo: float = 0.001,
                            qhi: float = 0.999) -> Optional[Tuple[float, float]]:
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    if a.size == 1:
        v = float(a[0]); return v, v
    return float(np.quantile(a, qlo)), float(np.quantile(a, qhi))


def _dynamic_rp_window(runs: Dict[str, Dict[str, np.ndarray]],
                       cloud_snaps: Dict[str, List[Dict]],
                       ti: int, t_tgt: float, *,
                       R0: float, P0: float, sigma_R: float, hbar: float,
                       mass: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Data-driven (R,P) plot window enclosing clouds and moment envelopes."""
    sigma_P = hbar / (2.0 * sigma_R)
    R_c = R0 + (P0 / mass) * t_tgt
    R_bounds: List[Tuple[float, float]] = []
    P_bounds: List[Tuple[float, float]] = []
    dR_spread = np.sqrt(sigma_R**2 + (t_tgt * sigma_P / mass) ** 2)
    R_bounds.append((R_c - 4.0 * dR_spread, R_c + 4.0 * dR_spread))
    P_bounds.append((P0 - 5.0 * max(sigma_P, 1.0), P0 + 5.0 * max(sigma_P, 1.0)))
    for r in runs.values():
        if "t" not in r:
            continue
        idx = _nearest_time_index(r, t_tgt)
        for mean_key, var_key, store, floor in (
            ("R_mean", "R_var", R_bounds, sigma_R),
            ("P_mean", "P_var", P_bounds, sigma_P),
        ):
            if mean_key in r and var_key in r:
                mu_arr = np.asarray(r[mean_key], dtype=np.float64).reshape(-1)
                va_arr = np.asarray(r[var_key], dtype=np.float64).reshape(-1)
                if idx < mu_arr.size and idx < va_arr.size:
                    mu = float(mu_arr[idx]); var = float(va_arr[idx])
                    if np.isfinite(mu) and np.isfinite(var):
                        sig = np.sqrt(max(var, 0.0))
                        half = max(5.0 * sig, 3.0 * floor)
                        store.append((mu - half, mu + half))
    for snaps in cloud_snaps.values():
        if ti < len(snaps):
            Z = np.asarray(snaps[ti].get("Z", []), dtype=np.float64)
            if Z.ndim == 2 and Z.shape[1] >= 2 and Z.shape[0] > 0:
                rb = _finite_quantile_bounds(Z[:, 0])
                pb = _finite_quantile_bounds(Z[:, 1])
                if rb is not None: R_bounds.append(rb)
                if pb is not None: P_bounds.append(pb)
    R_lo = min(b[0] for b in R_bounds); R_hi = max(b[1] for b in R_bounds)
    P_lo = min(b[0] for b in P_bounds); P_hi = max(b[1] for b in P_bounds)
    def _pad(lo: float, hi: float, min_half: float, frac: float = 0.08) -> Tuple[float, float]:
        if hi < lo:
            lo, hi = hi, lo
        cen = 0.5 * (lo + hi)
        half = max(0.5 * (hi - lo) * (1.0 + frac), min_half)
        return float(cen - half), float(cen + half)
    return (_pad(R_lo, R_hi, max(6.0 * sigma_R, 2.0)),
            _pad(P_lo, P_hi, max(6.0 * sigma_P, 3.0)))


def _subsample_cloud_for_overlay(Z: np.ndarray, max_points: int = 2500) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[1] < 2 or Z.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    if Z.shape[0] <= max_points:
        return Z[:, :2]
    stride = int(np.ceil(Z.shape[0] / max_points))
    return Z[::stride, :2]


# ============================================================================
# SE driver — TDSE with per-step observables on the SAME time grid as GP
# ============================================================================
def tdse_grid_metadata(
    R0: float,
    P0: float,
    sigma_R: float,
    dt: float,
    n_steps: int,
    *,
    mass: float = MASS_DEFAULT,
    hbar: float = HBAR_DEFAULT,
    model: Optional[TullyModel] = None,
    R_pad: float = 25.0,
    n_grid_min: int = 4096,
) -> Dict[str, float]:
    """Return the exact deterministic grid selected by :func:`run_tdse`."""
    if model is None:
        model = TullyModel(TullyParams.defaults("dual"))
    params = model.params
    velocity = max(P0, 1.0) / mass
    total_time = n_steps * dt
    travel = abs(velocity) * total_time
    R_lo = min(R0 - 6.0 * sigma_R, R0 - R_pad)
    R_upper = max(
        R0 + travel + 6.0 * sigma_R + R_pad,
        R0 + R_pad,
    )
    length = R_upper - R_lo
    maximum_gap = max(params.A + params.C, params.A)
    dynamic_momentum = np.sqrt(P0 ** 2 + 2.0 * mass * maximum_gap)
    sigma_P = hbar / (2.0 * sigma_R)
    required_k_max = (dynamic_momentum + 8.0 * sigma_P) / hbar
    n_grid_actual = max(
        n_grid_min,
        int(2 ** np.ceil(np.log2(2.0 * length * required_k_max / np.pi))),
    )
    n_grid_actual = min(n_grid_actual, 32768)
    dR = length / n_grid_actual
    return {
        "R_min": float(R_lo),
        # The FFT grid excludes the duplicated periodic upper endpoint.
        "R_max": float(R_upper - dR),
        "R_periodic_upper_endpoint": float(R_upper),
        "dR": float(dR),
        "n_grid_actual": int(n_grid_actual),
        "k_max": float(np.pi / dR),
        "k_max_required": float(required_k_max),
    }


def run_tdse(
    R0: float, P0: float, sigma_R: float,
    dt: float, n_steps: int,
    *,
    init_state: int = 0,
    mass: float = MASS_DEFAULT,
    hbar: float = HBAR_DEFAULT,
    model: Optional[TullyModel] = None,
    R_pad: float = 25.0,
    n_grid_min: int = 4096,
    save_every: int = 1,
    t_snapshots: Optional[List[float]] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Split-operator TDSE in the diabatic basis on a 1D R-grid, with
    per-step observables computed on the fly.  The R-grid is sized so
    the wavepacket cannot reach the boundaries within n_steps*dt.

    Returns dict with keys:
        t, P0, P1, coh_re, coh_im, coh_abs,
        R_mean, P_mean, R_var, P_var, trace, energy
    """
    if model is None:
        model = TullyModel(TullyParams.defaults("dual"))
    p = model.params

    # ---------- nuclear R-grid ----------------------------------------
    grid_meta = tdse_grid_metadata(
        R0, P0, sigma_R, dt, n_steps,
        mass=mass, hbar=hbar, model=model, R_pad=R_pad,
        n_grid_min=n_grid_min,
    )
    R_lo = grid_meta["R_min"]
    R_hi = grid_meta["R_periodic_upper_endpoint"]
    Lx = R_hi - R_lo
    nR = int(grid_meta["n_grid_actual"])
    dR = grid_meta["dR"]
    k_max_needed = grid_meta["k_max_required"]
    R = R_lo + np.arange(nR) * dR
    k = 2.0 * np.pi * np.fft.fftfreq(nR, d=dR)
    if verbose:
        print(f"  [SE]   N={nR}, dR={dR:.5f},  R∈[{R_lo:.1f},{R_hi:.1f}],  "
              f"k_max={np.pi/dR:.1f}  (need {k_max_needed:.1f})")

    # ---------- diabatic potentials and operators ---------------------
    H_d = model.diabatic_potential(R)                # (N, 2, 2)
    V11 = H_d[..., 0, 0]; V22 = H_d[..., 1, 1]; V12 = H_d[..., 0, 1]
    # Diabatic eigenvalues/eigenvectors at each R for the diagonal V step
    E_ad = model.adiabatic_energies(R)                # (N, 2)
    U_ad = model.adiabatic_states(R)                  # (N, 2, 2), columns = eigenvecs

    ph0 = np.exp(-1j * E_ad[:, 0] * dt / (2.0 * hbar))
    ph1 = np.exp(-1j * E_ad[:, 1] * dt / (2.0 * hbar))
    Tprop = np.exp(-1j * hbar * k ** 2 * dt / (2.0 * mass))

    # ---------- initial wavefunction ---------------------------------
    # |Ψ⟩ = |init_state⟩_diab ⊗ |Gaussian(R0,P0,σ_R)⟩
    gauss = ((2.0 * np.pi * sigma_R ** 2) ** -0.25
             * np.exp(-(R - R0) ** 2 / (4.0 * sigma_R ** 2)
                      + 1j * P0 * R / hbar))
    psi = np.zeros((2, nR), dtype=complex)
    psi[init_state] = gauss

    # ---------- step ---------------------------------------------------
    def vstep(p_in):
        """One split-operator step:  V/2 → T → V/2 (diagonalised in ad-basis)."""
        p0d, p1d = p_in[0], p_in[1]
        # diab -> ad
        a0 = U_ad[:, 0, 0] * p0d + U_ad[:, 1, 0] * p1d
        a1 = U_ad[:, 0, 1] * p0d + U_ad[:, 1, 1] * p1d
        a0 *= ph0; a1 *= ph1
        # ad -> diab
        p0d = U_ad[:, 0, 0] * a0 + U_ad[:, 0, 1] * a1
        p1d = U_ad[:, 1, 0] * a0 + U_ad[:, 1, 1] * a1
        # kinetic (FFT)
        p0d = np.fft.ifft(np.fft.fft(p0d) * Tprop)
        p1d = np.fft.ifft(np.fft.fft(p1d) * Tprop)
        # diab -> ad
        a0 = U_ad[:, 0, 0] * p0d + U_ad[:, 1, 0] * p1d
        a1 = U_ad[:, 0, 1] * p0d + U_ad[:, 1, 1] * p1d
        a0 *= ph0; a1 *= ph1
        # ad -> diab
        out = np.empty_like(p_in)
        out[0] = U_ad[:, 0, 0] * a0 + U_ad[:, 0, 1] * a1
        out[1] = U_ad[:, 1, 0] * a0 + U_ad[:, 1, 1] * a1
        return out

    # ---------- observables --------------------------------------------
    n_save = (n_steps // save_every) + 1
    t_out      = np.zeros(n_save)
    P0_t       = np.zeros(n_save)
    P1_t       = np.zeros(n_save)
    coh_re_t   = np.zeros(n_save)
    coh_im_t   = np.zeros(n_save)
    R_mean_t   = np.zeros(n_save)
    P_mean_t   = np.zeros(n_save)
    R_var_t    = np.zeros(n_save)
    P_var_t    = np.zeros(n_save)
    energy_t   = np.zeros(n_save)
    trace_t    = np.zeros(n_save)
    edge_mass_t = np.zeros(n_save)
    negative_momentum_probability_t = np.zeros(n_save)

    def measure(idx, t_now, psi):
        """One-pass observables computation."""
        rho00 = np.abs(psi[0]) ** 2
        rho11 = np.abs(psi[1]) ** 2
        rho_total = rho00 + rho11
        P_a = float(np.sum(rho00) * dR)
        P_b = float(np.sum(rho11) * dR)
        Z = P_a + P_b
        # off-diagonal: ρ_01 = psi_0 conj(psi_1).  Sum is complex; do NOT
        # cast to float here — that silently drops Im(ρ_01).
        rho01 = psi[0] * np.conj(psi[1])
        rho01_int = np.sum(rho01) * dR
        coh_re = float(np.real(rho01_int))
        coh_im = float(np.imag(rho01_int))
        # nuclear means (R-space)
        R_mean = float(np.sum(R * rho_total) * dR) / max(Z, 1e-30)
        R2     = float(np.sum(R * R * rho_total) * dR) / max(Z, 1e-30)
        R_var  = max(R2 - R_mean ** 2, 0.0)
        # momentum means (FFT-based)
        phi0 = np.fft.fft(psi[0]) * dR / np.sqrt(2.0 * np.pi)
        phi1 = np.fft.fft(psi[1]) * dR / np.sqrt(2.0 * np.pi)
        rho00_k = np.abs(phi0) ** 2
        rho11_k = np.abs(phi1) ** 2
        rho_k = rho00_k + rho11_k
        P_axis = hbar * k                        # k = 2π·fftfreq, so P = ℏk
        dk = 2.0 * np.pi / Lx
        P_mean = float(np.sum(P_axis * rho_k) * dk) / max(Z, 1e-30)
        P2     = float(np.sum(P_axis * P_axis * rho_k) * dk) / max(Z, 1e-30)
        P_var  = max(P2 - P_mean ** 2, 0.0)
        negative_momentum_probability = float(
            np.sum(rho_k[P_axis < 0.0]) * dk
        ) / max(Z, 1e-30)
        # diabatic-frame energy: <H> = <T> + <V>
        # <T> = ∫ Σ_λ |∂_R psi_λ|² / (2m) dR  via spectral form ℏ²k²/2m
        ekin = float(np.sum(rho_k * (hbar * k) ** 2) * dk) / (2.0 * mass)
        epot = float(
            np.sum(V11 * rho00 + V22 * rho11 + 2.0 * V12 * np.real(psi[0] * np.conj(psi[1])))
        ) * dR
        E_tot = ekin + epot
        n_edge = max(1, int(np.ceil(0.05 * nR)))
        edge_mass = float(
            (np.sum(rho_total[:n_edge]) + np.sum(rho_total[-n_edge:])) * dR
        ) / max(Z, 1e-30)

        t_out[idx]    = t_now
        P0_t[idx]     = P_a
        P1_t[idx]     = P_b
        trace_t[idx]  = Z
        coh_re_t[idx] = coh_re
        coh_im_t[idx] = coh_im
        R_mean_t[idx] = R_mean
        P_mean_t[idx] = P_mean
        R_var_t[idx]  = R_var
        P_var_t[idx]  = P_var
        energy_t[idx] = E_tot
        edge_mass_t[idx] = edge_mass
        negative_momentum_probability_t[idx] = negative_momentum_probability

    # ----- snapshot recording -----
    # If t_snapshots is given, the loop also stores psi(R) at the closest
    # integer step to each requested snapshot time.
    snap_steps: List[int] = []
    snap_t_targets: List[float] = []
    snap_psi: List[np.ndarray] = []
    snap_t: List[float] = []
    if t_snapshots is not None and len(t_snapshots) > 0:
        snap_steps = sorted({max(0, min(n_steps, int(round(ts / dt))))
                             for ts in t_snapshots})
        snap_t_targets = [s * dt for s in snap_steps]

    # initial measurement
    measure(0, 0.0, psi)
    idx = 1
    if snap_steps and snap_steps[0] == 0:
        snap_psi.append(psi.copy())
        snap_t.append(0.0)
    t0 = time.perf_counter()
    print_every = max(1, n_steps // 10)
    for s in range(1, n_steps + 1):
        psi = vstep(psi)
        if (s % save_every == 0) or (s == n_steps):
            if idx < n_save:
                measure(idx, s * dt, psi)
                idx += 1
        if s in snap_steps:
            snap_psi.append(psi.copy())
            snap_t.append(s * dt)
        if verbose and s % print_every == 0:
            P_a = float(np.sum(np.abs(psi[0]) ** 2) * dR)
            P_b = float(np.sum(np.abs(psi[1]) ** 2) * dR)
            print(f"  [SE]   step {s:6d}  t={s*dt:7.1f}  "
                  f"P0={P_a:.4f}  P1={P_b:.4f}  trace={P_a+P_b:.6f}")
    if verbose:
        print(f"  [SE]   done {time.perf_counter()-t0:.1f}s,  saved {idx}/{n_save} pts"
              + (f",  {len(snap_psi)} snapshots" if snap_psi else ""))

    # truncate if save_every did not divide n_steps exactly
    out = dict(
        t=t_out[:idx],         P0=P0_t[:idx],          P1=P1_t[:idx],
        coh_re=coh_re_t[:idx], coh_im=coh_im_t[:idx],
        coh_abs=np.sqrt(coh_re_t[:idx] ** 2 + coh_im_t[:idx] ** 2),
        R_mean=R_mean_t[:idx], P_mean=P_mean_t[:idx],
        R_var=R_var_t[:idx],   P_var=P_var_t[:idx],
        trace=trace_t[:idx],   energy=energy_t[:idx],
        edge_mass_5pct=edge_mass_t[:idx],
        negative_momentum_probability=negative_momentum_probability_t[:idx],
        # snapshot bundle for marginal-density panels
        snap_R=R, snap_dR=dR, snap_psi=snap_psi, snap_t=np.asarray(snap_t),
        snap_k=k, snap_Lx=Lx, snap_hbar=hbar,
    )
    return out


# ============================================================================
# QCLE driver — pseudospectral RK4 with per-step observables
# ============================================================================
def run_qcle(
    R0: float, P0: float, sigma_R: float,
    dt: float, n_steps: int,
    *,
    init_state: int = 0,
    mass: float = MASS_DEFAULT,
    hbar: float = HBAR_DEFAULT,
    save_every: int = 1,
    t_snapshots: Optional[List[float]] = None,
    qcle_params: Optional["QCLEGridParams"] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Drive QCLEGridSolver and accumulate observables per step.

    If ``qcle_params`` is provided (e.g. from :func:`_qcle_params_for_p0`) it is
    used verbatim — the caller controls the box and resolution.  When it is None
    the grid box is auto-sized internally so the wave packet never reaches the
    boundaries within n_steps*dt.
    """
    sigma_P = hbar / (2.0 * sigma_R)

    if qcle_params is not None:
        # Caller supplied an explicit grid (see _qcle_params_for_p0): use it
        # verbatim so the reference box / resolution is exactly as requested,
        # rather than the (capped) internal auto-sizer below.
        params = qcle_params
        mass = float(params.mass)
        hbar = float(params.hbar)
        sigma_P = hbar / (2.0 * sigma_R)
        if verbose:
            print(f"  [QCLE] grid {params.n_R}x{params.n_P}  "
                  f"R∈[{params.R_min:.1f},{params.R_max:.1f}]  "
                  f"P∈[{params.P_min:.1f},{params.P_max:.1f}]  (caller-supplied)")
    else:
        v0 = max(P0, 1.0) / mass
        t_total = n_steps * dt
        travel = abs(v0) * t_total

        # spatial box: cover initial Gaussian and full travel + 6σ pad on each side
        R_lo = min(R0 - 8.0 * sigma_R, R0 - 10.0)
        R_hi = max(R0 + travel + 8.0 * sigma_R + 10.0, R0 + 25.0)
        # momentum box: cover initial Gaussian and upper-state kick
        dV_max = 0.10 + 0.015     # max(A, A+C) for DAC defaults
        P_dyn = np.sqrt(P0 ** 2 + 2.0 * mass * dV_max)
        P_lo = -P_dyn - 8.0 * sigma_P - 5.0
        P_hi =  P_dyn + 8.0 * sigma_P + 5.0
        # resolution: sigma resolved by ~3.5 cells in each direction (a good
        # spectral-method default — finer than this is wasted work since the
        # error converges spectrally).
        nR = int(2 ** np.ceil(np.log2(max(256.0, (R_hi - R_lo) / (sigma_R / 3.5)))))
        nP = int(2 ** np.ceil(np.log2(max(192.0, (P_hi - P_lo) / (sigma_P / 3.5)))))
        nR = min(nR, 768); nP = min(nP, 384)
        if verbose:
            print(f"  [QCLE] grid {nR}x{nP}  R∈[{R_lo:.1f},{R_hi:.1f}]  "
                  f"P∈[{P_lo:.1f},{P_hi:.1f}]")

        params = QCLEGridParams(
            R_min=R_lo, R_max=R_hi, P_min=P_lo, P_max=P_hi,
            n_R=nR, n_P=nP, mass=mass, hbar=hbar,
        )
    solver = QCLEGridSolver(
        model=TullyModel(TullyParams.defaults("dual")),
        params=params,
    )
    state = solver.initial_diabat_gaussian(
        R0=R0, P0=P0, sigma_R=sigma_R, init_state=init_state,
    )

    # Pre-compute V11(R), V22(R), V12(R) on R-grid for energy expectation
    Rmesh, Pmesh = solver.meshgrid()
    Rg = Rmesh[:, 0]
    V11 = solver.model.V11(Rg)
    V22 = solver.model.V22(Rg)
    V12 = solver.model.V12(Rg)
    cA  = solver.cell_area

    # output arrays
    n_save = (n_steps // save_every) + 1
    t_out     = np.zeros(n_save)
    P0_t      = np.zeros(n_save)
    P1_t      = np.zeros(n_save)
    coh_re_t  = np.zeros(n_save)
    coh_im_t  = np.zeros(n_save)
    R_mean_t  = np.zeros(n_save)
    P_mean_t  = np.zeros(n_save)
    R_var_t   = np.zeros(n_save)
    P_var_t   = np.zeros(n_save)
    energy_t  = np.zeros(n_save)
    trace_t   = np.zeros(n_save)
    edge_R_mass_t = np.zeros(n_save)
    edge_P_mass_t = np.zeros(n_save)
    edge_phase_space_R_mass_t = np.zeros(n_save)
    edge_phase_space_P_mass_t = np.zeros(n_save)

    def measure(idx, t_now, st):
        A, C, bR_, bI_ = st.A, st.C, st.bR, st.bI
        rho_tot = A + C
        P_a = float(np.sum(A) * cA)
        P_b = float(np.sum(C) * cA)
        Z = P_a + P_b
        coh_re = float(np.sum(bR_) * cA)
        coh_im = float(np.sum(bI_) * cA)
        # nuclear marginals
        R_mean = float(np.sum(Rmesh * rho_tot) * cA) / max(Z, 1e-30)
        R2     = float(np.sum(Rmesh * Rmesh * rho_tot) * cA) / max(Z, 1e-30)
        R_var  = max(R2 - R_mean ** 2, 0.0)
        P_mean = float(np.sum(Pmesh * rho_tot) * cA) / max(Z, 1e-30)
        P2     = float(np.sum(Pmesh * Pmesh * rho_tot) * cA) / max(Z, 1e-30)
        P_var  = max(P2 - P_mean ** 2, 0.0)
        # energy: <T> + <V>; <V> = ∫(V11 A + V22 C + 2 V12 bR) dR dP
        ekin = float(np.sum((Pmesh ** 2) * rho_tot) * cA) / (2.0 * mass)
        epot = float(np.sum(V11[:, None] * A
                            + V22[:, None] * C
                            + 2.0 * V12[:, None] * bR_) * cA)
        boundary = qcle_boundary_masses(rho_tot, solver.dR, solver.dP)
        edge_R_mass = boundary["marginal_R"]
        edge_P_mass = boundary["marginal_P"]
        t_out[idx]    = t_now
        P0_t[idx]     = P_a
        P1_t[idx]     = P_b
        trace_t[idx]  = Z
        coh_re_t[idx] = coh_re
        coh_im_t[idx] = coh_im
        R_mean_t[idx] = R_mean
        P_mean_t[idx] = P_mean
        R_var_t[idx]  = R_var
        P_var_t[idx]  = P_var
        energy_t[idx] = ekin + epot
        edge_R_mass_t[idx] = edge_R_mass
        edge_P_mass_t[idx] = edge_P_mass
        edge_phase_space_R_mass_t[idx] = boundary["phase_space_R"]
        edge_phase_space_P_mass_t[idx] = boundary["phase_space_P"]

    # initial
    # ----- snapshot recording -----
    snap_steps: List[int] = []
    snap_states: List[QCLEGridState] = []
    snap_t: List[float] = []
    if t_snapshots is not None and len(t_snapshots) > 0:
        snap_steps = sorted({max(0, min(n_steps, int(round(ts / dt))))
                             for ts in t_snapshots})

    measure(0, 0.0, state)
    idx = 1
    if snap_steps and snap_steps[0] == 0:
        snap_states.append(state.copy()); snap_t.append(0.0)
    t0 = time.perf_counter()
    print_every = max(1, n_steps // 10)
    for s in range(1, n_steps + 1):
        state = solver.step(state, dt)
        if (s % save_every == 0) or (s == n_steps):
            if idx < n_save:
                measure(idx, s * dt, state)
                idx += 1
        if s in snap_steps:
            snap_states.append(state.copy()); snap_t.append(s * dt)
        if verbose and s % print_every == 0:
            P_a, P_b = solver.populations(state)
            print(f"  [QCLE] step {s:6d}  t={s*dt:7.1f}  "
                  f"P0={P_a:.4f}  P1={P_b:.4f}  trace={P_a+P_b:.6f}")
    if verbose:
        print(f"  [QCLE] done {time.perf_counter()-t0:.1f}s,  saved {idx}/{n_save} pts"
              + (f",  {len(snap_states)} snapshots" if snap_states else ""))

    out = dict(
        t=t_out[:idx],         P0=P0_t[:idx],          P1=P1_t[:idx],
        coh_re=coh_re_t[:idx], coh_im=coh_im_t[:idx],
        coh_abs=np.sqrt(coh_re_t[:idx] ** 2 + coh_im_t[:idx] ** 2),
        R_mean=R_mean_t[:idx], P_mean=P_mean_t[:idx],
        R_var=R_var_t[:idx],   P_var=P_var_t[:idx],
        trace=trace_t[:idx],   energy=energy_t[:idx],
        edge_R_mass_5pct=edge_R_mass_t[:idx],
        edge_P_mass_5pct=edge_P_mass_t[:idx],
        edge_phase_space_R_mass_5pct=edge_phase_space_R_mass_t[:idx],
        edge_phase_space_P_mass_5pct=edge_phase_space_P_mass_t[:idx],
        cfl_dt_max=np.asarray([solver.cfl_dt_max()["min"]], dtype=float),
        # snapshot bundle for marginal-density panels
        snap_Rg=Rg, snap_dR=solver.dR, snap_dP=solver.dP,
        snap_R_axis=Rmesh[:, 0], snap_P_axis=Pmesh[0, :],
        snap_states=snap_states, snap_t=np.asarray(snap_t),
    )
    return out


# ============================================================================
# Trajectory-method observables — read directly from any Collector NPZ
# ============================================================================
def load_collector_run(npz_path: str) -> Dict[str, np.ndarray]:
    """
    Extract the time-series observables produced by run.py (Collector output,
    e.g. midpoint.npz or pbme.npz) and put them on the same key schema as
    run_tdse / run_qcle output, so the panel code can iterate over methods
    uniformly.

    Both the midpoint (GP-RKHS-MInt) and PBME schemes carry the same
    ``lw_*`` label-weighted IS estimator and the same nuclear-moment /
    energy keys, so a single loader handles both.

    In addition to the legacy observables, the loader also forwards all
    new per-step diagnostic columns introduced by the corrected midpoint
    scheme — ``fc_*`` (flow correction), ``label_*`` and
    ``omega_A_residual_norm`` (label integrator), and ``faith_*``
    (surrogate-faithfulness battery) — so panels can read them directly.
    Missing keys are silently skipped (they don't exist in PBME runs, for
    example).
    """
    # This loader reads ONLY the per-step time-series members (t, lw_*,
    # cloud_weighted_*, nm_*, faith_*, fc_*, label_*, …).  The periodic
    # ``snap_*`` snapshot members — positions, labels, GP coefficients at
    # every saved step — are never touched here, yet ``dict(np.load(...))``
    # used to decompress and materialise all of them at once, which on a long,
    # finely-sampled run is hundreds of MB and the direct cause of the
    # MemoryError during comparison-figure generation.  Reading only the
    # non-snapshot members from the open NpzFile keeps this cheap.
    with np.load(npz_path) as z:
        d = {k: z[k] for k in z.files if not k.startswith("snap_")}

    def _prefer(*keys: str) -> np.ndarray:
        """Return the first present key's array as float64; raise if none exist.

        Closes over the materialised `d` above so callers can list a
        preference order of schema-equivalent keys (e.g. the self-normalized
        ``lw_*`` estimator first, then the raw ``cloud_weighted_*`` sum, then
        the GP-analytic fallback).
        """
        for key in keys:
            if key in d:
                return np.asarray(d[key], dtype=np.float64)
        raise KeyError(f"None of the requested keys exist in {npz_path}: {keys}")

    # Use the self-normalized label-weighted cloud estimator as the physical
    # primary for nuclear moments.  The analytic nm_* values include the
    # GP kernel bandwidth contribution (e.g. +ell_P^2 in <P^2>) and are useful
    # diagnostics, but they are not the support-cloud physical momentum
    # estimator used for populations/energy.
    coh_re_arr = _prefer("lw_coh_re", "cloud_weighted_coh_re", "dc_coh_re")
    coh_im_arr = _prefer("lw_coh_im", "cloud_weighted_coh_im", "dc_coh_im")
    out = dict(
        t=d["t"],
        P0=_prefer("lw_P0", "cloud_weighted_P0", "dp_P0"),
        P1=_prefer("lw_P1", "cloud_weighted_P1", "dp_P1"),
        coh_re=coh_re_arr,
        coh_im=coh_im_arr,
        coh_abs=np.sqrt(coh_re_arr ** 2 + coh_im_arr ** 2),
        R_mean=_prefer("lw_R_mean", "cloud_weighted_R_mean", "nm_R_mean"),
        P_mean=_prefer("lw_P_mean", "cloud_weighted_P_mean", "nm_P_mean"),
        R_var=_prefer("lw_R_var", "cloud_weighted_R_var", "nm_R_var"),
        P_var=_prefer("lw_P_var", "cloud_weighted_P_var", "nm_P_var"),
        trace=_prefer("lw_P_sum", "lw_trace", "cloud_weighted_trace", "km_trace"),
        energy=_prefer("H_expectation", "physical_energy", "lw_energy", "spe_E_density_sn", "km_energy"),
    )
    # Pass through every new diagnostic the panels know how to plot.
    diag_keys = (
        # Flow-correction diagnostics (Step 2.5 of MidpointScheme.step)
        "fc_dz_max", "fc_dz_rms", "fc_n_capped",
        "fc_grad_min", "fc_grad_median", "fc_applied",
        # Label-integrator diagnostics (Step 5)
        "label_scheme_id", "omega_A_residual_norm",
        "label_dy_max",   "label_dy_rms",   "label_probability_drift",
        # Surrogate-faithfulness battery (per-step health check)
        "faith_N",
        "faith_ess_alpha",        "faith_ess_alpha_frac",
        "faith_ess_alpha_naive",  "faith_alpha_sign_align",
        "faith_ess_wy",           "faith_ess_wy_frac",
        "faith_wy_sign_align",
        "faith_loo_rms",          "faith_loo_max",     "faith_loo_med",
        "faith_loo_std_max",      "faith_loo_n_3sig",
        "faith_cond_K_lo",        "faith_cond_K_lo_log10",
        "faith_predict_rms",      "faith_predict_max", "faith_predict_rms_rel",
    )
    for k in diag_keys:
        if k in d:
            out[k] = d[k]
    return out


# ============================================================================
# Comparison plotting — every panel = the same overlay structure
# ============================================================================
def _plot_overlay(ax, runs: Dict[str, Dict[str, np.ndarray]], key: str,
                  ylabel: str, title: Optional[str] = None,
                  hline: Optional[float] = None) -> None:
    del title
    for name, r in runs.items():
        ax.plot(r["t"], r[key], **_curve(name))
    ax.set_xlabel(r"Time  [a.u.]")
    ax.set_ylabel(ylabel)
    ax.set_title("")
    if hline is not None:
        ax.axhline(hline, color="0.6", lw=0.5, ls="-")


# ============================================================================
# R/P marginal density extraction — same schema for all four methods
# ============================================================================
def _marginals_from_se(snap_psi: np.ndarray,
                       R_axis: np.ndarray,
                       k_axis: np.ndarray,
                       Lx: float,
                       hbar: float,
                       P_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    From psi[2, nR] return (rho_R(R), rho_P(P)) — total diabatic-trace
    marginals on the requested P_grid (interpolated from the FFT axis).
    """
    dR = float(R_axis[1] - R_axis[0])
    rho_R = np.abs(snap_psi[0]) ** 2 + np.abs(snap_psi[1]) ** 2     # (nR,)
    # P-space densities via FFT
    phi0 = np.fft.fft(snap_psi[0]) * dR / np.sqrt(2.0 * np.pi)
    phi1 = np.fft.fft(snap_psi[1]) * dR / np.sqrt(2.0 * np.pi)
    P_fft = hbar * k_axis                                  # ℏ·k
    rho_P_fft = np.abs(phi0) ** 2 + np.abs(phi1) ** 2
    # Reorder FFT axis to monotonic, then interpolate onto requested P_grid
    order = np.argsort(P_fft)
    P_sorted = P_fft[order]; rho_P_sorted = rho_P_fft[order]
    rho_P = np.interp(P_grid, P_sorted, rho_P_sorted, left=0.0, right=0.0)
    return rho_R, rho_P


def _marginals_from_qcle(state: QCLEGridState,
                          R_axis: np.ndarray,
                          P_axis: np.ndarray,
                          dR: float, dP: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rho_R(R), rho_P(P)) from a QCLE phase-space state."""
    rho_full = state.A + state.C                # diabatic trace, shape (nR, nP)
    rho_R = np.sum(rho_full, axis=1) * dP
    rho_P = np.sum(rho_full, axis=0) * dR
    return rho_R, rho_P


def _bath_marginals_1d(snap: Dict[str, np.ndarray],
                       R_grid: np.ndarray,
                       P_grid: np.ndarray,
                       hbar: float = 1.0,
                       ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Analytic 1D bath-density marginals for the trajectory-based methods.

    These are the correct objects to compare against SE and QCLE.

    Mathematical content
    --------------------
    The partial-Wigner diagonal trace (the quantity SE and QCLE propagate) is

        rho_bath(R, P) = Sum_lambda rho_W^{lambda lambda}(R, P)
                       = (8 / hbar) * Int dr dp  exp(-x^2 / hbar) * (x^2 - hbar)
                                               * rho_hat(R, P, r, p)

    where x^2 = r0^2 + r1^2 + p0^2 + p1^2.  Against the ARD-RBF GP surrogate
    the 4D mapping integral has a closed form (polynomial-times-Gaussian moments;
    see _density_2d_from_gp_bath_marginal for the 2D case).

    The 1D marginals are:

        rho_bath^R(R) = Int dP rho_bath(R, P)
                      = (8 sigma_f^2 / hbar) * sqrt(2 pi) ell_P
                        * Sum_j  w_j * exp(-0.5 * (R - R_j)^2 / ell_R^2)

        rho_bath^P(P) = Int dR rho_bath(R, P)
                      = (8 sigma_f^2 / hbar) * sqrt(2 pi) ell_R
                        * Sum_j  w_j * exp(-0.5 * (P - P_j)^2 / ell_P^2)

    where the per-centre bath weight is (same as in _density_2d_from_gp_bath_marginal):

        a_d  = 1/hbar + 1/(2 ell_d^2)                for d in {r0, r1, p0, p1}
        mu_d = x_{j,d} * (1 / (2 ell_d^2)) / a_d
        C_d  = -x_{j,d}^2 / (2 ell_d^2) * (1 - (1/(2 ell_d^2)) / a_d)
        log_Pj = Sum_d [C_d + 0.5 * log(pi / a_d)]
        M_j    = Sum_d [mu_d^2 + 1 / (2 a_d)] - hbar
        w_j    = alpha_j * M_j * exp(log_Pj)

    Returns (rho_R, rho_P) or (None, None) if the snapshot lacks GP parameters.
    Falls back to the flat marginal (same shape for focused mode; different
    for seo_signed) when the bath integral yields all-zero or NaN, so the
    panel at least shows something plausible with an annotation.
    """
    # Preferred path for every modern snapshot: project the saved physical
    # cloud measure first, then integrate the common-support 2D GP.  This is
    # identifiable for focused PBME and uses the same object as the KDE
    # baseline.  The analytic 6D branch below remains only for legacy files
    # that lack the saved labels or measure.
    rho_2d = _density_2d_from_gp_moment_projection(snap, R_grid, P_grid)
    if rho_2d is not None:
        # Conditional expression is intentionally lazy: NumPy >=2.4 removes
        # ``np.trapz``, so passing it as an eager ``getattr`` default fails
        # even when ``np.trapezoid`` exists.
        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        rho_R = trap(rho_2d, P_grid, axis=1)
        rho_P = trap(rho_2d, R_grid, axis=0)
        return rho_R, rho_P

    def _component(alpha, Z_centers, ell, sigma_f, feature_zscore, feat_std):
        """(rho_R, rho_P) contribution of one kernel set, or (None, None)."""
        if any(v is None for v in (alpha, Z_centers, ell, sigma_f)):
            return None, None

        alpha     = np.asarray(alpha,     dtype=np.float64).reshape(-1)
        Z_centers = np.asarray(Z_centers, dtype=np.float64)
        ell       = np.asarray(ell,       dtype=np.float64).reshape(-1)
        sigma_f   = float(np.asarray(sigma_f).reshape(-1)[0])

        if Z_centers.shape[1] != 6 or ell.size != 6:
            return None, None

        # Physical lengthscales (undo z-score normalisation when active).
        is_zscored = (feature_zscore is not None
                      and int(np.asarray(feature_zscore).reshape(-1)[0]) == 1)
        if is_zscored and feat_std is not None and np.asarray(feat_std).size == 6:
            fs = np.asarray(feat_std, dtype=np.float64).reshape(-1)
            ell_phys = ell * fs
        else:
            ell_phys = ell.copy()

        ell_R, ell_P = ell_phys[0], ell_phys[1]
        ell_map      = ell_phys[2:6]                              # (4,)
        x_j          = Z_centers[:, 2:6]                         # (N, 4)  mapping centres

        # Per-mapping-axis Gaussian-moment constants (same as 2D bath function).
        a_d    = 1.0/hbar + 1.0/(2.0 * ell_map**2)               # (4,)
        inv2l2 = 1.0/(2.0 * ell_map**2)                          # (4,)
        mu_d   = (x_j * inv2l2[None, :]) / a_d[None, :]          # (N, 4)
        C_d    = -x_j**2 * inv2l2[None, :] \
                  * (1.0 - inv2l2[None, :] / a_d[None, :])       # (N, 4)

        log_sqrt_pi_over_a = 0.5 * np.log(np.pi / a_d)           # (4,)
        log_Pj = np.sum(C_d, axis=1) + float(np.sum(log_sqrt_pi_over_a))  # (N,)
        M_j    = np.sum(mu_d**2 + 0.5/a_d[None, :], axis=1) - hbar       # (N,)

        # Combined per-centre weight  w_j = alpha_j * M_j * exp(log_Pj).
        w_j = alpha * M_j * np.exp(log_Pj)                       # (N,)

        pref_base = 8.0 * (sigma_f**2) / hbar

        # R-marginal: integrate ρ_bath(R, P) over P analytically.
        #   Int dP exp(-0.5 (P - P_j)^2 / ell_P^2) = sqrt(2 pi) ell_P
        pref_R = pref_base * np.sqrt(2.0 * np.pi) * ell_P
        diff_R = R_grid[:, None] - Z_centers[None, :, 0]         # (nR, N)
        G_R    = np.exp(-0.5 * (diff_R / ell_R)**2)              # (nR, N)
        rho_R  = pref_R * (G_R * w_j[None, :]).sum(axis=1)       # (nR,)

        # P-marginal: integrate ρ_bath(R, P) over R analytically.
        pref_P = pref_base * np.sqrt(2.0 * np.pi) * ell_R
        diff_P = P_grid[:, None] - Z_centers[None, :, 1]         # (nP, N)
        G_P    = np.exp(-0.5 * (diff_P / ell_P)**2)              # (nP, N)
        rho_P  = pref_P * (G_P * w_j[None, :]).sum(axis=1)       # (nP,)
        return rho_R, rho_P

    fz = snap.get("feature_zscore")
    feat_std = snap.get("feature_std")

    # Primary component.  For vanilla GPDensity snapshots snap["alpha"] IS
    # the full density coefficient vector; for density-diff snapshots it is
    # the CORRECTION coefficient α_δ only (Collector.Snapshot docstring).
    rho_R, rho_P = _component(snap.get("alpha"), snap.get("Z"),
                              snap.get("lengthscales"), snap.get("sigma_f"),
                              fz, feat_std)

    # Density-diff regime: add the frozen baseline component
    # ρ̂(z) = Σ α₀ k₀(z, Z₀) + Σ α_δ k_δ(z, Z) so the rendered marginal is
    # the FULL density, not just the correction.
    idd = snap.get("is_density_diff")
    is_diff = (idd is not None and int(np.asarray(idd).reshape(-1)[0]) == 1)
    if is_diff:
        rho_R_b, rho_P_b = _component(
            snap.get("alpha_base"), snap.get("Z0"),
            snap.get("lengthscales_base"), snap.get("sigma_f_base"),
            # Baseline GP never uses feature z-scoring in the diff pipeline;
            # its stored lengthscales are physical.
            None, None)
        if rho_R_b is not None:
            if rho_R is None:
                rho_R, rho_P = rho_R_b, rho_P_b
            else:
                rho_R = rho_R + rho_R_b
                rho_P = rho_P + rho_P_b
        else:
            print("[_bath_marginals_1d] WARNING: density-diff snapshot is "
                  "missing baseline GP fields (alpha_base/Z0/...); the "
                  "rendered marginal is the CORRECTION δ̂ only — re-save the "
                  "run with the updated Collector to fix.")

    if rho_R is None:
        return None, None

    # Guard: if the bath integral is numerically degenerate (should not
    # happen in normal runs) fall back to the flat marginal so the panel
    # still renders.  The flat marginal has the correct shape for focused
    # mode (constant recovery kernel on the focus circle) but is wrong
    # in scale and for seo_signed mode — caller should annotate.
    if not np.any(np.isfinite(rho_R)) or float(np.nanmax(np.abs(rho_R))) == 0.0:
        rho_bath_2d = _density_2d_from_gp_marginal(snap, R_grid, P_grid)
        if rho_bath_2d is not None:
            rho_R = rho_bath_2d.sum(axis=1) * (P_grid[1] - P_grid[0])
            rho_P = rho_bath_2d.sum(axis=0) * (R_grid[1] - R_grid[0])

    return rho_R, rho_P


def _cloud_snapshots_from_npz(npz_path: str,
                              t_targets: List[float]) -> List[Dict]:
    """
    Pull trajectory cloud (Z, y, t) AND the fitted GP surrogate parameters
    (α, lengthscales, σ_f, feature normalization) at the snapshot steps
    closest to each requested time.  Returns a list aligned with
    t_targets.  Downstream consumers can use the GP parameters to
    evaluate the analytic 2D marginal of ρ̂ on (R, P) rather than a 2D
    KDE of the cloud.
    """
    # Memory-safe selective read.  Only ONE snapshot per requested target
    # time is ever needed (the nearest in time), so materialising every
    # snapshot via ``dict(np.load(...))`` — the previous behaviour — wasted
    # hundreds of MB on long runs and risked MemoryError.  Instead open the
    # archive lazily: the per-snapshot time stamps are tiny 1-element members
    # (``snap_XXXXXX_t``), so reading just those to locate the nearest steps
    # is cheap, and only the selected snapshots' full members are then
    # decompressed.
    out: List[Dict] = []
    with np.load(npz_path) as z:
        keys = set(z.files)
        snap_steps = sorted({int(k.split("_")[1]) for k in keys
                             if k.startswith("snap_") and k.endswith("_t")})
        if not snap_steps:
            return out
        snap_times = np.asarray([float(z[f"snap_{s:06d}_t"][0])
                                 for s in snap_steps])

        def _read(prefix: str, name: str):
            key = f"{prefix}_{name}"
            return z[key] if key in keys else None

        for tt in t_targets:
            i = int(np.argmin(np.abs(snap_times - tt)))
            s = snap_steps[i]
            prefix = f"snap_{s:06d}"
            out.append(dict(
                t=float(z[f"{prefix}_t"][0]),
                Z=z[f"{prefix}_Z"],
                y=z[f"{prefix}_y"],
                # GP surrogate parameters (present in all runs since the
                # collector saves them every snapshot)
                alpha=_read(prefix, "alpha"),
                lengthscales=_read(prefix, "lengthscales"),
                sigma_f=_read(prefix, "sigma_f"),
                feature_mean=_read(prefix, "feature_mean"),
                feature_std=_read(prefix, "feature_std"),
                # feature_zscore needed by _bath_marginals_1d to convert
                # stored (normalised-space) lengthscales to physical units.
                # Returns None from old NPZ files; _bath_marginals_1d
                # treats None as False (no z-scoring), which is correct
                # for the default feature_zscore=False pipeline runs.
                feature_zscore=_read(prefix, "feature_zscore"),
                # IS weight (focused-mode: omega_i = 1/(N q(z_i^0))).
                weight=_read(prefix, "weight"),
                geometric_measure=_read(prefix, "geometric_measure"),
                proposal_density=_read(prefix, "proposal_density"),
                is_product=_read(prefix, "is_product"),
                product_hbar=_read(prefix, "product_hbar"),
                product_init_state=_read(prefix, "product_init_state"),
                product_nstates=_read(prefix, "product_nstates"),
                product_g_floor_rel=_read(prefix, "product_g_floor_rel"),
                product_transported=_read(prefix, "product_transported"),
                # Density-difference regime.  CRITICAL: for diff runs the
                # top-level `alpha` is the CORRECTION coefficient vector
                # α_δ ONLY (see Collector.Snapshot docstring).  Rendering
                # the density from `alpha` alone silently plots the
                # correction instead of ρ̂ = ρ̂₀ + δ̂.  The baseline fields
                # below let _bath_marginals_1d / the 2D renderer assemble
                # the FULL density (baseline + correction).
                is_density_diff=_read(prefix, "is_density_diff"),
                alpha_base=_read(prefix, "alpha_base"),
                Z0=_read(prefix, "Z0"),
                sigma_f_base=_read(prefix, "sigma_f_base"),
                lengthscales_base=_read(prefix, "lengthscales_base"),
            ))
    return out


def panel_density_marginals(runs: Dict[str, Dict[str, np.ndarray]],
                             gp_dir: str,
                             out_dir: str,
                             t_targets: List[float],
                             R0: float = -15.0,
                             P0: float = 40.0,
                             sigma_R: float = 1.0,
                             hbar: float = 1.0) -> None:
    """
    Build the R- and P-marginal density panels for all four methods at
    the requested snapshot times.  All curves are normalised so that
    ∫ρ_R dR = 1 and ∫ρ_P dP = 1, removing any normalisation differences
    between method-specific weighting conventions.

    Layout:  2 rows × len(t_targets) columns.
        Row 0:  ρ(R)  at each time
        Row 1:  ρ(P)  at each time

    Methods overlaid:  SE, QCLE, PBME, midpoint  (whichever are present).

    Mathematical object comparison
    -------------------------------
    All four rows plot the same physical quantity: the nuclear density
    obtained by tracing over electronic states.

    * SE:        rho_R(R) = |psi_0(R)|^2 + |psi_1(R)|^2,
                 rho_P(P) = |phi_0(P)|^2 + |phi_1(P)|^2  (FT)

    * QCLE:      rho_R(R) = Int dP [A(R,P) + C(R,P)],
                 rho_P(P) = Int dR [A(R,P) + C(R,P)]
                 These equal the SE marginals for a pure quantum state.

    * PBME/midpoint (NEW): 1D marginals of the bath density
                 rho_bath(R,P) = Sum_lambda rho_W^{ll}(R,P)
                               = (8/hbar) Int dr dp e^{-x^2/hbar}(x^2-hbar)
                                          * rho_hat(R, P, r, p)
                 computed analytically from the GP parameters via
                 _bath_marginals_1d — same closed-form mapping integrals as
                 _density_2d_from_gp_bath_marginal, then integrated over the
                 remaining nuclear axis analytically.

    The former implementation used _marginals_from_cloud, a plain 1/N KDE on
    trajectory positions that silently ignored the labels y_i and the MMST
    recovery kernel e^{-x^2/hbar}(x^2-hbar).  For focused mode (constant
    recovery kernel on the focus circle) this gave the correct SHAPE after
    normalisation; for seo_signed mode or post-midpoint times where y_i
    varies, it was quantitatively and qualitatively wrong.
    """
    # ---------------- Plotting grid setup ----------------
    n_t = len(t_targets)
    # R grid: covers initial packet + travel distance for all methods
    travel = (P0 / 2000.0) * max(t_targets)            # rough estimate
    R_lo = min(R0 - 5.0 * sigma_R, R0 - 5.0)
    R_hi = R0 + travel + 8.0 * sigma_R + 5.0
    R_grid = np.linspace(R_lo, R_hi, 400)
    # P grid: cover P0 ± 6σ_P plus upper-state kick
    sigma_P = hbar / (2.0 * sigma_R)
    P_lo = P0 - 8.0 * sigma_P - 8.0
    P_hi = P0 + 8.0 * sigma_P + 8.0
    P_grid = np.linspace(P_lo, P_hi, 400)

    # Pre-extract cloud snapshots once each
    cloud_snaps: Dict[str, List[Dict]] = {}
    for name, npz in [("PBME",     os.path.join(gp_dir, "pbme.npz")),
                      ("midpoint", os.path.join(gp_dir, "midpoint.npz"))]:
        if name in runs and os.path.exists(npz):
            cloud_snaps[name] = _cloud_snapshots_from_npz(npz, t_targets)

    # ---------------- R-marginal row ----------------
    fig, axes = plt.subplots(2, n_t, figsize=(3.2 * n_t + 1.0, 6.6),
                              constrained_layout=True, sharey="row")
    if n_t == 1:
        axes = axes.reshape(2, 1)

    for col, t_tgt in enumerate(t_targets):
        ax_R = axes[0, col]; ax_P = axes[1, col]
        actual_t_label = t_tgt   # update with the actual nearest time per method

        # SE
        if "SE" in runs and runs["SE"].get("snap_psi"):
            r = runs["SE"]
            t_se = np.asarray(r["snap_t"])
            i = int(np.argmin(np.abs(t_se - t_tgt)))
            rho_R, rho_P = _marginals_from_se(
                r["snap_psi"][i], r["snap_R"], r["snap_k"],
                r["snap_Lx"], r["snap_hbar"], P_grid)
            # Restrict R curve to R_grid range via interpolation
            rho_R_on_grid = np.interp(R_grid, r["snap_R"], rho_R,
                                       left=0.0, right=0.0)
            ax_R.plot(R_grid, _normalize(rho_R_on_grid, R_grid), **_curve("SE"))
            ax_P.plot(P_grid, _normalize(rho_P,        P_grid), **_curve("SE"))

        # QCLE
        if "QCLE" in runs and runs["QCLE"].get("snap_states"):
            r = runs["QCLE"]
            t_qc = np.asarray(r["snap_t"])
            i = int(np.argmin(np.abs(t_qc - t_tgt)))
            rho_R_q, rho_P_q = _marginals_from_qcle(
                r["snap_states"][i], r["snap_R_axis"], r["snap_P_axis"],
                r["snap_dR"], r["snap_dP"])
            rho_R_on_grid = np.interp(R_grid, r["snap_R_axis"], rho_R_q,
                                       left=0.0, right=0.0)
            rho_P_on_grid = np.interp(P_grid, r["snap_P_axis"], rho_P_q,
                                       left=0.0, right=0.0)
            ax_R.plot(R_grid, _normalize(rho_R_on_grid, R_grid), **_curve("QCLE"))
            ax_P.plot(P_grid, _normalize(rho_P_on_grid, P_grid), **_curve("QCLE"))

        # PBME / midpoint — use analytic 1D bath-density marginals so that
        # these curves represent the same mathematical object as SE and QCLE:
        # the partial-Wigner diagonal trace marginalised over (P or R).
        # _bath_marginals_1d computes
        #   rho_R(R) = (8 sf2/hbar) sqrt(2pi) ell_P  Sum_j w_j G_R(R; R_j)
        #   rho_P(P) = (8 sf2/hbar) sqrt(2pi) ell_R  Sum_j w_j G_P(P; P_j)
        # with w_j = alpha_j M_j exp(log_Pj) from the closed-form 4D mapping
        # integral including the MMST recovery kernel e^{-x^2/hbar}(x^2-hbar).
        # The former _marginals_from_cloud was a plain 1/N KDE on positions
        # that ignored labels y_i and the recovery kernel entirely.
        for name in ("PBME", "midpoint"):
            if name in cloud_snaps:
                snap = cloud_snaps[name][col]
                rho_R_c, rho_P_c = _bath_marginals_1d(
                    snap, R_grid, P_grid, hbar=hbar)
                _path = "bath"
                if rho_R_c is None:
                    # Snapshot predates GP parameter saving (older NPZ):
                    # fall back to moment-projection KDE.
                    rho_2d = _density_2d_from_gp_moment_projection(
                        snap, R_grid, P_grid)
                    if rho_2d is not None:
                        dP = float(P_grid[1] - P_grid[0])
                        dR = float(R_grid[1] - R_grid[0])
                        rho_R_c = rho_2d.sum(axis=1) * dP
                        rho_P_c = rho_2d.sum(axis=0) * dR
                        _path = "moment-proj-fallback"
                    else:
                        _path = "no-gp-params-skipped"
                _rmax = float(np.nanmax(np.abs(rho_R_c))) if rho_R_c is not None else float("nan")
                print(f"  [1D marginals] {name} t={t_tgt:g}  path={_path}  max|rho_R|={_rmax:.3e}")
                if rho_R_c is not None:
                    ax_R.plot(R_grid, _normalize(rho_R_c, R_grid), **_curve(name))
                    ax_P.plot(P_grid, _normalize(rho_P_c, P_grid), **_curve(name))

        ax_R.set_xlabel(r"$R$  [a.u.]")
        ax_P.set_xlabel(r"$P$  [a.u.]")
        if col == 0:
            ax_R.set_ylabel(r"$\rho(R, t)$")
            ax_P.set_ylabel(r"$\rho(P, t)$")
        ax_R.set_xlim(R_grid[0], R_grid[-1])
        ax_P.set_xlim(P_grid[0], P_grid[-1])
        ax_R.set_ylim(bottom=0)
        ax_P.set_ylim(bottom=0)

    axes[0, 0].legend(loc="best")
    _save_pub(fig, os.path.join(out_dir, "panel_density_marginals"))
    plt.close(fig)
    print(f"  -> panel_density_marginals.{{pdf,png}}")


def _normalize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Normalise a density on x so ∫y dx = 1.  Used to compare shapes
    across methods whose absolute normalisation conventions differ."""
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    Z = float(_trapz(y, x))
    if not np.isfinite(Z) or abs(Z) < 1.0e-30:
        return y
    return y / Z


# ============================================================================
# 2D phase-space density extraction — same (R, P) target grid for all methods
# ============================================================================
def _wigner_2state_diag_trace(psi: np.ndarray,
                              R_axis: np.ndarray,
                              R_out: np.ndarray,
                              P_out: np.ndarray,
                              Qtr: float = 20.0,
                              hbar: float = 1.0) -> np.ndarray:
    """
    Partial Wigner transform of a 2-state diabatic wavefunction, returning
    the diagonal trace  W_00(R, P) + W_11(R, P) ≥ 0 on the output grid.

    Definition:
        W_λλ(R, P) = (1/πℏ) ∫ dQ  e^{-2iQP/ℏ}  ψ_λ*(R-Q)  ψ_λ(R+Q)

    The integral over Q is truncated to |Q| < Qtr (default 20 a.u. — wide
    enough that the Gaussian wavefunction is below 1e-16 at the boundary
    for σ_R ≤ 2) and a Hann window is applied to suppress the sinc-ring
    sidebands that rectangular truncation produces.  Together these two
    choices push the FFT periodic-image leakage well below 1e-4 of the
    wavepacket peak — invisible against the physical density.

    Returns a real (n_R_out, n_P_out) array.
    """
    R_in = np.asarray(R_axis, dtype=np.float64)
    dR = float(R_in[1] - R_in[0])
    N_in = R_in.size

    K = int(np.floor(Qtr / dR))
    Qv = np.arange(-K, K + 1, dtype=np.float64) * dR

    # Hann window over the Q-integral.  W_Hann(Q) = ½(1 + cos(πQ/Qtr)),
    # zero at the endpoints, unity at Q=0.  Multiplying the integrand by
    # this window suppresses the boundary-discontinuity sidebands without
    # introducing significant spectral broadening (FWHM increase ~10%).
    Qmax_actual = float(np.abs(Qv).max()) if Qv.size else 1.0
    Qmax_actual = max(Qmax_actual, 1.0e-30)
    hann = 0.5 * (1.0 + np.cos(np.pi * Qv / Qmax_actual))

    # phase[m, j] = exp(-2i Q_m P_j / hbar), shape (n_Q, n_P)
    phase = np.exp(-2.0j * Qv[:, None] * P_out[None, :] / hbar)
    # Pre-multiply phase by Hann window so the integration absorbs it
    # without extra runtime arithmetic per (m, j).
    phase = phase * hann[:, None]
    pref = dR / (np.pi * hbar)

    def _interp(c: int, R_eval: np.ndarray) -> np.ndarray:
        fr = (R_eval - R_in[0]) / dR
        lo = np.clip(fr.astype(np.int64), 0, N_in - 2)
        frac = fr - lo
        ok = (fr >= 0.0) & (fr <= N_in - 1)
        v = ((1.0 - frac) * psi[c, lo.clip(0, N_in - 2)]
             + frac      * psi[c, (lo + 1).clip(0, N_in - 1)])
        return np.where(ok, v, 0.0 + 0.0j)

    W_trace = np.zeros((R_out.size, P_out.size), dtype=np.float64)
    for cc in (0, 1):                                     # diagonal only
        W_c = np.zeros((R_out.size, P_out.size), dtype=complex)
        for m, R_m in enumerate(R_out):
            R_plus  = R_m + Qv
            R_minus = R_m - Qv
            ok = ((R_plus  >= R_in[0]) & (R_plus  <= R_in[-1]) &
                  (R_minus >= R_in[0]) & (R_minus <= R_in[-1]))
            integrand = _interp(cc, R_plus) * np.conj(_interp(cc, R_minus))
            integrand[~ok] = 0.0
            W_c[m] = integrand @ phase
        W_trace += np.real(W_c) * pref
    return W_trace


def _density_2d_from_se(snap_psi: np.ndarray,
                         R_axis: np.ndarray,
                         R_grid: np.ndarray,
                         P_grid: np.ndarray,
                         hbar: float = 1.0) -> np.ndarray:
    """Diabatic-trace Wigner density on a (R_grid, P_grid) mesh."""
    return _wigner_2state_diag_trace(snap_psi, R_axis, R_grid, P_grid,
                                     Qtr=20.0, hbar=hbar)


def _density_2d_from_qcle(state: QCLEGridState,
                           R_axis_in: np.ndarray,
                           P_axis_in: np.ndarray,
                           R_grid: np.ndarray,
                           P_grid: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of the QCLE A+C grid onto (R_grid, P_grid)."""
    rho_full = state.A + state.C
    # Build interpolation indices
    iR = np.searchsorted(R_axis_in, R_grid).clip(1, R_axis_in.size - 1)
    iP = np.searchsorted(P_axis_in, P_grid).clip(1, P_axis_in.size - 1)
    R0_ = R_axis_in[iR - 1]; R1_ = R_axis_in[iR]
    P0_ = P_axis_in[iP - 1]; P1_ = P_axis_in[iP]
    fR = ((R_grid - R0_) / (R1_ - R0_)).clip(0.0, 1.0)
    fP = ((P_grid - P0_) / (P1_ - P0_)).clip(0.0, 1.0)
    # gather corners
    f00 = rho_full[iR - 1][:, iP - 1]
    f10 = rho_full[iR    ][:, iP - 1]
    f01 = rho_full[iR - 1][:, iP    ]
    f11 = rho_full[iR    ][:, iP    ]
    # bilinear blend
    out = ((1.0 - fR[:, None]) * (1.0 - fP[None, :]) * f00
           +       fR[:, None]  * (1.0 - fP[None, :]) * f10
           + (1.0 - fR[:, None]) *       fP[None, :]  * f01
           +       fR[:, None]  *       fP[None, :]  * f11)
    # mask points outside input axis range
    inR = (R_grid >= R_axis_in[0]) & (R_grid <= R_axis_in[-1])
    inP = (P_grid >= P_axis_in[0]) & (P_grid <= P_axis_in[-1])
    out *= (inR[:, None] & inP[None, :]).astype(np.float64)
    return out


def _gaussian_smooth_2d(arr: np.ndarray, sigma_cells: float = 1.0) -> np.ndarray:
    """
    Cheap separable Gaussian convolution on a 2D array.

    Used to suppress grid-discretisation artifacts in the SE Wigner
    transform (spectral leakage from finite Q truncation, visible as
    horizontal stripes far from the wavepacket) and the QCLE bilinear-
    interpolation staircase, BEFORE rendering.  ``sigma_cells = 1`` is
    a light smoothing that removes one-cell quantisation without
    affecting structure broader than 2 cells.

    Returns an array of the same shape.  NaN entries are preserved.
    """
    if sigma_cells <= 0.0:
        return arr
    a = np.asarray(arr, dtype=np.float64)
    # 1D Gaussian kernel, truncated at 4σ
    k_half = max(1, int(np.ceil(4.0 * sigma_cells)))
    x = np.arange(-k_half, k_half + 1, dtype=np.float64)
    kern = np.exp(-0.5 * (x / sigma_cells) ** 2)
    kern /= kern.sum()
    # Separable convolution; preserve NaN locations by mask trick
    finite = np.isfinite(a)
    a_safe = np.where(finite, a, 0.0)
    # Axis 0
    from numpy.lib.stride_tricks import sliding_window_view
    def _conv_axis(x: np.ndarray, axis: int) -> np.ndarray:
        # Reflect-pad so edges don't dim
        pad = [(0, 0)] * x.ndim
        pad[axis] = (k_half, k_half)
        xp = np.pad(x, pad, mode="reflect")
        # Build view that slides the kernel along `axis`
        win = sliding_window_view(xp, window_shape=kern.size, axis=axis)
        return np.tensordot(win, kern, axes=([-1], [0]))
    sm = _conv_axis(_conv_axis(a_safe, axis=0), axis=1)
    w  = _conv_axis(_conv_axis(finite.astype(np.float64), axis=0), axis=1)
    out = np.where(w > 1e-12, sm / np.maximum(w, 1e-12), 0.0)
    # Restore NaN where the original was NaN throughout the smoothing window
    return out


def _mask_to_packet_tube(rho: np.ndarray,
                          R_grid: np.ndarray, P_grid: np.ndarray,
                          t: float,
                          R0: float, P0: float,
                          sigma_R: float, hbar: float,
                          mass: float,
                          r_widen_factor: float = 6.0,
                          p_widen_factor: float = 8.0) -> np.ndarray:
    """
    Mask SE/QCLE grid densities to a tube around the classical wave-packet
    center, setting everything outside to NaN.

    Outside the tube the Wigner-FFT periodic-image artefacts dominate
    the rendering even after smoothing, because the wavepacket sits
    *inside* the periodic image and the image-of-the-image bleeds back
    into the far field.  Cosmetically these can be misread as physics;
    the honest treatment is to hide them.

    The tube is centred on the classical packet position
        R_c(t) = R₀ + (P₀/m)·t
    and the initial momentum P₀, with widths
        ΔR = r_widen_factor · σ_R                       (grows with time
              + max(|R_grid|) - R_c, conservatively)     to bracket spread
        ΔP = p_widen_factor · σ_P
    The tube is therefore an axis-aligned rectangle.  Pixels outside
    are set to NaN so neither viridis nor magma overlays render them.
    """
    R_c = R0 + (P0 / mass) * t
    sigma_P = hbar / (2.0 * sigma_R)
    # Conservative R half-width: at least r_widen_factor·σ_R, but also
    # cover the wavepacket spread (which has grown by Δt·σ_P/m at time t).
    dR_spread = (np.sqrt(sigma_R**2 + (t * sigma_P / mass) ** 2)
                 if mass > 0 else sigma_R)
    R_halfw = max(r_widen_factor * sigma_R, r_widen_factor * dR_spread)
    P_halfw = p_widen_factor * max(sigma_P, 1.5)
    inR = np.abs(R_grid - R_c) <= R_halfw
    inP = np.abs(P_grid - P0)  <= P_halfw
    mask = inR[:, None] & inP[None, :]
    return np.where(mask, rho, np.nan)


def _density_2d_from_gp_marginal(snap: Dict[str, np.ndarray],
                                  R_grid: np.ndarray,
                                  P_grid: np.ndarray) -> Optional[np.ndarray]:
    """
    Analytic 2D marginal of the GP surrogate on (R, P).

    The GP fits a 6D density on z = (R, P, r₀, r₁, p₀, p₁) with the
    ARD-RBF kernel
        k(z, z') = σ_f² · exp(-½ Σ_d (z_d - z'_d)² / ℓ_d²) .
    For Gaussian kernels the marginal integration over the four mapping
    dimensions is closed-form:
        ∫ dr₀ dr₁ dp₀ dp₁  k(z, z_j) = σ_f² · C_map ·
            exp(-½ (R-R_j)² / ℓ_R²  -½ (P-P_j)² / ℓ_P²) ,
    with
        C_map = (2π)² · ℓ_{r₀} ℓ_{r₁} ℓ_{p₀} ℓ_{p₁} .
    The integrations are over PHYSICAL coordinates, so all lengthscales
    here must be in physical units.

    Normalization handling
    ----------------------
    The pipeline supports both raw and z-scored GP fits.  When
    `feature_zscore=False` (which is the default for the focused-mode
    runs in this study), the GP's stored lengthscales are in PHYSICAL
    units and we use them directly.  When `feature_zscore=True`, the
    stored lengthscales are in NORMALIZED units and we multiply them
    by `feature_std` to recover the physical-unit widths.

    The previous version of this function applied z-score normalization
    unconditionally, which was a bug: with `feature_zscore=False` the
    kernel was effectively evaluated at WIDTH = feature_std · ell rather
    than ell, producing kernels narrower than the GP actually uses by a
    factor of σ_phys.  That mismatched bandwidth manifested as severe
    ringing in the marginal (α-cancellation got amplified through too-
    narrow kernels), causing the figure to suggest the surrogate was
    pathologically unfaithful when in fact it was the visualizer.

    Returns None when the snapshot lacks GP parameters.
    """
    alpha = snap.get("alpha")
    Z_centers = snap.get("Z")
    ell = snap.get("lengthscales")
    sigma_f = snap.get("sigma_f")
    if any(v is None for v in (alpha, Z_centers, ell, sigma_f)):
        return None

    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    Z_centers = np.asarray(Z_centers, dtype=np.float64)
    ell = np.asarray(ell, dtype=np.float64).reshape(-1)
    sigma_f = float(np.asarray(sigma_f).reshape(-1)[0])

    if Z_centers.shape[1] != 6 or ell.size != 6:
        return None

    # Resolve whether the GP was fit on z-scored features.  Convert all
    # lengthscales to PHYSICAL units accordingly.
    fz = snap.get("feature_zscore")
    feat_std = snap.get("feature_std")
    is_zscored = (fz is not None and int(np.asarray(fz).reshape(-1)[0]) == 1)
    if is_zscored and feat_std is not None and np.asarray(feat_std).size == 6:
        fs = np.asarray(feat_std, dtype=np.float64).reshape(-1)
        ell_phys = ell * fs                                     # (6,) physical
    else:
        # Either zscore is off OR feature_std is missing — assume lengthscales
        # are already in physical units (which is the case for the focused-
        # mode runs we have in hand: feature_zscore=False).
        ell_phys = ell.copy()

    ell_R, ell_P = ell_phys[0], ell_phys[1]
    ell_map = ell_phys[2:6]

    # 2D Gaussian on (R, P) in PHYSICAL coordinates centred at each
    # support point's (R, P) projection, with widths ell_R, ell_P.
    diff_R = R_grid[:, None] - Z_centers[None, :, 0]            # (n_R, N)
    diff_P = P_grid[:, None] - Z_centers[None, :, 1]            # (n_P, N)
    G_R = np.exp(-0.5 * (diff_R / ell_R) ** 2)                  # (n_R, N)
    G_P = np.exp(-0.5 * (diff_P / ell_P) ** 2)                  # (n_P, N)

    # Marginal prefactor in physical units:
    #   σ_f² · (2π)² · ℓ_{r₀} ℓ_{r₁} ℓ_{p₀} ℓ_{p₁}
    C_map = float(np.prod(np.sqrt(2.0 * np.pi) * ell_map))
    pref = (sigma_f ** 2) * C_map

    # ρ_2D[i, k] = pref · Σ_j α_j G_R[i, j] G_P[k, j]
    weighted = G_R * alpha[None, :]                              # (n_R, N)
    rho_2D = pref * (weighted @ G_P.T)                           # (n_R, n_P)
    return rho_2D


def _density_2d_from_gp_bath_marginal(
        snap: Dict[str, np.ndarray],
        R_grid: np.ndarray,
        P_grid: np.ndarray,
        hbar: float = 1.0) -> Optional[np.ndarray]:
    """
    Bath density Tr_s ρ̂_W(R, P) — the partial-Wigner trace, which is the
    quantity the QCLE-grid solver propagates and plots.

    Recovery formula (Nassimi-Bonella-Kapral 2010, Eq. 16-17):

        ρ_W^{λλ'}(R, P) = ∫ dr dp  ρ_m(R, P, r, p) g_{λλ'}(r, p)

        g_{λλ'}(r, p) = (2^{N+1}/ℏ) e^{-x²/ℏ} ·
                         [r_λ r_{λ'} + p_λ p_{λ'}
                          - i(r_λ p_{λ'} - r_{λ'} p_λ)
                          - (ℏ/2) δ_{λλ'}]

    The trace over electronic states (diagonal sum) at N=2:

        Σ_λ g_{λλ}(r, p) = (8/ℏ) e^{-x²/ℏ} (x² - ℏ),    x² = r₀²+r₁²+p₀²+p₁²

    With the GP surrogate ρ_m(z) ≈ σ_f² Σ_j α_j Π_d exp(-(z_d-z_{j,d})²/(2ℓ_d²)),
    the integral over the 4 mapping dimensions factorizes.  For each
    dimension d ∈ {r_0, r_1, p_0, p_1} and each support point j, set
        a_d ≡ 1/ℏ + 1/(2ℓ_d²)
        μ_d^{j} ≡ (x_{j,d}/(2ℓ_d²)) / a_d
        C_d^{j} ≡ -x_{j,d}² / (2ℓ_d²) · (1 - (1/(2ℓ_d²))/a_d)
    so that the 1D integrals reduce to standard Gaussian moments:
        ∫ dx_d e^{-x_d²/ℏ - (x_d-x_{j,d})²/(2ℓ_d²)} · 1   = √(π/a_d) e^{C_d^{j}}
        ∫ dx_d e^{-x_d²/ℏ - (x_d-x_{j,d})²/(2ℓ_d²)} · x_d² = ditto · (μ² + 1/(2a))

    The bath density is

        ρ_bath(R,P) = (8 σ_f²/ℏ) · Σ_j α_j  G_R(R; R_j) G_P(P; P_j)
                                  · 𝒫^{j} · M^{j}

    with G_R, G_P the 2D-marginal Gaussians, 𝒫^{j} = Π_d √(π/a_d) e^{C_d^{j}}
    the product of mapping-axis normalizations, and
        M^{j} ≡ Σ_d (μ_d^{j} ² + 1/(2 a_d)) - ℏ.

    Note that M^{j} can be negative when the mapping centres are close to
    the origin (delta-function-like initial condition); the resulting
    negativity of ρ_bath is mathematically allowed (Wigner function) and
    physically meaningful (off-diagonal coherence).

    Verified against direct 4D Riemann integration to 6 decimal places.

    Returns None when the snapshot lacks GP parameters.
    """
    def _component_2d(alpha, Z_centers, ell, sigma_f, feature_zscore, feat_std):
        if any(v is None for v in (alpha, Z_centers, ell, sigma_f)):
            return None

        alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
        Z_centers = np.asarray(Z_centers, dtype=np.float64)
        ell = np.asarray(ell, dtype=np.float64).reshape(-1)
        sigma_f = float(np.asarray(sigma_f).reshape(-1)[0])

        if Z_centers.shape[1] != 6 or ell.size != 6:
            return None

        # Lengthscales: physical if zscore is off, else fold in feature_std.
        is_zscored = (feature_zscore is not None
                      and int(np.asarray(feature_zscore).reshape(-1)[0]) == 1)
        if is_zscored and feat_std is not None and np.asarray(feat_std).size == 6:
            fs = np.asarray(feat_std, dtype=np.float64).reshape(-1)
            ell_phys = ell * fs
        else:
            ell_phys = ell.copy()

        ell_R, ell_P = ell_phys[0], ell_phys[1]
        ell_map = ell_phys[2:6]                                       # (4,)
        x_j = Z_centers[:, 2:6]                                       # (N, 4)

        # Per-axis Gaussian-moment constants.
        a_d  = 1.0/hbar + 1.0/(2.0 * ell_map**2)                       # (4,)
        inv2l2 = 1.0/(2.0 * ell_map**2)                                # (4,)
        mu_d = (x_j * inv2l2[None, :]) / a_d[None, :]                  # (N, 4)
        C_d  = -x_j**2 * inv2l2[None, :] \
                * (1.0 - inv2l2[None, :] / a_d[None, :])               # (N, 4)

        # 𝒫^{j} = Π_d √(π/a_d) e^{C_d^{j}}   — use log to avoid underflow
        log_sqrt_pi_over_a = 0.5 * np.log(np.pi / a_d)                 # (4,)
        log_P_j = np.sum(C_d, axis=1) + float(np.sum(log_sqrt_pi_over_a))   # (N,)

        # M^{j} = Σ_d (μ_d² + 1/(2 a_d)) - ℏ
        M_j = np.sum(mu_d**2 + 0.5/a_d[None, :], axis=1) - hbar        # (N,)

        # Combined per-centre weight α_j · 𝒫^{j} · M^{j}, times overall scale 8 σ_f² / ℏ.
        w_j = alpha * M_j * np.exp(log_P_j)                            # (N,)

        # (R, P) Gaussian factor: ρ_bath[i,k] = pref · Σ_j w_j G_R[i,j] G_P[k,j]
        diff_R = R_grid[:, None] - Z_centers[None, :, 0]               # (n_R, N)
        diff_P = P_grid[:, None] - Z_centers[None, :, 1]               # (n_P, N)
        G_R = np.exp(-0.5 * (diff_R / ell_R)**2)                       # (n_R, N)
        G_P = np.exp(-0.5 * (diff_P / ell_P)**2)                       # (n_P, N)

        pref = 8.0 * (sigma_f**2) / hbar
        return pref * (G_R * w_j[None, :]) @ G_P.T                     # (n_R, n_P)

    rho_bath = _component_2d(snap.get("alpha"), snap.get("Z"),
                             snap.get("lengthscales"), snap.get("sigma_f"),
                             snap.get("feature_zscore"), snap.get("feature_std"))

    # Density-diff regime: snap["alpha"] is only the δ-correction — add the
    # frozen baseline component so the panel shows the full ρ̂ = ρ̂₀ + δ̂.
    idd = snap.get("is_density_diff")
    if idd is not None and int(np.asarray(idd).reshape(-1)[0]) == 1:
        rho_b = _component_2d(snap.get("alpha_base"), snap.get("Z0"),
                              snap.get("lengthscales_base"),
                              snap.get("sigma_f_base"), None, None)
        if rho_b is not None:
            rho_bath = rho_b if rho_bath is None else rho_bath + rho_b
        else:
            print("[_density_2d_from_gp_bath_marginal] WARNING: density-diff "
                  "snapshot lacks baseline GP fields; rendering δ̂ only.")
    return rho_bath


def _density_2d_from_gp_moment_projection(
        snap: Dict[str, np.ndarray],
        R_grid: np.ndarray,
        P_grid: np.ndarray) -> Optional[np.ndarray]:
    """Common-support projected GP estimate of the physical R--P marginal.

    The projected GP is conditioned on the all-trajectory KDE field using
    exactly the saved ``omega*y`` measure and the shared Scott/Silverman
    bandwidth.  This replaces the former direct 6D GP integration, which is
    non-identifiable for focused PBME because the mapping labels live on a
    lower-dimensional manifold.
    """
    Z = snap.get("Z")
    y = snap.get("y")
    if Z is None or y is None:
        return None

    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if Z.ndim != 2 or Z.shape[1] != 6 or y.size != Z.shape[0]:
        return None

    # Pull ω if it was saved; otherwise default to 1/N (uniform IS).
    # Under focused-mode IS:  ω_i = 1/(N · q(z_i^0)),  which is constant
    # along symplectic flow, so the t=0 value is the value at all times.
    omega = snap.get("geometric_measure")
    if omega is None or np.asarray(omega).size != Z.shape[0]:
        proposal = snap.get("proposal_density")
        if proposal is not None and np.asarray(proposal).size == Z.shape[0]:
            proposal = np.asarray(proposal, dtype=np.float64).reshape(-1)
            omega = 1.0 / (Z.shape[0] * np.maximum(proposal, np.finfo(float).tiny))
        else:
            omega = np.full(Z.shape[0], 1.0 / Z.shape[0], dtype=np.float64)
    else:
        omega = np.asarray(omega, dtype=np.float64).reshape(-1)

    saved_hbar = snap.get("product_hbar")
    hbar = (float(np.asarray(saved_hbar).reshape(-1)[0])
            if saved_hbar is not None else 1.0)
    trace_factor = ((np.sum(Z[:, 2:6] ** 2, axis=1) - 2.0 * hbar)
                    / (2.0 * hbar))
    projected = ProjectedNuclearGP().fit_from_cloud(
        Z, omega, y * trace_factor, dim_pair=(0, 1))
    # The shared estimator returns image ordering (P rows, R columns); this
    # comparison module uses matrix ordering (R rows, P columns).
    return projected.gp_grid(R_grid, P_grid).T


def _density_2d_from_cloud(Z: np.ndarray, y: np.ndarray,
                            R_grid: np.ndarray,
                            P_grid: np.ndarray) -> np.ndarray:
    """
    2D Gaussian KDE of the (R, P) cloud marginal.  Same focused-mode
    rationale as the 1D version: plain (unweighted) KDE on cloud positions
    reproduces ρ(R, P) for PBME; for midpoint it's the leading-order
    estimate of the corrected density.

    The 2D Gaussian factorises (the bandwidth matrix is diagonal), so the
    computation factors into outer products — O(N · n_R · n_P) flops.
    """
    R_i = np.asarray(Z[:, 0], dtype=np.float64)
    P_i = np.asarray(Z[:, 1], dtype=np.float64)
    N = R_i.size
    # Silverman 2D bandwidth (factor for 6-D effective sample reduced to 2D)
    h_R = 1.06 * float(np.std(R_i)) * N ** (-1.0 / 6.0)
    h_P = 1.06 * float(np.std(P_i)) * N ** (-1.0 / 6.0)
    h_R = max(h_R, 1.0e-3); h_P = max(h_P, 1.0e-3)
    # K_R[a, i] = exp(-0.5 ((R_a - R_i)/h_R)^2) / (h_R * sqrt(2π))
    sqrt2pi = np.sqrt(2.0 * np.pi)
    K_R = (np.exp(-0.5 * ((R_grid[:, None] - R_i[None, :]) / h_R) ** 2)
           / (h_R * sqrt2pi))                              # (n_R, N)
    K_P = (np.exp(-0.5 * ((P_grid[:, None] - P_i[None, :]) / h_P) ** 2)
           / (h_P * sqrt2pi))                              # (n_P, N)
    rho = (K_R @ K_P.T) / N                                # (n_R, n_P)
    return rho


# =============================================================================
# Reference-style phase-space figure: 2D density + 1D marginals per method
# =============================================================================

def _plot_phasespace_with_marginals(
    densities: Dict[str, Optional[np.ndarray]],
    R_grid: np.ndarray,
    P_grid: np.ndarray,
    t: float,
    savepath: str,
    *,
    vmax_quantile: float = 0.98,
    cloud_points: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Write the reference-style phase-space comparison figure.

    Layout per method column — replicates the attached reference image exactly:

        ┌──────┬──────────────────────────┬──┐
        │  ↑   │   Tr ρ_W  (2D map)      │  │
        │  P   │   seismic, sym. vmax     │cb│  top row
        │ marg │   R horizontal,          │  │
        │  ↓   │   P vertical             │  │
        └──────┴──────────────────────────┴──┘
               ┌──────────────────────────┐
               │   ∫ρ_W dP  (R-marginal)  │   bottom row
               └──────────────────────────┘

    Columns: SE | QCLE | PBME | midpoint (only those present in *densities*).
    One figure per snapshot time.  Saved as PDF + 300-dpi PNG.

    Parameters
    ----------
    densities  : ``{method_name: 2D array (n_R, n_P)}`` — None entries skipped.
    R_grid     : 1D array of R coordinates (length n_R).
    P_grid     : 1D array of P coordinates (length n_P).
    t          : snapshot time [a.u.].
    savepath   : output stem (no extension).
    """
    import warnings as _w
    import matplotlib.gridspec as _gs
    import matplotlib.ticker as _mtic

    METHOD_ORDER = ["SE", "QCLE", "PBME", "midpoint"]
    METHOD_COLOR = {"SE": "#333333", "QCLE": "#009E73",
                    "PBME": "#0072B2", "midpoint": "#D55E00"}
    CMAP = "seismic"     # blue(−) → white(0) → red(+), matches reference

    methods = [m for m in METHOD_ORDER
               if m in densities and densities[m] is not None]
    if not methods:
        return

    n_m = len(methods)
    dR  = float(R_grid[1] - R_grid[0])
    dP  = float(P_grid[1] - P_grid[0])

    # One symmetric scale across ALL compared methods.  Per-method color
    # normalization can make a weak surrogate look as intense as a reference
    # and was therefore unsuitable for a comparative thesis panel.
    finite_abs = [np.abs(np.asarray(densities[m], dtype=float))
                  for m in methods]
    finite_abs = np.concatenate([v[np.isfinite(v)] for v in finite_abs])
    shared_vmax = (float(np.quantile(finite_abs, vmax_quantile))
                   if finite_abs.size else 1.0)
    shared_vmax = max(shared_vmax, 1e-30)
    vmaxes: Dict[str, float] = {m: shared_vmax for m in methods}

    # ── figure geometry ───────────────────────────────────────────────────────
    # width per method: 1.0 (P-marg) + 3.5 (2D) + 0.25 (cb) = 4.75 in
    # total figure width: n_m * 4.75 + left/right margins
    # height: 3.8 (2D+P-marg) + 0.2 (gap) + 1.2 (R-marg) = 5.2 in
    cell_w  = 4.75
    fig_w   = n_m * cell_w + 0.35
    fig_h   = 5.4

    fig = plt.figure(figsize=(fig_w, fig_h))

    # outer GridSpec: one slot per method
    outer = _gs.GridSpec(
        1, n_m, figure=fig,
        left=0.07, right=0.99,
        top=0.91, bottom=0.08,
        wspace=0.30,
    )

    for mi, m in enumerate(methods):
        rho  = np.asarray(densities[m], dtype=float)
        vmax = vmaxes[m]
        mc   = METHOD_COLOR.get(m, "#777777")

        # inner GridSpec for this method: 2 rows × 3 cols
        inner = _gs.GridSpecFromSubplotSpec(
            2, 3,
            subplot_spec=outer[mi],
            height_ratios=[4, 1],
            width_ratios=[1, 4, 0.22],
            hspace=0.06,
            wspace=0.06,
        )
        ax2d  = fig.add_subplot(inner[0, 1])          # 2D density map
        axP   = fig.add_subplot(inner[0, 0], sharey=ax2d)  # P-marginal (left)
        axR   = fig.add_subplot(inner[1, 1], sharex=ax2d)  # R-marginal (bottom)
        axcb  = fig.add_subplot(inner[0, 2])           # colorbar

        # ── 2D density map ────────────────────────────────────────────────────
        rho_plot = np.where(np.isfinite(rho), rho, 0.0)
        im = ax2d.pcolormesh(
            R_grid, P_grid, rho_plot.T,
            cmap=CMAP, vmin=-vmax, vmax=vmax,
            shading="gouraud", rasterized=True,
        )
        if cloud_points is not None and m in cloud_points:
            pts = _subsample_cloud_for_overlay(cloud_points[m])
            if pts.size:
                ax2d.plot(pts[:, 0], pts[:, 1], ".", ms=0.7, color="k",
                          alpha=0.18, rasterized=True, zorder=3)
        ax2d.tick_params(labelbottom=False, labelleft=False,
                         which="both", direction="in", length=2.5, width=0.5)
        for sp in ax2d.spines.values(): sp.set_linewidth(0.5)

        # ── colorbar ─────────────────────────────────────────────────────────
        cb = plt.colorbar(im, cax=axcb)
        cb.ax.tick_params(labelsize=_TICK_FONT - 1, direction="in")
        # Limit tick labels: 5 ticks at ±vmax, ±vmax/2, 0
        ticks = [-vmax, -vmax/2, 0.0, vmax/2, vmax]
        cb.set_ticks(ticks)
        cb.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.2g}")
        )

        # ── P-marginal: ∫ρ dR vs P ───────────────────────────────────────────
        p_marg = np.nansum(rho_plot, axis=0) * dR   # (n_P,)
        axP.fill_betweenx(P_grid, 0.0, p_marg, color=mc, alpha=0.30)
        axP.plot(p_marg, P_grid, color=mc, lw=1.0)
        axP.invert_xaxis()
        axP.set_ylabel(r"$P$  [a.u.]", fontsize=_LABEL_FONT)
        axP.set_xlabel(r"$\int\rho_W\mathrm{d}R$", fontsize=_TICK_FONT - 1, labelpad=1)
        axP.tick_params(which="both", direction="in", labelsize=_TICK_FONT - 1,
                        length=2.5, width=0.5)
        axP.xaxis.set_minor_locator(_mtic.AutoMinorLocator())
        axP.yaxis.set_minor_locator(_mtic.AutoMinorLocator())
        for sp in axP.spines.values(): sp.set_linewidth(0.5)

        # ── R-marginal: ∫ρ dP vs R ───────────────────────────────────────────
        r_marg = np.nansum(rho_plot, axis=1) * dP   # (n_R,)
        axR.fill_between(R_grid, 0.0, r_marg, color=mc, alpha=0.30)
        axR.plot(R_grid, r_marg, color=mc, lw=1.0)
        axR.set_xlabel(r"$R$  [a.u.]", fontsize=_LABEL_FONT)
        axR.set_ylabel(r"$\int\rho_W\mathrm{d}P$", fontsize=_TICK_FONT - 1, labelpad=1)
        axR.tick_params(which="both", direction="in", labelsize=_TICK_FONT - 1,
                        length=2.5, width=0.5)
        axR.xaxis.set_minor_locator(_mtic.AutoMinorLocator())
        axR.yaxis.set_minor_locator(_mtic.AutoMinorLocator())
        for sp in axR.spines.values(): sp.set_linewidth(0.5)


    with _w.catch_warnings():
        _w.simplefilter("ignore", UserWarning)
        fig.savefig(savepath + ".pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(savepath + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def panel_phasespace_all_methods(
    runs: Dict[str, Dict[str, np.ndarray]],
    gp_dir: str,
    out_dir: str,
    t_targets: List[float],
    R0: float = -15.0,
    P0: float = 40.0,
    sigma_R: float = 1.0,
    hbar: float = 1.0,
    n_R: int = 128,
    n_P: int = 128,
    mass: float = MASS_DEFAULT,
) -> None:
    """Reference-style phase-space density panels — one figure per snapshot time.

    For each requested snapshot time, computes Tr ρ_W(R,P) for every method
    present in *runs* and writes the reference-style figure with 1D marginals
    on the left and bottom of each 2D panel.

    Output: ``{out_dir}/phasespace_t{t:06.0f}.{pdf,png}``
    """
    os.makedirs(out_dir, exist_ok=True)
    sigma_P = hbar / (2.0 * sigma_R)

    cloud_snaps: Dict[str, List] = {}
    for name, npz in [("PBME",     os.path.join(gp_dir, "pbme.npz")),
                       ("midpoint", os.path.join(gp_dir, "midpoint.npz"))]:
        if name in runs and os.path.exists(npz):
            cloud_snaps[name] = _cloud_snapshots_from_npz(npz, t_targets)

    for ti, t_tgt in enumerate(t_targets):
        # Data-driven box: include the actual support cloud and reference
        # moment envelopes, not just the classical centre.
        (R_lo, R_hi), (P_lo, P_hi) = _dynamic_rp_window(
            runs, cloud_snaps, ti, t_tgt,
            R0=R0, P0=P0, sigma_R=sigma_R, hbar=hbar, mass=mass,
        )
        R_grid = np.linspace(R_lo, R_hi, n_R)
        P_grid = np.linspace(P_lo, P_hi, n_P)

        densities: Dict[str, Optional[np.ndarray]] = {}
        cloud_overlay: Dict[str, np.ndarray] = {}

        # SE — Wigner transform of the wavefunction
        if "SE" in runs and runs["SE"].get("snap_psi"):
            t_se = np.asarray(runs["SE"]["snap_t"])
            idx  = int(np.argmin(np.abs(t_se - t_tgt)))
            rho  = _density_2d_from_se(
                runs["SE"]["snap_psi"][idx], runs["SE"]["snap_R"],
                R_grid, P_grid, hbar=hbar)
            densities["SE"] = _gaussian_smooth_2d(rho, sigma_cells=1.0)

        # QCLE — density from the propagated grid
        if "QCLE" in runs and runs["QCLE"].get("snap_states"):
            t_qc = np.asarray(runs["QCLE"]["snap_t"])
            idx  = int(np.argmin(np.abs(t_qc - t_tgt)))
            rho  = _density_2d_from_qcle(
                runs["QCLE"]["snap_states"][idx],
                runs["QCLE"]["snap_R_axis"],
                runs["QCLE"]["snap_P_axis"],
                R_grid, P_grid)
            densities["QCLE"] = _gaussian_smooth_2d(rho, sigma_cells=1.0)

        # PBME / midpoint — common-support projected physical marginal.
        for name in ("PBME", "midpoint"):
            if name not in cloud_snaps:
                continue
            snap = cloud_snaps[name][ti]
            rho = _density_2d_from_gp_moment_projection(snap, R_grid, P_grid)
            if rho is None:
                rho = _density_2d_from_cloud(snap["Z"], snap["y"], R_grid, P_grid)
            densities[name] = rho
            cloud_overlay[name] = snap["Z"]

        stem = os.path.join(out_dir, f"phasespace_t{t_tgt:06.0f}")
        print(f"  [phasespace] t={t_tgt:g} a.u.  methods present: {list(densities)}  "
              f"R∈[{R_lo:.2f},{R_hi:.2f}] P∈[{P_lo:.2f},{P_hi:.2f}]")
        _plot_phasespace_with_marginals(
            densities, R_grid, P_grid, t_tgt, stem, cloud_points=cloud_overlay)
        print(f"    → {stem}.pdf")


def panel_density_marginals_2d(runs: Dict[str, Dict[str, np.ndarray]],
                                gp_dir: str,
                                out_dir: str,
                                t_targets: List[float],
                                R0: float = -15.0,
                                P0: float = 40.0,
                                sigma_R: float = 1.0,
                                hbar: float = 1.0,
                                n_R: int = 144,
                                n_P: int = 144,
                                mass: float = MASS_DEFAULT,
                                vmax_quantile: float = 0.98) -> None:
    """
    Build the 2D phase-space density panel  ρ(R, P, t)  for all four
    methods at the requested snapshot times.

    Each column is zoomed to a tight (R, P) box around the wave-packet
    position at that time:
        R-window = R_center(t) ± 4 σ_R,  P-window = [P0-12, P0+12]
    where R_center(t) = R0 + P0/mass · t.  This is essential — using a
    fixed full-trajectory R-range collapses each packet into a few
    pixels and the structure is invisible.

    Colormap: ``viridis`` (perceptually uniform, dark→bright so even
    low-density regions are visible).  All four rows use the same map
    so the eye reads density values directly; row labels identify the
    method.  Per-column vmax is clipped to the ``vmax_quantile`` of the
    densities so the colorbar shows structure rather than saturating
    on a single peak pixel.

    Layout:  4 rows (SE, QCLE, PBME, midpoint) × N_times columns.
    """
    sigma_P = hbar / (2.0 * sigma_R)

    # Per-column plotting windows are now data-driven.  The previous
    # classical-centre window could miss reflected/transmitted branches and
    # PBME/midpoint support clouds after scattering.
    cloud_snaps: Dict[str, List[Dict]] = {}
    for name, npz in [("PBME",     os.path.join(gp_dir, "pbme.npz")),
                       ("midpoint", os.path.join(gp_dir, "midpoint.npz"))]:
        if name in runs and os.path.exists(npz):
            cloud_snaps[name] = _cloud_snapshots_from_npz(npz, t_targets)

    R_windows: List[Tuple[float, float]] = []
    P_windows: List[Tuple[float, float]] = []
    R_grids: List[np.ndarray] = []
    P_grids: List[np.ndarray] = []
    for ti, t_tgt in enumerate(t_targets):
        (R_lo, R_hi), (P_lo, P_hi) = _dynamic_rp_window(
            runs, cloud_snaps, ti, t_tgt,
            R0=R0, P0=P0, sigma_R=sigma_R, hbar=hbar, mass=mass,
        )
        R_windows.append((R_lo, R_hi))
        P_windows.append((P_lo, P_hi))
        R_grids.append(np.linspace(R_lo, R_hi, n_R))
        P_grids.append(np.linspace(P_lo, P_hi, n_P))

    for name, npz in [("PBME",     os.path.join(gp_dir, "pbme.npz")),
                      ("midpoint", os.path.join(gp_dir, "midpoint.npz"))]:
        if name in runs and os.path.exists(npz):
            cloud_snaps[name] = _cloud_snapshots_from_npz(npz, t_targets)

    methods_present = [m for m in ("SE", "QCLE", "PBME", "midpoint")
                        if m in runs]
    n_methods = len(methods_present)
    n_t = len(t_targets)

    # ---------------- compute all densities first (per-column zoomed grid) ----------------
    print("  Computing 2D densities on per-column zoom grids "
          "(Wigner transform is the slow step)...")
    densities: Dict[Tuple[str, int], np.ndarray] = {}
    for col, t_tgt in enumerate(t_targets):
        R_grid_c = R_grids[col]; P_grid_c = P_grids[col]
        for m in methods_present:
            if m == "SE" and runs["SE"].get("snap_psi"):
                t_se = np.asarray(runs["SE"]["snap_t"])
                i = int(np.argmin(np.abs(t_se - t_tgt)))
                rho = _density_2d_from_se(
                    runs["SE"]["snap_psi"][i],
                    runs["SE"]["snap_R"],
                    R_grid_c, P_grid_c, hbar=hbar)
                # Light Gaussian smoothing only — no NaN masking.
                # Earlier versions of this code masked out-of-tube
                # cells to NaN, but NaN combined with gouraud shading
                # made matplotlib render any quad with a NaN vertex
                # as the panel background colour, which read as black
                # bands around the wave packet.  The zoom window
                # itself bounds the visible region; residual FFT
                # periodic-image structure in the far field is a
                # known limitation of the truncated Wigner transform
                # but at least is not amplified by the renderer.
                rho = _gaussian_smooth_2d(rho, sigma_cells=1.0)
            elif m == "QCLE" and runs["QCLE"].get("snap_states"):
                t_qc = np.asarray(runs["QCLE"]["snap_t"])
                i = int(np.argmin(np.abs(t_qc - t_tgt)))
                rho = _density_2d_from_qcle(
                    runs["QCLE"]["snap_states"][i],
                    runs["QCLE"]["snap_R_axis"],
                    runs["QCLE"]["snap_P_axis"],
                    R_grid_c, P_grid_c)
                rho = _gaussian_smooth_2d(rho, sigma_cells=1.0)
            elif m in cloud_snaps:
                snap = cloud_snaps[m][col]
                # Physical electronic-trace nuclear density from the saved
                # importance-sampling measure, represented by the shared
                # projected GP.  This is the same object and bandwidth used
                # by the KDE baseline and is identifiable for focused PBME.
                rho = _density_2d_from_gp_moment_projection(
                    snap, R_grid_c, P_grid_c)
                if rho is None:
                    rho = _density_2d_from_cloud(
                        snap["Z"], snap["y"], R_grid_c, P_grid_c)
            else:
                rho = None
            densities[(m, col)] = rho
        print(f"    t = {t_tgt:7.1f} a.u.  done")

    # =================================================================
    # TWO SEPARATE PANELS — one for ρ>0 (viridis), one for ρ<0 (magma).
    # =================================================================
    # Each panel is a single-colormap rendering with no sign-boundary,
    # so there is no possibility of edge artifacts at the transition
    # between the two signs.  The two panels are independent figures
    # produced from the same densities dict, sharing the per-column
    # zoom windows and labels.
    #
    # Positive panel:
    #     One row per method (SE/QCLE/PBME/midpoint), one column per t.
    #     Renders max(ρ, 0) only; everything else is the viridis-zero
    #     background.  vmax is the column-wide quantile of ρ⁺ across
    #     all methods, so panels can be compared row-to-row.
    #
    # Negative panel:
    #     Same structure, but renders |ρ⁻| = -min(ρ, 0).  Only SE and
    #     QCLE rows have content; PBME and midpoint rows (positive KDE)
    #     are explicitly labeled "(no negativity — KDE)".  The
    #     magnitude scale is per-panel adaptive (each panel uses its
    #     own peak negativity) so even weakly-negative panels are
    #     legible.

    def _render_one_sign(filename_root: str,
                          sign: str,            # "pos" or "neg"
                          cmap_name: str,
                          fc_color: str,
                          cb_label: str,
                          title_suffix: str) -> None:
        del title_suffix
        fig, axes = plt.subplots(n_methods, n_t,
                                  figsize=(2.8 * n_t + 1.6, 2.6 * n_methods + 0.8),
                                  constrained_layout=True)
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        if n_t == 1:
            axes = axes.reshape(-1, 1)

        # One common magnitude scale across every method and time panel.  All
        # rows now represent the same normalized nuclear-density object, so a
        # per-row scale would hide genuine amplitude differences.
        all_vals = []
        for m in methods_present:
            for col in range(n_t):
                r = densities[(m, col)]
                if r is None: continue
                rv = np.asarray(r).ravel()
                rv = rv[np.isfinite(rv)]
                if sign == "pos":
                    all_vals.append(rv[rv > 0])
                else:
                    all_vals.append(-rv[rv < 0])
        arr = np.concatenate(all_vals) if all_vals else np.array([])
        v_max_common = max(
            float(np.quantile(arr, vmax_quantile)) if arr.size else 1.0,
            1.0e-30)

        for col, t_tgt in enumerate(t_targets):
            R_grid_c = R_grids[col]; P_grid_c = P_grids[col]
            for row, m in enumerate(methods_present):
                v_max = v_max_common
                ax = axes[row, col]
                rho = densities[(m, col)]
                if rho is None:
                    ax.set_visible(False); continue

                if sign == "pos":
                    # Render max(ρ, 0) directly.  We do NOT NaN-mask
                    # negative cells: gouraud shading interpolates across
                    # NaN/finite boundaries by producing transparent
                    # stripes, which on a dark background renders as
                    # visible bands.  Clipping to zero instead gives a
                    # clean dark viridis-zero in the low-density regions
                    # and avoids any interpolation artifacts at the
                    # tube-mask boundary.  Cells outside the tube remain
                    # NaN (the tube mask is honored), but cells inside
                    # the tube are always finite.
                    rho_show = np.where(np.isfinite(rho),
                                         np.maximum(rho, 0.0), np.nan)
                    show_panel = True
                else:
                    # For the NEGATIVE panel, all four methods are
                    # eligible: the GP surrogate is a signed sum of
                    # Gaussians (Σ_j α_j k_j), so PBME and midpoint can
                    # in principle produce small negativity from
                    # cancellation between adjacent α coefficients
                    # of opposite sign.  We treat them on the same
                    # footing as SE/QCLE.
                    # Per-panel test: is there ENOUGH negativity to
                    # be physically meaningful?  If the peak |ρ⁻|
                    # in this panel is below 0.2% of the peak ρ⁺,
                    # then everything we'd render is just machine-
                    # eps numerical noise.  Skip the panel rather
                    # than render speckle.
                    finite = rho[np.isfinite(rho)]
                    rho_pos_max = float(finite[finite > 0].max()) \
                                   if (finite > 0).any() else 0.0
                    neg_arr = -rho[np.isfinite(rho) & (rho < 0)]
                    neg_peak_here = (float(np.quantile(neg_arr, vmax_quantile))
                                      if neg_arr.size else 0.0)

                    if rho_pos_max > 0 and \
                       neg_peak_here < 0.002 * rho_pos_max:
                        show_panel = False
                    else:
                        rho_show = np.where(
                            np.isfinite(rho),
                            np.maximum(-rho, 0.0), np.nan)
                        show_panel = True

                if show_panel:
                    ax.pcolormesh(R_grid_c, P_grid_c, rho_show.T,
                                   cmap=cmap_name, vmin=0.0, vmax=v_max,
                                   shading="gouraud",
                                   edgecolors="none", rasterized=True,
                                   antialiased=False)
                else:
                    # Render a labeled placeholder so the row remains
                    # in the figure and the reader sees why it's empty.
                    if sign == "neg":
                        msg = "(below noise floor)"
                    else:
                        msg = "(no positive data)"
                    ax.text(0.5, 0.5, msg,
                             transform=ax.transAxes,
                             ha="center", va="center",
                             color="white", fontsize=8, style="italic")

                ax.set_xlim(*R_windows[col]); ax.set_ylim(*P_windows[col])
                ax.set_facecolor(fc_color)
                if row == n_methods - 1:
                    ax.set_xlabel(r"$R$  [a.u.]")
                else:
                    ax.tick_params(labelbottom=False)
                if col == 0:
                    ax.set_ylabel(rf"$\mathbf{{{m}}}$" + "\n" + r"$P$  [a.u.]")
                else:
                    ax.tick_params(labelleft=False)
                if col == n_t - 1 and show_panel:
                    # Per-row colorbar.  We use the last call to
                    # pcolormesh — Python's late binding captures the
                    # right artist because we render in row order.
                    from matplotlib.cm import ScalarMappable
                    from matplotlib.colors import Normalize
                    sm = ScalarMappable(cmap=cmap_name,
                                         norm=Normalize(vmin=0.0, vmax=v_max))
                    sm.set_array([])
                    cb = plt.colorbar(sm, ax=ax, fraction=0.05,
                                       pad=0.04, location="right")
                    cb.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"] - 2)
                    cb.set_label(cb_label,
                                  fontsize=plt.rcParams["axes.labelsize"] - 2)

        _save_pub(fig, os.path.join(out_dir, filename_root))
        plt.close(fig)
        print(f"  -> {filename_root}.{{pdf,png}}")

    _render_one_sign(
        filename_root="panel_density_2d_positive",
        sign="pos",
        cmap_name="viridis",
        fc_color="#0d0026",
        cb_label=r"$\rho > 0$",
        title_suffix=r"$\rho>0$ component, viridis colormap")
    _render_one_sign(
        filename_root="panel_density_2d_negative",
        sign="neg",
        cmap_name="magma",
        fc_color="#000000",
        cb_label=r"$|\rho|$ where $\rho<0$",
        title_suffix=r"$\rho<0$ component (magnitude), magma colormap; "
                      r"only methods with signed density representations "
                      r"(SE/QCLE Wigner, PBME/midpoint GP surrogate) can show negativity")


def make_individual_diagnostics(
    runs: Dict[str, Dict[str, np.ndarray]],
    out_dir: str,
) -> None:
    """Write one JCP-column PDF+PNG per per-step diagnostic.

    Only the trajectory methods (midpoint, PBME) carry GP diagnostics;
    SE and QCLE are silently skipped for any key they don't have.

    Output layout::

        {out_dir}/
          diagnostics/
            flow_correction/    dz_rms  n_capped  grad_median
            label_integrator/   omega_A_residual  prob_drift  dy_rms
            faithfulness/       ess_alpha  log_kappa  loo_max  loo_rms  pred_rms
    """
    cloud = {m: runs[m] for m in ("midpoint", "PBME") if m in runs}
    if not cloud:
        return
    os.makedirs(out_dir, exist_ok=True)

    def _sub(*parts: str) -> str:
        d = os.path.join(out_dir, *parts)
        os.makedirs(d, exist_ok=True)
        return d

    def _w(key: str, ylabel: str, subdir: str, fname: str, **kw) -> None:
        _plot_one_cmp(cloud, key, ylabel, os.path.join(subdir, fname), **kw)

    # ── flow correction ───────────────────────────────────────────────────────
    d = _sub("flow_correction")
    _w("fc_dz_rms",       r"Flow corr. $\|\Delta z\|_{\mathrm{rms}}$", d, "dz_rms",
       yscale="log")
    _w("fc_n_capped",     r"Flow corr. \# points capped",              d, "n_capped")
    _w("fc_grad_median",  r"Flow corr. median $|\nabla\hat\rho|$",     d, "grad_median",
       yscale="log")

    # ── label integrator ──────────────────────────────────────────────────────
    d = _sub("label_integrator")
    _w("omega_A_residual_norm",
       r"$\|\omega^T A\|$ (before projection)",                         d, "omega_A_residual",
       yscale="log")
    _w("label_probability_drift",
       r"Per-step $\sum_i \omega_i \Delta y_i$",                       d, "prob_drift")
    _w("label_dy_rms",
       r"$\|\Delta y\|_{\mathrm{rms}}$ per step",                      d, "dy_rms",
       yscale="log")

    # ── surrogate faithfulness ────────────────────────────────────────────────
    d = _sub("faithfulness")
    _w("faith_ess_alpha_frac",
       r"$\mathrm{ESS}(\alpha)/N$",                                     d, "ess_alpha",
       hline=0.10, ylim=(0.0, 1.05))
    _w("faith_cond_K_lo_log10",
       r"$\log_{10}\kappa(K)$ (lower bound)",                           d, "log_kappa",
       hline=12.0)
    _w("faith_loo_max",
       r"$\mathrm{LOO}_{\max}$ residual",                              d, "loo_max",
       yscale="log")
    _w("faith_loo_rms",
       r"$\mathrm{LOO}_{\mathrm{rms}}$",                               d, "loo_rms",
       yscale="log")
    _w("faith_predict_rms",
       r"Posterior predictive RMS",                                     d, "pred_rms",
       yscale="log")



def _diag_panel_curve(ax, runs, methods, key: str,
                       title: str, ylabel: str,
                       log: bool = False,
                       ylim: Optional[Tuple[float, float]] = None,
                       hline: Optional[float] = None,
                       hline_label: Optional[str] = None) -> None:
    """One subplot helper used by ``make_individual_diagnostics``."""
    any_data = False
    for m in methods:
        r = runs[m]
        if key not in r:
            continue
        v = np.asarray(r[key], dtype=np.float64).ravel()
        t = np.asarray(r["t"], dtype=np.float64).ravel()
        # NPZ time-series and observable time-series may be different
        # lengths (snapshots vs every step) — truncate / align here.
        n = min(t.size, v.size)
        if n < 2:
            continue
        # If log scale, mask non-positive (legitimate zeros, but log-undefined)
        plot_v = v[:n].copy()
        if log:
            plot_v = np.where(plot_v > 0, plot_v, np.nan)
        ax.plot(t[:n], plot_v, **_curve(m))
        any_data = True
    ax.set_title("")
    ax.set_xlabel(r"Time  [a.u.]")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if hline is not None:
        ax.axhline(hline, color="0.5", lw=0.7, ls=":",
                    label=hline_label)
        if hline_label:
            ax.legend(loc="best", fontsize=7)
    if not any_data:
        ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes,
                 ha="center", va="center", color="0.6", fontsize=8)


def make_individual_figures(
    runs: Dict[str, Dict[str, np.ndarray]],
    out_dir: str,
    P0_label: float = 40.0,
) -> None:
    """Write one JCP-column PDF+PNG per observable — no multi-panel figures.

    Output layout::

        {out_dir}/
          populations/   P0  P1  Psum
          coherences/    coh_re  coh_im  coh_abs
          nuclear/       R_mean  P_mean  R_var  P_var
          conservation/  trace  energy_drift
    """
    def _sub(*parts: str) -> str:
        d = os.path.join(out_dir, *parts)
        os.makedirs(d, exist_ok=True)
        return d

    def _w(key: str, ylabel: str, subdir: str, fname: str, **kw) -> None:
        _plot_one_cmp(runs, key, ylabel,
                      os.path.join(subdir, fname), **kw)

    # ── populations ──────────────────────────────────────────────────────────
    d = _sub("populations")
    _w("P0", r"$P_0(t)$",  d, "P0",  ylim=(0.0, 1.05))
    _w("P1", r"$P_1(t)$",  d, "P1",  ylim=(0.0, 1.05))
    # trace = P0 + P1 norm check
    _plot_one_cmp(
        runs, "trace", r"$\mathrm{Tr}\,\rho(t)$",
        os.path.join(d, "Psum"), hline=1.0, ylim=(0.95, 1.05),
    )

    # ── coherences ───────────────────────────────────────────────────────────
    d = _sub("coherences")
    _w("coh_re",  r"$\mathrm{Re}\,\rho_{01}(t)$",   d, "coh_re",  hline=0.0)
    _w("coh_im",  r"$\mathrm{Im}\,\rho_{01}(t)$",   d, "coh_im",  hline=0.0)
    _w("coh_abs", r"$|\rho_{01}(t)|$",               d, "coh_abs", ylim=(0.0, None))

    # ── nuclear moments ───────────────────────────────────────────────────────
    d = _sub("nuclear")
    _w("R_mean", r"$\langle R\rangle$ [a.u.]",         d, "R_mean")
    _w("P_mean", r"$\langle P\rangle$ [a.u.]",         d, "P_mean")
    _w("R_var",  r"$\mathrm{Var}(R)$ [a.u.$^2$]",     d, "R_var")
    _w("P_var",  r"$\mathrm{Var}(P)$ [a.u.$^2$]",     d, "P_var")

    # ── conservation ─────────────────────────────────────────────────────────
    d = _sub("conservation")
    _w("trace", r"$\mathrm{Tr}\,\rho(t)$",
       d, "trace", hline=1.0)

    def _edrift(t, y, name):
        E0 = y[0] if len(y) else 0.0
        return (y - E0) / max(abs(E0), 1e-30)

    _plot_one_cmp(
        runs, "energy",
        r"$(\langle H\rangle - \langle H\rangle_0)\,/\,|\langle H\rangle_0|$",
        os.path.join(d, "energy_drift"),
        transform=_edrift, hline=0.0,
    )



# ============================================================================
# Per-P0 runner
# ============================================================================

def _run_one_p0(
    P0: float,
    gp_dir: str,
    out_base: str,
    args,
) -> None:
    """Run SE + QCLE, load GP NPZ files, and write all figures for one P0.

    The output goes into ``{out_base}/P0_{P0:g}/``.
    """
    label = f"P0_{P0:g}"
    out_dir = os.path.join(out_base, label)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[compare] P0 = {P0:g}   output -> {out_dir}")
    print(f"{'='*60}")

    runs: Dict[str, Dict[str, np.ndarray]] = {}

    # ── GP trajectory runs ────────────────────────────────────────────────────
    npz_mid  = os.path.join(gp_dir, "midpoint.npz")
    npz_pbme = os.path.join(gp_dir, "pbme.npz")

    if os.path.exists(npz_mid):
        print(f"[compare] Loading midpoint (GP-RKHS-MInt) ...")
        runs["midpoint"] = load_collector_run(npz_mid)
        t_arr = runs["midpoint"]["t"]
        print(f"          n_obs={len(t_arr)}, t∈[{t_arr[0]:.1f}, {t_arr[-1]:.1f}]")
    else:
        print(f"[compare] WARNING: no midpoint.npz in {gp_dir}")

    if os.path.exists(npz_pbme) and not args.no_pbme:
        print(f"[compare] Loading PBME ...")
        runs["PBME"] = load_collector_run(npz_pbme)

    # ── reference methods ─────────────────────────────────────────────────────
    # Resolve the reference time grid per P0.  If midpoint/PBME are present,
    # this uses their actual saved Collector time axis, so SE/QCLE land on the
    # same physical endpoint even when run.py used automatic dt scaling.
    dt_eff, n_steps_eff, T_eff, grid_source = _infer_time_grid_from_runs(runs, args, P0)
    print(f"[compare] time grid: dt={dt_eff:.8g}, n_steps={n_steps_eff}, "
          f"T={T_eff:.8g} a.u.  ({grid_source})")

    density_times: List[float] = (
        [] if args.no_density
        else _parse_density_times(
            args.density_times, R0=args.R0, P0=P0, mass=args.mass,
            T_final=T_eff, include_final=args.include_final_density)
    )

    if not args.no_se:
        print(f"\n[compare] Running SE (TDSE) for P0={P0:g} ...")
        runs["SE"] = run_tdse(
            R0=args.R0, P0=P0, sigma_R=args.sigma_R,
            dt=dt_eff, n_steps=n_steps_eff,
            init_state=args.init_state,
            mass=args.mass, hbar=args.hbar,
            save_every=args.se_save_every,
            t_snapshots=density_times,
        )

    if not args.no_qcle:
        print(f"\n[compare] Running QCLE for P0={P0:g} ...")
        qcle_params = _qcle_params_for_p0(
            P0=P0, R0=args.R0, sigma_R=args.sigma_R,
            n_steps=n_steps_eff, dt=dt_eff,
            mass=args.mass, hbar=args.hbar,
        )
        print(f"          QCLE grid: n_R={qcle_params.n_R}, n_P={qcle_params.n_P}  "
              f"R∈[{qcle_params.R_min:.1f},{qcle_params.R_max:.1f}]  "
              f"P∈[{qcle_params.P_min:.1f},{qcle_params.P_max:.1f}]")
        runs["QCLE"] = run_qcle(
            R0=args.R0, P0=P0, sigma_R=args.sigma_R,
            dt=dt_eff, n_steps=n_steps_eff,
            init_state=args.init_state,
            mass=args.mass, hbar=args.hbar,
            save_every=args.qcle_save_every,
            t_snapshots=density_times,
            qcle_params=qcle_params,
        )

    # canonical method order for legends
    ordered: Dict[str, Dict[str, np.ndarray]] = {
        k: runs[k] for k in ("SE", "QCLE", "PBME", "midpoint") if k in runs
    }

    # ── figures ───────────────────────────────────────────────────────────────
    print(f"\n[compare] Writing figures: {', '.join(ordered.keys())}")
    make_individual_figures(ordered, out_dir, P0_label=P0)
    make_individual_diagnostics(ordered, out_dir)

    if density_times:
        print(f"[compare] 1D density marginals at t = {density_times} ...")
        panel_density_marginals(
            ordered, gp_dir=gp_dir, out_dir=os.path.join(out_dir, "density"),
            t_targets=density_times,
            R0=args.R0, P0=P0, sigma_R=args.sigma_R, hbar=args.hbar,
        )
        print(f"[compare] Reference-style phase-space panels (2D + marginals) ...")
        panel_phasespace_all_methods(
            ordered, gp_dir=gp_dir,
            out_dir=os.path.join(out_dir, "phasespace"),
            t_targets=density_times,
            R0=args.R0, P0=P0, sigma_R=args.sigma_R, hbar=args.hbar,
            mass=args.mass,
        )
        print(f"[compare] 2D phase-space density sign-split panels ...")
        panel_density_marginals_2d(
            ordered, gp_dir=gp_dir, out_dir=os.path.join(out_dir, "density"),
            t_targets=density_times,
            R0=args.R0, P0=P0, sigma_R=args.sigma_R, hbar=args.hbar,
        )

    print(f"[compare] P0={P0:g} done → {out_dir}")


# ============================================================================
# CLI
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Four-method comparison (SE / QCLE / PBME / midpoint) "
                    "on the Tully DAC model.  Supports multiple initial momenta.")
    parser.add_argument("gp_dir",
                        help="GP run directory containing midpoint.npz / pbme.npz.  "
                             "When --p0-list is used with multiple values, this must "
                             "point to a directory that contains one sub-folder per "
                             "P0 value named P0_{value}/ (e.g. P0_10/, P0_40/), or "
                             "a flat directory used for all P0 values.")
    parser.add_argument("--p0-list", type=float, nargs="+",
                        default=[40.0],
                        metavar="P0",
                        help="One or more initial momenta (a.u.).  "
                             "Default: 40.  Example: --p0-list 10 20 40 100")
    parser.add_argument("--R0",        type=float, default=-15.0)
    parser.add_argument("--sigma_R",   type=float, default=1.0)
    parser.add_argument("--mass",      type=float, default=MASS_DEFAULT)
    parser.add_argument("--hbar",      type=float, default=HBAR_DEFAULT)
    parser.add_argument("--init_state", type=int,  default=0)
    parser.add_argument("--dt",         type=float, default=0.5,
                        help="Maximum reference timestep [a.u.]. If GP Collector data are present, "
                             "dt is inferred from the saved run unless --ignore-gp-time-grid is used.")
    parser.add_argument("--n_steps",    type=int,   default=5600,
                        help="Legacy reference step count used only when no GP time grid, --t_final, or --scattering-cycles is supplied.")
    parser.add_argument("--t_final", type=float, default=None,
                        help="Reference final time in a.u. Used only when no GP time grid is inferred or when --ignore-gp-time-grid is set.")
    parser.add_argument("--scattering_cycles", type=float, default=None,
                        help="Alternative final time: T=scattering_cycles*M*abs(R0)/abs(P0). Ignored by --t_final.")
    parser.add_argument("--ignore-gp-time-grid", action="store_true",
                        help="Do not infer dt/T from midpoint.npz or pbme.npz; use CLI time-grid controls instead.")
    parser.add_argument("--auto_dt", dest="auto_dt", action="store_true",
                        help="When using CLI time-grid controls, reduce dt for low P0 via momentum scaling.")
    parser.add_argument("--no_auto_dt", dest="auto_dt", action="store_false")
    parser.set_defaults(auto_dt=True)
    parser.add_argument("--auto_dt_ref", type=float, default=0.5)
    parser.add_argument("--auto_dt_ref_p0", type=float, default=40.0)
    parser.add_argument("--auto_dt_power", type=float, default=1.0)
    parser.add_argument("--dt_min", type=float, default=0.02)
    parser.add_argument("--dt_max", type=float, default=0.5)
    parser.add_argument("--se-save-every",   type=int, default=4)
    parser.add_argument("--qcle-save-every", type=int, default=4)
    parser.add_argument("--talk",       action="store_true",
                        help="Use TALK styling (sans-serif, larger fonts).")
    parser.add_argument("--no-qcle",   action="store_true", help="Skip QCLE.")
    parser.add_argument("--no-se",     action="store_true", help="Skip SE.")
    parser.add_argument("--no-pbme",   action="store_true",
                        help="Skip PBME (don't load pbme.npz).")
    parser.add_argument("--density-times", type=str, default="auto",
                        help="Snapshot times for density/phase-space panels. Use 'auto' for "
                             "0, t_c, 2t_c with t_c=M|R0|/|P0|, 'final' for T, "
                             "or a comma-separated list such as '0,1500,3000,6000'.")
    parser.add_argument("--include-final-density", action="store_true",
                        help="Append the resolved final time to the density-times list.")
    parser.add_argument("--no-density", action="store_true",
                        help="Skip density marginal panels.")
    parser.add_argument("--out",        type=str, default=None,
                        help="Base output directory.  Default: "
                             "{gp_dir}/comparison_se_qcle/")
    args = parser.parse_args()

    _apply_rcparams("talk" if args.talk else "journal")

    gp_dir   = os.path.abspath(args.gp_dir)
    out_base = os.path.abspath(
        args.out if args.out else os.path.join(gp_dir, "comparison_se_qcle")
    )
    os.makedirs(out_base, exist_ok=True)

    p0_values = sorted(set(args.p0_list))
    print(f"[compare] GP dir  : {gp_dir}")
    print(f"[compare] Output  : {out_base}")
    print(f"[compare] P0 list : {p0_values}")
    print(f"[compare] requested dt upper bound={args.dt}, legacy n_steps={args.n_steps}")
    if args.t_final is not None:
        print(f"[compare] requested fixed T={args.t_final:g} a.u.")
    elif args.scattering_cycles is not None:
        print(f"[compare] requested scattering_cycles={args.scattering_cycles:g}")
    print(f"[compare] density-times spec: {args.density_times!r}")

    for P0 in p0_values:
        # If each P0 has its own GP sub-directory (P0_10/, P0_20/, etc.),
        # prefer that; otherwise fall back to the common gp_dir.
        candidate = os.path.join(gp_dir, f"P0_{P0:g}")
        run_dir   = candidate if os.path.isdir(candidate) else gp_dir
        _run_one_p0(P0=P0, gp_dir=run_dir, out_base=out_base, args=args)

    print(f"\n[compare] All P0 values done.  Figures in: {out_base}")


if __name__ == "__main__":
    main()
