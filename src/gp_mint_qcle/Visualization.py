from __future__ import annotations

"""
Visualization.py
================

Physics-category plotting for PBME vs midpoint QCLE comparisons.

Each plotting function takes a dict `{scheme_name: loaded_run_dict}` and
writes ONE figure per call to a user-specified path.  All categories are
also exposed through `produce_all_comparison_figures(...)` which saves the
full standard set at once.

Categories
----------
*   conservation        : normalization, trace, energy residuals
*   populations         : diabatic P_α and adiabatic P^{ad}_α
*   coherences          : Re/Im ρ^{el}_{01}, |ρ^{el}_{01}|
*   nuclear             : ⟨R⟩, ⟨P⟩, ⟨R²⟩, ⟨P²⟩, Var(R), Var(P)
*   mapping_moments     : ⟨r_α²⟩, ⟨p_α²⟩, cross moments
*   local_energy        : density-based energy from the GP moments
*   correction          : ‖Q‖_RMS, ‖Q‖_max, ΣΔ|y|
*   fit_quality         : fit RMS on support points, GP hyperparameters
*   density_slice_2d    : reconstruct ρ̂ at a snapshot, 2D plane;
                          includes analytic marginal of the R-P plane.

Each ρ-slice function can plot the PBME and midpoint surrogates on the
same figure (side-by-side panels) with a shared color scale.

The R-P "slice" issue
---------------------
A direct slice ρ̂(R, P, r_0=r_1=p_0=p_1=0) lands in the SEO negative lobe
because  w_α(0,0,0,0) = -1  and  ρ = q · w_λ ≤ 0  at the mapping origin.
The classically POSITIVE view is the MARGINAL

    ρ̂_cl(R, P)  ≡  ∫ ρ̂(R, P, r, p) dr dp.

Under the ARD-RBF kernel this integral factorizes analytically:

    ρ̂_cl(R, P) = Σ_i α_i σ_f²
                  · (√(2π) ℓ_R)(√(2π) ℓ_P)
                  · Π_{d∈map} √(2π) ℓ_d
                  · exp(-½ (R - Z_{i,R})²/ℓ_R² - ½ (P - Z_{i,P})²/ℓ_P²).

The mapping-direction (r_α, p_α) slice, in contrast, is physically
expected to be signed.
"""

from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import os
import warnings
import numpy as np

# matplotlib's PDF backend on Windows reads file creation/modification
# timestamps to embed in PDF metadata.  On Windows (and on network or
# virtual drives) freshly-written files can report epoch-0 timestamps,
# generating one noisy warning per savefig call.  Suppress it globally.
warnings.filterwarnings(
    "ignore",
    message=r".*timestamp seems very low.*",
    category=UserWarning,
)
from numpy.typing import NDArray

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .GP_Density import GPDensity, GPDensityConfig
from .Mint import D, PBMEMIntDynamics
from .Collector import Collector

FloatArray = NDArray[np.float64]


# ============================================================================
# Publication style — Journal of Chemical Physics / research-quality figures
# ============================================================================
#
# JCP single-column : 3.375 in   double-column : 6.75 in
# Research report   : up to 6.0 in single / 6.5 in full-page
# Font              : sans-serif (Helvetica/Arial), ≥8 pt axis labels
# DPI               : 300 (raster); PDF preferred for vector output
# Ticks             : inward, with minor ticks on all axes
# Grid              : off  (clean publication look)
# Spines            : all four edges, 0.75 pt weight
# ============================================================================

# Convenience width constants (inches) — use these for figsize arguments.
_W1 = 3.375   # JCP single column
_W15 = 5.0    # JCP 1.5 column  (common for square-ish panels)
_W2 = 6.75    # JCP double column  /  UofT full-width figure
_WU = 6.0     # Research-report comfortable single figure width

_BASE_FONT   = 9.0   # axis labels, tick labels (min 8 pt for JCP)
_TITLE_FONT  = 9.0   # panel/figure titles
_LEGEND_FONT = 8.0   # legend entries
_LABEL_FONT  = 9.0   # x/y axis labels
_TICK_FONT   = 8.0   # tick labels

_publication_rc = {
    # --- Figure ---
    "figure.dpi":            150,          # screen preview
    "savefig.dpi":           300,          # saved files
    "figure.facecolor":      "white",
    "figure.edgecolor":      "white",

    # --- Font ---
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":             _BASE_FONT,
    "axes.titlesize":        _TITLE_FONT,
    "axes.labelsize":        _LABEL_FONT,
    "xtick.labelsize":       _TICK_FONT,
    "ytick.labelsize":       _TICK_FONT,
    "legend.fontsize":       _LEGEND_FONT,
    "legend.title_fontsize": _LEGEND_FONT,

    # --- Axes ---
    "axes.linewidth":        0.75,
    "axes.spines.top":       True,
    "axes.spines.right":     True,
    "axes.facecolor":        "white",
    "axes.grid":             False,

    # --- Ticks: inward, both major & minor ---
    "xtick.direction":       "in",
    "ytick.direction":       "in",
    "xtick.major.size":      3.5,
    "ytick.major.size":      3.5,
    "xtick.minor.size":      1.8,
    "ytick.minor.size":      1.8,
    "xtick.major.width":     0.75,
    "ytick.major.width":     0.75,
    "xtick.minor.width":     0.50,
    "ytick.minor.width":     0.50,
    "xtick.top":             True,    # ticks on all four sides
    "ytick.right":           True,
    "xtick.minor.visible":   True,
    "ytick.minor.visible":   True,

    # --- Lines ---
    "lines.linewidth":       1.25,
    "lines.markersize":      4.0,

    # --- Legend ---
    "legend.frameon":        True,
    "legend.framealpha":     0.85,
    "legend.edgecolor":      "0.75",
    "legend.fancybox":       False,
    "legend.borderpad":      0.4,
    "legend.labelspacing":   0.3,
    "legend.handlelength":   2.0,
    "legend.handletextpad":  0.5,
    "legend.columnspacing":  1.0,

    # --- Saving ---
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.02,
    "savefig.format":        "pdf",      # vector by default
    "pdf.fonttype":          42,         # embeds fonts (Type 1 → TrueType)
    "ps.fonttype":           42,
}

matplotlib.rcParams.update(_publication_rc)



_COLORS = {
    "pbme":     "#0072B2",   # deep blue
    "midpoint": "#E69F00",   # orange
    "se":       "#000000",   # black  (exact/reference)
    "qcle":     "#009E73",   # teal-green
}

# Second visual cue: linestyle is independent of colour so traces are
# distinguishable in greyscale and for colorblind readers.
_LINESTYLES = {
    "pbme":     "-",
    "midpoint": "--",
    "se":       "-",
    "qcle":     ":",
}

# Line weights: 1.5 pt for publication; reference curve slightly heavier.
_LINEWIDTHS = {
    "pbme":     1.5,
    "midpoint": 1.5,
    "se":       2.0,
    "qcle":     1.5,
}


def _scheme_kw(name: str, lw: Optional[float] = None, **extra) -> dict:
    """
    Return a dict of matplotlib ``plot`` kwargs for the given scheme name.

    Encapsulates color + linestyle + linewidth so every plot call uses the
    same mapping and the legend is consistent across all figures.

    Parameters
    ----------
    name : scheme name string (case-insensitive; falls back to grey for unknowns)
    lw   : override linewidth (uses _LINEWIDTHS default when None)
    **extra : additional kwargs forwarded verbatim (e.g. alpha, zorder)
    """
    lname = str(name).lower()
    kw = {
        "color":     _COLORS.get(lname, "#888880"),
        "linestyle": _LINESTYLES.get(lname, "-"),
        "linewidth": lw if lw is not None else _LINEWIDTHS.get(lname, 1.8),
    }
    kw.update(extra)
    return kw


def _setup(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Apply JCP / research-quality axis styling to *ax*."""
    if title:
        ax.set_title(title, fontsize=_TITLE_FONT, pad=3)
    ax.set_xlabel(xlabel, fontsize=_LABEL_FONT, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=_LABEL_FONT, labelpad=3)
    ax.tick_params(axis="both", which="major",
                   labelsize=_TICK_FONT, direction="in",
                   length=3.5, width=0.75)
    ax.tick_params(axis="both", which="minor",
                   direction="in", length=1.8, width=0.50)
    # AutoMinorLocator is incompatible with log scales; guard both axes.
    if ax.get_xscale() == "linear":
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    if ax.get_yscale() == "linear":
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)


def _save(fig: plt.Figure, path: Optional[str]) -> None:
    """Save *fig* at publication/research quality.

    Writes companion PDF (vector, TrueType-embedded) and 300-dpi PNG.
    Timestamp UserWarnings from the matplotlib PDF backend are suppressed
    here because they fire on every savefig call on Windows (epoch-0 mtime).
    """
    if path is not None:
        base, _ = os.path.splitext(path)
        os.makedirs(os.path.dirname(os.path.abspath(base)) or ".", exist_ok=True)
        kw = dict(dpi=300, bbox_inches="tight", pad_inches=0.02)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.savefig(base + ".pdf", **kw)
            fig.savefig(base + ".png", **kw)
    plt.close(fig)


# =============================================================================
# Loaders
# =============================================================================

def load_run(path_no_ext: str) -> Dict:
    """Thin wrapper around Collector.load for convenience."""
    return Collector.load(path_no_ext)


def _truth_series(name: str, arrays: Dict[str, FloatArray], kind: str) -> FloatArray:
    """
    Choose the physically primary time series for each observable kind.

    Population policy (revised)
    ---------------------------
    lw_P0 / lw_P1  (label-weighted IS, w_eff = y(t)/q(z^0)):
        PRIMARY for ALL schemes.
        For PBME:    reduces identically to cloud_weighted_P0 (same formula).
        For midpoint: captures the QCLE correction through updated y_i(t).
        Available when initial_proposal_density was stored (seo_signed mode).

    cloud_weighted_P0 (Liouville IS, w = initial_weight, self-normalized):
        Fallback when lw_* is unavailable (e.g. focused sampling).
        Correct for PBME; gives PBME-only dynamics for midpoint (no QCLE).
        DO NOT splice this with dp_P0 — they use different normalisations.

    dp_P0 (cloud Liouville, unnormalized IS pre-fix / self-normalized post-fix):
        Second fallback.  Same Liouville physics as cloud_weighted_P0.

    Previous design used _splice_initial which mixed cloud_weighted_P0 at t=0
    with dp_P0 at t>0.  Those two quantities differed by the constant factor
    mean(w) ≠ 1, creating an artificial step-1 discontinuity.  That splice is
    removed; a single consistent series is used at all steps.

    Energy / trace policy
    ---------------------
    PBME: cloud-weighted MC (stable, no GP dependence).
    Others: GP kernel-integral km_* (includes QCLE correction, KKT-constrained).
    """
    lname = str(name).lower()

    def _prefer(*keys: str) -> FloatArray:
        """Return the first key that exists in arrays."""
        for k in keys:
            if k in arrays:
                return np.asarray(arrays[k], dtype=np.float64)
        raise KeyError(f"None of {keys} found in arrays.")

    if kind == "energy":
        # Prefer self-normalised lw_energy; fall back to raw (PBME only where
        # Σωy≈1 holds exactly) then GP kernel integral.
        return _prefer("lw_energy", "cloud_weighted_energy", "km_energy")

    if kind == "trace":
        return _prefer("lw_trace", "cloud_weighted_trace", "km_trace")

    if kind == "P0":
        # lw_P0: correct for PBME (=cloud) and midpoint (QCLE-corrected).
        # cloud_weighted_P0: correct for PBME; PBME-only for midpoint.
        # Never splice different estimators across time steps.
        return _prefer("lw_P0", "cloud_weighted_P0", "dp_P0")

    if kind == "P1":
        return _prefer("lw_P1", "cloud_weighted_P1", "dp_P1")

    raise ValueError(f"Unknown truth-series kind: {kind!r}")


def _mc_series(arrays: Dict[str, FloatArray], kind: str) -> Optional[FloatArray]:
    """Return self-normalised label-weighted (lw_*) series, falling back to
    raw cloud_weighted_* only when lw_* is absent (legacy NPZ files).

    lw_A = Σ_i ω_i y_i(t) A(z_i(t)) / Σ_i ω_i y_i(t)
    is the proper density-weighted expectation.  The raw cloud_weighted_*
    sums are not divided by Σ ω_i y_i, so they diverge whenever the
    midpoint QCLE labels drift.  lw_* is stable by construction.
    """
    # Prefer self-normalised lw_* produced by _weighted_support_diagnostics
    lw_map = {
        "energy": ("lw_energy",   "cloud_weighted_energy"),
        "trace":  ("lw_trace",    "cloud_weighted_trace"),
        "P0":     ("lw_P0",       "cloud_weighted_P0"),
        "P1":     ("lw_P1",       "cloud_weighted_P1"),
        "coh_re": ("lw_coh_re",   "cloud_weighted_coh_re"),
        "coh_im": ("lw_coh_im",   "cloud_weighted_coh_im"),
    }
    if kind not in lw_map:
        return None
    primary, fallback = lw_map[kind]
    for key in (primary, fallback):
        if key in arrays:
            return np.asarray(arrays[key], dtype=np.float64)
    return None


def _mc_coherence_abs(arrays: Dict[str, FloatArray]) -> Optional[FloatArray]:
    re = _mc_series(arrays, "coh_re")
    im = _mc_series(arrays, "coh_im")
    if re is None or im is None:
        return None
    return np.hypot(re, im)


def _plot_mc_overlay(ax, t: FloatArray, series: Optional[FloatArray],
                     label: str, color) -> None:
    """Dashed overlay for support-cloud Monte Carlo averages."""
    if series is None:
        return
    ax.plot(t, series, label=label, color=color, lw=1.2, ls="--", alpha=0.95)


# =============================================================================
# Conservation (norm, trace, energy residuals)
# =============================================================================


# =============================================================================
# Conservation (norm, trace, energy residuals)
# =============================================================================

def plot_conservation(
    runs: Dict[str, Dict], savepath: Optional[str] = None,
    target_energy: Optional[float] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 3, figsize=(_W2, 2.8))
    for name, run in runs.items():
        a = run["arrays"]
        t = a["t"]
        c  = _COLORS.get(name.lower(), "#888880")
        ls = _LINESTYLES.get(name.lower(), "-")
        lw = _LINEWIDTHS.get(name.lower(), 1.8)
        if "support_label_sum_er" in a:
            norm_resid = np.abs(a["support_label_sum_er"]) + 1e-18
        else:
            norm_resid = np.abs(1.0 - a["km_normalization"]) + 1e-18
        ax[0].semilogy(t, norm_resid, label=name, color=c, ls=ls, lw=lw)
        trace_series = _truth_series(name, a, "trace")
        energy_series = _truth_series(name, a, "energy")
        ax[1].semilogy(t, np.abs(1.0 - trace_series) + 1e-18,
                       label=name, color=c, ls=ls, lw=lw)
        e0 = target_energy if target_energy is not None \
             else float(energy_series[0])
        ax[2].semilogy(t, np.abs(energy_series - e0) + 1e-18,
                       label=name, color=c, ls=ls, lw=lw)

    _setup(ax[0], "Carried-label sum residual", r"$t$",
           r"$|\sum_i y_i(t)-\sum_i y_i(0)|$")
    _setup(ax[1], "Trace residual", r"$t$",
           r"$|1 - \langle c_{00} + c_{11}\rangle|$")
    _setup(ax[2], "Energy residual", r"$t$",
           r"$|\langle H\rangle - E_0|$")
    for a_ in ax: a_.legend(fontsize=_TICK_FONT)
    fig.suptitle("Conservation diagnostics (PBME cloud truth, QCLE GP density)", fontsize=_TITLE_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    _save(fig, savepath)
    return fig


# =============================================================================
# Populations  (diabatic + adiabatic, same figure)
# =============================================================================

def _robust_ylim(arrs, pad_factor: float = 0.15, cap: float = 50.0):
    """Ignore absurd outliers (|x| > cap) so initial-fit transients don't ruin the axis.

    Pass `cap=None` to disable clipping.
    """
    import numpy as np
    vals = np.concatenate([np.asarray(a).ravel() for a in arrs])
    mask = np.isfinite(vals)
    if cap is not None:
        mask &= np.abs(vals) < cap
    vals = vals[mask]
    if vals.size == 0:
        return None
    lo, hi = float(vals.min()), float(vals.max())
    if hi == lo:
        return lo - 1.0, hi + 1.0
    pad = pad_factor * (hi - lo)
    return lo - pad, hi + pad


def plot_populations(runs: Dict[str, Dict],
                     savepath: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(2, 2, figsize=(_W2, 4.5))
    series_P0, series_P1, series_Pad0, series_Pad1 = [], [], [], []
    for name, run in runs.items():
        a = run["arrays"]
        t = a["t"]
        lname = str(name).lower()
        c  = _COLORS.get(lname, "#888880")
        ls = _LINESTYLES.get(lname, "-")
        lw = _LINEWIDTHS.get(lname, 1.8)

        P0_series = _truth_series(name, a, "P0")
        P1_series = _truth_series(name, a, "P1")
        ax[0, 0].plot(t, P0_series, label=f"{name} (density)", color=c, ls=ls, lw=1.8)
        ax[0, 1].plot(t, P1_series, label=f"{name} (density)", color=c, ls=ls, lw=1.8)
        series_P0.append(P0_series)
        series_P1.append(P1_series)

        if lname != "pbme":
            _plot_mc_overlay(ax[0, 0], t, _mc_series(a, "P0"), f"{name} (MC)", c)
            _plot_mc_overlay(ax[0, 1], t, _mc_series(a, "P1"), f"{name} (MC)", c)

        if "ap_Pad_0" in a:
            ax[1, 0].plot(t, a["ap_Pad_0"], label=f"{name} (density)", color=c, ls=ls, lw=1.8)
            ax[1, 1].plot(t, a["ap_Pad_1"], label=f"{name} (density)", color=c, ls=ls, lw=1.8)
            series_Pad0.append(a["ap_Pad_0"])
            series_Pad1.append(a["ap_Pad_1"])

    for axi, ser in [(ax[0, 0], series_P0), (ax[0, 1], series_P1),
                     (ax[1, 0], series_Pad0), (ax[1, 1], series_Pad1)]:
        yl = _robust_ylim(ser, cap=10.0)
        if yl is not None:
            axi.set_ylim(*yl)

    _setup(ax[0, 0], r"Diabatic $P_0$", r"$t$", r"$\langle P_0\rangle$")
    _setup(ax[0, 1], r"Diabatic $P_1$", r"$t$", r"$\langle P_1\rangle$")
    _setup(ax[1, 0], r"Adiabatic $P^{\mathrm{ad}}_0$", r"$t$",
           r"$\langle P^{\mathrm{ad}}_0\rangle$")
    _setup(ax[1, 1], r"Adiabatic $P^{\mathrm{ad}}_1$", r"$t$",
           r"$\langle P^{\mathrm{ad}}_1\rangle$")
    for axr in ax.ravel():
        axr.legend(fontsize=_LEGEND_FONT)
    fig.suptitle("Electronic populations  |  PBME: blue solid  ·  Midpoint: orange dashed  |  density primary, MC cloud overlay", fontsize=_LABEL_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    _save(fig, savepath)
    return fig


# =============================================================================
# Coherences  (Re, Im, |·|)
# =============================================================================


# =============================================================================
# Coherences  (Re, Im, |·|)
# =============================================================================

def plot_coherences(runs: Dict[str, Dict],
                    savepath: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(1, 3, figsize=(_W2, 2.8))
    ser_re, ser_im, ser_abs = [], [], []
    for name, run in runs.items():
        a = run["arrays"]
        t = a["t"]
        lname = str(name).lower()
        c  = _COLORS.get(lname, "#888880")
        ls = _LINESTYLES.get(lname, "-")
        lw = _LINEWIDTHS.get(lname, 1.8)

        ax[0].plot(t, a["dc_coh_re"],  label=f"{name} (density)", color=c, ls=ls, lw=1.8)
        ax[1].plot(t, a["dc_coh_im"],  label=f"{name} (density)", color=c, ls=ls, lw=1.8)
        ax[2].plot(t, a["dc_coh_abs"], label=f"{name} (density)", color=c, ls=ls, lw=1.8)
        ser_re.append(a["dc_coh_re"])
        ser_im.append(a["dc_coh_im"])
        ser_abs.append(a["dc_coh_abs"])

        if lname != "pbme":
            _plot_mc_overlay(ax[0], t, _mc_series(a, "coh_re"), f"{name} (MC)", c)
            _plot_mc_overlay(ax[1], t, _mc_series(a, "coh_im"), f"{name} (MC)", c)
            _plot_mc_overlay(ax[2], t, _mc_coherence_abs(a), f"{name} (MC)", c)

    for axi, ser in [(ax[0], ser_re), (ax[1], ser_im), (ax[2], ser_abs)]:
        yl = _robust_ylim(ser)
        if yl is not None:
            axi.set_ylim(*yl)

    _setup(ax[0], r"Re $\rho^{\mathrm{el}}_{01}$", r"$t$", r"$\mathrm{Re}\,\rho^{\mathrm{el}}_{01}$")
    _setup(ax[1], r"Im $\rho^{\mathrm{el}}_{01}$", r"$t$", r"$\mathrm{Im}\,\rho^{\mathrm{el}}_{01}$")
    _setup(ax[2], r"$|\rho^{\mathrm{el}}_{01}|$", r"$t$", r"$|\rho^{\mathrm{el}}_{01}|$")
    for a_ in ax:
        a_.legend(fontsize=_LEGEND_FONT)
    fig.suptitle("Diabatic electronic coherences (solid = density/GP, dashed = Monte Carlo cloud)", fontsize=_TITLE_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    _save(fig, savepath)
    return fig


# =============================================================================
# Nuclear expectations
# =============================================================================


# =============================================================================
# Nuclear expectations
# =============================================================================

def plot_nuclear(runs: Dict[str, Dict],
                 savepath: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(2, 3, figsize=(_W2, 4.2))
    keys = [("nm_R_mean", r"$\langle R\rangle$",   50.0),
            ("nm_P_mean", r"$\langle P\rangle$",   500.0),
            ("nm_R_sq",   r"$\langle R^2\rangle$", 500.0),
            ("nm_P_sq",   r"$\langle P^2\rangle$", 5.0e4),
            ("nm_R_var",  r"$\mathrm{Var}(R)$",    100.0),
            ("nm_P_var",  r"$\mathrm{Var}(P)$",    500.0)]
    for (axi, (k, lab, cap)) in zip(ax.ravel(), keys):
        series = []
        for name, run in runs.items():
            a = run["arrays"]
            c  = _COLORS.get(name.lower(), "#888880")
            ls = _LINESTYLES.get(name.lower(), "-")
            lw = _LINEWIDTHS.get(name.lower(), 1.8)
            axi.plot(a["t"], a[k], label=name, color=c, ls=ls, lw=1.6)
            series.append(a[k])
        yl = _robust_ylim(series, cap=cap)
        if yl is not None: axi.set_ylim(*yl)
        _setup(axi, lab, r"$t$", lab); axi.legend(fontsize=_TICK_FONT)
    fig.suptitle("Nuclear expectations", fontsize=_TITLE_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    _save(fig, savepath)
    return fig


# =============================================================================
# Mapping moments
# =============================================================================

def plot_mapping_moments(runs: Dict[str, Dict],
                         savepath: Optional[str] = None) -> plt.Figure:
    """Legacy single-panel entry-point; kept for backward compat."""
    figs = plot_mapping_moment_panels(runs, out_dir=None)
    fig = next(iter(figs.values()))
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig


def plot_mapping_moment_panels(
    runs: Dict[str, Dict],
    out_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    One figure per mapping quadratic moment.

    Returns {moment_key -> Figure}.  When *out_dir* is given each figure
    is written to {out_dir}/fig_qm_{key}.png.
    """
    moment_specs = [
        ("r0_sq", "qm_r0_sq", "$\\langle r_0^2\\rangle$"),
        ("r1_sq", "qm_r1_sq", "$\\langle r_1^2\\rangle$"),
        ("p0_sq", "qm_p0_sq", "$\\langle p_0^2\\rangle$"),
        ("p1_sq", "qm_p1_sq", "$\\langle p_1^2\\rangle$"),
        ("r0_r1", "qm_r0_r1", "$\\langle r_0 r_1\\rangle$"),
        ("p0_p1", "qm_p0_p1", "$\\langle p_0 p_1\\rangle$"),
    ]
    panels: Dict[str, plt.Figure] = {}
    for key, arr_key, label in moment_specs:
        fig, ax = plt.subplots(figsize=(_W15, 2.6))
        series = []
        for name, run in runs.items():
            a = run["arrays"]
            c  = _COLORS.get(name.lower(), "#888880")
            ls = _LINESTYLES.get(name.lower(), "-")
            lw = _LINEWIDTHS.get(name.lower(), 1.8)
            if arr_key in a:
                ax.plot(a["t"], a[arr_key], label=name, color=c, ls=ls, lw=1.6)
                series.append(a[arr_key])
        yl = _robust_ylim(series, cap=20.0)
        if yl is not None:
            ax.set_ylim(*yl)
        _setup(ax, label, r"$t$", label)
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=_TICK_FONT)
        fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
        if out_dir:
            fig.savefig(os.path.join(out_dir, f"fig_qm_{key}.png"),
                        dpi=300, bbox_inches="tight")
        panels[key] = fig
    return panels


# =============================================================================
# Density-based energy (from analytic GP integrals)
# =============================================================================

def plot_local_energy(runs: Dict[str, Dict],
                      savepath: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4))
    for name, run in runs.items():
        a = run["arrays"]
        t = a["t"]
        lname = str(name).lower()
        c  = _COLORS.get(lname, "#888880")
        ls = _LINESTYLES.get(lname, "-")
        lw = _LINEWIDTHS.get(lname, 1.8)
        energy_series = _truth_series(name, a, "energy")
        e0 = float(energy_series[0])
        ax[0].plot(t, energy_series, label=f"{name} (density)", color=c, ls=ls, lw=1.8)
        ax[1].semilogy(t, np.abs(energy_series - e0) + 1e-18,
                       label=f"{name} (density)", color=c, ls=ls, lw=1.8)

        if lname != "pbme":
            mc_energy = _mc_series(a, "energy")
            _plot_mc_overlay(ax[0], t, mc_energy, f"{name} (MC)", c)
            if mc_energy is not None and np.size(mc_energy):
                ax[1].semilogy(t, np.abs(mc_energy - float(mc_energy[0])) + 1e-18,
                               label=f"{name} (MC)", color=c, lw=1.2, ls="--", alpha=0.95)

    _setup(ax[0], r"Density energy $\langle H\rangle$", r"$t$", r"$\langle H\rangle$")
    _setup(ax[1], r"Energy residual", r"$t$", r"$|\langle H\rangle - E_0|$")
    for a_ in ax:
        a_.legend(fontsize=_LEGEND_FONT)
    fig.suptitle("Energy diagnostics (solid = density/GP, dashed = Monte Carlo cloud)", fontsize=_TITLE_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    _save(fig, savepath)
    return fig


# =============================================================================
# Correction magnitude
# =============================================================================


# =============================================================================
# Correction magnitude
# =============================================================================

def plot_correction(runs: Dict[str, Dict],
                    savepath: Optional[str] = None) -> plt.Figure:
    """Legacy single-panel entry-point; kept for backward compat."""
    figs = plot_correction_panels(runs, out_dir=None)
    fig = next(iter(figs.values())) if figs else plt.figure()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig


def plot_correction_panels(
    runs: Dict[str, Dict],
    out_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    One figure per correction diagnostic.  t=0 is excluded because the
    correction is identically zero there (the GP has not yet been updated),
    which would compress the log-scale y-axis around machine epsilon.

    Returns {key -> Figure}.  When *out_dir* is given each figure is
    written to {out_dir}/fig_correction_{key}.png.
    """
    mid_runs = {name: r for name, r in runs.items()
                if name.lower() != "pbme"}
    panels: Dict[str, plt.Figure] = {}

    def _save_panel(fig: plt.Figure, key: str) -> None:
        if out_dir:
            fig.savefig(os.path.join(out_dir, f"fig_correction_{key}.png"),
                        dpi=300, bbox_inches="tight")
        panels[key] = fig

    specs = [
        ("rms",        "cs_q_rms",         r"$\|Q\|_{\mathrm{RMS}}$", r"RMS$(Q)$"),
        ("max",        "cs_q_max",          r"$\max_i |Q_i|$",         r"$\max|Q|$"),
        ("dy_over_y",  "cs_dq_over_y_rms",  r"$\|\Delta y\|/\|y\|$ per step",
                                            r"$\|dt\cdot Q\|/\|y\|$"),
    ]

    for key, arr_key, title, ylabel in specs:
        fig, ax = plt.subplots(figsize=(_W15, 2.6))
        plotted = False
        for name, run in mid_runs.items():
            a = run["arrays"]; t = a["t"]
            c  = _COLORS.get(name.lower(), "#888880")
            ls = _LINESTYLES.get(name.lower(), "-")
            lw = _LINEWIDTHS.get(name.lower(), 1.8)
            if arr_key not in a:
                continue
            # exclude t=0: the correction is trivially zero at the initial step
            mask = t > 0
            if not np.any(mask):
                mask = np.ones(len(t), dtype=bool)
            ax.semilogy(t[mask], np.maximum(a[arr_key][mask], 1e-18),
                        label=name, color=c, ls=ls, lw=1.6)
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "No non-PBME data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
        suptitle = (r"Midpoint extra-Liouvillian correction  "
                    r"(PBME: $Q\equiv 0$, excluded)")
        _setup(ax, title, r"$t$", ylabel)
        ax.set_title(suptitle, fontsize=_LABEL_FONT)
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=_TICK_FONT)
        fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
        _save_panel(fig, key)

    return panels




# =============================================================================
# Signed-weight / cancellation diagnostics
# =============================================================================

def plot_signed_weight_diagnostics(runs: Dict[str, Dict],
                                   savepath: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(1, 3, figsize=(_W2, 2.8))
    for name, run in runs.items():
        a = run["arrays"]; t = a["t"]
        c  = _COLORS.get(name.lower(), "#888880")
        ls = _LINESTYLES.get(name.lower(), "-")
        lw = _LINEWIDTHS.get(name.lower(), 1.8)
        if "sw_ess_frac" in a:
            ax[0].plot(t, a["sw_ess_frac"], label=name, color=c, ls=ls, lw=1.6)
        if "sw_cancel_ratio" in a:
            ax[1].plot(t, a["sw_cancel_ratio"], label=name, color=c, ls=ls, lw=1.6)
        if "sw_neg_frac" in a:
            ax[2].plot(t, a["sw_neg_frac"], label=name, color=c, ls=ls, lw=1.6)

    _setup(ax[0], r"Signed-label ESS fraction", r"$t$", r"$N_{\mathrm{eff}}/N$")
    _setup(ax[1], r"Cancellation ratio", r"$t$", r"$|\sum y_i|/\sum |y_i|$")
    _setup(ax[2], r"Negative-label fraction", r"$t$", r"fraction$(y_i<0)$")
    for a_ in ax: a_.legend(fontsize=_TICK_FONT)
    fig.suptitle("Fit-cloud signed-weight diagnostics", fontsize=_TITLE_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    _save(fig, savepath)
    return fig

def plot_sampling_statistics(runs: Dict[str, Dict],
                             savepath: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 6.8))
    ax = np.asarray(ax)
    for name, run in runs.items():
        a = run["arrays"]; t = a["t"]
        c  = _COLORS.get(name.lower(), "#888880")
        ls = _LINESTYLES.get(name.lower(), "-")
        lw = _LINEWIDTHS.get(name.lower(), 1.8)
        if "cloud_mean_R" in a:
            ax[0, 0].plot(t, a["cloud_mean_R"], label=name, color=c, ls=ls, lw=1.6)
        if "cloud_mean_P" in a:
            ax[0, 1].plot(t, a["cloud_mean_P"], label=name, color=c, ls=ls, lw=1.6)
        if "cloud_var_R" in a:
            ax[1, 0].plot(t, a["cloud_var_R"], label=name, color=c, ls=ls, lw=1.6)
        if "cloud_var_P" in a:
            ax[1, 1].plot(t, a["cloud_var_P"], label=name, color=c, ls=ls, lw=1.6)

    _setup(ax[0, 0], r"Support-cloud mean in $R$", r"$t$", r"$\mu_R$")
    _setup(ax[0, 1], r"Support-cloud mean in $P$", r"$t$", r"$\mu_P$")
    _setup(ax[1, 0], r"Support-cloud variance in $R$", r"$t$", r"Var$(R)$")
    _setup(ax[1, 1], r"Support-cloud variance in $P$", r"$t$", r"Var$(P)$")
    for a_ in ax.ravel():
        a_.legend(fontsize=_TICK_FONT)
    fig.suptitle("Sampling / support-cloud mean and variance diagnostics", fontsize=_TITLE_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    _save(fig, savepath)
    return fig


def plot_density_diff_diagnostics(runs: Dict[str, Dict],
                                  savepath: Optional[str] = None
                                  ) -> plt.Figure:
    """
    Density-difference architecture diagnostics (produced only for runs
    that used --use_density_diff).  Shows:

      [0,0]  ||y(t) - y0||_2         vs t  (how much has y drifted?)
      [0,1]  ||delta||_inf           vs t  (max per-point correction)
      [1,0]  ||alpha_delta||_2 / ||alpha_base||_2   (relative fit magnitude)
      [1,1]  max|alpha_delta| / max|alpha_base|      (max-norm ratio)

    Any run whose snapshots do NOT carry density-diff data (`is_density_diff`
    is False everywhere) is silently skipped.  If no run has diff data at
    all, a single "no data" panel is produced so the figure still renders.
    """
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.2))
    ax = np.asarray(ax)

    any_data = False
    for name, run in runs.items():
        snaps = run.get("snapshots", {}) or {}
        if not snaps:
            continue
        style = {"color": _COLORS.get(name.lower(), "#888880"),
                  "linestyle": _LINESTYLES.get(name.lower(), "-"),
                  "linewidth": 1.8, "alpha": 0.95}

        # Collect the relevant per-snapshot stats
        t_list    = []
        dnorm2    = []    # ||delta||_2
        dmaxabs   = []    # ||delta||_inf
        rel_l2    = []    # ||alpha_delta||_2 / ||alpha_base||_2
        rel_max   = []    # max|alpha_delta| / max|alpha_base|
        for step in sorted(snaps.keys()):
            s = snaps[step]
            if not getattr(s, "is_density_diff", False):
                continue
            if s.delta is None or s.alpha_base is None:
                continue
            any_data = True
            t_list.append(float(s.t))
            d = np.asarray(s.delta, dtype=np.float64).reshape(-1)
            a_b = np.asarray(s.alpha_base, dtype=np.float64).reshape(-1)
            a_d = np.asarray(s.alpha, dtype=np.float64).reshape(-1)

            dnorm2.append(float(np.linalg.norm(d)))
            dmaxabs.append(float(np.max(np.abs(d))))

            norm_b  = float(np.linalg.norm(a_b)) + 1e-300
            max_b   = float(np.max(np.abs(a_b)))  + 1e-300
            rel_l2.append(float(np.linalg.norm(a_d)) / norm_b)
            rel_max.append(float(np.max(np.abs(a_d))) / max_b)

        if not t_list:
            continue
        t_arr = np.asarray(t_list)
        ax[0, 0].semilogy(t_arr, dnorm2,  label=name, **style)
        ax[0, 1].semilogy(t_arr, dmaxabs, label=name, **style)
        ax[1, 0].semilogy(t_arr, rel_l2,  label=name, **style)
        ax[1, 1].semilogy(t_arr, rel_max, label=name, **style)

    if not any_data:
        for a_ in ax.ravel():
            a_.text(0.5, 0.5,
                    "No density-diff snapshots in the loaded runs.\n"
                    "(Run with --use_density_diff to populate these.)",
                    ha="center", va="center", transform=a_.transAxes,
                    fontsize=_LABEL_FONT, color="gray")

    _setup(ax[0, 0], r"$\|y(t) - y_0\|_2$", r"$t$", r"$\ell_2$ norm")
    _setup(ax[0, 1], r"$\|y(t) - y_0\|_\infty$", r"$t$", r"$\ell_\infty$ norm")
    _setup(ax[1, 0], r"$\|\alpha_\delta\|_2 / \|\alpha_{\mathrm{base}}\|_2$",
           r"$t$", "ratio")
    _setup(ax[1, 1], r"$\max|\alpha_\delta| / \max|\alpha_{\mathrm{base}}|$",
           r"$t$", "ratio")
    for a_ in ax.ravel():
        if a_.get_legend_handles_labels()[0]:
            a_.legend(fontsize=_TICK_FONT)
    fig.suptitle(
        "Density-difference diagnostics  "
        r"($\hat\rho = \hat\rho_0 \circ \Phi_{-t} + \hat\delta$)",
        fontsize=_TITLE_FONT,
    )
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    _save(fig, savepath)
    return fig


# =============================================================================
# Fit quality
# =============================================================================

def plot_fit_quality(runs: Dict[str, Dict],
                     savepath: Optional[str] = None) -> plt.Figure:
    """Legacy single-panel entry-point; kept for backward compat."""
    figs = plot_fit_quality_panels(runs, out_dir=None)
    # return the first figure so callers expecting one Figure still work
    fig = next(iter(figs.values()))
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig


def plot_fit_quality_panels(
    runs: Dict[str, Dict],
    out_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    One figure per diagnostic panel.

    Returns {key -> Figure}.  When *out_dir* is given each figure is written
    to  {out_dir}/fig_fit_{key}.png.
    """
    panels: Dict[str, plt.Figure] = {}

    def _save_panel(fig: plt.Figure, key: str) -> None:
        if out_dir:
            fig.savefig(os.path.join(out_dir, f"fig_fit_{key}.png"),
                        dpi=300, bbox_inches="tight")
        panels[key] = fig

    # ── Fit RMS ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(_W15, 2.6))
    for name, run in runs.items():
        a = run["arrays"]; t = a["t"]; c  = _COLORS.get(name.lower(), "#888880")
        ls = _LINESTYLES.get(name.lower(), "-")
        lw = _LINEWIDTHS.get(name.lower(), 1.8)
        ax.semilogy(t, np.maximum(a["fit_rms_on_support"], 1e-18),
                    label=name, color=c, ls=ls, lw=1.6)
    _setup(ax, r"Fit RMS on support", r"$t$", "RMS$(\\hat\\rho(Z_i)-y_i)$")
    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=_TICK_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5); _save_panel(fig, "rms")

    # ── Fit MAE ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(_W15, 2.6))
    plotted = False
    for name, run in runs.items():
        a = run["arrays"]; t = a["t"]; c  = _COLORS.get(name.lower(), "#888880")
        ls = _LINESTYLES.get(name.lower(), "-")
        lw = _LINEWIDTHS.get(name.lower(), 1.8)
        if "gp_fit_mae" in a:
            ax.semilogy(t, np.maximum(np.abs(a["gp_fit_mae"]), 1e-18),
                        label=name, color=c, ls=ls, lw=1.6); plotted = True
    if not plotted: ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes, color="gray")
    _setup(ax, r"Fit MAE on support", r"$t$", "MAE$(\\hat\\rho(Z_i)-y_i)$")
    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=_TICK_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5); _save_panel(fig, "mae")

    # ── Fit R² ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(_W15, 2.6))
    plotted = False
    for name, run in runs.items():
        a = run["arrays"]; t = a["t"]; c  = _COLORS.get(name.lower(), "#888880")
        ls = _LINESTYLES.get(name.lower(), "-")
        lw = _LINEWIDTHS.get(name.lower(), 1.8)
        if "gp_fit_r2" in a:
            ax.plot(t, a["gp_fit_r2"], label=name, color=c, ls=ls, lw=1.6); plotted = True
    if not plotted: ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes, color="gray")
    _setup(ax, r"Fit $R^2$ on support", r"$t$", r"$R^2$")
    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=_TICK_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5); _save_panel(fig, "r2")

    # ── Optimizer loss ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(_W15, 2.6))
    plotted = False
    for name, run in runs.items():
        a = run["arrays"]; t = a["t"]; c  = _COLORS.get(name.lower(), "#888880")
        ls = _LINESTYLES.get(name.lower(), "-")
        lw = _LINEWIDTHS.get(name.lower(), 1.8)
        if "gp_opt_total_loss" in a:
            ax.semilogy(t, np.maximum(np.abs(a["gp_opt_total_loss"]), 1e-18),
                        label=f"{name} total", color=c, ls=ls, lw=1.6)
            if "gp_opt_reg_loss" in a:
                ax.semilogy(t, np.maximum(np.abs(a["gp_opt_reg_loss"]), 1e-18),
                            label=f"{name} reg", color=c, lw=1.0, ls="--")
            plotted = True
    if not plotted: ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes, color="gray")
    _setup(ax, "Optimizer loss", r"$t$", "loss")
    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=_TICK_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5); _save_panel(fig, "optimizer_loss")

    # ── GP hyperparameters ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(_W15, 2.6))
    for name, run in runs.items():
        a = run["arrays"]; t = a["t"]; c  = _COLORS.get(name.lower(), "#888880")
        ls = _LINESTYLES.get(name.lower(), "-")
        lw = _LINEWIDTHS.get(name.lower(), 1.8)
        ax.plot(t, a["sigma_f"], label=f"{name} σ_f(norm)", color=c, ls=ls, lw=1.6)
        ax.plot(t, a["sigma_n"], label=f"{name} σ_n(raw)",  color=c, lw=1.1, ls="--")
        if "sigma_n_normalized" in a:
            ax.plot(t, a["sigma_n_normalized"], label=f"{name} σ_n(norm)",
                    color=c, lw=1.0, ls=":")
    _setup(ax, "GP hyperparameters", r"$t$", "value")
    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=_TICK_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5); _save_panel(fig, "hyperparams")

    # ── Early-stop monitor ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(_W15, 2.6))
    plotted = False
    for name, run in runs.items():
        a = run["arrays"]; t = a["t"]; c  = _COLORS.get(name.lower(), "#888880")
        ls = _LINESTYLES.get(name.lower(), "-")
        lw = _LINEWIDTHS.get(name.lower(), 1.8)
        if "gp_opt_val_mae" in a:
            ax.semilogy(t, np.maximum(np.abs(a["gp_opt_val_mae"]), 1e-18),
                        label=f"{name} val MAE", color=c, ls=ls, lw=1.6)
            if "gp_opt_train_mae" in a:
                ax.semilogy(t, np.maximum(np.abs(a["gp_opt_train_mae"]), 1e-18),
                            label=f"{name} train MAE", color=c, lw=1.0, ls="--")
            plotted = True
    if not plotted: ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes, color="gray")
    _setup(ax, "Early-stop monitor", r"$t$", "MAE")
    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=_TICK_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5); _save_panel(fig, "early_stop")

    # ── Optimizer budget / SNR ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(_W15, 2.6))
    plotted = False
    for name, run in runs.items():
        a = run["arrays"]; t = a["t"]; c  = _COLORS.get(name.lower(), "#888880")
        ls = _LINESTYLES.get(name.lower(), "-")
        lw = _LINEWIDTHS.get(name.lower(), 1.8)
        if "gp_opt_steps" in a:
            ax.plot(t, a["gp_opt_steps"], label=f"{name} opt steps", color=c, ls=ls, lw=1.6)
            if "sigma_f_over_sigma_n" in a:
                ax.plot(t, a["sigma_f_over_sigma_n"],
                        label=f"{name} σ_f/σ_n(norm)", color=c, lw=1.0, ls="--")
            plotted = True
    if not plotted: ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes, color="gray")
    _setup(ax, "Optimizer budget / normalized SNR", r"$t$", "value")
    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=_TICK_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5); _save_panel(fig, "optimizer_budget")

    return panels



# =============================================================================
# Separated analytic-GP vs Monte Carlo observable comparisons
# =============================================================================
#
# Design
# ------
# Each observable family is split into two functions:
#   plot_<family>_analytic(runs, out_dir)  — GP surrogate integrals only
#   plot_<family>_mc(runs, out_dir)        — cloud-weighted MC only
#
# Both functions produce ONE figure per observable, saved individually.
# Each figure overlays PBME (blue) vs Midpoint (orange) so the two schemes
# can be compared on equal footing within the same evaluation method.
#
# Shared helper
# -------------

def _two_scheme_fig(
    runs: Dict[str, Dict],
    arr_key: str,
    title: str,
    ylabel: str,
    out_dir: Optional[str],
    filename: str,
    semilogy: bool = False,
    cap: float = 1e30,
) -> plt.Figure:
    """
    One figure: every scheme in `runs` plotted on the same axes,
    straight comparison PBME vs Midpoint (no MC/analytic mixing).
    """
    fig, ax = plt.subplots(figsize=(_W15, 2.6))
    series = []
    for name, run in runs.items():
        a = run["arrays"]; t = a["t"]
        c  = _COLORS.get(name.lower(), "#888880")
        ls = _LINESTYLES.get(name.lower(), "-")
        lw = _LINEWIDTHS.get(name.lower(), 1.8)
        if arr_key not in a:
            continue
        vals = np.asarray(a[arr_key], dtype=np.float64)
        (ax.semilogy if semilogy else ax.plot)(
            t, np.maximum(np.abs(vals), 1e-18) if semilogy else vals,
            label=name, color=c, ls=ls, lw=lw)
        series.append(vals)
    if not series:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    else:
        yl = _robust_ylim(series, cap=cap)
        if yl is not None:
            ax.set_ylim(*yl)
    _setup(ax, title, r"$t$", ylabel)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=_TICK_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    if out_dir:
        fig.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches="tight")
    return fig


def _two_scheme_fig_prefer(
    runs: Dict[str, Dict],
    arr_keys: Sequence[str],
    title: str,
    ylabel: str,
    out_dir: Optional[str],
    filename: str,
    semilogy: bool = False,
    cap: float = 1e30,
) -> plt.Figure:
    """Like _two_scheme_fig but accepts a priority list of keys.

    For each scheme the first key found in its arrays dict is used.
    This lets callers prefer ``lw_*`` (self-normalised) while silently
    falling back to raw ``cloud_weighted_*`` for legacy NPZ files that
    predate the self-normalisation fix.
    """
    fig, ax = plt.subplots(figsize=(_W15, 2.6))
    series = []
    for name, run in runs.items():
        a = run["arrays"]; t = a["t"]
        c  = _COLORS.get(name.lower(), "#888880")
        ls = _LINESTYLES.get(name.lower(), "-")
        lw = _LINEWIDTHS.get(name.lower(), 1.8)
        chosen = next((k for k in arr_keys if k in a), None)
        if chosen is None:
            continue
        vals = np.asarray(a[chosen], dtype=np.float64)
        (ax.semilogy if semilogy else ax.plot)(
            t, np.maximum(np.abs(vals), 1e-18) if semilogy else vals,
            label=name, color=c, ls=ls, lw=lw)
        series.append(vals)
    if not series:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    else:
        yl = _robust_ylim(series, cap=cap)
        if yl is not None:
            ax.set_ylim(*yl)
    _setup(ax, title, r"$t$", ylabel)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=_TICK_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    if out_dir:
        fig.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Populations — analytic GP
# ──────────────────────────────────────────────────────────────────────────────

def plot_populations_analytic(
    runs: Dict[str, Dict],
    out_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    Per-scheme diabatic and adiabatic population time series.

    Primary panels (lw_*): label-weighted IS estimator.
        For PBME: reduces to the Liouville cloud estimator (w_i = ρ_0/q).
        For midpoint: w_eff_i = y_i(t)/q(z_i^0) captures the QCLE correction
        through the carried density labels, without GP kernel integrals.

    Adiabatic populations (ap_*): GP Gauss-Hermite quadrature in R.
        Inherently GP-integral-based; no cloud equivalent exists.

    GP-integral diagnostic panels (gpi_*): analytic kernel integrals.
        Correct in principle (includes QCLE via alpha), oscillates when fit
        degrades.  Shown separately so oscillations don't pollute the primary.
    """
    panels: Dict[str, plt.Figure] = {}
    specs = [
        ("lw_P0",    "Diabatic $P_0$  — label-weighted IS",
                     r"$\langle P_0\rangle$",        "fig_an_pop_P0_diab.png"),
        ("lw_P1",    "Diabatic $P_1$  — label-weighted IS",
                     r"$\langle P_1\rangle$",        "fig_an_pop_P1_diab.png"),
        ("ap_Pad_0", r"Adiabatic $P^{\mathrm{ad}}_0$  — GP integral",
                     r"$\langle P^{\mathrm{ad}}_0\rangle$", "fig_an_pop_P0_ad.png"),
        ("ap_Pad_1", r"Adiabatic $P^{\mathrm{ad}}_1$  — GP integral",
                     r"$\langle P^{\mathrm{ad}}_1\rangle$", "fig_an_pop_P1_ad.png"),
        ("lw_P_sum", "Trace (diabatic sum)  — label-weighted IS",
                     "$P_0 + P_1$",                   "fig_an_pop_trace.png"),
        # GP kernel-integral diagnostic panels (separate; known to oscillate)
        ("gpi_P0",   "Diabatic $P_0$  — GP kernel integral (diagnostic)",
                     r"$\langle P_0\rangle_{\mathrm{gpi}}$", "fig_gpi_pop_P0_diab.png"),
        ("gpi_P1",   "Diabatic $P_1$  — GP kernel integral (diagnostic)",
                     r"$\langle P_1\rangle_{\mathrm{gpi}}$", "fig_gpi_pop_P1_diab.png"),
    ]
    for key, title, ylabel, fname in specs:
        fig = _two_scheme_fig(runs, key, title, ylabel, out_dir, fname, cap=10.0)
        panels[key] = fig
    return panels


# ──────────────────────────────────────────────────────────────────────────────
# Populations — Monte Carlo
# ──────────────────────────────────────────────────────────────────────────────

def plot_populations_mc(
    runs: Dict[str, Dict],
    out_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    Cloud-weighted Monte Carlo diabatic populations.
    Each panel: PBME (MC) vs Midpoint (MC).
    """
    panels: Dict[str, plt.Figure] = {}
    # Use self-normalised lw_* (prefer) or raw cloud_weighted_* (legacy fallback).
    # lw_P0 = Σ ω_i y_i P0_i / Σ ω_i y_i  — bounded even when y diverges.
    specs = [
        ("lw_P0",    "cloud_weighted_P0",    r"Diabatic $P_0$  — MC cloud (self-norm.)",
                                              r"$\langle P_0\rangle_{\mathrm{MC}}$",
                                              "fig_mc_pop_P0.png"),
        ("lw_P1",    "cloud_weighted_P1",    r"Diabatic $P_1$  — MC cloud (self-norm.)",
                                              r"$\langle P_1\rangle_{\mathrm{MC}}$",
                                              "fig_mc_pop_P1.png"),
        ("lw_trace", "cloud_weighted_trace", "Trace  — MC cloud (self-norm.)",
                                              r"$\langle P_0+P_1\rangle_{\mathrm{MC}}$",
                                              "fig_mc_pop_trace.png"),
    ]
    for primary, fallback, title, ylabel, fname in specs:
        # Pick whichever key is available in each run
        best_key = primary  # will fall back inside _two_scheme_fig_prefer
        fig = _two_scheme_fig_prefer(runs, [primary, fallback], title, ylabel,
                                     out_dir, fname, cap=5.0)
        panels[primary] = fig
    # Keep legacy key for backward compat
    if "lw_P0" in panels:
        panels["cloud_weighted_P0"] = panels["lw_P0"]

    # ─────────────────────────────────────────────────────────────────
    # Combined-populations plot: P0 and P1 from every scheme on one
    # figure, so the PBME-vs-midpoint divergence on each diabat is
    # readable at a glance.
    #
    # Convention:
    #   color     ↔  state   (state 0 = blue, state 1 = red)
    #   linestyle ↔  scheme  (PBME = solid, midpoint = dashed)
    # so each (scheme, state) pair is uniquely identified and the eye
    # naturally groups by state.  Reads "the orange band is excited-
    # state population; PBME and midpoint disagree about its growth".
    # ─────────────────────────────────────────────────────────────────
    combined_fig = _plot_populations_combined(runs, out_dir,
                                              filename="fig_mc_pop_combined.png")
    if combined_fig is not None:
        panels["lw_P_combined"] = combined_fig
    return panels


_STATE_COLORS = {
    0: "#1f77b4",   # tab:blue   — ground
    1: "#d62728",   # tab:red    — excited
}
_SCHEME_LS = {
    "pbme":     "-",
    "midpoint": "--",
}


def _plot_populations_combined(
    runs: Dict[str, Dict],
    out_dir: Optional[str],
    filename: str = "fig_mc_pop_combined.png",
) -> Optional[plt.Figure]:
    r"""
    Plot ⟨P_0⟩ and ⟨P_1⟩ (diabatic, MC self-normalised) for every
    scheme on one set of axes.

    Returns None if no scheme has the necessary keys (e.g. an old NPZ
    without lw_P0 or cloud_weighted_P0).  Otherwise saves to
    out_dir/filename and returns the Figure.

    Visual encoding
    ---------------
    Color  : state   (state 0 = blue, state 1 = red)
    Style  : scheme  (PBME = solid, midpoint = dashed)

    The y-axis is clipped to [-0.05, 1.10] so transient overshoots from
    Q-clipping or sampling noise don't blow out the scale; the bulk
    population dynamics for the Tully dual problem stay in [0, 1].
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    plotted_any = False
    for scheme, run in runs.items():
        a = run["arrays"]
        t = a.get("t", None)
        if t is None:
            continue
        sl = scheme.lower()
        ls = _SCHEME_LS.get(sl, "-")
        for state, primary, fallback in (
            (0, "lw_P0", "cloud_weighted_P0"),
            (1, "lw_P1", "cloud_weighted_P1"),
        ):
            key = primary if primary in a else (fallback if fallback in a else None)
            if key is None:
                continue
            vals = np.asarray(a[key], dtype=np.float64)
            color = _STATE_COLORS.get(state, "#888888")
            ax.plot(t, vals, color=color, linestyle=ls, linewidth=2.0,
                    label=rf"{scheme}  $\langle P_{state}\rangle$")
            plotted_any = True
    if not plotted_any:
        plt.close(fig)
        return None

    ax.set_ylim(-0.05, 1.10)
    ax.axhline(1.0, color="gray", linewidth=0.6, alpha=0.5, zorder=0)
    ax.axhline(0.0, color="gray", linewidth=0.6, alpha=0.5, zorder=0)
    _setup(ax, r"Diabatic populations  $\langle P_\alpha\rangle$  — PBME vs midpoint",
           r"$t$", r"$\langle P_\alpha\rangle_{\mathrm{MC}}$")
    # Two-column legend: pair (scheme, state) entries naturally.
    ax.legend(fontsize=_TICK_FONT, ncol=2, loc="best", framealpha=0.9)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    if out_dir:
        fig.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches="tight")
    return fig



# ──────────────────────────────────────────────────────────────────────────────
# Coherences — analytic GP
# ──────────────────────────────────────────────────────────────────────────────

def plot_coherences_analytic(
    runs: Dict[str, Dict],
    out_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    GP-surrogate diabatic coherences Re, Im, |rho_01|.
    Each panel: PBME (analytic) vs Midpoint (analytic).
    """
    panels: Dict[str, plt.Figure] = {}
    specs = [
        ("dc_coh_re",  r"Re $\rho^{\mathrm{el}}_{01}$  — analytic GP",
                        r"$\mathrm{Re}\,\rho^{\mathrm{el}}_{01}$",
                        "fig_an_coh_re.png"),
        ("dc_coh_im",  r"Im $\rho^{\mathrm{el}}_{01}$  — analytic GP",
                        r"$\mathrm{Im}\,\rho^{\mathrm{el}}_{01}$",
                        "fig_an_coh_im.png"),
        ("dc_coh_abs", r"$|\rho^{\mathrm{el}}_{01}|$  — analytic GP",
                        r"$|\rho^{\mathrm{el}}_{01}|$",
                        "fig_an_coh_abs.png"),
    ]
    for key, title, ylabel, fname in specs:
        fig = _two_scheme_fig(runs, key, title, ylabel, out_dir, fname)
        panels[key] = fig
    return panels


# ──────────────────────────────────────────────────────────────────────────────
# Coherences — Monte Carlo
# ──────────────────────────────────────────────────────────────────────────────

def plot_coherences_mc(
    runs: Dict[str, Dict],
    out_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    Self-normalised MC coherences Re, Im, |rho_01|.
    Prefers lw_coh_re/im (self-normalised by Σω_iy_i) over the raw
    cloud_weighted_coh_re/im which diverge when labels drift.
    """
    panels: Dict[str, plt.Figure] = {}
    for key, prim, fall, title, ylabel, fname in [
        ("coh_re", "lw_coh_re", "cloud_weighted_coh_re",
         r"Re $\rho^{\mathrm{el}}_{01}$  — MC cloud (self-norm.)",
         r"$\mathrm{Re}\,\rho^{\mathrm{el}}_{01,\mathrm{MC}}$",
         "fig_mc_coh_re.png"),
        ("coh_im", "lw_coh_im", "cloud_weighted_coh_im",
         r"Im $\rho^{\mathrm{el}}_{01}$  — MC cloud (self-norm.)",
         r"$\mathrm{Im}\,\rho^{\mathrm{el}}_{01,\mathrm{MC}}$",
         "fig_mc_coh_im.png"),
    ]:
        fig = _two_scheme_fig_prefer(runs, [prim, fall], title, ylabel,
                                     out_dir, fname, cap=3.0)
        panels[key] = fig

    # |rho_01|_MC  — derived from Re² + Im² using lw_coh_* when available
    fig, ax = plt.subplots(figsize=(_W15, 2.6))
    series = []
    for name, run in runs.items():
        a = run["arrays"]; t = a["t"]; c  = _COLORS.get(name.lower(), "#888880")
        ls = _LINESTYLES.get(name.lower(), "-")
        lw = _LINEWIDTHS.get(name.lower(), 1.8)
        re_arr = a.get("lw_coh_re", a.get("cloud_weighted_coh_re", None))
        im_arr = a.get("lw_coh_im", a.get("cloud_weighted_coh_im", None))
        if re_arr is not None and im_arr is not None:
            vals = np.hypot(np.asarray(re_arr), np.asarray(im_arr))
            ax.plot(t, vals, label=name, color=c, ls=ls, lw=lw)
            series.append(vals)
    if not series:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    else:
        yl = _robust_ylim(series, cap=3.0)
        if yl is not None: ax.set_ylim(*yl)
    _setup(ax, r"$|\rho^{\mathrm{el}}_{01}|$  — MC cloud (self-norm.)",
           r"$t$", r"$|\rho^{\mathrm{el}}_{01,\mathrm{MC}}|$")
    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=_TICK_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    if out_dir:
        fig.savefig(os.path.join(out_dir, "fig_mc_coh_abs.png"),
                    dpi=300, bbox_inches="tight")
    panels["coh_abs"] = fig
    return panels



# ──────────────────────────────────────────────────────────────────────────────
# Energy — analytic GP
# ──────────────────────────────────────────────────────────────────────────────

def plot_energy_analytic(
    runs: Dict[str, Dict],
    out_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    GP-surrogate energy and normalization.
    Each panel: PBME (analytic) vs Midpoint (analytic).
    """
    panels: Dict[str, plt.Figure] = {}
    specs = [
        ("km_energy",        r"Energy $\langle H\rangle$  — analytic GP",
                              r"$\langle H\rangle$",         "fig_an_energy.png",     False),
        ("km_normalization",  r"Normalization $\langle 1\rangle$  — analytic GP",
                              r"$\langle 1\rangle$",         "fig_an_norm.png",       False),
        ("km_trace",          r"Trace $\langle c_{00}+c_{11}\rangle$  — analytic GP",
                              r"$\mathrm{tr}\,\rho$",       "fig_an_trace.png",      False),
        ("km_energy_raw",     "Raw energy integral  — analytic GP",
                              r"$A_{\mathrm{energy}}\cdot\alpha$", "fig_an_energy_raw.png", False),
    ]
    for key, title, ylabel, fname, slog in specs:
        fig = _two_scheme_fig(runs, key, title, ylabel, out_dir, fname, semilogy=slog)
        panels[key] = fig

    # Energy residual |<H> - E0| on log scale
    fig, ax = plt.subplots(figsize=(_W15, 2.6))
    for name, run in runs.items():
        a = run["arrays"]; t = a["t"]; c  = _COLORS.get(name.lower(), "#888880")
        ls = _LINESTYLES.get(name.lower(), "-")
        lw = _LINEWIDTHS.get(name.lower(), 1.8)
        if "km_energy" not in a:
            continue
        E = np.asarray(a["km_energy"], dtype=np.float64)
        E0 = float(E[0])
        ax.semilogy(t, np.abs(E - E0) + 1e-18, label=name, color=c, ls=ls, lw=lw)
    _setup(ax, r"Energy residual $|\langle H\rangle - E_0|$  — analytic GP",
           r"$t$", r"$|\langle H\rangle - E_0|$")
    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=_TICK_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    if out_dir:
        fig.savefig(os.path.join(out_dir, "fig_an_energy_resid.png"),
                    dpi=300, bbox_inches="tight")
    panels["km_energy_resid"] = fig
    return panels


# ──────────────────────────────────────────────────────────────────────────────
# Energy — Monte Carlo
# ──────────────────────────────────────────────────────────────────────────────

def plot_energy_mc(
    runs: Dict[str, Dict],
    out_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    Cloud-weighted MC energy and trace.
    Each panel: PBME (MC) vs Midpoint (MC).
    """
    panels: Dict[str, plt.Figure] = {}
    specs = [
        (["lw_energy",  "cloud_weighted_energy"], r"Energy $\langle H\rangle$  — MC cloud (self-norm.)",
                                                   r"$\langle H\rangle_{\mathrm{MC}}$",
                                                   "fig_mc_energy.png"),
        (["lw_trace",   "cloud_weighted_trace"],  "Trace  — MC cloud (self-norm.)",
                                                   r"$\mathrm{tr}\,\rho_{\mathrm{MC}}$",
                                                   "fig_mc_trace.png"),
    ]
    for keys, title, ylabel, fname in specs:
        fig = _two_scheme_fig_prefer(runs, keys, title, ylabel, out_dir, fname)
        panels[keys[0]] = fig

    # Energy residual on log scale — use lw_energy (self-norm) with raw fallback
    fig, ax = plt.subplots(figsize=(_W15, 2.6))
    for name, run in runs.items():
        a = run["arrays"]; t = a["t"]; c  = _COLORS.get(name.lower(), "#888880")
        ls = _LINESTYLES.get(name.lower(), "-")
        lw = _LINEWIDTHS.get(name.lower(), 1.8)
        E_arr = a.get("lw_energy", a.get("cloud_weighted_energy", None))
        if E_arr is None:
            continue
        E = np.asarray(E_arr, dtype=np.float64)
        E0 = float(E[0])
        ax.semilogy(t, np.abs(E - E0) + 1e-18, label=name, color=c, ls=ls, lw=lw)
    _setup(ax, r"Energy residual $|\langle H\rangle - E_0|$  — MC cloud",
           r"$t$", r"$|\langle H\rangle_{\mathrm{MC}} - E_0|$")
    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=_TICK_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    if out_dir:
        fig.savefig(os.path.join(out_dir, "fig_mc_energy_resid.png"),
                    dpi=300, bbox_inches="tight")
    panels["cloud_weighted_energy_resid"] = fig
    return panels


# ──────────────────────────────────────────────────────────────────────────────
# Nuclear moments — analytic GP
# ──────────────────────────────────────────────────────────────────────────────

def plot_nuclear_analytic(
    runs: Dict[str, Dict],
    out_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    GP-surrogate nuclear moments: mean R, mean P, Var(R), Var(P).
    Each panel: PBME (analytic) vs Midpoint (analytic).
    """
    panels: Dict[str, plt.Figure] = {}
    specs = [
        ("nm_R_mean", r"$ \langle R\rangle$  — analytic GP",
                       r"$\langle R\rangle$", "fig_an_nuc_R_mean.png", 50.0),
        ("nm_P_mean", r"$\langle P\rangle$  — analytic GP",
                       r"$\langle P\rangle$", "fig_an_nuc_P_mean.png", 500.0),
        ("nm_R_var",  "Var$(R)$  — analytic GP",
                       r"$\mathrm{Var}(R)$",  "fig_an_nuc_R_var.png",  100.0),
        ("nm_P_var",  "Var$(P)$  — analytic GP",
                       r"$\mathrm{Var}(P)$",  "fig_an_nuc_P_var.png",  500.0),
        ("nm_R_sq",   r"$\langle R^2\rangle$  — analytic GP",
                       r"$\langle R^2\rangle$", "fig_an_nuc_R_sq.png", 500.0),
        ("nm_P_sq",   r"$\langle P^2\rangle$  — analytic GP",
                       r"$\langle P^2\rangle$", "fig_an_nuc_P_sq.png", 5e4),
    ]
    for key, title, ylabel, fname, cap in specs:
        fig = _two_scheme_fig(runs, key, title, ylabel, out_dir, fname, cap=cap)
        panels[key] = fig
    return panels


# ──────────────────────────────────────────────────────────────────────────────
# Nuclear moments — Monte Carlo
# ──────────────────────────────────────────────────────────────────────────────

def plot_nuclear_mc(
    runs: Dict[str, Dict],
    out_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    Self-normalised MC nuclear moments (lw_* preferred over raw cloud_weighted_*).
    Var(R) and Var(P) are not self-normalised individually since they depend on
    both first and second moments; they use raw values which are less sensitive
    to drift than the populations.
    """
    panels: Dict[str, plt.Figure] = {}
    # Mean R and mean P: use lw_* (self-normalised) with raw fallback
    mean_specs = [
        (["lw_R_mean", "cloud_weighted_R_mean"],
         r"$\langle R\rangle$  — MC cloud (self-norm.)",
         r"$\langle R\rangle_{\mathrm{MC}}$",
         "fig_mc_nuc_R_mean.png", 50.0),
        (["lw_P_mean", "cloud_weighted_P_mean"],
         r"$\langle P\rangle$  — MC cloud (self-norm.)",
         r"$\langle P\rangle_{\mathrm{MC}}$",
         "fig_mc_nuc_P_mean.png", 500.0),
    ]
    for keys, title, ylabel, fname, cap in mean_specs:
        fig = _two_scheme_fig_prefer(runs, keys, title, ylabel, out_dir, fname, cap=cap)
        panels[keys[0]] = fig

    # Variance: raw cloud_weighted_* (less critical — variance doesn't diverge
    # as badly as unnormalised means)
    var_specs = [
        ("cloud_weighted_R_var", "Var$(R)$  — MC cloud",
         r"$\mathrm{Var}(R)_{\mathrm{MC}}$",
         "fig_mc_nuc_R_var.png", 100.0),
        ("cloud_weighted_P_var", "Var$(P)$  — MC cloud",
         r"$\mathrm{Var}(P)_{\mathrm{MC}}$",
         "fig_mc_nuc_P_var.png", 500.0),
    ]
    for key, title, ylabel, fname, cap in var_specs:
        fig = _two_scheme_fig(runs, key, title, ylabel, out_dir, fname, cap=cap)
        panels[key] = fig
    return panels


# ──────────────────────────────────────────────────────────────────────────────
# Mapping quadratic moments — analytic GP
# ──────────────────────────────────────────────────────────────────────────────

def plot_mapping_analytic(
    runs: Dict[str, Dict],
    out_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    GP-surrogate mapping quadratic moments.
    Each panel: PBME (analytic) vs Midpoint (analytic).
    """
    panels: Dict[str, plt.Figure] = {}
    specs = [
        ("qm_r0_sq",          r"$\langle r_0^2\rangle$  — analytic GP",
                               r"$\langle r_0^2\rangle$", "fig_an_qm_r0_sq.png"),
        ("qm_r1_sq",          r"$\langle r_1^2\rangle$  — analytic GP",
                               r"$\langle r_1^2\rangle$", "fig_an_qm_r1_sq.png"),
        ("qm_p0_sq",          r"$\langle p_0^2\rangle$  — analytic GP",
                               r"$\langle p_0^2\rangle$", "fig_an_qm_p0_sq.png"),
        ("qm_p1_sq",          r"$\langle p_1^2\rangle$  — analytic GP",
                               r"$\langle p_1^2\rangle$", "fig_an_qm_p1_sq.png"),
        ("qm_r0_r1",          r"$\langle r_0 r_1\rangle$  — analytic GP",
                               r"$\langle r_0 r_1\rangle$", "fig_an_qm_r0_r1.png"),
        ("qm_p0_p1",          r"$\langle p_0 p_1\rangle$  — analytic GP",
                               r"$\langle p_0 p_1\rangle$", "fig_an_qm_p0_p1.png"),
        ("qm_mapping_radius_sq", r"$\langle r^2+p^2\rangle$  — analytic GP",
                                  r"$\langle r_0^2+r_1^2+p_0^2+p_1^2\rangle$",
                                  "fig_an_qm_radius_sq.png"),
    ]
    for key, title, ylabel, fname in specs:
        fig = _two_scheme_fig(runs, key, title, ylabel, out_dir, fname, cap=20.0)
        panels[key] = fig
    return panels


# ──────────────────────────────────────────────────────────────────────────────
# Mapping radius — Monte Carlo
# ──────────────────────────────────────────────────────────────────────────────

def plot_mapping_mc(
    runs: Dict[str, Dict],
    out_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    Cloud-weighted MC mapping radius squared (the only MC mapping moment).
    PBME vs Midpoint.
    """
    # Use self-normalised lw_mapping_radius_sq (prefer) or raw cloud_weighted_*
    fig = _two_scheme_fig_prefer(
        runs,
        ["lw_mapping_radius_sq", "cloud_weighted_mapping_radius_sq"],
        r"$\langle r^2+p^2\rangle$  — MC cloud (self-norm.)",
        r"$\langle r_0^2+r_1^2+p_0^2+p_1^2\rangle_{\mathrm{MC}}$",
        out_dir,
        "fig_mc_qm_radius_sq.png",
        cap=8.0,
    )
    return {"cloud_weighted_mapping_radius_sq": fig}


# =============================================================================
# Density marginals — full analytic integration
# =============================================================================
#
# Under the ARD-RBF GP the density factorizes across dimensions:
#
#     ρ̂(z) = σ_f² Σ_i α_i Π_d exp(-½ (z_d - Z_{i,d})² / ℓ_d²).
#
# Marginalising out any subset S of dimensions gives:
#
#     ∫ ρ̂(z) Π_{d∈S} dz_d
#       = σ_f² [ Π_{d∈S} √(2π) ℓ_d ]
#         · Σ_i α_i Π_{d∉S} exp(-½ (z_d - Z_{i,d})² / ℓ_d²).
#
# In particular:
#
#   • 1D marginal in axis d:  integrate out all other 5 dims.
#   • 2D marginal in axes (d₁, d₂):  integrate out the other 4 dims.
#
# The 2D marginal over (R, P) is strictly the classical nuclear distribution
# (physically positive).  Marginals involving mapping axes inherit the SEO
# signed structure: e.g. the (r₀, p₀) marginal of a |0⟩-initial state is
# negative at the origin and positive on a ring (since ∫w_0(r₀,p₀)dp_1 dr_1
# preserves the sign of w_0).

_AXES_6  = ("R", "P", "r_0", "r_1", "p_0", "p_1")
_AXIS_NAMES = {a: i for i, a in enumerate(_AXES_6)}
_AXIS_LABEL = {"R": r"$R$", "P": r"$P$",
               "r_0": r"$r_0$", "r_1": r"$r_1$",
               "p_0": r"$p_0$", "p_1": r"$p_1$"}


def _gp_arrays(gp: GPDensity):
    """Extract numpy copies of the stored GP state."""
    Zbase = gp.raw_training_centers if hasattr(gp, "raw_training_centers") else gp._Z_train
    Z_tr  = Zbase.detach().cpu().numpy() \
            if hasattr(Zbase, "detach") else np.asarray(Zbase)
    alpha = gp._alpha.detach().cpu().numpy() \
            if hasattr(gp._alpha, "detach")   else np.asarray(gp._alpha)
    ell   = gp.lengthscales
    sf2   = gp.sigma_f ** 2
    return np.asarray(Z_tr), np.asarray(alpha), np.asarray(ell), float(sf2)


def _analytic_1d_marginal(gp: GPDensity, dim_idx: int,
                          grid: FloatArray) -> FloatArray:
    r"""∫ ρ̂(z) Π_{d' ≠ dim_idx} dz_{d'}  evaluated on `grid`."""
    Z_tr, alpha, ell, sf2 = _gp_arrays(gp)
    out_dims = [d for d in range(D) if d != dim_idx]
    prefac = sf2 * float(np.prod([np.sqrt(2.0 * np.pi) * ell[d]
                                  for d in out_dims]))
    # (N, Ng)
    term = np.exp(-0.5 * ((grid[None, :] - Z_tr[:, dim_idx:dim_idx + 1])
                          / ell[dim_idx]) ** 2)
    return (alpha[:, None] * term).sum(axis=0) * prefac


def _analytic_2d_marginal(gp: GPDensity, dim_pair: Tuple[int, int],
                          g1: FloatArray, g2: FloatArray) -> FloatArray:
    r"""
    ∫ ρ̂(z) Π_{d ∉ dim_pair} dz_d

    Returns shape (len(g2), len(g1)) — row=g2 (y-axis), col=g1 (x-axis),
    matching matplotlib imshow(origin='lower', extent=(g1[0], g1[-1],
    g2[0], g2[-1])).
    """
    Z_tr, alpha, ell, sf2 = _gp_arrays(gp)
    d1, d2 = dim_pair
    out_dims = [d for d in range(D) if d not in (d1, d2)]
    prefac = sf2 * float(np.prod([np.sqrt(2.0 * np.pi) * ell[d]
                                  for d in out_dims]))

    # (N, Ng1) and (N, Ng2)
    g1_term = np.exp(-0.5 * ((g1[None, :] - Z_tr[:, d1:d1 + 1])
                             / ell[d1]) ** 2)
    g2_term = np.exp(-0.5 * ((g2[None, :] - Z_tr[:, d2:d2 + 1])
                             / ell[d2]) ** 2)
    # sum_i (alpha_i · prefac) · g2_term[i, j] · g1_term[i, k]  →  [j, k]
    return np.einsum("i,ij,ik->jk", alpha * prefac, g2_term, g1_term)


def _rebuild_gp(snap) -> GPDensity:
    """Reconstruct a predict-only GP from a Snapshot.

    .. warning::

       For density-difference snapshots (``snap.is_density_diff == True``)
       the returned GP only carries the CORRECTION coefficients; the baseline
       is not reconstructed.  Use this helper only for callers that know to
       handle the density-difference split themselves (e.g. fit diagnostics
       that only need the δ-GP).  For marginal integration of the FULL
       density use :func:`marginal_1d_from_snap` / :func:`marginal_2d_from_snap`.
    """
    feature_zscore = bool(getattr(snap, "feature_zscore", False))
    if feature_zscore and getattr(snap, "feature_std", None) is not None:
        ls_norm = np.asarray(snap.lengthscales, dtype=np.float64) / np.asarray(snap.feature_std, dtype=np.float64)
    else:
        ls_norm = np.asarray(snap.lengthscales, dtype=np.float64)
    cfg = GPDensityConfig(
        init_log_sigma_f=float(np.log(snap.sigma_f)),
        init_log_lengthscales=np.log(ls_norm).tolist(),
        init_log_sigma_n=float(np.log(max(snap.sigma_n, 1e-16))),
        n_opt_steps=0,
        reinit_lengthscales=False, fix_sigma_n=True,
        feature_zscore=feature_zscore,
        interpolate_targets=False,
        constraints_enabled=False,
    )
    dyn = PBMEMIntDynamics()
    gp = GPDensity(cfg, dynamics=dyn)

    import torch
    if feature_zscore and getattr(snap, "feature_mean", None) is not None and getattr(snap, "feature_std", None) is not None:
        gp._feature_mean = torch.as_tensor(np.asarray(snap.feature_mean, dtype=np.float64), dtype=torch.float64)
        gp._feature_std = torch.as_tensor(np.asarray(snap.feature_std, dtype=np.float64), dtype=torch.float64)
    gp._set_training_data(snap.Z, snap.y)
    gp._alpha = torch.as_tensor(snap.alpha, dtype=torch.float64).reshape(-1)
    gp._alpha0 = gp._alpha.clone()
    return gp


# =============================================================================
# Snapshot-aware marginal integrators (regime-agnostic)
# =============================================================================
#
# These integrate ρ̂ directly from a Snapshot, bypassing _rebuild_gp.  They
# handle BOTH regimes correctly:
#
#   • vanilla GPDensity      → single kernel integral with (sigma_f, ell).
#   • density-difference GP  → sum of baseline kernel integral (sigma_f_base,
#                              lengthscales_base, alpha_base) plus correction
#                              kernel integral (sigma_f, lengthscales, alpha).
#
# Rationale
# ---------
# Under GPDensityDiff the surrogate is
#
#     ρ̂(z, t) = k_0(z, Z_t) @ α_0  +  k_δ(z, Z_t) @ α_δ,
#
# where k_0 and k_δ are ARD-RBF kernels with INDEPENDENT (σ_f, ℓ) but the
# SAME support centers Z_t.  By Fubini the marginal is additive:
#
#     ∫ ρ̂ dS  =  ∫ (k_0 @ α_0) dS  +  ∫ (k_δ @ α_δ) dS,
#
# and each piece is a standard single-GP kernel integral that factorizes
# across dimensions under the ARD-RBF product form.  The pre-existing
# _rebuild_gp cannot represent this because it loads a single α into a
# single GPDensity carrying a single (σ_f, ℓ) pair.  In particular at t=0
# the correction α_δ is identically zero (δ_i = y_i - y_i(0) = 0), so a
# marginal routed through _rebuild_gp returns zero everywhere — which is
# the symptom "even t=0 seems incorrect" that motivated this rewrite.


def _kernel_marginal_1d(Z_tr: FloatArray, alpha: FloatArray,
                        ell: FloatArray, sf2: float,
                        dim_idx: int, grid: FloatArray) -> FloatArray:
    r"""Single-kernel 1D marginal on ``grid`` in axis ``dim_idx``.

    Analytically integrates

        ρ(z) = σ_f² Σ_i α_i ∏_d exp(-½ (z_d - Z_{i,d})² / ℓ_d²)

    over all dims except ``dim_idx``.  The Gaussian integrals collapse to
    a product of √(2π) ℓ_d prefactors, leaving one Gaussian sum over i
    along the un-integrated axis.
    """
    out_dims = [d for d in range(D) if d != dim_idx]
    prefac = sf2 * float(np.prod([np.sqrt(2.0 * np.pi) * ell[d]
                                  for d in out_dims]))
    term = np.exp(-0.5 * ((grid[None, :] - Z_tr[:, dim_idx:dim_idx + 1])
                          / ell[dim_idx]) ** 2)
    return (alpha[:, None] * term).sum(axis=0) * prefac


def _kernel_marginal_2d(Z_tr: FloatArray, alpha: FloatArray,
                        ell: FloatArray, sf2: float,
                        dim_pair: Tuple[int, int],
                        g1: FloatArray, g2: FloatArray) -> FloatArray:
    r"""Single-kernel 2D marginal on ``(g1, g2)`` for the axis pair.

    Returns shape ``(len(g2), len(g1))`` — row index = g2 (y-axis), column
    index = g1 (x-axis) — matching matplotlib imshow(origin='lower',
    extent=(g1[0], g1[-1], g2[0], g2[-1])).
    """
    d1, d2 = dim_pair
    out_dims = [d for d in range(D) if d not in (d1, d2)]
    prefac = sf2 * float(np.prod([np.sqrt(2.0 * np.pi) * ell[d]
                                  for d in out_dims]))
    g1_term = np.exp(-0.5 * ((g1[None, :] - Z_tr[:, d1:d1 + 1]) / ell[d1]) ** 2)
    g2_term = np.exp(-0.5 * ((g2[None, :] - Z_tr[:, d2:d2 + 1]) / ell[d2]) ** 2)
    return np.einsum("i,ij,ik->jk", alpha * prefac, g2_term, g1_term)


def _require_density_diff_baseline(snap) -> None:
    """Raise a helpful error if a density-diff snapshot is missing baseline state."""
    missing = [n for n, v in (
        ("alpha_base",        snap.alpha_base),
        ("lengthscales_base", snap.lengthscales_base),
        ("sigma_f_base",      snap.sigma_f_base),
    ) if v is None]
    if missing:
        raise ValueError(
            f"density-diff Snapshot at step {snap.step_index} is missing "
            f"required baseline fields: {missing}. "
            "Check that Dynamics._snapshot populates these for density-diff "
            "runs and that the .npz round-trip in Collector preserves them."
        )


def marginal_1d_from_snap(snap, dim_idx: int,
                          grid: FloatArray) -> FloatArray:
    r"""1D marginal of the surrogate density ρ̂(z) from a Snapshot.

    Correct for BOTH vanilla GPDensity and GPDensityDiff snapshots.  In the
    density-diff regime the returned marginal is the sum of the baseline
    and correction kernel integrals, each using its own (σ_f, ℓ).

    Parameters
    ----------
    snap      : Snapshot                  — carries Z, α, (α_base, …).
    dim_idx   : int                       — axis to keep, 0..D-1.
    grid      : (Ng,) array               — axis grid in physical units.

    Returns
    -------
    (Ng,) array   — marginal ρ̂ integrated over the other D-1 dims.
    """
    Z_tr = np.asarray(snap.Z, dtype=np.float64)

    if getattr(snap, "is_density_diff", False):
        _require_density_diff_baseline(snap)
        out = _kernel_marginal_1d(
            Z_tr,
            np.asarray(snap.alpha_base, dtype=np.float64).reshape(-1),
            np.asarray(snap.lengthscales_base, dtype=np.float64),
            float(snap.sigma_f_base) ** 2,
            dim_idx, grid,
        )
        out = out + _kernel_marginal_1d(
            Z_tr,
            np.asarray(snap.alpha, dtype=np.float64).reshape(-1),
            np.asarray(snap.lengthscales, dtype=np.float64),
            float(snap.sigma_f) ** 2,
            dim_idx, grid,
        )
        return out

    # vanilla single GP
    return _kernel_marginal_1d(
        Z_tr,
        np.asarray(snap.alpha, dtype=np.float64).reshape(-1),
        np.asarray(snap.lengthscales, dtype=np.float64),
        float(snap.sigma_f) ** 2,
        dim_idx, grid,
    )


def marginal_2d_from_snap(snap, dim_pair: Tuple[int, int],
                          g1: FloatArray, g2: FloatArray) -> FloatArray:
    r"""2D marginal of the surrogate density ρ̂(z) from a Snapshot.

    Correct for BOTH vanilla GPDensity and GPDensityDiff snapshots.  See
    :func:`marginal_1d_from_snap` for the density-diff math and the shape
    convention for the returned array.
    """
    Z_tr = np.asarray(snap.Z, dtype=np.float64)

    if getattr(snap, "is_density_diff", False):
        _require_density_diff_baseline(snap)
        out = _kernel_marginal_2d(
            Z_tr,
            np.asarray(snap.alpha_base, dtype=np.float64).reshape(-1),
            np.asarray(snap.lengthscales_base, dtype=np.float64),
            float(snap.sigma_f_base) ** 2,
            dim_pair, g1, g2,
        )
        out = out + _kernel_marginal_2d(
            Z_tr,
            np.asarray(snap.alpha, dtype=np.float64).reshape(-1),
            np.asarray(snap.lengthscales, dtype=np.float64),
            float(snap.sigma_f) ** 2,
            dim_pair, g1, g2,
        )
        return out

    # vanilla single GP
    return _kernel_marginal_2d(
        Z_tr,
        np.asarray(snap.alpha, dtype=np.float64).reshape(-1),
        np.asarray(snap.lengthscales, dtype=np.float64),
        float(snap.sigma_f) ** 2,
        dim_pair, g1, g2,
    )


# =============================================================================
# 1D marginal plot — overlay both schemes on one axis
# =============================================================================

def plot_density_1d_marginal(
    snaps: Dict[str, "Snapshot"],
    axis: str,
    n_grid: int = 200, padding: float = 2.0,
    savepath: Optional[str] = None,
) -> plt.Figure:
    """
    Overlay the 1D marginal of ρ̂ in axis `axis` for every scheme in `snaps`.
    """
    d = _AXIS_NAMES[axis]
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.0))

    lo_all = min(snap.Z[:, d].min() for snap in snaps.values()) - padding
    hi_all = max(snap.Z[:, d].max() for snap in snaps.values()) + padding
    grid = np.linspace(lo_all, hi_all, n_grid)

    for name, snap in snaps.items():
        # Use the snapshot-aware marginal integrator so that density-diff
        # snapshots correctly receive baseline + correction contributions.
        # (The old `_rebuild_gp` path silently dropped the baseline.)
        rho1d = marginal_1d_from_snap(snap, d, grid)
        c  = _COLORS.get(name.lower(), "#888880")
        ls = _LINESTYLES.get(name.lower(), "-")
        lw = _LINEWIDTHS.get(name.lower(), 1.8)
        ax.plot(grid, rho1d, label=name, color=c, ls=ls, lw=lw)

    ax.axhline(0.0, color="k", lw=0.5, ls="--", alpha=0.4)
    _setup(ax,
           rf"1D marginal: $\int\hat\rho\,\prod_{{d'\neq d}}dz_{{d'}}$",
           _AXIS_LABEL[axis],
           fr"$\rho_{{\mathrm{{marg}}}}(\,${_AXIS_LABEL[axis][1:-1]}$\,)$")
    ax.legend(fontsize=_TICK_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    _save(fig, savepath)
    return fig


# =============================================================================
# 2D marginal plot — side-by-side, shared colormap
# =============================================================================

def plot_density_2d_marginal(
    snaps: Dict[str, "Snapshot"],
    axes_pair: Tuple[str, str],
    n_grid: int = 70, padding: float = 2.0,
    savepath: Optional[str] = None,
) -> plt.Figure:
    r"""
    2D marginal of ρ̂ in the two named axes, integrated analytically over
    the remaining four dimensions.

    If both PBME and one QCLE snapshot are present, generate three panels:
    PBME, QCLE, and Difference = QCLE - PBME.
    """
    d1 = _AXIS_NAMES[axes_pair[0]]
    d2 = _AXIS_NAMES[axes_pair[1]]

    lo1 = min(s.Z[:, d1].min() for s in snaps.values()) - padding
    hi1 = max(s.Z[:, d1].max() for s in snaps.values()) + padding
    lo2 = min(s.Z[:, d2].min() for s in snaps.values()) - padding
    hi2 = max(s.Z[:, d2].max() for s in snaps.values()) + padding
    g1 = np.linspace(lo1, hi1, n_grid)
    g2 = np.linspace(lo2, hi2, n_grid)

    rhos = {}
    for name, snap in snaps.items():
        # Use the snapshot-aware marginal integrator so that density-diff
        # snapshots correctly receive baseline + correction contributions.
        rhos[name] = marginal_2d_from_snap(snap, (d1, d2), g1, g2)

    pbme_key = next((k for k in snaps if str(k).lower() == "pbme"), None)
    qcle_keys = [k for k in snaps if str(k).lower() != "pbme"]

    if pbme_key is not None and len(qcle_keys) == 1 and len(snaps) == 2:
        qcle_key = qcle_keys[0]
        rho_pbme = rhos[pbme_key]
        rho_qcle = rhos[qcle_key]
        rho_diff = rho_qcle - rho_pbme
        vmax = max(float(np.max(np.abs(rho_pbme))), float(np.max(np.abs(rho_qcle))), 1.0e-16)
        vmax_diff = max(float(np.max(np.abs(rho_diff))), 1.0e-16)

        fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.5), squeeze=False)
        ax0, ax1, ax2 = axes.ravel()

        im0 = ax0.imshow(rho_pbme, origin="lower", aspect="auto",
                         extent=(g1[0], g1[-1], g2[0], g2[-1]),
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax0.scatter(snaps[pbme_key].Z[:, d1], snaps[pbme_key].Z[:, d2],
                    s=3, color="k", alpha=0.2, rasterized=True)
        _setup(ax0, f"PBME   (step {snaps[pbme_key].step_index},  t={snaps[pbme_key].t:.2f})",
               _AXIS_LABEL[axes_pair[0]], _AXIS_LABEL[axes_pair[1]])
        plt.colorbar(im0, ax=ax0)

        im1 = ax1.imshow(rho_qcle, origin="lower", aspect="auto",
                         extent=(g1[0], g1[-1], g2[0], g2[-1]),
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax1.scatter(snaps[qcle_key].Z[:, d1], snaps[qcle_key].Z[:, d2],
                    s=3, color="k", alpha=0.2, rasterized=True)
        _setup(ax1, f"{qcle_key}   (step {snaps[qcle_key].step_index},  t={snaps[qcle_key].t:.2f})",
               _AXIS_LABEL[axes_pair[0]], _AXIS_LABEL[axes_pair[1]])
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(rho_diff, origin="lower", aspect="auto",
                         extent=(g1[0], g1[-1], g2[0], g2[-1]),
                         cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff)
        _setup(ax2, f"Difference = {qcle_key} - PBME",
               _AXIS_LABEL[axes_pair[0]], _AXIS_LABEL[axes_pair[1]])
        plt.colorbar(im2, ax=ax2)
    else:
        vmax = max(np.max(np.abs(r)) for r in rhos.values())
        if vmax == 0.0:
            vmax = 1.0

        n = len(snaps)
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.5), squeeze=False)
        for ax, (name, snap) in zip(axes.ravel(), snaps.items()):
            im = ax.imshow(rhos[name], origin="lower", aspect="auto",
                           extent=(g1[0], g1[-1], g2[0], g2[-1]),
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.scatter(snap.Z[:, d1], snap.Z[:, d2],
                       s=3, color="k", alpha=0.2, rasterized=True)
            _setup(ax, f"{name}   (step {snap.step_index},  t={snap.t:.2f})",
                   _AXIS_LABEL[axes_pair[0]], _AXIS_LABEL[axes_pair[1]])
            plt.colorbar(im, ax=ax)

    fig.suptitle(
        rf"2D marginal  "
        rf"$\int\hat\rho\,\prod_{{d\notin\{{{axes_pair[0]},{axes_pair[1]}\}}}}dz_d$",
        fontsize=_TITLE_FONT,
    )
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    _save(fig, savepath)
    return fig


# =============================================================================
# Faithfulness-Test #3:  cloud-Riemann KDE  vs  GP-integral marginal
# =============================================================================
#
# Background.  The 2D marginal plotted by ``plot_density_2d_marginal`` is the
# analytic kernel integral of the surrogate,
#     ρ̂_marg(x, y) = ∫ ρ̂(R, P, r_0, r_1, p_0, p_1) ∏_{d ∈ out} dz_d,
# which is what the GP "believes" the marginal should be given its fitted α.
# That object reflects whatever sign-oscillations α has acquired (e.g. when
# the kernel bandwidth is too small for the spread cloud after the avoided
# crossing — see the focused-mode post-crossing analysis).
#
# The cloud-Riemann KDE marginal evaluates the same physical integral
# directly from the trajectory cloud,
#     ρ̂_KDE(x, y) = Σ_i (ω_i y_i) · 𝒩((x, y) | (R_i, P_i), h²),
# with 𝒩 a fixed-bandwidth 2D Gaussian (Silverman's rule on the cloud
# spread, so it tracks the actual cloud).  This estimator cannot
# sign-oscillate because all kernel coefficients are ω_i y_i ≥ 0 in
# focused mode.
#
# Plotting them side by side reveals when the GP fit is unfaithful to
# the cloud: the GP panel shows red/blue dipoles while the KDE panel
# shows the actual positive-density cloud.


def _cloud_kde_2d_marginal(
    Z_tr: FloatArray,
    omega: FloatArray,
    y_train: FloatArray,
    dim_pair: Tuple[int, int],
    g1: FloatArray,
    g2: FloatArray,
    bandwidth: Optional[Tuple[float, float]] = None,
) -> FloatArray:
    r"""
    Kernel density estimate of the 2D (d1, d2) marginal directly from
    the trajectory cloud, with weights ω_i · y_i.

    bandwidth=None uses Silverman's rule  h_d = 1.06 · σ_d · N^{-1/6}
    (2D version) — adapts to the cloud spread snapshot-by-snapshot.

    Returns shape (Ng2, Ng1) — rows = g2, cols = g1 — matching the
    layout used by ``_kernel_marginal_2d`` so subplots share axes.
    """
    d1, d2 = dim_pair
    z1 = Z_tr[:, d1]
    z2 = Z_tr[:, d2]
    w  = np.asarray(omega, dtype=np.float64).reshape(-1) \
       * np.asarray(y_train, dtype=np.float64).reshape(-1)
    N = int(z1.size)
    if bandwidth is None:
        s1 = float(np.std(z1))
        s2 = float(np.std(z2))
        Nsafe = max(N, 2)
        h1 = max(1.06 * s1 * Nsafe ** (-1.0 / 6.0), 1.0e-6)
        h2 = max(1.06 * s2 * Nsafe ** (-1.0 / 6.0), 1.0e-6)
    else:
        h1, h2 = bandwidth
        h1 = float(max(h1, 1.0e-6))
        h2 = float(max(h2, 1.0e-6))
    norm = 1.0 / (2.0 * np.pi * h1 * h2)
    A = np.exp(-0.5 * ((g1[None, :] - z1[:, None]) / h1) ** 2)  # (N, Ng1)
    B = np.exp(-0.5 * ((g2[None, :] - z2[:, None]) / h2) ** 2)  # (N, Ng2)
    return norm * np.einsum("i,ij,ik->jk", w, B, A)


def plot_faithfulness_2d_marginal(
    snaps: Dict[str, "Snapshot"],
    axes_pair: Tuple[str, str],
    n_grid: int = 70,
    padding: float = 2.0,
    savepath: Optional[str] = None,
) -> plt.Figure:
    r"""
    Side-by-side comparison of GP-integral and cloud-Riemann-KDE 2D
    marginals, per scheme (PBME, midpoint, …).

    Each row: one scheme; three panels per row:
        [GP-integral marginal]  [cloud-KDE marginal]  [GP - KDE]
    so visual sign-oscillation in the GP panel that is absent from the
    KDE panel is unambiguously a GP-fit artifact, not a physical feature.

    Required Snapshot fields beyond the standard set:
        snap.y                   — carried labels (live; QCLE-corrected
                                    for midpoint, frozen at ρ_0 for PBME).
        snap.proposal_density    — used to reconstruct ω_i = 1/(N q(z_i^0))
                                    when present; falls back to ω = 1/N
                                    when missing (exact for focused-mode
                                    proposals where q ∝ ρ_0 anyway).
    """
    d1 = _AXIS_NAMES[axes_pair[0]]
    d2 = _AXIS_NAMES[axes_pair[1]]

    lo1 = min(s.Z[:, d1].min() for s in snaps.values()) - padding
    hi1 = max(s.Z[:, d1].max() for s in snaps.values()) + padding
    lo2 = min(s.Z[:, d2].min() for s in snaps.values()) - padding
    hi2 = max(s.Z[:, d2].max() for s in snaps.values()) + padding
    g1 = np.linspace(lo1, hi1, n_grid)
    g2 = np.linspace(lo2, hi2, n_grid)

    n_schemes = len(snaps)
    fig, axes = plt.subplots(n_schemes, 3, figsize=(15.8, 4.5 * n_schemes),
                              squeeze=False)

    rho_gp:   Dict[str, FloatArray] = {}
    rho_kde:  Dict[str, FloatArray] = {}
    rho_diff: Dict[str, FloatArray] = {}
    _warned_about_missing_proposal = False
    for name, snap in snaps.items():
        rho_gp[name] = marginal_2d_from_snap(snap, (d1, d2), g1, g2)
        # ω reconstruction chain.  The geometric measure ω_i = 1/(N·q(z_i⁰))
        # is FROZEN at t=0 (Liouville) so any snapshot's frozen-state field
        # gives the right answer.
        N = int(np.asarray(snap.Z).shape[0])
        prop = getattr(snap, "proposal_density", None)
        snap_weight = getattr(snap, "weight", None)
        if prop is not None:
            # ω_i = 1/(N · q(z_i⁰))
            omega = 1.0 / (N * np.asarray(prop, dtype=np.float64).reshape(-1))
        elif snap_weight is not None:
            # snap.weight = initial_weight = w_i = ω_i · target_density(z_i⁰)
            # Not what we want directly, but for focused mode where
            # proposal_density = target_density (the IS cancellation),
            # ω_i = w_i / target_density(z_i⁰) = w_i · target_inv.  Since
            # we don't have target_density either, fall through to 1/N
            # below for safety.
            omega = np.full(N, 1.0 / N, dtype=np.float64)
            if not _warned_about_missing_proposal:
                print(f"[plot_faithfulness_2d_marginal] WARNING: snap for "
                      f"'{name}' has no proposal_density (only weight).  "
                      f"Falling back to uniform ω = 1/N — the KDE panel "
                      f"will be SCALED INCORRECTLY (typically O(1e-3) too "
                      f"small for focused mode).  To fix: re-run with a "
                      f"Dynamics.py that saves proposal_density at every "
                      f"snapshot, OR patch existing snapshots with "
                      f"refix_faithfulness_figs.py.")
                _warned_about_missing_proposal = True
        else:
            # Last-resort fallback — gives the qualitatively-correct cloud
            # KDE up to an overall scale.  Will be visibly off if the
            # cloud is not equi-weighted (i.e. for seo_signed where
            # different trajectories carry different ω).
            omega = np.full(N, 1.0 / N, dtype=np.float64)
            if not _warned_about_missing_proposal:
                print(f"[plot_faithfulness_2d_marginal] WARNING: snap for "
                      f"'{name}' has no proposal_density and no weight.  "
                      f"Falling back to uniform ω = 1/N — the KDE panel "
                      f"will be SCALED INCORRECTLY by ~K_focus·W_cl (1e-3) "
                      f"in focused mode, making it appear as ~0.  "
                      f"Use refix_faithfulness_figs.py to recover ω from "
                      f"the step-0 snapshot.")
                _warned_about_missing_proposal = True
        # Defensive: if omega has any non-finite entries (legacy snapshots
        # with proposal=0 somewhere) fall back to 1/N rather than NaN-out
        # the whole plot.
        if not np.all(np.isfinite(omega)):
            omega = np.full(N, 1.0 / N, dtype=np.float64)
        y_arr = np.asarray(snap.y, dtype=np.float64).reshape(-1)
        rho_kde[name] = _cloud_kde_2d_marginal(
            np.asarray(snap.Z, dtype=np.float64),
            omega, y_arr,
            (d1, d2), g1, g2,
        )
        rho_diff[name] = rho_gp[name] - rho_kde[name]

    vmax_main = max(
        max(float(np.max(np.abs(rho_gp[k]))) for k in rho_gp),
        max(float(np.max(np.abs(rho_kde[k]))) for k in rho_kde),
        1.0e-16,
    )
    vmax_diff = max(
        max(float(np.max(np.abs(rho_diff[k]))) for k in rho_diff),
        1.0e-16,
    )

    for row, (name, snap) in enumerate(snaps.items()):
        ax0, ax1, ax2 = axes[row]
        im0 = ax0.imshow(rho_gp[name], origin="lower", aspect="auto",
                         extent=(g1[0], g1[-1], g2[0], g2[-1]),
                         cmap="RdBu_r", vmin=-vmax_main, vmax=vmax_main)
        ax0.scatter(snap.Z[:, d1], snap.Z[:, d2], s=3, color="k",
                    alpha=0.2, rasterized=True)
        _setup(ax0,
               f"{name}  GP-integral  (step {snap.step_index}, t={snap.t:.1f})",
               _AXIS_LABEL[axes_pair[0]], _AXIS_LABEL[axes_pair[1]])
        plt.colorbar(im0, ax=ax0)

        im1 = ax1.imshow(rho_kde[name], origin="lower", aspect="auto",
                         extent=(g1[0], g1[-1], g2[0], g2[-1]),
                         cmap="RdBu_r", vmin=-vmax_main, vmax=vmax_main)
        ax1.scatter(snap.Z[:, d1], snap.Z[:, d2], s=3, color="k",
                    alpha=0.2, rasterized=True)
        _setup(ax1, f"{name}  cloud-KDE (Silverman)",
               _AXIS_LABEL[axes_pair[0]], _AXIS_LABEL[axes_pair[1]])
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(rho_diff[name], origin="lower", aspect="auto",
                         extent=(g1[0], g1[-1], g2[0], g2[-1]),
                         cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff)
        _setup(ax2, f"{name}  GP - KDE",
               _AXIS_LABEL[axes_pair[0]], _AXIS_LABEL[axes_pair[1]])
        plt.colorbar(im2, ax=ax2)

    fig.suptitle(
        r"Faithfulness test:  GP-integral marginal  vs  cloud-Riemann KDE",
        fontsize=_TITLE_FONT,
    )
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    _save(fig, savepath)
    return fig


# =============================================================================
# Snapshot-aware 2D and 1D CONDITIONALS (slices)
# =============================================================================
#
# These are the point-evaluation companions of the marginal helpers.  A
# conditional (slice) is the density ρ̂(z_S, z_{S^c} = z*) evaluated at a
# fixed anchor z* in the un-plotted axes.  Unlike marginals, slices do
# NOT integrate — they preserve the polynomial's sign pointwise.  In
# particular, pinning the mapping variables r_α = p_α = 0 places the
# slice inside the SEO negative disc, so the resulting 2D image in any
# plane shows the signed Wigner structure that the marginal integrates
# away.
#
# Regime-aware: vanilla (single kernel) and density-difference (sum of
# baseline + correction kernel evaluations, each with its own sigma_f
# and lengthscales) are handled by the same snapshot-level API.


def _kernel_predict_at(Z_tr: FloatArray, alpha: FloatArray,
                       ell: FloatArray, sf2: float,
                       z_query: FloatArray) -> FloatArray:
    r"""Evaluate σ_f² Σ_i α_i ∏_d exp(-½(z_d-Z_{i,d})²/ℓ_d²) at query points.

    Parameters
    ----------
    Z_tr     : (N, D)  training centers (physical coordinates).
    alpha    : (N,)    kernel coefficients.
    ell      : (D,)    per-axis lengthscales (physical).
    sf2      : float   σ_f² scalar.
    z_query  : (M, D)  query points (physical coordinates).

    Returns
    -------
    (M,) density values.  Can have any sign; no clipping.
    """
    diff = z_query[:, None, :] - Z_tr[None, :, :]               # (M, N, D)
    arg = -0.5 * np.sum((diff / ell[None, None, :]) ** 2, axis=-1)   # (M, N)
    return sf2 * (np.exp(arg) * alpha[None, :]).sum(axis=1)


def density_predict_from_snap(snap, z_query: FloatArray) -> FloatArray:
    r"""Evaluate ρ̂(z) at arbitrary query points from a Snapshot.

    Correct for BOTH vanilla GPDensity and GPDensityDiff snapshots.  In
    the density-difference regime the result is the sum of the baseline
    and correction kernel evaluations — each with its own (σ_f, ℓ) —
    analogous to what :func:`marginal_1d_from_snap` does for integrals.
    """
    z_query = np.atleast_2d(np.asarray(z_query, dtype=np.float64))
    assert z_query.shape[1] == D, \
        f"z_query must have shape (M, {D}); got {z_query.shape}"
    Z_tr = np.asarray(snap.Z, dtype=np.float64)

    if getattr(snap, "is_density_diff", False):
        _require_density_diff_baseline(snap)
        out = _kernel_predict_at(
            Z_tr,
            np.asarray(snap.alpha_base, dtype=np.float64).reshape(-1),
            np.asarray(snap.lengthscales_base, dtype=np.float64),
            float(snap.sigma_f_base) ** 2,
            z_query,
        )
        out = out + _kernel_predict_at(
            Z_tr,
            np.asarray(snap.alpha, dtype=np.float64).reshape(-1),
            np.asarray(snap.lengthscales, dtype=np.float64),
            float(snap.sigma_f) ** 2,
            z_query,
        )
        return out

    # vanilla single GP
    return _kernel_predict_at(
        Z_tr,
        np.asarray(snap.alpha, dtype=np.float64).reshape(-1),
        np.asarray(snap.lengthscales, dtype=np.float64),
        float(snap.sigma_f) ** 2,
        z_query,
    )


def _default_slice_anchor(snap, kept_axes) -> Dict[str, float]:
    r"""Default fixed values for the non-plotted axes in a slice.

    * Nuclear axes (R, P): density-weighted centroid (approximated by the
      cloud mean, which for signed-SEO clouds agrees with the density
      centroid up to MC noise).
    * Mapping axes (r_α, p_α): 0.0.  This choice pins the slice inside the
      SEO negative lobe so the resulting image shows the signed Wigner
      structure — the main reason to look at slices.

    Parameters
    ----------
    snap         : Snapshot
    kept_axes    : tuple or set of axis names that ARE being plotted
                   (so we set anchors for the complementary axes).
    """
    kept = set(kept_axes)
    Z = np.asarray(snap.Z, dtype=np.float64)
    cloud_mean = Z.mean(axis=0)
    out: Dict[str, float] = {}
    for a in _AXES_6:
        if a in kept:
            continue
        if a in ("R", "P"):
            out[a] = float(cloud_mean[_AXIS_NAMES[a]])
        else:
            out[a] = 0.0
    return out


def plot_density_1d_slice(
    snaps: Dict[str, "Snapshot"],
    axis: str,
    fixed_vals: Optional[Dict[str, float]] = None,
    n_grid: int = 200, padding: float = 2.0,
    savepath: Optional[str] = None,
) -> plt.Figure:
    r"""1D SLICE of ρ̂(z) along `axis`, with the other five axes held at
    `fixed_vals` (default: nuclear axes at the cloud mean, mapping axes
    at zero).

    Complementary to :func:`plot_density_1d_marginal`: the marginal
    integrates over the other five axes while the slice evaluates at
    fixed points on them.  Slices preserve the signed Wigner structure
    that marginals integrate away.
    """
    d = _AXIS_NAMES[axis]
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.0))

    lo_all = min(snap.Z[:, d].min() for snap in snaps.values()) - padding
    hi_all = max(snap.Z[:, d].max() for snap in snaps.values()) + padding
    grid = np.linspace(lo_all, hi_all, n_grid)

    anchors = {}
    for name, snap in snaps.items():
        anchor = fixed_vals if fixed_vals is not None \
                              else _default_slice_anchor(snap, (axis,))
        anchors[name] = anchor

        z_query = np.zeros((grid.size, D), dtype=np.float64)
        for a, v in anchor.items():
            z_query[:, _AXIS_NAMES[a]] = v
        z_query[:, d] = grid
        rho = density_predict_from_snap(snap, z_query)
        c  = _COLORS.get(name.lower(), "#888880")
        ls = _LINESTYLES.get(name.lower(), "-")
        lw = _LINEWIDTHS.get(name.lower(), 1.8)
        ax.plot(grid, rho, label=name, color=c, ls=ls, lw=lw)

    ax.axhline(0.0, color="k", lw=0.5, ls="--", alpha=0.4)
    any_anchor = next(iter(anchors.values()))
    anchor_str = ", ".join(f"{a}={v:.2f}" for a, v in sorted(any_anchor.items()))
    _setup(ax,
           rf"1D slice: $\hat\rho({axis}\,|\,${anchor_str}$)$",
           _AXIS_LABEL[axis],
           r"$\hat\rho$")
    ax.legend(fontsize=_TICK_FONT)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    _save(fig, savepath)
    return fig


def plot_density_2d_slice(
    snaps: Dict[str, "Snapshot"],
    axes_pair: Tuple[str, str],
    fixed_vals: Optional[Dict[str, float]] = None,
    n_grid: int = 100, padding: float = 2.0,
    savepath: Optional[str] = None,
) -> plt.Figure:
    r"""2D SLICE of ρ̂(z) in `axes_pair`, non-plotted axes held at `fixed_vals`.

    Default `fixed_vals` comes from :func:`_default_slice_anchor` —
    nuclear axes at the cloud mean, mapping axes at zero.  Under that
    anchor, the slice sits inside the SEO negative disc, so every 2D
    slice through a plane that includes a mapping axis or the $(R, P)$
    plane shows the signed Wigner structure.

    Same three-panel layout (PBME | QCLE | difference) as
    :func:`plot_density_2d_marginal` when both are present.
    """
    d1 = _AXIS_NAMES[axes_pair[0]]
    d2 = _AXIS_NAMES[axes_pair[1]]

    lo1 = min(s.Z[:, d1].min() for s in snaps.values()) - padding
    hi1 = max(s.Z[:, d1].max() for s in snaps.values()) + padding
    lo2 = min(s.Z[:, d2].min() for s in snaps.values()) - padding
    hi2 = max(s.Z[:, d2].max() for s in snaps.values()) + padding
    g1 = np.linspace(lo1, hi1, n_grid)
    g2 = np.linspace(lo2, hi2, n_grid)
    G1, G2 = np.meshgrid(g1, g2, indexing="xy")     # (Ng2, Ng1)

    rhos: Dict[str, FloatArray] = {}
    anchors: Dict[str, Dict[str, float]] = {}
    for name, snap in snaps.items():
        anchor = fixed_vals if fixed_vals is not None \
                              else _default_slice_anchor(snap, axes_pair)
        anchors[name] = anchor

        z_query = np.zeros((G1.size, D), dtype=np.float64)
        for a, v in anchor.items():
            z_query[:, _AXIS_NAMES[a]] = v
        z_query[:, d1] = G1.ravel()
        z_query[:, d2] = G2.ravel()
        rhos[name] = density_predict_from_snap(snap, z_query).reshape(G1.shape)

    pbme_key = next((k for k in snaps if str(k).lower() == "pbme"), None)
    qcle_keys = [k for k in snaps if str(k).lower() != "pbme"]

    if pbme_key is not None and len(qcle_keys) == 1 and len(snaps) == 2:
        qcle_key = qcle_keys[0]
        rho_pbme = rhos[pbme_key]; rho_qcle = rhos[qcle_key]
        rho_diff = rho_qcle - rho_pbme
        vmax = max(float(np.max(np.abs(rho_pbme))),
                   float(np.max(np.abs(rho_qcle))), 1.0e-16)
        vmax_diff = max(float(np.max(np.abs(rho_diff))), 1.0e-16)

        fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.5), squeeze=False)
        ax0, ax1, ax2 = axes.ravel()

        for ax, name, rho, vlim in [
            (ax0, pbme_key, rho_pbme, vmax),
            (ax1, qcle_key, rho_qcle, vmax),
            (ax2, f"{qcle_key} - {pbme_key}", rho_diff, vmax_diff),
        ]:
            im = ax.imshow(rho, origin="lower", aspect="auto",
                           extent=(g1[0], g1[-1], g2[0], g2[-1]),
                           cmap="RdBu_r", vmin=-vlim, vmax=vlim)
            if name in snaps:
                ax.scatter(snaps[name].Z[:, d1], snaps[name].Z[:, d2],
                           s=3, color="k", alpha=0.2, rasterized=True)
                _setup(ax,
                       f"{name}   (step {snaps[name].step_index}, "
                       f"t={snaps[name].t:.2f})",
                       _AXIS_LABEL[axes_pair[0]], _AXIS_LABEL[axes_pair[1]])
            else:
                _setup(ax, name, _AXIS_LABEL[axes_pair[0]],
                       _AXIS_LABEL[axes_pair[1]])
            plt.colorbar(im, ax=ax)
    else:
        vmax = max(np.max(np.abs(r)) for r in rhos.values())
        if vmax == 0.0:
            vmax = 1.0
        n = len(snaps)
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.5), squeeze=False)
        for ax, (name, snap) in zip(axes.ravel(), snaps.items()):
            im = ax.imshow(rhos[name], origin="lower", aspect="auto",
                           extent=(g1[0], g1[-1], g2[0], g2[-1]),
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.scatter(snap.Z[:, d1], snap.Z[:, d2],
                       s=3, color="k", alpha=0.2, rasterized=True)
            _setup(ax, f"{name}   (step {snap.step_index}, t={snap.t:.2f})",
                   _AXIS_LABEL[axes_pair[0]], _AXIS_LABEL[axes_pair[1]])
            plt.colorbar(im, ax=ax)

    any_anchor = next(iter(anchors.values()))
    anchor_str = ", ".join(f"{a}={v:.2f}" for a, v in sorted(any_anchor.items()))
    fig.suptitle(
        rf"2D slice  $\hat\rho({axes_pair[0]},{axes_pair[1]}\,|\,$"
        f"{anchor_str}$)$",
        fontsize=_TITLE_FONT,
    )
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    _save(fig, savepath)
    return fig


# =============================================================================
# Enumerate: all 6 × 1D  + all 15 × 2D marginals (pairs)
# =============================================================================


# =============================================================================
# Enumerate: all 6 × 1D  + all 15 × 2D marginals (pairs)
# =============================================================================

_ALL_2D_PAIRS: Tuple[Tuple[str, str], ...] = tuple(
    (_AXES_6[i], _AXES_6[j])
    for i in range(len(_AXES_6))
    for j in range(i + 1, len(_AXES_6))
)   # 15 pairs



# =============================================================================
# QCLE correction term diagnostics  (density-weighted)
# =============================================================================

def plot_qcle_correction_diagnostics(
    runs: Dict[str, Dict],
    out_dir: Optional[str] = None,
    savepath: Optional[str] = None,
) -> plt.Figure:
    r"""
    Density-weighted QCLE correction term diagnostics for all non-PBME runs.

    The four curves mirror the FLV pipeline diagnostic:

      rho-weighted mean   Σ_m y_m Q_m / Σ_m y_m      (signed, tracks net drift)
      rho-weighted RMS    √(Σ_m |y_m| Q_m² / Σ_m |y_m|)  (magnitude under ρ)
      support max |Q|     max_m |Q_m|                  (worst-case correction)
      Σ_m ρ_m c_m         Σ_m y_m Q_m                  (raw normalization flux)

    t = 0 is excluded (the correction is trivially zero before the first step).

    Physical interpretation
    -----------------------
    * Σ_m ρ_m c_m  measures the net normalization flux injected per step:
      if this is consistently near zero the QCLE correction is approximately
      normalization-preserving.
    * The rho-weighted mean tracks any systematic signed drift in Q across the
      density support.  A non-zero steady-state value signals a bias in the
      correction operator relative to the carried density.
    * The rho-weighted RMS quantifies the typical correction magnitude felt by
      the "important" (high-ρ) support points.
    * support max |Q| gives the absolute envelope; large values relative to
      the weighted RMS mean the correction is concentrated in low-ρ tails.
    """
    mid_runs = {name: r for name, r in runs.items()
                if name.lower() != "pbme"}

    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    if not mid_runs:
        ax.text(0.5, 0.5, "No non-PBME runs available.",
                ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches="tight")
        return fig

    COLOR_MEAN  = "#E07020"   # warm orange   — rho-weighted mean
    COLOR_RMS   = "#2E3A8C"   # dark blue     — rho-weighted RMS
    COLOR_MAX   = "#A040B0"   # violet/pink   — support max |Q|
    COLOR_SUM   = "#207830"   # dark green    — Σ ρ c (dashed)

    for name, run in mid_runs.items():
        a = run["arrays"]
        t = np.asarray(a["t"], dtype=np.float64)

        # Exclude t = 0 (correction undefined / trivially zero)
        mask = t > 0.0
        if not np.any(mask):
            mask = np.ones(len(t), dtype=bool)
        tm = t[mask]

        def _get(key):
            v = a.get(key, None)
            if v is None:
                return None
            return np.asarray(v, dtype=np.float64)[mask]

        mean_w  = _get("cs_q_y_weighted_mean")
        rms_w   = _get("cs_q_y_weighted_rms")
        max_q   = _get("cs_q_max")
        sum_yc  = _get("cs_q_sum_yc")

        suffix = f"  ({name})" if len(mid_runs) > 1 else ""

        if mean_w is not None and np.any(np.isfinite(mean_w)):
            ax.plot(tm, mean_w, color=COLOR_MEAN, lw=2.0,
                    label="$\\rho$-weighted mean" + suffix)
        if rms_w is not None and np.any(np.isfinite(rms_w)):
            ax.plot(tm, rms_w,  color=COLOR_RMS,  lw=2.0,
                    label="$\\rho$-weighted RMS" + suffix)
        if max_q is not None and np.any(np.isfinite(max_q)):
            ax.plot(tm, max_q,  color=COLOR_MAX,  lw=1.6,
                    label=r"support $\max|Q|$" + suffix)
        if sum_yc is not None and np.any(np.isfinite(sum_yc)):
            ax.plot(tm, sum_yc, color=COLOR_SUM,  lw=1.6, ls="--",
                    label="$\\sum_m \\rho_m Q_m$" + suffix)

    ax.axhline(0.0, color="black", lw=0.8, ls="-", alpha=0.6)
    ax.set_xlabel(r"$t$  [a.u.]", fontsize=_TITLE_FONT)
    ax.set_ylabel("QCLE correction diagnostics", fontsize=_TITLE_FONT)
    ax.set_title(
        "QCLE Midpoint: density-weighted correction diagnostics  "
        r"(PBME: $Q\equiv 0$, excluded)",
        fontsize=_TITLE_FONT)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=_LABEL_FONT, framealpha=0.9)
    ax.grid(False)
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)

    path = savepath or (os.path.join(out_dir, "fig_qcle_diagnostics.png") if out_dir else None)
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig

def produce_all_marginal_slices(
    snaps: Dict[str, "Snapshot"], out_dir: str,
    step_tag: str = "",
    n_grid_2d: int = 70, n_grid_1d: int = 200,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Produce the full 6+15 set of density marginals — one figure each.

    `step_tag` is appended to filenames (e.g. "_step15").
    """
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, str] = {}

    for ax_name in _AXES_6:
        fn = f"fig_marginal_1d_{ax_name}{step_tag}.png"
        p  = os.path.join(out_dir, fn)
        plot_density_1d_marginal(snaps, ax_name, savepath=p,
                                 n_grid=n_grid_1d)
        out[f"1d_{ax_name}"] = p
        if verbose: print(f"    wrote {p}")

    for pair in _ALL_2D_PAIRS:
        key = f"{pair[0]}_{pair[1]}"
        fn = f"fig_marginal_2d_{key}{step_tag}.png"
        p  = os.path.join(out_dir, fn)
        plot_density_2d_marginal(snaps, pair, savepath=p,
                                 n_grid=n_grid_2d)
        out[f"2d_{key}"] = p
        if verbose: print(f"    wrote {p}")

    # Faithfulness-Test #3 figures: GP-integral marginal vs cloud-Riemann
    # KDE side-by-side, per scheme.  Only the (R, P) pair is generated
    # by default since that's the diagnostic of interest for the focused
    # post-crossing regime; to add other axis pairs, append to the loop.
    for pair in (("R", "P"),):
        key = f"{pair[0]}_{pair[1]}"
        fn = f"fig_faithfulness_2d_{key}{step_tag}.png"
        p  = os.path.join(out_dir, fn)
        plot_faithfulness_2d_marginal(snaps, pair, savepath=p,
                                       n_grid=n_grid_2d)
        out[f"faith_2d_{key}"] = p
        if verbose: print(f"    wrote {p}")

    plt.close("all")
    return out


# =============================================================================
# Master entry point
# =============================================================================

def produce_all_comparison_figures(
    pbme_path_no_ext: str,
    midpoint_path_no_ext: str,
    out_dir: str,
    snapshot_step: Optional[int] = None,
    snapshot_stride: Optional[int] = None,
) -> Dict[str, str]:
    """
    Generate and save all comparison figures, organized into sub-directories:

        {out_dir}/
          time_series/          conservation, populations, coherences, nuclear,
                                local_energy, signed_weights, sampling_stats
          fit_quality/          one file per GP diagnostic panel
          correction/           one file per Q-correction panel (t>0 only)
          mapping_moments/      one file per quadratic moment
          marginals/
            step{K}/            1D and 2D density slices at snapshot step K

    Returns a flat dict {label -> absolute_path} for all saved figures.
    """
    os.makedirs(out_dir, exist_ok=True)
    runs = {"pbme":     load_run(pbme_path_no_ext),
            "midpoint": load_run(midpoint_path_no_ext)}

    # Sub-directory helpers
    def _subdir(*parts: str) -> str:
        d = os.path.join(out_dir, *parts)
        os.makedirs(d, exist_ok=True)
        return d

    out: Dict[str, str] = {}

    # ── analytic/ — GP-surrogate observables: PBME vs Midpoint ──────────────
    an_dir = _subdir("analytic")

    for key, fig in plot_populations_analytic(runs, out_dir=an_dir).items():
        out[f"an_pop_{key}"] = os.path.join(an_dir, f"fig_an_pop_{key.replace('dp_','').replace('ap_','')}.png")
        plt.close(fig)

    for key, fig in plot_coherences_analytic(runs, out_dir=an_dir).items():
        out[f"an_coh_{key}"] = os.path.join(an_dir, f"fig_an_coh_{key.replace('dc_coh_','')}.png")
        plt.close(fig)

    for key, fig in plot_energy_analytic(runs, out_dir=an_dir).items():
        out[f"an_energy_{key}"] = os.path.join(an_dir, f"fig_an_energy_{key.replace('km_','')}.png")
        plt.close(fig)

    for key, fig in plot_nuclear_analytic(runs, out_dir=an_dir).items():
        out[f"an_nuc_{key}"] = os.path.join(an_dir, f"fig_an_nuc_{key.replace('nm_','')}.png")
        plt.close(fig)

    for key, fig in plot_mapping_analytic(runs, out_dir=an_dir).items():
        out[f"an_map_{key}"] = os.path.join(an_dir, f"fig_an_qm_{key.replace('qm_','')}.png")
        plt.close(fig)

    # ── mc/ — cloud-weighted Monte Carlo observables: PBME vs Midpoint ───────
    mc_dir = _subdir("mc")

    for key, fig in plot_populations_mc(runs, out_dir=mc_dir).items():
        out[f"mc_pop_{key}"] = os.path.join(mc_dir, f"fig_mc_pop_{key.replace('cloud_weighted_','')}.png")
        plt.close(fig)

    for key, fig in plot_coherences_mc(runs, out_dir=mc_dir).items():
        out[f"mc_coh_{key}"] = os.path.join(mc_dir, f"fig_mc_coh_{key}.png")
        plt.close(fig)

    for key, fig in plot_energy_mc(runs, out_dir=mc_dir).items():
        out[f"mc_energy_{key}"] = os.path.join(mc_dir, f"fig_mc_energy_{key.replace('cloud_weighted_','')}.png")
        plt.close(fig)

    for key, fig in plot_nuclear_mc(runs, out_dir=mc_dir).items():
        out[f"mc_nuc_{key}"] = os.path.join(mc_dir, f"fig_mc_nuc_{key.replace('cloud_weighted_','')}.png")
        plt.close(fig)

    for key, fig in plot_mapping_mc(runs, out_dir=mc_dir).items():
        out[f"mc_map_{key}"] = os.path.join(mc_dir, "fig_mc_qm_radius_sq.png")
        plt.close(fig)

    # ── time_series/ — scalar conservation / weight diagnostics ──────────────
    ts_dir = _subdir("time_series")
    for name, func in [
        ("conservation",    plot_conservation),
        ("signed_weights",  plot_signed_weight_diagnostics),
        ("sampling_stats",  plot_sampling_statistics),
        ("density_diff",    plot_density_diff_diagnostics),
    ]:
        p = os.path.join(ts_dir, f"fig_{name}.png")
        func(runs, savepath=p)
        out[name] = p
        plt.close("all")

    # ── fit_quality/ ─────────────────────────────────────────────────────────
    fq_dir = _subdir("fit_quality")
    fq_figs = plot_fit_quality_panels(runs, out_dir=fq_dir)
    for key, fig in fq_figs.items():
        out[f"fit_{key}"] = os.path.join(fq_dir, f"fig_fit_{key}.png")
        plt.close(fig)

    # ── correction/ ──────────────────────────────────────────────────────────
    corr_dir = _subdir("correction")
    corr_figs = plot_correction_panels(runs, out_dir=corr_dir)
    for key, fig in corr_figs.items():
        out[f"correction_{key}"] = os.path.join(corr_dir, f"fig_correction_{key}.png")
        plt.close(fig)

    # Density-weighted QCLE diagnostics (rho-weighted mean/RMS, Σρc, max|Q|)
    p_qcle = os.path.join(corr_dir, "fig_qcle_diagnostics.png")
    fig_qcle = plot_qcle_correction_diagnostics(runs, savepath=p_qcle)
    out["qcle_diagnostics"] = p_qcle
    plt.close(fig_qcle)

    # ── mapping_moments/ ─────────────────────────────────────────────────────
    mm_dir = _subdir("mapping_moments")
    mm_figs = plot_mapping_moment_panels(runs, out_dir=mm_dir)
    for key, fig in mm_figs.items():
        out[f"qm_{key}"] = os.path.join(mm_dir, f"fig_qm_{key}.png")
        plt.close(fig)

    # ── marginals/step{K}/ ───────────────────────────────────────────────────
    panel_steps = []
    if snapshot_stride is not None and snapshot_stride > 0:
        common_steps = None
        for r in runs.values():
            steps = set(r["snapshots"].keys())
            common_steps = steps if common_steps is None else (common_steps & steps)
        common_steps = sorted(common_steps) if common_steps is not None else []
        panel_steps = [s for s in common_steps if (s == 0 or s % snapshot_stride == 0)]
        if common_steps:
            final_step = common_steps[-1]
            if final_step not in panel_steps:
                panel_steps.append(final_step)
    elif snapshot_step is not None:
        panel_steps = [snapshot_step]

    for step in panel_steps:
        snaps = {k: r["snapshots"].get(step) for k, r in runs.items()}
        snaps = {k: v for k, v in snaps.items() if v is not None}
        if len(snaps) >= 1:
            step_dir = _subdir("marginals", f"step{step:06d}")
            tag = f"_step{step}"
            marg = produce_all_marginal_slices(snaps, out_dir=step_dir,
                                               step_tag=tag, verbose=False)
            out.update({f"marg_step{step}_{k}": v for k, v in marg.items()})
            plt.close("all")

    plt.close("all")
    return out



# =============================================================================
# Universal single-observable time-series plotter (research-quality style)
# =============================================================================

def _plot_one(
    runs: Dict[str, Dict],
    keys: Sequence[str],
    ylabel: str,
    title: str = "",
    savepath: Optional[str] = None,
    yscale: str = "linear",
    hline: Optional[float] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> Optional[plt.Figure]:
    """One JCP-column figure for a single scalar time-series.

    Parameters
    ----------
    runs      : ``{scheme_name: run_dict}`` from :func:`load_run`.
    keys      : NPZ key names tried in order; first finite match is used.
    ylabel    : y-axis label (LaTeX accepted).
    yscale    : ``"linear"`` or ``"log"``.
    hline     : draw a horizontal reference line at this y-value.
    """
    fig, ax = plt.subplots(figsize=(_W15, 2.4))
    plotted = False
    for name, run in runs.items():
        a = run.get("arrays", run)
        t = a.get("t")
        if t is None:
            continue
        y = None
        for k in keys:
            arr = a.get(k)
            if arr is not None and np.any(np.isfinite(arr)):
                y = np.asarray(arr, dtype=float)
                break
        if y is None:
            continue
        kw = _scheme_kw(name)
        ax.plot(t, y, **kw, label=name.upper())
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    if hline is not None:
        ax.axhline(hline, color="0.45", lw=0.75, ls=":")
    ax.set_yscale(yscale)
    if ylim:
        ax.set_ylim(*ylim)
    _setup(ax, title, r"$t$ [a.u.]", ylabel)
    ax.legend(fontsize=_LEGEND_FONT, loc="best")
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    _save(fig, savepath)
    return fig


# =============================================================================
# Individual time-series figures — one file per observable
# =============================================================================

def produce_individual_timeseries_figures(
    pbme_path_no_ext: str,
    midpoint_path_no_ext: str,
    out_dir: str,
) -> Dict[str, str]:
    """Write one JCP-column PDF + PNG per physical time-series observable.

    Layout::

        {out_dir}/
          populations/   P0  P1  Psum
          coherences/    coh_re  coh_im  coh_abs
          nuclear/       R_mean  P_mean  R_var  P_var
          energy/        E_phys
          conservation/  norm  delta_norm  delta_E
          weights/       essf_y  essf_w  chi  essf_c

    Returns ``{label: pdf_path}``.
    """
    os.makedirs(out_dir, exist_ok=True)
    runs = {
        "pbme":     load_run(pbme_path_no_ext),
        "midpoint": load_run(midpoint_path_no_ext),
    }

    def _sub(*parts: str) -> str:
        d = os.path.join(out_dir, *parts)
        os.makedirs(d, exist_ok=True)
        return d

    out: Dict[str, str] = {}

    def _w(label: str, keys: Sequence[str], ylabel: str,
           subdir: str, fname: str, **kw) -> None:
        p = os.path.join(subdir, fname)
        fig = _plot_one(runs, keys, ylabel, savepath=p, **kw)
        if fig is not None:
            out[label] = p + ".pdf"

    # ── populations ──────────────────────────────────────────────────────────
    d = _sub("populations")
    _w("P0",   ["lw_P0",    "cloud_weighted_P0"],   r"$P_0(t)$",            d, "P0",
       ylim=(0.0, 1.05))
    _w("P1",   ["lw_P1",    "cloud_weighted_P1"],   r"$P_1(t)$",            d, "P1",
       ylim=(0.0, 1.05))
    _w("Psum", ["lw_P_sum", "cloud_weighted_trace"], r"$P_0 + P_1$",        d, "Psum",
       hline=1.0, ylim=(0.95, 1.05))

    # ── coherences ───────────────────────────────────────────────────────────
    d = _sub("coherences")
    _w("coh_re",  ["lw_coh_re",  "dc_coh_re"],
       r"$\mathrm{Re}\,\rho^{\mathrm{el}}_{01}(t)$",  d, "coh_re")
    _w("coh_im",  ["lw_coh_im",  "dc_coh_im"],
       r"$\mathrm{Im}\,\rho^{\mathrm{el}}_{01}(t)$",  d, "coh_im")
    _w("coh_abs", ["lw_coh_abs", "dc_coh_abs"],
       r"$|\rho^{\mathrm{el}}_{01}(t)|$",              d, "coh_abs",
       ylim=(0.0, None))

    # ── nuclear moments ───────────────────────────────────────────────────────
    d = _sub("nuclear")
    _w("R_mean", ["cloud_mean_R", "nm_R_mean"],  r"$\langle R\rangle$ [a.u.]", d, "R_mean")
    _w("P_mean", ["cloud_mean_P", "nm_P_mean"],  r"$\langle P\rangle$ [a.u.]", d, "P_mean")
    _w("R_var",  ["cloud_var_R",  "nm_R_var"],
       r"$\mathrm{Var}(R)$ [a.u.$^2$]",  d, "R_var")
    _w("P_var",  ["cloud_var_P",  "nm_P_var"],
       r"$\mathrm{Var}(P)$ [a.u.$^2$]",  d, "P_var")

    # ── energy ────────────────────────────────────────────────────────────────
    d = _sub("energy")
    _w("E_phys", ["lw_energy", "spe_E_density"],
       r"$\langle H\rangle$ [a.u.]",  d, "E_phys")

    # ── conservation ─────────────────────────────────────────────────────────
    d = _sub("conservation")
    _w("norm",       ["km_normalization"],
       r"$\int\hat{\rho}\,\mathrm{d}z$",       d, "norm",       hline=1.0)
    _w("delta_norm", ["gp_vs_cloud_norm_rel"],
       r"$|\Delta\,\mathrm{norm}(t)|$",          d, "delta_norm", yscale="log")
    _w("delta_E",    ["gp_vs_cloud_energy_rel"],
       r"$|\Delta E(t)|$ [a.u.]",                d, "delta_E",    yscale="log")

    # ── weight / sampling quality ─────────────────────────────────────────────
    d = _sub("weights")
    _w("essf_y", ["sw_ess_frac"],
       r"$\mathrm{ESS}_y / N$  (label ESS)",          d, "essf_y",
       ylim=(0.0, 1.05))
    _w("essf_w", ["init_sw_ess_frac"],
       r"$\mathrm{ESS}_w / N$  (weight ESS)",          d, "essf_w",
       ylim=(0.0, 1.05))
    _w("chi",    ["sw_cancel_ratio"],
       r"$\chi$  (sign cancellation ratio)",           d, "chi",
       ylim=(0.0, 1.05))
    _w("essf_c", ["ce_ess_c_frac"],
       r"$\mathrm{ESS}_c / N$  (coefficient vector)",  d, "essf_c",
       ylim=(0.0, 1.05))

    return out


# =============================================================================
# GP surrogate health / reliability figures — one file per diagnostic
# =============================================================================

def _plot_qcle_correction_combined(
    mid_only: Dict[str, Dict],
    savepath: str,
) -> Optional[plt.Figure]:
    """Combined QCLE correction diagnostic plot replicating the reference figure.

    Four density-weighted statistics of Q (the extra Liouvillian term) on one
    axis — PBME excluded because Q ≡ 0 there by construction.

    Curves
    ------
    ρ-weighted mean   cs_q_y_weighted_mean   orange  solid
    ρ-weighted RMS    cs_q_y_weighted_rms    blue    solid
    support max|Q|    cs_q_max               magenta solid
    Σ_m ρ_m Q_m       cs_q_sum_yc            green   dashed
    """
    _CSTYLE = [
        ("cs_q_y_weighted_mean", r"$\rho$-weighted mean",    "#E69F00", "-",  1.25),
        ("cs_q_y_weighted_rms",  r"$\rho$-weighted RMS",     "#0072B2", "-",  1.25),
        ("cs_q_max",             r"support $\max|Q|$",       "#CC79A7", "-",  1.0),
        ("cs_q_sum_yc",          r"$\sum_m \rho_m Q_m$",    "#009E73", "--", 1.25),
    ]

    fig, ax = plt.subplots(figsize=(_W2, 2.8))
    plotted = False
    for name, run in mid_only.items():
        a = run.get("arrays", run)
        t = a.get("t")
        if t is None:
            continue
        for key, label, color, ls, lw in _CSTYLE:
            arr = a.get(key)
            if arr is None or not np.any(np.isfinite(arr)):
                continue
            ax.plot(t, np.asarray(arr, dtype=float),
                    color=color, ls=ls, lw=lw, label=label)
            plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.axhline(0.0, color="0.5", lw=0.5, ls="-")
    ax.set_title(
        r"QCLE Midpoint: density-weighted correction diagnostics"
        r"  (PBME: $Q\equiv 0$, excluded)",
        fontsize=_TITLE_FONT,
    )
    _setup(ax, "", r"$t$  [a.u.]", r"QCLE correction diagnostics")
    ax.legend(fontsize=_LEGEND_FONT, loc="upper right",
              frameon=True, framealpha=0.85, edgecolor="0.75")
    fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
    _save(fig, savepath)
    return fig


def produce_surrogate_health_figures(
    pbme_path_no_ext: str,
    midpoint_path_no_ext: str,
    out_dir: str,
) -> Dict[str, str]:
    """Write one JCP-column PDF + PNG per GP-surrogate reliability diagnostic.

    Folders are named to make their physical meaning explicit:

    ``gp_reconstruction/``   — How faithfully the GP re-produces the training
                               labels (fit RMS, R², MAE, Liouville residual).
    ``gp_prediction/``       — Predictive accuracy at unseen points via
                               leave-one-out cross-validation (LOO).
    ``gp_coefficients/``     — Health of the α expansion coefficients
                               (ESS, sign-oscillation, condition number).
    ``gp_kernel/``           — Kernel hyperparameters (σ_f, σ_n, SNR,
                               log κ(K), lengthscales per phase-space dim).
    ``qcle_correction/``     — Magnitude of the extra quantum-classical
                               Liouvillian term Q applied to the density by
                               the midpoint scheme only.  Multiple metrics:
                               raw ‖Q‖_rms, applied ‖dt Q‖_rms, relative
                               ‖Q/ρ̂‖_rms, and the clipping/overflow counts.

    Returns ``{label: pdf_path}``.
    """
    os.makedirs(out_dir, exist_ok=True)
    runs = {
        "pbme":     load_run(pbme_path_no_ext),
        "midpoint": load_run(midpoint_path_no_ext),
    }

    def _sub(*parts: str) -> str:
        d = os.path.join(out_dir, *parts)
        os.makedirs(d, exist_ok=True)
        return d

    out: Dict[str, str] = {}

    def _w(label: str, keys: Sequence[str], ylabel: str,
           subdir: str, fname: str, **kw) -> None:
        p = os.path.join(subdir, fname)
        fig = _plot_one(runs, keys, ylabel, savepath=p, **kw)
        if fig is not None:
            out[label] = p + ".pdf"

    # midpoint-only dict: QCLE correction is only computed for this scheme
    mid_only = {k: v for k, v in runs.items() if k.lower() != "pbme"}

    def _wm(label: str, keys: Sequence[str], ylabel: str,
            subdir: str, fname: str, **kw) -> None:
        """Emit a figure for midpoint-only keys."""
        if not mid_only:
            return
        p = os.path.join(subdir, fname)
        fig = _plot_one(mid_only, keys, ylabel, savepath=p, **kw)
        if fig is not None:
            out[label] = p + ".pdf"

    # ── GP reconstruction at training points ──────────────────────────────────
    # These measure how faithfully ρ̂ reproduces the supplied labels y_i.
    d = _sub("gp_reconstruction")
    _w("fit_rms",
       ["fit_rms_on_support"],
       r"$\|\hat{\rho}(Z_i) - y_i\|_{\mathrm{rms}}$ — GP reconstruction RMS",
       d, "fit_rms",  yscale="log")
    _w("fit_r2",
       ["gp_fit_r2"],
       r"$R^2$ — GP reconstruction coefficient",
       d, "fit_r2",   hline=1.0, ylim=(0.999, 1.0001))
    _w("fit_mae",
       ["gp_fit_mae"],
       r"$\|\hat{\rho}(Z_i) - y_i\|_{\mathrm{mae}}$ — GP reconstruction MAE",
       d, "fit_mae",  yscale="log")
    _w("liou_rms",
       ["gp_liouville_rms"],
       r"$\|\mathcal{L}\hat{\rho}\|_{\mathrm{rms}}$ — Liouville residual",
       d, "liou_rms", yscale="log")

    # ── GP predictive accuracy (leave-one-out cross-validation) ──────────────
    # LOO measures how well ρ̂ predicts withheld support points.
    d = _sub("gp_prediction")
    _w("LOO_rms",
       ["faith_loo_rms"],
       r"$\mathrm{LOO\text{-}CV}_{\mathrm{rms}}$ — GP predictive RMS",
       d, "LOO_rms",  yscale="log")
    _w("LOO_max",
       ["faith_loo_max"],
       r"$\mathrm{LOO\text{-}CV}_{\max}$ — GP predictive max error",
       d, "LOO_max",  yscale="log")
    _w("LOO_n3sig",
       ["faith_loo_n_3sig"],
       r"LOO $3\sigma$ outlier count",
       d, "LOO_n3sig", hline=0.0)
    _w("pred_rms",
       ["faith_predict_rms"],
       r"Posterior predictive RMS — GP reconstruction at new points",
       d, "pred_rms", yscale="log")

    # ── GP coefficient (α) health ─────────────────────────────────────────────
    d = _sub("gp_coefficients")
    _w("ESS_alpha",
       ["faith_ess_alpha_frac"],
       r"$\mathrm{ESS}(\alpha)/N$ — GP coefficient sign coherence",
       d, "ESS_alpha",
       hline=0.35, ylim=(0.0, 1.05))
    _w("sign_align",
       ["faith_alpha_sign_align"],
       r"$|\Sigma\alpha|/\Sigma|\alpha|$ — GP sign alignment",
       d, "sign_align",
       hline=0.5, ylim=(0.0, 1.05))
    _w("alpha_neg_frac",
       ["alpha_neg_frac"],
       r"Fraction of $\alpha_i < 0$ — GP coefficient sign oscillation",
       d, "alpha_neg_frac",
       hline=0.5, ylim=(0.0, 1.0))
    _w("alpha_l1",
       ["alpha_l1"],
       r"$\|\alpha\|_1$ — GP coefficient L1 norm",
       d, "alpha_l1",  yscale="log")

    # ── GP kernel hyperparameters ─────────────────────────────────────────────
    d = _sub("gp_kernel")
    _w("sigma_n",
       ["sigma_n"],
       r"$\sigma_n$ — GP noise level",
       d, "sigma_n",  yscale="log")
    _w("sigma_f",
       ["sigma_f"],
       r"$\sigma_f$ — GP signal amplitude",
       d, "sigma_f")
    _w("snr",
       ["sigma_f_over_sigma_n"],
       r"$\sigma_f/\sigma_n$ — GP signal-to-noise ratio",
       d, "snr",      yscale="log")
    _w("log_kappa",
       ["faith_cond_K_lo_log10"],
       r"$\log_{10}\kappa(K)$ — GP kernel matrix condition number",
       d, "log_kappa", hline=12.0)

    # Lengthscales — one figure per phase-space dimension
    _DIM_NAMES = ["R", "P", "r_0", "r_1", "p_0", "p_1"]
    _DIM_LATEX = [r"$\ell_R$", r"$\ell_P$",
                  r"$\ell_{r_0}$", r"$\ell_{r_1}$",
                  r"$\ell_{p_0}$", r"$\ell_{p_1}$"]
    n_dims = 0
    for run in runs.values():
        a = run.get("arrays", run)
        ell = a.get("lengthscales")
        if ell is not None:
            ell = np.asarray(ell)
            if ell.ndim == 2:
                n_dims = ell.shape[1]
                break
    for dim in range(n_dims):
        lbl  = _DIM_NAMES[dim] if dim < len(_DIM_NAMES) else f"d{dim}"
        ylab = (_DIM_LATEX[dim] if dim < len(_DIM_LATEX) else rf"$\ell_{{{dim}}}$")
        ylab += r" — GP lengthscale"
        fname = f"ell_{lbl.replace(' ', '_')}"
        fig, ax = plt.subplots(figsize=(_W15, 2.4))
        plotted = False
        for name, run in runs.items():
            a = run.get("arrays", run)
            t = a.get("t")
            ell = a.get("lengthscales")
            if t is None or ell is None:
                continue
            ell = np.asarray(ell)
            if ell.ndim != 2 or ell.shape[1] <= dim:
                continue
            ax.plot(t, ell[:, dim], **_scheme_kw(name), label=name.upper())
            plotted = True
        if plotted:
            _setup(ax, "", r"$t$ [a.u.]", ylab)
            ax.legend(fontsize=_LEGEND_FONT)
            fig.tight_layout(pad=0.4, h_pad=0.5, w_pad=0.5)
            p = os.path.join(d, fname)
            _save(fig, p)
            out[f"ell_{lbl}"] = p + ".pdf"
        else:
            plt.close(fig)

    # ── QCLE correction: extra Liouvillian term ‖Q‖ on the density ───────────
    # Q is the quantum-classical correction beyond PBME:
    #   ∂ρ̂/∂t|_extra = Q[ρ̂]
    # Applied ONLY by the midpoint scheme; PBME has Q ≡ 0 by construction.
    # Multiple metrics show raw magnitude, timestep-scaled, and relative.
    d = _sub("qcle_correction")

    # ---- Combined figure (matches the reference plot) ----
    combined = _plot_qcle_correction_combined(
        mid_only, savepath=os.path.join(d, "Q_combined"))
    if combined is not None:
        out["Q_combined"] = os.path.join(d, "Q_combined.pdf")

    # ---- Individual metrics ----
    _wm("cs_q_rms",
        ["cs_q_rms"],
        r"$\|Q\|_{\mathrm{rms}}$ — raw QCLE correction operator",
        d, "Q_rms")
    _wm("cs_q_max",
        ["cs_q_max"],
        r"$\|Q\|_{\max}$ — QCLE correction peak magnitude",
        d, "Q_max")
    _wm("cs_q_y_weighted_rms",
        ["cs_q_y_weighted_rms"],
        r"$\rho$-weighted RMS of $Q$",
        d, "Q_rho_rms")
    _wm("cs_q_y_weighted_mean",
        ["cs_q_y_weighted_mean"],
        r"$\rho$-weighted mean of $Q$",
        d, "Q_rho_mean")
    _wm("cs_q_sum_yc",
        ["cs_q_sum_yc"],
        r"$\sum_m \rho_m Q_m$ — density-weighted correction integral",
        d, "Q_sum_rho")
    _wm("cs_dtq_rms",
        ["cs_dtq_rms"],
        r"$\|dt \cdot Q\|_{\mathrm{rms}}$ — applied correction magnitude",
        d, "dtQ_rms")
    _wm("cs_dq_over_y_rms",
        ["cs_dq_over_y_rms"],
        r"$\|Q/\hat{\rho}\|_{\mathrm{rms}}$ — relative QCLE correction",
        d, "Q_over_rho_rms", yscale="log")
    _wm("applied_cs_q_rms",
        ["applied_cs_q_rms"],
        r"$\|Q_{\mathrm{applied}}\|_{\mathrm{rms}}$ — post-clip QCLE correction",
        d, "Q_applied_rms")
    _wm("n_q_clipped",
        ["n_q_clipped"],
        r"$N_{\mathrm{clip}}$ — correction values clipped per step",
        d, "n_clipped",    hline=0.0)
    _wm("n_q_nonfinite",
        ["n_q_nonfinite"],
        r"$N_{\mathrm{nan}}$ — non-finite correction values per step",
        d, "n_nonfinite",  hline=0.0)

    return out


# =============================================================================
# produce_all_figures_publication — replaces produce_all_comparison_figures
# =============================================================================

def produce_all_figures_publication(
    pbme_path_no_ext: str,
    midpoint_path_no_ext: str,
    out_dir: str,
    snapshot_step: Optional[int] = None,
    snapshot_stride: Optional[int] = None,
) -> Dict[str, str]:
    """Full publication figure set: individual time-series + surrogate health.

    This supersedes :func:`produce_all_comparison_figures` for journal/research-report
    output.  Every observable and every diagnostic is a separate PDF + PNG file;
    no multi-panel figures are produced.

    Output layout::

        {out_dir}/
          populations/          P0  P1  Psum
          coherences/           coh_re  coh_im  coh_abs
          nuclear/              R_mean  P_mean  R_var  P_var
          energy/               E_phys
          conservation/         norm  delta_norm  delta_E
          weights/              essf_y  essf_w  chi  essf_c
          surrogate_health/
            fit_quality/        fit_rms  fit_r2  fit_mae  liou_rms
            loo/                LOO_rms  LOO_max  LOO_n3sig  pred_rms
            alpha_health/       ESS_alpha_f  negL1  neg_frac  minmax  l1  linf
            hyperparameters/    sigma_n  sigma_f  snr  log_kappa
            lengthscales/       ell_R  ell_P  ell_r0  ell_r1  ell_p0  ell_p1
            correction/         q_rms
          marginals/step{K}/    GP density slices at snapshot steps

    Returns ``{label: absolute_pdf_path}`` for every figure produced.
    """
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, str] = {}

    # ── individual physics time-series ────────────────────────────────────────
    ts = produce_individual_timeseries_figures(
        pbme_path_no_ext, midpoint_path_no_ext, out_dir)
    out.update(ts)

    # ── surrogate health diagnostics ──────────────────────────────────────────
    health_dir = os.path.join(out_dir, "surrogate_health")
    health = produce_surrogate_health_figures(
        pbme_path_no_ext, midpoint_path_no_ext, health_dir)
    out.update({f"health_{k}": v for k, v in health.items()})

    # ── density marginals at snapshot steps ───────────────────────────────────
    runs = {"pbme":     load_run(pbme_path_no_ext),
            "midpoint": load_run(midpoint_path_no_ext)}

    def _subdir(*parts: str) -> str:
        d = os.path.join(out_dir, *parts)
        os.makedirs(d, exist_ok=True)
        return d

    panel_steps: list = []
    if snapshot_stride is not None and snapshot_stride > 0:
        common: Optional[set] = None
        for r in runs.values():
            s = set(r["snapshots"].keys())
            common = s if common is None else common & s
        common_sorted = sorted(common) if common else []
        panel_steps = [s for s in common_sorted
                       if s == 0 or s % snapshot_stride == 0]
        if common_sorted and common_sorted[-1] not in panel_steps:
            panel_steps.append(common_sorted[-1])
    elif snapshot_step is not None:
        panel_steps = [snapshot_step]

    for step in panel_steps:
        snaps = {k: r["snapshots"].get(step) for k, r in runs.items()}
        snaps = {k: v for k, v in snaps.items() if v is not None}
        if snaps:
            step_dir = _subdir("marginals", f"step{step:06d}")
            marg = produce_all_marginal_slices(
                snaps, out_dir=step_dir,
                step_tag=f"_step{step}", verbose=False)
            out.update({f"marg_step{step}_{k}": v for k, v in marg.items()})
            plt.close("all")

    plt.close("all")
    return out
