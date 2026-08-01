"""Build reader-facing numerical tables for Chapters 6 and 7.

The numerical archive remains the source of the calculations, but the output
of this script is deliberately written as a scientific results section: it
contains model parameters, observables, uncertainty, and physical
interpretation, without software paths or execution metadata.
"""
from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "final_reviewer_closure"
OUTPUT = ROOT / "Thesis" / "PhysicsResultsTables.tex"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def value(row: Mapping[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def mean(values: Iterable[float]) -> float:
    finite = [x for x in values if math.isfinite(x)]
    return statistics.fmean(finite) if finite else math.nan


def sample_sd(values: Iterable[float]) -> float:
    finite = [x for x in values if math.isfinite(x)]
    return statistics.stdev(finite) if len(finite) > 1 else 0.0


def fmt(x: float, digits: int = 3) -> str:
    if not math.isfinite(x):
        return r"---"
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1.0e3 or ax < 1.0e-3:
        mantissa_text, exponent_text = f"{x:.{digits - 1}e}".split("e")
        return rf"{mantissa_text}\times10^{{{int(exponent_text)}}}"
    return f"{x:.{digits}g}"


# Largest observed order accepted as interpretable convergence evidence.
# Declared formal orders of the reference schemes do not exceed four; observed
# values far above this bound indicate that the finer difference has entered a
# cancellation- or roundoff-limited regime and the sequence is not asymptotic.
MAX_PLAUSIBLE_ORDER = 6.0


def pm(mu: float, sd: float) -> str:
    return rf"${fmt(mu)}\pm {fmt(sd)}$"


def mathnum(x: float) -> str:
    return rf"${fmt(x)}$"


def obs(name: str) -> str:
    labels = {
        "P0": r"$\rho_{11}^{\mathrm{SN}}$",
        "P1": r"$\rho_{22}^{\mathrm{SN}}$",
        "trace": r"electronic trace",
        "energy": r"energy",
        "R_mean": r"$\langle R\rangle$",
        "P_mean": r"$\langle P\rangle$",
        "R_var": r"$\operatorname{var}(R)$",
        "P_var": r"$\operatorname{var}(P)$",
        "mapping-integrated R-P density": r"$R$--$P$ density",
        "nuclear R marginal": r"nuclear $R$ marginal",
    }
    return labels.get(name, name.replace("_", r"\_"))


def method(name: str) -> str:
    return "MIDPOINT" if name.lower() == "midpoint" else "PBME"


def longtable(
    label: str,
    caption: str,
    columns: str,
    header: str,
    rows: Sequence[str],
    landscape: bool = False,
    font: str = r"\scriptsize",
) -> str:
    parts: list[str] = []
    if landscape:
        parts.append(r"\begin{landscape}")
    parts.extend(
        [
            r"\begingroup",
            font,
            r"\setlength{\tabcolsep}{3.5pt}",
            rf"\begin{{longtable}}{{{columns}}}",
            rf"\caption{{{caption}}}\label{{{label}}}\\",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            r"\endfirsthead",
            rf"\multicolumn{{{header.count('&') + 1}}}{{l}}{{\tablename\ \thetable\ (continued)}}\\",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            r"\endhead",
            r"\midrule",
            rf"\multicolumn{{{header.count('&') + 1}}}{{r}}{{Continued on next page}}\\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
        ]
    )
    parts.extend(row + r" \\" for row in rows)
    parts.extend([r"\end{longtable}", r"\endgroup"])
    if landscape:
        parts.append(r"\end{landscape}")
    return "\n".join(parts)


def study_design_table() -> str:
    rows = [
        r"Manufactured operator & $\rho$, $\nabla\rho$, and $Q[\rho]$ on and away from the training cloud & $\ell_2=10^{-6},0.01,0.05$; $N=300,600,1200,2400$; seeds 123--125 & Relative $L_1$, $L_2$, and $L_\infty$ errors",
        r"Time-step refinement & Dynamical observables at three common physical-time grids & $\Delta t=0.5,0.25,0.125$; $P_{\rm init}=20,100$; four independent seeds & Successive differences compared with seed dispersion",
        r"Cloud-size sensitivity & Effect of enlarging independently sampled phase-space clouds & $N=500,1000,2000$; three independent seeds & Mean, sample standard deviation, and change relative to seed dispersion",
        r"Replication & Sensitivity of endpoint observables to independently sampled initial clouds & Seeds 11, 29, 47, and 73; $N=1000$; $\Delta t=0.25$ & Mean, sample standard deviation, and Student-$t$ interval",
        r"Estimator structure & SEO leakage, raw invariants, and signed-label conditioning & Three collision-time snapshots and threshold sweep & Projection residual, raw drift, effective sample sizes, and excluded physical mass",
        r"Reference calculations & Numerical control of TDSE and grid-QCLE baselines & Separate three-level time and grid refinements & Values, successive differences, observed order, edge mass, and CFL ratio",
        r"Physical comparison & PBME and MIDPOINT against the same TDSE or grid-QCLE reference & Paired four-seed comparisons & Density and observable error differences with paired Student-$t$ intervals",
    ]
    return longtable(
        "tab:numerical-study-design",
        "Numerical studies used to distinguish interpolation accuracy, operator fidelity, dynamical stability, estimator structure, and physical predictive accuracy.  Here and below, $N$ denotes the number of independently sampled phase-space trajectories in a cloud.",
        r"L{0.16\textwidth}L{0.28\textwidth}L{0.25\textwidth}L{0.23\textwidth}",
        r"Question & Quantity examined & Controlled variation & Evidence reported",
        rows,
    )


def manufactured_tables() -> str:
    data = read_csv(EVIDENCE / "manufactured" / "manufactured_complete.csv")
    domains = [("on_support", "training cloud"), ("off_support", "independent query cloud")]
    blocks: list[str] = []
    for quantity, qlabel, short_label in (
        ("density", r"density $\rho$", "density"),
        ("gradient", r"density gradient $\nabla\rho$", "gradient"),
        ("Q", r"excess action $Q[\rho]$", "operator"),
    ):
        rows: list[str] = []
        for l2 in (1.0e-6, 0.01, 0.05):
            for n in (300, 600, 1200, 2400):
                for domain_key, domain_label in domains:
                    group = [
                        row
                        for row in data
                        if math.isclose(value(row, "l2_regularization"), l2)
                        and int(value(row, "N")) == n
                        and row["query_type"] == domain_key
                    ]
                    metrics = []
                    for suffix in ("relative_l1", "relative_l2", "relative_linf"):
                        vals = [value(row, f"{quantity}_{suffix}") for row in group]
                        metrics.append(pm(mean(vals), sample_sd(vals)))
                    rows.append(
                        rf"${fmt(l2)}$ & {n} & {domain_label} & " + " & ".join(metrics)
                    )
        blocks.append(
            longtable(
                f"tab:manufactured-{short_label}",
                rf"Manufactured-function errors for the reconstructed {qlabel}.  Entries are mean $\pm$ sample standard deviation over three independently generated training/query-cloud pairs.  The relative norms use the corresponding analytic quantity in the denominator.",
                r"r r L{0.25\textwidth} r r r",
                r"$\ell_2$ & $N$ & Evaluation domain & $E_1$ & $E_2$ & $E_\infty$",
                rows,
            )
        )
    return "\n\n".join(blocks)


def timestep_table() -> str:
    data = read_csv(EVIDENCE / "timestep" / "timestep_run_by_run.csv")
    grouped: dict[tuple[str, float, str], list[dict[str, str]]] = defaultdict(list)
    for row in data:
        grouped[(row["method"], value(row, "P0"), row["observable"])].append(row)
    order = ["P0", "P1", "trace", "energy", "R_mean", "P_mean", "R_var", "P_var"]
    rows: list[str] = []
    for m in ("PBME", "MIDPOINT"):
        for p0 in (20.0, 100.0):
            for observable in order:
                group = grouped[(m, p0, observable)]
                means = [mean(value(row, key) for row in group) for key in ("value1", "value2", "value3")]
                d12 = mean(value(row, "D12") for row in group)
                d23 = mean(value(row, "D23") for row in group)
                spread = mean(value(row, "pooled_seed_spread") for row in group)
                noise = max(value(row, "roundoff_threshold") for row in group)
                if min(d12, d23) <= noise:
                    interpretation = (
                        "roundoff- or saturation-limited; order not interpreted"
                    )
                elif d23 >= d12:
                    interpretation = "nonmonotone step-size response"
                elif min(d12, d23) <= spread:
                    interpretation = "step-size signal is not resolved above seed spread"
                else:
                    interpretation = "contracting signal resolved above noise and seed spread"
                rows.append(
                    rf"{m} & {int(p0)} & {obs(observable)} & "
                    + " & ".join(mathnum(x) for x in means)
                    + rf" & {mathnum(d12)} & {mathnum(d23)} & {interpretation}"
                )
    return longtable(
        "tab:timestep-refinement-physics",
        "Time-step sensitivity of endpoint observables. Values and time-normalized successive differences are averages over seeds 11, 29, 47, and 73 after interpolation to common physical times without extrapolation. Both differences must exceed the declared absolute-plus-relative floor $\\tau_{\\rm noise}=10^{-12}+10^{-12}\\max_k\\operatorname{RMS}(O_k)$ before an order is interpreted. For both stochastic moving-cloud methods, PBME and MIDPOINT, both differences must also exceed pooled independent-seed dispersion.",
        r"l r l r r r r r L{0.20\textwidth}",
        r"Method & $P_{\rm init}$ & Observable & $\langle y_{0.5}\rangle$ & $\langle y_{0.25}\rangle$ & $\langle y_{0.125}\rangle$ & $\langle D_{12}\rangle$ & $\langle D_{23}\rangle$ & Interpretation",
        rows,
        landscape=True,
    )


def cloud_size_table() -> str:
    data = read_csv(EVIDENCE / "support" / "independent_cloud_summary.csv")
    grouped: dict[tuple[str, float, str], dict[int, dict[str, str]]] = defaultdict(dict)
    for row in data:
        grouped[(row["method"], value(row, "P0"), row["observable"])][int(value(row, "N"))] = row
    order = ["P0", "P1", "trace", "energy", "R_mean", "P_mean", "R_var", "P_var"]
    rows: list[str] = []
    for m in ("PBME", "MIDPOINT"):
        for p0 in (20.0, 100.0):
            for observable in order:
                levels = grouped[(m, p0, observable)]
                cells = [pm(value(levels[n], "mean"), value(levels[n], "sample_sd")) for n in (500, 1000, 2000)]
                flagged = any(levels[n]["change_exceeds_seed_variability"].lower() == "true" for n in (500, 1000))
                interpretation = "cloud-size change exceeds seed spread" if flagged else "change remains within seed spread"
                rows.append(rf"{m} & {int(p0)} & {obs(observable)} & " + " & ".join(cells) + rf" & {interpretation}")
    return longtable(
        "tab:independent-cloud-size",
        r"Sensitivity to enlarging independently sampled trajectory clouds.  Each entry is mean $\pm$ sample standard deviation over seeds 11, 29, and 47.  Because the clouds are independently sampled rather than nested, this is a stochastic cloud-size test, not a deterministic convergence order.",
        r"l r l r r r L{0.24\textwidth}",
        r"Method & $P_{\rm init}$ & Observable & $N=500$ & $N=1000$ & $N=2000$ & Interpretation",
        rows,
        landscape=True,
    )


def replication_table() -> str:
    data = read_csv(EVIDENCE / "replication" / "four_seed_summary.csv")
    lookup = {(row["method"], value(row, "P0"), row["observable"]): row for row in data}
    order = ["P0", "P1", "trace", "energy", "R_mean", "P_mean", "R_var", "P_var"]
    rows: list[str] = []
    for m in ("PBME", "MIDPOINT"):
        for p0 in (20.0, 100.0):
            for observable in order:
                row = lookup[(m, p0, observable)]
                rows.append(
                    rf"{m} & {int(p0)} & {obs(observable)} & {mathnum(value(row, 'mean'))} & {mathnum(value(row, 'sample_sd'))} & "
                    rf"$[{fmt(value(row, 'ci95_low'))},{fmt(value(row, 'ci95_high'))}]$ & {mathnum(value(row, 'maximum_spread'))}"
                )
    return longtable(
        "tab:four-seed-replication-physics",
        r"Four-seed endpoint replication at $N=1000$ and $\Delta t=0.25$.  The interval is a two-sided Student-$t$ interval with three degrees of freedom and is interpreted as a small-sample sensitivity measure, not as strong uncertainty calibration.",
        r"l r l r r r r",
        r"Method & $P_{\rm init}$ & Observable & Mean & Sample SD & 95\% interval & Max. spread",
        rows,
        landscape=True,
    )


def projection_and_baseline_tables() -> str:
    projection = read_csv(EVIDENCE / "preserved_evidence" / "projection_leakage.csv")
    grouped: dict[tuple[str, int, float], list[dict[str, str]]] = defaultdict(list)
    for row in projection:
        grouped[(row["method"], int(float(row["P0"])), value(row, "t_over_tc"))].append(row)
    rows: list[str] = []
    for m in ("pbme", "midpoint"):
        for p0 in (20, 100):
            for tau in (0.0, 1.0, 2.0):
                group = grouped[(m, p0, tau)]
                vals = [value(row, "mean_relative_l2_leakage") for row in group]
                maxvals = [value(row, "maximum_relative_l2_leakage") for row in group]
                rows.append(rf"{method(m)} & {p0} & ${fmt(tau)}$ & {pm(mean(vals), sample_sd(vals))} & {mathnum(max(maxvals))}")
    projection_table = longtable(
        "tab:seo-projection-physics",
        r"Relative $L_2$ distance from the four-field physical SEO image at three collision-time snapshots.  The entries are mean $\pm$ sample standard deviation over four independently propagated clouds; projection is diagnosed by least squares but is not imposed during propagation.",
        r"l r r r r",
        r"Method & $P_{\rm init}$ & $t/t_c$ & Mean leakage & Largest local leakage",
        rows,
    )
    baseline = read_csv(EVIDENCE / "preserved_evidence" / "kde_gp_baseline.csv")
    brows = []
    for row in sorted(baseline, key=lambda r: float(r["P0"])):
        criterion = "criterion met" if value(row, "max_E1") <= value(row, "threshold_E1") else "criterion not met"
        brows.append(
            rf"{int(value(row, 'P0'))} & {int(value(row, 'n_cases'))} & {mathnum(value(row, 'max_E1'))} & {mathnum(value(row, 'max_E2'))} & {mathnum(value(row, 'max_Einf'))} & $E_1\leq {fmt(value(row, 'threshold_E1'))}$ ({criterion})"
        )
    baseline_table = longtable(
        "tab:identical-cloud-reconstruction",
        "Identical-cloud KDE/GP reconstruction control.  Both estimators use the same PBME snapshot, weights, bandwidth convention, normalization, and evaluation grid.  The declared criterion concerns $E_1$ only and does not test derivatives or propagation.",
        r"r r r r r L{0.27\textwidth}",
        r"$P_{\rm init}$ & Cases & $\max E_1$ & $\max E_2$ & $\max E_\infty$ & Declared reconstruction criterion",
        brows,
    )
    return projection_table + "\n\n" + baseline_table


def tail_table() -> str:
    data = read_csv(EVIDENCE / "tail_sensitivity" / "tail_summary.csv")
    rows: list[str] = []
    for row in sorted(data, key=lambda r: (r["method"], float(r["P0"]), int(r["seed"]))):
        rows.append(
            rf"{row['method']} & {int(value(row, 'P0'))} & {row['seed']} & {mathnum(value(row, 'eta0_ratio_abs_max'))} & {mathnum(value(row, 'eta0_signed_ESS'))} & {mathnum(value(row, 'eta0_absolute_ESS'))} & {mathnum(value(row, 'eta0_raw_normalization'))} & {mathnum(value(row, 'first_excluding_eta'))} & {mathnum(value(row, 'first_excluding_absolute_mass_fraction'))}"
        )
    return longtable(
        "tab:signed-label-tail-physics",
        "Signed-label conditioning and tail sensitivity.  The first nonzero threshold excludes about $10^{-3}$ of the absolute physical mass, already above the predeclared negligible-mass allowance $10^{-6}$.  Consequently no nontrivial negligible-mass plateau exists from which to attribute the instability uniquely to a vanishing-$|y_i^0|$ tail.",
        r"l r r r r r r r r",
        r"Method & $P_{\rm init}$ & Seed & $\max|y/y^0|$ & $N_{\rm eff}^{\rm signed}$ & $N_{\rm eff}^{\rm abs}$ & Raw norm & First $\eta$ & Excluded mass",
        rows,
        landscape=True,
    )


def reference_tables() -> str:
    blocks: list[str] = []
    parameter_rows: list[str] = []
    for name, relpath, label in (
        ("TDSE", "reference_tdse/tdse_three_level.csv", "tdse"),
        ("grid QCLE", "reference_grid_qcle/qcle_three_level.csv", "qcle"),
    ):
        data = read_csv(EVIDENCE / relpath)
        for mode in ("time", "grid"):
            representative = next(row for row in data if row["refinement_mode"] == mode and value(row, "P0") == 20.0)
            if name == "TDSE":
                levels = ", ".join(
                    rf"$\Delta t={fmt(value(representative, f'level{i}_dt'))}$, $N_R={int(value(representative, f'level{i}_n_grid_actual'))}$"
                    for i in (1, 2, 3)
                )
                domain = rf"$R\in[{fmt(value(representative, 'level3_R_min'))},{fmt(value(representative, 'level3_R_max'))}]$"
                edge = max(value(row, "level3_maximum_edge_mass_5pct") for row in data if row["refinement_mode"] == mode)
                control = rf"max. edge probability ${fmt(edge)}$"
            else:
                levels = ", ".join(
                    rf"$\Delta t={fmt(value(representative, f'level{i}_dt'))}$, ${int(value(representative, f'level{i}_n_R'))}\!\times\!{int(value(representative, f'level{i}_n_P'))}$"
                    for i in (1, 2, 3)
                )
                domain = rf"$R\in[{fmt(value(representative, 'level3_R_min'))},{fmt(value(representative, 'level3_R_max'))}]$, $P\in[{fmt(value(representative, 'level3_P_min'))},{fmt(value(representative, 'level3_P_max'))}]$"
                edge_r = max(value(row, "level3_maximum_edge_R_mass_5pct") for row in data if row["refinement_mode"] == mode)
                edge_p = max(value(row, "level3_maximum_edge_P_mass_5pct") for row in data if row["refinement_mode"] == mode)
                cfl = max(value(row, "level1_cfl_ratio") for row in data if row["refinement_mode"] == mode)
                control = rf"edge fractions $({fmt(edge_r)},{fmt(edge_p)})$; max. CFL ratio ${fmt(cfl)}$"
            parameter_rows.append(rf"{name} & {mode} & {domain} & {levels} & {control}")

        rows: list[str] = []
        for mode in ("time", "grid"):
            for p0 in (20.0, 100.0):
                group = [row for row in data if row["refinement_mode"] == mode and value(row, "P0") == p0]
                for observable in ("P0", "P1", "trace", "energy", "R_mean", "P_mean", "R_var", "P_var"):
                    row = next(r for r in group if r["observable"] == observable)
                    reason = row["order_reason"]
                    # Observed orders far above every declared formal order
                    # indicate a non-asymptotic sequence (the finer difference
                    # has entered a cancellation- or roundoff-limited regime)
                    # and are not interpreted as convergence orders.
                    if (
                        reason == "ok"
                        and 0 < value(row, "p_observed") <= MAX_PLAUSIBLE_ORDER
                    ):
                        interpretation = rf"$p={fmt(value(row, 'p_observed'))}$"
                    elif reason == "ok" and value(row, "p_observed") > MAX_PLAUSIBLE_ORDER:
                        interpretation = (
                            "rapid contraction; order not interpreted "
                            "(not demonstrably asymptotic)"
                        )
                    elif "not_asymptotic" in reason or "rapid_contraction" in reason:
                        interpretation = (
                            "rapid contraction; order not interpreted "
                            "(not demonstrably asymptotic)"
                        )
                    elif "roundoff" in reason.lower() or "saturation" in reason.lower():
                        interpretation = (
                            "roundoff- or saturation-limited; order not interpreted"
                        )
                    else:
                        interpretation = "non-monotone or unresolved"
                    rows.append(
                        rf"{mode} & {int(p0)} & {obs(observable)} & {mathnum(value(row, 'value1'))} & {mathnum(value(row, 'value2'))} & {mathnum(value(row, 'value3'))} & {mathnum(value(row, 'delta12'))} & {mathnum(value(row, 'delta23'))} & {interpretation}"
                    )
        blocks.append(
            longtable(
                f"tab:{label}-reference-refinement",
                rf"Three-level {name} reference study. Time and spatial/grid refinement are performed separately. $\delta_{{12}}$ and $\delta_{{23}}$ are successive absolute differences. Both must exceed $\tau_{{\rm noise}}=10^{{-12}}+10^{{-12}}\max_k|O_k|$; nonmonotone rows and values above the plausibility bound $p\le{MAX_PLAUSIBLE_ORDER:g}$ are retained but not interpreted as convergence orders.",
                r"l r l r r r r r L{0.18\textwidth}",
                r"Refinement & $P_{\rm init}$ & Observable & Level 1 & Level 2 & Level 3 & $\delta_{12}$ & $\delta_{23}$ & Interpretation",
                rows,
                landscape=True,
            )
        )
    # This table is generated directly from the same full-precision reference
    # CSVs as Appendix F. Keeping it as an input eliminates the former
    # hand-copied P_init=20-only summary that could drift from Table F.1.
    parameter_table = (
        r"\input{../final_reviewer_closure/tables/ReferenceSettingsByMomentum.tex}"
    )
    return parameter_table + "\n\n" + "\n\n".join(blocks)


def physical_comparison_table() -> str:
    data = read_csv(EVIDENCE / "physical_comparison" / "paired_improvement_summary.csv")
    selected = [
        row
        for row in data
        if (
            row["comparison_kind"] == "density"
            and row["observable"] == "mapping-integrated R-P density"
            and row["metric"] in {"raw_E1", "shape_E1"}
        )
        or (
            row["comparison_kind"] == "observable"
            and row["observable"] in {"P0", "P1", "R_mean", "P_mean"}
            and row["metric"] == "E_RMS"
        )
    ]
    metric_labels = {
        "raw_E1": r"raw $E_1$",
        "shape_E1": r"unit-mass shape $E_1$",
        "E_RMS": r"time-series RMS error",
    }
    rows: list[str] = []
    for row in sorted(selected, key=lambda r: (r["reference"], float(r["P0"]), r["comparison_kind"], r["observable"], r["metric"])):
        lo, hi = value(row, "ci95_low"), value(row, "ci95_high")
        if hi < 0:
            interpretation = "MIDPOINT smaller for this metric"
        elif lo > 0:
            interpretation = "PBME smaller for this metric"
        else:
            interpretation = "interval includes zero"
        rows.append(
            rf"{row['reference']} & {int(value(row, 'P0'))} & {obs(row['observable'])} & {metric_labels[row['metric']]} & {mathnum(value(row, 'mean_paired_difference'))} & $[{fmt(lo)},{fmt(hi)}]$ & {interpretation}"
        )
    return longtable(
        "tab:paired-physical-errors",
        r"Paired physical-error comparison against common references.  The reported difference is $\Delta E=E_{\mathrm{MIDPOINT}}-E_{\mathrm{PBME}}$ over the same four independent seeds; negative values favour MIDPOINT for that metric.  Intervals are paired Student-$t$ 95\% intervals with three degrees of freedom.  Raw and unit-mass density errors are distinguished so that shape normalization cannot hide normalization error.",
        r"l r l l r r L{0.23\textwidth}",
        r"Reference & $P_{\rm init}$ & Quantity & Error measure & $\langle\Delta E\rangle$ & 95\% interval & Interpretation",
        rows,
        landscape=True,
    )


def conservation_table() -> str:
    data = read_csv(EVIDENCE / "preserved_evidence" / "raw_conservation.csv")
    canonical = [
        row
        for row in data
        if value(row, "P0") in {20.0, 100.0}
        and int(value(row, "N")) == 1000
        and math.isclose(value(row, "dt"), 0.25)
        and int(value(row, "seed")) in {11, 29, 47, 73}
        and "reviewer_closure_20260726_174927/step9_repl_" in row["source_file"].replace("\\", "/")
    ]
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in canonical:
        grouped[(row["method"], int(value(row, "P0")), row["quantity"])].append(row)
    rows: list[str] = []
    for m in ("pbme", "midpoint"):
        for p0 in (20, 100):
            for quantity in ("normalization", "trace", "energy"):
                group = grouped[(m, p0, quantity)]
                maxdrifts = [value(row, "maximum_absolute_drift") for row in group]
                rmsdrifts = [value(row, "rms_drift") for row in group]
                endpoints = [value(row, "endpoint_drift") for row in group]
                rows.append(
                    rf"{method(m)} & {p0} & {quantity} & {mathnum(max(maxdrifts))} & {mathnum(mean(rmsdrifts))} & $[{fmt(min(endpoints))},{fmt(max(endpoints))}]$"
                )
    return longtable(
        "tab:raw-conservation-physics",
        r"Raw pre-normalization conservation diagnostics for the canonical four-seed calculations ($N=1000$, $\Delta t=0.25$).  The largest drift is the maximum over both time and seeds; the RMS column is averaged over seeds.  These raw quantities are evaluated before any display normalization.",
        r"l r l r r r",
        r"Method & $P_{\rm init}$ & Quantity & Largest $|\Delta|$ & Mean RMS drift & Endpoint-drift range",
        rows,
    )


def build_tables() -> str:
    sections = [
        "% Reader-facing physics tables; generated from the completed numerical studies.",
        study_design_table(),
        manufactured_tables(),
        projection_and_baseline_tables(),
        timestep_table(),
        cloud_size_table(),
        replication_table(),
        conservation_table(),
        tail_table(),
        reference_tables(),
        physical_comparison_table(),
    ]
    return "\n\n".join(sections) + "\n"


def main() -> int:
    OUTPUT.write_text(build_tables(), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
