#!/usr/bin/env python3
"""Build the small, fully traceable figure set used in the revised thesis.

The script performs no dynamics.  It reads only the quantitative CSV outputs
produced by ``reviewer_final_closure.py --mode analyze`` and writes reader-facing
PNG figures plus a SHA-256 crosswalk.  The closure verifier is then run on both
the numbers and the rendered figure provenance.  Selection rules are explicit
so that a plot cannot silently substitute a legacy run or a differently
normalized metric.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "final_reviewer_closure"
OUT = EVIDENCE / "figures"
DATA = OUT / "derived_figure_data"
SCRIPT = Path(__file__).resolve()

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7B2CBF"
GREY = "#5F6368"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty figure data: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "build_verified_thesis_figures.py"},
    )
    plt.close(fig)


def manufactured_figure() -> tuple[Path, list[Path], str]:
    source = EVIDENCE / "manufactured" / "manufactured_summary.csv"
    rows = [
        row
        for row in read_rows(source)
        if row["metric"] == "Q_relative_l2"
    ]
    if len(rows) != 24:
        raise ValueError(f"Expected 24 manufactured Q/L2 rows; found {len(rows)}")

    colors = {"1e-06": BLUE, "0.01": GREEN, "0.05": ORANGE}
    labels = {"1e-06": r"$\ell_2=10^{-6}$", "0.01": r"$\ell_2=0.01$", "0.05": r"$\ell_2=0.05$"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    for ax, query in zip(axes, ("on_support", "off_support")):
        for regularization in ("1e-06", "0.01", "0.05"):
            selected = sorted(
                (
                    row for row in rows
                    if row["query_type"] == query
                    and row["l2_regularization"] == regularization
                ),
                key=lambda row: int(row["N"]),
            )
            if len(selected) != 4:
                raise ValueError((query, regularization, len(selected)))
            x = np.asarray([int(row["N"]) for row in selected])
            y = np.asarray([float(row["mean"]) for row in selected])
            yerr = np.asarray([float(row["sample_sd"]) for row in selected])
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                markersize=4.5,
                linewidth=1.35,
                capsize=2.5,
                color=colors[regularization],
                label=labels[regularization],
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([300, 600, 1200, 2400], ["300", "600", "1200", "2400"])
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Support size, $N$")
        ax.text(
            0.03, 0.97,
            "Training support" if query == "on_support"
            else "Independent query cloud",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
        )
    axes[0].set_ylabel(r"Relative $L_2$ error in $Q[\rho]$")
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout()
    output = OUT / "manufactured_operator_regularization.png"
    save_figure(fig, output)
    return output, [source], (
        "metric=Q_relative_l2; query_type split into on_support/off_support; "
        "all N={300,600,1200,2400}, seeds summarized by source CSV, and "
        "l2={1e-6,0.01,0.05}; error bars are sample SD over three seeds"
    )


def projection_figure() -> tuple[Path, list[Path], str]:
    source = EVIDENCE / "preserved_evidence" / "projection_leakage.csv"
    rows = read_rows(source)
    grouped: dict[tuple[int, str, float], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(float(row["P0"])), row["method"].upper(), float(row["t_over_tc"]))].append(
            float(row["mean_relative_l2_leakage"])
        )
    derived: list[dict[str, object]] = []
    for (p0, method, scaled_time), values in sorted(grouped.items()):
        if len(values) != 4:
            raise ValueError((p0, method, scaled_time, len(values)))
        derived.append(
            {
                "P0": p0,
                "method": method,
                "t_over_tc": scaled_time,
                "n_seeds": len(values),
                "mean_relative_l2_leakage": mean(values),
                "sample_sd": stdev(values),
            }
        )
    derived_path = DATA / "projection_leakage_plot_data.csv"
    write_rows(derived_path, derived)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    styles = {"PBME": (BLUE, "o", "-"), "MIDPOINT": (ORANGE, "s", "--")}
    for ax, p0 in zip(axes, (20, 100)):
        for method in ("PBME", "MIDPOINT"):
            selected = [row for row in derived if row["P0"] == p0 and row["method"] == method]
            x = np.asarray([float(row["t_over_tc"]) for row in selected])
            y = np.asarray([float(row["mean_relative_l2_leakage"]) for row in selected])
            yerr = np.asarray([float(row["sample_sd"]) for row in selected])
            color, marker, line = styles[method]
            ax.errorbar(x, y, yerr=yerr, color=color, marker=marker, linestyle=line,
                        linewidth=1.35, markersize=4.5, capsize=2.5, label=method)
        ax.text(
            0.03, 0.97, fr"$P_{{\rm init}}={p0}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
        )
        ax.set_xlabel(r"Scaled time, $t/t_c$")
        ax.set_xticks([0, 1, 2])
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel(r"Relative $L_2$ SEO projection leakage")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    output = OUT / "seo_projection_leakage_verified.png"
    save_figure(fig, output)
    return output, [source, derived_path], (
        "all 48 verified projection diagnostics; mean and sample SD over four "
        "independent propagation seeds at t/tc={0,1,2}; projection is diagnostic, not enforced"
    )


def replication_conservation_figure() -> tuple[Path, list[Path], str]:
    values_source = EVIDENCE / "replication" / "four_seed_values.csv"
    conservation_source = EVIDENCE / "preserved_evidence" / "raw_conservation.csv"
    values = [
        row for row in read_rows(values_source)
        if row["observable"] == "P0" and row["record_type"] == "seed_value"
    ]
    if len(values) != 16:
        raise ValueError(f"Expected 16 P0 replication values; found {len(values)}")
    conservation = [
        row for row in read_rows(conservation_source)
        if row["quantity"] == "normalization"
        and "reviewer_closure_20260726_174927/step9_repl" in row["run_directory"].replace("\\", "/")
        and int(float(row["seed"])) in (11, 29, 47, 73)
        and int(float(row["P0"])) in (20, 100)
    ]
    unique: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in conservation:
        unique[(row["method"].upper(), int(float(row["P0"])), int(float(row["seed"])))] = row
    conservation = list(unique.values())
    if len(conservation) != 16:
        raise ValueError(f"Expected 16 replication conservation rows; found {len(conservation)}")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), sharex="col")
    offsets = {"PBME": -0.08, "MIDPOINT": 0.08}
    styles = {"PBME": (BLUE, "o"), "MIDPOINT": (ORANGE, "s")}
    seeds = [11, 29, 47, 73]
    for column, p0 in enumerate((20, 100)):
        top = axes[0, column]
        bottom = axes[1, column]
        top.axhspan(0.0, 1.0, color=GREEN, alpha=0.08, zorder=0)
        for method in ("PBME", "MIDPOINT"):
            color, marker = styles[method]
            selected = {
                int(float(row["seed"])): float(row["endpoint_value"])
                for row in values
                if row["method"].upper() == method and int(float(row["P0"])) == p0
            }
            top.scatter(
                np.asarray(seeds) + offsets[method],
                [selected[seed] for seed in seeds],
                color=color,
                marker=marker,
                s=32,
                label=method,
                zorder=3,
            )
            drift = {
                int(float(row["seed"])): float(row["maximum_absolute_drift"])
                for row in conservation
                if row["method"].upper() == method and int(float(row["P0"])) == p0
            }
            bottom.scatter(
                np.asarray(seeds) + offsets[method],
                [max(drift[seed], 1.0e-16) for seed in seeds],
                color=color,
                marker=marker,
                s=32,
                label=method,
                zorder=3,
            )
        top.axhline(0.0, color=GREY, linewidth=0.8)
        top.axhline(1.0, color=GREY, linewidth=0.8)
        top.text(
            0.03, 0.97, fr"$P_{{\rm init}}={p0}$",
            transform=top.transAxes, va="top", ha="left", fontsize=8.5,
        )
        bottom.set_yscale("log")
        bottom.set_xticks(seeds)
        bottom.set_xlabel("Propagation seed")
    axes[0, 0].set_ylabel(r"Endpoint $\rho_{11}^{\rm SN}$")
    axes[1, 0].set_ylabel("Maximum raw normalization drift")
    axes[0, 1].legend(frameon=False, loc="best")
    axes[1, 0].text(11, 1.8e-16, r"PBME values at plotting floor $10^{-16}$", fontsize=7, color=GREY)
    fig.tight_layout()
    output = OUT / "replication_and_raw_conservation.png"
    save_figure(fig, output)
    return output, [values_source, conservation_source], (
        "observable=P0 endpoint values from the four-seed replication campaign; "
        "raw normalization uses quantity=normalization and only the canonical "
        "reviewer_closure_20260726_174927/step9_repl runs; zeros plotted at 1e-16"
    )


def physical_comparison_figure() -> tuple[Path, list[Path], str]:
    source = EVIDENCE / "physical_comparison" / "paired_improvement_summary.csv"
    rows = [
        row for row in read_rows(source)
        if row["comparison_kind"] == "density" and row["metric"] in ("raw_E1", "shape_E1")
    ]
    if len(rows) != 8:
        raise ValueError(f"Expected 8 density E1 comparison rows; found {len(rows)}")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55))
    for ax, metric, title in zip(
        axes,
        ("raw_E1", "shape_E1"),
        ("Raw-density $E_1$", "Unit-mass shape $E_1$"),
    ):
        selected = sorted(
            (row for row in rows if row["metric"] == metric),
            key=lambda row: (int(float(row["P0"])), row["reference"]),
        )
        y = np.arange(len(selected))
        labels = [fr"{row['reference']}, $P_{{\rm init}}={int(float(row['P0']))}$" for row in selected]
        x = np.asarray([float(row["mean_paired_difference"]) for row in selected])
        low = np.asarray([float(row["ci95_low"]) for row in selected])
        high = np.asarray([float(row["ci95_high"]) for row in selected])
        if metric == "raw_E1":
            transform = lambda values: np.sign(values) * np.log10(1.0 + np.abs(values))
            x_plot = transform(x)
            low_plot = transform(low)
            high_plot = transform(high)
        else:
            x_plot, low_plot, high_plot = x, low, high
        colors = [ORANGE if row["verdict_before_scientific_gates"] == "MIDPOINT_ERROR_LARGER" else BLUE for row in selected]
        ax.axvline(0.0, color=GREY, linewidth=0.9)
        for index in range(len(selected)):
            ax.errorbar(
                x_plot[index],
                y[index],
                xerr=[
                    [x_plot[index] - low_plot[index]],
                    [high_plot[index] - x_plot[index]],
                ],
                fmt="o",
                color=colors[index],
                capsize=2.5,
                markersize=4.5,
            )
        ax.set_yticks(y, labels)
        ax.set_xlabel(
            r"Paired difference: $E_{\rm MIDPOINT}-E_{\rm PBME}$"
            + "\n" + f"({title})"
        )
        if metric == "raw_E1":
            ax.set_xticks(
                [-21, -14, -7, 0, 7, 14, 21],
                [r"$-10^{21}$", r"$-10^{14}$", r"$-10^7$", "$0$", r"$10^7$", r"$10^{14}$", r"$10^{21}$"],
            )
            ax.set_xlabel(
                r"Paired difference on signed $\log_{10}(1+|\Delta E|)$ scale"
                + "\n" + f"({title})"
            )
        ax.invert_yaxis()
    fig.tight_layout()
    output = OUT / "paired_physical_reference_differences.png"
    save_figure(fig, output)
    return output, [source], (
        "comparison_kind=density and metric in {raw_E1,shape_E1}; all reference/P0 "
        "cells retained; points are four-seed paired means and bars are two-sided "
        "Student-t 95% intervals; positive values mean larger MIDPOINT error; "
        "raw panel uses the declared signed log10(1+abs(delta E)) display transform"
    )


def tail_distribution_figure() -> tuple[Path, list[Path], str]:
    distribution_source = (
        EVIDENCE / "tail_sensitivity" / "y0_distribution_paired.csv"
    )
    sweep_source = EVIDENCE / "tail_sensitivity" / "threshold_sweep.csv"
    distributions = read_rows(distribution_source)
    sweep = [
        row for row in read_rows(sweep_source)
        if row["method"] == "PBME" and float(row["eta"]) > 0.0
    ]
    if len(distributions) != 8:
        raise ValueError(
            f"Expected eight paired initial-label distributions; found {len(distributions)}"
        )
    if len(sweep) != 72:
        raise ValueError(f"Expected 72 positive-threshold rows; found {len(sweep)}")

    quantiles = (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999)
    seeds = (11, 29, 47, 73)
    colors = {11: BLUE, 29: ORANGE, 47: GREEN, 73: PURPLE}
    floor = 1.0e-7
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    for row_index, p0 in enumerate((20, 100)):
        distribution_ax = axes[row_index, 0]
        threshold_ax = axes[row_index, 1]
        for seed in seeds:
            selected_distribution = [
                row for row in distributions
                if int(float(row["P0"])) == p0 and int(row["seed"]) == seed
            ]
            if len(selected_distribution) != 1:
                raise ValueError((p0, seed, len(selected_distribution)))
            row = selected_distribution[0]
            distribution_ax.plot(
                quantiles,
                [float(row[f"q{quantile:g}"]) for quantile in quantiles],
                color=colors[seed],
                marker="o",
                markersize=2.8,
                linewidth=1.1,
                label=f"seed {seed}",
            )

            selected_sweep = sorted(
                (
                    row for row in sweep
                    if int(float(row["P0"])) == p0 and int(row["seed"]) == seed
                ),
                key=lambda row: float(row["eta"]),
            )
            if len(selected_sweep) != 9:
                raise ValueError((p0, seed, len(selected_sweep)))
            eta = np.asarray([float(row["eta"]) for row in selected_sweep])
            excluded_points = np.asarray(
                [float(row["excluded_fraction"]) for row in selected_sweep]
            )
            excluded_mass = np.asarray(
                [
                    float(row["excluded_absolute_physical_mass_fraction"])
                    for row in selected_sweep
                ]
            )
            threshold_ax.plot(
                eta,
                np.maximum(excluded_points, floor),
                color=colors[seed],
                linewidth=1.1,
            )
            threshold_ax.plot(
                eta,
                np.maximum(excluded_mass, floor),
                color=colors[seed],
                linestyle="--",
                linewidth=1.1,
            )

        distribution_ax.set_yscale("log")
        distribution_ax.set_ylabel(r"Quantile of $|y_i^0|$")
        distribution_ax.text(
            0.03, 0.97, fr"Initial labels, $P_{{\rm init}}={p0}$",
            transform=distribution_ax.transAxes, va="top", ha="left",
            fontsize=8.5,
        )
        distribution_ax.set_xlim(0.0, 1.0)
        threshold_ax.set_xscale("log")
        threshold_ax.set_yscale("log")
        threshold_ax.set_ylim(floor * 0.8, 2.0e-2)
        threshold_ax.text(
            0.03, 0.97, fr"Threshold effect, $P_{{\rm init}}={p0}$",
            transform=threshold_ax.transAxes, va="top", ha="left",
            fontsize=8.5,
        )
        threshold_ax.set_ylabel("Excluded fraction")
    axes[1, 0].set_xlabel("Empirical quantile")
    axes[1, 1].set_xlabel(r"Relative threshold $\eta$")
    axes[0, 0].legend(frameon=False, ncol=2, loc="best")
    axes[0, 1].plot([], [], color=GREY, linestyle="-", label="points")
    axes[0, 1].plot([], [], color=GREY, linestyle="--", label="absolute physical mass")
    axes[0, 1].legend(frameon=False, loc="best")
    axes[1, 1].text(
        1.2e-14,
        1.35e-7,
        r"zero exclusions displayed at $10^{-7}$",
        fontsize=7,
        color=GREY,
    )
    fig.tight_layout()
    output = OUT / "initial_label_tail_sensitivity.png"
    save_figure(fig, output)
    return output, [distribution_source, sweep_source], (
        "all eight paired Pinit/seed initial-label distributions; quantile panel "
        "uses q={0.001,...,0.999}; threshold panels use the PBME copy of the "
        "method-identical inclusion masks at every positive eta; solid lines are "
        "excluded point fractions and dashed lines are excluded absolute physical "
        "mass fractions; exact zeros are plotted at 1e-7"
    )


def main() -> int:
    analysis = EVIDENCE / "analysis_manifest.json"
    if not analysis.exists():
        raise RuntimeError(
            "Analysis manifest is absent; run reviewer_final_closure.py "
            "--mode analyze before building figures"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    products = [
        manufactured_figure(),
        projection_figure(),
        replication_conservation_figure(),
        physical_comparison_figure(),
        tail_distribution_figure(),
    ]
    artifact_ids = {
        "manufactured_operator_regularization.png": "fig:manufactured-operator-regularization",
        "seo_projection_leakage_verified.png": "fig:seo-projection-leakage-verified",
        "replication_and_raw_conservation.png": "fig:replication-raw-conservation",
        "paired_physical_reference_differences.png": "fig:paired-physical-reference-differences",
        "initial_label_tail_sensitivity.png": "fig:initial-label-tail-sensitivity",
    }
    script_hash = sha256(SCRIPT)
    rows: list[dict[str, object]] = []
    for figure, sources, selection_rule in products:
        rows.append(
            {
                "artifact_id": artifact_ids[figure.name],
                "figure": figure.relative_to(ROOT).as_posix(),
                "figure_sha256": sha256(figure),
                "source_csvs": ";".join(path.relative_to(ROOT).as_posix() for path in sources),
                "source_csv_sha256": ";".join(sha256(path) for path in sources),
                "generator": SCRIPT.relative_to(ROOT).as_posix(),
                "generator_sha256": script_hash,
                "selection_rule": selection_rule,
            }
        )
    crosswalk = OUT / "FIGURE_DATA_CROSSWALK.csv"
    write_rows(crosswalk, rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_manifest": analysis.relative_to(ROOT).as_posix(),
        "analysis_manifest_sha256": sha256(analysis),
        "generator": SCRIPT.relative_to(ROOT).as_posix(),
        "generator_sha256": script_hash,
        "figure_count": len(rows),
        "crosswalk": crosswalk.relative_to(ROOT).as_posix(),
        "crosswalk_sha256": sha256(crosswalk),
        "figures": rows,
    }
    (OUT / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
