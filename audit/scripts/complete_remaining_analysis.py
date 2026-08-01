"""Complete quantitative analyses supported by the saved four-seed campaign.

This script is deliberately read-only with respect to simulation artifacts.  All
derived JSON, NPZ, and CSV files are written below ``reviewer_data_audit``.

The calculations add:

* pairwise independent-seed trajectory distances and method-level summaries;
* threshold-free MIDPOINT/PBME seed-dispersion ratios;
* population-physicality and GP-health diagnostics for every replication run;
* SEO projection diagnostics for all saved replication snapshots;
* identical-support KDE/projected-GP diagnostics for all saved snapshots.

The 1% conservation and 5% ESS values are labelled audit visibility gates.  They
are not retrospective scientific acceptance criteria.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT = Path(__file__).resolve()
AUDIT = SCRIPT.parents[1]
ROOT = AUDIT.parent
TABLES = AUDIT / "tables"
DERIVED = AUDIT / "derived_validations"
CANON = ROOT / "reviewer_closure_20260726_174927"
SEEDS = (11, 29, 47, 73)
P0_VALUES = (20, 100)
METHODS = ("pbme", "midpoint")
OBSERVABLES = (
    "lw_P0",
    "lw_P1",
    "lw_P_sum",
    "lw_trace",
    "lw_energy",
    "nm_R_mean",
    "nm_P_mean",
    "nm_R_var",
    "nm_P_var",
    "raw_norm_drift",
    "raw_trace_drift",
    "raw_energy_drift",
    "cs_q_rms",
    "applied_cs_q_rms",
)
T_CRIT_975_DF3 = 3.182446305284263


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path.resolve())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def trapz_mean_abs(tau: np.ndarray, values: np.ndarray) -> float:
    return float(np.trapezoid(np.abs(values), tau) / (tau[-1] - tau[0]))


def trapz_rms(tau: np.ndarray, values: np.ndarray) -> float:
    return float(
        np.sqrt(np.trapezoid(values * values, tau) / (tau[-1] - tau[0]))
    )


def stats(values: Iterable[float]) -> dict[str, Any]:
    vals = np.asarray(list(values), dtype=float)
    n = int(vals.size)
    mean = float(np.mean(vals)) if n else float("nan")
    sd = float(np.std(vals, ddof=1)) if n > 1 else "NOT COMPUTED"
    se = float(sd / math.sqrt(n)) if n > 1 else "NOT COMPUTED"
    half = float(T_CRIT_975_DF3 * se) if n == 4 else "NOT COMPUTED"
    return {
        "n": n,
        "mean": mean,
        "sample_sd": sd,
        "standard_error": se,
        "ci95_lower": mean - half if n == 4 else "NOT COMPUTED",
        "ci95_upper": mean + half if n == 4 else "NOT COMPUTED",
        "ci_method": "two-sided Student-t, df=3" if n == 4 else "NOT COMPUTED",
        "minimum": float(np.min(vals)) if n else float("nan"),
        "maximum": float(np.max(vals)) if n else float("nan"),
    }


def stem_for(p0: int, seed: int, method: str) -> Path:
    return CANON / f"step9_repl_P0{p0}" / f"seed{seed}" / method


def common_seed_arrays(
    p0: int, method: str, observable: str, tau_grid: np.ndarray
) -> dict[int, np.ndarray]:
    tc = 30000.0 / p0
    result: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        path = stem_for(p0, seed, method).with_suffix(".npz")
        with np.load(path, allow_pickle=False) as z:
            if observable not in z:
                continue
            t = np.asarray(z["t"], dtype=float)
            u = np.asarray(z[observable], dtype=float)
            result[seed] = np.interp(tau_grid * tc, t, u)
    return result


def seed_reliability() -> None:
    tau = np.linspace(0.0, 2.0, 41)
    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    population_rows: list[dict[str, Any]] = []
    health_rows: list[dict[str, Any]] = []

    by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    for p0 in P0_VALUES:
        for method in METHODS:
            for observable in OBSERVABLES:
                arrays = common_seed_arrays(p0, method, observable, tau)
                if len(arrays) < 2:
                    continue
                pairs: list[dict[str, Any]] = []
                for seed_a, seed_b in itertools.combinations(sorted(arrays), 2):
                    diff = arrays[seed_a] - arrays[seed_b]
                    row = {
                        "P0": p0,
                        "method": method,
                        "observable": observable,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "n_common_times": len(tau),
                        "time_coordinate": "t/tc",
                        "endpoint_signed_difference": float(diff[-1]),
                        "endpoint_absolute_difference": float(abs(diff[-1])),
                        "maximum_in_time_absolute_difference": float(
                            np.max(np.abs(diff))
                        ),
                        "time_of_maximum_t_over_tc": float(
                            tau[int(np.argmax(np.abs(diff)))]
                        ),
                        "time_averaged_L1_difference": trapz_mean_abs(tau, diff),
                        "time_averaged_L2_difference": trapz_rms(tau, diff),
                        "source_a": relative(
                            stem_for(p0, seed_a, method).with_suffix(".npz")
                        ),
                        "source_b": relative(
                            stem_for(p0, seed_b, method).with_suffix(".npz")
                        ),
                    }
                    pairs.append(row)
                    pair_rows.append(row)

                endpoint_values = np.asarray(
                    [arrays[seed][-1] for seed in sorted(arrays)], dtype=float
                )
                stacked = np.vstack([arrays[seed] for seed in sorted(arrays)])
                endpoint_stats = stats(endpoint_values)
                pair_l2 = [r["time_averaged_L2_difference"] for r in pairs]
                pair_l1 = [r["time_averaged_L1_difference"] for r in pairs]
                pair_max = [
                    r["maximum_in_time_absolute_difference"] for r in pairs
                ]
                seed_sd_t = np.std(stacked, axis=0, ddof=1)
                seed_spread_t = np.max(stacked, axis=0) - np.min(stacked, axis=0)
                summary = {
                    "P0": p0,
                    "method": method,
                    "observable": observable,
                    "n_independent_seeds": len(arrays),
                    "n_seed_pairs": len(pairs),
                    "endpoint_mean": endpoint_stats["mean"],
                    "endpoint_sample_sd": endpoint_stats["sample_sd"],
                    "endpoint_standard_error": endpoint_stats["standard_error"],
                    "endpoint_ci95_lower": endpoint_stats["ci95_lower"],
                    "endpoint_ci95_upper": endpoint_stats["ci95_upper"],
                    "endpoint_minimum": endpoint_stats["minimum"],
                    "endpoint_maximum": endpoint_stats["maximum"],
                    "endpoint_spread": float(np.ptp(endpoint_values)),
                    "endpoint_coefficient_of_variation": (
                        float(endpoint_stats["sample_sd"])
                        / abs(float(endpoint_stats["mean"]))
                        if isinstance(endpoint_stats["sample_sd"], float)
                        and abs(float(endpoint_stats["mean"])) > 1.0e-14
                        else "NOT COMPUTED"
                    ),
                    "mean_pairwise_time_L1": float(np.mean(pair_l1)),
                    "maximum_pairwise_time_L1": float(np.max(pair_l1)),
                    "mean_pairwise_time_L2": float(np.mean(pair_l2)),
                    "maximum_pairwise_time_L2": float(np.max(pair_l2)),
                    "maximum_pairwise_difference": float(np.max(pair_max)),
                    "maximum_seed_sample_sd_over_time": float(np.max(seed_sd_t)),
                    "time_of_maximum_seed_sd_t_over_tc": float(
                        tau[int(np.argmax(seed_sd_t))]
                    ),
                    "maximum_seed_spread_over_time": float(np.max(seed_spread_t)),
                    "time_of_maximum_seed_spread_t_over_tc": float(
                        tau[int(np.argmax(seed_spread_t))]
                    ),
                    "interpretation": (
                        "descriptive four-seed reproducibility; no acceptance "
                        "threshold was predeclared"
                    ),
                    "source_files": "; ".join(
                        relative(stem_for(p0, seed, method).with_suffix(".npz"))
                        for seed in sorted(arrays)
                    ),
                }
                summary_rows.append(summary)
                by_key[(p0, method, observable)] = summary

            for seed in SEEDS:
                path = stem_for(p0, seed, method).with_suffix(".npz")
                with np.load(path, allow_pickle=False) as z:
                    t = np.asarray(z["t"], dtype=float)
                    tc = 30000.0 / p0
                    for observable in ("lw_P0", "lw_P1"):
                        u = np.asarray(z[observable], dtype=float)
                        outside = np.isfinite(u) & (
                            (u < -1.0e-6) | (u > 1.0 + 1.0e-6)
                        )
                        first = int(np.where(outside)[0][0]) if np.any(outside) else -1
                        population_rows.append(
                            {
                                "P0": p0,
                                "method": method,
                                "seed": seed,
                                "observable": observable,
                                "minimum": float(np.nanmin(u)),
                                "maximum": float(np.nanmax(u)),
                                "n_values": len(u),
                                "n_outside_0_1_tolerance": int(
                                    np.count_nonzero(outside)
                                ),
                                "fraction_outside_0_1_tolerance": float(
                                    np.mean(outside)
                                ),
                                "first_affected_step": (
                                    int(np.asarray(z["step_index"])[first])
                                    if first >= 0
                                    else "NOT APPLICABLE"
                                ),
                                "first_affected_time": (
                                    float(t[first])
                                    if first >= 0
                                    else "NOT APPLICABLE"
                                ),
                                "first_affected_t_over_tc": (
                                    float(t[first] / tc)
                                    if first >= 0
                                    else "NOT APPLICABLE"
                                ),
                                "rule": "outside [0,1] by more than 1e-6",
                                "source_file": relative(path),
                            }
                        )

                    def arr(name: str) -> np.ndarray | None:
                        return (
                            np.asarray(z[name], dtype=float)
                            if name in z
                            else None
                        )

                    sigma_n = arr("sigma_n")
                    alpha = arr("alpha_linf")
                    ess = arr("sw_abs_ess_frac")
                    q = arr("cs_q_rms")
                    defined = arr("cs_q_weighted_mean_defined")
                    denominator = arr("cs_q_weight_denominator")
                    raw_norm = arr("raw_norm_drift")
                    raw_energy_rel = arr("raw_energy_relative_drift")
                    adapt_failed = arr("adapt_refit_failed")
                    opt_total = arr("gp_opt_total_loss")
                    opt_reg = arr("gp_opt_reg_loss")

                    def extremum_time(
                        values: np.ndarray | None, kind: str
                    ) -> tuple[Any, Any]:
                        if values is None or not np.any(np.isfinite(values)):
                            return "DATA ABSENT", "DATA ABSENT"
                        idx = (
                            int(np.nanargmin(values))
                            if kind == "min"
                            else int(np.nanargmax(np.abs(values)))
                        )
                        value = (
                            float(np.nanmin(values))
                            if kind == "min"
                            else float(np.nanmax(np.abs(values)))
                        )
                        return value, float(t[idx])

                    sigma_max, sigma_t = extremum_time(sigma_n, "max")
                    alpha_max, alpha_t = extremum_time(alpha, "max")
                    ess_min, ess_t = extremum_time(ess, "min")
                    q_max, q_t = extremum_time(q, "max")
                    norm_max, norm_t = extremum_time(raw_norm, "max")
                    energy_max, energy_t = extremum_time(raw_energy_rel, "max")
                    undefined_q = (
                        (np.arange(len(t)) > 0)
                        & (np.abs(q) > 0)
                        & (defined < 0.5)
                        if q is not None and defined is not None
                        else np.zeros(len(t), dtype=bool)
                    )
                    health_rows.append(
                        {
                            "P0": p0,
                            "method": method,
                            "seed": seed,
                            "sigma_n_max": sigma_max,
                            "time_of_sigma_n_max": sigma_t,
                            "sigma_n_upper_bound_hits": (
                                int(
                                    np.count_nonzero(
                                        np.isclose(
                                            sigma_n,
                                            math.e,
                                            rtol=1.0e-10,
                                            atol=1.0e-12,
                                        )
                                    )
                                )
                                if sigma_n is not None
                                else "DATA ABSENT"
                            ),
                            "alpha_linf_max_abs": alpha_max,
                            "time_of_alpha_linf_max": alpha_t,
                            "sw_abs_ess_fraction_min": ess_min,
                            "time_of_ess_min": ess_t,
                            "ess_below_0_05_count": (
                                int(np.count_nonzero(ess < 0.05))
                                if ess is not None
                                else "DATA ABSENT"
                            ),
                            "cs_q_rms_max_abs": q_max,
                            "time_of_cs_q_rms_max": q_t,
                            "undefined_weighted_Q_mean_count": int(
                                np.count_nonzero(undefined_q)
                            ),
                            "minimum_abs_Q_denominator_when_undefined": (
                                float(np.min(np.abs(denominator[undefined_q])))
                                if denominator is not None
                                and np.any(undefined_q)
                                else "NOT APPLICABLE"
                            ),
                            "raw_norm_max_abs_drift": norm_max,
                            "time_of_raw_norm_max_abs_drift": norm_t,
                            "raw_energy_max_abs_relative_drift": energy_max,
                            "time_of_raw_energy_max_abs_relative_drift": energy_t,
                            "adaptive_refit_failure_count": (
                                int(np.count_nonzero(adapt_failed > 0.5))
                                if adapt_failed is not None
                                else "DATA ABSENT"
                            ),
                            "nonfinite_gp_total_loss_count": (
                                int(np.count_nonzero(~np.isfinite(opt_total)))
                                if opt_total is not None
                                else "DATA ABSENT"
                            ),
                            "nonfinite_gp_regularization_loss_count": (
                                int(np.count_nonzero(~np.isfinite(opt_reg)))
                                if opt_reg is not None
                                else "DATA ABSENT"
                            ),
                            "audit_gate_note": (
                                "ESS 0.05 is the configured resampling threshold; "
                                "1% conservation is an audit visibility gate, not "
                                "a predeclared scientific acceptance criterion"
                            ),
                            "source_file": relative(path),
                        }
                    )

    for p0 in P0_VALUES:
        for observable in OBSERVABLES:
            pbme = by_key.get((p0, "pbme", observable))
            midpoint = by_key.get((p0, "midpoint", observable))
            if not pbme or not midpoint:
                continue

            def ratio(num: Any, den: Any) -> Any:
                if not isinstance(num, (float, int)) or not isinstance(
                    den, (float, int)
                ):
                    return "NOT COMPUTED"
                return float(num / den) if abs(float(den)) > 1.0e-30 else "NOT IDENTIFIABLE"

            method_rows.append(
                {
                    "P0": p0,
                    "observable": observable,
                    "PBME_endpoint_spread": pbme["endpoint_spread"],
                    "MIDPOINT_endpoint_spread": midpoint["endpoint_spread"],
                    "MIDPOINT_to_PBME_endpoint_spread_ratio": ratio(
                        midpoint["endpoint_spread"], pbme["endpoint_spread"]
                    ),
                    "PBME_mean_pairwise_time_L2": pbme[
                        "mean_pairwise_time_L2"
                    ],
                    "MIDPOINT_mean_pairwise_time_L2": midpoint[
                        "mean_pairwise_time_L2"
                    ],
                    "MIDPOINT_to_PBME_pairwise_L2_ratio": ratio(
                        midpoint["mean_pairwise_time_L2"],
                        pbme["mean_pairwise_time_L2"],
                    ),
                    "PBME_maximum_seed_spread_over_time": pbme[
                        "maximum_seed_spread_over_time"
                    ],
                    "MIDPOINT_maximum_seed_spread_over_time": midpoint[
                        "maximum_seed_spread_over_time"
                    ],
                    "MIDPOINT_to_PBME_maximum_spread_ratio": ratio(
                        midpoint["maximum_seed_spread_over_time"],
                        pbme["maximum_seed_spread_over_time"],
                    ),
                    "assessment_basis": (
                        "threshold-free comparison of cross-seed dispersion; "
                        "ratio >1 means MIDPOINT is more seed-sensitive than PBME"
                    ),
                    "PBME_source_files": pbme["source_files"],
                    "MIDPOINT_source_files": midpoint["source_files"],
                }
            )

    write_csv(TABLES / "seed_replication_pairwise_distances.csv", pair_rows)
    write_csv(TABLES / "seed_replication_reliability_summary.csv", summary_rows)
    write_csv(TABLES / "seed_replication_method_dispersion_ratios.csv", method_rows)
    write_csv(TABLES / "population_physicality_audit.csv", population_rows)
    write_csv(TABLES / "replication_gp_health.csv", health_rows)


def snapshot_validations(
    selected_p0: int | None = None,
    selected_seed: int | None = None,
    selected_method: str | None = None,
    collect_only: bool = False,
) -> None:
    sys.path.insert(0, str(ROOT))
    from ReviewerValidation import baseline_kde_gp, projection_diagnostic

    projection_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for p0 in P0_VALUES:
        if selected_p0 is not None and p0 != selected_p0:
            continue
        tc = 30000.0 / p0
        for seed in SEEDS:
            if selected_seed is not None and seed != selected_seed:
                continue
            for method in METHODS:
                if selected_method is not None and method != selected_method:
                    continue
                stem = stem_for(p0, seed, method)
                meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
                steps = [int(x) for x in meta["snapshot_steps"]]
                with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
                    times = {
                        step: float(np.asarray(z[f"snap_{step:06d}_t"])[0])
                        for step in steps
                    }
                for step in steps:
                    out = (
                        DERIVED
                        / "snapshots"
                        / f"P0_{p0}"
                        / f"seed_{seed}"
                        / method
                        / f"step_{step:06d}"
                    )
                    projection_path = (
                        out / "projection" / "projection_leakage.json"
                    )
                    baseline_path = (
                        out / "baseline" / "kde_gp_identical_support.json"
                    )
                    if not projection_path.exists() and not collect_only:
                        projection = projection_diagnostic(
                            stem, out / "projection", step, 20, 400, 123
                        )
                    elif projection_path.exists():
                        projection = json.loads(
                            projection_path.read_text(encoding="utf-8")
                        )
                    else:
                        continue
                    leakage = [
                        float(row["relative_l2_leakage"])
                        for row in projection["per_anchor"]
                    ]
                    projection_rows.append(
                        {
                            "P0": p0,
                            "method": method,
                            "propagation_seed": seed,
                            "diagnostic_probe_seed": 123,
                            "snapshot_step": step,
                            "physical_time": times[step],
                            "t_over_tc": times[step] / tc,
                            "n_bath_anchors": projection["n_bath_anchors"],
                            "n_mapping_probes": projection["n_mapping_probes"],
                            "basis_rank": projection["basis_rank"],
                            "mean_relative_l2_leakage": float(np.mean(leakage)),
                            "median_relative_l2_leakage": float(np.median(leakage)),
                            "sample_sd_relative_l2_leakage": float(
                                np.std(leakage, ddof=1)
                            ),
                            "maximum_relative_l2_leakage": float(np.max(leakage)),
                            "normalization": "||y-Bc||2/max(||y||2,1e-30)",
                            "projection_policy": (
                                "diagnostic least-squares projection; not enforced"
                            ),
                            "source_file": relative(stem.with_suffix(".npz")),
                            "derived_json": relative(
                                out / "projection" / "projection_leakage.json"
                            ),
                        }
                    )
                    if not baseline_path.exists() and not collect_only:
                        baseline = baseline_kde_gp(
                            stem, out / "baseline", step, n_grid=120
                        )
                    elif baseline_path.exists():
                        baseline = json.loads(
                            baseline_path.read_text(encoding="utf-8")
                        )
                    else:
                        continue
                    errors = baseline["shape_errors"]
                    acceptance = baseline["acceptance"]
                    baseline_rows.append(
                        {
                            "P0": p0,
                            "method": method,
                            "seed": seed,
                            "snapshot_step": step,
                            "physical_time": times[step],
                            "t_over_tc": times[step] / tc,
                            "n_support": baseline["n_support"],
                            "initial_cloud_sha256": baseline[
                                "initial_cloud_sha256"
                            ],
                            "weight_policy": baseline["weight_policy"],
                            "n_R": baseline["grid"]["n_R"],
                            "n_P": baseline["grid"]["n_P"],
                            "R_min": baseline["grid"]["R_range"][0],
                            "R_max": baseline["grid"]["R_range"][1],
                            "P_min": baseline["grid"]["P_range"][0],
                            "P_max": baseline["grid"]["P_range"][1],
                            "bandwidth_R": baseline["bandwidth"]["R"],
                            "bandwidth_P": baseline["bandwidth"]["P"],
                            "E1": errors["E1"],
                            "E2": errors["E2"],
                            "Einf": errors["Einf"],
                            "raw_E1": errors["raw_E1"],
                            "raw_Einf": errors["raw_Einf"],
                            "acceptance_applies": acceptance["applies"],
                            "acceptance_threshold_E1": (
                                acceptance["threshold"]
                                if acceptance["applies"]
                                else "NOT APPLICABLE"
                            ),
                            "passed_declared_threshold": (
                                acceptance["passed"]
                                if acceptance["applies"]
                                else "NOT APPLICABLE"
                            ),
                            "estimator_contract": baseline[
                                "estimator_contract"
                            ],
                            "source_file": relative(stem.with_suffix(".npz")),
                            "derived_json": relative(
                                out
                                / "baseline"
                                / "kde_gp_identical_support.json"
                            ),
                            "derived_npz": relative(
                                out
                                / "baseline"
                                / "kde_gp_identical_support.npz"
                            ),
                        }
                    )
    # Filtered invocations are used to keep memory bounded.  Only an unfiltered
    # or collect-only invocation writes the combined tables.
    if any(
        x is not None for x in (selected_p0, selected_seed, selected_method)
    ) and not collect_only:
        return
    write_csv(TABLES / "seo_projection_leakage_all_snapshots.csv", projection_rows)
    write_csv(TABLES / "kde_gp_identical_support_all_snapshots.csv", baseline_rows)

    aggregate: list[dict[str, Any]] = []
    groups: dict[tuple[int, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in projection_rows:
        groups[(row["P0"], row["method"], row["t_over_tc"])].append(row)
    for (p0, method, tau), rows in sorted(groups.items()):
        seed_means = [r["mean_relative_l2_leakage"] for r in rows]
        seed_stats = stats(seed_means)
        aggregate.append(
            {
                "P0": p0,
                "method": method,
                "t_over_tc": tau,
                "n_independent_propagation_seeds": len(rows),
                "mean_of_seed_mean_leakage": seed_stats["mean"],
                "sample_sd_across_seed_means": seed_stats["sample_sd"],
                "standard_error_across_seed_means": seed_stats[
                    "standard_error"
                ],
                "ci95_lower": seed_stats["ci95_lower"],
                "ci95_upper": seed_stats["ci95_upper"],
                "ci_method": seed_stats["ci_method"],
                "minimum_seed_mean": seed_stats["minimum"],
                "maximum_seed_mean": seed_stats["maximum"],
                "maximum_anchor_leakage": max(
                    r["maximum_relative_l2_leakage"] for r in rows
                ),
                "note": (
                    "uncertainty uses four independent propagation seeds; "
                    "within-run bath anchors are diagnostic probes"
                ),
                "source_files": "; ".join(r["source_file"] for r in rows),
            }
        )
    write_csv(TABLES / "seo_projection_leakage_seed_summary.csv", aggregate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-snapshot-validations",
        action="store_true",
        help="Compute seed reliability only.",
    )
    parser.add_argument("--p0", type=int, choices=P0_VALUES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect existing derived snapshot JSONs into combined CSVs.",
    )
    args = parser.parse_args()
    seed_reliability()
    if not args.skip_snapshot_validations:
        snapshot_validations(
            selected_p0=args.p0,
            selected_seed=args.seed,
            selected_method=args.method,
            collect_only=args.collect_only,
        )
    print("Remaining supported analyses completed.")


if __name__ == "__main__":
    main()
