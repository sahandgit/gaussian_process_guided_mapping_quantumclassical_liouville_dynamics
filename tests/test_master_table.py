from __future__ import annotations

r"""
test_master_table.py
====================

Verifies ``ReviewerValidation.build_master_table`` (the ``report`` subcommand)
consolidates every validation artifact into one master table.  Uses synthetic
JSON/CSV fixtures matching the real schemas so it runs without torch/jax.
"""

import csv
import json
from pathlib import Path

import pytest

import ReviewerValidation as RV


def _write_fixtures(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    json.dump({
        "n_completed_scheme_runs": 8,
        "pbme_convergence": [{"case_id": "dt_N500_h0.5", "against": "dt_N500_h0.25",
                              "absolute_endpoint_differences": {"lw_P0": 0.0123}}],
        "pbme_replication": {"lw_P0": {"mean": 0.31, "sample_std": 0.004, "n": 3}},
    }, open(root / "campaign_metrics.json", "w"))

    with open(root / "endpoint_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case_id", "scheme", "dt", "lw_P0"])
        w.writeheader()
        w.writerow({"case_id": "dt_N500_h0.5", "scheme": "pbme", "dt": 0.5, "lw_P0": 0.31})

    (root / "manufactured").mkdir()
    json.dump({"n_train": 600, "n_query": 100, "seed": 123, "metrics": {
        "on_support": {"operator_Q": {"rmse": 2e-2, "linf": 6e-2, "relative_l2": 3e-2}},
        "off_support": {"operator_Q": {"rmse": 8e-2, "linf": 2e-1, "relative_l2": 1.1e-1}}}},
        open(root / "manufactured" / "manufactured_operator_metrics.json", "w"))

    (root / "projection").mkdir()
    json.dump({"source": "results/P0_20/midpoint", "snapshot_step": 800, "basis_rank": 4,
               "mean_relative_l2_leakage": 0.017, "max_relative_l2_leakage": 0.041,
               "n_bath_anchors": 20, "n_mapping_probes": 400},
              open(root / "projection" / "projection_leakage.json", "w"))

    (root / "baseline").mkdir()
    json.dump({"source": "results/P0_20/midpoint", "n_support": 1000,
               "weight_policy": "saved frozen geometric measure",
               "shape_errors": {"E1": 0.015, "E2": 0.02, "Einf": 0.05},
               "raw_norms": {"gp_on_grid": 0.98, "kde_on_grid": 0.99},
               "acceptance": {"applies": True, "metric": "E1", "threshold": 0.02, "passed": True}},
              open(root / "baseline" / "kde_gp_identical_support.json", "w"))

    (root / "reference").mkdir()
    json.dump({"tdse": {"P0": {"coarse": 0.5, "fine": 0.5001, "absolute_difference": 1e-4}},
               "qcle": {"P0": {"coarse": 0.48, "fine": 0.482, "absolute_difference": 2e-3}}},
              open(root / "reference" / "reference_convergence.json", "w"))

    json.dump({"cases": [{"case_id": "x"}]}, open(root / "campaign_plan.json", "w"))


def test_report_captures_every_test_group(tmp_path):
    _write_fixtures(tmp_path)
    summary = RV.build_master_table(tmp_path)
    groups = set(summary["tests"])
    expected = {"convergence_campaign", "manufactured_operator",
                "seo_projection_leakage", "kde_gp_identical_support",
                "reference_convergence"}
    assert expected <= groups, groups
    assert summary["n_rows"] >= 20
    assert Path(summary["csv"]).exists()
    assert Path(summary["markdown"]).exists()


def test_report_csv_contains_key_numbers(tmp_path):
    _write_fixtures(tmp_path)
    RV.build_master_table(tmp_path)
    rows = list(csv.DictReader(open(tmp_path / "master_validation_table.csv",
                                    encoding="utf-8")))
    # on/off-support manufactured Q must both be present
    got = {(r["test"], r["item"], r["quantity"]): r["value"] for r in rows}
    assert got[("manufactured_operator", "off_support:operator_Q", "rmse")] == "0.08"
    assert got[("manufactured_operator", "on_support:operator_Q", "rmse")] == "0.02"
    # KDE/GP acceptance decision preserved
    assert got[("kde_gp_identical_support",
                "results/P0_20/midpoint:acceptance", "passed")] == "True"


def test_report_markdown_escapes_pipes(tmp_path):
    _write_fixtures(tmp_path)
    RV.build_master_table(tmp_path)
    md = (tmp_path / "master_validation_table.md").read_text(encoding="utf-8")
    # the "|Δ endpoint| lw_P0" label must have its inner pipes escaped so the
    # Markdown table renders with the correct number of columns.
    assert "\\|Δ endpoint\\|" in md
    for line in md.splitlines():
        if line.startswith("|") and "---" not in line:
            assert line.count("|") - line.count("\\|") == 5   # 4 cells -> 5 bars


def test_report_empty_tree_is_graceful(tmp_path):
    summary = RV.build_master_table(tmp_path)
    assert summary["n_rows"] == 0
    assert Path(summary["markdown"]).exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
