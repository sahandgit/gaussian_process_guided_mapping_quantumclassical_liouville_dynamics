#!/usr/bin/env python3
"""Independent consistency checks for generated reviewer audit artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "reviewer_data_audit"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    required = [
        "PIPELINE_DATA_AUDIT_AND_THESIS_EVIDENCE.md",
        "THESIS_EVIDENCE_TABLES.tex",
        "EXAMINER_RESPONSE_EVIDENCE.md",
        "MISSING_DATA_AND_ANALYSES.md",
        "file_inventory.csv", "run_inventory.csv",
        "numerical_stability_audit.csv", "figure_metadata.csv",
        "metric_provenance.csv", "checksums_sha256.csv",
        "tables/run_status_matrix.csv",
        "tables/manufactured_operator_summary.csv",
        "tables/raw_conservation.csv",
        "tables/seed_replication_pairwise_distances.csv",
        "tables/seed_replication_reliability_summary.csv",
        "tables/seed_replication_method_dispersion_ratios.csv",
        "tables/population_physicality_audit.csv",
        "tables/replication_gp_health.csv",
        "tables/seo_projection_leakage_all_snapshots.csv",
        "tables/seo_projection_leakage_seed_summary.csv",
        "tables/kde_gp_identical_support_all_snapshots.csv",
        "plots/seed_replication_lw_P0.png",
        "plots/seo_projection_leakage_replication_snapshots.png",
    ]
    missing = [name for name in required if not (AUDIT/name).exists()]
    assert not missing, f"missing required outputs: {missing}"

    status = rows(AUDIT/"tables/run_status_matrix.csv")
    counts = {s: sum(r["status"] == s for r in status) for s in ("COMPLETE","INCOMPLETE","FAILED","MISSING","CONFIGURATION CONFLICT")}
    summary = json.loads((AUDIT/"audit_summary.json").read_text(encoding="utf-8"))
    for key, status_key in (("complete","COMPLETE"),("incomplete","INCOMPLETE"),("failed","FAILED"),("missing","MISSING"),("configuration_conflicts","CONFIGURATION CONFLICT")):
        assert summary[key] == counts[status_key], (key, summary[key], counts[status_key])
    assert len(status) == 38, len(status)

    # Six incomplete expected configurations must be missing MIDPOINT only.
    incomplete = [r for r in status if r["status"]=="INCOMPLETE"]
    assert len(incomplete) == 6
    assert all("midpoint.npz" in r["missing_outputs"] and "pbme.npz" not in r["missing_outputs"] for r in incomplete)

    # Stored manufactured E2 is copied exactly from its JSON.
    mfg = rows(AUDIT/"tables/manufactured_operator_metrics.csv")
    sample = next(r for r in mfg if r["canonical_latest_campaign"]=="True" and r["N"]=="300" and r["seed"]=="123" and r["query_set"]=="on_support" and r["quantity"]=="operator_Q")
    src = json.loads((ROOT/sample["source_file"]).read_text(encoding="utf-8"))
    assert float(sample["E2_relative"]) == float(src["metrics"]["on_support"]["operator_Q"]["relative_l2"])
    assert sample["E1_relative"] == "NOT COMPUTED"
    assert sample["Einf_relative"] == "NOT COMPUTED"

    # Baseline E1/E2/Einf copied exactly and P0=100 values are present.
    base = rows(AUDIT/"tables/kde_gp_identical_support.csv")
    p100 = next(r for r in base if r["canonical"]=="True" and r["P0"]=="100")
    src = json.loads((ROOT/p100["source_file"]).read_text(encoding="utf-8"))
    for key in ("E1","E2","Einf"):
        assert float(p100[key]) == float(src["shape_errors"][key])

    # Expanded common-support validation covers every saved replication snapshot.
    base_all = rows(AUDIT/"tables/kde_gp_identical_support_all_snapshots.csv")
    assert len(base_all) == 48, len(base_all)
    pbme_all = [r for r in base_all if r["method"] == "pbme"]
    assert len(pbme_all) == 24
    assert all(r["acceptance_applies"] == "True" for r in pbme_all)
    assert all(r["passed_declared_threshold"] == "True" for r in pbme_all)

    # Four-seed projection summaries have one row per P0/method/saved time.
    projection = rows(AUDIT/"tables/seo_projection_leakage_seed_summary.csv")
    assert len(projection) == 12, len(projection)
    assert all(r["n_independent_propagation_seeds"] == "4" for r in projection)

    # Independently recompute one pairwise seed trajectory L2 distance.
    pair_rows = rows(AUDIT/"tables/seed_replication_pairwise_distances.csv")
    sample_pair = next(
        r for r in pair_rows
        if r["P0"] == "20" and r["method"] == "midpoint"
        and r["observable"] == "lw_P0"
        and r["seed_a"] == "11" and r["seed_b"] == "29"
    )
    tau = np.linspace(0.0, 2.0, 41)
    arrays = []
    for seed in (11, 29):
        path = (
            ROOT / "reviewer_closure_20260726_174927"
            / "step9_repl_P020" / f"seed{seed}" / "midpoint.npz"
        )
        with np.load(path, allow_pickle=False) as z:
            arrays.append(
                np.interp(tau * 1500.0, np.asarray(z["t"]), np.asarray(z["lw_P0"]))
            )
    diff = arrays[0] - arrays[1]
    expected_l2 = float(np.sqrt(np.trapezoid(diff * diff, tau) / 2.0))
    assert np.isclose(
        float(sample_pair["time_averaged_L2_difference"]),
        expected_l2,
        rtol=1e-13,
        atol=1e-15,
    )

    # Every generated quantitative table has a provenance family.
    prov = rows(AUDIT/"metric_provenance.csv")
    assert len(prov) >= 22
    report = (AUDIT/"PIPELINE_DATA_AUDIT_AND_THESIS_EVIDENCE.md").read_text(encoding="utf-8")
    for phrase in ("NOT COMPUTED", "DATA ABSENT", "RUN INCOMPLETE", "independently sampled", "one-step refinement", "systematic MIDPOINT"):
        assert phrase in report, phrase

    # Check every checksum whose current file is immutable or generated and still exists.
    checksum_rows = rows(AUDIT/"checksums_sha256.csv")
    checked = 0
    for row in checksum_rows:
        path = ROOT/row["path"]
        if not path.exists() or path == AUDIT/"checksums_sha256.csv":
            continue
        assert sha(path) == row["sha256"], row["path"]
        checked += 1
    result = {
        "status": "PASS",
        "required_outputs": len(required),
        "status_rows": len(status),
        "checksums_verified": checked,
        "complete": counts["COMPLETE"],
        "incomplete": counts["INCOMPLETE"],
        "failed": counts["FAILED"],
        "missing": counts["MISSING"],
    }
    (AUDIT/"verification_result.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
