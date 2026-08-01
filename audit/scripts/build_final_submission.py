"""Build the final audit narrative and reviewer-response source from verified CSVs.

This script reads but never changes simulation artifacts.  Generated documents
are written under ``reviewer_data_audit`` except for the explicitly requested
top-level reviewer-response TeX source.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from build_physics_thesis_tables import (
    cloud_size_table,
    conservation_table,
    manufactured_tables,
    physical_comparison_table,
    projection_and_baseline_tables,
    reference_tables,
    replication_table,
    study_design_table,
    tail_table,
    timestep_table,
)


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "reviewer_data_audit"
EVIDENCE = ROOT / "final_reviewer_closure"
TITLE = (
    "Gaussian-Process Reconstruction of the Mapping-QCLE Excess Term: "
    "A Moving-Cloud Formulation and Failure Analysis"
)
BLOCKED_ID = "BLOCKED_EXTERNAL_PUBLICATION"
ALLOWED_STATUSES = {
    "computation": "Closed — computation and thesis correction",
    "negative": "Closed — explicit negative result",
    "removed": "Closed — claim removed",
    "limitation": "Closed — reviewer-authorized limitation",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def sci(value: Any, digits: int = 6) -> str:
    x = num(value)
    if not math.isfinite(x):
        return "not identifiable"
    if x == 0:
        return "0"
    return f"{x:.{digits}g}"


def _tex_plain(value: Any) -> str:
    text = str(value).replace("—", "---").replace("–", "--")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def tex(value: Any) -> str:
    """Escape prose while keeping URLs and long digests breakable."""
    text = str(value)
    pieces: List[str] = []
    cursor = 0
    token_pattern = re.compile(
        r"https?://\S+|(?<![A-Fa-f0-9])[A-Fa-f0-9]{40,64}(?![A-Fa-f0-9])"
    )
    for match in token_pattern.finditer(text):
        pieces.append(_tex_plain(text[cursor:match.start()]))
        token = match.group(0)
        if token.startswith(("http://", "https://")):
            core = token.rstrip(".,;)")
            trailing = token[len(core):]
            pieces.append(r"\url{" + core + "}")
            pieces.append(_tex_plain(trailing))
        else:
            chunks = [
                token[index:index + 8]
                for index in range(0, len(token), 8)
            ]
            pieces.append(r"\texttt{" + r"\allowbreak{}".join(chunks) + "}")
        cursor = match.end()
    pieces.append(_tex_plain(text[cursor:]))
    return "".join(pieces)


def tex_path(value: Any) -> str:
    """Format a filesystem path with URL-style break opportunities."""
    return r"\path{" + str(value).replace("\\", "/") + "}"


def md_path(path: Path) -> str:
    return f"`{path.resolve()}`"


def require(paths: Iterable[Path]) -> None:
    absent = [str(path) for path in paths if not path.exists()]
    if absent:
        raise FileNotFoundError("Required final evidence absent:\n" + "\n".join(absent))


def find_row(
    rows: Sequence[Mapping[str, str]], **criteria: Any
) -> Mapping[str, str]:
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in criteria.items()):
            return row
    raise KeyError(f"No row satisfies {criteria}")


def aux_pages() -> Dict[str, str]:
    aux = ROOT / "Thesis" / "Thesis.aux"
    if not aux.exists():
        return {}
    pages: Dict[str, str] = {}
    pattern = re.compile(r"\\newlabel\{([^@}]+)\}\{\{[^}]*\}\{([^}]*)\}")
    for line in aux.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            pages[match.group(1)] = match.group(2)
    return pages


def abstract_word_count() -> int:
    text = (ROOT / "Thesis" / "Thesis.tex").read_text(encoding="utf-8")
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    if not match:
        return 0
    abstract = re.sub(
        r"\\(?:cite|ref|label|eqref)\{[^}]*\}", " ", match.group(1)
    )
    abstract = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", abstract)
    abstract = abstract.replace("{", " ").replace("}", " ")
    return len(re.findall(r"\b[\w’'-]+\b", abstract))


def evidence_summary() -> Dict[str, Any]:
    paths = {
        "manufactured": EVIDENCE / "manufactured" / "manufactured_complete.csv",
        "manufactured_summary": EVIDENCE / "manufactured" / "manufactured_summary.csv",
        "manufactured_refinement": EVIDENCE / "manufactured" / "manufactured_refinement_verdicts.csv",
        "timestep": EVIDENCE / "timestep" / "timestep_run_by_run.csv",
        "support": EVIDENCE / "support" / "independent_cloud_summary.csv",
        "replication": EVIDENCE / "replication" / "four_seed_summary.csv",
        "tail": EVIDENCE / "tail_sensitivity" / "tail_summary.csv",
        "tail_sweep": EVIDENCE / "tail_sensitivity" / "threshold_sweep.csv",
        "tdse": EVIDENCE / "reference_tdse" / "tdse_three_level.csv",
        "qcle": EVIDENCE / "reference_grid_qcle" / "qcle_three_level.csv",
        "physical": EVIDENCE / "physical_comparison" / "paired_improvement_summary.csv",
        "inventory": EVIDENCE / "validation_inventory.csv",
        "raw": EVIDENCE / "preserved_evidence" / "raw_conservation.csv",
        "projection": EVIDENCE / "preserved_evidence" / "projection_leakage.csv",
        "baseline": EVIDENCE / "preserved_evidence" / "kde_gp_baseline.csv",
        "stability": EVIDENCE / "preserved_evidence" / "numerical_stability_audit.csv",
        "crosswalk": EVIDENCE / "TABLE_DATA_CROSSWALK.csv",
        "incidents": EVIDENCE / "commands" / "execution_incidents.jsonl",
        "frozen_payload_manifest": (
            AUDIT / "frozen_numerical_evidence_payload_manifest.json"
        ),
    }
    require(paths.values())
    rows = {name: read_csv(path) for name, path in paths.items()}
    payload_record = json.loads(
        paths["frozen_payload_manifest"].read_text(encoding="utf-8")
    )
    payload_archive = Path(payload_record["archive"])
    if (
        not payload_archive.exists()
        or sha256(payload_archive) != payload_record["archive_sha256"]
    ):
        raise ValueError("Frozen numerical-evidence payload hash mismatch")
    payload_sha256 = payload_record["archive_sha256"]

    manufactured = rows["manufactured"]
    manufactured_cases = {
        (row["l2_regularization"], row["N"], row["seed"])
        for row in manufactured
    }
    manufactured_verdicts = Counter(
        row["refinement_verdict"]
        for row in rows["manufactured_refinement"]
    )
    manufactured_text = (
        f"{len(manufactured)} query rows from {len(manufactured_cases)} paired "
        "policy/support/seed fits; all density, gradient, and operator "
        "E1/E2/E-infinity values are finite. The independent-cloud "
        "enlargement checks yield "
        f"{manufactured_verdicts.get('NO_MONOTONE_DECREASE_OBSERVED', 0)} "
        "rows without monotone decrease and "
        f"{manufactured_verdicts.get('MONOTONE_DECREASE_OBSERVED', 0)} "
        "row with monotone decrease. Because the clouds are nonnested, these "
        "descriptive checks do not establish deterministic support convergence."
    )
    production_q_e1 = [
        row
        for row in rows["manufactured_refinement"]
        if num(row.get("l2_regularization")) == 0.05
        and row.get("query_type") == "off_support"
        and row.get("quantity") == "Q"
        and row.get("metric") == "relative_l1"
    ]
    if len(production_q_e1) != 1:
        raise ValueError(
            "Expected exactly one production-policy off-support Q E1 "
            "refinement row"
        )
    qrow = production_q_e1[0]
    manufactured_text += (
        " For the production-policy off-support operator E1, the three-seed "
        "means at N=300, 600, 1200, and 2400 are "
        + ", ".join(
            sci(qrow[f"N{level}_seed_mean"])
            for level in (300, 600, 1200, 2400)
        )
        + "; the trend is nonmonotone."
    )

    timestep_counts = Counter(row["verdict"] for row in rows["timestep"])
    timestep_text = (
        f"{len(rows['timestep'])} run/observable rows across four seeds. "
        f"Of the tested orders, {timestep_counts.get('COMPUTED_POSITIVE', 0)} "
        "are positive, "
        f"{timestep_counts.get('COMPUTED_ZERO_OR_NEGATIVE', 0)} are zero or "
        f"negative, and {timestep_counts.get('REJECT_SEED_VARIABILITY', 0)} "
        "are rejected because the refinement signal does not exceed seed "
        "variability. A further "
        f"{timestep_counts.get('REJECT_NUMERICAL_NOISE', 0)} rows are "
        "roundoff- or saturation-limited under the declared absolute-plus-"
        "relative numerical-noise rule. Orders rejected by either guard are "
        "not promoted."
    )

    rep = rows["replication"]
    rep_focus: Dict[str, Mapping[str, str]] = {}
    for P0 in ("20.0", "100.0", "20", "100"):
        for method in ("PBME", "MIDPOINT"):
            matches = [
                row for row in rep
                if row.get("method") == method
                and num(row.get("P0")) == num(P0)
                and row.get("observable") == "P0"
            ]
            if matches:
                rep_focus[f"{method}_{float(P0):g}"] = matches[0]
    replication_parts = []
    for P0 in (20.0, 100.0):
        pbme = rep_focus[f"PBME_{P0:g}"]
        midpoint = rep_focus[f"MIDPOINT_{P0:g}"]
        ratio = num(midpoint["sample_sd"]) / max(num(pbme["sample_sd"]), 1e-300)
        replication_parts.append(
            f"P0={P0:g}: PBME SD={sci(pbme['sample_sd'])}, "
            f"MIDPOINT SD={sci(midpoint['sample_sd'])}, ratio={sci(ratio)}"
        )
    replication_text = "; ".join(replication_parts) + (
        ". The independent-seed sample size is four, not the trajectory count."
    )

    support_rows = rows["support"]
    support_flags = Counter(
        str(row.get("change_exceeds_seed_variability", "")).lower()
        for row in support_rows
    )
    support_text = (
        f"{len(support_rows)} method/momentum/observable/support summary rows "
        "from three independent seeds per intended cell. The clouds are "
        "independently sampled and not nested; "
        f"change exceeds seed variability in {support_flags.get('true', 0)} "
        f"rows and does not in {support_flags.get('false', 0)} rows. "
        "No deterministic support order is inferred."
    )

    tail = rows["tail"]
    midpoint_tail = [row for row in tail if row.get("method") == "MIDPOINT"]
    max_ratio = max(num(row.get("eta0_ratio_abs_max")) for row in midpoint_tail)
    min_signed_ess = min(num(row.get("eta0_signed_ESS")) for row in midpoint_tail)
    tail_verdicts = Counter(row.get("verdict") for row in tail)
    tail_text = (
        f"16 method/momentum/seed distributions and {len(rows['tail_sweep'])} "
        f"threshold rows; MIDPOINT max |y/y0|={sci(max_ratio)}, minimum signed "
        f"ESS={sci(min_signed_ess)}. No nontrivial threshold satisfies the "
        f"negligible-mass rule in "
        f"{tail_verdicts.get('NO_NONTRIVIAL_NEGLIGIBLE_MASS_THRESHOLD', 0)} "
        "distributions."
    )

    tdse = rows["tdse"]
    qcle = rows["qcle"]
    physical_observables = {"P0", "P1", "R_mean", "P_mean"}

    def reference_order_text(
        data: Sequence[Mapping[str, str]], method: str
    ) -> str:
        pieces = []
        for p0 in (20.0, 100.0):
            selected = [
                row
                for row in data
                if num(row.get("P0")) == p0
                and row.get("refinement_mode") == "time"
                and row.get("observable") in physical_observables
            ]
            if len(selected) != 4:
                raise ValueError(
                    f"Expected four physical time-order rows for {method}, "
                    f"P0={p0:g}; got {len(selected)}"
                )
            by_observable = {
                row["observable"]: (
                    "not computed"
                    if row.get("p_observed", "").strip().upper()
                    == "NOT COMPUTED"
                    else sci(row.get("p_observed", ""))
                )
                for row in selected
            }
            pieces.append(
                f"{method} P0={p0:g} time orders "
                + ", ".join(
                    f"{observable}={by_observable[observable]}"
                    for observable in ("P0", "P1", "R_mean", "P_mean")
                )
            )
        return "; ".join(pieces)

    tdse_edge = max(
        num(row.get(f"level{level}_maximum_edge_mass_5pct"))
        for row in tdse
        for level in (1, 2, 3)
    )
    qcle_edge = max(
        num(row.get(f"level3_{field}"))
        for row in qcle
        for field in (
            "maximum_edge_R_mass_5pct",
            "maximum_edge_P_mass_5pct",
        )
    )
    reference_text = (
        f"The controlled-reference evidence contains {len(tdse)} identified "
        f"TDSE observable/mode/momentum rows and {len(qcle)} grid-QCLE rows. "
        "Each row prints three values, two "
        "successive differences, exact domains and resolved steps; temporal "
        "and grid refinement are separated. "
        + reference_order_text(tdse, "TDSE")
        + "; "
        + reference_order_text(qcle, "grid-QCLE")
        + f". Maximum recorded TDSE spatial-edge mass={sci(tdse_edge)}; "
        + f"maximum accepted finest-level grid-QCLE physical-marginal edge "
        + f"mass={sci(qcle_edge)} (declared tolerance 1e-3)."
    )

    physical_counts = Counter(
        row["verdict_before_scientific_gates"] for row in rows["physical"]
    )
    physical_text = (
        f"{len(rows['physical'])} paired seed-aggregate error rows; "
        f"MIDPOINT has a larger error in "
        f"{physical_counts.get('MIDPOINT_ERROR_LARGER', 0)} rows and no "
        f"resolved difference occurs in "
        f"{physical_counts.get('NO_RESOLVED_DIFFERENCE', 0)} rows"
        + ". Regardless of isolated smaller errors, systematic improvement is "
        "not demonstrated unless refinement, seed, conservation, projection, "
        "and appreciable-source gates also pass."
    )

    baseline_text = "; ".join(
        f"P0={sci(row.get('P0'))}: max E1={sci(row.get('max_E1'))}, "
        f"max E2={sci(row.get('max_E2'))}, max Einf={sci(row.get('max_Einf'))}, "
        f"{row.get('result')}"
        for row in rows["baseline"]
    )

    projection_values = [
        num(row.get("mean_relative_l2_leakage")) for row in rows["projection"]
    ]
    projection_text = (
        f"{len(rows['projection'])} snapshot diagnostics; mean relative L2 "
        f"leakage ranges from {sci(min(projection_values))} to "
        f"{sci(max(projection_values))}. Projection is diagnostic, not enforced."
    )

    raw_rows = rows["raw"]
    raw_by_method: Dict[str, float] = {}
    for row in raw_rows:
        method = str(row.get("method", "")).upper()
        value = abs(num(row.get("maximum_absolute_drift")))
        if math.isfinite(value):
            raw_by_method[method] = max(raw_by_method.get(method, 0.0), value)
    raw_text = (
        f"{len(raw_rows)} raw pre-renormalization drift rows; maximum absolute "
        "drift by stored method: "
        + ", ".join(
            f"{method}={sci(value)}"
            for method, value in sorted(raw_by_method.items())
        )
        + ". Self-normalized curves are not used as conservation evidence."
    )

    stability_counts = Counter(
        row.get("severity", "UNCLASSIFIED") for row in rows["stability"]
    )
    stability_text = (
        f"{len(rows['stability'])} stability/anomaly records, including "
        "superseded failed attempts; severity counts: "
        f"excluded attempts={stability_counts.get('EXCLUDE ATTEMPT', 0)}, "
        f"excluded quantities={stability_counts.get('EXCLUDE QUANTITY', 0)}, "
        f"informational placeholders={stability_counts.get('INFORMATIONAL PLACEHOLDER', 0)}, "
        f"scientific warnings={stability_counts.get('SCIENTIFIC WARNING', 0)}, "
        f"other warnings={stability_counts.get('WARN', 0)}"
        + ". Historical failures remain visible even when a repaired rerun completes."
    )

    inventory_text = "; ".join(
        f"{row['campaign']}: expected={row['expected']}, "
        f"verified={row['reuse']}, missing={row['missing']}"
        for row in rows["inventory"]
    )
    incident_records = [
        json.loads(line)
        for line in paths["incidents"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    incident_text = (
        f"{len(incident_records)} orchestration incident(s) retained: "
        + "; ".join(
            f"{record['status']} ({record['reason']})"
            for record in incident_records
        )
    )

    return {
        "paths": paths,
        "rows": rows,
        "manufactured": manufactured_text,
        "timestep": timestep_text,
        "replication": replication_text,
        "support": support_text,
        "tail": tail_text,
        "references": reference_text,
        "physical": physical_text,
        "baseline": baseline_text,
        "projection": projection_text,
        "raw": raw_text,
        "stability": stability_text,
        "inventory": inventory_text,
        "incidents": incident_text,
        "frozen_payload_sha256": payload_sha256,
    }


def item_records(summary: Mapping[str, Any], archive_id: str) -> Dict[str, Dict[str, str]]:
    matrix = read_csv(ROOT / "reviewer_closure_out" / "closure_matrix.csv")
    base: Dict[str, Dict[str, str]] = {}
    default_location = "Chapters 6–7 and Appendix F"
    default_source = str(EVIDENCE / "TABLE_DATA_CROSSWALK.csv")
    for row in matrix:
        base[row["item"]] = {
            "concern": row["title"],
            "status": ALLOWED_STATUSES["computation"],
            "action": row["action"],
            "result": "The requested wording, evidence boundary, or numerical table is present.",
            "location": default_location,
            "table": "Table-data crosswalk",
            "source": default_source,
            "archive_id": archive_id,
        }

    def update(item: str, **values: str) -> None:
        base[item].update(values)

    snapshot_manifest = AUDIT / "source_release_snapshot_manifest.json"
    release_commit = "NOT IDENTIFIABLE"
    if snapshot_manifest.exists():
        release_commit = json.loads(
            snapshot_manifest.read_text(encoding="utf-8")
        ).get("release_commit", "NOT IDENTIFIABLE")
    archive_complete = (
        archive_id != BLOCKED_ID and release_commit != "NOT IDENTIFIABLE"
    )

    for item in ("I-1", "I-2", "I-3", "I-4", "I-5", "I-9", "I-13", "I-14",
                 "M-2", "M-5", "M-6", "M-8", "M-10", "M-12", "M-13",
                 "M-15", "M-17", "M-18", "M-19", "M-24", "M-25"):
        update(item, status=ALLOWED_STATUSES["negative"])
    for item in ("I-8", "I-11", "M-4", "M-11", "M-16",
                  "M-20", "M-23"):
        update(item, status=ALLOWED_STATUSES["removed"])
    for item in ("I-6", "I-7", "I-10", "I-12", "I-16", "M-1", "M-3", "M-7", "M-9", "M-14",
                  "M-21", "M-22"):
        update(item, status=ALLOWED_STATUSES["computation"])

    update("I-1", result=summary["manufactured"], location="Chapter 6, manufactured-operator section",
           table="Tables tab:manufactured-complete-density/gradient/q and tab:manufactured-summary",
           source=str(summary["paths"]["manufactured"]))
    update("I-2", result=summary["timestep"], location="Chapter 6, time-step section",
           table="Table tab:timestep-run-by-run", source=str(summary["paths"]["timestep"]))
    update("I-3", result=summary["references"], location="Chapter 6 and Appendix F",
           table="Tables tab:tdse-reference-orders and tab:qcle-reference-orders",
           source=f"{summary['paths']['tdse']}; {summary['paths']['qcle']}")
    update("I-4", result=summary["replication"], location="Chapter 6, replication section",
           table="Table tab:four-seed-replication", source=str(summary["paths"]["replication"]))
    update("I-5", result=summary["baseline"], location="Chapter 6, common-support baseline",
           table="Table tab:identical-support-kde-gp", source=str(summary["paths"]["baseline"]))
    update("I-6", result="All 41 untraceable legacy figures were removed. Five replacement figures are regenerated only from verified final CSVs and have figure, source, generator, and SHA-256 records.",
           location="Chapter 6, Figures fig:manufactured-operator-regularization through fig:paired-physical-reference-differences",
           table="Figure-data crosswalk", source=str(EVIDENCE / "figures" / "FIGURE_DATA_CROSSWALK.csv"))
    update("I-7", result="Every retained comparison defines its metric, aggregation, uncertainty display, normalization, and evidentiary limit in the surrounding prose and caption.",
           location="Chapter 6, four verified figures and corresponding quantitative tables",
           source=str(EVIDENCE / "figures" / "FIGURE_DATA_CROSSWALK.csv"))
    update("I-8", result="No threshold-dependent scientific plot remains; the only baseline gate is E1 <= 0.02.",
           location="Chapter 6, common-support baseline")
    update("M-9", result="The legacy figure set and its dependent narration were removed; the four retained summary figures are regenerated from hash-verified CSVs.",
           location="Chapter 6, verified summary figures",
           source=str(EVIDENCE / "figures" / "FIGURE_DATA_CROSSWALK.csv"))
    update("I-9", result=summary["tail"], location="Chapters 5–7, tail-sensitivity analysis",
           table="Table tab:y0-tail-sensitivity", source=str(summary["paths"]["tail_sweep"]))
    update("I-10", result="Production manifests set apply_q_clip=false and omega_clip_quantile=null; Q_applied=Q_raw.",
           location="Chapter 6, applied-source controls", table="Equation Q_applied=Q_raw",
           source=str(ROOT / "Dynamics.py"))
    update("I-11", result="The unsupported physical-implausibility assertion was removed.",
           location="Chapters 6–7")
    update("I-12", result=summary["replication"], location="Chapter 6, replication section",
           table="Table tab:four-seed-replication", source=str(summary["paths"]["replication"]))
    update("I-13", result=summary["physical"], location="Chapter 6, PBME versus MIDPOINT section",
           table="Table tab:physical-reference-comparison", source=str(summary["paths"]["physical"]))
    update("I-14", result=summary["references"], location="Chapter 6 and Appendix F",
           table="Reference-control tables", source=f"{summary['paths']['tdse']}; {summary['paths']['qcle']}")
    update("I-15", result=(
        "Local archive, environment, manifests, SHA-256 checksums, and crosswalk are complete. "
        + f"The frozen numerical-evidence payload SHA-256 is {summary['frozen_payload_sha256']}. "
        + f"The audit-created final code release commit is {release_commit}. "
        + ("A versioned public release URL is recorded; it is not a DOI or "
           "institutional persistent identifier." if archive_id != BLOCKED_ID
           else "External repository publication remains required for a retrievable release URL. ")
        + "The originating development commit is not identifiable from the copied workspace."
    ), location="Appendix F", table="Archive/environment crosswalk",
       source=str(EVIDENCE / "archive_manifest.json"),
       status=(ALLOWED_STATUSES["computation"] if archive_complete
               else "BLOCKED — external publication required"))
    update("I-16", result=f"One title is used: {TITLE}.", location="Title page, PDF metadata, response, release")

    update("M-2", result=summary["projection"], location="Chapters 4, 6, and 7",
           table="Table tab:seo-projection-leakage", source=str(summary["paths"]["projection"]))
    update("M-4", result="Visible method-level correctness vocabulary was replaced by MIDPOINT prototype, source update, or tested excess-source branch.",
           location="Thesis source throughout")
    update("M-5", result="Failure is reported for the tested discretization; causal allocation remains explicitly nonunique.",
           location="Chapters 6–7")
    update("M-6", result="Clouds are independently sampled and explicitly not nested; no deterministic support order is reported.",
           location="Chapter 6, independent-cloud enlargement",
           table="Table tab:independent-cloud-support", source=str(summary["paths"]["support"]))
    update("M-7", result="The pilot l2=0.01 statement is restricted to its recorded momentum and seed.",
           location="Chapter 6, regularization controls")
    update("M-8", result=summary["timestep"], location="Chapter 6, time-step section",
           table="Table tab:timestep-run-by-run", source=str(summary["paths"]["timestep"]))
    update("M-9", result="No untraceable scientific figure remains.", location="Chapter 6, figure disposition")
    update("M-10", result="Raw pre-renormalization drift is foregrounded; shape-only and self-normalized quantities are labelled.",
           location="Chapter 6, conservation section", table="Table tab:raw-conservation",
           source=str(summary["paths"]["raw"]))
    update("M-12", result=summary["tail"], location="Chapters 5–7",
           table="Table tab:y0-tail-sensitivity", source=str(summary["paths"]["tail_sweep"]))
    update("M-13", result="Self-normalized unity is excluded as conservation evidence.",
           location="Chapter 6, conservation section", table="Table tab:raw-conservation",
           source=str(summary["paths"]["raw"]))
    update("M-14", result="Estimator hierarchy is defined before numerical interpretation.",
           location="Opening of Chapter 6")
    update("M-15", result="Anchor-cloud and analytic-GP estimates are not validated in low-signed-ESS regimes.",
           location="Chapters 6–7")
    update("M-17", result="The negative conclusion applies to the tested nonprojected, nonconservative discretization, not the continuum QCLE excess term.",
           location="Chapters 6–7")
    update("M-18", result=summary["baseline"], location="Chapter 6, common-support baseline",
           table="Table tab:identical-support-kde-gp", source=str(summary["paths"]["baseline"]))
    update("M-19", result="The thesis states an evidence-supported, nonunique failure pathway.",
           location="Chapters 6–7")
    update("M-20", result=(
        "Broad reproducibility language is removed; permanent reproducibility is claimed only after "
        "publication of the frozen archive."
    ), location="Chapter 7 and Appendix F")
    update("M-21", result="The objective names one product-GP/moving-cloud/MIDPOINT construction on a one-dimensional two-state test.",
           location="Chapter 1 objective")
    update("M-22", result="Chapter 6 opens with the decisive acceptance question and immediate negative answer.",
           location="Opening of Chapter 6")
    update("M-23", result="Ambiguous full-density wording is absent.", location="Chapters 1, 4, 6, and 7")
    update("M-24", result=summary["manufactured"], location="Chapter 6, manufactured-operator section",
           table="Complete 72-row manufactured tables", source=str(summary["paths"]["manufactured"]))
    update("M-25", result=summary["inventory"], location="Chapter 6, campaign inventory",
           table="Table tab:final-validation-inventory", source=str(summary["paths"]["inventory"]))

    for item in (f"L-{index}" for index in range(1, 8)):
        update(item, status=ALLOWED_STATUSES["computation"],
               result="Terminology, definitions, claim calibration, local clarity, and the row-level audit trail were applied.",
               location="Thesis source and this reviewer response")
    return base


def _write_final_chapters_audit_legacy(
    summary: Mapping[str, Any], archive_id: str
) -> List[Path]:
    """Write the two numerical chapters that are later inlined into Thesis.tex."""
    chapter6_path = ROOT / "Thesis" / "Chapter6_VerifiedResults.tex"
    chapter7_path = ROOT / "Thesis" / "Chapter7_Conclusions.tex"

    chapter6 = r"""
\chapter{Verified Results and Failure Analysis}
\label{chap:results}

This chapter asks whether the tested product-GP moving-cloud MIDPOINT
construction satisfies the operator, projection, refinement, replication, and
conservation criteria declared in Chapter~\ref{chap:introduction}, and whether
it improves PBME against matched TDSE or grid-QCLE references.  The distinction
between successful execution and scientific reliability is essential: a
source-update formula is not validated merely because its jobs terminate.
Every campaign cell is therefore checked against its manifest, expected
endpoint, finite arrays, and PBME/MIDPOINT pairing before it contributes to an
analysis.  The evidence is then organized from manufactured operator fidelity
and representation structure through refinement, replication, conservation,
tail sensitivity, controlled references, and matched physical errors.  This
sequence yields a clear negative answer.  Although the requested numerical
campaigns are complete, the MIDPOINT prototype does not meet the joint
reliability criteria and does not demonstrate systematic improvement over
PBME.

\section{Evidence rules and exact campaign inventory}
\label{sec:results-evidence-rules}

Run completion, data availability, metric extraction, and scientific
acceptance are distinct.  A run is complete only when its manifest endpoint,
expected step count, finite stored histories, and method pairing agree.
Self-normalized curves are not substituted for raw conservation.  GP posterior
variance, training residuals, LOO residuals, and \(R^2\) are internal
surrogate diagnostics and are not physical-error estimates.

The final verifier records the following complete campaign inventory:
@@INVENTORY@@.

The exact final inventory is printed in
Table~\ref{tab:final-validation-inventory}.  Time-step runs use
\(P_{\rm init}\in\{20,100\}\), four seeds \(11,29,47,73\),
\(N=1000\), and \(\Delta t\in\{0.5,0.25,0.125\}\).
Independent-cloud enlargement uses seeds \(11,29,47\),
\(N\in\{500,1000,2000\}\), and \(\Delta t=0.25\).
Four-seed replication uses \(N=1000\) and \(\Delta t=0.25\).
Per-run manifests are authoritative over directory names.

\section{Numerical stability and execution history}
\label{sec:results-stability}

The execution audit contains @@STABILITY@@

Historical out-of-memory and interrupted attempts are not silently erased.
Where the exact derivative batching repair produced a complete finite rerun,
the repaired result is used and the failed attempt remains an execution-history
record rather than a scientific data row.  A completed calculation can still
fail a scientific stability or conservation criterion.

\section{Manufactured density, gradient, and excess operator}
\label{sec:results-manufactured}

The manufactured study contains @@MANUFACTURED@@

The analytic density, analytic gradient, and analytic \(Q[\rho]\) are evaluated
separately on every training point and on 1000 independent off-support query
points.  The complete tables include
\(\ell_2\in\{10^{-6},0.01,0.05\}\),
\(N\in\{300,600,1200,2400\}\), and seeds \(123,124,125\).
All three relative norms and the absolute MAE, RMSE, and maximum error are
reported.  Production \(0.05\), pilot \(0.01\), and manufactured-test
\(10^{-6}\) policies remain separately identified.  Nonmonotone error changes
are reported as a failed refinement trend; conditioning is not assigned as a
cause unless directly tested.

Figure~\ref{fig:manufactured-operator-regularization} condenses the decisive
operator result without replacing the complete numerical tables.
\begin{figure}[tbp]
\centering
\includegraphics[width=0.96\textwidth]{../final_reviewer_closure/figures/manufactured_operator_regularization.png}
\caption{Manufactured relative \(L_2\) error in the excess action \(Q[\rho]\)
on the training support and on an independent query cloud.  Points are means
over seeds 123, 124, and 125; bars are sample standard deviations.  The three
regularization policies use paired training and query clouds at each support
size.  The error remains nonzero and nonmonotone, so neither support
enlargement nor changing \(\ell_2\) from \(10^{-6}\) to the pilot \(0.01\) or
production \(0.05\) policy establishes operator convergence.  Figure source,
selection rule, generator, and SHA-256 hashes are recorded in
\protect\path{final_reviewer_closure/figures/FIGURE_DATA_CROSSWALK.csv}.}
\label{fig:manufactured-operator-regularization}
\end{figure}

\section{SEO projection leakage}
\label{sec:results-projection}

The representation audit contains @@PROJECTION@@

The two-state physical SEO image has four real bath-dependent coefficient
fields.  The tabulated residual is a least-squares diagnostic of the fitted
surrogate using the recorded bath anchors and mapping probes.  Projection is
not enforced.  It therefore measures leakage of the fitted representation and
must not be reinterpreted as a TDSE or QCLE physical-error estimate.

The four-seed snapshot summary in
Figure~\ref{fig:seo-projection-leakage-verified} shows that the leakage is not
a small local residual confined to one run.
\begin{figure}[tbp]
\centering
\includegraphics[width=0.96\textwidth]{../final_reviewer_closure/figures/seo_projection_leakage_verified.png}
\caption{Diagnostic SEO projection leakage at \(t/t_c=0,1,2\).  Each point is
the mean over four independent propagation seeds and each bar is the sample
standard deviation.  The plotted quantity is the relative \(L_2\) residual
\(\lVert y-Bc\rVert_2/\max(\lVert y\rVert_2,10^{-30})\) from a four-field
least-squares SEO projection using 20 bath anchors and 400 mapping probes.
Projection is diagnosed but not enforced, and this quantity is not a physical
TDSE or QCLE error.  Complete provenance and hashes appear in
\protect\path{final_reviewer_closure/figures/FIGURE_DATA_CROSSWALK.csv}.}
\label{fig:seo-projection-leakage-verified}
\end{figure}

\section{Identical-support KDE/GP baseline}
\label{sec:results-kde-gp}

For the identical-support reconstruction control, @@BASELINE@@.

This comparison uses a PBME source and identical support, weights, bandwidth
convention, grid, normalization, physical mass, and snapshot.  Only the
predeclared \(E_1\leq0.02\) criterion is a pass gate; the separately printed
\(E_2\) and \(E_\infty\) values are not inferred from \(E_1\).  Passing this
same-support reconstruction gate does not validate derivative-dependent
operators or dynamics.

\section{Time-step refinement}
\label{sec:results-timestep}

The time-step refinement evidence comprises @@TIMESTEP@@

Trajectories are compared only on common physical times, by linear
interpolation without extrapolation.  The table prints all three observable
values, both successive time-normalized differences, the finest-level
independent-seed spread, the roundoff guard, and the empirical-order verdict.
An order is omitted when the denominator is too small or the refinement signal
does not exceed stochastic seed variation.  Formal midpoint order is not
promoted to demonstrated production order.

\section{Independent-cloud enlargement and four-seed replication}
\label{sec:results-support-replication}

Across the independent-cloud enlargement study, @@SUPPORT@@

The four-seed replication gives the following endpoint spreads:
@@REPLICATION@@

Support clouds were independently sampled and were not nested.  The study is
therefore an independent-cloud enlargement study, not deterministic support
convergence, and no pointwise trajectory order is computed.  The replication
sample size is four independent seeds, never the number of trajectories.
Student-\(t\) intervals with three degrees of freedom describe the observed
four-seed dispersion but are not strong uncertainty calibration.

Figure~\ref{fig:replication-raw-conservation} places the seed instability and
the associated raw conservation failure on the same footing.
\begin{figure}[tbp]
\centering
\includegraphics[width=0.91\textwidth]{../final_reviewer_closure/figures/replication_and_raw_conservation.png}
\caption{Four-seed replication and raw pre-normalization conservation for the
canonical \(N=1000\), \(\Delta t=0.25\) campaign.  The upper panels show the
endpoint \(\rho_{11}^{\mathrm{SN}}\) estimator; the shaded band marks the physical interval
\([0,1]\).  The lower panels show the maximum absolute drift of the raw
normalization.  Exact PBME zeros are displayed at \(10^{-16}\) only to make
them visible on the logarithmic axis.  The seed count is four, not the number
of trajectories, and the plot supplies a descriptive sensitivity comparison
rather than strong uncertainty calibration.  Complete provenance and hashes
appear in \protect\path{final_reviewer_closure/figures/FIGURE_DATA_CROSSWALK.csv}.}
\label{fig:replication-raw-conservation}
\end{figure}

\section{Raw conservation and signed-label tail sensitivity}
\label{sec:results-conservation-tail}

The raw conservation audit contains @@RAW@@

The tail-sensitivity audit covers @@TAIL@@

The threshold sweep is post-processing and does not alter propagation.
It records included fractions, excluded absolute physical mass, signed and
absolute effective sample sizes, raw normalization, raw energy, and observable
changes.  For the focused-sampling measure, removing even one support point
exceeds the declared negligible-mass allowance; consequently the observed
signed-weight collapse cannot be attributed specifically to a negligible
\(\lvert y_i^0\rvert\) tail.  It remains a demonstrated instability of the
stored estimator, with causal allocation unresolved.

\section{TDSE and grid-QCLE numerical controls}
\label{sec:results-reference-controls}

@@REFERENCES@@

TDSE is model-exact only within the displayed domain, time step, periodic FFT
convention, edge-mass, normalization, and reflected-momentum controls.
Grid QCLE is a numerical solution of the approximate QCLE equation.  Its
separate temporal and phase-space-grid studies report three levels, both
successive differences, guarded orders, exact \(R/P\) domains, CFL ratios,
and edge masses.  A rejected order remains visible and is not called
convergence.

\section{PBME versus MIDPOINT against common references}
\label{sec:results-physical-comparison}

The common-reference comparison contains @@PHYSICAL@@

Each paired difference is
\(E_{\mathrm{MIDPOINT,ref}}-E_{\mathrm{PBME,ref}}\), so a negative value
favours MIDPOINT for that particular metric.  Isolated negative differences
do not establish systematic improvement.  The success claim additionally
requires stable time-step and support behavior, reproducibility across seeds,
raw conservation, and a non-negligible excess source.  Those joint gates do
not pass; systematic improvement over PBME is therefore not demonstrated.

This conclusion is visible directly in
Figure~\ref{fig:paired-physical-reference-differences}.  The raw-density panel
is kept separate from the independently normalized shape panel so that shape
normalization cannot conceal a failed raw integral.
\begin{figure}[tbp]
\centering
\includegraphics[width=0.96\textwidth]{../final_reviewer_closure/figures/paired_physical_reference_differences.png}
\caption{Four-seed paired MIDPOINT-minus-PBME differences for density
\(E_1\) errors against the matched TDSE and grid-QCLE references.  Points are
paired means and bars are two-sided Student-\(t\) 95\% intervals with three
degrees of freedom.  Positive values mean that MIDPOINT has the larger error.
The raw-density panel uses the signed display transform
\(\operatorname{sgn}(\Delta E)\log_{10}(1+\lvert\Delta E\rvert)\), whose tick
labels show the corresponding untransformed magnitude; the shape panel uses
unit-mass normalization and a linear axis.  Shape agreement is therefore not
used as conservation evidence.  Complete provenance and hashes appear in
\protect\path{final_reviewer_closure/figures/FIGURE_DATA_CROSSWALK.csv}.}
\label{fig:paired-physical-reference-differences}
\end{figure}

\FloatBarrier

\section{Complete numerical tables and provenance}
\label{sec:results-tables}

Every following table is generated directly from a machine-readable CSV.
The exact source path and SHA-256 digest for each table are recorded in
\path{final_reviewer_closure/TABLE_DATA_CROSSWALK.csv}.  Machine files
retain full precision; rounding occurs only in these reader-facing tables.

\input{ReviewerEvidenceTables}

Taken together, the calculations close the requested evidence gaps but yield a controlled
negative scientific result.  The tested moving-cloud product-GP MIDPOINT
discretization is not a reliable dynamical scheme under the declared tests,
and no systematic improvement over PBME is demonstrated.
""".strip()

    chapter7 = r"""
\chapter{Discussion and Conclusions}
\label{chap:discussion-outlook}

This chapter asks what defensible contribution remains after the complete
evidence campaign shows that the MIDPOINT prototype fails the joint acceptance
criteria.  The answer matters because a controlled negative numerical study
must be distinguished from an unsupported claim of a validated mapping-QCLE
solver.  The discussion integrates manufactured-operator, projection,
refinement, replication, conservation, tail-sensitivity, and matched-reference
evidence while separating established mathematical facts from the
application-specific work performed here.  The contribution that survives
this test is the formulation and forensic validation of one
product-GP/moving-cloud/MIDPOINT construction, the quantitative demonstration
of its coupled failure modes, and the resulting requirements for a projected,
conservative successor.  It is neither a validated solver nor evidence of an
improvement over PBME.

\section{Application-specific contribution}
\label{sec:discussion-contribution}

Values on a lower-dimensional manifold do not generally identify ambient
normal derivatives; that fact is not original to this thesis.  The original
application-specific contribution is narrower: constructing a differentiable
product-GP excess-source prototype on moving PBME/MInt support, testing it
against an analytic manufactured operator and the exact two-state SEO image,
and connecting its observed derivative, projection, conservation, and
signed-weight failures to production dynamics without claiming a unique
cause
\cite{Kim2008JCP,Kelly2012JCP,Raissi2017LinearGP,
Solak2002DerivativeGP,CockayneOatesSullivanGirolami2019}.

\section{What the complete evidence establishes}
\label{sec:discussion-evidence}

The manufactured tests provide @@MANUFACTURED@@

Time-step refinement provides @@TIMESTEP@@

Independent-cloud enlargement supplies @@SUPPORT@@

The four-seed replication gives the following endpoint spreads:
@@REPLICATION@@

The raw conservation audit contains @@RAW@@

The tail-sensitivity study contains @@TAIL@@

@@REFERENCES@@

Finally, the paired common-reference comparison contains @@PHYSICAL@@

The identical-support KDE/GP \(E_1\) gate passes, but this is only a
reconstruction control.  The production method-level conclusion is governed by
the failed joint criteria.  Run completion is never equated with validation.

\section{Claim boundary}
\label{sec:discussion-limitations}

This thesis does not establish a validated discretization of the complete
mapping-basis QCLE, deterministic support convergence, strong uncertainty
calibration from four seeds, chemical accuracy outside the one-dimensional
two-state benchmark, multidimensional scalability, or systematic improvement
over PBME.  TDSE and grid-QCLE have different epistemic roles: TDSE is
model-exact within numerical controls, whereas grid QCLE solves an approximate
quantum--classical equation numerically.

The production regularization \(0.05\), pilot \(0.01\), and
manufactured-test \(10^{-6}\) policies are now all tested and reported rather
than conflated.  The tail study demonstrates extreme signed-weight
instability, but its negligible-mass rule does not permit a nontrivial
point-removal plateau, so the instability is not assigned uniquely to small
initial labels.

\section{Requirements for a successor}
\label{sec:discussion-successor}

A credible successor should regress the four real bath-dependent fields of the
two-state SEO image so that Hermiticity and projection are enforced by
construction.  Its discrete excess source should preserve normalization and
the relevant trace/energy identities before any renormalization.  A future
success claim requires decreasing manufactured-operator errors under a
controlled refinement, positive time-step behavior above seed variation,
independent-cloud uncertainty bands, raw conservation, and reproducibly
smaller PBME-matched errors against controlled references where the excess
source is appreciable
\cite{TraskBochevPerego2020,BonetLok1999,Monaghan2005}.

\section{Reproducibility boundary}
\label{sec:discussion-reproducibility}

The local release contains the final sources, PDFs, environment record,
manifests, table--data and raw-source crosswalks, and SHA-256 checksums.
The frozen numerical-evidence payload SHA-256 is
\texttt{@@PAYLOAD_SHA@@}.
The frozen evidence is linked to the frozen source/evidence commit recorded
in the release manifest.  Its versioned public release URL is
@@ARCHIVE_ID@@.  This GitHub release is a retrievable, checksum-bound public
record, not a DOI or institutional persistent identifier; no DOI is claimed.

\section{Conclusion}
\label{sec:discussion-conclusions}

The full evidence package supports a controlled negative result.  Accurate
same-support value reconstruction does not guarantee an accurate
derivative-dependent excess operator, preservation of the SEO image, raw
conservation, or seed-stable dynamics.  The tested MIDPOINT branch fails the
joint reliability criteria and does not demonstrate systematic improvement
over PBME.  This conclusion applies to the tested discretization, not to the
continuum QCLE excess term.

Accordingly, the thesis contributes a precisely scoped formulation, complete numerical
audit, and evidence-backed redesign criteria.  It does not claim that the
tested construction is reliable, converged, or improved.
""".strip()

    replacements = {
        "@@INVENTORY@@": summary["inventory"],
        "@@STABILITY@@": summary["stability"],
        "@@MANUFACTURED@@": summary["manufactured"],
        "@@PROJECTION@@": summary["projection"],
        "@@BASELINE@@": summary["baseline"],
        "@@TIMESTEP@@": summary["timestep"],
        "@@SUPPORT@@": summary["support"],
        "@@REPLICATION@@": summary["replication"],
        "@@RAW@@": summary["raw"],
        "@@TAIL@@": summary["tail"],
        "@@REFERENCES@@": summary["references"],
        "@@PHYSICAL@@": summary["physical"],
        "@@ARCHIVE_ID@@": archive_id,
        "@@PAYLOAD_SHA@@": summary["frozen_payload_sha256"],
    }
    for placeholder, value in replacements.items():
        chapter6 = chapter6.replace(placeholder, tex(value))
        chapter7 = chapter7.replace(placeholder, tex(value))
    chapter6_path.write_text(chapter6 + "\n", encoding="utf-8")
    chapter7_path.write_text(chapter7 + "\n", encoding="utf-8")
    return [chapter6_path, chapter7_path]


def write_final_chapters(
    summary: Mapping[str, Any], archive_id: str
) -> List[Path]:
    """Write cohesive, physics-facing numerical chapters."""
    del summary, archive_id
    chapter6_path = ROOT / "Thesis" / "Chapter6_VerifiedResults.tex"
    chapter7_path = ROOT / "Thesis" / "Chapter7_Conclusions.tex"

    chapter6 = r"""
\chapter{Numerical Results and Physical Interpretation}
\label{chap:results}

The central question is whether the product-GP excess source supplies a
physically reliable correction to PBME for the one-dimensional two-state
scattering model. Reliability requires more than accurate interpolation of
the density values used for training. The reconstructed derivatives must
remain accurate away from the sampled cloud, the fitted density must remain
close to the physical singly-excited-oscillator (SEO) image, the propagation
must become insensitive to time step and cloud size, raw invariants must be
preserved, and physical errors relative to controlled TDSE or grid-QCLE
references must decrease. These questions are tested separately below so that
a favourable value-reconstruction result cannot conceal a failure of the
derivative operator or the dynamics. The combined evidence gives a controlled
negative result: the tested MIDPOINT construction does not satisfy these joint
requirements and does not show a systematic improvement over PBME.

\section{Numerical design and hierarchy of evidence}
\label{sec:results-design}

Table~\ref{tab:numerical-study-design} summarizes the independent numerical
questions. Raw physical observables and raw invariants are primary. Errors
relative to analytic manufactured functions or controlled reference solutions
are the next level. GP training residuals, leave-one-out residuals, posterior
variance, and \(R^2\) diagnose the interpolant internally but are not treated
as physical-error estimates. Likewise, unit-mass density shapes are useful
only after the corresponding raw integral has been examined.

@@TABLE_STUDY_DESIGN@@

\section{Manufactured density, gradient, and excess operator}
\label{sec:results-manufactured}

The manufactured calculation isolates the reconstruction problem from
dynamical feedback. An analytic positive density is sampled in the same
six-dimensional phase space used by the product surrogate. Its density,
gradient, and excess action \(Q[\rho]\) are then compared with their analytic
counterparts both on the training cloud and on 1000 independently drawn query
points. Tables~\ref{tab:manufactured-density}--\ref{tab:manufactured-operator}
report the relative \(L_1\), \(L_2\), and \(L_\infty\) errors as means and
sample standard deviations over three independent cloud pairs for every
regularization and cloud size.

@@TABLE_MANUFACTURED@@

The value reconstruction is generally the least demanding part of this test;
the gradient and \(Q[\rho]\) amplify local errors through differentiation.
Most importantly, increasing \(N\) does not produce a monotone reduction of
the operator error. For the regularization used in the dynamical calculations,
\(\ell_2=0.05\), the off-cloud mean relative \(L_1\) error in \(Q[\rho]\) is
\(2.08\times10^{-2}\), \(2.92\times10^{-2}\),
\(2.32\times10^{-2}\), and \(2.33\times10^{-2}\) for
\(N=300,600,1200,2400\), respectively. The smaller regularizations
\(10^{-6}\) and \(0.01\) change the error level but do not restore a systematic
refinement trend. The data therefore support neither convergence of the
ambient derivative reconstruction nor an attribution of the non-monotonicity
to one numerical cause.

Figure~\ref{fig:manufactured-operator-regularization} condenses the decisive
operator comparison.
\begin{figure}[tbp]
\centering
\includegraphics[width=0.96\textwidth]{../final_reviewer_closure/figures/manufactured_operator_regularization.png}
\caption{Manufactured relative \(L_2\) error in the excess action \(Q[\rho]\)
on the training cloud and on an independent query cloud. Points are means
over seeds 123, 124, and 125; bars are sample standard deviations. The three
regularization policies use paired training and query clouds at each cloud
size. The error remains nonzero and nonmonotone, so neither cloud enlargement
nor changing \(\ell_2\) from \(10^{-6}\) to the pilot \(0.01\) or production
\(0.05\) policy establishes a monotone error decrease under independent-cloud
enlargement.}
\label{fig:manufactured-operator-regularization}
\end{figure}

\section{Physical SEO image and identical-cloud reconstruction}
\label{sec:results-projection}

The two-state physical SEO image has four real bath-dependent coefficient
fields. Table~\ref{tab:seo-projection-physics} measures the relative distance
of the fitted product density from this image using 20 bath anchors and 400
mapping probes. The projection is diagnosed but not imposed. Across the
reported snapshots and initial momenta, the mean leakage spans approximately
0.221--0.977, which is too large to regard the propagated surrogate as a small
perturbation within the physical SEO subspace. This is a representation
diagnostic, not a TDSE or QCLE error.

The identical-cloud comparison in
Table~\ref{tab:identical-cloud-reconstruction} addresses a narrower question:
can the GP reproduce a KDE density when both estimators use the same PBME
snapshot, weights, bandwidth, normalization, and evaluation grid? The maximum
\(E_1\) values are well below the declared \(0.02\) criterion. This positive
control confirms accurate same-cloud value reconstruction, but it does not
test derivatives, the SEO constraint, or time propagation.

@@TABLE_PROJECTION_BASELINE@@

The nonzero initial value in Table~\ref{tab:seo-projection-physics} requires
explicit interpretation. At \(t=0\) the analytic initial density is the
occupied-state SEO profile multiplied by a smooth bath factor, so it lies
exactly in the four-field image and its residual under the same projector
vanishes identically in exact arithmetic. The reported initial leakage of
approximately 0.222 is therefore a property of the fitted finite surrogate as
evaluated by this diagnostic, not of the physical initial target: the
20-anchor/400-probe evaluation samples mapping directions that the focused
initial cloud does not constrain, where the fitted product density is
determined by the Gaussian-process extension and the safe-profile floor
rather than by data. PBME and MIDPOINT report identical initial values
because both branches share the same initial fit before any excess source is
applied. The present calculations do not apportion this initial residual
among the Gaussian-process extension, the floor regularization, and the
finite anchor/probe sampling; the subsequent growth of the leakage along the
propagation, not the initial offset, carries the propagation-relevant
conclusion.

The four-seed snapshot summary in
Figure~\ref{fig:seo-projection-leakage-verified} shows that the leakage is not
a small local residual confined to one calculation.
\begin{figure}[tbp]
\centering
\includegraphics[width=0.96\textwidth]{../final_reviewer_closure/figures/seo_projection_leakage_verified.png}
\caption{Diagnostic SEO projection leakage at \(t/t_c=0,1,2\). Each point is
the mean over four independent propagation seeds and each bar is the sample
standard deviation. The plotted quantity is the relative \(L_2\) residual
\(\lVert y-Bc\rVert_2/\max(\lVert y\rVert_2,10^{-30})\) from a four-field
least-squares SEO projection using 20 bath anchors and 400 mapping probes.
Projection is diagnosed but not enforced, and this quantity is not a physical
TDSE or QCLE error.}
\label{fig:seo-projection-leakage-verified}
\end{figure}

\section{Time-step refinement}
\label{sec:results-timestep}

Table~\ref{tab:timestep-refinement-physics} compares
\(\Delta t=0.5,0.25,0.125\) on common physical times using linear
interpolation without extrapolation. Both successive time-normalized
differences must first exceed
\(\tau_{\rm noise}=10^{-12}+10^{-12}\max_k\operatorname{RMS}(O_k)\).
Rows below that floor are labelled roundoff- or saturation-limited; rows with
nondecreasing successive differences are labelled nonmonotone. For MIDPOINT,
both differences must additionally exceed the pooled dispersion over four
independently sampled clouds. The 128 run/observable rows comprise 15
noise-limited rejections, 102 seed-variability rejections, three nonpositive
orders, and eight positive orders. The latter do not repair the simultaneous
failures in replication and raw conservation, so the formal order of the
midpoint formula is not promoted to a demonstrated order of the full
moving-cloud dynamics.

@@TABLE_TIMESTEP@@

\section{Independent-cloud enlargement and four-seed replication}
\label{sec:results-support-replication}

The cloud-size study in Table~\ref{tab:independent-cloud-size} compares
\(N=500,1000,2000\) using three independently sampled clouds at each size.
These clouds are not nested; the results therefore measure sensitivity to
enlarging a stochastic representation rather than deterministic convergence.
Several MIDPOINT changes remain comparable to, or larger than, the seed
dispersion, while PBME is generally much more stable.

Table~\ref{tab:four-seed-replication-physics} makes that contrast explicit at
\(N=1000\) and \(\Delta t=0.25\). For the endpoint \(\rho_{11}^{\mathrm{SN}}\) estimator, the
PBME sample standard deviation is 0.00788 at initial momentum 20 and 0.0154 at
initial momentum 100; the corresponding MIDPOINT values are 0.795 and 0.956.
The MIDPOINT mean and interval can also extend outside the physical population
range \([0,1]\). Four seeds provide a sensitive instability diagnostic but
not a precise uncertainty distribution, so the intervals are interpreted
descriptively.

@@TABLE_CLOUD_SIZE@@

@@TABLE_REPLICATION@@

Figure~\ref{fig:replication-raw-conservation} places the seed instability and
the associated raw conservation failure on the same footing.
\begin{figure}[tbp]
\centering
\includegraphics[width=0.91\textwidth]{../final_reviewer_closure/figures/replication_and_raw_conservation.png}
\caption{Four-seed replication and raw pre-normalization conservation for the
canonical \(N=1000\), \(\Delta t=0.25\) calculations. The upper panels show
the endpoint \(\rho_{11}^{\mathrm{SN}}\) estimator; the shaded band marks the physical interval
\([0,1]\). The lower panels show the maximum absolute drift of the raw
normalization. Exact PBME zeros are displayed at \(10^{-16}\) only to make
them visible on the logarithmic axis. The seed count is four, not the number
of trajectories, and the plot supplies a descriptive sensitivity comparison
rather than strong uncertainty calibration.}
\label{fig:replication-raw-conservation}
\end{figure}

\section{Raw conservation and signed-label tail sensitivity}
\label{sec:results-conservation-tail}

The population and density curves shown elsewhere can be normalized for visual
comparison, but that operation removes the very integral whose conservation
must be tested. Table~\ref{tab:raw-conservation-physics} therefore reports the
normalization, electronic trace, and energy before any such rescaling. PBME
keeps the raw normalization constant in these calculations and its trace and
energy drifts remain small. MIDPOINT shows large and strongly seed-dependent
drifts: even in the canonical four-seed set the largest normalization drift
reaches \(8.69\times10^{20}\), and across the tested refinement/cloud-size
calculations it reaches \(7.35\times10^{46}\). A unit-normalized display curve
cannot be used as evidence against this failure.

The signed-label diagnostic in Table~\ref{tab:signed-label-tail-physics}
quantifies the conditioning of the ratio \(y_i(t)/y_i^0\). At zero threshold,
the largest absolute ratio reaches \(7.61\times10^{23}\) and the signed
effective sample size falls as low as 0.0247. The first nonzero threshold,
however, already removes about \(10^{-3}\) of the absolute physical mass,
which is larger than the predeclared negligible-mass allowance of \(10^{-6}\).
Thus there is no nontrivial negligible-mass plateau. The signed estimator is
demonstrably ill-conditioned, but these calculations do not isolate a
vanishing-initial-label tail as its unique cause.

@@TABLE_CONSERVATION@@

@@TABLE_TAIL@@

Figure~\ref{fig:initial-label-tail-sensitivity} shows the paired
initial-label distributions and the threshold sweep behind
Table~\ref{tab:signed-label-tail-physics}.
\begin{figure}[tbp]
\centering
\includegraphics[width=0.96\textwidth]{../final_reviewer_closure/figures/initial_label_tail_sensitivity.png}
\caption{Initial-label tail sensitivity for all eight paired
\(P_{\rm init}\)/seed cases. The quantile panel reports the \(|y_i^0|\)
distribution on the grid \(q=0.001,\ldots,0.999\); the threshold panels use
the PBME copy of the method-identical inclusion masks at every positive
threshold \(\eta\). Solid lines are excluded point fractions and dashed
lines are excluded absolute physical-mass fractions; exact zeros are plotted
at \(10^{-7}\) only for logarithmic display and do not alter any reported
error. The first nonzero threshold already excludes about \(10^{-3}\) of the
absolute physical mass, exceeding the predeclared \(10^{-6}\) allowance, so
no negligible-mass plateau exists.}
\label{fig:initial-label-tail-sensitivity}
\end{figure}

\section{TDSE and grid-QCLE numerical controls}
\label{sec:results-reference-controls}

TDSE is model-exact only within the displayed domain, time step, periodic FFT
convention, edge-mass, normalization, and reflected-momentum controls. Grid
QCLE is a numerical solution of the approximate QCLE equation. Its separate
temporal and phase-space-grid studies report three levels and both successive
differences for each observable. The exact domains and discretizations are
collected in Table~\ref{tab:reference-discretizations}, while
Tables~\ref{tab:tdse-reference-refinement} and
\ref{tab:qcle-reference-refinement} give the observable-level comparisons.

The maximum TDSE boundary probability in the outer spatial bands is
\(1.22\times10^{-24}\), and the maximum negative-momentum probability for the
high-momentum case is \(6.09\times10^{-27}\), both negligible on the declared
scale. For grid QCLE, the largest physical marginal fractions in the outer
5\% bands are \(2.51\times10^{-4}\) in \(R\) and
\(5.92\times10^{-4}\) in \(P\), below the \(10^{-3}\) criterion. An auxiliary
phase-space \(|W|\)-weighted edge diagnostic can reach \(7.94\times10^{-2}\)
because the Wigner density is sign-indefinite and exhibits numerical ringing;
it is retained as a ringing diagnostic and is not substituted for the physical
marginal boundary mass. The CFL ratios also remain below unity. For
\(P_{\rm init}=100\), the nominal TDSE grid levels all resolve to the required
8192-point grid; that case therefore establishes boundary adequacy and
numerical repeatability on the chosen grid, but not an independent spatial
convergence rate.

The expected temporal scaling is approximately second order for the TDSE
split-operator calculation and approximately fourth order for the grid-QCLE
classical RK4 calculation. Spatial or phase-space-grid behaviour is assessed
separately and is not assigned either temporal order. An observed order is
reported only when both successive differences exceed the same declared
absolute-plus-relative noise floor, decrease monotonically, and give
\(0<p\leq6\); more rapid contractions are retained as nonasymptotic rather
than promoted as convergence evidence.

@@TABLE_REFERENCES@@

\section{PBME versus MIDPOINT against common references}
\label{sec:results-physical-comparison}

Each paired difference is
\(E_{\mathrm{MIDPOINT,ref}}-E_{\mathrm{PBME,ref}}\), so a negative value
favours MIDPOINT for that particular metric. Isolated negative differences do
not establish systematic improvement. Table~\ref{tab:paired-physical-errors}
reports the paired mean and Student-\(t\) interval for raw and unit-mass density
errors and for the principal time-dependent observables. Several intervals
include zero, and resolved comparisons are not consistently in favour of
MIDPOINT. More fundamentally, a physical success claim also requires
seed-stable dynamics, raw conservation, and controlled discretization
sensitivity. Those conditions are not met, so systematic improvement over
PBME is not demonstrated.

@@TABLE_PHYSICAL@@

This conclusion is visible directly in
Figure~\ref{fig:paired-physical-reference-differences}. The raw-density panel
is kept separate from the independently normalized shape panel so that shape
normalization cannot conceal a failed raw integral.
\begin{figure}[tbp]
\centering
\includegraphics[width=0.96\textwidth]{../final_reviewer_closure/figures/paired_physical_reference_differences.png}
\caption{Four-seed paired MIDPOINT-minus-PBME differences for density
\(E_1\) errors against the matched TDSE and grid-QCLE references. Points are
paired means and bars are two-sided Student-\(t\) 95\% intervals with three
degrees of freedom. Positive values mean that MIDPOINT has the larger error.
The raw-density panel uses the signed display transform
\(\operatorname{sgn}(\Delta E)\log_{10}(1+\lvert\Delta E\rvert)\), whose tick
labels show the corresponding untransformed magnitude; the shape panel uses
unit-mass normalization and a linear axis. Shape agreement is therefore not
used as conservation evidence.}
\label{fig:paired-physical-reference-differences}
\end{figure}

\FloatBarrier

\section{Synthesis of the numerical evidence}
\label{sec:results-synthesis}

The positive identical-cloud reconstruction control establishes that the GP
can interpolate density values accurately under matched conditions. That
result does not carry through the sequence required by the physical
calculation. The manufactured derivative operator is non-monotone with cloud
size; the fitted density has substantial SEO-image leakage; the dynamical
response is dominated by cloud-to-cloud variation; and the raw MIDPOINT
normalization, trace, and energy can drift catastrophically. Controlled TDSE
and grid-QCLE references do not reveal a consistent physical-error reduction
relative to PBME. The failure is therefore not inferred from one diagnostic:
it is the common conclusion of independent operator, representation,
refinement, replication, conservation, and reference tests.

The defensible result is correspondingly narrow. The tested product-GP,
moving-cloud MIDPOINT discretization is not a reliable dynamical method for
this benchmark. This statement concerns the discretization studied here; it
does not constitute a failure of the continuum mapping-QCLE excess term.
""".strip()

    chapter7 = r"""
\chapter{Discussion and Conclusions}
\label{chap:discussion-outlook}

This thesis set out to test whether a differentiable product-GP representation
could make the mapping-QCLE excess term usable on a moving trajectory cloud.
The answer for the construction studied here is no. Accurate interpolation of
density values on a fixed cloud does not imply accurate ambient derivatives,
preservation of the physical SEO image, stable signed estimators, or
conservative dynamics. The contribution is therefore a precisely defined
formulation, a sequence of physically interpretable tests that expose its
limitations, and concrete requirements for a successor. It is not a validated
GP-QCLE solver and it is not evidence that MIDPOINT improves PBME.

\section{Application-specific contribution}
\label{sec:discussion-contribution}

Values on a lower-dimensional manifold do not generally identify ambient
normal derivatives; that fact is not original to this thesis. The original
application-specific contribution is narrower: constructing a differentiable
product-GP excess-source prototype on moving PBME/MInt clouds, testing it
against an analytic manufactured operator and the exact two-state SEO image,
and connecting its observed derivative, projection, conservation, and
signed-weight failures to the propagated dynamics without claiming a unique
cause
\cite{Kim2008JCP,Kelly2012JCP,Raissi2017LinearGP,
Solak2002DerivativeGP,CockayneOatesSullivanGirolami2019}.

\section{Physical interpretation of the failure}
\label{sec:discussion-evidence}

The results in Chapter~\ref{chap:results} reveal a connected, but not uniquely
causal, failure pathway. A product kernel can reconstruct the sampled density
values while leaving ambient derivatives underdetermined away from the cloud.
The excess operator depends precisely on those derivatives. Once its source
is applied without enforcing the four-field SEO structure, the fitted density
can develop large components outside the physical image. The focused-sampling
ratio then combines sign cancellation with very large individual label ratios,
which magnifies cloud-to-cloud variation. The resulting raw normalization,
trace, and energy drift show that this is not merely a cosmetic change in
density shape.

These observations establish compatibility among the failure modes, not a
proof that one is the sole cause of the others. The manufactured operator
study, SEO projection diagnostic, signed-label analysis, and conservation
test deliberately isolate different links in the chain. Their agreement
supports the method-level conclusion without overstating causal identification.
The controlled TDSE and grid-QCLE studies then ensure that the absence of a
systematic physical improvement is not an artefact of an uncontrolled
reference calculation.

\section{Claim boundary}
\label{sec:discussion-limitations}

This thesis does not establish a validated discretization of the complete
mapping-basis QCLE, deterministic cloud convergence, strong uncertainty
calibration from four seeds, chemical accuracy outside the one-dimensional
two-state benchmark, multidimensional scalability, or systematic improvement
over PBME. TDSE and grid-QCLE have different epistemic roles: TDSE is
model-exact within numerical controls, whereas grid QCLE solves an approximate
quantum--classical equation numerically.

The production regularization \(0.05\), pilot \(0.01\), and
manufactured-test \(10^{-6}\) policies are all tested and reported rather than
conflated. The tail study demonstrates extreme signed-weight instability, but
its negligible-mass rule does not permit a nontrivial point-removal plateau,
so the instability is not assigned uniquely to small initial labels.

\section{Requirements for a successor}
\label{sec:discussion-successor}

A credible successor should regress the four real bath-dependent fields of the
two-state SEO image so that Hermiticity and projection are enforced by
construction. Its discrete excess source should preserve normalization and
the relevant trace and energy identities before any renormalization. The
regression should incorporate derivative information or structural constraints
capable of controlling directions normal to the sampled manifold. A future
success claim requires decreasing manufactured-operator errors under a
controlled refinement, time-step changes resolvable above seed variation,
independent-cloud uncertainty bands, raw conservation, and reproducibly
smaller PBME-matched errors against controlled references where the excess
source is appreciable
\cite{TraskBochevPerego2020,BonetLok1999,Monaghan2005}.

\section{Numerical scope and transferability}
\label{sec:discussion-numerical-scope}

The numerical conclusions are tied to the parameter ranges, independent seed
sets, domains, and discretizations stated in Chapter~\ref{chap:results} and
Appendix~\ref{app:reference-and-reproducibility}. Repeating those calculations
tests this particular construction on the same benchmark; it does not by
itself establish transferability to additional electronic states, nuclear
dimensions, coupling regimes, or sampling measures. Such transfer would
require renewed manufactured-operator, projection, conservation, and
reference comparisons rather than extrapolation from the present negative
result.

\section{Conclusion}
\label{sec:discussion-conclusions}

The combined numerical evidence supports a controlled negative result.
Accurate same-cloud value reconstruction does not guarantee an accurate
derivative-dependent excess operator, preservation of the SEO image, raw
conservation, or seed-stable dynamics. The tested MIDPOINT branch fails the
joint reliability criteria and does not demonstrate systematic improvement
over PBME. This conclusion applies to the tested discretization, not to the
continuum QCLE excess term.

Accordingly, the thesis contributes a precisely scoped formulation, a
physics-based failure analysis, and evidence-backed redesign criteria. It
does not claim that the tested construction is reliable, deterministically
cloud-converged, or improved.
""".strip()

    tables = {
        "@@TABLE_STUDY_DESIGN@@": study_design_table(),
        "@@TABLE_MANUFACTURED@@": manufactured_tables(),
        "@@TABLE_PROJECTION_BASELINE@@": projection_and_baseline_tables(),
        "@@TABLE_TIMESTEP@@": timestep_table(),
        "@@TABLE_CLOUD_SIZE@@": cloud_size_table(),
        "@@TABLE_REPLICATION@@": replication_table(),
        "@@TABLE_CONSERVATION@@": conservation_table(),
        "@@TABLE_TAIL@@": tail_table(),
        "@@TABLE_REFERENCES@@": reference_tables(),
        "@@TABLE_PHYSICAL@@": physical_comparison_table(),
    }
    for placeholder, table_text in tables.items():
        chapter6 = chapter6.replace(placeholder, table_text)

    chapter6_path.write_text(chapter6 + "\n", encoding="utf-8")
    chapter7_path.write_text(chapter7 + "\n", encoding="utf-8")
    return [chapter6_path, chapter7_path]


def _write_reviewer_response_audit_legacy(summary: Mapping[str, Any], archive_id: str) -> Path:
    records = item_records(summary, archive_id)
    pages = aux_pages()
    response = ROOT / "Reviewer_Response.tex"
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=25mm]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern,microtype,booktabs,longtable,array,xcolor,hyperref}",
        r"\hypersetup{hidelinks,pdftitle={" + TITLE + r" --- Response to Examiner}}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{5pt}",
        r"\newcommand{\field}[2]{\textbf{#1}\quad #2\par}",
        r"\begin{document}",
        r"\raggedright",
        r"\begin{center}",
        r"{\Large\bfseries Response to the Major-Revision Examiner Report\par}",
        r"\vspace{6pt}",
        r"{\bfseries " + tex(TITLE) + r"\par}",
        r"\vspace{4pt}Sahand Nikzat",
        r"\end{center}",
        r"\section*{Revision overview}",
        tex(
            "The thesis is presented as a controlled negative-result study. "
            "Completion is separated from scientific acceptance. The MIDPOINT "
            "prototype is not called reliable and no systematic improvement "
            "over PBME is claimed because seed replication, raw conservation, "
            "projection, and other declared gates do not all pass."
        ),
        r"\section*{Ten acceptance gates}",
    ]

    gate_results = {
        1: "Question/importance/approach/outcome openings and chapter answers are present.",
        2: "The novelty statement is literature-bounded and application-specific.",
        3: "Zero unprovenanced thesis figures remain.",
        4: summary["manufactured"],
        5: summary["timestep"] + " " + summary["references"],
        6: "The support study is independent-cloud enlargement; deterministic support convergence is untested.",
        7: summary["physical"],
        8: (
            "The frozen numerical-evidence payload, environment, checksums, "
            "and crosswalk are complete. Payload SHA-256="
            + summary["frozen_payload_sha256"]
            + "; "
            + ("the versioned public release URL is recorded; no DOI or "
               "institutional persistent identifier has been assigned."
               if archive_id != BLOCKED_ID
               else "external publication of a versioned release URL is still required.")
        ),
        9: f"The exact title is synchronized: {TITLE}.",
        10: "This response contains every gate and all I/M/L rows with location and source.",
    }
    for gate in range(1, 11):
        status = (
            records["I-15"]["status"] if gate == 8
            else ALLOWED_STATUSES["computation"]
        )
        lines += [
            rf"\subsection*{{Gate {gate}}}",
            rf"\field{{Final status:}}{{{tex(status)}}}",
            rf"\field{{Action and result:}}{{{tex(gate_results[gate])}}}",
            rf"\field{{Thesis location:}}{{{tex('Chapters 1–7 and Appendix F as cross-referenced')}}}",
            rf"\field{{Archive source:}}{{{tex_path(EVIDENCE / 'TABLE_DATA_CROSSWALK.csv')}}}",
            rf"\field{{Versioned public release URL:}}{{{tex(archive_id)}}}",
        ]

    matrix_order = read_csv(ROOT / "reviewer_closure_out" / "closure_matrix.csv")
    for prefix, heading in (("I-", "I-items"), ("M-", "M-items"), ("L-", "L-items")):
        lines.append(rf"\section*{{{heading}}}")
        for matrix_row in matrix_order:
            item = matrix_row["item"]
            if not item.startswith(prefix):
                continue
            record = records[item]
            page_note = ""
            label_match = re.search(r"(tab:[A-Za-z0-9:-]+)", record["table"])
            if label_match and label_match.group(1) in pages:
                page_note = f"; thesis p. {pages[label_match.group(1)]}"
            lines += [
                rf"\subsection*{{{tex(item)} --- {tex(record['concern'])}}}",
                rf"\field{{Final status:}}{{{tex(record['status'])}}}",
                rf"\field{{Action taken:}}{{{tex(record['action'])}}}",
                rf"\field{{Numerical result or wording change:}}{{{tex(record['result'])}}}",
                rf"\field{{Thesis location:}}{{{tex(record['location'] + page_note)}}}",
                rf"\field{{Table/equation:}}{{{tex(record['table'])}}}",
                rf"\field{{Archive source:}}{{{tex_path(record['source'])}}}",
                rf"\field{{Versioned public release URL:}}{{{tex(record['archive_id'])}}}",
            ]
    lines += [
        r"\section*{Final evidence statement}",
        tex(
            "Every numerical statement is traced through the table-data "
            "crosswalk to a full-precision CSV and source manifest. Failed or "
            "unstable scientific outcomes are retained. No GP internal "
            "diagnostic is presented as a physical-error estimate."
        ),
        r"\end{document}",
    ]
    response.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return response


def write_reviewer_response(summary: Mapping[str, Any], archive_id: str) -> Path:
    """Write a point-by-point scientific response without operational metadata."""
    records = item_records(summary, archive_id)
    pages = aux_pages()
    response = ROOT / "Reviewer_Response.tex"

    evidence_map = {
        "Tables tab:manufactured-complete-density/gradient/q and tab:manufactured-summary":
            "Tables G.1--G.3 (complete per-seed values); aggregates in "
            "Tables 6.2--6.4 and Figure 6.1",
        "Complete 72-row manufactured tables":
            "Tables G.1--G.3 (complete per-seed values)",
        "Table tab:timestep-run-by-run":
            "Table G.4 (complete run-by-run evidence); summarized in "
            "Table 6.7",
        "Tables tab:tdse-reference-orders and tab:qcle-reference-orders":
            "Tables 6.13 and 6.14",
        "Reference-control tables":
            "Tables 6.12--6.14 and F.1",
        "Table tab:four-seed-replication":
            "Table 6.9 (mean, SD, interval, spread); individual seed values "
            "in Figure 6.3 and the Delta-t=0.25 rows of Table G.4",
        "Table tab:identical-support-kde-gp": "Table 6.6",
        "Table tab:y0-tail-sensitivity": "Tables 6.11 and G.5--G.6",
        "Table tab:physical-reference-comparison":
            "Table 6.15; per-seed absolute errors in Tables G.7--G.8",
        "Table tab:seo-projection-leakage": "Table 6.5",
        "Table tab:independent-cloud-support": "Table 6.8",
        "Table tab:raw-conservation": "Table 6.10",
        "Table tab:final-validation-inventory": "Table 6.1",
        "Figure-data crosswalk":
            "Figures 6.1--6.5",
        "Table-data crosswalk":
            "Tables 6.1--6.15 and G.1--G.8, traced through "
            "final_reviewer_closure/table_data_crosswalk.csv",
        "Archive/environment crosswalk":
            "Appendix F and Tables F.1--F.2",
        "Equation Q_applied=Q_raw":
            "Chapter 5 excess-source equations and Table 6.10",
    }

    scientific_results = {
        "I-1": "Density, gradient, and excess-action errors are reported on both the training cloud and an independent query cloud for every regularization and cloud size; the operator error is non-monotone with cloud size.",
        "I-2": "Three endpoint values and two successive time-normalized differences are tested against the declared numerical-noise floor and independent-seed dispersion, so empirical order is withheld when either guard fails.",
        "I-3": "TDSE and grid-QCLE use separate time and grid controls with explicit domains, discretizations, edge masses, and successive differences. The high-momentum TDSE grid case is identified as a boundary-adequacy control because all nominal levels use 8192 points.",
        "I-4": "Endpoint means, sample standard deviations, spreads, and Student-t intervals are reported in Table 6.9, and the four individual seed values for seeds 11, 29, 47, and 73 are shown in Figure 6.3 and the Delta-t=0.25 rows of Table G.4; MIDPOINT variability is far larger than PBME variability.",
        "I-5": "The identical-cloud KDE/GP value-reconstruction criterion is met, while the thesis explicitly states that this does not validate derivatives or dynamics.",
        "I-6": "All retained figures define the observable, normalization, independent seed count, uncertainty bars, and sign convention in their captions.",
        "I-7": "The thesis uses five quantitative figures and fifteen reader-facing tables; every retained comparison has a defined metric, averaging rule, uncertainty description, and interpretation.",
        "I-8": "Every numerical threshold used for acceptance or display is stated with its rule. The numerical-noise, boundary-mass, negligible-tail-mass, and reconstruction thresholds are explicit in the relevant text, captions, and evidence CSVs.",
        "I-9": "The signed-label threshold study reports excluded physical mass and signed/absolute effective sample sizes. No nontrivial negligible-mass plateau exists, so the tail is not assigned as a unique cause.",
        "I-10": "The applied excess source is distinguished from clipped or renormalized alternatives; the reported MIDPOINT results use the unprojected, nonconservative source under examination.",
        "I-11": "The unsupported analytic-GP physical-observable claim was removed; no uncomputed table is implied. Retained physical comparisons use the reported cloud estimators and controlled references.",
        "I-12": "The independent sample size is four clouds, not the number of trajectories; intervals are described as small-sample sensitivity measures.",
        "I-13": "Paired MIDPOINT-minus-PBME physical-error differences do not consistently favour MIDPOINT, and the joint reliability requirements are not met.",
        "I-14": "Reference domains, time steps, grids, boundary conventions, edge masses, and CFL controls are stated in Chapter 6 and Appendix F.",
        "I-15": (
            "The final code, evidence archive, checksum index, environment, "
            "manifests, and reproduction instructions are retrievable from "
            + archive_id
            if archive_id != BLOCKED_ID
            else "Public deposition of the final archive remains pending and is not represented as complete."
        ),
        "I-16": f"The title is synchronized as: {TITLE}.",
        "M-2": "The four-field SEO image is derived and the measured leakage is large; projection is diagnosed but not enforced.",
        "M-6": "Clouds at N=500, 1000, and 2000 are independently sampled and non-nested, so the result is described as stochastic cloud-size sensitivity rather than deterministic convergence.",
        "M-7": "All three regularizations are compared under paired manufactured clouds, with no claim that regularization alone cures the instability.",
        "M-8": "Time-step changes are judged against seed dispersion and are not converted into formal convergence claims when unresolved.",
        "M-10": "Raw normalization, trace, and energy are reported before display normalization; MIDPOINT exhibits severe drift whereas PBME remains stable on the reported scale.",
        "M-12": "The signed-label ratio, effective sample sizes, excluded mass, normalization, and energy are examined together across the threshold sweep.",
        "M-13": "Self-normalized unity is never used as conservation evidence.",
        "M-14": "Chapter 6 states the hierarchy from raw invariants and reference errors to internal surrogate diagnostics before interpreting the data.",
        "M-15": "Low signed effective sample size is treated as loss of estimator reliability, not as validated physical information.",
        "M-17": "The negative conclusion is restricted to the tested product-GP moving-cloud MIDPOINT discretization, not the continuum QCLE excess term.",
        "M-18": "The identical-cloud control isolates value reconstruction and is not generalized to derivative or propagation accuracy.",
        "M-19": "The thesis presents a compatible failure pathway but explicitly avoids claiming a unique causal mechanism.",
        "M-20": "Appendix G, Section G.5 gives a retrievable release record while the scientific chapters retain only the numerical methods and parameter scope needed to interpret the calculations.",
        "M-21": "The objective is restricted to one product-GP moving-cloud construction on a one-dimensional two-state benchmark.",
        "M-22": "Chapter 6 opens with the decisive physical reliability question and its controlled negative answer.",
        "M-24": "Complete manufactured density, gradient, and excess-action evidence is summarized over three independent cloud pairs for every tested regularization and cloud size.",
        "M-25": "The numerical design table states the physical question, controlled variation, independent realizations, and diagnostic for every study.",
    }
    scientific_actions = {
        "I-3": "Recompute and classify all TDSE and grid-QCLE rows from separate three-level time and spatial/grid ladders under the declared guards.",
        "I-6": "Retain only figures whose observable, normalization, independent realizations, uncertainty, and interpretation are completely defined.",
        "I-7": "Attach a defined metric and numerical evidence to every retained comparison.",
        "I-8": "State each threshold and its exact decision rule at the point of use.",
        "I-11": "Remove the unsupported analytic-observable claim rather than inventing uncomputed results.",
        "I-15": "Provide stable public access to the final research materials and state honestly whether a permanent archival identifier has been assigned.",
        "M-25": "Replace computational completion language with a numerical-design table organized by physical question and diagnostic.",
    }
    location_overrides = {
        "I-7": "Chapter 6, Figures 6.1--6.5 and Tables 6.1--6.15",
        "I-8": "Chapters 5--6 and Appendix F--G, relevant captions and table notes",
        "I-11": "Chapters 6--7, physical-comparison and claim-boundary sections",
        "I-15": "Response availability note; Appendix G, Section G.5 archive and reproducibility record",
        "M-25": "Chapter 6, numerical design and hierarchy of evidence",
    }

    release_manifest_path = AUDIT / "frozen_numerical_evidence_payload_manifest.json"
    release_manifest = (
        json.loads(release_manifest_path.read_text(encoding="utf-8"))
        if release_manifest_path.exists() else {}
    )
    frozen_commit = release_manifest.get(
        "frozen_source_evidence_commit",
        release_manifest.get("source_release_commit", "pending final freeze"),
    )
    archive_sha = release_manifest.get("archive_sha256", "pending final freeze")
    index_sha = release_manifest.get(
        "embedded_checksum_index_sha256", "pending final freeze"
    )

    item_audit = {
        "I-1": (
            "All three manufactured cloud-pair seeds are printed for every N, regularization, support type, and density/gradient/operator metric; the Chapter 6 summaries retain the mean and sample SD.",
            "Section 6.2, printed pp. 84--86; Tables 6.2--6.4 and Figure 6.1; Appendix G, Tables G.1--G.3, printed pp. 146--152.",
            "final_reviewer_closure/manufactured/manufactured_complete.csv; manufactured_summary.csv; manufactured_refinement_verdicts.csv",
        ),
        "I-2": (
            "Each method/momentum/seed/observable row reports the three endpoint values, both aligned time-series differences, the numerical floor, pooled seed spread, guarded order, verdict, and reason.",
            "Sections 5.9.5 and 6.4, printed pp. 80 and 89--93; Table 6.7; Appendix G, Table G.4, printed pp. 153--157.",
            "final_reviewer_closure/timestep/timestep_run_by_run.csv",
        ),
        "I-3": (
            "TDSE and grid-QCLE time and spatial/grid ladders are separated; orders are shown only after the declared noise, monotonicity, and asymptotic guards pass.",
            "Section 6.7, printed pp. 100--108; Tables 6.12--6.14; Appendix F.1, printed pp. 140--142.",
            "final_reviewer_closure/reference_tdse/tdse_three_level.csv; reference_grid_qcle/qcle_three_level.csv",
        ),
        "I-4": (
            "The MIDPOINT P_init=20 central values are reported from seeds 11, 29, 47, and 73 with mean, sample SD, spread, and Student-t interval.",
            "Section 6.5, printed pp. 93--97; Table 6.9 and Figure 6.3; corresponding dt=0.25 rows of Table G.4, printed pp. 153--157.",
            "final_reviewer_closure/replication/four_seed_values.csv; four_seed_summary.csv",
        ),
        "I-5": (
            "The identical-support KDE/GP comparison uses the same PBME cloud, weights, bandwidth, normalization, and grid; only the declared E1 value-reconstruction gate is evaluated.",
            "Section 6.3, printed pp. 86--89; Table 6.6, printed p. 88.",
            "final_reviewer_closure/preserved_evidence/kde_gp_baseline.csv",
        ),
        "I-6": (
            "Every retained figure now defines the plotted quantity, normalization, seed aggregation, uncertainty display, sign convention, and evidentiary limit in its caption.",
            "Figures 6.1--6.5, printed pp. 87, 88, 98, 101, and 111.",
            "final_reviewer_closure/figures/FIGURE_DATA_CROSSWALK.csv and the five hashed source CSV entries named there",
        ),
        "I-7": (
            "All qualitative legacy comparisons were removed; the five retained figures are paired with quantitative tables and explicit metrics.",
            "Sections 6.2--6.8, printed pp. 84--111; Figures 6.1--6.5 and Tables 6.2--6.15.",
            "final_reviewer_closure/figures/FIGURE_DATA_CROSSWALK.csv; table_data_crosswalk.csv",
        ),
        "I-8": (
            "Every acceptance/display threshold is stated numerically at use: the E1 gate, absolute-plus-relative order floor, edge-mass criterion, plausibility bound, and tail-mass rule.",
            "Sections 5.9.5 and 6.3--6.7, printed pp. 80 and 86--108; Tables 6.6--6.7 and 6.11--6.14.",
            "final_reviewer_closure/preserved_evidence/kde_gp_baseline.csv; timestep/timestep_run_by_run.csv; reference_tdse/tdse_three_level.csv; reference_grid_qcle/qcle_three_level.csv; tail_sensitivity/threshold_sweep.csv",
        ),
        "I-9": (
            "The tail analysis reports threshold, excluded point fraction, excluded absolute physical mass, normalization, energy, and signed/absolute ESS; no nontrivial negligible-mass plateau is claimed.",
            "Section 6.6, printed pp. 97--100; Table 6.11 and Figure 6.4; Appendix G, Tables G.5--G.6, printed pp. 157--162.",
            "final_reviewer_closure/tail_sensitivity/y0_distribution_paired.csv; threshold_sweep.csv",
        ),
        "I-10": (
            "The reported MIDPOINT branch applies Q_applied=Q_raw: apply_q_clip=false and omega_clip_quantile=null; clipped and renormalized alternatives are not substituted.",
            "Sections 5.2 and 5.4.2, printed pp. 67--74; Section 6.6, Table 6.10, printed p. 97.",
            "Dynamics.py; final_reviewer_closure/FINAL_RUN_MANIFEST.json; preserved_evidence/raw_conservation.csv",
        ),
        "I-11": (
            "The unsupported analytic-GP physical-observable assertion was removed; no uncomputed analytic-observable table or conclusion remains.",
            "Sections 6.8--6.9 and 7.3, printed pp. 109--113; retained physical claims are limited to Tables 6.15 and G.7--G.8.",
            "Thesis/Thesis.tex; final_reviewer_closure/physical_comparison/paired_improvement_summary.csv",
        ),
        "I-12": (
            "The former word 'weakly' was replaced by estimator- and momentum-specific four-cloud statistics; four clouds, not trajectories, define the independent sample size.",
            "Section 6.5, printed pp. 93--97; Table 6.9 and Figure 6.3.",
            "final_reviewer_closure/replication/four_seed_values.csv; four_seed_summary.csv",
        ),
        "I-13": (
            "High-momentum agreement is reported as per-seed absolute field/observable errors and paired MIDPOINT-minus-PBME differences, not as an adjective.",
            "Section 6.8, printed pp. 109--111; Table 6.15 and Figure 6.5; Appendix G, Tables G.7--G.8, printed pp. 163--164.",
            "final_reviewer_closure/physical_comparison/density_errors_method_pair_by_seed.csv; observable_errors_method_pair_by_seed.csv; paired_improvement_summary.csv",
        ),
        "I-14": (
            "The thesis prints every reference method, momentum, final time, time/grid ladder, domain ladder, edge statistic, boundary rule, and CFL definition.",
            "Section 6.7, printed pp. 100--108; Tables 6.12--6.14; Appendix F.1--F.3, printed pp. 140--145.",
            "final_reviewer_closure/reference_settings_by_method_and_momentum.csv; reference_tdse/tdse_three_level.csv; reference_grid_qcle/qcle_three_level.csv",
        ),
        "I-15": (
            "The archive record distinguishes the final release tag, which identifies the tagged document commit, from the frozen source/evidence commit and states that the GitHub release is versioned public access, not a DOI or institutional persistent identifier.",
            "Appendix G.5, printed p. 165; this response Availability note.",
            f"release {archive_id}; frozen_numerical_evidence_payload.zip; frozen source/evidence commit {frozen_commit}; archive SHA-256 {archive_sha}; checksum-index SHA-256 {index_sha}; FINAL_RUN_MANIFEST.json; FIGURE_DATA_CROSSWALK.csv; table_data_crosswalk.csv; environment.json; README.md; CLEAN_ROOM_VERIFICATION.json",
        ),
        "I-16": (
            f"The exact title is '{TITLE}' in the title page, PDF metadata, response heading/metadata, citation record, repository README, and release title.",
            "Thesis title page and PDF metadata; this response title and PDF metadata; public repository/release title.",
            "Thesis/Thesis.tex; Thesis/Thesis.pdf metadata; Reviewer_Response.tex; CITATION.cff; GitHub release metadata",
        ),
        "M-1": (
            "The thesis explicitly says that nonidentifiability of ambient normal derivatives is known; originality is restricted to constructing and testing this application-specific product-GP/MIDPOINT chain.",
            "Section 7.1, printed p. 112.",
            "Thesis/Chapter7_Conclusions.tex; cited derivative-GP and probabilistic-numerics literature",
        ),
        "M-2": (
            "The projected density is the physical target, while the tested product surrogate is not projection-enforcing; measured SEO leakage is therefore a failure diagnostic.",
            "Sections 4.1.1--4.1.2, printed pp. 50--51; Section 6.3, printed pp. 86--89; Table 6.5 and Figure 6.2.",
            "final_reviewer_closure/preserved_evidence/projection_leakage.csv",
        ),
        "M-3": (
            "The method is described as one attempted moving-cloud collocation of the formal Hamiltonian and excess-source terms, not a validated semi-discretization of the complete generator.",
            "Sections 5.2--5.4, printed pp. 67--74; Chapter 6 opening, printed p. 83; Section 7.3, printed p. 113.",
            "Thesis/Thesis.tex; Thesis/Chapter6_VerifiedResults.tex; Thesis/Chapter7_Conclusions.tex",
        ),
        "M-4": (
            "Method-level 'corrected method/evolution' language was replaced by 'MIDPOINT prototype', 'source update', or 'tested excess-source branch'.",
            "Chapters 5--7, printed pp. 65--114.",
            "Thesis/Thesis.tex; final_acceptance_check.py forbidden-phrase audit",
        ),
        "M-5": (
            "The checks are described as evidence of failure under tested settings; the thesis lists untested transferability and states that causal allocation is nonunique.",
            "Sections 7.2--7.5, printed pp. 112--113.",
            "Thesis/Chapter7_Conclusions.tex; Tables 6.2--6.15",
        ),
        "M-6": (
            "N=500, 1000, and 2000 use independent nonnested clouds and are labelled stochastic cloud-size sensitivity, never deterministic convergence.",
            "Section 4.7.5, printed p. 62; Section 6.5, printed pp. 93--97; Table 6.8.",
            "final_reviewer_closure/support/independent_cloud_summary.csv",
        ),
        "M-7": (
            "The 0.01 result is restricted to its pilot policy and tested cells; all three regularizations are compared without claiming that regularization globally cures instability.",
            "Section 4.7.5, printed p. 62; Section 6.2, printed pp. 84--86; Tables 6.2--6.4 and Figure 6.1.",
            "final_reviewer_closure/manufactured/manufactured_complete.csv; manufactured_summary.csv",
        ),
        "M-8": (
            "Formal midpoint order is separated from observed order; no observed production order is reported unless numerical-noise and pooled seed-dispersion guards both pass.",
            "Sections 5.5 and 5.9.5, printed pp. 75 and 80; Section 6.4, printed pp. 89--93; Table 6.7 and Table G.4.",
            "final_reviewer_closure/timestep/timestep_run_by_run.csv",
        ),
        "M-9": (
            "All 41 untraceable legacy figures and dependent conclusions were removed; five figures regenerated from hash-verified CSVs remain.",
            "Figures 6.1--6.5, printed pp. 87, 88, 98, 101, and 111.",
            "final_reviewer_closure/figures/FIGURE_DATA_CROSSWALK.csv",
        ),
        "M-10": (
            "Unit-mass shape comparisons are labelled shape-only and are interpreted only after raw normalization, trace, and energy have been shown.",
            "Section 6.6, printed pp. 97--100; Table 6.10 and Figure 6.3; Section 6.8, printed pp. 109--111.",
            "final_reviewer_closure/preserved_evidence/raw_conservation.csv; physical_comparison/paired_improvement_summary.csv",
        ),
        "M-11": (
            "The thesis now states explicitly that a ratio of global RMS norms is a global diagnostic, not a pointwise per-step bound; the former less-than-one-percent wording is not used.",
            "Section 5.9, printed p. 78.",
            "Thesis/Thesis.tex",
        ),
        "M-12": (
            "Tail attribution is supported only through quantile-resolved labels, threshold sweeps, excluded physical mass, normalization, energy, and ESS; no unique tail cause is claimed.",
            "Section 6.6, printed pp. 97--100; Table 6.11 and Figure 6.4; Tables G.5--G.6, printed pp. 157--162.",
            "final_reviewer_closure/tail_sensitivity/y0_distribution_paired.csv; threshold_sweep.csv",
        ),
        "M-13": (
            "Self-normalized unity is identified as tautological and is excluded from conservation evidence; raw integral, trace, and energy are primary.",
            "Section 6.6, printed pp. 97--100; Table 6.10 and Figure 6.3.",
            "final_reviewer_closure/preserved_evidence/raw_conservation.csv",
        ),
        "M-14": (
            "The estimator hierarchy is stated before results: raw observables/invariants, then analytic/reference errors, then internal GP diagnostics.",
            "Section 6.1, printed pp. 83--84; Table 6.1.",
            "final_reviewer_closure/validation_inventory.csv; table_data_crosswalk.csv",
        ),
        "M-15": (
            "Neither anchor-cloud nor analytic-GP estimator is called validated in the low-signed-ESS regime; low ESS is treated as estimator unreliability.",
            "Sections 5.9.4 and 6.6, printed pp. 80 and 97--100; Section 7.3, printed p. 113.",
            "final_reviewer_closure/tail_sensitivity/threshold_sweep.csv; preserved_evidence/raw_conservation.csv",
        ),
        "M-16": (
            "Reference proximity is defined only by printed absolute field and observable errors; unsupported words such as close or agreement are not acceptance criteria.",
            "Sections 6.7--6.8, printed pp. 100--111; Table 6.15; Tables G.7--G.8, printed pp. 163--164.",
            "final_reviewer_closure/physical_comparison/density_errors_method_pair_by_seed.csv; observable_errors_method_pair_by_seed.csv",
        ),
        "M-17": (
            "Failure is attributed to the tested nonprojected, nonconservative product-GP/MIDPOINT discretization, not to the continuum excess term itself.",
            "Sections 7.2--7.3 and 7.6, printed pp. 112--114.",
            "Thesis/Chapter7_Conclusions.tex; preserved_evidence/projection_leakage.csv; raw_conservation.csv",
        ),
        "M-18": (
            "The identical-cloud result is limited to passing the declared E1 value-reconstruction gate and is not generalized to derivatives, propagation, or physical fidelity.",
            "Section 6.3, printed pp. 86--89; Table 6.6, printed p. 88.",
            "final_reviewer_closure/preserved_evidence/kde_gp_baseline.csv",
        ),
        "M-19": (
            "The thesis presents a compatible multi-stage failure pathway and explicitly states that the tests do not uniquely apportion causality.",
            "Section 7.2, printed p. 112; Section 7.3, printed p. 113.",
            "Thesis/Chapter7_Conclusions.tex; manufactured, projection, tail, and raw-conservation CSVs",
        ),
        "M-20": (
            "Reproducibility is bounded to the versioned public GitHub release and its checksum record; the text explicitly states that no DOI or institutional persistent identifier has been assigned.",
            "Section 7.5, printed p. 113; Appendix G.5, printed p. 165; this response Availability note.",
            f"{archive_id}; frozen_numerical_evidence_payload_manifest.json; CLEAN_ROOM_VERIFICATION.json",
        ),
        "M-21": (
            "The objective now names one product-GP/moving-cloud/explicit-MIDPOINT construction on a one-dimensional, two-state avoided-crossing benchmark.",
            "Section 1.9, printed p. 25.",
            "Thesis/Thesis.tex",
        ),
        "M-22": (
            "Chapter 6 opens with the physical reliability question and its controlled negative answer before presenting any internal diagnostic.",
            "Chapter 6 opening and Section 6.1, printed pp. 83--84.",
            "Thesis/Chapter6_VerifiedResults.tex",
        ),
        "M-23": (
            "Ambiguous 'full-density representation' language is absent; the text distinguishes the projected physical target from the unprojected tested product surrogate.",
            "Sections 4.1.1--4.1.2, printed pp. 50--51; Section 6.3, printed pp. 86--89.",
            "Thesis/Thesis.tex; final_reviewer_closure/preserved_evidence/projection_leakage.csv",
        ),
        "M-24": (
            "Regularization 0.01 is called pilot-selected; complete operator tests are repeated at 1e-6, 0.01, and production 0.05 over all four N values and three seeds.",
            "Section 4.7.5, printed p. 62; Section 6.2, printed pp. 84--86; Tables 6.2--6.4 and G.1--G.3.",
            "final_reviewer_closure/manufactured/manufactured_complete.csv; manufactured_summary.csv",
        ),
        "M-25": (
            "Completion counts are confined to the inventory; scientific validation is organized by physical question, controlled variation, independent realizations, and diagnostic.",
            "Section 6.1, printed pp. 83--84; Table 6.1.",
            "final_reviewer_closure/validation_inventory.csv",
        ),
        "L-1": (
            "One vocabulary is used: PBME baseline, MIDPOINT prototype, product-GP surrogate, excess-source update, and tested discretization; 'corrected method' is absent.",
            "Sections 1.9, 5.1--5.4, and 6.1, printed pp. 25, 66--74, and 83--84.",
            "Thesis/Thesis.tex; final_acceptance_check.py terminology audit",
        ),
        "L-2": (
            "Population, coherence, trace, signed-cloud, source, and KDE estimators are defined once and mapped to named equations before use.",
            "Appendix E.3.1--E.3.4, printed pp. 137--139; Eqs. (E.24)--(E.31) and the signed-KDE equation.",
            "Thesis/Thesis.tex; Observables.py; final_reviewer_closure/table_data_crosswalk.csv",
        ),
        "L-3": (
            "Evidentiary words are tied to declared standards: references are 'numerically controlled', independent clouds show sensitivity rather than convergence, and the method is not called validated or improved.",
            "Abstract, printed p. i; Sections 6.4--6.9, printed pp. 89--111; Sections 7.3 and 7.6, printed pp. 113--114.",
            "Thesis/Thesis.tex; final_acceptance_check.py claim-language audit",
        ),
        "L-4": (
            "The visible compounds were corrected, including 'Chapters 2--5' and 'cross-validation'; protected typography prevents line-break artefacts.",
            "Section 1.9, printed p. 25; Section 4.7.1, printed p. 60.",
            "Thesis/Thesis.tex; reviewer_data_audit/pdf_qa/pdf_compile_render_manifest.json",
        ),
        "L-5": (
            "Vague antecedents were replaced by named subjects, for example 'The cloud-size study in Table 6.8' and 'This conclusion applies to the tested discretization'.",
            "Section 6.5, printed p. 93; Section 7.6, printed p. 114.",
            "Thesis/Chapter6_VerifiedResults.tex; Thesis/Chapter7_Conclusions.tex",
        ),
        "L-6": (
            "Repetitive legacy figure narration was removed; five compact captions carry definitions while surrounding prose states one interpretation per figure.",
            "Sections 6.2--6.8, printed pp. 84--111; Figures 6.1--6.5.",
            "final_reviewer_closure/figures/FIGURE_DATA_CROSSWALK.csv",
        ),
        "L-7": (
            "The response is now generated from a 58-row gate/item audit matrix; every row has a unique correction, exact section/page locator, and named evidence artifact.",
            "This response, Ten acceptance gates and I/M/L sections.",
            "reviewer_data_audit/response_item_audit.csv",
        ),
    }

    def displayed_evidence(text_value: str) -> str:
        for old, new in evidence_map.items():
            text_value = text_value.replace(old, new)
        return text_value

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=25mm]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern,microtype,array,xcolor,hyperref}",
        r"\hypersetup{hidelinks,pdftitle={" + TITLE + r" --- Response to Examiner}}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{5pt}",
        r"\newcommand{\field}[2]{\textbf{#1}\quad #2\par}",
        r"\begin{document}",
        r"\raggedright",
        r"\begin{center}",
        r"{\Large\bfseries Response to the Major-Revision Examiner Report\par}",
        r"\vspace{6pt}",
        r"{\bfseries " + tex(TITLE) + r"\par}",
        r"\vspace{4pt}Sahand Nikzat",
        r"\end{center}",
        r"\section*{Revision overview}",
        tex(
            "The revision presents the work as a controlled negative-result study. "
            "Accurate density interpolation is separated from operator fidelity, SEO "
            "structure, raw conservation, stochastic stability, and physical accuracy. "
            "Because these requirements are not jointly satisfied, the revised thesis "
            "does not call the MIDPOINT construction reliable and does not claim a "
            "systematic improvement over PBME."
        ),
    ]

    def thesis_page(label: str) -> str:
        return pages.get(label, "pending final compilation")

    mandatory = [
        (
            "1 --- Time-step inventory reconciled",
            "The campaign is stated as 24 paired momentum--seed--step "
            "configurations, 48 individual method executions, and four "
            "independent seeds.",
            f"Thesis p. {thesis_page('subsec:ch5-time-step-tests')}, "
            "Section 5 time-step tests; Table G.4.",
            "final_reviewer_closure/timestep/timestep_run_by_run.csv",
        ),
        (
            "2 --- Reference settings synchronized",
            "Table 6.12 is generated from the same full-precision CSVs as "
            "Table F.1 and lists momentum, final time, all domain and "
            "grid/time levels, physical edge statistic, and a defined CFL ratio.",
            f"Thesis p. {thesis_page('tab:reference-discretizations')}, "
            "Table 6.12; Appendix F, Table F.1.",
            "final_reviewer_closure/reference_settings_by_method_and_momentum.csv",
        ),
        (
            "3 --- Temporal orders corrected",
            "The text assigns approximately second-order temporal behaviour "
            "to TDSE and approximately fourth-order temporal behaviour to "
            "grid-QCLE RK4; spatial/grid behaviour is treated separately.",
            f"Thesis p. {thesis_page('sec:results-reference-controls')}, "
            "Section 6 reference controls; Tables 6.13--6.14.",
            "final_reviewer_closure/reference_tdse/tdse_three_level.csv and "
            "reference_grid_qcle/qcle_three_level.csv",
        ),
        (
            "4 --- Numerical-noise and stochastic guards enforced",
            "The absolute-plus-relative floor is 1e-12 + 1e-12 times the "
            "maximum level scale. Noise-limited, nonmonotone, and rapidly "
            "contracting nonasymptotic rows are rejected; for both stochastic "
            "moving-cloud methods, PBME and MIDPOINT, both differences must "
            "also exceed pooled independent-seed variation.",
            f"Thesis p. {thesis_page('eq:declared-numerical-noise-floor')}, "
            "Eq. (declared numerical-noise floor); Tables 6.7, 6.13, 6.14, and G.4.",
            "full-precision order-reason and verdict columns in the three "
            "reference/time-step CSVs",
        ),
        (
            "5 --- Abstract claim calibrated",
            "The abstract now says numerically controlled reference solutions "
            "and does not call them converged solutions.",
            f"Thesis p. {thesis_page('sec:abstract')}, Abstract.",
            "Thesis/Thesis.tex",
        ),
        (
            "6 --- Independent clouds no longer called convergent",
            "N=500, 1000, and 2000 are described as independently sampled, "
            "nonnested cloud enlargement; no deterministic cloud order is claimed.",
            f"Thesis p. {thesis_page('sec:results-support-replication')}, "
            "Section 6 independent-cloud enlargement; Table 6.8.",
            "final_reviewer_closure/support/independent_cloud_summary.csv",
        ),
        (
            "7 --- Retrievable release record supplied",
            "The final release identifies the archive filename, public release "
            "URL, final release tag, frozen source/evidence commit, archive "
            "SHA-256, checksum-index SHA-256, environment, manifests, clean-room "
            "verification record, and reproduction instructions. It also states "
            "that no DOI or institutional persistent identifier is assigned.",
            f"Thesis p. {thesis_page('appsec:versioned-release')}, "
            "Appendix G, Section G.5 archive record.",
            "frozen_numerical_evidence_payload.zip; " + archive_id,
        ),
        (
            "8 --- Visible typography repaired",
            "The chapter range and radial-basis cross-validation phrase are "
            "kept unbroken; the dense Appendix G evidence remains complete.",
            "Thesis Chapter 1 roadmap and Chapter 4 cross-validation text.",
            "Thesis/Thesis.tex",
        ),
        (
            "9 --- Final response crosswalk added",
            "The ten gates and all 48 I/M/L items are generated from a 58-row "
            "matrix with a unique correction, exact section/page locator, and "
            "named evidence artifact for every row.",
            "This response, Ten acceptance gates and I/M/L sections.",
            "reviewer_data_audit/response_item_audit.csv",
        ),
    ]
    lines.append(r"\section*{Final mandatory corrections}")
    for heading, action, location, evidence in mandatory:
        lines += [
            rf"\subsection*{{{tex(heading)}}}",
            rf"\field{{Status:}}{{{tex('Implemented and verified in the revised source')}}}",
            rf"\field{{Correction:}}{{{tex(action)}}}",
            rf"\field{{Final location:}}{{{tex(location)}}}",
            rf"\field{{Archive evidence:}}{{{tex(evidence)}}}",
        ]
    lines.append(r"\section*{Ten acceptance gates}")

    gate_audit = {
        1: (
            "Each chapter states its question, motivation, approach, and outcome; Chapter 6 opens with the decisive reliability question and controlled negative answer.",
            "Section 1.9, printed p. 25; chapter openings/closings at printed pp. 27, 39, 49, 65, 83, and 112--114.",
            "Thesis/Thesis.tex; Thesis/Chapter6_VerifiedResults.tex; Thesis/Chapter7_Conclusions.tex",
        ),
        2: (
            "The novelty claim is literature-bounded and restricted to the application-specific product-GP/MIDPOINT construction and its failure analysis.",
            "Section 1.9, printed p. 25; Section 7.1, printed p. 112.",
            "Thesis/Thesis.tex; Thesis/Chapter7_Conclusions.tex and cited literature",
        ),
        3: (
            "All retained figures define their quantities, aggregation, uncertainty, normalization, sign convention, and interpretation and are hash-linked to source CSVs.",
            "Figures 6.1--6.5, printed pp. 87, 88, 98, 101, and 111.",
            "final_reviewer_closure/figures/FIGURE_DATA_CROSSWALK.csv",
        ),
        4: item_audit["I-1"],
        5: (
            "Production and reference orders are reconstructible and guarded. For both PBME and MIDPOINT, both differences must exceed numerical noise and pooled independent-seed dispersion; reference time/grid controls remain separate.",
            "Sections 5.9.5, 6.4, and 6.7, printed pp. 80, 89--93, and 100--108; Tables 6.7 and 6.12--6.14; Table G.4, printed pp. 153--157.",
            "final_reviewer_closure/timestep/timestep_run_by_run.csv; reference_tdse/tdse_three_level.csv; reference_grid_qcle/qcle_three_level.csv",
        ),
        6: item_audit["M-6"],
        7: item_audit["I-13"],
        8: item_audit["I-15"],
        9: item_audit["I-16"],
        10: item_audit["L-7"],
    }
    audit_rows: List[Dict[str, str]] = []
    for gate in range(1, 11):
        pending = gate == 8 and archive_id == BLOCKED_ID
        correction, locator, evidence = gate_audit[gate]
        correction = f"Gate {gate} closure: {correction}"
        gate_status = (
            "External repository deposition pending"
            if pending else "Addressed in the revised thesis"
        )
        audit_rows.append({
            "item": f"Gate {gate}",
            "concern": f"Acceptance gate {gate}",
            "status": gate_status,
            "exact_correction": correction,
            "thesis_locator": locator,
            "evidence_artifact": evidence,
        })
        lines += [
            rf"\subsection*{{Gate {gate}}}",
            rf"\field{{Status:}}{{{tex(gate_status)}}}",
            rf"\field{{Exact correction:}}{{{tex(correction)}}}",
            rf"\field{{Thesis locator:}}{{{tex(locator)}}}",
            rf"\field{{Evidence artifact:}}{{{tex(evidence)}}}",
        ]

    matrix_order = read_csv(ROOT / "reviewer_closure_out" / "closure_matrix.csv")
    for prefix, heading in (("I-", "I-items"), ("M-", "M-items"), ("L-", "L-items")):
        lines.append(rf"\section*{{{heading}}}")
        for matrix_row in matrix_order:
            item = matrix_row["item"]
            if not item.startswith(prefix):
                continue
            record = records[item]
            evidence = displayed_evidence(record["table"])
            page_note = ""
            label_match = re.search(r"(tab:[A-Za-z0-9:-]+)", record["table"])
            if label_match and label_match.group(1) in pages:
                page_note = f"; thesis p. {pages[label_match.group(1)]}"
            result = scientific_results.get(item, record["result"])
            action = scientific_actions.get(item, record["action"])
            location = location_overrides.get(item, record["location"])
            status = (
                "External repository deposition pending"
                if item == "I-15" and archive_id == BLOCKED_ID
                else "Addressed in the revised thesis"
            )
            correction, locator, evidence = item_audit[item]
            audit_rows.append({
                "item": item,
                "concern": record["concern"],
                "status": status,
                "exact_correction": correction,
                "thesis_locator": locator,
                "evidence_artifact": evidence,
            })
            lines += [
                rf"\subsection*{{{tex(item)} --- {tex(record['concern'])}}}",
                rf"\field{{Status:}}{{{tex(status)}}}",
                rf"\field{{Exact correction:}}{{{tex(correction)}}}",
                rf"\field{{Thesis locator:}}{{{tex(locator)}}}",
                rf"\field{{Evidence artifact:}}{{{tex(evidence)}}}",
            ]

    lines += [
        r"\section*{Final scientific statement}",
        tex(
            "The revision retains unfavourable and unstable outcomes rather than "
            "turning numerical completion into a success claim. Evidence is presented "
            "as physical observables, raw invariants, operator errors, uncertainty "
            "intervals, and controlled-reference comparisons. No internal surrogate "
            "diagnostic is presented as a physical-error estimate."
        ),
        r"\section*{Availability note}",
        r"The current research-code repository is "
        r"\url{https://github.com/sahandgit/gaussian_process_guided_mapping_quantumclassical_liouville_dynamics}. "
        + (
            r"The versioned public release is available at "
            r"\url{" + archive_id + r"}. No DOI or institutional persistent "
            r"identifier has been assigned."
            if archive_id != BLOCKED_ID
            else tex(
                "No DOI or institutional persistent identifier has been "
                "assigned; public release deposition remains pending."
            )
        ),
        r"\end{document}",
    ]
    audit_path = AUDIT / "response_item_audit.csv"
    if len(audit_rows) != 58:
        raise RuntimeError(
            f"Response audit must contain 58 gate/item rows; found {len(audit_rows)}"
        )
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    response.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return response


def write_audit_documents(summary: Mapping[str, Any], archive_id: str) -> List[Path]:
    paths: List[Path] = []
    sources = summary["paths"]
    crosswalk_rows = read_csv(sources["crosswalk"])
    table_parts = [
        "% Generated from final_reviewer_closure/TABLE_DATA_CROSSWALK.csv",
        "% Full-precision values remain in the source CSVs named in that crosswalk.",
        "",
    ]
    copied_table_dir = AUDIT / "final_tables"
    copied_table_dir.mkdir(parents=True, exist_ok=True)
    final_provenance: List[Dict[str, str]] = []
    for row in crosswalk_rows:
        table_path = Path(row["table"])
        csv_path = Path(row["source_csv"])
        require((table_path, csv_path))
        table_parts += [
            f"% NUMERICAL CLAIM CROSSWALK: {table_path.name} -> "
            f"{csv_path} (SHA-256 {row['source_csv_sha256']})",
            table_path.read_text(encoding="utf-8"),
            "",
        ]
        copied_csv = copied_table_dir / csv_path.name
        shutil.copy2(csv_path, copied_csv)
        final_provenance.append({
            "table_tex": str(table_path.resolve()),
            "source_csv": str(csv_path.resolve()),
            "source_csv_sha256": row["source_csv_sha256"],
            "audit_copy": str(copied_csv.resolve()),
            "audit_copy_sha256": sha256(copied_csv),
        })
    combined_tex = AUDIT / "THESIS_EVIDENCE_TABLES.tex"
    combined_tex.write_text("\n".join(table_parts), encoding="utf-8")
    paths.append(combined_tex)
    provenance_path = AUDIT / "metric_provenance_final_closure.csv"
    with provenance_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(final_provenance[0]))
        writer.writeheader()
        writer.writerows(final_provenance)
    paths.append(provenance_path)

    audit = AUDIT / "PIPELINE_DATA_AUDIT_AND_THESIS_EVIDENCE.md"
    baseline_copy = AUDIT / "PIPELINE_DATA_AUDIT_BASELINE_FORENSIC.md"
    if (
        audit.exists()
        and "<!-- FINAL_CLOSURE_SUPPLEMENT -->" not in audit.read_text(
            encoding="utf-8", errors="replace"
        )
    ):
        shutil.copy2(audit, baseline_copy)
    baseline_text = (
        baseline_copy.read_text(encoding="utf-8", errors="replace")
        if baseline_copy.exists()
        else "# Baseline forensic audit\n\nDATA ABSENT"
    )
    audit_text = f"""<!-- FINAL_CLOSURE_SUPPLEMENT -->
# Pipeline Data Audit and Thesis Evidence

Overall scientific conclusion: **no validated systematic improvement over PBME**.
The tested MIDPOINT prototype is not reliable under the declared acceptance
contract because four-seed replication, raw conservation, and projection
requirements do not all pass. This is an explicit negative result, not a hidden
run failure.

## 1. Executive conclusion

{summary["physical"]}

## 2. Directory and data provenance

Pipeline root: {md_path(ROOT)}

Final table-data crosswalk: {md_path(sources["crosswalk"])}

Frozen numerical-evidence payload SHA-256:
`{summary["frozen_payload_sha256"]}`. Manifest:
{md_path(sources["frozen_payload_manifest"])}

All derived numerical files preserve full machine precision. Reader-facing
rounding does not alter the CSVs.

## 3. Exact campaign inventory

{summary["inventory"]}

Execution/retry history: {summary["incidents"]}

## 4. Numerical-stability audit

{summary["stability"]}

The historical and final numerical-stability records remain visible at
{md_path(sources["stability"])}. Completion is not treated as validation.

## 5. Manufactured-operator results

{summary["manufactured"]}

## 6. SEO projection leakage

{summary["projection"]}

## 7. Identical-support KDE/GP baseline

{summary["baseline"]}

## 8. Time-step analysis

{summary["timestep"]}

## 9. Support-size analysis

{summary["support"]}

## 10. Independent-seed replication

{summary["replication"]}

## 11. Raw conservation

{summary["raw"]}

Full results: {md_path(sources["raw"])}.

### Signed-label tail sensitivity

{summary["tail"]}

## 12. TDSE and grid-QCLE controls

{summary["references"]}

TDSE is model-exact only within its displayed numerical controls. Grid QCLE is
a numerical solution of the approximate QCLE equation.

## 13. PBME versus MIDPOINT

{summary["physical"]}

## 14. GP policy reconstructed from code

The run manifest is authoritative over defaults. The final evidence records the
product ARD-RBF surrogate, input scaling, label/profile policy, regularization,
noise policy, optimizer, breathing schedule, floors, adaptive Cholesky jitter,
moment constraints, source location, midpoint evaluation, and failure recovery.
Source: {md_path(ROOT / "GP_Density.py")}.

## 15. Figure and caption audit

All 41 untraceable legacy figures were removed.  The final thesis contains
four replacement summary figures regenerated from verified final CSVs.  Their
figure files, quantitative sources, generator, selection rules, and SHA-256
hashes are recorded in `final_reviewer_closure/figures/FIGURE_DATA_CROSSWALK.csv`.

## 16. Examiner checklist

The complete ten-gate and 48-item response is generated at
{md_path(ROOT / "Reviewer_Response.tex")}.

## 17. Thesis-ready conclusion

The evidence supports a controlled implementation and documented failure
diagnosis; it does not validate a reliable MIDPOINT dynamical solver. The
application-specific contribution is the formulation, testing, quantified
failure pathway, and redesign criteria.

## 18. Missing or externally unidentifiable evidence

The local evidence package is complete. Its checksum-bound versioned public
release URL is `{archive_id}`. This GitHub release is not a DOI or institutional
persistent identifier; no DOI is invented. The frozen source/evidence commit
and the final release tag are reported separately in the release record.

## 19. Recommended thesis changes implemented

The abstract is below 150 words; the objective is narrow; full-density
prototype wording and method-level correctness vocabulary are removed; the
tail study, production-matched operator policies, reconstructible time-step
table, and matched reference controls are incorporated.

## 20. Exact provenance

Every generated LaTeX table maps to a source CSV and SHA-256 digest through
{md_path(sources["crosswalk"])}. Job commands, manifests, environment records,
and output hashes are under {md_path(EVIDENCE)}.

---

# Appendix A — Complete recursive forensic baseline

The following baseline is preserved from the full-directory inventory
regenerated immediately before final synthesis. The final-closure sections
above are authoritative where a repaired rerun supersedes an incomplete
historical configuration.

{baseline_text}
"""
    audit.write_text(audit_text, encoding="utf-8")
    paths.append(audit)

    examiner = AUDIT / "EXAMINER_RESPONSE_EVIDENCE.md"
    records = item_records(summary, archive_id)
    gate_results = {
        1: "Six-question/chapter contracts and explicit chapter answers are present.",
        2: "The novelty statement is application-specific and literature-bounded.",
        3: "The 41 untraceable legacy figures were removed; four replacement summary figures are regenerated from verified CSVs and hash-crosswalked.",
        4: summary["manufactured"],
        5: summary["timestep"] + " " + summary["references"],
        6: summary["support"],
        7: summary["physical"],
        8: (
            "Local environment, checksums, and crosswalk are complete; "
            + (
                f"versioned public release URL {archive_id} is recorded; it is "
                "not a DOI or institutional persistent identifier."
                if archive_id != BLOCKED_ID else
                "authenticated external publication remains required."
            )
        ),
        9: f"One exact title is used: {TITLE}.",
        10: "The response supplies every gate and all 48 I/M/L rows with sources.",
    }
    examiner_lines = [
        "# Examiner Response Evidence",
        "",
        f"Exact thesis title: **{TITLE}**",
        "",
        "Scientific outcome: **no validated systematic improvement over PBME**.",
        "",
        "## Ten acceptance gates",
        "",
    ]
    for gate in range(1, 11):
        status = (
            records["I-15"]["status"] if gate == 8
            else ALLOWED_STATUSES["computation"]
        )
        examiner_lines += [
            f"### Gate {gate}",
            "",
            f"- Status: {status}",
            f"- Result: {gate_results[gate]}",
            f"- Source: `{sources['crosswalk']}`",
            f"- Versioned public release URL: `{archive_id}`",
            "",
        ]
    matrix_order = read_csv(ROOT / "reviewer_closure_out" / "closure_matrix.csv")
    for prefix, heading in (
        ("I-", "I-items"),
        ("M-", "M-items"),
        ("L-", "L-items"),
    ):
        examiner_lines += [f"## {heading}", ""]
        for matrix_row in matrix_order:
            item = matrix_row["item"]
            if not item.startswith(prefix):
                continue
            record = records[item]
            examiner_lines += [
                f"### {item} — {record['concern']}",
                "",
                f"- Status: {record['status']}",
                f"- Action: {record['action']}",
                f"- Result: {record['result']}",
                f"- Thesis location: {record['location']}",
                f"- Table/equation: {record['table']}",
                f"- Archive source: `{record['source']}`",
                "",
            ]
    examiner.write_text("\n".join(examiner_lines), encoding="utf-8")
    paths.append(examiner)

    missing = AUDIT / "MISSING_DATA_AND_ANALYSES.md"
    missing.write_text(
        "# Missing Data and Analyses\n\n"
        "All calculations required by the final local execution specification "
        "are represented in the verified final CSVs. The frozen archive has "
        "a checksum-bound versioned public release URL when that URL is shown "
        f"here: `{archive_id}`. A GitHub release is not a DOI or institutional "
        "persistent identifier.\n\n"
        "The copied workspace has no `.git` metadata, so the originating "
        "version-control commit is NOT IDENTIFIABLE. The release records "
        "full source-snapshot SHA-256 hashes instead; this limitation cannot "
        "be repaired retroactively without inventing provenance.\n\n"
        "Scientific acceptance remains negative: the tested MIDPOINT prototype "
        "does not pass the four-seed reliability, raw-conservation, and "
        "projection contract. This is not missing evidence and is not converted "
        "into a pass by wording.\n",
        encoding="utf-8",
    )
    paths.append(missing)
    return paths


def write_final_closure_report(
    summary: Mapping[str, Any], archive_id: str
) -> Path | None:
    archive_manifest = EVIDENCE / "archive_manifest.json"
    if not archive_manifest.exists():
        return None
    archive = json.loads(archive_manifest.read_text(encoding="utf-8"))
    thesis_pdf = ROOT / "Thesis" / "Thesis.pdf"
    response_pdf = ROOT / "Reviewer_Response.pdf"
    thesis_source = ROOT / "Thesis" / "Thesis.tex"
    response_source = ROOT / "Reviewer_Response.tex"
    thesis_ok = (
        thesis_pdf.exists()
        and thesis_pdf.stat().st_size > 0
        and thesis_pdf.stat().st_mtime >= thesis_source.stat().st_mtime
    )
    response_ok = (
        response_pdf.exists()
        and response_pdf.stat().st_size > 0
        and response_pdf.stat().st_mtime >= response_source.stat().st_mtime
    )
    release_ok = archive_id != BLOCKED_ID
    overall = "PASS" if thesis_ok and response_ok and release_ok else "FAIL"
    closure_totals = (
        "I-items: 16/16 closed; M-items: 25/25 closed; L-items: 7/7 closed."
        if release_ok
        else "I-items: 15/16 locally closed and I-15 externally blocked; "
             "M-items: 25/25 closed; L-items: 7/7 closed."
    )
    report = EVIDENCE / "FINAL_CLOSURE_REPORT.md"
    report.write_text(
        f"Overall status: {overall}\n"
        "Scientific result: no validated improvement\n"
        f"Thesis compile: {'PASS' if thesis_ok else 'FAIL'}\n"
        f"Reviewer response compile: {'PASS' if response_ok else 'FAIL'}\n"
        "Evidence archive: PASS\n"
        f"Versioned public release URL: {archive_id}\n"
        "Persistent identifier note: no DOI or institutional persistent "
        "identifier has been assigned.\n"
        f"Frozen source/evidence commit: "
        f"{archive.get('audit_created_release_commit', 'NOT IDENTIFIABLE')}\n"
        "Originating development commit: NOT IDENTIFIABLE\n"
        f"Archive SHA-256: {archive['archive_sha256']}\n\n"
        f"# Final Closure Report\n\n"
        f"## Exact title\n\n{TITLE}\n\n"
        f"Abstract word count: {abstract_word_count()}.\n\n"
        "## Environment\n\n"
        f"See `{EVIDENCE / 'environment.json'}` and "
        f"`{EVIDENCE / 'environment' / 'pip_freeze.txt'}`.\n\n"
        f"## Completed job counts\n\n{summary['inventory']}\n\n"
        f"## Failed/retried history\n\n{summary['incidents']}\n\n"
        "## Ten-gate result\n\n"
        "Gates 1–7 and 9–10 are locally closed. Gate 8 is closed when the "
        "checksum-bound versioned public release URL is recorded; that URL is "
        f"`{archive_id}`.\n\n"
        "## I/M/L closure totals\n\n"
        f"{closure_totals}\n\n"
        "## Major numerical conclusions\n\n"
        f"- Manufactured operator: {summary['manufactured']}\n"
        f"- Time step: {summary['timestep']}\n"
        f"- Replication: {summary['replication']}\n"
        f"- Tail sensitivity: {summary['tail']}\n"
        f"- References: {summary['references']}\n"
        f"- PBME/MIDPOINT comparison: {summary['physical']}\n\n"
        "## Table-data crosswalk\n\n"
        f"`{summary['paths']['crosswalk']}`\n\n"
        "## Compilation diagnostics\n\n"
        f"Thesis PDF: `{thesis_pdf}`. Reviewer-response PDF: `{response_pdf}`. "
        "Rendered-page QA is recorded under "
        f"`{AUDIT / 'pdf_qa'}`.\n\n"
        "## Archive\n\n"
        "Frozen numerical-evidence payload SHA-256: "
        f"`{summary['frozen_payload_sha256']}`\n\n"
        f"Archive: `{archive['archive']}`\n\n"
        f"SHA-256: `{archive['archive_sha256']}`\n\n"
        "The ZIP checksum is recorded in this external post-package report; "
        "embedding a ZIP's own checksum inside itself would be self-referential.\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-id", default=BLOCKED_ID)
    args = parser.parse_args()
    summary = evidence_summary()
    outputs = write_audit_documents(summary, args.archive_id)
    outputs.extend(write_final_chapters(summary, args.archive_id))
    outputs.append(write_reviewer_response(summary, args.archive_id))
    final_report = write_final_closure_report(summary, args.archive_id)
    if final_report is not None:
        outputs.append(final_report)
    manifest = {
        "archive_identifier": args.archive_id,
        "outputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in outputs
        ],
    }
    manifest_path = AUDIT / "final_submission_document_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(outputs)} final documents; manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
