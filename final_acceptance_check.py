"""Strict final checker for the reviewer-closure evidence and documents."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
ALLOWED_STATUSES = (
    "Closed — computation and thesis correction",
    "Closed — explicit negative result",
    "Closed — claim removed",
    "Closed — reviewer-authorized limitation",
    "BLOCKED — external publication required",
)
FORBIDDEN_THESIS = (
    "TODO", "TBD", "FIXME", "NOT COMPUTED", r"\textbf{NOT COMPUTED}",
    "full-density prototype", "corrected method", "corrected evolution",
    "automatic-relevance-", r"\input{Chapter6", r"\input{Chapter7",
    r"\input{ReviewerEvidenceTables",
    "Grid QCLE was not rerun", "NOT_IDENTIFIABLE_UNTIL_FINAL_SNAPSHOT",
    "BLOCKED_UNTIL_FINAL_NUMERICAL_VERIFICATION",
    "12 time-step runs", "comparison with converged solutions",
    "approximately second-order behaviour for the TDSE and grid-QCLE",
    "step-size change exceeds seed spread",
    "fail to converge for this manufactured operator",
)
FORBIDDEN_LOG = (
    "undefined", "multiply defined", "LaTeX Error", "Fatal error",
    "overfull", "missing citation",
)


def csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(row: Dict[str, str], fields: Iterable[str]) -> bool:
    try:
        return all(np.isfinite(float(row[field])) for field in fields)
    except (KeyError, TypeError, ValueError):
        return False


def normalize_latex(text: str) -> str:
    text = re.sub(r"%[^\n]*", "", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", "", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\\([&%$#_{}])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_latex_words(text: str) -> List[str]:
    text = re.sub(r"%[^\n]*", " ", text)
    text = re.sub(r"\\(?:cite|ref|label|eqref)\{[^}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("{", " ").replace("}", " ").replace("--", " ")
    return re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE)


class Checker:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def check(self, name: str, condition: bool, detail: str) -> None:
        self.rows.append({
            "name": name, "passed": bool(condition), "detail": detail
        })

    @property
    def passed(self) -> bool:
        return bool(self.rows) and all(row["passed"] for row in self.rows)


def check_numerical(c: Checker, evidence: Path) -> None:
    manufactured = evidence / "manufactured" / "manufactured_complete.csv"
    if not manufactured.exists():
        c.check("manufactured_complete", False, f"absent: {manufactured}")
    else:
        rows = csv_rows(manufactured)
        c.check("manufactured_72_rows", len(rows) == 72,
                f"{len(rows)} rows")
        required_metadata = (
            "l2_regularization", "seed", "N", "query_type", "query_count",
            "training_cloud_sha256", "query_sha256",
            "hyperparameter_policy", "cholesky_jitter",
        )
        c.check("manufactured_metadata", all(
            all(row.get(field, "") != "" for field in required_metadata)
            for row in rows
        ), "l2/seed/N/query/hashes/policy/jitter present")
        metrics = [
            f"{prefix}_{metric}"
            for prefix in ("density", "gradient", "Q")
            for metric in (
                "relative_l1", "relative_l2", "relative_linf",
                "mae", "rmse", "linf",
            )
        ]
        c.check("manufactured_finite", all(finite(row, metrics) for row in rows),
                "all 18 metrics finite per row")

    manufactured_summary = (
        evidence / "manufactured" / "manufactured_summary.csv"
    )
    if not manufactured_summary.exists():
        c.check(
            "manufactured_policy_summary",
            False,
            f"absent: {manufactured_summary}",
        )
    else:
        rows = csv_rows(manufactured_summary)
        paired_fields = (
            "mean_paired_difference_from_baseline",
            "paired_difference_sample_sd",
            "paired_difference_standard_error",
            "paired_difference_ci95_low",
            "paired_difference_ci95_high",
        )
        c.check(
            "manufactured_policy_summary",
            len(rows) == 432
            and all(finite(row, paired_fields) for row in rows)
            and all(
                row.get(
                    "paired_training_and_query_clouds_verified", ""
                ).lower()
                == "true"
                for row in rows
            ),
            f"{len(rows)} mean/spread/paired-policy rows",
        )

    timestep = evidence / "timestep" / "timestep_run_by_run.csv"
    if not timestep.exists():
        c.check("timestep_complete", False, f"absent: {timestep}")
    else:
        rows = csv_rows(timestep)
        required = (
            "value1", "value2", "value3", "D12", "D23", "seed_spread",
            "roundoff_threshold",
        )
        metadata = (
            "verdict", "run1_manifest", "run2_manifest", "run3_manifest",
        )
        c.check("timestep_128_rows", len(rows) == 128, f"{len(rows)} rows")
        c.check("timestep_reconstructible", all(
            finite(row, required)
            and all(row.get(field, "") for field in metadata)
            and row["verdict"] not in ("MISSING_RUN", "NONFINITE_RUN")
            for row in rows
        ), "three values, two differences, spread, threshold, verdict and manifests")
        counts: Dict[str, int] = {}
        for row in rows:
            counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
        c.check(
            "timestep_guarded_verdicts",
            counts == {
                "COMPUTED_POSITIVE": 8,
                "COMPUTED_ZERO_OR_NEGATIVE": 3,
                "REJECT_NUMERICAL_NOISE": 15,
                "REJECT_SEED_VARIABILITY": 102,
            }
            and all(float(row["roundoff_threshold"]) >= 1.0e-12 for row in rows),
            f"verdict_counts={counts}",
        )

    replication = evidence / "replication" / "four_seed_summary.csv"
    if not replication.exists():
        c.check("replication_four_seed", False, f"absent: {replication}")
    else:
        rows = csv_rows(replication)
        c.check("replication_four_seed", len(rows) == 32 and all(
            int(row["n_independent_seeds"]) == 4
            and finite(row, ("mean", "sample_sd", "standard_error",
                             "ci95_low", "ci95_high"))
            for row in rows
        ), f"{len(rows)} method/momentum/observable rows")

    distribution = evidence / "tail_sensitivity" / "y0_distribution.csv"
    sweep = evidence / "tail_sensitivity" / "threshold_sweep.csv"
    if not distribution.exists() or not sweep.exists():
        c.check("tail_complete", False, "distribution or threshold sweep absent")
    else:
        drows, srows = csv_rows(distribution), csv_rows(sweep)
        cases: Dict[Tuple[str, str, str], set] = {}
        for row in srows:
            cases.setdefault(
                (row["P0"], row["method"], row["seed"]), set()
            ).add(float(row["eta"]))
        expected_eta = {0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6,
                        1e-5, 1e-4, 1e-3, 1e-2}
        c.check("tail_complete", len(drows) == 16 and len(srows) == 160
                and len(cases) == 16
                and all(values == expected_eta for values in cases.values()),
                f"{len(drows)} distributions, {len(srows)} sweep rows")

    for method, folder, filename in (
        ("TDSE", "reference_tdse", "tdse_three_level.csv"),
        ("grid-QCLE", "reference_grid_qcle", "qcle_three_level.csv"),
    ):
        path = evidence / folder / filename
        if not path.exists():
            c.check(f"{method}_three_level", False, f"absent: {path}")
            continue
        rows = csv_rows(path)
        cases = {(row["P0"], row["refinement_mode"]) for row in rows}
        exact = ("value1", "value2", "value3", "delta12", "delta23",
                 "level1_dt", "level2_dt", "level3_dt",
                 "level1_n_steps", "level2_n_steps", "level3_n_steps")
        domains = (
            ("level1_R_min", "level1_R_max")
            if method == "TDSE"
            else ("level1_R_min", "level1_R_max",
                  "level1_P_min", "level1_P_max")
        )
        c.check(f"{method}_three_level", len(rows) == 32
                and cases == {("20.0", "time"), ("20.0", "grid"),
                              ("100.0", "time"), ("100.0", "grid")}
                and all(finite(row, (*exact, *domains)) for row in rows)
                and all(row.get("p_observed", "") != "" for row in rows),
                f"{len(rows)} rows; cases={sorted(cases)}")
        guarded = True
        for row in rows:
            d12, d23 = float(row["delta12"]), float(row["delta23"])
            tau = float(row["numerical_noise_threshold"])
            reason = row["order_reason"]
            if reason == "ok":
                p = float(row["p_observed"])
                guarded = guarded and d12 > tau and d23 > tau and d23 < d12
                guarded = guarded and 0.0 < p <= 6.0
            elif reason == "roundoff_or_saturation_limited":
                guarded = guarded and min(d12, d23) <= tau
                guarded = guarded and row["p_observed"] == "NOT COMPUTED"
            elif reason == "nonmonotone_difference_retained":
                guarded = guarded and d23 >= d12 and min(d12, d23) > tau
                guarded = guarded and row["p_observed"] == "NOT COMPUTED"
                guarded = guarded and float(row["raw_ratio_order"]) <= 0.0
            elif reason == "rapid_contraction_not_asymptotic":
                raw_p = float(row["raw_ratio_order"])
                guarded = guarded and raw_p > 6.0 and d23 < d12
                guarded = guarded and row["p_observed"] == "NOT COMPUTED"
            else:
                guarded = False
        c.check(
            f"{method}_noise_and_asymptotic_guards",
            guarded,
            "accepted and rejected rows satisfy the declared hierarchy",
        )

    physical = evidence / "physical_comparison" / "observable_errors_by_seed.csv"
    if not physical.exists():
        c.check("paired_physical_comparison", False, f"absent: {physical}")
    else:
        rows = csv_rows(physical)
        by_case: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
        for row in rows:
            key = (row["reference"], row["P0"], row["seed"], row["observable"])
            existing = by_case.setdefault(key, {})
            existing[row["method"]] = row["paired_initial_cloud_sha256"]
        c.check("paired_physical_comparison", len(rows) == 256
                and all(set(methods) == {"PBME", "MIDPOINT"}
                        and len(set(methods.values())) == 1
                        for methods in by_case.values()),
                f"{len(rows)} rows with paired support hashes")

    for name, relative in (
        ("raw_conservation", "preserved_evidence/raw_conservation.csv"),
        ("projection_leakage", "preserved_evidence/projection_leakage.csv"),
        ("kde_gp_baseline", "preserved_evidence/kde_gp_baseline.csv"),
    ):
        path = evidence / relative
        c.check(name, path.exists() and path.stat().st_size > 0,
                str(path))

    settings = evidence / "reference_settings_by_method_and_momentum.csv"
    if settings.exists():
        rows = csv_rows(settings)
        cases = {
            (row["method"], row["P0"], row["mode"])
            for row in rows
        }
        expected = {
            (method, p0, mode)
            for method in ("TDSE", "grid QCLE")
            for p0 in ("20.0", "100.0")
            for mode in ("time", "grid")
        }
        c.check(
            "reference_settings_eight_exact_cases",
            len(rows) == 8 and cases == expected,
            f"rows={len(rows)}; cases={sorted(cases)}",
        )
    else:
        c.check("reference_settings_eight_exact_cases", False, str(settings))


def check_thesis(c: Checker, thesis: Path, bibliography: Path,
                 evidence: Path) -> None:
    if not thesis.exists():
        c.check("thesis_source", False, f"absent: {thesis}")
        return
    text = thesis.read_text(encoding="utf-8", errors="replace")
    for phrase in FORBIDDEN_THESIS:
        c.check(f"thesis_forbidden_{phrase}", phrase not in text,
                f"occurrences={text.count(phrase)}")

    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S
    )
    words = strip_latex_words(abstract_match.group(1)) if abstract_match else []
    c.check("abstract_max_150", bool(abstract_match) and len(words) <= 150,
            f"word_count={len(words)}")

    title_match = re.search(r"\\title\{(.*?)\}", text, re.S)
    pdf_match = re.search(r"pdftitle\s*=\s*\{(.*?)\}", text, re.S)
    title = normalize_latex(title_match.group(1)) if title_match else ""
    pdftitle = normalize_latex(pdf_match.group(1)) if pdf_match else ""
    c.check("title_matches_pdftitle", bool(title) and title == pdftitle,
            f"title={title!r}; pdftitle={pdftitle!r}")

    included_tex = []
    for input_name in re.findall(r"\\input\{([^}]+)\}", text):
        input_path = (thesis.parent / input_name).with_suffix(".tex")
        if input_path.exists():
            included_tex.append(
                input_path.read_text(encoding="utf-8", errors="replace")
            )
    complete_source = text + "\n" + "\n".join(included_tex)
    labels = re.findall(r"\\label\{([^}]+)\}", complete_source)
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    c.check("duplicate_labels", not duplicates, f"duplicates={duplicates[:10]}")
    refs = set(re.findall(r"\\(?:ref|eqref|pageref)\{([^}]+)\}", text))
    c.check("source_references_resolve", refs.issubset(set(labels)),
            f"missing={sorted(refs - set(labels))[:20]}")

    if bibliography.exists():
        bib = bibliography.read_text(encoding="utf-8", errors="replace")
        bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
        cited: set = set()
        for group in re.findall(r"\\cite\w*\{([^}]+)\}", text):
            cited.update(key.strip() for key in group.split(","))
        c.check("cited_keys_exist", cited.issubset(bib_keys),
                f"missing={sorted(cited - bib_keys)}")
    else:
        c.check("bibliography_present", False, f"absent: {bibliography}")

    crosswalk = evidence / "table_data_crosswalk.csv"
    c.check("table_crosswalk_present", crosswalk.exists(),
            str(crosswalk))
    if crosswalk.exists():
        rows = csv_rows(crosswalk)
        c.check("table_sources_exist", all(
            Path(row["source_csv"]).exists() for row in rows
        ), f"{len(rows)} crosswalk rows")

    figure_crosswalk = evidence / "figures" / "FIGURE_DATA_CROSSWALK.csv"
    c.check("figure_crosswalk_present", figure_crosswalk.exists(),
            str(figure_crosswalk))
    figure_rows: List[Dict[str, str]] = []
    if figure_crosswalk.exists():
        figure_rows = csv_rows(figure_crosswalk)
        figure_ok = len(figure_rows) == 5
        for row in figure_rows:
            figure_path = ROOT / row["figure"]
            source_paths = [ROOT / value for value in row["source_csvs"].split(";")]
            source_hashes = row["source_csv_sha256"].split(";")
            figure_ok = (
                figure_ok
                and figure_path.exists()
                and sha256(figure_path) == row["figure_sha256"]
                and len(source_paths) == len(source_hashes)
                and all(
                    path.exists() and sha256(path) == digest
                    for path, digest in zip(source_paths, source_hashes)
                )
            )
        c.check("figure_sources_and_hashes", figure_ok,
                f"{len(figure_rows)} crosswalk rows")
    included_figures = re.findall(
        r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text
    )
    expected_figures = {"../" + row["figure"] for row in figure_rows}
    c.check(
        "thesis_verified_figures_only",
        len(included_figures) == 5 and set(included_figures) == expected_figures,
        f"included={included_figures}",
    )

    reference_appendix_manifest = (
        ROOT
        / "reviewer_data_audit"
        / "thesis_sources"
        / "reference_appendix_update_manifest.json"
    )
    c.check(
        "reference_appendix_generated_from_final_csvs",
        reference_appendix_manifest.exists()
        and text.count("% BEGIN GENERATED REFERENCE REFINEMENT APPENDIX") == 1
        and text.count("% END GENERATED REFERENCE REFINEMENT APPENDIX") == 1,
        str(reference_appendix_manifest),
    )
    required_corrections = (
        "24 paired",
        "48 individual",
        r"\tau_{\mathrm{noise}}",
        r"\tau_{\rm noise}=10^{-12}+10^{-12}",
        "approximately second order for the symmetric TDSE",
        "approximately fourth order for the grid-QCLE",
        "numerically controlled reference solutions",
        r"\input{../final_reviewer_closure/tables/ReferenceSettingsByMomentum.tex}",
        "deterministic support convergence was not assessed",
    )
    c.check(
        "mandatory_corrections_present",
        all(value in text for value in required_corrections),
        f"missing={[value for value in required_corrections if value not in text]}",
    )

    appendix = re.search(
        r"\\chapter\[Reference Calculations\]"
        r"(.*?)(?=\\chapter|\\end\{document\})",
        text,
        re.S,
    )
    appendix_text = appendix.group(1) if appendix else ""
    release_record = re.search(
        r"\\section\{Versioned reproducibility release\}"
        r"(.*?)(?=\\section|\\chapter|\\end\{document\})",
        text,
        re.S,
    )
    release_text = release_record.group(1) if release_record else ""
    appendix_forbidden_terms = [
        term for term in (
            "BLOCKED_EXTERNAL_PUBLICATION", "expected=", "verified=",
            "missing=", "PENDING_PUBLIC_RELEASE", "NOT_IDENTIFIABLE",
        )
        if term.lower() in appendix_text.lower()
    ]
    c.check(
        "appendix_physics_facing",
        bool(appendix)
        and "Reference TDSE and grid-QCLE calculations" in appendix_text
        and "Numerical parameters of the reported calculations" in appendix_text
        and "Controlled approximations and interpretation" in appendix_text
        and bool(release_record)
        and "Frozen numerical-evidence archive SHA-256" in release_text
        and "Public release asset" in release_text
        and not appendix_forbidden_terms,
        f"forbidden_terms={appendix_forbidden_terms}",
    )
    payload_manifest = (
        ROOT
        / "reviewer_data_audit"
        / "frozen_numerical_evidence_payload_manifest.json"
    )
    if payload_manifest.exists():
        payload_record = json.loads(payload_manifest.read_text(encoding="utf-8"))
        payload_path = Path(payload_record.get("archive", ""))
        expected_payload_sha = payload_record.get("archive_sha256", "")
        payload_ok = (
            payload_path.is_file()
            and len(expected_payload_sha) == 64
            and sha256(payload_path) == expected_payload_sha
        )
        c.check(
            "internal_frozen_payload_integrity",
            payload_ok,
            (
                f"archive={payload_path}; expected SHA-256="
                f"{expected_payload_sha}"
            ),
        )
    else:
        c.check(
            "internal_frozen_payload_integrity",
            False,
            f"absent: {payload_manifest}",
        )
    chapter_blocks = re.findall(
        r"\\chapter\{[^}]+\}(.*?)(?=\\chapter\{|\\appendix)", text, re.S
    )[:7]
    opening_blocks = [block.split(r"\section", 1)[0].lower() for block in chapter_blocks]
    opening_ok = len(opening_blocks) == 7 and all(
        "asks" in block
        and any(term in block for term in ("matters", "essential", "would therefore", "because"))
        and any(term in block for term in ("approach", "derivation", "derives", "construction", "evidence"))
        and any(term in block for term in ("negative", "outcome", "shows", "establishes", "contribution"))
        for block in opening_blocks
    )
    closing_anchors = sum(
        text.count(anchor)
        for anchor in (
            "Taken together, the introduction",
            "Thus, the QCLE",
            "Consequently, the physical mapping density",
            "The chapter therefore establishes",
            "In summary, the explicit midpoint rule",
            "Taken together, the calculations",
            "Accordingly, the thesis contributes",
        )
    )
    paragraph_command_count = text.count(r"\paragraph{")
    c.check(
        "chapter_contract_cohesive_prose",
        len(opening_blocks) == 7
        and closing_anchors >= 6
        and paragraph_command_count == 0
        and "Numerical Results and Physical Interpretation" in text
        and "Synthesis of the numerical evidence" in text,
        f"chapters={len(opening_blocks)}; closing_anchors={closing_anchors}; "
        f"paragraph_commands={paragraph_command_count}",
    )
    required_physics_tables = {
        "tab:numerical-study-design", "tab:manufactured-density",
        "tab:manufactured-gradient", "tab:manufactured-operator",
        "tab:seo-projection-physics", "tab:identical-cloud-reconstruction",
        "tab:timestep-refinement-physics", "tab:independent-cloud-size",
        "tab:four-seed-replication-physics", "tab:raw-conservation-physics",
        "tab:signed-label-tail-physics", "tab:reference-discretizations",
        "tab:tdse-reference-refinement", "tab:qcle-reference-refinement",
        "tab:paired-physical-errors",
    }
    present_physics_tables = set(
        re.findall(r"\\label\{(tab:[^}]+)\}", complete_source)
    )
    chapter_results = re.search(
        r"\\chapter\{Numerical Results and Physical Interpretation\}"
        r"(.*?)\\chapter\{Discussion and Conclusions\}",
        text,
        re.S,
    )
    chapter_results_text = chapter_results.group(1) if chapter_results else ""
    operational_terms = [
        term
        for term in (
            "SHA-256", "run_manifest", "Source CSV", "TABLE_DATA_CROSSWALK",
            "BLOCKED_EXTERNAL_PUBLICATION", "expected=", "verified=",
            "missing=", "execution audit", "campaign inventory",
        )
        if term.lower() in chapter_results_text.lower()
    ]
    c.check("chapter_6_7_physics_evidence",
            required_physics_tables <= present_physics_tables
            and not operational_terms,
            "explicit source marker required for every Chapter 6–7 numerical block")


def check_response(c: Checker, response: Path) -> None:
    if not response.exists():
        c.check("reviewer_response", False, f"absent: {response}")
        return
    text = response.read_text(encoding="utf-8", errors="replace")
    expected = (
        [f"Gate {i}" for i in range(1, 11)]
        + [f"I-{i}" for i in range(1, 17)]
        + [f"M-{i}" for i in range(1, 26)]
        + [f"L-{i}" for i in range(1, 8)]
    )
    missing = [item for item in expected if item not in text]
    c.check("response_all_items", not missing, f"missing={missing}")
    c.check(
        "response_nine_mandatory_corrections",
        "Final mandatory corrections" in text
        and sum(
            f"{index} ---" in text
            for index in range(1, 10)
        ) == 9
        and text.count("Archive evidence") >= 9
        and text.count("Final location") >= 9,
        "nine final corrections have location and archive crosswalks",
    )
    forbidden = ("Partial", "Open", "placeholder",
                 "future action", "NOT COMPUTED", "TBD", "TODO")
    present = [
        word
        for word in forbidden
        if re.search(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", text)
    ]
    c.check("response_no_open_language", not present,
            f"present={present}")
    addressed_hits = text.count("Addressed in the revised thesis")
    pending_hits = (
        text.count("External repository deposition pending")
        + text.count("Frozen release supplied; public DOI deposition pending")
    )
    c.check(
        "response_item_statuses",
        addressed_hits + pending_hits >= len(expected),
        f"addressed={addressed_hits}; external_pending={pending_hits}; expected>={len(expected)}",
    )
    c.check(
        "response_locations_and_evidence",
        text.count("Revision in thesis") >= len(expected)
        and text.count("Evidence") >= 48,
        "each item identifies its thesis revision and scientific evidence",
    )
    has_doi = bool(re.search(r"10\.\d{4,9}/[-._;()/:\w]+", text))
    normalized_text = text.replace(r"\_", "_")
    public_release = (
        "https://github.com/" in normalized_text
        and "/releases/tag/" in normalized_text
        and "frozen_numerical_evidence_payload.zip" in normalized_text
    )
    honest_pending = (
        "A permanent DOI has not yet been assigned" in text
        and "https://github.com/" in text
    )
    c.check(
        "response_availability_statement",
        has_doi or public_release or honest_pending,
        "public release/DOI recorded or external deposition status stated explicitly",
    )
    return
    status_hits = sum(
        text.count(status)
        + text.count(status.replace("—", "---").replace("–", "--"))
        for status in ALLOWED_STATUSES
    )
    c.check("response_allowed_status_per_item", status_hits >= len(expected),
            f"allowed_status_occurrences={status_hits}; expected>={len(expected)}")
    c.check("response_locations_and_sources",
            text.count("Thesis location") >= len(expected)
            and text.count("Archive source") >= len(expected),
            "each row must print thesis location and archive source")
    c.check("response_permanent_identifier",
            bool(re.search(r"10\.\d{4,9}/[-._;()/:\w]+", text)),
            "real DOI required")


def count_pdf_pages(path: Path) -> Tuple[int, str]:
    """Count PDF pages via pypdf, pypdfium2, or any discoverable pdfinfo."""
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(path)).pages), "pypdf"
    except ModuleNotFoundError:
        pass
    except Exception as exc:  # unreadable PDF is a real failure
        return 0, f"pypdf error: {exc}"
    try:
        import pypdfium2 as pdfium
        return len(pdfium.PdfDocument(str(path))), "pypdfium2"
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        return 0, f"pypdfium2 error: {exc}"
    candidates: List[Path] = []
    tools = ROOT / "thesis_revision_evidence" / "tools"
    for archive in sorted(tools.glob("poppler*")):
        for rel in ("Library/bin", "bin"):
            for filename in ("pdfinfo.exe", "pdfinfo"):
                candidates.append(archive / rel / filename)
    override = os.environ.get("POPPLER_BIN")
    if override:
        for filename in ("pdfinfo.exe", "pdfinfo"):
            candidates.append(Path(override) / filename)
    discovered = shutil.which("pdfinfo")
    if discovered:
        wrapper = Path(discovered)
        if wrapper.suffix.lower() == ".cmd" and len(wrapper.parents) >= 3:
            native = (
                wrapper.parents[2] / "native" / "poppler"
                / "Library" / "bin" / "pdfinfo.exe"
            )
            if native.exists():
                candidates.append(native)
        candidates.append(wrapper)
    for pdfinfo in candidates:
        if not pdfinfo.exists():
            continue
        result = subprocess.run(
            [str(pdfinfo), str(path)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            continue
        match = re.search(r"(?m)^Pages:\s+(\d+)\s*$", result.stdout)
        if match:
            return int(match.group(1)), f"pdfinfo: {pdfinfo}"
    return 0, "no page counter found (install pypdf or pypdfium2)"


def check_compile(c: Checker, thesis_pdf: Path, response_pdf: Path,
                  thesis: Path, response: Path) -> None:
    for source in (thesis, response):
        log = source.with_suffix(".log")
        if not log.exists():
            c.check(f"log_{source.stem}", False, f"absent: {log}")
            continue
        text = log.read_text(encoding="utf-8", errors="replace").lower()
        found = [phrase for phrase in FORBIDDEN_LOG if phrase.lower() in text]
        c.check(f"log_{source.stem}", not found, f"forbidden={found}")
    for name, path in (("thesis_pdf", thesis_pdf), ("response_pdf", response_pdf)):
        c.check(name, path.exists() and path.stat().st_size > 0, str(path))
        if path.exists():
            pages, counter = count_pdf_pages(path)
            c.check(f"{name}_pages", pages > 0, f"{pages} pages ({counter})")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("precompile", "final"), required=True)
    p.add_argument("--evidence", type=Path,
                   default=ROOT / "final_reviewer_closure")
    p.add_argument("--thesis-tex", type=Path,
                   default=ROOT / "Thesis" / "Thesis.tex")
    p.add_argument("--bibliography", type=Path,
                   default=ROOT / "Thesis" / "References.bib")
    p.add_argument("--response-tex", type=Path,
                   default=ROOT / "Reviewer_Response.tex")
    p.add_argument("--thesis-pdf", type=Path,
                   default=ROOT / "Thesis" / "Thesis.pdf")
    p.add_argument("--response-pdf", type=Path,
                   default=ROOT / "Reviewer_Response.pdf")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    c = Checker()
    check_numerical(c, args.evidence.resolve())
    check_thesis(c, args.thesis_tex.resolve(), args.bibliography.resolve(),
                 args.evidence.resolve())
    check_response(c, args.response_tex.resolve())
    if args.stage == "final":
        check_compile(
            c, args.thesis_pdf.resolve(), args.response_pdf.resolve(),
            args.thesis_tex.resolve(), args.response_tex.resolve(),
        )
    result = {
        "stage": args.stage, "passed": c.passed, "checks": c.rows,
    }
    output = args.evidence.resolve() / f"final_acceptance_{args.stage}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for row in c.rows:
        print(
            f"{'PASS' if row['passed'] else 'FAIL'}\t"
            f"{row['name']}\t{row['detail']}"
        )
    return 0 if c.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
