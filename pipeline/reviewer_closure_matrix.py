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
reviewer_closure_matrix.py
==========================

Closure matrix and evidence extraction for the *Revised MSc Thesis Review*
(28 July 2026), which is the scientific requirements authority.

The review defines 48 numbered items in section 8 -- I-1..I-16 (results that
cannot be audited uniquely), M-1..M-25 (ambiguous or misleading claims), and
L-1..L-7 (terminology) -- plus ten non-negotiable acceptance gates in section 9.

Two facts drive the design of this module:

1. **Most items are not new computation.** They are text corrections, or tables
   that can be extracted from data already on disk. Distinguishing these from
   the genuinely missing evidence is the single highest-value thing the
   pipeline can do, because it separates a week of writing from a month of
   compute.

2. **The pipeline cannot close a text item.** A statement is corrected by
   editing the thesis, not by running code. This module therefore reports
   *evidence readiness* per item and never claims a thesis item is "resolved".

Subcommands
-----------
``tables``   extract the evidence tables the review demands (I-1, I-3, I-14)
``matrix``   resolve all 48 items against artifacts on disk
``--self-test``  validate the pure logic

Torch-free, NumPy-free. Reads only JSON already produced by the pipeline.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

EPS = sys.float_info.epsilon
ROUNDOFF_FACTOR = 100.0

# Terminal evidence states. "PARTIAL"/"OPEN"/"UNKNOWN" are deliberately absent.
READY = "EVIDENCE READY"
EXTRACTABLE = "EVIDENCE ON DISK - EXTRACTION REQUIRED"
MISSING = "EVIDENCE MISSING - COMPUTATION REQUIRED"
TEXT_ONLY = "NO EVIDENCE REQUIRED - THESIS EDIT"
ESCAPE = "CLOSABLE BY REVIEWER-SANCTIONED NARROWING"
ARCHIVAL = "ARCHIVAL ACTION REQUIRED"


# ===========================================================================
# Order estimation with the roundoff guard the acceptance contract requires
# ===========================================================================

def guarded_observed_order(coarse: float, fine: float, finer: float,
                           scale: Optional[float] = None
                           ) -> Tuple[Optional[float], str]:
    """
    p_obs = log2(|u_h - u_h/2| / |u_h/2 - u_h/4|), suppressed when either
    difference is at roundoff.

    The review (I-3) asks that the negative outlier be retained as an
    *identified* row. It is identified here as a roundoff artifact rather than
    silently dropped or reported as a negative convergence order.
    """
    for v in (coarse, fine, finer):
        if not math.isfinite(v):
            return None, "REJECTED: nonfinite level value"
    a = abs(coarse - fine)
    b = abs(fine - finer)
    if scale is None:
        scale = max(abs(coarse), abs(fine), abs(finer), 1.0)
    floor = ROUNDOFF_FACTOR * EPS * scale
    if b <= floor:
        return None, (f"REJECTED: fine-to-finer difference {b:.3e} at or below "
                      f"roundoff floor {floor:.3e}")
    if a <= floor:
        return None, (f"REJECTED: coarse-to-fine difference {a:.3e} at or below "
                      f"roundoff floor {floor:.3e}")
    return math.log2(a / b), "ok"


# ===========================================================================
# I-1: complete manufactured-operator table
# ===========================================================================

QUANTITIES = ("density", "gradient", "operator_Q")
METRICS = ("rmse", "relative_l2", "linf")


def extract_manufactured_table(root: Path) -> Tuple[List[Dict[str, Any]],
                                                    List[Dict[str, Any]]]:
    """
    Every seed, every N, on- and off-support density/gradient/operator errors.

    Returns (rows, summary). Duplicate (N, seed) pairs across campaign
    directories are deduplicated on identical content and flagged otherwise --
    the review's central complaint is that numbers cannot be uniquely
    identified, so a silent overwrite would reproduce the defect.
    """
    seen: Dict[Tuple[int, int], Tuple[Path, Dict[str, Any]]] = {}
    conflicts: List[str] = []
    for p in sorted(Path(root).rglob("manufactured_operator_metrics.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        n, s = obj.get("n_train"), obj.get("seed")
        if n is None or s is None:
            continue
        key = (int(n), int(s))
        if key in seen and seen[key][1].get("metrics") != obj.get("metrics"):
            conflicts.append(f"N={n} seed={s}: {seen[key][0]} vs {p}")
        seen.setdefault(key, (p, obj))

    rows: List[Dict[str, Any]] = []
    for (n, s), (p, obj) in sorted(seen.items()):
        m = obj.get("metrics", {})
        for support in ("on_support", "off_support"):
            block = m.get(support, {})
            for q in QUANTITIES:
                vals = block.get(q, {})
                rows.append({
                    "N": n, "seed": s, "support": support, "quantity": q,
                    "rmse": vals.get("rmse"),
                    "relative_l2": vals.get("relative_l2"),
                    "linf": vals.get("linf"),
                    "n_query": obj.get("n_query"),
                    "source": str(p.relative_to(root)),
                })

    # per-(N, support, quantity) seed statistics -- the means and spreads the
    # review says are asserted in the text but never shown
    summary: List[Dict[str, Any]] = []
    groups: Dict[Tuple[int, str, str], List[Tuple[int, float]]] = {}
    for r in rows:
        v = r["relative_l2"]
        if isinstance(v, (int, float)) and math.isfinite(v):
            groups.setdefault((r["N"], r["support"], r["quantity"]), []).append(
                (r["seed"], float(v)))
    for (n, support, q), pairs in sorted(groups.items()):
        vals = [v for _, v in sorted(pairs)]
        summary.append({
            "N": n, "support": support, "quantity": q,
            "n_seeds": len(vals),
            "seeds": ";".join(str(s) for s, _ in sorted(pairs)),
            "mean_relative_l2": statistics.fmean(vals),
            "sd_relative_l2": (statistics.stdev(vals) if len(vals) > 1 else 0.0),
            "min_relative_l2": min(vals), "max_relative_l2": max(vals),
        })
    if conflicts:
        summary.append({"N": -1, "support": "CONFLICT", "quantity": "",
                        "n_seeds": len(conflicts), "seeds": "",
                        "mean_relative_l2": None, "sd_relative_l2": None,
                        "min_relative_l2": None, "max_relative_l2": None,
                        "note": " | ".join(conflicts)})
    return rows, summary


# ===========================================================================
# I-3 / I-14: fully identified reference-convergence table with exact settings
# ===========================================================================

def extract_reference_table(root: Path) -> List[Dict[str, Any]]:
    """
    One row per (method, momentum, refinement mode, observable), carrying the
    three level values, both differences, the guarded order, the rejection
    reason, and the exact numerical settings (grid, dt, t_final, boundary).

    This is what the review demands in I-3 ("identify every order by method,
    momentum, refinement mode, numerical levels, domain, resolved step,
    observable, and successive errors") and I-14 ("include the exact settings
    in Appendix F").
    """
    rows: List[Dict[str, Any]] = []
    for p in sorted(Path(root).rglob("reference_convergence_3level.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key, blk in doc.items():
            if not isinstance(blk, dict):
                continue
            levels = {lv.get("label"): lv for lv in blk.get("levels", [])
                      if isinstance(lv, dict)}
            meta = {lab: (lv.get("metadata") or {}) for lab, lv in levels.items()}
            momentum = ""
            for token in key.replace("-", "_").split("_"):
                if token.startswith("P0") and token[2:].isdigit():
                    momentum = token[2:]
            for obs, o in (blk.get("observables") or {}).items():
                if not isinstance(o, dict):
                    continue
                c, f, fr = o.get("coarse"), o.get("fine"), o.get("finer")
                p_guarded, reason = (None, "REJECTED: missing level value")
                if all(isinstance(x, (int, float)) for x in (c, f, fr)):
                    p_guarded, reason = guarded_observed_order(
                        float(c), float(f), float(fr))
                rows.append({
                    "key": key,
                    "method": blk.get("method", ""),
                    "momentum_P0": momentum,
                    "refine_mode": blk.get("refine_mode", ""),
                    "observable": obs,
                    "coarse": c, "fine": f, "finer": fr,
                    "diff_coarse_fine": o.get("abs_diff_coarse_fine"),
                    "diff_fine_finer": o.get("abs_diff_fine_finer"),
                    "p_obs_reported": o.get("p_obs"),
                    "p_obs_guarded": p_guarded,
                    "guard_verdict": reason,
                    "n_grid_coarse": levels.get("coarse", {}).get("n_grid"),
                    "n_grid_fine": levels.get("fine", {}).get("n_grid"),
                    "n_grid_finer": levels.get("finer", {}).get("n_grid"),
                    "dt_coarse": levels.get("coarse", {}).get("dt"),
                    "dt_fine": levels.get("fine", {}).get("dt"),
                    "dt_finer": levels.get("finer", {}).get("dt"),
                    "t_final": meta.get("coarse", {}).get("t_final"),
                    "boundary_rule": meta.get("coarse", {}).get("boundary_rule"),
                    "source": str(p.relative_to(root)),
                })
    return rows


# ===========================================================================
# The 48 review items
# ===========================================================================

@dataclass(frozen=True)
class Item:
    id: str
    gate: int                       # acceptance gate 1..10 from section 9, 0 = none
    kind: str                       # TEXT | EXTRACT | COMPUTE | REGENERATE | ARCHIVE
    title: str
    thesis_loc: str
    needs: Tuple[str, ...] = ()     # glob patterns that must resolve
    escape: str = ""                # reviewer-sanctioned narrowing, if offered
    action: str = ""


def review_items() -> List[Item]:
    """The 48 numbered items of section 8, with the section-9 gate they feed."""
    I = [
        Item("I-1", 4, "EXTRACT", "Complete three-seed manufactured result not shown",
             "pp. 62, 87-88; Table 6.4",
             ("**/manufactured_operator_metrics.json",),
             action="Emit every seed x N x on/off-support table with means and spreads."),
        Item("I-2", 5, "EXTRACT", "Time-step order conclusion cannot be reconstructed",
             "p. 80, pp. 87-89; Tables 6.3, 6.6",
             ("**/step7_dt_P0*/seed*_dt*/run_manifest.json",),
             action="One row per method/momentum/seed/observable/step triplet with "
                    "both differences, guarded order, rejection reason, seed spread."),
        Item("I-3", 5, "EXTRACT", "Reference-convergence orders are unassigned",
             "p. 89, Table 6.8; App. F pp. 184-185",
             ("**/reference_convergence_3level.json",),
             action="Identified table; the -3.391 outlier is the TDSE P0=20 'both' "
                    "trace row and its differences are at roundoff."),
        Item("I-4", 5, "EXTRACT", "MIDPOINT central value at P0=20 absent",
             "p. 89, Table 6.6",
             ("**/step9_repl_P0*/seed*/run_manifest.json",),
             action="Report all four seed values, mean and SD for every estimator."),
        Item("I-5", 7, "EXTRACT", "Identical-support pass incompletely reported",
             "pp. 87-88; Table 6.5",
             ("**/kde_gp_identical_support.json",),
             action="Report E1, E2, Einf at both momenta, or state which were not evaluated."),
        Item("I-6", 3, "REGENERATE", "41 retained figures have no complete provenance",
             "pp. 83-155; Figs. 6.1-6.41",
             ("**/figure_catalog.json",),
             action="SUBMISSION BLOCKER. Regenerate each retained scientific figure "
                    "from a verified manifest, or remove it."),
        Item("I-7", 3, "TEXT", "Figure-dependent descriptions not interpretable",
             "pp. 92-148", escape="Restrict text to neutral description; do not use as evidence.",
             action="Attach a defined metric and number to each retained comparison."),
        Item("I-8", 3, "TEXT", "Plotting thresholds undefined",
             "pp. 109-117; Figs. 6.13-6.18",
             action="State the numerical threshold and rule in every caption; show insensitivity."),
        Item("I-9", 0, "COMPUTE", "Diagnostics divide by undocumented |y0| tail",
             "p. 73, pp. 133-144",
             ("**/y0_tail_audit.json",),
             escape="Qualify every multiplier and cloud-weight claim as potentially tail-sensitive.",
             action="Report |y0| distribution and minimum, truncation rule, affected "
                    "fraction of points and mass, threshold sensitivity."),
        Item("I-10", 0, "EXTRACT", "'Applied source after stabilization' unspecified",
             "pp. 129-131; Fig. 6.29",
             ("**/source_application_contract.json",),
             action="Read the production branch and state the exact formula, or state "
                    "unambiguously that applied source equals raw source."),
        Item("I-11", 0, "EXTRACT", "'Physically implausible' analytic observables not reported",
             "pp. 147-148",
             ("**/step9_repl_P0*/seed*/run_manifest.json",),
             escape="Remove the claim.",
             action="Table of raw and normalized analytic-GP and cloud estimates vs references."),
        Item("I-12", 0, "EXTRACT", "'Only weakly' undefined and conflicts with Table 6.6",
             "p. 147",
             ("**/step9_repl_P0*/seed*/run_manifest.json",),
             action="Replace the global statement with per-estimator, per-momentum numbers."),
        Item("I-13", 7, "COMPUTE", "High-momentum agreement not quantitative",
             "pp. 147-148",
             ("**/physical_reference_errors.json",),
             escape="Write only that no validated improvement over PBME was demonstrated.",
             action="Seed-resolved L1/L2 field errors and observable errors, or take the escape."),
        Item("I-14", 8, "EXTRACT", "Exact reference settings outside the thesis",
             "App. F pp. 184-185",
             ("**/reference_convergence_3level.json",),
             action="Reproduce grid, domain, dt, t_final, boundary rule in Appendix F."),
        Item("I-15", 8, "ARCHIVE", "Immutable configuration / versioned record not identified",
             "p. 87; App. F p. 184",
             ("**/frozen_release.json",),
             action="Freeze a release, record commit/tag/DOI and SHA-256 of every artifact."),
        Item("I-16", 9, "TEXT", "Thesis title not uniquely identified across artifact",
             "title page vs PDF metadata",
             action="One exact title in title page, PDF metadata, repository, response, paperwork."),
    ]

    m_specs: List[Tuple[str, int, str, str, str, str]] = [
        ("M-1", 2, "TEXT", "Contribution not separated from known mathematical fact", "p. 1",
         "State the application-specific chain, not the generic identifiability fact."),
        ("M-2", 0, "TEXT", "'Projected density is represented' is misleading", "pp. 48-50, 149",
         "Target is projected; the tested surrogate does not remain in its image."),
        ("M-3", 0, "TEXT", "'Semi-discretization of the complete generator' too strong", "pp. 149-150",
         "Call it an attempted moving-cloud collocation of both formal generator terms."),
        ("M-4", 0, "TEXT", "'Corrected method' implies correctness", "throughout",
         "Use 'MIDPOINT prototype' / 'tested excess-update branch'."),
        ("M-5", 0, "TEXT", "'Four checks form a controlled argument' overstates", "p. 87",
         "Say the checks evidence failure under tested settings; list what stays unaudited."),
        ("M-6", 6, "TEXT", "'Support refinement/convergence' used too freely", "pp. 62, 80, 87-88, 155, 158",
         "Call it an independent-cloud enlargement study; state convergence untested."),
        ("M-7", 0, "TEXT", "l2=0.01 control does not globally rule out regularization", "pp. 87, 151, 158",
         "State only: 0.05 -> 0.01 did not remove instability for P0=100, seed 11."),
        ("M-8", 5, "TEXT", "Formal second order is not demonstrated order", "pp. 74, 80-81, 152, 158",
         "State assumptions unverified and positive production order not observed."),
        ("M-9", 3, "TEXT", "'Qualitative visualization' does not admit untraceable images", "pp. 104-148",
         "Regenerate/trace, or remove images and all figure-dependent conclusions."),
        ("M-10", 0, "TEXT", "Independent shape normalization conceals the failure", "pp. 105-109",
         "Label conclusions shape-only; show raw integral beside each."),
        ("M-11", 0, "TEXT", "'Less than ~1% per step' reads as a pointwise bound", "p. 130",
         "State it is a ratio of global RMS norms and bounds nothing pointwise."),
        ("M-12", 0, "COMPUTE", "'Not confined to negligible-label tails' unsupported", "p. 132",
         "Supply quantile-resolved source contributions and tail-threshold sensitivity."),
        ("M-13", 0, "TEXT", "Self-normalized GP normalization too prominent", "pp. 144-146",
         "Label as tautological; foreground the raw integral and GP-cloud discrepancy."),
        ("M-14", 0, "TEXT", "No explicit estimator hierarchy", "pp. 145-148",
         "Define each estimator once with validity domain and admissibility order."),
        ("M-15", 0, "TEXT", "'Anchor-cloud estimator is more defensible' too strong", "p. 148",
         "Neither estimator is validated in the low-ESS regime."),
        ("M-16", 7, "TEXT", "Undefined proximity language for the reference benchmark", "pp. 92-104, 150",
         "Define by numerical field and observable errors with thresholds."),
        ("M-17", 0, "TEXT", "Causation attributed to the excess update, not the discretization", "pp. 151, 158",
         "Attribute to the tested nonprojected, nonconservative MIDPOINT discretization."),
        ("M-18", 7, "TEXT", "'PBME reconstruction is faithful' broader than the gate", "p. 158",
         "State exactly that the declared E1 gate passed on the tested support."),
        ("M-19", 0, "TEXT", "Failure attribution not fully isolated", "p. 158",
         "Describe as an evidence-supported pathway; causality not uniquely apportioned."),
        ("M-20", 8, "TEXT", "'Reproducible basis' contradicts archive defects", "p. 158",
         "Use 'documented but presently incomplete basis for future development'."),
        ("M-21", 9, "TEXT", "Objective broader than the tested construction", "p. 25",
         "Name the single product-GP/moving-cloud/MIDPOINT construction and its 1D two-state test."),
        ("M-22", 1, "TEXT", "Chapter 6 opens with the wrong decision question", "p. 82",
         "Open with the four Chapter 1 acceptance criteria and the negative answer."),
        ("M-23", 0, "TEXT", "'Full-density representation' ambiguous", "pp. 83, 87",
         "Define at first use as a software architecture; rename to avoid implying exactness."),
        ("M-24", 4, "COMPUTE", "'Selected l2 = 0.01' needs qualification", "pp. 83, 87, 89, App. F",
         "Call it pilot-selected; repeat the operator test at production 0.05 and pilot 0.01."),
        ("M-25", 5, "TEXT", "Completion counts confused with validation", "p. 88, Table 6.3",
         "Separate run inventory completed from scientific criterion passed."),
    ]
    M = [Item(i, g, k, t, loc, action=act) for i, g, k, t, loc, act in m_specs]

    l_specs = [
        ("L-1", "Use one method vocabulary", "throughout"),
        ("L-2", "Define every estimator at first use and map it to one equation", "throughout"),
        ("L-3", "Reserve evidentiary words for declared standards", "throughout"),
        ("L-4", "Correct compound-word and hyphenation defects", "pp. 52, 59"),
        ("L-5", "Replace vague antecedents", "throughout"),
        ("L-6", "Compress repetitive figure narration", "Secs. 6.4-6.10"),
        ("L-7", "Make the response document an audit trail", "response letter"),
    ]
    L = [Item(i, 10 if i == "L-7" else 0, "TEXT", t, loc) for i, t, loc in l_specs]

    items = I + M + L
    assert len(items) == 48, f"expected 48 review items, built {len(items)}"
    return items


# ===========================================================================
# Resolution
# ===========================================================================

def resolve_item(item: Item, root: Path) -> Dict[str, Any]:
    found: List[str] = []
    for pat in item.needs:
        hits = list(Path(root).glob(pat))
        if hits:
            found.append(f"{pat} -> {len(hits)} file(s)")
    have_all = bool(item.needs) and len(found) == len(item.needs)

    if item.kind == "TEXT":
        status = TEXT_ONLY
    elif item.kind == "ARCHIVE":
        status = READY if have_all else ARCHIVAL
    elif item.kind == "REGENERATE":
        status = MISSING if not have_all else EXTRACTABLE
    elif item.kind == "EXTRACT":
        status = EXTRACTABLE if have_all else MISSING
    else:  # COMPUTE
        status = READY if have_all else (ESCAPE if item.escape else MISSING)

    return {
        "item": item.id,
        "gate": item.gate or "",
        "kind": item.kind,
        "title": item.title,
        "thesis_location": item.thesis_loc,
        "evidence_status": status,
        "evidence_found": "; ".join(found),
        "escape_hatch": item.escape,
        "action": item.action,
    }


def resolve_matrix(root: Path) -> List[Dict[str, Any]]:
    return [resolve_item(it, Path(root)) for it in review_items()]


def matrix_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tally: Dict[str, int] = {}
    for r in rows:
        tally[r["evidence_status"]] = tally.get(r["evidence_status"], 0) + 1
    blocking = [r["item"] for r in rows if r["evidence_status"] == MISSING]
    return {"total_items": len(rows), "by_status": tally,
            "computation_blocked_items": blocking,
            "n_text_only": tally.get(TEXT_ONLY, 0)}


# ===========================================================================
# I/O
# ===========================================================================

def write_csv(path: Path, rows: Sequence[Dict[str, Any]],
              columns: Optional[Sequence[str]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: List[str] = list(columns) if columns else []
    if not cols:
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path


def run_tables(root: Path, out: Path) -> Dict[str, Any]:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)

    rows, summary = extract_manufactured_table(root)
    write_csv(out / "I1_manufactured_complete.csv", rows)
    write_csv(out / "I1_manufactured_seed_statistics.csv", summary)

    ref = extract_reference_table(root)
    write_csv(out / "I3_I14_reference_convergence_identified.csv", ref)

    suppressed = [r for r in ref if r["p_obs_guarded"] is None]
    info = {
        "manufactured_rows": len(rows),
        "manufactured_groups": len(summary),
        "reference_rows": len(ref),
        "reference_rows_suppressed_by_roundoff_guard": len(suppressed),
        "suppressed_detail": [
            {"key": r["key"], "observable": r["observable"],
             "p_obs_reported": r["p_obs_reported"],
             "verdict": r["guard_verdict"]} for r in suppressed],
    }
    (out / "tables_summary.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8")

    print(f"[tables] manufactured rows        : {len(rows)}")
    print(f"[tables] manufactured seed groups : {len(summary)}")
    print(f"[tables] reference rows           : {len(ref)}")
    print(f"[tables] suppressed by guard      : {len(suppressed)}")
    for s in suppressed:
        print(f"    {s['key']}/{s['observable']}: reported p_obs="
              f"{s['p_obs_reported']}, {s['guard_verdict']}")
    return info


def run_matrix(root: Path, out: Path) -> Dict[str, Any]:
    rows = resolve_matrix(root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "closure_matrix.csv", rows,
              ["item", "gate", "kind", "title", "thesis_location",
               "evidence_status", "evidence_found", "escape_hatch", "action"])
    s = matrix_summary(rows)
    (out / "closure_matrix_summary.json").write_text(
        json.dumps(s, indent=2), encoding="utf-8")

    print(f"[matrix] items: {s['total_items']}")
    for k, v in sorted(s["by_status"].items(), key=lambda kv: -kv[1]):
        print(f"    {v:3d}  {k}")
    if s["computation_blocked_items"]:
        print("[matrix] computation required for: "
              + ", ".join(s["computation_blocked_items"]))
    return s


# ===========================================================================
# Self-test
# ===========================================================================

def run_self_test() -> None:
    import tempfile

    # -- roundoff guard ------------------------------------------------
    p, why = guarded_observed_order(1.0, 0.5, 0.25)
    assert why == "ok" and abs(p - 1.0) < 1e-12, (p, why)
    p, why = guarded_observed_order(1.0, 0.25, 0.0625)
    assert abs(p - 2.0) < 1e-12
    # the real trace row: differences 1.82e-14 -> 1.91e-13 on a value near 1
    p, why = guarded_observed_order(1.0000000000000675, 1.0000000000000857,
                                    1.0000000000002767)
    assert p is None and "roundoff" in why, (p, why)
    # the real P0 row must survive: 8.04e-12 -> 2.20e-12 on a value near 1
    p, why = guarded_observed_order(0.9999787335391701, 0.9999787335472097,
                                    0.9999787335494058)
    assert p is not None and abs(p - 1.8721532832054133) < 1e-9, (p, why)
    assert guarded_observed_order(float("nan"), 1.0, 2.0)[0] is None
    # exactly-equal fine levels must be rejected, not divide by zero
    assert guarded_observed_order(1.0, 2.0, 2.0)[0] is None

    # -- item register -------------------------------------------------
    items = review_items()
    ids = [i.id for i in items]
    assert len(set(ids)) == 48
    assert sum(1 for i in ids if i.startswith("I-")) == 16
    assert sum(1 for i in ids if i.startswith("M-")) == 25
    assert sum(1 for i in ids if i.startswith("L-")) == 7
    assert all(1 <= i.gate <= 10 or i.gate == 0 for i in items)
    # every acceptance gate must be reachable from at least one item
    covered = {i.gate for i in items if i.gate}
    assert covered >= {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, sorted(covered)
    assert all(i.kind in ("TEXT", "EXTRACT", "COMPUTE", "REGENERATE", "ARCHIVE")
               for i in items)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "r"
        (root / "a" / "N300_seed123").mkdir(parents=True)
        (root / "a" / "N300_seed124").mkdir(parents=True)

        def man(n, s, val):
            return {"n_train": n, "seed": s, "n_query": 1000, "metrics": {
                "on_support": {q: {"rmse": val, "relative_l2": val,
                                   "linf": val} for q in QUANTITIES},
                "off_support": {q: {"rmse": 2 * val, "relative_l2": 2 * val,
                                    "linf": 2 * val} for q in QUANTITIES}}}

        (root / "a" / "N300_seed123" / "manufactured_operator_metrics.json"
         ).write_text(json.dumps(man(300, 123, 0.02)), encoding="utf-8")
        (root / "a" / "N300_seed124" / "manufactured_operator_metrics.json"
         ).write_text(json.dumps(man(300, 124, 0.03)), encoding="utf-8")

        rows, summary = extract_manufactured_table(root)
        # 2 runs x 2 supports x 3 quantities
        assert len(rows) == 12, len(rows)
        g = next(s for s in summary if s["N"] == 300
                 and s["support"] == "on_support"
                 and s["quantity"] == "operator_Q")
        assert g["n_seeds"] == 2 and g["seeds"] == "123;124"
        assert abs(g["mean_relative_l2"] - 0.025) < 1e-12
        assert abs(g["sd_relative_l2"] - statistics.stdev([0.02, 0.03])) < 1e-12
        off = next(s for s in summary if s["N"] == 300
                   and s["support"] == "off_support"
                   and s["quantity"] == "operator_Q")
        assert abs(off["mean_relative_l2"] - 0.05) < 1e-12

        # duplicate with identical content must NOT be flagged
        (root / "b" / "N300_seed123").mkdir(parents=True)
        (root / "b" / "N300_seed123" / "manufactured_operator_metrics.json"
         ).write_text(json.dumps(man(300, 123, 0.02)), encoding="utf-8")
        _, s2 = extract_manufactured_table(root)
        assert not any(x["support"] == "CONFLICT" for x in s2)
        # duplicate with different content MUST be flagged
        (root / "c" / "N300_seed123").mkdir(parents=True)
        (root / "c" / "N300_seed123" / "manufactured_operator_metrics.json"
         ).write_text(json.dumps(man(300, 123, 0.09)), encoding="utf-8")
        _, s3 = extract_manufactured_table(root)
        assert any(x["support"] == "CONFLICT" for x in s3), \
            "conflicting duplicate metrics were silently merged"

        # -- reference extraction --------------------------------------
        (root / "ref").mkdir()
        (root / "ref" / "reference_convergence_3level.json").write_text(
            json.dumps({"tdse_P020_both": {
                "method": "tdse", "refine_mode": "both",
                "levels": [
                    {"label": "coarse", "dt": 0.2, "n_grid": 2048,
                     "metadata": {"t_final": 40.0, "boundary_rule": "periodic"}},
                    {"label": "fine", "dt": 0.1, "n_grid": 4096,
                     "metadata": {"t_final": 40.0}},
                    {"label": "finer", "dt": 0.05, "n_grid": 8192,
                     "metadata": {"t_final": 40.0}}],
                "observables": {
                    "P0": {"coarse": 0.9999787335391701,
                           "fine": 0.9999787335472097,
                           "finer": 0.9999787335494058, "p_obs": 1.8721532832},
                    "trace": {"coarse": 1.0000000000000675,
                              "fine": 1.0000000000000857,
                              "finer": 1.0000000000002767,
                              "p_obs": -3.3906408449713763}}}}),
            encoding="utf-8")
        ref = extract_reference_table(root)
        assert len(ref) == 2
        by_obs = {r["observable"]: r for r in ref}
        assert by_obs["P0"]["momentum_P0"] == "20", by_obs["P0"]["momentum_P0"]
        assert by_obs["P0"]["method"] == "tdse"
        assert by_obs["P0"]["n_grid_finer"] == 8192
        assert by_obs["P0"]["t_final"] == 40.0
        assert by_obs["P0"]["boundary_rule"] == "periodic"
        assert by_obs["P0"]["p_obs_guarded"] is not None
        # the -3.391 outlier must be identified and suppressed, not reported
        assert by_obs["trace"]["p_obs_reported"] == -3.3906408449713763
        assert by_obs["trace"]["p_obs_guarded"] is None
        assert "roundoff" in by_obs["trace"]["guard_verdict"]

        # -- matrix ----------------------------------------------------
        rows = resolve_matrix(root)
        assert len(rows) == 48
        bid = {r["item"]: r for r in rows}
        assert bid["I-1"]["evidence_status"] == EXTRACTABLE, bid["I-1"]
        assert bid["I-3"]["evidence_status"] == EXTRACTABLE
        assert bid["I-2"]["evidence_status"] == MISSING   # no dt runs in fixture
        assert bid["M-1"]["evidence_status"] == TEXT_ONLY
        assert bid["L-4"]["evidence_status"] == TEXT_ONLY
        assert bid["I-13"]["evidence_status"] == ESCAPE   # escape hatch offered
        assert bid["I-15"]["evidence_status"] == ARCHIVAL
        s = matrix_summary(rows)
        assert s["total_items"] == 48
        assert sum(s["by_status"].values()) == 48
        assert s["n_text_only"] >= 30, s["n_text_only"]

    print("[self-test] reviewer_closure_matrix checks passed "
          "(roundoff guard incl. real trace and P0 rows, 48-item register, "
          "gate coverage 1-10, manufactured aggregation and conflict "
          "detection, reference identification, matrix resolution).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=("tables", "matrix", "all"))
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("reviewer_closure_out"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        run_self_test(); return
    if a.command in ("tables", "all"):
        run_tables(a.root, a.out)
    if a.command in ("matrix", "all"):
        run_matrix(a.root, a.out)
    if not a.command:
        print("No command. Use --self-test, or: tables | matrix | all")


if __name__ == "__main__":
    main()
