"""Replace Appendix F's reference block from verified full-precision CSVs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
THESIS = ROOT / "Thesis" / "Thesis.tex"
TDSE = ROOT / "final_reviewer_closure" / "reference_tdse" / "tdse_three_level.csv"
QCLE = ROOT / "final_reviewer_closure" / "reference_grid_qcle" / "qcle_three_level.csv"
OUT = ROOT / "reviewer_data_audit" / "thesis_sources"
BEGIN = "% BEGIN GENERATED REFERENCE REFINEMENT APPENDIX"
END = "% END GENERATED REFERENCE REFINEMENT APPENDIX"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def unique_row(data: list[dict[str, str]], p0: float, mode: str) -> dict[str, str]:
    matches = [
        row
        for row in data
        if float(row["P0"]) == p0 and row["refinement_mode"] == mode
    ]
    if not matches:
        raise ValueError(f"No row for P0={p0:g}, mode={mode}")
    keys = [
        key
        for key in matches[0]
        if key.startswith("level") or key.startswith("declared_")
    ]
    for key in keys:
        if len({row.get(key, "") for row in matches}) != 1:
            raise ValueError(f"Inconsistent {key} for P0={p0:g}, mode={mode}")
    return matches[0]


def number(value: str, digits: int = 7) -> str:
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return r"\textsc{Not Computed}"


def integer(value: str) -> str:
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return r"\textsc{Not Computed}"


def ladder(row: dict[str, str], key: str, *, integer_values: bool = False) -> str:
    convert = integer if integer_values else number
    return "/".join(convert(row[f"level{index}_{key}"]) for index in (1, 2, 3))


def domain_ladder(row: dict[str, str], axis: str) -> str:
    return "/".join(
        f"[{number(row[f'level{index}_{axis}_min'])},"
        f"{number(row[f'level{index}_{axis}_max'])}]"
        for index in (1, 2, 3)
    )


def maximum(row: dict[str, str], key: str) -> str:
    return f"{max(float(row[f'level{index}_{key}']) for index in (1, 2, 3)):.6e}"


def build_block(
    tdse_rows: list[dict[str, str]], qcle_rows: list[dict[str, str]]
) -> str:
    lines = [
        BEGIN,
        r"\begin{landscape}",
        r"\begin{table}[p]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Exact, separate interaction-window reference-refinement "
        r"settings. Slash-separated entries list coarse/fine/finer values; "
        r"no domain or nominal level is collapsed. Observable endpoints, both "
        r"successive differences, guarded orders, and rejection reasons are "
        r"reported in Tables~\ref{tab:tdse-reference-refinement} and "
        r"\ref{tab:qcle-reference-refinement}.}",
        r"\label{apptab:reference-refinement-status}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lllrllllll}",
        r"\toprule",
        r"Method & $P_0$ & Mode & $t_{\rm final}$ & $\Delta t$ ladder & "
        r"Grid ladder & $R$ domain ladder & $P$ domain ladder & "
        r"accepted edge mass & finest CFL ratio \\",
        r"\midrule",
    ]
    for p0 in (20.0, 100.0):
        for mode in ("time", "grid"):
            row = unique_row(tdse_rows, p0, mode)
            lines.append(
                f"TDSE & {p0:g} & {mode} & {number(row['level1_t_final'])}"
                f" & {ladder(row, 'dt')}"
                f" & {ladder(row, 'n_grid_actual', integer_values=True)}"
                f" & {domain_ladder(row, 'R')} & ---"
                f" & {maximum(row, 'maximum_edge_mass_5pct')} & --- " + r"\\"
            )
    lines.append(r"\midrule")
    for p0 in (20.0, 100.0):
        for mode in ("time", "grid"):
            row = unique_row(qcle_rows, p0, mode)
            grids = "/".join(
                f"{integer(row[f'level{index}_n_R'])}"
                + r"$\times$"
                + f"{integer(row[f'level{index}_n_P'])}"
                for index in (1, 2, 3)
            )
            edge = max(
                float(row[f"level3_{key}"])
                for key in (
                    "maximum_edge_R_mass_5pct",
                    "maximum_edge_P_mass_5pct",
                )
            )
            lines.append(
                f"grid QCLE & {p0:g} & {mode} & {number(row['level1_t_final'])}"
                f" & {ladder(row, 'dt')} & {grids}"
                f" & {domain_ladder(row, 'R')} & {domain_ladder(row, 'P')}"
                f" & {edge:.6e} & {number(row['level3_cfl_ratio'])} " + r"\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            r"\end{landscape}",
            "",
            r"The TDSE and grid-QCLE studies use \(R_0=-15\) and propagate to "
            r"\(t_{\rm final}=2t_c\), so every case traverses the avoided-crossing "
            r"region. Time and grid refinement are separate: a time ladder holds "
            r"the finest grid fixed, while a grid ladder holds the finest time step "
            r"fixed. Both successive differences must exceed "
            r"\(\tau_{\rm noise}=10^{-12}+10^{-12}\max_k|O_k|\), must decrease "
            r"monotonically, and must give \(0<p\leq6\); other rows are retained "
            r"as roundoff- or saturation-limited, nonmonotone, or rapidly "
            r"contracting but nonasymptotic. Expected temporal scaling is "
            r"approximately second order for TDSE and fourth order for grid-QCLE "
            r"RK4; spatial or phase-space-grid behaviour is assessed separately. "
            r"The TDSE edge statistic is the maximum spatial-edge mass across the "
            r"three levels. The grid-QCLE statistic is the larger finest-grid "
            r"absolute physical-marginal edge fraction, with declared tolerance "
            r"\(10^{-3}\); the sign-indefinite phase-space \(\lvert W\rvert\) "
            r"ringing diagnostic is not substituted for this boundary gate. For "
            r"grid QCLE the dimensionless CFL ratio is "
            r"\(\Delta t/\Delta t_{\max}\), where \(\Delta t_{\max}\) is the "
            r"minimum of the RK4 imaginary-axis bound \(2.828\) divided by the "
            r"spectral advection, force, and electronic-frequency scales; ratios "
            r"below one pass this declared linear-stability check. TDSE is "
            r"model-exact only within its displayed numerical controls; grid QCLE "
            r"remains a numerical solution of the approximate QCLE.",
            END,
        ]
    )
    return "\n".join(lines)


def main() -> int:
    for path in (THESIS, TDSE, QCLE):
        if not path.exists():
            raise FileNotFoundError(path)
    tdse_rows, qcle_rows = read_rows(TDSE), read_rows(QCLE)
    if len(tdse_rows) != 32 or len(qcle_rows) != 32:
        raise ValueError(
            f"Expected 32 rows per reference table; got "
            f"{len(tdse_rows)} and {len(qcle_rows)}"
        )
    text = THESIS.read_text(encoding="utf-8")
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError("Expected exactly one generated reference block")
    start = text.index(BEGIN)
    stop = text.index(END, start) + len(END)
    THESIS.write_text(
        text[:start] + build_block(tdse_rows, qcle_rows) + text[stop:],
        encoding="utf-8",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "thesis": str(THESIS.resolve()),
        "thesis_sha256": sha256(THESIS),
        "tdse_csv": str(TDSE.resolve()),
        "tdse_csv_sha256": sha256(TDSE),
        "qcle_csv": str(QCLE.resolve()),
        "qcle_csv_sha256": sha256(QCLE),
        "tdse_rows": len(tdse_rows),
        "qcle_rows": len(qcle_rows),
    }
    (OUT / "reference_appendix_update_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Updated Appendix F from verified CSVs: {THESIS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
