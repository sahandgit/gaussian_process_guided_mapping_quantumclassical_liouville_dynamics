"""Assemble Chapters 6–7 and final evidence tables into one Thesis.tex source."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
THESIS_DIR = ROOT / "Thesis"
MASTER = THESIS_DIR / "Thesis.tex"
CHAPTER6 = THESIS_DIR / "Chapter6_VerifiedResults.tex"
CHAPTER7 = THESIS_DIR / "Chapter7_Conclusions.tex"
OUT = ROOT / "reviewer_data_audit" / "thesis_sources"
FIGURE_CROSSWALK = (
    ROOT / "final_reviewer_closure" / "figures" / "FIGURE_DATA_CROSSWALK.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def replace_component(
    text: str, name: str, input_command: str, content: str
) -> str:
    begin = f"% BEGIN INLINED {name}"
    end = f"% END INLINED {name}"
    block = begin + "\n" + content.rstrip() + "\n" + end
    pattern = re.compile(
        re.escape(begin) + r".*?" + re.escape(end), flags=re.S
    )
    if pattern.search(text):
        return pattern.sub(lambda _: block, text, count=1)
    if input_command not in text:
        raise ValueError(
            f"Neither {input_command!r} nor existing {name} markers found"
        )
    return text.replace(input_command, block, 1)


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    for path in (MASTER, CHAPTER6, CHAPTER7):
        if not path.exists():
            raise FileNotFoundError(path)
    OUT.mkdir(parents=True, exist_ok=True)
    backup = OUT / "Thesis_before_final_inline.tex"
    if not backup.exists():
        backup.write_bytes(MASTER.read_bytes())

    chapter6 = CHAPTER6.read_text(encoding="utf-8")

    master = MASTER.read_text(encoding="utf-8")
    master = replace_component(
        master,
        "CHAPTER 6",
        r"\input{Chapter6_VerifiedResults}",
        chapter6,
    )
    master = replace_component(
        master,
        "CHAPTER 7",
        r"\input{Chapter7_Conclusions}",
        CHAPTER7.read_text(encoding="utf-8"),
    )
    master = re.sub(
        r"\\begin\{comment\}\s*"
        r"\\section\{Implementation cross-reference and reproducibility settings\}"
        r".*?\\end\{comment\}",
        "",
        master,
        flags=re.S,
    )

    forbidden = (
        r"\input{Chapter6",
        r"\input{Chapter7",
        r"\input{ReviewerEvidenceTables",
    )
    present = [token for token in forbidden if token in master]
    if present:
        raise ValueError(f"Forbidden final-source tokens remain: {present}")
    included_figures = re.findall(
        r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", master
    )
    with FIGURE_CROSSWALK.open(newline="", encoding="utf-8-sig") as handle:
        verified_figures = sorted(
            row["figure"].replace("\\", "/").rsplit("/", 1)[-1]
            for row in csv.DictReader(handle)
        )
    included_names = sorted(
        path.replace("\\", "/").rsplit("/", 1)[-1] for path in included_figures
    )
    if included_names != verified_figures or any(
        "final_reviewer_closure/figures/" not in path
        for path in included_figures
    ):
        raise ValueError(
            "Final source must contain exactly the hash-verified crosswalk "
            f"figures {verified_figures}; found {included_figures}"
        )

    MASTER.write_text(master, encoding="utf-8")
    manifest = {
        "master": str(MASTER.resolve()),
        "master_sha256": sha256(MASTER),
        "backup": str(backup.resolve()),
        "backup_sha256": sha256(backup),
        "chapter6_source": str(CHAPTER6.resolve()),
        "chapter6_sha256": sha256(CHAPTER6),
        "chapter7_source": str(CHAPTER7.resolve()),
        "chapter7_sha256": sha256(CHAPTER7),
        "physics_table_labels": re.findall(
            r"\\label\{(tab:[^}]+)\}", chapter6
        ),
        "included_figures": included_figures,
        "single_source_contract": True,
    }
    (OUT / "final_thesis_assembly_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"Assembled one-source thesis: {MASTER}; SHA-256 "
        f"{manifest['master_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
