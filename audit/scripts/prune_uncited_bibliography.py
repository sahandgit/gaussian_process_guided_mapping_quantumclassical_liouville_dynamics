"""Remove uncited BibTeX entries after the final one-file thesis is assembled.

The original file is never overwritten silently: a byte-identical backup and a
CSV disposition record are written under ``reviewer_data_audit/bibliography``.
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parents[2]
THESIS = ROOT / "Thesis" / "Thesis.tex"
BIB = ROOT / "Thesis" / "References.bib"
OUT = ROOT / "reviewer_data_audit" / "bibliography"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def entries(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    starts = list(re.finditer(r"(?m)^[ \t]*@", text))
    if not starts:
        raise ValueError("No BibTeX entries found")
    prefix = text[: starts[0].start()]
    blocks: List[Tuple[str, str]] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[match.start():end]
        key_match = re.match(
            r"\s*@\w+\s*[\{\(]\s*([^,\s]+)\s*,", block, flags=re.S
        )
        if not key_match:
            raise ValueError(f"Cannot parse BibTeX key near: {block[:100]!r}")
        blocks.append((key_match.group(1), block))
    return prefix, blocks


def main() -> int:
    thesis_text = THESIS.read_text(encoding="utf-8")
    cited = set()
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", thesis_text):
        cited.update(key.strip() for key in group.split(",") if key.strip())

    original = BIB.read_text(encoding="utf-8")
    prefix, blocks = entries(original)
    available = {key for key, _ in blocks}
    missing = sorted(cited - available)
    if missing:
        raise ValueError(f"Cited keys absent from bibliography: {missing}")

    OUT.mkdir(parents=True, exist_ok=True)
    backup = OUT / "References_before_uncited_prune.bib"
    if not backup.exists():
        backup.write_bytes(BIB.read_bytes())
    else:
        backup_keys = {
            key
            for key, _ in entries(
                backup.read_text(encoding="utf-8")
            )[1]
        }
        if not available.issubset(backup_keys):
            raise RuntimeError(
                "Existing bibliography backup does not contain the current "
                "source lineage"
            )

    kept_blocks = [(key, block) for key, block in blocks if key in cited]
    removed = sorted(available - cited)
    final_text = prefix + "".join(block for _, block in kept_blocks)
    BIB.write_text(final_text, encoding="utf-8")

    disposition = OUT / "bibliography_disposition.csv"
    with disposition.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("key", "status", "reason")
        )
        writer.writeheader()
        for key, _ in blocks:
            writer.writerow({
                "key": key,
                "status": "RETAINED" if key in cited else "REMOVED",
                "reason": (
                    "cited in final Thesis.tex"
                    if key in cited else "uncited in final Thesis.tex"
                ),
            })
    manifest = OUT / "bibliography_hashes.txt"
    manifest.write_text(
        f"backup_sha256={sha256(backup)}\n"
        f"final_sha256={sha256(BIB)}\n"
        f"cited_keys={len(cited)}\n"
        f"retained_entries={len(kept_blocks)}\n"
        f"removed_entries={len(removed)}\n"
        f"removed_keys={','.join(removed)}\n",
        encoding="utf-8",
    )
    print(
        f"Retained {len(kept_blocks)} cited entries; removed {len(removed)} "
        f"uncited entries; backup={backup}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
