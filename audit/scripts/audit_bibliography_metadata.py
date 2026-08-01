"""Audit final BibTeX field completeness, DOI syntax, and optional DOI metadata.

With ``--crossref``, DOI records are fetched read-only from Crossref and cached
under ``reviewer_data_audit/bibliography/crossref_cache``.  The script never
changes the bibliography.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from prune_uncited_bibliography import BIB, OUT, THESIS, entries


DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.I)


def fields(block: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    position = block.find(",") + 1
    while position > 0 and position < len(block):
        match = re.search(r"(?m)^\s*([A-Za-z][A-Za-z0-9_-]*)\s*=\s*", block[position:])
        if not match:
            break
        name = match.group(1).lower()
        value_start = position + match.end()
        if value_start >= len(block):
            break
        opening = block[value_start]
        if opening in "{(":
            closing = "}" if opening == "{" else ")"
            depth = 0
            cursor = value_start
            while cursor < len(block):
                character = block[cursor]
                if character == opening and (cursor == 0 or block[cursor - 1] != "\\"):
                    depth += 1
                elif character == closing and (cursor == 0 or block[cursor - 1] != "\\"):
                    depth -= 1
                    if depth == 0:
                        break
                cursor += 1
            value = block[value_start + 1:cursor].strip()
            position = cursor + 1
        elif opening == '"':
            cursor = value_start + 1
            while cursor < len(block):
                if block[cursor] == '"' and block[cursor - 1] != "\\":
                    break
                cursor += 1
            value = block[value_start + 1:cursor].strip()
            position = cursor + 1
        else:
            cursor = block.find(",", value_start)
            if cursor < 0:
                cursor = len(block)
            value = block[value_start:cursor].strip()
            position = cursor + 1
        result[name] = value
    return result


def normalized_title(value: str) -> str:
    # Preserve the letter modified by common BibTeX accent commands before
    # removing remaining TeX markup.  NFKD also makes Crossref Unicode titles
    # comparable with their BibTeX ASCII/TeX forms.
    value = re.sub(r"""\\["'`~^=.uvHckbd]\s*\{?([A-Za-z])\}?""", r"\1", value)
    value = value.replace(r"\&", " and ")
    value = re.sub(r"\\[A-Za-z]+\*?\s*", " ", value)
    value = value.replace("{", "").replace("}", "")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(character for character in value if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def crossref_years(message: Dict[str, object]) -> List[str]:
    years = set()
    for field in (
        "issued",
        "published",
        "published-print",
        "published-online",
        "created",
    ):
        payload = message.get(field)
        if not isinstance(payload, dict):
            continue
        parts = payload.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            years.add(str(parts[0][0]))
        date_time = payload.get("date-time")
        if isinstance(date_time, str) and re.match(r"^\d{4}", date_time):
            years.add(date_time[:4])
    return sorted(years)


def required_fields(entry_type: str) -> Tuple[str, ...]:
    common = ("author", "title", "year")
    if entry_type == "article":
        return (*common, "journal", "volume")
    if entry_type == "book":
        return (*common, "publisher")
    if entry_type in ("incollection", "inproceedings"):
        return (*common, "booktitle")
    return common


def crossref_record(doi: str, cache: Path) -> Dict[str, object]:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (re.sub(r"[^A-Za-z0-9._-]+", "_", doi) + ".json")
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "MSc-thesis-bibliography-audit/1.0"
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    time.sleep(0.1)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossref", action="store_true")
    args = parser.parse_args()
    thesis = THESIS.read_text(encoding="utf-8")
    cited = set()
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", thesis):
        cited.update(key.strip() for key in group.split(",") if key.strip())

    _, blocks = entries(BIB.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for key, block in blocks:
        type_match = re.match(r"\s*@(\w+)", block)
        entry_type = type_match.group(1).lower() if type_match else "unknown"
        data = fields(block)
        required = list(required_fields(entry_type))
        if entry_type == "book" and data.get("editor") and "author" in required:
            required.remove("author")
        if (
            entry_type == "article"
            and data.get("journal", "").lower().startswith("arxiv")
            and "volume" in required
        ):
            required.remove("volume")
        missing = [name for name in required if not data.get(name)]
        doi = data.get("doi", "").strip()
        row: Dict[str, object] = {
            "key": key,
            "cited": key in cited,
            "entry_type": entry_type,
            "author": data.get("author", ""),
            "title": data.get("title", ""),
            "container": data.get("journal", data.get("booktitle", data.get("publisher", ""))),
            "year": data.get("year", ""),
            "volume": data.get("volume", ""),
            "pages_or_article": data.get("pages", data.get("eid", data.get("number", ""))),
            "doi": doi,
            "required_fields_complete": not missing,
            "missing_required_fields": ";".join(missing),
            "doi_syntax_valid_or_absent": (not doi or bool(DOI_RE.match(doi))),
            "crossref_status": "NOT_REQUESTED",
            "crossref_title": "",
            "crossref_title_similarity": "",
            "crossref_years": "",
            "crossref_title_match": "",
            "crossref_year_match": "",
        }
        if args.crossref and doi:
            try:
                payload = crossref_record(doi, OUT / "crossref_cache")
                message = payload["message"]
                titles = message.get("title") or []
                crossref_title = titles[0] if titles else ""
                candidate_years = crossref_years(message)
                local_title = normalized_title(data.get("title", ""))
                remote_title = normalized_title(crossref_title)
                similarity = difflib.SequenceMatcher(
                    None, local_title, remote_title
                ).ratio()
                row["crossref_status"] = "FOUND"
                row["crossref_title"] = crossref_title
                row["crossref_title_similarity"] = f"{similarity:.6f}"
                row["crossref_years"] = ";".join(candidate_years)
                # Crossref commonly differs only in punctuation, subtitle
                # styling, or article-number wording.  A high normalized
                # similarity is accepted but the exact remote title remains in
                # the CSV for human review.
                row["crossref_title_match"] = bool(
                    local_title and remote_title
                    and (local_title == remote_title or similarity >= 0.95)
                )
                row["crossref_year_match"] = data.get("year", "") in candidate_years
            except Exception as exc:
                row["crossref_status"] = f"ERROR: {type(exc).__name__}: {exc}"
        rows.append(row)

    output = OUT / "bibliography_metadata_audit.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "entries": len(rows),
        "cited_entries": sum(bool(row["cited"]) for row in rows),
        "uncited_entries": sum(not bool(row["cited"]) for row in rows),
        "required_fields_complete": sum(
            bool(row["required_fields_complete"]) for row in rows
        ),
        "doi_syntax_failures": [
            row["key"] for row in rows if not row["doi_syntax_valid_or_absent"]
        ],
        "crossref_requested": args.crossref,
        "crossref_errors": [
            row["key"] for row in rows
            if str(row["crossref_status"]).startswith("ERROR")
        ],
        "crossref_title_mismatches": [
            row["key"] for row in rows
            if row["crossref_title_match"] is False
        ],
        "crossref_year_mismatches": [
            row["key"] for row in rows
            if row["crossref_year_match"] is False
        ],
    }
    (OUT / "bibliography_metadata_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if not summary["doi_syntax_failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
