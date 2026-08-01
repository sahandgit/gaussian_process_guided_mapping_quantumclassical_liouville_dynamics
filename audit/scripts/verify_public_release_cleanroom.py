"""Download and independently verify a public thesis release in a fresh tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from urllib.request import Request, urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "reviewer_data_audit"
OUTPUT = AUDIT / "CLEAN_ROOM_VERIFICATION.json"
OWNER = "sahandgit"
REPOSITORY = "gaussian_process_guided_mapping_quantumclassical_liouville_dynamics"
DEFAULT_TAG = "thesis-final-2026-08-01-r3"
REQUIRED_EVIDENCE = (
    "final_reviewer_closure/PAYLOAD_SHA256SUMS.csv",
    "final_reviewer_closure/FINAL_RUN_MANIFEST.json",
    "final_reviewer_closure/figures/FIGURE_DATA_CROSSWALK.csv",
    "final_reviewer_closure/table_data_crosswalk.csv",
    "final_reviewer_closure/environment.json",
    "final_reviewer_closure/environment/pip_freeze.txt",
    "final_reviewer_closure/README.md",
)
REQUIRED_SOURCE = (
    "thesis/Thesis.tex",
    "thesis/References.bib",
    "thesis/ut-thesis.cls",
    "Reviewer_Response.tex",
    "final_reviewer_closure/figures/FIGURE_DATA_CROSSWALK.csv",
    "final_reviewer_closure/table_data_crosswalk.csv",
    "README.md",
)
FORBIDDEN_LOG = (
    "error:", "undefined references", "undefined citations",
    "multiply defined", "overfull \\hbox", "overfull \\vbox",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    request = Request(url, headers={"User-Agent": "thesis-clean-room-verifier"})
    received = 0
    next_report = 64 * 1024 * 1024
    with urlopen(request, timeout=600) as response, target.open("wb") as stream:
        while block := response.read(8 * 1024 * 1024):
            stream.write(block)
            received += len(block)
            if received >= next_report:
                print(f"Downloaded {target.name}: {received} bytes", flush=True)
                next_report += 64 * 1024 * 1024
    print(f"Downloaded {target.name}: {received} bytes", flush=True)


def find_compiler(explicit: str | None) -> Path:
    candidates = [
        explicit,
        os.environ.get("TECTONIC_BIN"),
        str(
            ROOT / "thesis_revision_evidence" / "tools"
            / "tectonic-0.17.0" / "tectonic.exe"
        ),
        shutil.which("tectonic"),
    ]
    for value in candidates:
        if value and Path(value).is_file():
            return Path(value).resolve()
    raise FileNotFoundError(
        "Tectonic was not found; pass --tectonic or set TECTONIC_BIN"
    )


def compile_tex(compiler: Path, source: Path) -> dict[str, object]:
    command = [
        str(compiler), "-X", "compile", source.name,
        "--reruns", "2", "--keep-logs", "--only-cached",
    ]
    result = subprocess.run(
        command, cwd=source.parent, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Clean compilation failed for {source.name}: "
            f"{result.stderr[-2000:]}"
        )
    pdf = source.with_suffix(".pdf")
    log = source.with_suffix(".log")
    if not pdf.is_file() or not log.is_file():
        raise FileNotFoundError(f"Compiler did not create PDF/log for {source}")
    log_text = log.read_text(encoding="utf-8", errors="replace").lower()
    forbidden = [token for token in FORBIDDEN_LOG if token in log_text]
    if forbidden:
        raise RuntimeError(f"{source.name} log contains {forbidden}")
    pages = None
    page_counter = None
    candidates = list(
        (ROOT / "thesis_revision_evidence" / "tools").glob(
            "poppler*/Library/bin/pdfinfo.exe"
        )
    )
    discovered = shutil.which("pdfinfo")
    if discovered:
        candidates.append(Path(discovered))
    for pdfinfo in candidates:
        counter = subprocess.run(
            [str(pdfinfo), str(pdf)], capture_output=True, text=True, check=False
        )
        match = re.search(r"(?m)^Pages:\s+(\d+)\s*$", counter.stdout)
        if counter.returncode == 0 and match:
            pages = int(match.group(1))
            page_counter = str(pdfinfo.resolve())
            break
    if pages is None:
        raise RuntimeError(f"Could not count pages in clean-room PDF {pdf}")
    return {
        "source": source.name,
        "command": command[1:],
        "returncode": result.returncode,
        "pdf_sha256": sha256(pdf),
        "pdf_size_bytes": pdf.stat().st_size,
        "page_count": pages,
        "page_counter": page_counter,
        "forbidden_log_diagnostics": forbidden,
    }


def verify_embedded_index(extract_root: Path) -> int:
    index = extract_root / "final_reviewer_closure" / "PAYLOAD_SHA256SUMS.csv"
    with index.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = extract_root / Path(row["relative_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError(f"Embedded size mismatch: {row['relative_path']}")
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"Embedded SHA-256 mismatch: {row['relative_path']}")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--tectonic", default=None)
    args = parser.parse_args()

    release_page = (
        f"https://github.com/{OWNER}/{REPOSITORY}/releases/tag/{args.tag}"
    )
    evidence_url = (
        f"https://github.com/{OWNER}/{REPOSITORY}/releases/download/"
        f"{args.tag}/frozen_numerical_evidence_payload.zip"
    )
    source_url = (
        f"https://github.com/{OWNER}/{REPOSITORY}/archive/refs/tags/"
        f"{args.tag}.zip"
    )
    compiler = find_compiler(args.tectonic)
    version = subprocess.run(
        [str(compiler), "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()

    # Use the system temporary root so the extracted GitHub repository name
    # does not exceed Windows' legacy working-directory path limit.
    with tempfile.TemporaryDirectory(prefix="thesis-clean-room-") as temp:
        work = Path(temp)
        evidence_zip = work / "evidence.zip"
        source_zip = work / "source.zip"
        download(evidence_url, evidence_zip)
        evidence_digest = sha256(evidence_zip)
        if evidence_digest != args.archive_sha256:
            raise RuntimeError(
                f"Downloaded archive SHA-256 {evidence_digest} does not match "
                f"expected {args.archive_sha256}"
            )
        download(source_url, source_zip)

        evidence_extract = work / "evidence-extracted"
        source_extract = work / "source-extracted"
        with zipfile.ZipFile(evidence_zip) as bundle:
            bundle.extractall(evidence_extract)
        with zipfile.ZipFile(source_zip) as bundle:
            bundle.extractall(source_extract)

        missing_evidence = [
            name for name in REQUIRED_EVIDENCE
            if not (evidence_extract / name).is_file()
        ]
        if missing_evidence:
            raise FileNotFoundError(f"Missing evidence files: {missing_evidence}")
        verified_files = verify_embedded_index(evidence_extract)

        source_roots = [path for path in source_extract.iterdir() if path.is_dir()]
        if len(source_roots) != 1:
            raise RuntimeError(f"Expected one source root; found {source_roots}")
        source_root = source_roots[0]
        missing_source = [
            name for name in REQUIRED_SOURCE if not (source_root / name).is_file()
        ]
        if missing_source:
            raise FileNotFoundError(f"Missing source files: {missing_source}")
        compile_results = [
            compile_tex(compiler, source_root / "thesis" / "Thesis.tex"),
            compile_tex(compiler, source_root / "Reviewer_Response.tex"),
        ]

        record = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASSED",
            "release_page": release_page,
            "release_tag": args.tag,
            "evidence_download_url": evidence_url,
            "evidence_expected_sha256": args.archive_sha256,
            "evidence_download_sha256": evidence_digest,
            "evidence_size_bytes": evidence_zip.stat().st_size,
            "source_download_url": source_url,
            "source_download_sha256": sha256(source_zip),
            "source_size_bytes": source_zip.stat().st_size,
            "fresh_temporary_directory_used": True,
            "required_evidence_files": list(REQUIRED_EVIDENCE),
            "required_source_files": list(REQUIRED_SOURCE),
            "embedded_index_rows_verified": verified_files,
            "compiler": {
                "version": version,
                "sha256": sha256(compiler),
                "external_to_extracted_source": True,
            },
            "clean_compilations": compile_results,
            "cleanup": "temporary downloads and extractions removed after verification",
        }
        OUTPUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
