"""Compile the one-source thesis and reviewer response, then render every page.

The script uses the locally archived Tectonic binary and Poppler.  Rendered
pages and contact sheets are written only below ``reviewer_data_audit/pdf_qa``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "reviewer_data_audit"
QA = AUDIT / "pdf_qa"
TOOLS = ROOT / "thesis_revision_evidence" / "tools"
TECTONIC = TOOLS / "tectonic-0.17.0" / "tectonic.exe"
TARGETS = (
    ("thesis", ROOT / "Thesis" / "Thesis.tex"),
    ("reviewer_response", ROOT / "Reviewer_Response.tex"),
)
FORBIDDEN_LOG = (
    "error:",
    "undefined references",
    "undefined citations",
    "multiply defined",
    "overfull \\hbox",
    "overfull \\vbox",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_owned_directory(path: Path) -> None:
    resolved = path.resolve()
    qa_root = QA.resolve()
    if resolved == qa_root or qa_root not in resolved.parents:
        raise RuntimeError(f"Refusing to clean path outside owned QA tree: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def poppler_tool(name: str) -> str:
    filenames = (f"{name}.exe", name)

    # 1) Locally archived Poppler, same convention as the archived Tectonic
    #    binary (e.g. ``tools/poppler-24.08.0/Library/bin/pdftoppm.exe``).
    for archive in sorted(TOOLS.glob("poppler*")):
        for rel in ("Library/bin", "bin"):
            for filename in filenames:
                candidate = archive / rel / filename
                if candidate.exists():
                    return str(candidate)

    # 2) Explicit override: directory containing the Poppler executables.
    override = os.environ.get("POPPLER_BIN")
    if override:
        for filename in filenames:
            candidate = Path(override) / filename
            if candidate.exists():
                return str(candidate)

    # 3) PATH.
    discovered = shutil.which(name)
    if discovered:
        candidate = Path(discovered)
        # The bundled Windows runtime may expose a wrapper that still points
        # to an obsolete ``native/poppler/bin`` location. Resolve the shipped
        # executable under ``native/poppler/Library/bin`` when present.
        if candidate.suffix.lower() == ".cmd" and len(candidate.parents) >= 3:
            native = (
                candidate.parents[2] / "native" / "poppler"
                / "Library" / "bin" / f"{name}.exe"
            )
            if native.exists():
                return str(native)
        return discovered

    # 4) Common Windows install locations (conda, manual installs).
    bases = [Path(sys.prefix) / "Library" / "bin"]
    for parent in (Path("C:/Program Files"), Path("C:/")):
        try:
            bases.extend(sorted(parent.glob("poppler*")))
        except OSError:
            pass
    for base in bases:
        for rel in ("", "Library/bin", "bin"):
            for filename in filenames:
                candidate = base / rel / filename if rel else base / filename
                if candidate.exists():
                    return str(candidate)

    raise FileNotFoundError(
        f"{name} not found. Searched the archived tools directory ({TOOLS}), "
        "the POPPLER_BIN environment variable, PATH, and common install "
        "locations. Run 'python thesis_revision_evidence/tools/"
        "fetch_poppler.py' once to download and archive Poppler locally."
    )


def ensure_poppler() -> "Tuple[str, str] | None":
    """Return (pdftoppm, pdfinfo) paths, auto-archiving Poppler if needed.

    Returns None when Poppler cannot be obtained; the caller then uses the
    pure-Python pypdfium2 fallback renderer.
    """
    try:
        return poppler_tool("pdftoppm"), poppler_tool("pdfinfo")
    except FileNotFoundError:
        pass
    fetch = TOOLS / "fetch_poppler.py"
    if fetch.exists():
        print("Poppler not found; downloading the pinned archive once ...")
        result = subprocess.run(
            [sys.executable, str(fetch)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(
                "Automatic Poppler download failed "
                f"(exit={result.returncode}): {result.stderr.strip()[:500]}"
            )
        try:
            return poppler_tool("pdftoppm"), poppler_tool("pdfinfo")
        except FileNotFoundError:
            pass
    return None


def render_with_pdfium(
    pdf: Path, render_dir: Path
) -> Tuple[List[Path], int]:
    """Pure-Python fallback renderer used when Poppler is unavailable."""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise FileNotFoundError(
            "Poppler is unavailable and its automatic download failed, and "
            "the pypdfium2 fallback is not installed. Either run 'python "
            "thesis_revision_evidence/tools/fetch_poppler.py' on a machine "
            "with network access, set POPPLER_BIN, or run "
            "'pip install pypdfium2'."
        ) from exc
    document = pdfium.PdfDocument(str(pdf))
    expected = len(document)
    pages: List[Path] = []
    for index in range(expected):
        bitmap = document[index].render(scale=110 / 72)
        image = bitmap.to_pil()
        target = render_dir / f"page-{index + 1:04d}.png"
        image.save(target)
        pages.append(target)
    return pages, expected


def unlock_output(target: Path) -> None:
    """Move a locked output file aside so Tectonic can write a fresh copy.

    Windows PDF viewers, the Explorer preview pane, and sync/antivirus tools
    open output files without write sharing, which makes Tectonic fail with
    'os error 32'. Renaming a locked file usually still succeeds, so the
    locked copy is moved aside instead of blocking the build.
    """
    if not target.exists():
        return
    try:
        with target.open("r+b"):
            return  # writable; no lock
    except PermissionError:
        pass
    stale = target.with_name(
        f"{target.stem}.locked-{time.strftime('%Y%m%d%H%M%S')}{target.suffix}"
    )
    try:
        target.rename(stale)
        print(
            f"warning: {target.name} was locked by another program "
            f"(PDF viewer, Explorer preview pane, or sync tool); the locked "
            f"copy was moved aside to {stale.name} and can be deleted."
        )
    except PermissionError as exc:
        raise RuntimeError(
            f"{target} is locked by another program and cannot be replaced "
            "or renamed. Close any PDF viewer or Explorer preview pane "
            "showing it (and pause sync/antivirus for the folder), then "
            "rerun this script."
        ) from exc


def compile_tex(source: Path) -> Tuple[Path, Path, str]:
    if not TECTONIC.exists():
        raise FileNotFoundError(TECTONIC)
    command = [
        str(TECTONIC),
        "-X", "compile", source.name,
        "--reruns", "2",
        "--keep-logs",
        "--keep-intermediates",
        "--only-cached",
    ]
    for produced in (source.with_suffix(".pdf"), source.with_suffix(".log")):
        unlock_output(produced)
    result = subprocess.run(
        command,
        cwd=source.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 and "os error 32" in result.stderr:
        print("Output file locked during compile; retrying once in 3 s ...")
        time.sleep(3)
        for produced in (
            source.with_suffix(".pdf"), source.with_suffix(".log")
        ):
            unlock_output(produced)
        result = subprocess.run(
            command,
            cwd=source.parent,
            capture_output=True,
            text=True,
            check=False,
        )
    build_log = QA / f"{source.stem}_tectonic_console.log"
    build_log.parent.mkdir(parents=True, exist_ok=True)
    build_log.write_text(
        "COMMAND: " + subprocess.list2cmdline(command) + "\n\nSTDOUT\n"
        + result.stdout + "\nSTDERR\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Tectonic failed for {source}: exit={result.returncode}; "
            f"console={build_log}"
        )
    pdf = source.with_suffix(".pdf")
    log = source.with_suffix(".log")
    if not pdf.exists() or not log.exists():
        raise FileNotFoundError(f"Expected PDF/log absent for {source}")
    log_text = log.read_text(encoding="utf-8", errors="replace").lower()
    bad = [token for token in FORBIDDEN_LOG if token.lower() in log_text]
    if bad:
        raise RuntimeError(f"{source}: forbidden log diagnostics: {bad}")
    return pdf, log, str(build_log)


def render_pdf(name: str, pdf: Path) -> Tuple[List[Path], List[Path]]:
    tools = ensure_poppler()
    render_dir = QA / "renders" / name
    clean_owned_directory(render_dir)
    if tools is not None:
        pdftoppm, pdfinfo = tools
        prefix = render_dir / "page"
        result = subprocess.run(
            [pdftoppm, "-png", "-r", "110", str(pdf), str(prefix)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pdftoppm failed for {pdf}: {result.stderr}")
        pages = sorted(render_dir.glob("page-*.png"))
        info = subprocess.run(
            [pdfinfo, str(pdf)], capture_output=True, text=True, check=False
        )
        if info.returncode != 0:
            raise RuntimeError(f"pdfinfo failed for {pdf}: {info.stderr}")
        match = re.search(r"(?m)^Pages:\s+(\d+)\s*$", info.stdout)
        if not match:
            raise RuntimeError(
                f"pdfinfo did not report a page count for {pdf}"
            )
        expected = int(match.group(1))
    else:
        print(f"Rendering {pdf.name} with the pypdfium2 fallback renderer.")
        pages, expected = render_with_pdfium(pdf, render_dir)
    if len(pages) != expected:
        raise RuntimeError(f"{pdf}: rendered {len(pages)} of {expected} pages")

    sheets_dir = QA / "contact_sheets" / name
    clean_owned_directory(sheets_dir)
    sheets: List[Path] = []
    per_sheet = 16
    columns = 4
    thumb_w, thumb_h = 360, 510
    label_h = 24
    font = ImageFont.load_default()
    for offset in range(0, len(pages), per_sheet):
        batch = pages[offset:offset + per_sheet]
        rows = (len(batch) + columns - 1) // columns
        sheet = Image.new(
            "RGB",
            (columns * thumb_w, rows * (thumb_h + label_h)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for slot, page in enumerate(batch):
            image = Image.open(page).convert("RGB")
            image.thumbnail((thumb_w - 8, thumb_h - 8))
            x = (slot % columns) * thumb_w + (thumb_w - image.width) // 2
            y0 = (slot // columns) * (thumb_h + label_h)
            y = y0 + label_h + (thumb_h - image.height) // 2
            sheet.paste(image, (x, y))
            draw.text(
                (x, y0 + 4),
                f"{name} page {offset + slot + 1}",
                fill="black",
                font=font,
            )
        target = sheets_dir / f"pages_{offset + 1:04d}_{offset + len(batch):04d}.png"
        sheet.save(target)
        sheets.append(target)
    return pages, sheets


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    QA.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []
    for name, source in TARGETS:
        if not source.exists():
            raise FileNotFoundError(source)
        pdf, log, console = compile_tex(source)
        pages, sheets = render_pdf(name, pdf)
        records.append({
            "name": name,
            "source": str(source.resolve()),
            "source_sha256": sha256(source),
            "pdf": str(pdf.resolve()),
            "pdf_sha256": sha256(pdf),
            "latex_log": str(log.resolve()),
            "latex_log_sha256": sha256(log),
            "console_log": console,
            "page_count": len(pages),
            "rendered_page_count": len(pages),
            "contact_sheet_count": len(sheets),
            "visual_status": "PENDING_MANUAL_CONTACT_SHEET_INSPECTION",
        })
    manifest = QA / "pdf_compile_render_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (QA / "pdf_compile_render_manifest.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(
        "Compiled and rendered PDFs. Manual visual status remains pending until "
        f"all contact sheets under {QA / 'contact_sheets'} are inspected."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
