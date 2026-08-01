from __future__ import annotations

"""Build thesis-ready stand-alone caption/configuration catalogs."""

from pathlib import Path
from typing import Any

import json

from Reproducibility import write_json


def build_figure_catalog(out_dir: str | Path) -> tuple[str, str]:
    root = Path(out_dir)
    manifest_path = root / "run_manifest.json"
    manifest: dict[str, Any] = (json.loads(manifest_path.read_text(encoding="utf-8"))
                                if manifest_path.exists() else {})
    cli = manifest.get("cli_arguments", {})
    n = cli.get("n_train", "not recorded"); seed = cli.get("seed", "not recorded")
    dt = cli.get("dt", "not recorded"); floor = cli.get("log_sn_floor", "not recorded")
    surrogate = cli.get("surrogate", "not recorded")
    gfloor = cli.get("product_g_floor_rel", "not applicable")
    production = (
        f"Unless a caption states otherwise, figures use N={n}, seed={seed}, "
        f"resolved Δt={dt} a.u., surrogate={surrogate}, log σ_n floor={floor}, "
        f"and product-profile relative floor={gfloor}. PBME and MIDPOINT use "
        "the identical initial support cloud. Raw quantities are explicitly "
        "labeled raw; self-normalized quantities are labeled in the axis or legend."
    )
    entries=[]
    for pdf in sorted(root.rglob("*.pdf")):
        sidecar=Path(str(pdf)+".meta.json")
        meta=json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        title=meta.get("title",pdf.stem.replace("_"," ").title())
        scale=meta.get("scale_policy","encoded by the axes")
        norm=meta.get("normalization","identified by axis labels")
        entries.append({"file":str(pdf.relative_to(root)),"title":title,
                        "scale_policy":scale,"normalization":norm,
                        "caption":f"{title}. {production} Scale policy: {scale}. Normalization: {norm}."})
        if not sidecar.exists():
            write_json(sidecar, {"figure": pdf.name, "title": title,
                                 "production_configuration": production,
                                 "scale_policy": scale,
                                 "normalization": norm,
                                 "run_manifest": str(manifest_path.name)})
    payload={"production_configuration":production,"figures":entries}
    json_path=write_json(root/"figure_catalog.json",payload)
    md_path=root/"THESIS_FIGURE_CAPTIONS.md"
    lines=["# Thesis Figure Caption Catalog","",production,"",
           "The graphics intentionally contain no visible figure or subplot headers; panel order, time, method, and configuration are stated in the captions and sidecars.",""]
    for e in entries:
        lines.extend([f"## {e['file']}","",e["caption"],""])
    md_path.write_text("\n".join(lines),encoding="utf-8")
    return str(md_path),json_path
