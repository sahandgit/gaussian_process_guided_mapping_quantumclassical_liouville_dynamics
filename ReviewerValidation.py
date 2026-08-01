from __future__ import annotations

"""Reviewer-facing validation and convergence driver.

This module turns the reviewer's open numerical requests into reproducible
commands and machine-readable outputs.  It never labels an unexecuted test as
passed: campaign plans, completed runs, metrics, and acceptance decisions are
stored separately.

Examples
--------
    python ReviewerValidation.py plan --out validation
    python ReviewerValidation.py campaign --out validation --execute
    python ReviewerValidation.py analyze --out validation
    python ReviewerValidation.py manufactured --out validation/manufactured
    python ReviewerValidation.py projection midpoint --out validation/projection
    python ReviewerValidation.py reference --out validation/reference
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

from KDEDensity import ProjectedNuclearGP
from Reproducibility import array_fingerprint, environment_metadata, write_json


OBSERVABLE_KEYS = (
    "cloud_weighted_P0", "cloud_weighted_P1", "lw_P0", "lw_P1",
    "raw_norm_drift", "raw_energy_drift", "raw_trace_drift",
    "cs_q_y_weighted_rms", "cs_q_max", "cs_q_sum_yc",
)


@dataclass(frozen=True)
class CampaignCase:
    case_id: str
    category: str
    n_train: int
    dt: float
    seed: int
    t_final: float


def campaign_cases(n_base: int, dt_base: float, seeds: Iterable[int],
                   t_final: float) -> list[CampaignCase]:
    seeds = list(dict.fromkeys(int(s) for s in seeds))
    cases = [
        CampaignCase(f"dt_N{n_base}_h{dt_base:g}", "time_step", n_base, dt_base, seeds[0], t_final),
        CampaignCase(f"dt_N{n_base}_h{dt_base/2:g}", "time_step", n_base, dt_base / 2.0, seeds[0], t_final),
        CampaignCase(f"support_N{2*n_base}", "support", 2 * n_base, dt_base / 2.0, seeds[0], t_final),
    ]
    for seed in seeds:
        cases.append(CampaignCase(f"seed_{seed}", "replication", n_base,
                                  dt_base / 2.0, seed, t_final))
    # Remove exact duplicate parameter tuples while retaining all category
    # labels in the plan through aliases.
    unique: dict[tuple[int, float, int, float], CampaignCase] = {}
    for case in cases:
        key=(case.n_train,case.dt,case.seed,case.t_final)
        if key in unique:
            prior=unique[key]
            categories=",".join(dict.fromkeys((prior.category+","+case.category).split(",")))
            unique[key]=CampaignCase(prior.case_id,categories,prior.n_train,prior.dt,prior.seed,prior.t_final)
        else:
            unique[key]=case
    return list(unique.values())


def command_for_case(case: CampaignCase, out_root: Path,
                     density_mode: str, sampling_mode: str,
                     surrogate: str) -> list[str]:
    run_py = Path(__file__).with_name("run.py")
    final_snapshot_interval = max(1, int(round(case.t_final / case.dt)))
    cmd = [sys.executable, str(run_py),
           "--out", str(out_root / case.case_id),
           "--n_train", str(case.n_train),
           "--seed", str(case.seed),
           "--dt", repr(case.dt), "--no_auto_dt",
           "--t_final", repr(case.t_final),
           "--density_mode", density_mode,
           "--sampling_mode", sampling_mode,
           "--surrogate", surrogate,
           "--snapshot_every", str(final_snapshot_interval),
           "--skip_figures", "--quiet"]
    if sampling_mode == "focused":
        cmd.append("--no_abs_target")
    return cmd


def write_campaign_plan(out: Path, args) -> list[dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    cases = campaign_cases(args.n_base, args.dt_base, args.seeds, args.t_final)
    rows = []
    for c in cases:
        command = command_for_case(c, out, args.density_mode,
                                   args.sampling_mode, args.surrogate)
        rows.append({**asdict(c), "command": command,
                     "status": "planned", "result_directory": str(out / c.case_id)})
    write_json(out / "campaign_plan.json", {
        "schema_version": 1,
        "purpose": {
            "time_step": "compare dt and dt/2 at fixed support/seed/endpoint",
            "support": "compare N and 2N at the finer dt with the same seed",
            "replication": "independent support clouds at fixed N, dt, and endpoint",
        },
        "environment": environment_metadata(), "cases": rows,
    })
    with open(out / "campaign_commands.sh", "w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        for row in rows:
            import shlex
            handle.write(" ".join(shlex.quote(x) for x in row["command"]) + "\n")
    return rows


def run_campaign(out: Path, args) -> None:
    rows = write_campaign_plan(out, args)
    if not args.execute:
        print(f"Wrote {len(rows)} planned cases to {out}; no solver was executed.")
        return
    statuses = []
    for row in rows:
        case_dir = Path(row["result_directory"])
        case_dir.mkdir(parents=True, exist_ok=True)
        log_path = case_dir / "validation_run.log"
        print(f"[validation] executing {row['case_id']}")
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(row["command"], stdout=log,
                                  stderr=subprocess.STDOUT, text=True)
        status = dict(row)
        status.update({"status": "completed" if proc.returncode == 0 else "failed",
                       "returncode": proc.returncode, "log": str(log_path)})
        statuses.append(status)
        write_json(out / "campaign_status.json", {"cases": statuses})
        if proc.returncode != 0 and not args.keep_going:
            raise SystemExit(f"Case {row['case_id']} failed; see {log_path}")


def _load_arrays(stem: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with open(str(stem) + ".json", "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    with np.load(str(stem) + ".npz") as z:
        arrays = {k: z[k] for k in z.files if not k.startswith("snap_")}
    return meta, arrays


def _endpoint_row(case: dict, scheme: str, arrays: dict[str, np.ndarray]) -> dict:
    row = {k: case[k] for k in ("case_id", "category", "n_train", "dt", "seed", "t_final")}
    row["scheme"] = scheme
    row["actual_t_final"] = float(np.asarray(arrays["t"])[-1])
    for key in OBSERVABLE_KEYS:
        values = arrays.get(key)
        row[key] = float(np.asarray(values)[-1]) if values is not None else float("nan")
    return row


def analyze_campaign(out: Path) -> dict[str, Any]:
    plan_path = out / "campaign_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing {plan_path}; run the plan/campaign command first.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = []
    for case in plan["cases"]:
        for scheme in ("pbme", "midpoint"):
            stem = Path(case["result_directory"]) / scheme
            if Path(str(stem) + ".npz").exists():
                _, arrays = _load_arrays(stem)
                rows.append(_endpoint_row(case, scheme, arrays))
    if not rows:
        raise RuntimeError("No completed campaign NPZ files were found; metrics were not invented.")

    keys = list(rows[0].keys())
    with open(out / "endpoint_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)

    summary: dict[str, Any] = {"n_completed_scheme_runs": len(rows)}
    # Finest run within the time-step category is the comparison reference.
    for scheme in ("pbme", "midpoint"):
        candidates = [r for r in rows if r["scheme"] == scheme]
        if not candidates:
            continue
        finest = min(candidates, key=lambda r: (r["dt"], -r["n_train"]))
        comparisons = []
        for r in candidates:
            delta = {k: float(r[k] - finest[k]) for k in OBSERVABLE_KEYS
                     if np.isfinite(r[k]) and np.isfinite(finest[k])}
            comparisons.append({"case_id": r["case_id"], "against": finest["case_id"],
                                "absolute_endpoint_differences": delta})
        summary[f"{scheme}_convergence"] = comparisons

        reps = [r for r in candidates if "replication" in str(r["category"])]
        if len(reps) >= 2:
            summary[f"{scheme}_replication"] = {
                k: {"mean": float(np.nanmean([r[k] for r in reps])),
                    "sample_std": float(np.nanstd([r[k] for r in reps], ddof=1)),
                    "n": len(reps)} for k in OBSERVABLE_KEYS
            }
    write_json(out / "campaign_metrics.json", summary)
    return summary


def _snapshot(stem: Path, step: Optional[int] = None) -> tuple[dict, dict[str, Any]]:
    with open(str(stem) + ".json", "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    steps = [int(s) for s in meta.get("snapshot_steps", [])]
    if not steps:
        raise ValueError(f"{stem} has no snapshots; rerun with --snapshot_every > 0.")
    step = steps[-1] if step is None else int(step)
    pref = f"snap_{step:06d}_"
    with np.load(str(stem) + ".npz") as z:
        snap = {k[len(pref):]: z[k] for k in z.files if k.startswith(pref)}
    return meta, snap


def _kernel_predict(snap: dict, points: np.ndarray) -> np.ndarray:
    Z = np.asarray(snap["Z"]); ell = np.asarray(snap["lengthscales"])
    alpha = np.asarray(snap["alpha"]).reshape(-1)
    sf2 = float(np.asarray(snap["sigma_f"])[0]) ** 2
    def one(a, ls, sf):
        d = points[:, None, :] - Z[None, :, :]
        return sf * np.exp(-0.5 * np.sum((d / ls[None, None, :]) ** 2, axis=-1)) @ a
    value = one(alpha, ell, sf2)
    is_diff = bool(int(np.asarray(snap.get("is_density_diff", [0]))[0]))
    if is_diff:
        value += one(np.asarray(snap["alpha_base"]).reshape(-1),
                     np.asarray(snap["lengthscales_base"]),
                     float(np.asarray(snap["sigma_f_base"])[0]) ** 2)
    is_product = bool(int(np.asarray(snap.get("is_product", [0]))[0]))
    if is_product:
        if bool(int(np.asarray(snap.get("product_transported", [0]))[0])):
            raise NotImplementedError("Projection probes require a static/global product profile.")
        h = float(np.asarray(snap.get("product_hbar", [1.0]))[0])
        active = int(np.asarray(snap.get("product_init_state", [0]))[0])
        ns = int(np.asarray(snap.get("product_nstates", [2]))[0])
        x = points[:, 2:6]
        g = ((np.pi * h) ** (-ns) * np.exp(-np.sum(x*x, axis=1) / h)
             * ((2.0/h) * (x[:, active]**2 + x[:, 2+active]**2) - 1.0))
        value *= g
    return value


def seo_basis_matrix(x: np.ndarray, hbar: float = 1.0) -> np.ndarray:
    """Real two-state SEO basis spanning diagonal/Re/Im density elements."""
    x = np.asarray(x, dtype=float).reshape(-1, 4)
    envelope = (np.pi * hbar) ** -2 * np.exp(-np.sum(x*x, axis=1) / hbar)
    r0, r1, p0, p1 = x.T
    return envelope[:, None] * np.column_stack([
        (2.0/hbar)*(r0*r0+p0*p0)-1.0,
        (2.0/hbar)*(r1*r1+p1*p1)-1.0,
        (2.0/hbar)*(r0*r1+p0*p1),
        (2.0/hbar)*(r0*p1-r1*p0),
    ])


def projection_diagnostic(stem: Path, out: Path, step: Optional[int],
                          n_bath: int, n_mapping: int, seed: int) -> dict[str, Any]:
    meta, snap = _snapshot(stem, step)
    rng = np.random.default_rng(seed)
    Z = np.asarray(snap["Z"])
    bath_idx = rng.choice(Z.shape[0], size=min(n_bath, Z.shape[0]), replace=False)
    x = rng.normal(scale=np.sqrt(0.5), size=(n_mapping, 4))
    B = seo_basis_matrix(x)
    residuals = []
    for idx in bath_idx:
        pts = np.empty((n_mapping, 6)); pts[:, :2] = Z[idx, :2]; pts[:, 2:] = x
        y = _kernel_predict(snap, pts)
        coeff, *_ = np.linalg.lstsq(B, y, rcond=None)
        resid = y - B @ coeff
        residuals.append({"support_index": int(idx),
                          "relative_l2_leakage": float(np.linalg.norm(resid) / max(np.linalg.norm(y), 1e-30)),
                          "absolute_rms_leakage": float(np.sqrt(np.mean(resid*resid)))})
    result = {"source": str(stem), "snapshot_step": int(step if step is not None else max(meta["snapshot_steps"])),
              "n_bath_anchors": len(residuals), "n_mapping_probes": n_mapping,
              "seed": seed, "basis_rank": int(np.linalg.matrix_rank(B)),
              "mean_relative_l2_leakage": float(np.mean([r["relative_l2_leakage"] for r in residuals])),
              "max_relative_l2_leakage": float(np.max([r["relative_l2_leakage"] for r in residuals])),
              "per_anchor": residuals}
    out.mkdir(parents=True, exist_ok=True); write_json(out / "projection_leakage.json", result)
    return result


def _gp_rp_marginal(snap: dict, R: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Analytic R-P marginal on the exact same support used by the KDE."""
    Z=np.asarray(snap["Z"],dtype=float); ell=np.asarray(snap["lengthscales"],dtype=float)
    alpha=np.asarray(snap["alpha"],dtype=float).reshape(-1)
    sf2=float(np.asarray(snap["sigma_f"])[0])**2
    rterm=np.exp(-0.5*((R[None,:]-Z[:,0:1])/ell[0])**2)
    pterm=np.exp(-0.5*((P[None,:]-Z[:,1:2])/ell[1])**2)
    is_product=bool(int(np.asarray(snap.get("is_product",[0]))[0]))
    if is_product:
        if bool(int(np.asarray(snap.get("product_transported",[0]))[0])):
            raise NotImplementedError("Static R-P marginal is undefined for a row-indexed transported profile.")
        h=float(np.asarray(snap.get("product_hbar",[1.0]))[0]); active=int(np.asarray(snap.get("product_init_state",[0]))[0]); ns=int(np.asarray(snap.get("product_nstates",[2]))[0])
        map_factor=np.ones(len(Z)); means=[]; variances=[]
        for d in range(2,6):
            den=h+2*ell[d]**2; var=h*ell[d]**2/den; mean=Z[:,d]*h/den
            map_factor*=np.sqrt(2*np.pi*var)*np.exp(-Z[:,d]**2/den)
            means.append(mean); variances.append(var)
        q=-np.ones(len(Z))
        for j in (active,2+active): q+=(2/h)*(means[j]**2+variances[j])
        coeff=alpha*sf2*(np.pi*h)**(-ns)*map_factor*q
    else:
        coeff=alpha*sf2*float(np.prod(np.sqrt(2*np.pi)*ell[2:]))
    rho=np.einsum("i,ir,ip->pr",coeff,rterm,pterm)
    if bool(int(np.asarray(snap.get("is_density_diff",[0]))[0])):
        ell0=np.asarray(snap["lengthscales_base"],dtype=float); a0=np.asarray(snap["alpha_base"],dtype=float).reshape(-1); sf0=float(np.asarray(snap["sigma_f_base"])[0])**2
        rr=np.exp(-0.5*((R[None,:]-Z[:,0:1])/ell0[0])**2); pp=np.exp(-0.5*((P[None,:]-Z[:,1:2])/ell0[1])**2)
        rho+=np.einsum("i,ir,ip->pr",a0*sf0*float(np.prod(np.sqrt(2*np.pi)*ell0[2:])),rr,pp)
    return rho


def baseline_kde_gp(stem: Path, out: Path, step: Optional[int],
                    n_grid: int = 120) -> dict[str, Any]:
    """Projected KDE-vs-GP baseline on one physical estimator contract.

    The former implementation compared a 2D importance-sampling KDE with an
    unconstrained 6D GP integral over focused mapping coordinates.  Those are
    different mathematical objects.  This routine now uses one saved frozen
    measure, one support cloud, one bandwidth, one grid and one target mass.
    """
    meta,snap=_snapshot(stem,step); Z=np.asarray(snap["Z"],dtype=float); y=np.asarray(snap["y"],dtype=float).reshape(-1)
    geometric=snap.get("geometric_measure")
    proposal=snap.get("proposal_density")
    if geometric is not None:
        omega=np.asarray(geometric,dtype=float).reshape(-1); weight_policy="saved frozen geometric measure"
    elif proposal is not None:
        q=np.asarray(proposal,dtype=float).reshape(-1); omega=1.0/(len(Z)*np.maximum(q,np.finfo(float).tiny)); weight_policy="omega=1/(N q_initial)"
    else:
        omega=np.full(len(Z),1.0/len(Z)); weight_policy="equal weights (legacy snapshot lacked proposal_density)"
    R=np.linspace(np.quantile(Z[:,0],.005)-2*np.std(Z[:,0]),np.quantile(Z[:,0],.995)+2*np.std(Z[:,0]),n_grid)
    P=np.linspace(np.quantile(Z[:,1],.005)-2*np.std(Z[:,1]),np.quantile(Z[:,1],.995)+2*np.std(Z[:,1]),n_grid)
    projected=ProjectedNuclearGP().fit_from_cloud(Z,omega,y,dim_pair=(0,1))
    kde=projected.kde_grid(R,P); gp=projected.gp_grid(R,P)
    gp_meta=projected.metadata()
    # Keep compatibility with both older NumPy (trapz) and newer NumPy
    # releases where trapz has been removed.  The conditional is lazy.
    trap=np.trapezoid if hasattr(np,"trapezoid") else np.trapz
    integ=lambda a: float(trap(trap(a,P,axis=0),R,axis=0))
    norm_kde=integ(kde); norm_gp=integ(gp)
    kde_n=kde/norm_kde if abs(norm_kde)>1e-30 else kde; gp_n=gp/norm_gp if abs(norm_gp)>1e-30 else gp
    diff=gp_n-kde_n
    raw_diff=gp-kde
    errors={"E1":integ(np.abs(diff)),
            "E2":float(np.sqrt(max(integ(diff*diff),0.0))),
            "Einf":float(np.max(np.abs(diff))),
            "raw_E1":integ(np.abs(raw_diff)),
            "raw_Einf":float(np.max(np.abs(raw_diff)))}
    scheme=str(meta.get("scheme",meta.get("run_metadata",{}).get("scheme",stem.name))).lower()
    pbme_contract=("pbme" in scheme or "pbme" in stem.name.lower())
    acceptance={"applies":pbme_contract,"metric":"E1","threshold":0.02,
                "passed":(bool(errors["E1"] <= 0.02) if pbme_contract else None),
                "rationale":"PBME projected-GP shape must reproduce the shared-cloud KDE to within 2% L1 error."}
    result={"source":str(stem),"step":int(step if step is not None else max(meta["snapshot_steps"])),"n_support":len(Z),"initial_cloud_sha256":array_fingerprint(Z),"weight_policy":weight_policy,"estimator_contract":"KDE and sparse 2D GP use identical support, omega*y weights, Scott/Silverman bandwidth, R-P grid, and raw-mass constraint","grid":{"n_R":n_grid,"n_P":n_grid,"R_range":[float(R[0]),float(R[-1])],"P_range":[float(P[0]),float(P[-1])]},"bandwidth":{"R":gp_meta["bandwidth_1"],"P":gp_meta["bandwidth_2"],"policy":"Scott/Silverman d=2 on the shared cloud"},"projected_gp":gp_meta,"raw_norms":{"gp_on_grid":norm_gp,"kde_on_grid":norm_kde,"target_infinite_domain":gp_meta["target_raw_mass"]},"shape_errors":errors,"acceptance":acceptance,"scale_policy":"common raw mass is imposed analytically; normalized arrays are additional shape-only outputs","six_dimensional_gp_note":"The unconstrained 6D mapping integral is excluded here and belongs only in the separate off-manifold leakage diagnostic."}
    out.mkdir(parents=True,exist_ok=True); write_json(out/"kde_gp_identical_support.json",result)
    np.savez_compressed(out/"kde_gp_identical_support.npz",R=R,P=P,gp=gp,kde=kde,gp_normalized=gp_n,kde_normalized=kde_n)
    return result


def manufactured_exact(Z: np.ndarray, *, R0=0.0, P0=8.0,
                       sigma_R=1.2, sigma_P=0.7, hbar=1.0,
                       init_state=0):
    """Exact manufactured density, gradient, and excess operator Q."""
    from GP_Density import seo_profile_derivs
    from Models import TullyModel, TullyParams
    Z = np.asarray(Z, dtype=float).reshape(-1, 6)
    R, P, x = Z[:, 0], Z[:, 1], Z[:, 2:]
    W = np.exp(-0.5*((R-R0)/sigma_R)**2 - 0.5*((P-P0)/sigma_P)**2)
    g, dg, d2g = seo_profile_derivs(x, hbar, init_state, 2)
    rho = W*g
    grad = np.zeros_like(Z)
    grad[:, 0] = -(R-R0)/sigma_R**2 * rho
    grad[:, 1] = -(P-P0)/sigma_P**2 * rho
    grad[:, 2:] = W[:, None]*dg
    dH = TullyModel(TullyParams.defaults("dual")).d_diabatic_potential_dR(R)
    tr = 0.5*(dH[:,0,0]+dH[:,1,1]); dh=dH.copy(); dh[:,0,0]-=tr; dh[:,1,1]-=tr
    dW_dP = -(P-P0)/sigma_P**2 * W
    contraction = np.zeros(len(Z))
    for a in range(2):
        for b in range(2):
            contraction += dh[:,a,b]*(d2g[:,a,b] + d2g[:,2+a,2+b])
    Q = -(hbar/8.0)*dW_dP*contraction
    return rho, grad, Q


def manufactured_test(out: Path, n_train: int, n_query: int, seed: int,
                      l2: float = 1.0e-6) -> dict:
    """Fit the product surrogate and compare rho, grad, Q on/off support."""
    try:
        import torch  # noqa: F401
        import jax  # noqa: F401
    except Exception as exc:
        raise RuntimeError("manufactured test requires torch and jax; install requirements.txt") from exc
    from GP_Density import GPDensity, GPDensityConfig, GPDensityProduct
    from GPDerivatives import rho_derivative_bundle
    from Mint import PBMEMIntDynamics, PBMEMIntParams
    from Models import TullyModel, TullyParams
    from Operator import compute_Q_at_points
    rng = np.random.default_rng(seed)
    def points(n):
        Z = np.empty((n,6)); Z[:,0]=rng.normal(0,1.2,n); Z[:,1]=rng.normal(8,0.7,n)
        Z[:,2:]=rng.normal(0,np.sqrt(0.5),(n,4)); return Z
    Zt=points(n_train); yt,_,_=manufactured_exact(Zt)
    dyn=PBMEMIntDynamics(TullyModel(TullyParams.defaults("dual")), PBMEMIntParams())
    cfg=GPDensityConfig(n_opt_steps=80, constraints_enabled=False,
                        refit_hyper_policy="frozen", fix_sigma_n=False,
                        l2_regularization=float(l2))
    gp=GPDensityProduct(GPDensity(cfg,dynamics=dyn), hbar=1.0, init_state=0)
    gp.fit(Zt,yt,moment_targets=None,apply_constraints=False)
    sets={"on_support":Zt, "off_support":points(n_query)}
    results={}
    for name,Z in sets.items():
        rho,grad,Q=manufactured_exact(Z); pred=gp.predict(Z)
        grad_gp,_,_=rho_derivative_bundle(gp._inner,Z)
        g,dg,_=gp.profile_derivs_current(Z)
        mu=gp._inner.predict(Z); grad_product=dg*mu[:,None]+g[:,None]*grad_gp
        Q_gp=compute_Q_at_points(Z,gp,dyn)[0]
        def metric(ref,cand):
            d=np.asarray(cand)-np.asarray(ref)
            ref=np.asarray(ref)
            ref_l1=float(np.sum(np.abs(ref)))
            ref_l2=float(np.linalg.norm(ref))
            ref_linf=float(np.max(np.abs(ref)))
            floor=1.0e-30
            return {
                "mae":float(np.mean(np.abs(d))),
                "rmse":float(np.sqrt(np.mean(d*d))),
                "linf":float(np.max(np.abs(d))),
                "relative_l1":float(np.sum(np.abs(d))/max(ref_l1,floor)),
                "relative_l2":float(np.linalg.norm(d)/max(ref_l2,floor)),
                "relative_linf":float(np.max(np.abs(d))/max(ref_linf,floor)),
                "denominator_floor":floor,
                "denominator_floor_used":bool(
                    ref_l1 <= floor or ref_l2 <= floor or ref_linf <= floor
                ),
            }
        results[name]={"density":metric(rho,pred),"gradient":metric(grad,grad_product),
                       "operator_Q":metric(Q,Q_gp),
                       "query_count":int(len(Z)),
                       "query_sha256":array_fingerprint(Z)}
    inner=gp._inner
    result={
        "n_train":int(n_train),
        "n_query_off_support":int(n_query),
        "n_query_on_support":int(n_train),
        "seed":int(seed),
        "training_cloud_sha256":array_fingerprint(Zt),
        "l2_regularization":float(l2),
        "cholesky_jitter":float(inner.last_cholesky_effective_jitter),
        "cholesky_adaptive_jitter":float(inner.last_cholesky_adaptive_jitter),
        "cholesky_attempts":int(inner.last_cholesky_attempts),
        "minimum_eigenvalue_estimate":float(inner.last_cholesky_min_eigenvalue),
        "minimum_eigenvalue_estimate_method":"Gershgorin lower bound",
        "sigma_n":float(inner.sigma_n),
        "sigma_f":float(inner.sigma_f),
        "lengthscales":[float(v) for v in inner.lengthscales],
        "input_scaling":{
            "feature_zscore":bool(inner.config.feature_zscore),
            "feature_mean":(
                None if inner._feature_mean is None
                else [float(v) for v in inner._feature_mean.detach().cpu().numpy()]
            ),
            "feature_std":(
                None if inner._feature_std is None
                else [float(v) for v in inner._feature_std.detach().cpu().numpy()]
            ),
        },
        "hyperparameter_policy":"optimized once, then frozen",
        "dtype":"float64",
        "metrics":results,
    }
    out.mkdir(parents=True,exist_ok=True); write_json(out/"manufactured_operator_metrics.json",result)
    return result


def reference_convergence(out: Path, dt: float, n_steps: int, P0: float) -> dict:
    """Run explicit TDSE time/grid and QCLE time/support refinements."""
    from Compare_gp_se_qcle import run_tdse, run_qcle
    from qcle_grid_tully import QCLEGridParams
    out.mkdir(parents=True, exist_ok=True)
    tdse = {}
    for label, h, ngrid in (("coarse", dt, 2048), ("fine", dt/2, 4096)):
        tdse[label] = run_tdse(-10.0, P0, 1.0, h,
                               int(round(n_steps*dt/h)), n_grid_min=ngrid,
                               save_every=max(1,int(round(n_steps*dt/h))), verbose=False)
    qcle = {}
    for label, h, nr, np_ in (("coarse", dt, 192, 128), ("fine", dt/2, 384, 256)):
        params=QCLEGridParams(R_min=-25,R_max=25,P_min=-35,P_max=35,
                              n_R=nr,n_P=np_,mass=2000,hbar=1)
        qcle[label]=run_qcle(-10.0,P0,1.0,h,int(round(n_steps*dt/h)),
                             qcle_params=params,save_every=max(1,int(round(n_steps*dt/h))),verbose=False)
    result={}
    for method,data in (("tdse",tdse),("qcle",qcle)):
        result[method]={}
        for key in ("P0","P1","trace","energy","R_mean","P_mean"):
            a=float(data["coarse"][key][-1]); b=float(data["fine"][key][-1])
            result[method][key]={"coarse":a,"fine":b,"absolute_difference":abs(a-b)}
    result["configuration"]={"dt_coarse":dt,"dt_fine":dt/2,"n_steps_coarse":n_steps,
                             "P0":P0,"tdse_grids":[2048,4096],"qcle_grids":[[192,128],[384,256]]}
    write_json(out/"reference_convergence.json",result); return result


# =============================================================================
# Master validation table
# =============================================================================
#
# Consolidates every machine-readable validation artifact under an output tree
# into ONE long-form table (test, source, item, quantity, value) plus a
# rendered Markdown view.  Each known artifact has a dedicated flattener so the
# rows are labelled meaningfully; any other *.json is deep-flattened generically
# so no numerical result is silently dropped.  Reads only JSON/CSV — no torch.

_MASTER_COLUMNS = ("test", "source", "item", "quantity", "value")


def _fmt_value(v: Any) -> str:
    """Render a scalar for the Markdown table; leave strings/None readable."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v != v:                      # NaN
            return "nan"
        if v == 0.0:
            return "0"
        av = abs(v)
        return f"{v:.6g}" if (1e-4 <= av < 1e6) else f"{v:.6e}"
    return "" if v is None else str(v)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _deep_rows(test: str, source: str, obj: Any, prefix: str = ""):
    """Yield (test, source, item, quantity, value) rows for every scalar leaf."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            yield from _deep_rows(test, source, v, key)
    elif isinstance(obj, (list, tuple)):
        # Only descend into lists of dicts/scalars that carry numbers; index them.
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            yield from _deep_rows(test, source, v, key)
    else:
        item, _, quantity = prefix.rpartition(".")
        if not quantity:
            quantity = prefix
        yield {"test": test, "source": source, "item": item,
               "quantity": quantity, "value": obj}


def _rows_campaign_metrics(path: Path):
    test, src = "convergence_campaign", path.name
    data = json.loads(path.read_text(encoding="utf-8"))
    yield {"test": test, "source": src, "item": "",
           "quantity": "n_completed_scheme_runs",
           "value": data.get("n_completed_scheme_runs")}
    for scheme in ("pbme", "midpoint"):
        for comp in data.get(f"{scheme}_convergence", []):
            item = f"{scheme}:{comp.get('case_id')} vs {comp.get('against')}"
            for q, v in (comp.get("absolute_endpoint_differences") or {}).items():
                yield {"test": test, "source": src, "item": item,
                       "quantity": f"|Δ endpoint| {q}", "value": v}
        rep = data.get(f"{scheme}_replication")
        if rep:
            for q, stats in rep.items():
                for stat_name, val in stats.items():
                    yield {"test": test, "source": src,
                           "item": f"{scheme}:replication",
                           "quantity": f"{q}.{stat_name}", "value": val}


def _rows_endpoint_csv(path: Path):
    test, src = "convergence_campaign", path.name
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            case = f"{row.get('case_id')}:{row.get('scheme')}"
            for q, v in row.items():
                if q in ("case_id", "scheme"):
                    continue
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    pass
                yield {"test": test, "source": src, "item": case,
                       "quantity": q, "value": v}


def _rows_manufactured(path: Path):
    test, src = "manufactured_operator", path.name
    data = json.loads(path.read_text(encoding="utf-8"))
    for meta_k in ("n_train", "n_query", "seed"):
        yield {"test": test, "source": src, "item": "config",
               "quantity": meta_k, "value": data.get(meta_k)}
    for support, fields in (data.get("metrics") or {}).items():
        for field_name, metrics in fields.items():
            for q, v in metrics.items():
                yield {"test": test, "source": src,
                       "item": f"{support}:{field_name}",
                       "quantity": q, "value": v}


def _rows_projection(path: Path):
    test, src = "seo_projection_leakage", path.name
    data = json.loads(path.read_text(encoding="utf-8"))
    for q in ("snapshot_step", "n_bath_anchors", "n_mapping_probes",
              "basis_rank", "mean_relative_l2_leakage", "max_relative_l2_leakage"):
        if q in data:
            yield {"test": test, "source": src, "item": str(data.get("source", "")),
                   "quantity": q, "value": data[q]}


def _rows_kde_gp(path: Path):
    test, src = "kde_gp_identical_support", path.name
    data = json.loads(path.read_text(encoding="utf-8"))
    item = str(data.get("source", ""))
    for q in ("n_support", "weight_policy"):
        if q in data:
            yield {"test": test, "source": src, "item": item,
                   "quantity": q, "value": data[q]}
    for q, v in (data.get("shape_errors") or {}).items():
        yield {"test": test, "source": src, "item": f"{item}:shape_error",
               "quantity": q, "value": v}
    for q, v in (data.get("raw_norms") or {}).items():
        yield {"test": test, "source": src, "item": f"{item}:raw_norm",
               "quantity": q, "value": v}
    acc = data.get("acceptance") or {}
    for q in ("applies", "metric", "threshold", "passed"):
        if q in acc:
            yield {"test": test, "source": src, "item": f"{item}:acceptance",
                   "quantity": q, "value": acc[q]}


def _rows_reference(path: Path):
    test, src = "reference_convergence", path.name
    data = json.loads(path.read_text(encoding="utf-8"))
    for method in ("tdse", "qcle"):
        for q, stats in (data.get(method) or {}).items():
            for stat_name, val in stats.items():
                yield {"test": test, "source": src, "item": f"{method}:{q}",
                       "quantity": stat_name, "value": val}


# filename -> dedicated flattener
_KNOWN_ARTIFACTS = {
    "campaign_metrics.json": _rows_campaign_metrics,
    "endpoint_metrics.csv": _rows_endpoint_csv,
    "manufactured_operator_metrics.json": _rows_manufactured,
    "projection_leakage.json": _rows_projection,
    "kde_gp_identical_support.json": _rows_kde_gp,
    "reference_convergence.json": _rows_reference,
}
# Artifacts intentionally NOT deep-flattened generically (bulky/plumbing).
_SKIP_GENERIC = {"campaign_plan.json", "campaign_status.json"}


def build_master_table(out: Path, include_unknown_json: bool = True) -> dict[str, Any]:
    """
    Aggregate every validation artifact under *out* into one master table.

    Writes ``master_validation_table.csv`` (long form) and
    ``master_validation_table.md`` (grouped by test) into *out* and returns a
    summary dict.  Recurses into subdirectories, so per-test outputs written to
    ``out/manufactured``, ``out/projection`` etc. are all picked up.
    """
    out = Path(out)
    if not out.exists():
        raise FileNotFoundError(f"Output tree {out} does not exist.")

    rows: list[dict] = []
    seen: set[Path] = set()

    # 1. Known artifacts (dedicated, well-labelled flatteners).
    for fname, flat in _KNOWN_ARTIFACTS.items():
        for path in sorted(out.rglob(fname)):
            seen.add(path.resolve())
            try:
                rows.extend(flat(path))
            except Exception as exc:                       # pragma: no cover
                rows.append({"test": "PARSE_ERROR", "source": str(path),
                             "item": "", "quantity": type(exc).__name__,
                             "value": str(exc)})

    # 2. Generic fallback for any other JSON (nothing silently dropped).
    if include_unknown_json:
        for path in sorted(out.rglob("*.json")):
            if path.resolve() in seen or path.name in _SKIP_GENERIC:
                continue
            if path.name.endswith(".meta.json"):           # figure sidecars
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.extend(_deep_rows(f"other:{path.stem}", path.name, data))

    # Stable ordering: by test, then source, then item, then quantity.
    rows.sort(key=lambda r: (str(r["test"]), str(r["source"]),
                             str(r["item"]), str(r["quantity"])))

    csv_path = out / "master_validation_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MASTER_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in _MASTER_COLUMNS})

    md_path = out / "master_validation_table.md"
    tests = sorted({str(r["test"]) for r in rows})
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Master validation table\n\n")
        fh.write(f"Generated from `{out}` — {len(rows)} numerical results "
                 f"across {len(tests)} test group(s).\n\n")
        if not rows:
            fh.write("_No validation artifacts found. Run the plan/campaign/"
                     "manufactured/projection/baseline/reference commands first._\n")
        for test in tests:
            trows = [r for r in rows if str(r["test"]) == test]
            fh.write(f"## {test}  ({len(trows)} results)\n\n")
            fh.write("| source | item | quantity | value |\n")
            fh.write("|---|---|---|---|\n")
            def _cell(x):
                return str(x).replace("|", "\\|")
            for r in trows:
                fh.write(f"| {_cell(r['source'])} | {_cell(r['item'])} "
                         f"| {_cell(r['quantity'])} | {_cell(_fmt_value(r['value']))} |\n")
            fh.write("\n")

    summary = {"n_rows": len(rows), "n_test_groups": len(tests),
               "tests": tests, "csv": str(csv_path), "markdown": str(md_path)}
    print(f"[report] {len(rows)} results across {len(tests)} test group(s)")
    print(f"[report] wrote {csv_path}")
    print(f"[report] wrote {md_path}")
    return summary


def add_campaign_args(p):
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-base", type=int, default=500)
    p.add_argument("--dt-base", type=float, default=0.5)
    p.add_argument("--seeds", type=int, nargs="+", default=[11,29,47])
    p.add_argument("--t-final", type=float, default=200.0)
    p.add_argument("--density-mode", choices=["full","diff"], default="full")
    p.add_argument("--sampling-mode", choices=["focused","seo_signed"], default="focused")
    p.add_argument("--surrogate", choices=["gp","product","product_transported"], default="product")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("plan"); add_campaign_args(p)
    p=sub.add_parser("campaign"); add_campaign_args(p); p.add_argument("--execute",action="store_true"); p.add_argument("--keep-going",action="store_true")
    p=sub.add_parser("analyze"); p.add_argument("--out",type=Path,required=True)
    p=sub.add_parser("projection"); p.add_argument("stem",type=Path); p.add_argument("--out",type=Path,required=True); p.add_argument("--step",type=int); p.add_argument("--n-bath",type=int,default=20); p.add_argument("--n-mapping",type=int,default=400); p.add_argument("--seed",type=int,default=123)
    p=sub.add_parser("manufactured"); p.add_argument("--out",type=Path,required=True); p.add_argument("--n-train",type=int,default=600); p.add_argument("--n-query",type=int,default=100); p.add_argument("--seed",type=int,default=123); p.add_argument("--l2",type=float,default=1.0e-6)
    p=sub.add_parser("baseline"); p.add_argument("stem",type=Path); p.add_argument("--out",type=Path,required=True); p.add_argument("--step",type=int); p.add_argument("--n-grid",type=int,default=120)
    p=sub.add_parser("reference"); p.add_argument("--out",type=Path,required=True); p.add_argument("--dt",type=float,default=0.2); p.add_argument("--n-steps",type=int,default=200); p.add_argument("--P0",type=float,default=20.0)
    p=sub.add_parser("report"); p.add_argument("--out",type=Path,required=True); p.add_argument("--no-unknown-json",action="store_true",help="Skip generic flattening of unrecognized JSON files.")
    args=parser.parse_args()
    if args.command=="plan": write_campaign_plan(args.out,args)
    elif args.command=="campaign": run_campaign(args.out,args)
    elif args.command=="analyze": analyze_campaign(args.out)
    elif args.command=="projection": projection_diagnostic(args.stem,args.out,args.step,args.n_bath,args.n_mapping,args.seed)
    elif args.command=="manufactured": manufactured_test(args.out,args.n_train,args.n_query,args.seed,args.l2)
    elif args.command=="baseline": baseline_kde_gp(args.stem,args.out,args.step,args.n_grid)
    elif args.command=="reference": reference_convergence(args.out,args.dt,args.n_steps,args.P0)
    elif args.command=="report": build_master_table(args.out, include_unknown_json=not args.no_unknown_json)


if __name__ == "__main__":
    main()
