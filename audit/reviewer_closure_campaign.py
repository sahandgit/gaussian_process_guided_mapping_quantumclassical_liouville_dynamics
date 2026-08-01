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
reviewer_closure_campaign.py
============================

Runnable driver that implements Steps 3-12 of the reviewer-closure
specification by orchestrating the EXISTING pipeline entry points
(``run.py`` and ``ReviewerValidation.py``).  It does not reimplement any
physics and it never fabricates numbers: it constructs the exact commands,
runs them into a fresh timestamped directory, records manifests, and then
performs the torch-free *analysis* the spec prescribes (observed order,
nested-vs-seed comparison, raw-drift extraction, E1 acceptance).

Design guarantees (mirroring the spec's non-negotiable rules)
-------------------------------------------------------------
* **Never invent results.** Missing prerequisites yield ``NOT RUN`` /
  ``INSUFFICIENT EVIDENCE`` in the step manifest; nothing is faked.
* **Never overwrite production results.** Everything goes under a new
  ``reviewer_closure_<UTC timestamp>`` root.
* **One control at a time.** The refinement builders vary exactly one of
  {dt, N, seed} and hold everything else fixed.
* **Raw conservation.** Analysis reads the ``raw_*`` drift arrays, not the
  self-normalized ones.
* **Three levels + observed order**, with guards that refuse to report an order
  when the denominator is zero or dominated by seed noise.

Modes
-----
* ``--mode dry-run``  (default): print the full command plan; run nothing.
* ``--mode execute`` : actually run (requires torch on this machine).
* ``--mode self-test``: run the pure-helper unit checks and exit.

The physics runs require PyTorch + JAX (see requirements.txt).  The analysis
helpers and command construction are pure NumPy/stdlib and are unit-tested in
``test_reviewer_closure.py``.
"""

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ===========================================================================
# Pure helpers  (unit-tested; no torch)
# ===========================================================================

def utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit(cwd: Path) -> Optional[str]:
    """Return HEAD sha, or None if not a git repo (never invent one)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_dirty(cwd: Path) -> Optional[bool]:
    try:
        out = subprocess.run(["git", "status", "--short"], cwd=str(cwd),
                             capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        return bool(out.stdout.strip())
    except Exception:
        return None


def package_versions() -> Dict[str, Optional[str]]:
    vers: Dict[str, Optional[str]] = {}
    for mod in ("numpy", "scipy", "torch", "jax", "matplotlib"):
        try:
            m = __import__(mod)
            vers[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            vers[mod] = None
    return vers


def environment_manifest(argv: Sequence[str], repo: Path) -> Dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "executable": sys.executable,
        "packages": package_versions(),
        "git_commit": git_commit(repo),
        "git_dirty_worktree": git_dirty(repo),
        "platform": platform.platform(),
        "command_line": list(argv),
        "utc_start": utc_iso(),
    }


def collision_time(mass: float, R0: float, P0: float) -> float:
    r"""Scattering time  t_c = M |R0| / |P0|."""
    p = abs(float(P0))
    if p <= 0.0:
        raise ValueError("P0 must be non-zero for a scattering time.")
    return float(mass) * abs(float(R0)) / p


def array_sha256(a: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    h = sha256()
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def nested_subset_indices(n_max: int, levels: Sequence[int],
                          seed: int) -> Dict[int, np.ndarray]:
    r"""
    Deterministic **nested** support subsets.

    Returns {level_size -> sorted index array} where each smaller level is a
    strict subset of every larger level (Step 8's requirement: the N=500 cloud
    is a subset of the N=1000 cloud is a subset of N=2000).  Implemented by
    permuting ``range(n_max)`` once with a fixed seed and taking prefixes.
    """
    levels = sorted({int(v) for v in levels})
    if levels[-1] > n_max:
        raise ValueError(f"largest level {levels[-1]} exceeds n_max {n_max}")
    perm = np.random.default_rng(seed).permutation(n_max)
    out: Dict[int, np.ndarray] = {}
    for L in levels:
        out[L] = np.sort(perm[:L])
    # verify nesting (prefix property under the shared permutation)
    for small, large in zip(levels, levels[1:]):
        assert set(out[small].tolist()) <= set(out[large].tolist())
    return out


def shell_distance(X: np.ndarray, S: np.ndarray, ell: np.ndarray) -> np.ndarray:
    r"""
    Normalized-kernel-coordinate distance of each point in ``X`` to the support
    ``S``:  d_ell(x, S) = min_i sqrt( sum_d (x_d - s_{i,d})^2 / ell_d^2 ).
    """
    X = np.asarray(X, float); S = np.asarray(S, float)
    ell = np.asarray(ell, float).reshape(-1)
    ell = np.where(ell > 0, ell, 1.0)
    Xs = X / ell; Ss = S / ell
    d2 = (np.sum(Xs**2, 1)[:, None] + np.sum(Ss**2, 1)[None, :]
          - 2.0 * Xs @ Ss.T)
    return np.sqrt(np.clip(d2.min(axis=1), 0.0, None))


def shell_bin_indices(dist: np.ndarray,
                      edges: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 4.0)
                      ) -> Dict[str, np.ndarray]:
    dist = np.asarray(dist, float)
    bins: Dict[str, np.ndarray] = {}
    for lo, hi in zip(edges, edges[1:]):
        label = f"[{lo:g},{hi:g})"
        bins[label] = np.where((dist >= lo) & (dist < hi))[0]
    return bins


def observed_order(u_h: np.ndarray, u_h2: np.ndarray, u_h4: np.ndarray,
                   ord: Optional[int] = None,
                   seed_noise: Optional[float] = None,
                   rel_floor: float = 1e-14) -> Tuple[Optional[float], str]:
    r"""
    Observed refinement order from three levels:

        p_obs = log2( ||u_h - u_h2|| / ||u_h2 - u_h4|| ).

    Returns (p_obs or None, reason).  Refuses to report when the denominator is
    ~0 or when either difference is below the supplied independent-seed noise
    (Step 7's guard against reporting seed noise as an order).
    """
    a = float(np.linalg.norm(np.asarray(u_h) - np.asarray(u_h2), ord=ord))
    b = float(np.linalg.norm(np.asarray(u_h2) - np.asarray(u_h4), ord=ord))
    scale = max(float(np.linalg.norm(np.asarray(u_h4), ord=ord)), 1.0)
    if b <= rel_floor * scale:
        return None, "INSUFFICIENT EVIDENCE: finer difference ~0 (denominator underflow)"
    if a <= rel_floor * scale:
        # Coarse and mid levels are identical: log2(0/b) = -inf. Refuse rather
        # than emit a spurious large negative order (and a numpy warning).
        return None, ("INSUFFICIENT EVIDENCE: coarse-to-mid difference ~0 "
                      "(numerator underflow)")
    if seed_noise is not None and (a < seed_noise or b < seed_noise):
        return None, "INSUFFICIENT EVIDENCE: refinement differences below seed noise"
    return float(np.log2(a / b)), "ok"


def interp_to_grid(t_src: np.ndarray, u_src: np.ndarray,
                   t_dst: np.ndarray) -> np.ndarray:
    """Linear interpolation of a time series onto a (coarser) common grid."""
    t_src = np.asarray(t_src, float); u_src = np.asarray(u_src, float)
    return np.interp(np.asarray(t_dst, float), t_src, u_src)


def timeseries_norms(a: np.ndarray, b: np.ndarray,
                     t: Optional[np.ndarray] = None) -> Dict[str, float]:
    """L1, L2, Linf differences of two aligned time series (trapezoid L1/L2)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = np.abs(a - b)
    if t is not None:
        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        span = max(float(t[-1] - t[0]), 1e-30)
        L1 = float(trap(d, t) / span)
        L2 = float(np.sqrt(trap(d * d, t) / span))
    else:
        L1 = float(np.mean(d)); L2 = float(np.sqrt(np.mean(d * d)))
    return {"L1": L1, "L2": L2, "Linf": float(np.max(d) if d.size else 0.0)}


def raw_drift_summary(t: np.ndarray, drift: np.ndarray,
                      t_c: Optional[float] = None) -> Dict[str, Any]:
    """
    Endpoint, max-abs, time-of-max, and (if t_c given) pre/inter/post-interaction
    maxima of a raw cumulative drift curve.
    """
    t = np.asarray(t, float); d = np.asarray(drift, float)
    out: Dict[str, Any] = {
        "endpoint": float(d[-1]) if d.size else float("nan"),
        "max_abs": float(np.max(np.abs(d))) if d.size else float("nan"),
        "t_at_max_abs": float(t[int(np.argmax(np.abs(d)))]) if d.size else float("nan"),
    }
    if t_c is not None and d.size:
        pre = d[t < t_c]; inter = d[(t >= t_c) & (t < 2 * t_c)]; post = d[t >= 2 * t_c]
        out["pre_interaction_max_abs"] = float(np.max(np.abs(pre))) if pre.size else None
        out["interaction_max_abs"] = float(np.max(np.abs(inter))) if inter.size else None
        out["post_interaction_max_abs"] = float(np.max(np.abs(post))) if post.size else None
    return out


def grid_shape_errors(gp: np.ndarray, kde: np.ndarray,
                      R: np.ndarray, P: np.ndarray) -> Dict[str, float]:
    r"""E1, E2, Einf between two normalized R-P fields (Step 11)."""
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    integ = lambda a: float(trap(trap(a, P, axis=0), R, axis=0))
    diff = np.asarray(gp, float) - np.asarray(kde, float)
    return {"E1": integ(np.abs(diff)),
            "E2": float(np.sqrt(max(integ(diff * diff), 0.0))),
            "Einf": float(np.max(np.abs(diff)))}


def convergence_slopes(support_levels: Sequence[int],
                       errors: Sequence[float]) -> Optional[float]:
    """Empirical error slope d(log err)/d(log N) from the last three levels."""
    N = np.asarray(support_levels, float); e = np.asarray(errors, float)
    m = np.isfinite(e) & (e > 0) & (N > 0)
    if m.sum() < 2:
        return None
    N, e = N[m][-3:], e[m][-3:]
    if N.size < 2:
        return None
    return float(np.polyfit(np.log(N), np.log(e), 1)[0])


# ===========================================================================
# Command builders
# ===========================================================================

def run_py_cmd(python: str, run_py: Path, out: Path, *, P0: float, n_train: int,
               dt: float, t_final: float, seed: int, snapshot_every: int,
               density_mode: str, sampling_mode: str, surrogate: str,
               l2_regularization: float, R0: float, sigma_R: float,
               mass: float, hbar: float, abs_target: bool,
               refit_hyper_policy: str, extra: Sequence[str] = (),
               quiet: bool = True) -> List[str]:
    """Construct one paired PBME+MIDPOINT production run.  --no_auto_dt so dt is
    exactly as prescribed (one control at a time)."""
    cmd = [python, str(run_py), "--out", str(out),
           "--P0", repr(float(P0)), "--n_train", str(int(n_train)),
           "--dt", repr(float(dt)), "--no_auto_dt", "--t_final", repr(float(t_final)),
           "--seed", str(int(seed)), "--snapshot_every", str(int(snapshot_every)),
           "--density_mode", density_mode, "--sampling_mode", sampling_mode,
           "--surrogate", surrogate,
           "--l2_regularization", repr(float(l2_regularization)),
           "--R0", repr(float(R0)), "--sigma_R", repr(float(sigma_R)),
           "--mass", repr(float(mass)), "--hbar", repr(float(hbar)),
           "--refit_hyper_policy", refit_hyper_policy,
           "--skip_figures",
           ("--abs_target" if abs_target else "--no_abs_target")]
    if quiet:
        cmd.append("--quiet")
    cmd += list(extra)
    return cmd


def rv_cmd(python: str, rv_py: Path, sub: str, **kw) -> List[str]:
    cmd = [python, str(rv_py), sub]
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        elif isinstance(v, (list, tuple)):
            cmd.append(flag); cmd += [str(x) for x in v]
        else:
            cmd += [flag, str(v)]
    return cmd


def stem_tag(stem: Path, n_parts: int = 3) -> str:
    """
    Directory/label tag for a run stem, built from the last ``n_parts`` path
    components.

    Run stems end in a generic basename ('pbme'), so keying outputs on the name
    alone collides.  One parent level is not enough either:

        step9_repl_P020/seed11/pbme
        step9_repl_P0100/seed11/pbme

    both end in 'seed11/pbme', so a 2-part tag still collides and the second
    result silently overwrites the first.  Three parts separates them.  Use
    ``unique_stem_tags`` when tagging a whole list - it guarantees uniqueness
    even if three parts are somehow still ambiguous.
    """
    parts = [p for p in Path(stem).parts if p not in ("/", "\\")]
    parts = [p for p in parts if not p.endswith(":")]        # drop drive letter
    return "_".join(parts[-max(1, int(n_parts)):]) if parts else str(stem)


def unique_stem_tags(stems: Sequence[Path], n_parts: int = 3) -> List[str]:
    """
    Tags for a list of stems, guaranteed collision-free.

    Falls back to appending a short hash of the full path for any tag that
    would otherwise repeat, so two inputs can never share an output directory.
    """
    tags = [stem_tag(s, n_parts) for s in stems]
    counts: Dict[str, int] = {}
    for t in tags:
        counts[t] = counts.get(t, 0) + 1
    out: List[str] = []
    for s, t in zip(stems, tags):
        if counts[t] > 1:
            h = sha256(str(s).encode("utf-8")).hexdigest()[:6]
            out.append(f"{t}_{h}")
        else:
            out.append(t)
    return out


def total_ram_gb() -> Optional[float]:
    """
    Physical RAM in GiB, without requiring psutil.

    Order: psutil (if present) -> Windows GlobalMemoryStatusEx via ctypes ->
    POSIX sysconf.  Returns None if it cannot be determined (callers must then
    fall back to the user's explicit --jobs rather than guessing).
    """
    try:
        import psutil                                  # type: ignore
        return float(psutil.virtual_memory().total) / 1024 ** 3
    except Exception:
        pass
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class _MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MEMSTAT()
            st.dwLength = ctypes.sizeof(_MEMSTAT)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return float(st.ullTotalPhys) / 1024 ** 3
        except Exception:
            pass
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                / 1024 ** 3)
    except (AttributeError, ValueError, OSError):
        return None


def recommended_jobs(ram_gb: Optional[float] = None,
                     per_job_gb: float = 4.0,
                     reserve_gb: float = 4.0) -> Optional[int]:
    """
    Memory-safe worker count.

    Each child imports torch AND jax and holds N x N GP matrices plus autograd
    graphs during hyperparameter optimisation.  MEASURED: 4 workers on a 16 GB
    machine exhausted memory and the parent process was killed, so the earlier
    2.5 GB/worker figure was optimistic; 4 GB is the calibrated value.  The
    heaviest single job in the campaign is the Step 5 manufactured fit at
    N_train=2400 - if RAM is tight, run Step 5 on its own with --jobs 1.

    Exceeding RAM does not degrade gracefully: children die with MemoryError
    (often inside ``import jax``), and the parent can be killed outright.
    """
    if ram_gb is None:
        ram_gb = total_ram_gb()
    if ram_gb is None:
        return None
    return max(1, int((float(ram_gb) - reserve_gb) // per_job_gb))


_OOM_MARKERS = ("MemoryError", "Unable to allocate", "_ArrayMemoryError",
                "std::bad_alloc", "CUDA out of memory")


def log_indicates_oom(log_path: Optional[str]) -> bool:
    """True if a child's log shows an out-of-memory death (retryable)."""
    if not log_path:
        return False
    p = Path(log_path)
    if not p.exists():
        return False
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    return any(m in txt for m in _OOM_MARKERS)


def snapshot_every_for(t_c: float, dt: float) -> int:
    """Choose snapshot stride so snapshots land on t_c and 2 t_c."""
    steps_to_tc = int(round(t_c / dt))
    return max(1, steps_to_tc)


# ===========================================================================
# Campaign orchestrator
# ===========================================================================

@dataclass
class Campaign:
    repo: Path
    root: Path
    python: str
    mode: str                     # "dry-run" | "execute"
    jobs: int = 1                 # parallel workers in execute mode
    resume: bool = False          # skip cases whose completion artifact exists
    allow_oversubscribe: bool = False   # bypass the RAM-based --jobs clamp
    start_stagger_s: float = 5.0        # delay between child launches
    heartbeat_s: float = 60.0           # progress ping while jobs are running
    per_job_gb: float = 4.0             # calibrated peak RSS per child
    manifest: Dict[str, Any] = field(default_factory=dict)
    plan: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.run_py = self.repo / "run.py"
        self.rv_py = self.repo / "ReviewerValidation.py"
        self.queue: List[tuple] = []
        self._start_lock = threading.RLock()
        self._running: Dict[str, float] = {}
        self._stop_heartbeat = threading.Event()
        self.jobs_requested = int(self.jobs)

        # Memory guard.  Oversubscribing RAM does not slow the campaign down --
        # children are killed with MemoryError (often inside `import jax`), so
        # clamp to what this machine can actually hold unless overridden.
        ram = total_ram_gb()
        safe = recommended_jobs(ram, per_job_gb=self.per_job_gb)
        if safe is not None and self.jobs > safe and not self.allow_oversubscribe:
            print(f"[campaign] --jobs {self.jobs} needs ~"
                  f"{self.per_job_gb*self.jobs:.0f} GB but this machine has "
                  f"{ram:.0f} GB; clamping to --jobs {safe} to avoid "
                  f"MemoryError. Override with --allow-oversubscribe.")
            self.jobs = safe

        self.manifest.setdefault("environment",
                                 environment_manifest(sys.argv, self.repo))
        self.manifest.setdefault("steps", {})
        self.manifest["environment"].update({
            "jobs_requested": self.jobs_requested,
            "jobs_effective": int(self.jobs),
            "total_ram_gb": ram,
            "memory_safe_jobs": safe,
            "resume": bool(self.resume),
        })

    def _completion_marker(self, cmd: List[str], out_dir: Path) -> Optional[Path]:
        """The artifact whose presence means this command already finished."""
        s = " ".join(cmd)
        if str(self.run_py) in s:
            return out_dir / "pbme.npz"
        if "manufactured" in cmd:
            return out_dir / "manufactured_operator_metrics.json"
        if "projection" in cmd:
            return out_dir / "projection_leakage.json"
        if "baseline" in cmd:
            return out_dir / "kde_gp_identical_support.json"
        if "reference" in cmd:
            return out_dir / "reference_convergence.json"
        return None

    # -- execution -----------------------------------------------------------
    def _record(self, label: str, cmd: List[str], out_dir: Path):
        entry = {"label": label, "command": " ".join(shlex.quote(c) for c in cmd),
                 "out_dir": str(out_dir)}
        self.plan.append(entry)
        return entry

    def run_cmd(self, label: str, cmd: List[str], out_dir: Path,
                status_holder: Dict[str, Any], inline: bool = False
                ) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        entry = self._record(label, cmd, out_dir)
        marker = self._completion_marker(cmd, out_dir)
        entry["completion_marker"] = str(marker) if marker else None

        if self.mode == "dry-run":
            entry["status"] = "PLANNED (dry-run)"
            return entry

        # --resume: skip cases whose completion artifact already exists.
        if self.resume and marker is not None and marker.exists():
            entry["status"] = "SKIPPED (resume: output exists)"
            status_holder.setdefault("commands", []).append(entry)
            return entry

        # Parallel mode: queue for the pool unless forced inline (e.g. report,
        # which must run after all data is produced).
        if self.jobs > 1 and not inline:
            entry["status"] = "QUEUED"
            self.queue.append((entry, cmd, out_dir, label, status_holder))
            return entry

        self._exec_one(entry, cmd, out_dir, label)
        status_holder.setdefault("commands", []).append(entry)
        return entry

    def _worker_env(self) -> Dict[str, str]:
        """
        Environment for a child run.

        With ``--jobs N`` every child would otherwise let NumPy/MKL/OpenBLAS/torch
        grab *all* cores, so N children oversubscribe the machine and thrash
        (often slower than serial).  Divide the cores between workers instead.
        """
        env = dict(os.environ)
        # Unbuffered child stdout: otherwise Python block-buffers (~8 KB) when
        # output is redirected to a file, so a running job's log stays EMPTY for
        # a long time and the campaign looks hung when it is fine.
        env["PYTHONUNBUFFERED"] = "1"
        if self.jobs > 1:
            per = max(1, (os.cpu_count() or 1) // self.jobs)
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS"):
                env[var] = str(per)
            env.setdefault("TORCH_NUM_THREADS", str(per))
        return env

    def _exec_one(self, entry: Dict[str, Any], cmd: List[str], out_dir: Path,
                  label: str, stream: bool = False) -> Dict[str, Any]:
        """
        Run one command to completion, recording timing/return code.

        ``stream=True`` tees the child's output to this console as it arrives
        (used for calibration, so a long-running job never looks frozen).
        """
        log = out_dir / f"{label}.log"
        # Stagger launches: torch+jax import is the peak-memory moment, so
        # starting every worker simultaneously creates a spike that kills
        # children before any physics runs.  Serialise the start instants.
        if self.jobs > 1 and self.start_stagger_s > 0 and not stream:
            with self._start_lock:
                time.sleep(self.start_stagger_s)
        entry["utc_start"] = utc_iso(); t0 = time.time()
        # Announce the launch: long cases would otherwise produce no console
        # output at all until they finish, hours later.
        with self._start_lock:
            self._running[label] = t0
            print(f"  [start] {label}", flush=True)
        env = self._worker_env()
        if stream:
            with open(log, "w", encoding="utf-8") as fh:
                proc = subprocess.Popen(
                    cmd, cwd=str(self.repo), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                    encoding="utf-8", errors="replace", env=env)
                assert proc.stdout is not None
                for line in proc.stdout:
                    sys.stdout.write(line); sys.stdout.flush()
                    fh.write(line)
                proc.wait()
            returncode = proc.returncode
        else:
            with open(log, "w", encoding="utf-8") as fh:
                proc = subprocess.run(cmd, cwd=str(self.repo), stdout=fh,
                                      stderr=subprocess.STDOUT, text=True, env=env)
            returncode = proc.returncode
        entry["returncode"] = returncode
        entry["seconds"] = round(time.time() - t0, 2)
        entry["utc_end"] = utc_iso()
        entry["log"] = str(log)
        entry["status"] = "OK" if returncode == 0 else "FAILED"
        with self._start_lock:
            self._running.pop(label, None)
        return entry

    def dispatch(self) -> None:
        """Execute all queued jobs concurrently with up to ``self.jobs`` workers.

        The 38 dynamics/validation cases are independent processes, so this is a
        near-linear speedup.  Results are collected in the main thread (each
        worker only mutates its own entry dict), so no locking is required.
        """
        if not self.queue:
            return
        from concurrent.futures import ThreadPoolExecutor, as_completed
        n = len(self.queue)
        print(f"[dispatch] running {n} jobs with {self.jobs} worker(s)")
        # Memory guard: each child imports torch AND jax (~2-3 GB peak at
        # N ~ 1000-1400).  Oversubscribing RAM does not degrade gracefully --
        # children die with MemoryError, frequently inside `import jax`.
        rec = recommended_jobs(per_job_gb=self.per_job_gb)
        est_gb = self.per_job_gb * self.jobs
        print(f"[dispatch] estimated peak memory ~{est_gb:.0f} GB "
              f"({self.jobs} x ~{self.per_job_gb:.1f} GB)")
        if rec is not None and self.jobs > rec:
            print(f"[dispatch] WARNING: this machine looks safe for about "
                  f"--jobs {rec}. Jobs may die with MemoryError.")
        elif rec is None:
            print("[dispatch] (install psutil for an automatic RAM check; "
                  "rule of thumb: jobs <= (RAM_GB - 4) / 2.5)")
        # Heartbeat: individual cases can run for hours, so without this the
        # console shows nothing between completions and looks hung.
        def _heartbeat():
            t_start = time.time()
            while not self._stop_heartbeat.wait(self.heartbeat_s):
                with self._start_lock:
                    inflight = sorted(self._running.items(), key=lambda kv: kv[1])
                mins = (time.time() - t_start) / 60.0
                if inflight:
                    detail = ", ".join(
                        f"{lab} ({(time.time()-t0)/60.0:.0f}m)"
                        for lab, t0 in inflight[:4])
                    print(f"  [alive {mins:.0f}m] running {len(inflight)}: "
                          f"{detail}", flush=True)
                else:
                    print(f"  [alive {mins:.0f}m] no jobs in flight", flush=True)

        hb = threading.Thread(target=_heartbeat, daemon=True)
        self._stop_heartbeat.clear()
        if self.heartbeat_s > 0:
            hb.start()

        failures: List[tuple] = []
        with ThreadPoolExecutor(max_workers=self.jobs) as ex:
            futs = {ex.submit(self._exec_one, e, c, o, l): (e, c, o, l, sh)
                    for (e, c, o, l, sh) in self.queue}
            done = 0
            for fut in as_completed(futs):
                entry, cmd, out_dir, label, sh = futs[fut]
                try:
                    fut.result()
                except Exception as exc:                     # pragma: no cover
                    entry["status"] = "FAILED"
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                sh.setdefault("commands", []).append(entry)
                done += 1
                print(f"  [{done}/{n}] {entry['status']}: {entry['label']} "
                      f"({entry.get('seconds', '?')}s)")
                if entry["status"] == "FAILED":
                    failures.append((entry, cmd, out_dir, label))
        self._stop_heartbeat.set()
        self.queue = []

        # ---- automatic recovery -------------------------------------------
        # Cases killed by memory pressure are not real failures: the same
        # command usually succeeds once it is not competing for RAM.  Retry
        # them one at a time (full memory + all cores each) so the campaign
        # self-heals instead of leaving a wall of FAILED rows.
        oom = [f for f in failures if log_indicates_oom(f[0].get("log"))]
        other = len(failures) - len(oom)
        if oom:
            print(f"\n[recovery] {len(oom)} case(s) died out-of-memory; "
                  f"retrying serially with full RAM per case...")
            saved_jobs = self.jobs
            self.jobs = 1                    # full threads, no stagger, no clamp
            try:
                for i, (entry, cmd, out_dir, label) in enumerate(oom, 1):
                    marker = self._completion_marker(cmd, out_dir)
                    if marker is not None and marker.exists():
                        entry["status"] = "OK (completed before retry)"
                        continue
                    entry["retried_after_oom"] = True
                    self._exec_one(entry, cmd, out_dir, label)
                    print(f"  [retry {i}/{len(oom)}] {entry['status']}: {label} "
                          f"({entry.get('seconds', '?')}s)")
            finally:
                self.jobs = saved_jobs
        if other:
            print(f"[recovery] {other} case(s) failed for non-memory reasons; "
                  f"see their logs (not retried).")

    def save(self):
        self.manifest["plan"] = self.plan
        self.manifest["environment"]["utc_end"] = utc_iso()
        (self.root / "campaign_manifest.json").write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8")
        print(f"[campaign] manifest -> {self.root / 'campaign_manifest.json'}")

    # -- Step 3: production contract ----------------------------------------
    def step3_contract(self, production_dirs: Dict[float, Optional[Path]],
                       masses=2000.0):
        st: Dict[str, Any] = {"step": 3, "name": "production_contract"}
        contract: Dict[str, Any] = {}
        keys = ["R0", "P0", "sigma_R", "mass", "hbar", "init_state",
                "sampling_mode", "density_mode", "surrogate", "label_scheme",
                "weight_scheme", "dt", "t_final", "normalization",
                "profile_floor", "noise_floor", "l2_regularization",
                "refit_hyper_policy"]
        for P0, pdir in production_dirs.items():
            if pdir is None or not Path(pdir).exists():
                contract[str(P0)] = {"status": "NOT RUN",
                                     "reason": f"no production manifest for P0={P0}"}
                continue
            man = None
            for name in ("run_manifest.json", "pbme.json", "midpoint.json"):
                p = Path(pdir) / name
                if p.exists():
                    man = json.loads(p.read_text(encoding="utf-8")); break
            if man is None:
                contract[str(P0)] = {"status": "INSUFFICIENT EVIDENCE",
                                     "reason": f"no manifest json under {pdir}"}
                continue
            flat = _flatten(man)
            contract[str(P0)] = {"status": "READ",
                                 "values": {k: _find_key(flat, k) for k in keys},
                                 "source": str(p)}
        (self.root / "production_contract.json").write_text(
            json.dumps(contract, indent=2), encoding="utf-8")
        st["artifact"] = str(self.root / "production_contract.json")
        st["status"] = "COMPLETE" if any(
            v.get("status") == "READ" for v in contract.values()) else "NOT RUN"
        self.manifest["steps"]["step3"] = st
        return contract

    # -- Step 7: dt convergence (one control varied: dt) --------------------
    def step7_dt_convergence(self, *, P0: float, n_train: int, dt_base: float,
                             seeds: Sequence[int], R0: float, sigma_R: float,
                             mass: float, hbar: float, l2: float,
                             density_mode="full", sampling_mode="focused",
                             surrogate="product", abs_target=False,
                             refit_hyper_policy="breathing",
                             ladder: Optional[Sequence[float]] = None):
        """
        Refine the timestep with everything else held fixed.

        ``ladder`` overrides the default (dt_base, dt_base/2, dt_base/4).
        Anchoring the ladder AT the production timestep -- e.g. (1.0, 0.5, 0.25)
        when production runs at 0.25 -- demonstrates convergence *to* the
        setting actually used, and costs half as much as refining below it.
        """
        st: Dict[str, Any] = {"step": 7, "name": "dt_convergence",
                              "P0": P0, "levels": {}}
        t_c = collision_time(mass, R0, P0)
        t_final = 2.0 * t_c
        base = self.root / f"step7_dt_P0{P0:g}"
        dts = (tuple(float(x) for x in ladder) if ladder
               else (dt_base, dt_base / 2, dt_base / 4))
        st["dt_ladder"] = list(dts)
        for seed in seeds:
            for dt in dts:
                out = base / f"seed{seed}_dt{dt:g}"
                cmd = run_py_cmd(self.python, self.run_py, out, P0=P0,
                                 n_train=n_train, dt=dt, t_final=t_final, seed=seed,
                                 snapshot_every=snapshot_every_for(t_c, dt),
                                 density_mode=density_mode, sampling_mode=sampling_mode,
                                 surrogate=surrogate, l2_regularization=l2, R0=R0,
                                 sigma_R=sigma_R, mass=mass, hbar=hbar,
                                 abs_target=abs_target,
                                 refit_hyper_policy=refit_hyper_policy)
                self.run_cmd(f"dt_P0{P0:g}_seed{seed}_h{dt:g}", cmd, out, st)
        st["t_c"] = t_c; st["t_final"] = t_final
        st["analysis"] = "pending execution" if self.mode == "dry-run" else \
            "run analyze_dt_convergence() after execute"
        st.setdefault("status", "PLANNED" if self.mode == "dry-run" else "RAN")
        self.manifest["steps"]["step7"] = st

    # -- Step 8: nested support convergence (one control varied: N) ---------
    def step8_support_convergence(self, *, P0: float, dt: float,
                                  levels=(350, 700, 1400), seeds=(11, 29, 47),
                                  R0=-15.0, sigma_R=1.0, mass=2000.0, hbar=1.0,
                                  l2=0.0, density_mode="full",
                                  sampling_mode="focused", surrogate="product",
                                  abs_target=False):
        st: Dict[str, Any] = {"step": 8, "name": "support_convergence",
                              "P0": P0, "note":
            "NESTED subsets require run.py to expose a fixed-cloud/prefix policy; "
            "if unavailable the driver records this and does NOT claim a "
            "deterministic support order (Rule 4/8)."}
        t_c = collision_time(mass, R0, P0); t_final = 2 * t_c
        base = self.root / f"step8_support_P0{P0:g}"
        for seed in seeds:
            for N in levels:
                out = base / f"seed{seed}_N{N}"
                cmd = run_py_cmd(self.python, self.run_py, out, P0=P0, n_train=N,
                                 dt=dt, t_final=t_final, seed=seed,
                                 snapshot_every=snapshot_every_for(t_c, dt),
                                 density_mode=density_mode, sampling_mode=sampling_mode,
                                 surrogate=surrogate, l2_regularization=l2, R0=R0,
                                 sigma_R=sigma_R, mass=mass, hbar=hbar,
                                 abs_target=abs_target, refit_hyper_policy="breathing")
                self.run_cmd(f"support_P0{P0:g}_seed{seed}_N{N}", cmd, out, st)
        st["levels"] = list(levels); st["seeds"] = list(seeds)
        st.setdefault("status", "PLANNED" if self.mode == "dry-run" else "RAN")
        self.manifest["steps"]["step8"] = st

    # -- Step 9: independent replication (one control varied: seed) ---------
    def step9_replication(self, *, P0_list=(20.0, 100.0), n_train=1000, dt=0.25,
                          seeds=(11, 29, 47, 73), R0=-15.0, sigma_R=1.0,
                          mass=2000.0, hbar=1.0, l2=0.0, density_mode="full",
                          sampling_mode="focused", surrogate="product",
                          abs_target=False):
        st: Dict[str, Any] = {"step": 9, "name": "replication",
                              "P0_list": list(P0_list), "seeds": list(seeds)}
        for P0 in P0_list:
            t_c = collision_time(mass, R0, P0); t_final = 2 * t_c
            base = self.root / f"step9_repl_P0{P0:g}"
            for seed in seeds:
                out = base / f"seed{seed}"
                cmd = run_py_cmd(self.python, self.run_py, out, P0=P0,
                                 n_train=n_train, dt=dt, t_final=t_final, seed=seed,
                                 snapshot_every=snapshot_every_for(t_c, dt),
                                 density_mode=density_mode, sampling_mode=sampling_mode,
                                 surrogate=surrogate, l2_regularization=l2, R0=R0,
                                 sigma_R=sigma_R, mass=mass, hbar=hbar,
                                 abs_target=abs_target, refit_hyper_policy="breathing")
                self.run_cmd(f"repl_P0{P0:g}_seed{seed}", cmd, out, st)
        st.setdefault("status", "PLANNED" if self.mode == "dry-run" else "RAN")
        self.manifest["steps"]["step9"] = st

    # -- Steps 5, 6, 11, 12: delegate to ReviewerValidation subcommands -----
    def step5_manufactured(self, n_trains=(300, 600, 1200, 2400),
                           n_query=1000, seeds=(123, 124, 125)):
        st: Dict[str, Any] = {"step": 5, "name": "manufactured_operator",
                              "n_trains": list(n_trains), "seeds": list(seeds)}
        for N in n_trains:
            for seed in seeds:
                out = self.root / "step5_manufactured" / f"N{N}_seed{seed}"
                cmd = rv_cmd(self.python, self.rv_py, "manufactured",
                             out=out, n_train=N, n_query=n_query, seed=seed)
                self.run_cmd(f"manufactured_N{N}_seed{seed}", cmd, out, st)
        st.setdefault("status", "PLANNED" if self.mode == "dry-run" else "RAN")
        self.manifest["steps"]["step5"] = st

    def step6_projection(self, pbme_stems: Sequence[Path]):
        st: Dict[str, Any] = {"step": 6, "name": "projection_leakage",
                              "route": "A (minimal defensible)"}
        for stem, tag in zip(pbme_stems, unique_stem_tags(pbme_stems)):
            out = self.root / "step6_projection" / tag
            # 'projection' takes the stem as a positional argument.
            cmd = [self.python, str(self.rv_py), "projection", str(stem),
                   "--out", str(out)]
            self.run_cmd(f"projection_{tag}", cmd, out, st)
        st.setdefault("status", "PLANNED" if self.mode == "dry-run" else "RAN")
        self.manifest["steps"]["step6"] = st

    def step11_baseline(self, pbme_stems: Sequence[Path]):
        st: Dict[str, Any] = {"step": 11, "name": "pbme_kde_gp_baseline",
                              "acceptance": "E1 <= 0.02 (PBME source only)"}
        for stem, tag in zip(pbme_stems, unique_stem_tags(pbme_stems)):
            out = self.root / "step11_baseline" / tag
            cmd = [self.python, str(self.rv_py), "baseline", str(stem),
                   "--out", str(out)]
            self.run_cmd(f"baseline_{tag}", cmd, out, st)
        st.setdefault("status", "PLANNED" if self.mode == "dry-run" else "RAN")
        self.manifest["steps"]["step11"] = st

    def step12_reference(self, dt=0.2, n_steps=200, P0_list=(20.0, 100.0)):
        st: Dict[str, Any] = {"step": 12, "name": "tdse_qcle_reference",
                              "P0_list": list(P0_list)}
        for P0 in P0_list:
            out = self.root / "step12_reference" / f"P0{P0:g}"
            cmd = rv_cmd(self.python, self.rv_py, "reference",
                         out=out, dt=dt, n_steps=n_steps, P0=P0)
            self.run_cmd(f"reference_P0{P0:g}", cmd, out, st)
        st.setdefault("status", "PLANNED" if self.mode == "dry-run" else "RAN")
        self.manifest["steps"]["step12"] = st

    def step_report(self):
        """Step 16/deliverable #6: consolidate everything into the master table."""
        out = self.root
        cmd = [self.python, str(self.rv_py), "report", "--out", str(out)]
        st: Dict[str, Any] = {"step": 16, "name": "master_table"}
        self.run_cmd("master_report", cmd, out, st, inline=True)   # after dispatch
        self.manifest["steps"]["step16"] = st


# ---- manifest flattening helpers (Step 3) ---------------------------------

def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        flat[prefix] = obj
    return flat


def _find_key(flat: Dict[str, Any], key: str) -> Any:
    """Return the first flattened entry whose leaf name matches ``key``."""
    for full, val in flat.items():
        if full.split(".")[-1] == key:
            return val
    return None


# ===========================================================================
# Runtime estimation
# ===========================================================================

def total_campaign_steps(*, R0=-15.0, mass=2000.0,
                         dt_base=0.5, dt_support=0.25, dt_repl=0.25,
                         P0_dt=(20.0, 100.0), seeds_dt=(11, 29),
                         P0_support=(20.0, 100.0), seeds_support=(11, 29, 47),
                         support_levels=(350, 700, 1400),
                         P0_repl=(20.0, 100.0), seeds_repl=(11, 29, 47, 73),
                         n_train_dt=1000, n_train_repl=1000, n_calib=500
                         ) -> Dict[str, float]:
    r"""
    Total paired integration steps the campaign will run, plus an N-weighted
    total that accounts for the O(N^2)-ish growth of the per-step operator
    evaluation (so a projected wall time reflects that N=2000 cases cost more
    than N=500 cases).  Weight = (N / n_calib)^2.
    """
    def steps(P0, dt):
        t_c = collision_time(mass, R0, P0)
        return int(round(2.0 * t_c / dt))

    raw = 0.0
    weighted = 0.0
    n_cases = 0
    w = lambda N: (float(N) / float(n_calib)) ** 2

    for P0 in P0_dt:
        for _ in seeds_dt:
            for dt in (dt_base, dt_base / 2, dt_base / 4):
                s = steps(P0, dt); raw += s; weighted += s * w(n_train_dt); n_cases += 1
    for P0 in P0_support:
        for _ in seeds_support:
            for N in support_levels:
                s = steps(P0, dt_support); raw += s; weighted += s * w(N); n_cases += 1
    for P0 in P0_repl:
        for _ in seeds_repl:
            s = steps(P0, dt_repl); raw += s; weighted += s * w(n_train_repl); n_cases += 1
    # Critical path: the single most expensive case.  A trajectory cannot be
    # split across workers, so the campaign can NEVER finish faster than this,
    # no matter how many jobs are used.
    longest = 0.0
    for P0 in P0_dt:
        for dt in (dt_base, dt_base / 2, dt_base / 4):
            longest = max(longest, steps(P0, dt) * w(n_train_dt))
    for P0 in P0_support:
        for N in support_levels:
            longest = max(longest, steps(P0, dt_support) * w(N))
    for P0 in P0_repl:
        longest = max(longest, steps(P0, dt_repl) * w(n_train_repl))

    return {"raw_steps": raw, "weighted_steps": weighted,
            "n_calib": n_calib, "n_cases": n_cases,
            "longest_case_weighted_steps": longest}


def project_runtime(per_step_seconds: float, overhead_seconds: float,
                    totals: Dict[str, float], jobs: int = 1) -> Dict[str, Any]:
    """
    Project campaign wall time from a per-step cost and a per-case fixed
    overhead (process start, initial GP fit, JAX compilation).

        serial = n_cases * overhead + per_step * weighted_steps
    """
    n_cases = float(totals.get("n_cases", 0))
    fixed = n_cases * max(overhead_seconds, 0.0)
    serial_weighted = fixed + per_step_seconds * totals["weighted_steps"]
    serial_flat = fixed + per_step_seconds * totals["raw_steps"]

    # A single case is inherently sequential -> hard lower bound on wall time.
    critical_h = (max(overhead_seconds, 0.0)
                  + per_step_seconds
                  * totals.get("longest_case_weighted_steps", 0.0)) / 3600.0
    ideal_h = serial_weighted / 3600.0 / max(jobs, 1)
    makespan_h = max(ideal_h, critical_h)

    return {
        "seconds_per_paired_step_at_calib_N": per_step_seconds,
        "per_case_overhead_seconds": max(overhead_seconds, 0.0),
        "n_cases": n_cases,
        "serial_hours_flat": serial_flat / 3600.0,
        "serial_hours_N_weighted": serial_weighted / 3600.0,
        "parallel_hours_N_weighted": ideal_h,
        "critical_path_hours": critical_h,
        "expected_wall_hours": makespan_h,
        "parallelism_is_critical_path_bound": bool(critical_h > ideal_h),
        "jobs": jobs,
        "note": ("N-weighted uses (N/n_calib)^2 for the per-step operator cost; "
                 "periodic O(N^3) refits make the largest-N cases the upper end. "
                 "expected_wall_hours = max(serial/jobs, longest single case): "
                 "no amount of parallelism beats the critical path."),
    }


def run_estimate(repo: Path, python: str, jobs: int,
                 calib_steps: Tuple[int, int] = (30, 90),
                 n_train: int = 500, P0: float = 100.0, dt: float = 0.5,
                 ) -> Dict[str, Any]:
    """
    Fast, visible runtime calibration.

    Runs TWO very short trajectories (default 30 and 90 paired steps) and takes
    the *difference* so that fixed startup cost (process launch, initial GP fit,
    JAX tracing/compilation) cancels:

        per_step = (t_long - t_short) / (n_long - n_short)
        overhead =  t_short - per_step * n_short

    Both runs stream their output to the console, so progress is visible.  This
    replaces the previous version, which ran a full 1200-step trajectory with
    ``--quiet`` and therefore looked frozen for a very long time.
    """
    root = repo / f"reviewer_estimate_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    root.mkdir(parents=True, exist_ok=True)
    camp = Campaign(repo=repo, root=root, python=python, mode="execute", jobs=1)
    t_c = collision_time(2000.0, -15.0, P0)

    timings: List[Tuple[int, float]] = []
    n_short, n_long = int(calib_steps[0]), int(calib_steps[1])
    for n_steps in (n_short, n_long):
        t_final = n_steps * dt          # deliberately tiny: a probe, not a run
        out = root / f"calib_{n_steps}steps"
        cmd = run_py_cmd(python, repo / "run.py", out, P0=P0, n_train=n_train,
                         dt=dt, t_final=t_final, seed=11,
                         snapshot_every=max(1, n_steps),
                         density_mode="full", sampling_mode="focused",
                         surrogate="product", l2_regularization=0.0, R0=-15.0,
                         sigma_R=1.0, mass=2000.0, hbar=1.0, abs_target=False,
                         refit_hyper_policy="breathing", quiet=False)
        st: Dict[str, Any] = {"step": "estimate", "name": f"calib_{n_steps}"}
        entry = camp._record(f"calib_{n_steps}", cmd, out)
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n[estimate] timing {n_steps} paired steps (N={n_train}, "
              f"P0={P0:g}, dt={dt:g}) — live output follows\n")
        camp._exec_one(entry, cmd, out, f"calib_{n_steps}", stream=True)
        if entry.get("status") != "OK":
            print(f"[estimate] calibration FAILED; see {entry.get('log')}")
            return {"status": "FAILED", "entry": entry}
        timings.append((n_steps, float(entry["seconds"])))
        print(f"[estimate] {n_steps} steps -> {entry['seconds']:.1f} s")

    (n1, t1), (n2, t2) = timings
    per_step = (t2 - t1) / max(n2 - n1, 1)
    if per_step <= 0:                       # noise dominated; fall back
        per_step = t2 / max(n2, 1)
        overhead = 0.0
    else:
        overhead = max(t1 - per_step * n1, 0.0)

    totals = total_campaign_steps(n_calib=n_train)
    proj = project_runtime(per_step, overhead, totals, jobs=jobs)
    proj.update({"calibration_timings": [{"paired_steps": n, "seconds": s}
                                         for n, s in timings],
                 "totals": totals})
    (root / "runtime_estimate.json").write_text(json.dumps(proj, indent=2),
                                                encoding="utf-8")
    print("\n" + "=" * 66)
    print(f"  per paired step (N={n_train}): {per_step*1000:.0f} ms")
    print(f"  per-case fixed overhead:      {overhead:.0f} s")
    print(f"  cases in campaign:            {int(totals['n_cases'])}")
    print(f"  PROJECTED serial:             {proj['serial_hours_N_weighted']:.1f} h")
    print(f"  ideal with --jobs {jobs}:          "
          f"{proj['parallel_hours_N_weighted']:.1f} h")
    print(f"  critical path (longest case): {proj['critical_path_hours']:.1f} h")
    print(f"  EXPECTED WALL TIME:           {proj['expected_wall_hours']:.1f} h")
    if proj["parallelism_is_critical_path_bound"]:
        print("  ^ critical-path bound: more workers will NOT help.")
        print("    Lower --support-levels to shorten the longest case.")
    print("=" * 66)
    print(f"[estimate] wrote {root / 'runtime_estimate.json'}")
    return proj


# ===========================================================================
# Self-test  (pure helpers; no torch)
# ===========================================================================

def run_self_test() -> None:
    # collision time
    assert abs(collision_time(2000.0, -15.0, 20.0) - 1500.0) < 1e-9
    # nested subsets
    sub = nested_subset_indices(2000, (500, 1000, 2000), seed=11)
    assert set(sub[500]) <= set(sub[1000]) <= set(sub[2000])
    assert sub[2000].size == 2000 and sub[500].size == 500
    # observed order: exact 2nd-order sequence err ~ (dt)^2 halving -> ratio 4 -> p=2
    u_h = np.array([4.0]); u_h2 = np.array([1.0]); u_h4 = np.array([0.25])
    p, why = observed_order(u_h, u_h2, u_h4)
    assert why == "ok" and abs(p - 2.0) < 1e-9, (p, why)
    # guard: zero finer difference
    p2, why2 = observed_order(np.array([1.0]), np.array([0.0]), np.array([0.0]))
    assert p2 is None and "INSUFFICIENT" in why2
    # guard: below seed noise
    p3, why3 = observed_order(np.array([1e-6]), np.array([5e-7]), np.array([2e-7]),
                              seed_noise=1e-3)
    assert p3 is None and "seed noise" in why3
    # shell distance / bins
    S = np.zeros((1, 2)); X = np.array([[0.3, 0.0], [0.9, 0.0], [3.0, 0.0]])
    d = shell_distance(X, S, np.ones(2))
    bins = shell_bin_indices(d)
    assert bins["[0,0.5)"].tolist() == [0]
    assert bins["[0.5,1)"].tolist() == [1]
    assert bins["[2,4)"].tolist() == [2]
    # interp + timeseries norms
    t = np.linspace(0, 1, 11); a = t; b = t + 0.1
    an = interp_to_grid(t, a, t)
    n = timeseries_norms(an, b, t)
    assert abs(n["Linf"] - 0.1) < 1e-9
    # raw drift summary
    tt = np.linspace(0, 3000, 301); dd = 1e-9 * tt
    s = raw_drift_summary(tt, dd, t_c=1500.0)
    assert abs(s["endpoint"] - dd[-1]) < 1e-18
    assert s["pre_interaction_max_abs"] is not None
    # E1 gate
    R = np.linspace(-1, 1, 5); P = np.linspace(-1, 1, 5)
    g = np.ones((5, 5)); k = np.ones((5, 5))
    e = grid_shape_errors(g, k, R, P)
    assert e["E1"] == 0.0
    # convergence slope
    slope = convergence_slopes([300, 600, 1200], [1e-1, 5e-2, 2.5e-2])
    assert slope is not None and slope < 0
    # command builders
    cmd = run_py_cmd("python", Path("run.py"), Path("out"), P0=20.0, n_train=1000,
                     dt=0.25, t_final=3000.0, seed=11, snapshot_every=6000,
                     density_mode="full", sampling_mode="focused",
                     surrogate="product", l2_regularization=0.0, R0=-15.0,
                     sigma_R=1.0, mass=2000.0, hbar=1.0, abs_target=False,
                     refit_hyper_policy="breathing")
    assert "--no_auto_dt" in cmd and "--no_abs_target" in cmd and "--P0" in cmd
    rv = rv_cmd("python", Path("ReviewerValidation.py"), "manufactured",
                out=Path("o"), n_train=600, n_query=1000, seed=123)
    assert rv[:3] == ["python", "ReviewerValidation.py", "manufactured"]
    assert "--n-train" in rv and "600" in rv
    # runtime estimation helpers
    totals = total_campaign_steps()
    assert totals["raw_steps"] > 0 and totals["n_cases"] == 38
    assert totals["weighted_steps"] >= totals["raw_steps"]   # weights >= 1
    proj = project_runtime(0.05, 10.0, totals, jobs=4)
    assert abs(proj["parallel_hours_N_weighted"]
               - proj["serial_hours_N_weighted"] / 4) < 1e-9
    # the fixed overhead must contribute exactly n_cases * overhead
    no_oh = project_runtime(0.05, 0.0, totals, jobs=1)
    delta = proj["serial_hours_N_weighted"] - no_oh["serial_hours_N_weighted"]
    assert abs(delta - 38 * 10.0 / 3600.0) < 1e-9
    print("[self-test] all pure-helper checks passed.")


# ===========================================================================
# CLI
# ===========================================================================

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["dry-run", "execute", "self-test", "estimate"],
                   default="dry-run")
    p.add_argument("--jobs", type=int, default=1,
                   help="Parallel workers in execute mode (the cases are independent).")
    p.add_argument("--resume", action="store_true",
                   help="Skip cases whose completion artifact already exists.")
    p.add_argument("--allow-oversubscribe", action="store_true",
                   help="Bypass the RAM-based --jobs clamp (risks MemoryError).")
    p.add_argument("--start-stagger", type=float, default=5.0,
                   help="Seconds between child launches; smooths the torch+jax "
                        "import memory spike. 0 disables.")
    p.add_argument("--heartbeat", type=float, default=60.0,
                   help="Seconds between progress pings while jobs run. "
                        "0 disables.")
    p.add_argument("--per-job-gb", type=float, default=4.0,
                   help="Assumed peak RAM per worker for the --jobs clamp. "
                        "Measured ~4 GB; raise if you still hit MemoryError.")
    p.add_argument("--manufactured-levels", type=str, default="300,600,1200,2400",
                   help="Step 5 support levels. N=2400 is the single heaviest "
                        "job in the campaign; drop it on low-RAM machines "
                        "(the spec allows 'subject to available memory').")
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--root", type=Path, default=None,
                   help="Campaign root; default reviewer_closure_<UTC> under --repo.")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--steps", type=str, default="3,5,6,7,8,9,11,12,16",
                   help="Comma-separated step numbers to include.")
    p.add_argument("--production-dir-P0-20", type=Path, default=None)
    p.add_argument("--production-dir-P0-40", type=Path, default=None)
    p.add_argument("--production-dir-P0-100", type=Path, default=None)
    p.add_argument("--pbme-stem", type=Path, action="append", default=[],
                   help="PBME run stem (dir/pbme) for projection & baseline; repeatable.")
    p.add_argument("--l2", type=float, default=0.0,
                   help="l2_regularization (use L2* from select_regularization).")
    p.add_argument("--support-levels", type=str, default="350,700,1400",
                   help="Three-level nested support ladder for Step 8. The top "
                        "level sets the campaign's critical path: a single "
                        "P0=20 case costs ~(N/500)^2 x 12000 x per_step "
                        "seconds, and no amount of parallelism shortens it.")
    p.add_argument("--dt-ladder", type=str, default="0.5,0.25,0.125",
                   help="Step 7 timestep ladder (coarse,mid,fine).")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    if args.mode == "self-test":
        run_self_test(); return
    if args.mode == "estimate":
        run_estimate(args.repo, args.python, jobs=args.jobs); return

    root = args.root or (args.repo /
                         f"reviewer_closure_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}")
    root.mkdir(parents=True, exist_ok=True)
    camp = Campaign(repo=args.repo, root=root, python=args.python, mode=args.mode,
                    jobs=args.jobs, resume=args.resume,
                    allow_oversubscribe=args.allow_oversubscribe,
                    start_stagger_s=args.start_stagger,
                    heartbeat_s=args.heartbeat,
                    per_job_gb=args.per_job_gb)
    steps = {s.strip() for s in args.steps.split(",") if s.strip()}

    if "3" in steps:
        camp.step3_contract({20.0: args.production_dir_P0_20,
                             40.0: args.production_dir_P0_40,
                             100.0: args.production_dir_P0_100})
    if "5" in steps:
        camp.step5_manufactured(
            n_trains=tuple(int(x) for x in args.manufactured_levels.split(",")
                           if x.strip()))
    if "6" in steps:
        camp.step6_projection(args.pbme_stem)
    if "7" in steps:
        ladder = tuple(float(x) for x in args.dt_ladder.split(",") if x.strip())
        for _P0 in (20.0, 100.0):
            camp.step7_dt_convergence(P0=_P0, n_train=1000, dt_base=ladder[0],
                                      seeds=(11, 29), R0=-15.0, sigma_R=1.0,
                                      mass=2000.0, hbar=1.0, l2=args.l2,
                                      ladder=ladder)
    if "8" in steps:
        levels = tuple(int(x) for x in args.support_levels.split(",") if x.strip())
        camp.step8_support_convergence(P0=20.0, dt=0.25, levels=levels, l2=args.l2)
        camp.step8_support_convergence(P0=100.0, dt=0.25, levels=levels, l2=args.l2)
    if "9" in steps:
        camp.step9_replication(l2=args.l2)
    if "11" in steps:
        camp.step11_baseline(args.pbme_stem)
    if "12" in steps:
        camp.step12_reference()

    # In parallel execute mode the dynamics/validation jobs were queued; run them
    # now, then produce the master table LAST (it aggregates their outputs).
    camp.dispatch()
    if "16" in steps:
        camp.step_report()

    camp.save()
    if args.mode == "dry-run":
        print(f"\n[dry-run] {len(camp.plan)} commands planned under {root}")
        for e in camp.plan:
            print("  " + e["command"])
        print("\nRe-run with --mode execute (add --jobs N --resume) on a machine "
              "with PyTorch to run them.")


if __name__ == "__main__":
    main()
