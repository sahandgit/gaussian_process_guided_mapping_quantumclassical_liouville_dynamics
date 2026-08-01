from __future__ import annotations

"""
run.py
======

Single-entry driver for the refactored FLV pipeline.

Steps
-----
1. MC-sample an initial signed-SEO cloud.
2. Fit an initial GP surrogate with KKT-constrained moments
   (normalization, trace, energy).
3. Deep-copy the resulting SimulationState so PBME and midpoint schemes
   start from identical (Z, y, GP, moment_targets).
4. Run both schemes for the same number of steps at the same dt.
5. Save both runs via Collector (.npz + .json sidecar).
6. Generate the full comparison-figure set via Visualization.py.

Output directory layout
-----------------------
    <out>/pbme.npz       <out>/pbme.json
    <out>/midpoint.npz   <out>/midpoint.json
    <out>/fig_conservation.png
    <out>/fig_populations.png
    <out>/fig_coherences.png
    <out>/fig_nuclear.png
    <out>/fig_mapping_moments.png
    <out>/fig_local_energy.png
    <out>/fig_correction.png
    <out>/fig_fit_quality.png
    <out>/fig_marginal_1d_{ax}_step<K>.png      (one per axis, per snapshot step)
    <out>/fig_marginal_2d_{ax0}_{ax1}_step<K>.png  (one per pair, per snapshot step)

    BUG FIX (doc): the legacy names fig_slice_*.png listed here previously were
    wrong.  Visualization.py writes fig_marginal_{1d,2d}_* as shown above.
"""

import argparse
import copy
import os
import sys
import time

# ---------------------------------------------------------------------------
# UTF-8 console safety.
#
# The banners and diagnostics below print non-ASCII physics notation
# (e.g. alpha 'α', rho-hat 'ρ̂', 'Δt', '→', 'ħ').  On Windows the default
# stdout/stderr encoding is cp1252, which cannot encode these characters, so
# a plain print() raises UnicodeEncodeError and aborts the whole run (this is
# what broke the dt_N500_h0.5 validation case).  Reconfiguring the streams to
# UTF-8 here — before any output is produced — makes the encoding lossless and
# platform-independent.  Guarded so it is a no-op on interpreters/streams that
# do not support reconfigure().
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np

from GP_Density import GPDensityConfig
from Sampling import GaussianWavePacketParams, MappingInitParams
from Dynamics import DynamicsConfig, Simulation
from Visualization import produce_all_figures_publication
from Mint import PBMEMIntDynamics, PBMEMIntParams


def _positive_abs_p0(P0: float) -> float:
    """Return |P0| and reject the degenerate zero-momentum case."""
    pabs = abs(float(P0))
    if not np.isfinite(pabs) or pabs <= 0.0:
        raise ValueError("P0 must be non-zero when automatic scattering time/dt controls are used.")
    return pabs


def _electronic_dt_ceiling(args, steps_per_period: float) -> float:
    """
    P0-independent Δt ceiling that resolves the fastest electronic/mapping
    phase of the model under study.

    The SEO/MMST mapping variables rotate under the diabatic Hamiltonian, so
    the fastest phase rate encountered over the scattering window
    R ∈ [−|R0|, +|R0|] is bounded by the span of its eigenvalues there,
    ω_max = ΔE_span / ħ.  Requiring ``steps_per_period`` steps per oscillation
    period 2πħ/ΔE_span gives

        Δt_elec = 2πħ / (steps_per_period · ΔE_span).

    This is a physical resolution bound, NOT a momentum policy — it does not
    depend on P0.  For the default dual Tully model ΔE_span ≈ 0.10 a.u.
    (T_el ≈ 60 a.u.), so with 30 steps/period the ceiling is ≈ 2 a.u.: far
    looser than the dynamics dt, hence usually slack behind --dt_max.  It only
    bites if --dt_max is raised or a stiffer model/params are used.

    NOTE: the kind here ("dual") must match the TullyModel that
    PBMEMIntDynamics builds (it defaults to the dual model).  If a Tully-kind
    CLI flag is ever added, thread it through to this constructor.
    """
    from Models import TullyModel, TullyParams
    model = TullyModel(TullyParams.defaults("dual"))
    R = np.linspace(-abs(float(args.R0)), abs(float(args.R0)), 4001)
    E = model.adiabatic_energies(R)                 # (4001, 2), ascending
    dE_span = float(np.max(E[..., 1]) - np.min(E[..., 0]))
    if not np.isfinite(dE_span) or dE_span <= 0.0:
        return float("inf")
    return 2.0 * np.pi * float(args.hbar) / (float(steps_per_period) * dE_span)


def _resolve_time_grid(args) -> None:
    """
    Resolve the physical final time, timestep, and number of steps.

    The scattering timescale for the incoming packet is
        t_c = M |R0| / |P0|,
    and M/|P0| is the time required to move one bohr at the incoming speed.

    Final time
    ----------
    Every incoming momentum is made to traverse the seam: the endpoint is
    T = N_cyc · t_c (default N_cyc = 2, the incoming/crossing/outgoing window),
    or T = --t_final if given.  Because t_c ∝ 1/|P0|, slower packets get a
    correspondingly longer run.  --n_steps does NOT set the endpoint anymore;
    it is a derived output.

    Timestep
    --------
    With --auto_dt (on by default), dt is the tightest of:
      * the requested --dt (upper bound),
      * an accuracy-balanced momentum term dt ∝ |P0|^(--auto_dt_power), with
        default power 0.5 — the exponent that holds the 2nd-order global error
        (~ T·dt², T ∝ 1/|P0|) fixed across momenta, giving finer dt for slower
        packets without the 1/|P0|-style step blow-up, and
      * a P0-independent electronic-phase ceiling from the model's diabatic
        eigenvalue span (see _electronic_dt_ceiling),
    then clamped to [--dt_min, --dt_max].

    The final dt is adjusted downward so n_steps * dt hits the requested
    physical endpoint exactly (important when comparing runs across P0).
    """
    pabs = _positive_abs_p0(args.P0)
    mass = float(args.mass)
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass must be positive.")

    args.collision_time = mass * abs(float(args.R0)) / pabs
    args.time_per_bohr = mass / pabs

    # Start from the user-supplied dt as a hard upper bound.
    dt_nominal = float(args.dt)
    if not np.isfinite(dt_nominal) or dt_nominal <= 0.0:
        raise ValueError("dt must be positive.")

    args.dt_electronic_ceiling = float("inf")
    dt_eff = dt_nominal
    if args.auto_dt:
        pref = _positive_abs_p0(args.auto_dt_ref_p0)
        # Accuracy-balanced momentum scaling.  For the 2nd-order Strang/midpoint
        # scheme the global error over a run of length T ∝ 1/|P0| scales as
        # T·dt², so holding the error fixed gives dt ∝ |P0|^0.5 (the default
        # --auto_dt_power=0.5): finer dt for slower packets, but only as √|P0|,
        # NOT the 1/|P0|-style blow-up.  The nuclear seam-crossing and the
        # electronic phase both TOLERATE a larger dt at low |P0| (longer dwell),
        # so this accuracy term is what actually drives "finer for slow".
        scaled = float(args.auto_dt_ref) * (pabs / pref) ** float(args.auto_dt_power)
        # Hard physical ceiling: resolve the electronic/mapping phase (flat in P0).
        args.dt_electronic_ceiling = _electronic_dt_ceiling(args, args.steps_per_eperiod)
        # Never exceed the requested --dt; take the tightest of the policy term
        # and the physical ceiling; then clamp to the validated [dt_min, dt_max].
        dt_eff = min(dt_nominal, scaled, args.dt_electronic_ceiling)
        dt_eff = max(float(args.dt_min), min(float(args.dt_max), dt_eff))

    if not np.isfinite(dt_eff) or dt_eff <= 0.0:
        raise ValueError("resolved dt is not positive; check --auto_dt_ref, --dt_min, --dt_max.")

    # Final-time resolution.  Goal: EVERY incoming momentum traverses the seam.
    # T = N_cyc · t_c = N_cyc · M|R0|/|P0| carries the packet R0 → 0 → +|R0| for
    # any P0 (since t_c ∝ 1/|P0|).  --t_final overrides for manual control;
    # otherwise the number of scattering cycles is used, defaulting to 2 (the
    # standard incoming/crossing/outgoing window) when --scattering_cycles is
    # not given.  NOTE: --n_steps no longer sets the endpoint — it is now a
    # derived OUTPUT (n_eff below); use --t_final to fix the run length manually.
    if args.t_final is not None:
        T_target = float(args.t_final)
    else:
        cyc = 2.0 if args.scattering_cycles is None else float(args.scattering_cycles)
        T_target = cyc * args.collision_time

    if not np.isfinite(T_target) or T_target <= 0.0:
        raise ValueError("target final time must be positive.")

    n_eff = int(np.ceil(T_target / dt_eff - 1.0e-12))
    n_eff = max(1, n_eff)
    # Hit the endpoint exactly; this prevents comparing P0 runs that stop at
    # slightly different physical times because of ceiling/roundoff.
    dt_exact = T_target / float(n_eff)

    args.dt_requested = dt_nominal
    args.dt_auto_nominal = dt_eff
    args.t_final_resolved = T_target
    args.dt = float(dt_exact)
    args.n_steps = int(n_eff)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_train",        type=int,   default=1000)
    p.add_argument("--n_steps",        type=int,   default=12000,
                   help="IGNORED for the endpoint and overwritten with the resolved "
                        "ceil(T/dt). The run length T is set by --t_final or "
                        "--scattering_cycles (default 2 cycles). Use --t_final to fix "
                        "the number of steps manually.")
    p.add_argument("--dt",             type=float, default=0.5,
                   help="Maximum timestep dt [a.u.]. With --auto_dt this is an upper bound; the resolved dt may be smaller.")
    p.add_argument("--snapshot_every", type=int,   default=5)
    p.add_argument("--panel_every",    type=int,   default=100,
                   help="Generate density comparison panels every this many saved steps.")
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--out",            type=str,
                   default="results/run_default")
    p.add_argument(
        "--run_methods", nargs="+", choices=("pbme", "midpoint"),
        default=["pbme", "midpoint"],
        help=(
            "Execute requested scheme(s); midpoint-only is reserved for "
            "hash-matched recovery of an interrupted paired run."
        ),
    )
    p.add_argument("--R0",             type=float, default=-15.0)
    p.add_argument("--P0",             type=float, default= 40.0)
    p.add_argument("--sigma_R",        type=float, default=1.0)
    p.add_argument("--mass",           type=float, default=2000.0,
                   help="Nuclear mass [a.u.]. Used by the dynamics and by the automatic scattering-time controls.")
    p.add_argument("--hbar",           type=float, default=1.0,
                   help="Planck constant in atomic units (default 1).")

    # ------------------------------------------------------------------
    # Physical time-grid controls.
    # ------------------------------------------------------------------
    p.add_argument("--t_final", type=float, default=None,
                   help="Physical final time in a.u. If supplied, overrides --n_steps; "
                        "the code chooses n_steps = ceil(t_final/dt_eff) and adjusts "
                        "dt downward so the run ends exactly at t_final. Example: --t_final 12000.")
    p.add_argument("--scattering_cycles", type=float, default=None,
                   help="Final time T = scattering_cycles * M*abs(R0)/abs(P0). "
                        "Defaults to 2 (the standard incoming/crossing/outgoing window) "
                        "when omitted, so every momentum traverses the seam. "
                        "Ignored when --t_final is supplied.")
    p.add_argument("--auto_dt", dest="auto_dt", action="store_true",
                   help="Automatically reduce dt for lower incoming momenta using "
                        "dt_eff = min(--dt, --auto_dt_ref*(|P0|/--auto_dt_ref_p0)^power).")
    p.add_argument("--no_auto_dt", dest="auto_dt", action="store_false",
                   help="Disable momentum-dependent dt scaling and use --dt directly.")
    p.set_defaults(auto_dt=True)
    p.add_argument("--auto_dt_ref", type=float, default=0.5,
                   help="Reference stable timestep at --auto_dt_ref_p0. Default: 0.5 a.u. at P0=40.")
    p.add_argument("--auto_dt_ref_p0", type=float, default=40.0,
                   help="Reference momentum for automatic timestep scaling. Default: 40.")
    p.add_argument("--auto_dt_power", type=float, default=0.5,
                   help="Exponent in dt ∝ (|P0|/auto_dt_ref_p0)^power. Default 0.5 is "
                        "the accuracy-balanced exponent for the 2nd-order scheme (holds "
                        "the global error ~ T·dt² fixed since T ∝ 1/|P0|), giving finer "
                        "dt for slower packets. Raise toward 1.0 ONLY if a dt-convergence "
                        "study shows the weight-ODE stiffness needs it.")
    p.add_argument("--steps_per_eperiod", type=float, default=30.0,
                   help="Steps per electronic/mapping oscillation period. Sets a "
                        "P0-independent dt ceiling from the model's diabatic eigenvalue "
                        "span (≈2 a.u. for the default dual model, so usually slack behind "
                        "--dt_max). Lower it to force finer resolution of the mapping phase.")
    p.add_argument("--dt_min", type=float, default=0.02,
                   help="Lower safety bound for automatic dt.")
    p.add_argument("--dt_max", type=float, default=0.5,
                   help="Upper safety bound for automatic dt. Also prevents high-P0 runs from using a larger dt than validated.")
    p.add_argument("--init_state",     type=int,   default=0)
    p.add_argument("--n_opt_steps",    type=int,   default=250)
    # Start in a bona-fide ridge-regression regime with a visible diagonal
    # noise level, then let optimization adapt sigma_n during refits.
    p.add_argument("--fix_sigma_n", action="store_true",
                   help="Pin sigma_n at init_log_sigma_n instead of optimizing "
                        "it jointly with sigma_f and the lengthscales "
                        "(pre-2026-07 behaviour).")
    p.add_argument("--init_log_sigma_n", type=float, default=-10.0)
    p.add_argument("--log_sn_floor",     type=float, default=-8.0)
    p.add_argument("--feature_zscore", dest="feature_zscore", action="store_true",
                   help="Use z-score normalized internal feature coordinates.")
    p.add_argument("--no_feature_zscore", dest="feature_zscore", action="store_false")
    p.set_defaults(feature_zscore=False)
    p.add_argument("--recompute_feature_zscore", action="store_true",
                   help="Recompute z-score statistics on each refit (normally off).")
    p.add_argument("--validation_fraction", type=float, default=0.10,
                   help="Deterministic validation fraction used for early stopping.")
    p.add_argument("--early_stop_patience", type=int, default=25,
                   help="Stop hyperparameter optimization when validation MAE stalls.")
    p.add_argument("--early_stop_min_delta", type=float, default=1.0e-6,
                   help="Minimum MAE improvement required to reset early-stop patience.")
    p.add_argument("--l2_regularization", type=float, default=0.0,
                   help="L2 regularization applied to log-hyperparameters during optimization.")
    p.add_argument("--constraints", dest="constraints_enabled", action="store_true",
                   help="Enable KKT moment constraints in the GP alpha projection.")
    p.add_argument("--no_constraints", dest="constraints_enabled", action="store_false")
    p.set_defaults(constraints_enabled=True)
    # Hyperparameter policy.  The production default is bounded-L-BFGS
    # "breathing": sigma_f is anchored, lengthscales adapt under a shrinkage
    # prior, and sigma_n floats unless --fix_sigma_n is requested.  The legacy
    # hard-freeze path is retained behind
    # --freeze_hypers for diagnostic comparisons.
    p.add_argument("--freeze_hypers", dest="freeze_hypers_after_initial_fit",
                   action="store_true",
                   help="Force legacy hard-freeze of (sigma_f, ell, sigma_n) at initial-fit values. "
                        "Overrides --refit_hyper_policy.  Diagnostic only.")
    p.add_argument("--no_freeze_hypers", dest="freeze_hypers_after_initial_fit",
                   action="store_false",
                   help="Honor --refit_hyper_policy (default).")
    p.set_defaults(freeze_hypers_after_initial_fit=False)

    # New per-refit hyperparameter policy knobs (surfaced through
    # GPDensityConfig).
    p.add_argument("--refit_hyper_policy", type=str, default="breathing",
                   choices=["frozen", "breathing", "adaptive", "free"],
                   help="Refit policy. breathing (default) uses bounded L-BFGS for "
                        "lengthscales and floating sigma_n with shrinkage/trust-region "
                        "guards; adaptive invokes the same solve only when cloud spread "
                        "triggers it; frozen is the legacy no-optimization control; free "
                        "is an unanchored diagnostic policy.")
    p.add_argument("--refit_opt_steps", type=int, default=100,
                   help="Projected L-BFGS outer steps per refit in breathing mode; "
                        "L-BFGS outer steps in free mode.")
    p.add_argument("--adaptive_fit_rms_target", type=float, default=1.0e-6,
                   help="Adaptive hyperparameter policy trigger: refit when frozen-kernel support RMS exceeds this target.")
    p.add_argument("--adaptive_cloud_ratio_target", type=float, default=4.0,
                   help="Adaptive hyperparameter policy trigger: refit when Var(Z_d)/ell_d^2 exceeds this value on an informative axis.")
    p.add_argument("--adaptive_opt_steps", type=int, default=5,
                   help="Small LBFGS/optimizer budget for each adaptive hyperparameter update.")
    p.add_argument("--adaptive_cooldown", type=int, default=20,
                   help="Minimum number of refits between adaptive hyperparameter triggers.")
    p.add_argument("--lengthscale_prior_weight", type=float, default=0.1,
                   help="Coefficient of (1/D)||log ell - log ell_0||^2 shrinkage prior "
                        "in breathing mode.  Calibration: too large (≳1) and the prior "
                        "gradient dominates the LOOCV data gradient for signed SEO labels "
                        "(|y|≲1e-5) so ℓ freezes at ℓ_0; too small (=0) and MLL can drift "
                        "ℓ monotonically upward (oversmoothing).  0.1 is a reasonable middle "
                        "that lets ℓ track the evolving cloud while the trust-region clip "
                        "prevents runaway.  Prior runs with 10.0 produced ℓ frozen at ℓ_0 "
                        "and density-reconstruction jumps of 130-160%% between snapshots.")
    p.add_argument("--lengthscale_prior_clip", type=float, default=0.5,
                   help="Trust-region clip: |log ell - log ell_0| <= clip (natural log units). "
                        "0.5 corresponds to ℓ ∈ [ℓ_0/e^0.5, ℓ_0·e^0.5] ≈ [ℓ_0/1.65, ℓ_0·1.65].")
    # Breathing anchor — what does the prior shrink TOWARD?
    #
    # Long PBME runs near avoided crossings showed fit_rms creeping up
    # monotonically (e.g. 8.9e-4 -> 1.2e-3 over 35 steps) while the cloud
    # broadened by ~10% — the prior was anchored at t=0 lengthscales and
    # actively prevented ℓ from following the cloud.  The 'ewma' policy
    # (default) tracks cloud geometry with time constant ~10 refits, so
    # ℓ_anchor smoothly follows VR/VP growth.  Set to 'initial' to
    # reproduce the legacy frozen-anchor behavior.
    p.add_argument("--breathing_anchor_policy", type=str, default="ewma",
                   choices=["initial", "cloud_mad", "ewma"],
                   help="Where the breathing prior shrinks lengthscales TOWARD: "
                        "'initial' (legacy: frozen at t=0 fit values), "
                        "'cloud_mad' (per-refit: kappa·MAD of current cloud — best fidelity, can be jittery), "
                        "'ewma' (recommended: EWMA of cloud_mad with t=0 init).")
    p.add_argument("--breathing_anchor_mad_factor", type=float, default=0.4,
                   help="Bandwidth heuristic: ell_anchor[d] = factor·MAD_d. "
                        "0.4 is the standard ARD-RBF bandwidth choice. "
                        "With ewma this means the anchor starts at ell_0 and slowly "
                        "drifts toward 0.4·MAD ≈ 0.27·sigma_R, providing a mild "
                        "shrinkage prior that reduces KKT correction size without "
                        "collapsing to the Silverman bandwidth immediately.")
    p.add_argument("--breathing_anchor_ewma_beta", type=float, default=0.9,
                   help="EWMA decay: anchor(t) = β·anchor(t-1) + (1-β)·cloud_mad(t). "
                        "β=0.9 → time constant ~10 refits. The anchor starts at ell_0 "
                        "(reg≈0 at t=0, LOOCV dominates) and drifts toward 0.4·MAD "
                        "over ~10-20 steps, gently nudging ell smaller to improve "
                        "KKT constraint satisfaction while LOOCV resists over-shrinkage.")
    p.add_argument("--lengthscale_prior_weight_per_dim", type=str, default="",
                   help="Per-dim prior weights (comma-separated 6 floats).  Overrides the scalar "
                        "--lengthscale_prior_weight.  Example: '0.01,0.01,0.1,0.1,0.1,0.1' "
                        "loosens the (R, P) prior 10× vs the mapping dims, allowing nuclear "
                        "lengthscales to track de-Broglie-scale density structure near avoided "
                        "crossings.  Leave empty to use the scalar weight uniformly.")

    # ESS resampling (opt-in).  Fixes the norm->0 pathology at the cost of
    # biasing the estimator toward the current surrogate.
    p.add_argument("--enable_ess_resampling", action="store_true",
                   help="When ESS/N falls below threshold, relabel y <- rho_hat(Z) and refit. "
                        "Destroys carried-label Liouville conservation but prevents catastrophic "
                        "norm blowup in long runs where signed-weight cancellation is severe.")
    # BUG FIX 1: bare '%' in argparse help strings causes ValueError when
    # argparse tries to %-interpolate the help text (e.g. when -h is passed).
    # The character sequence '%)'  is parsed as a format specifier where ')'
    # is not a valid conversion character, crashing with:
    #   ValueError: unsupported format character ')' (0x29) at index 55
    # Fix: escape any literal '%' as '%%' in argparse help strings.
    p.add_argument("--ess_resample_threshold", type=float, default=0.05,
                   help="ESS/N fraction below which relabeling fires (default 5%%).")
    p.add_argument("--ess_resample_cooldown", type=int, default=25,
                   help="Minimum steps between consecutive resamples.")
    p.add_argument("--ess_resample_max", type=int, default=100,
                   help="Hard cap on total resamples over the run.")
    # Legacy safeguard: per-step cap on |dt·Q| at q_clip_frac of |y|.
    # OFF by default — the csz label integrator already conserves the
    # discrete probability Σ_i ω_i y_i to machine precision per step,
    # so q_clip is no longer needed.  Kept as an opt-in for diagnostic
    # comparison with the older label-Euler scheme.
    p.add_argument("--apply_q_clip",    dest="apply_q_clip", action="store_true",
                   help="Enable legacy per-point cap on midpoint Q "
                        "corrections (default: off; csz integrator handles this).")
    p.add_argument("--no_apply_q_clip", dest="apply_q_clip", action="store_false",
                   help="(Default) disable Q clipping.")
    p.set_defaults(apply_q_clip=False)
    p.add_argument("--q_clip_frac",      type=float, default=0.3,
                   help="Per-step, per-point cap on |dt*Q/y| if q_clip is on.")
    p.add_argument("--include_abs_integral", action="store_true")
    p.add_argument("--quiet",          action="store_true")
    p.add_argument("--skip_figures", action="store_true",
                   help="Save numerical data/manifests without rendering the full figure suite (validation campaigns).")

    # ---------------------------------------------------------------
    # Flow-correction geometric knobs.
    #
    # CORRECTED 2026-07: this comment used to say the flow correction "is
    # ALWAYS applied", while Dynamics.MidpointScheme's own docstring
    # simultaneously called these same knobs dead legacy params. Neither
    # was accurate — flow correction is now a real mechanism, gated by
    # --flow_fraction (0 by default, so existing runs are unaffected).
    # axes/grad_floor/step_cap only take effect when flow_fraction > 0.
    # ---------------------------------------------------------------
    p.add_argument("--flow_correction_axes", choices=["P_only", "all"],
                   default="P_only",
                   help="Which phase-space axes feel the flow correction. "
                        "'P_only' (default) restricts to nuclear momentum, "
                        "preserving mapping Casimir and the kinematic R-P "
                        "relation.  'all' uses the full 6-D gradient.")
    p.add_argument("--flow_correction_grad_floor", type=float, default=1.0e-8,
                   help="Tikhonov regulariser for |∇ρ|² in the v_corr "
                        "denominator (default 1e-8).")
    p.add_argument("--flow_correction_step_cap", type=float, default=0.5,
                   help="Per-axis cap on |Δz_corr| in characteristic units "
                        "(σ_R for R, σ_P for P, sqrt(ℏ) for mapping). "
                        "Default 0.5.")
    p.add_argument("--flow_fraction", type=float, default=0.0,
                   help="Fraction of Q routed through the flow-correction "
                        "displacement instead of the weight ODE at each "
                        "stage; the remaining (1 - flow_fraction) still "
                        "goes through --weight_scheme. Default 0.0 "
                        "reproduces the previous (flow-correction-inactive) "
                        "behaviour exactly. Mutually exclusive with "
                        "--label_scheme=linear.")

    # ---------------------------------------------------------------
    # MidpointScheme weight-update variant
    # ---------------------------------------------------------------
    # Both variants update a per-trajectory correction weight w_i via a
    # midpoint refit of the GP.  "midpoint" = explicit Heun (k¹ then k²)
    # on  ̇w = −Q/y.  "cayley"  = symmetric Cayley map on rate
    # σ = −Q/ρ̂, which is 2nd-order time-symmetric and bounded for any
    # real σ.  Only consulted when --label_scheme=weight.
    p.add_argument("--weight_scheme", choices=["midpoint", "cayley"],
                   default="midpoint",
                   help="QCLE weight-update rule used inside MidpointScheme: "
                        "'midpoint' = explicit-midpoint Heun (default); "
                        "'cayley' = symmetric Cayley (1+Δt/2 σ)/(1−Δt/2 σ).")
    p.add_argument("--label_scheme", choices=["weight", "linear", "strang"],
                   default="weight",
                   help="'weight' (default): scalar per-point Heun/Cayley "
                        "scheme via --weight_scheme, unchanged behaviour. "
                        "'linear': EXPERIMENTAL Crank-Nicolson integrator "
                        "of the linear label-product ODE b_dot = A b, "
                        "A = L K^-1 (Operator.compute_L_matrix / "
                        "GPDensity.solve_K). Not yet validated against "
                        "finite differences — treat as a research "
                        "prototype, not a production default.")

    # ---------------------------------------------------------------
    # Density-construction architecture  (REQUIRED, NO DEFAULT)
    # ---------------------------------------------------------------
    # You MUST explicitly choose one of:
    #
    #   full  — legacy single-GP pipeline.  At every step, one GP fits
    #           ρ̂(z, t) directly against the full signed-SEO labels.
    #           Tends to produce large α oscillations and density-
    #           reconstruction jumps in long midpoint runs because
    #           the coefficient vector must absorb all geometric change.
    #
    #   diff  — density-difference pipeline (GPDensityDiff).  Writes
    #           ρ̂(z,t) = ρ̂₀(Φ_{-t}(z)) + δ̂(z,t) where ρ̂₀ is a
    #           frozen baseline GP fit at t=0 and δ̂ is a correction
    #           GP trained on residuals δᵢ = yᵢ(t) − yᵢ(0) (zero at
    #           t=0).  Recommended for long propagation.
    #
    # This was previously a boolean flag (--use_density_diff /
    # --no_use_density_diff) with a silent default of False, which
    # caused runs intended to be density-diff to silently use the
    # legacy path.  The new explicit required flag prevents that
    # failure mode.
    p.add_argument("--density_mode",
                   type=str, choices=["full", "diff"], required=True,
                   help="REQUIRED.  'full' = legacy single-GP pipeline; "
                        "'diff' = density-difference pipeline (baseline + correction). "
                        "No default: you must pick one explicitly so that architecture "
                        "choice is always visible in the command line and the log.")

    # Sampling-variance reduction.
    # abs_target=True (default): sample from |ρ_0| so ω_i·y_i = ±Z_abs/N
    # is a bounded constant — eliminates heavy-tailed IS weights that cause
    # MC cloud estimator divergence.  --no_abs_target recovers the raw Gaussian
    # proposal for backward-compatibility or diagnostic comparison.

    # ------------------------------------------------------------------
    # Sampling mode.  CRITICAL: prior to this flag's existence run.py
    # silently used seo_signed because build_initial_state defaults to it.
    # The "focused" sampler concentrates all trajectories on a single
    # diabatic surface (the active mapping circle), gives uncoupled
    # uniform-weight trajectories (chi=1, no sign cancellation), and is
    # the right choice for single-surface PBME/midpoint dynamics where
    # the diabatic populations should stay bounded in [0,1] up to QCLE
    # corrections.  The "seo_signed" sampler covers the full 6D Wigner
    # envelope with signed weights w_i = ±1; population estimators
    # converge only in the N→∞ limit and at finite N can drift outside
    # [0,1] proportional to (1 - chi).  Use "focused" for single-surface
    # studies; use "seo_signed" only when you specifically need the full
    # signed-mapping estimator.
    p.add_argument("--sampling_mode", type=str, default="focused",
                   choices=["seo_signed", "focused"],
                   help="Initial-cloud sampler.  'focused' (default): all "
                        "trajectories on the active diabatic surface "
                        "(uncoupled, w_i≡1, chi=1).  'seo_signed': full "
                        "6D signed Wigner envelope (signed weights, "
                        "finite-N populations can exceed [0,1] when chi<1).")
    p.add_argument("--surrogate", type=str, default="gp",
                   choices=["gp", "product", "product_transported"],
                   help="Density surrogate.  'gp' (default): plain ARD-RBF GP "
                        "fitted to the labels — original pipeline.  'product': "
                        "reference-profile factorization rho_hat = g(x)*mu(z) "
                        "with g the analytic SEO mapping profile, so the "
                        "excess operator's second mapping derivatives are "
                        "exact (removes the ~527x operator-input suppression "
                        "under focused sampling; the GP carries only the "
                        "smooth modulation).  'product_transported': as "
                        "'product' but g rides the exact MInt flow via "
                        "footpoint pullback, keeping the operator accurate as "
                        "the density's mapping structure evolves, not only at "
                        "t=0.  NOTE for both product modes: GP-analytic "
                        "moments are analytic for the static product profile; the "
                        "transported profile reports those global moments as NaN and "
                        "uses the raw cloud estimators lw_P0/lw_P1; the "
                        "operator now delivers full-strength O(1e-3) input, so "
                        "use a smaller dt and watch chi through the crossing.")
    p.add_argument("--product_g_floor_rel", type=float, default=1.0e-3,
                   help="Signed relative floor used only in the product y/g label transform. Its absolute value and affected support fraction are saved each step.")
    p.add_argument("--abs_target",    dest="abs_target", action="store_true",
                   help="Sample from |ρ_0| via rejection (bounded ω, default: on).")
    p.add_argument("--no_abs_target", dest="abs_target", action="store_false",
                   help="Sample from Gaussian q (raw signed SEO, heavy-tailed ω).")
    p.set_defaults(abs_target=True)
    p.add_argument("--abs_cap_quantile", type=float, default=0.999,
                   help="Truncation quantile of |w_poly| for the abs-target rejection sampler. "
                        "Higher → less truncation bias, lower → faster.  Default: 0.999.")
    p.add_argument("--omega_clip_quantile", type=float, default=None,
                   help="If given, cap ω_i at the empirical quantile q of ω.  E.g. 0.99 "
                        "drops the top 1%% of ω values.  Cheap variance reducer with a "
                        "small documented bias.  Default: off (no clipping).")

    args = p.parse_args()

    # Translate the explicit CLI mode ('full' / 'diff') into the boolean
    # that downstream modules still expect.  Keeping this bridge local
    # to run.py means Dynamics.py / GP_DensityDiff.py / Collector.py
    # don't need to know about the CLI naming change.
    args.use_density_diff = (args.density_mode == "diff")

    # The product surrogates factor rho_hat = g*mu on the full density and do
    # not compose with the baseline+correction split in v1; fail early with a
    # clear message rather than deep inside build_initial_state.
    if args.surrogate in ("product", "product_transported") \
            and args.use_density_diff:
        raise SystemExit(
            f"--surrogate {args.surrogate} is incompatible with "
            "--density_mode diff in v1 (the product factorization applies to "
            "the full density, not the baseline+correction split).  Use "
            "--density_mode full.")

    # Resolve dt/n_steps from physical scattering controls before any dynamics,
    # consistency checks, or output logging.
    _resolve_time_grid(args)

    os.makedirs(args.out, exist_ok=True)
    abs_out = os.path.abspath(args.out)
    verbose = not args.quiet
    t0 = time.time()

    banner = "=" * 72
    print(banner)
    print(f"  OUTPUT DIRECTORY (absolute): {abs_out}")
    print(f"  cwd when launched:           {os.getcwd()}")
    if args.density_mode == "diff":
        print(f"  Density mode:                DIFF  "
              f"(ρ̂ = baseline + correction)")
    else:
        print(f"  Density mode:                FULL  "
              f"(legacy single-GP surrogate)")
        print(f"")
        print(f"  !!! Long midpoint runs with --density_mode full tend to show")
        print(f"      large α oscillations and density-reconstruction jumps.")
        print(f"      Pass --density_mode diff for the baseline+correction path.")
    print(f"  Sampling mode:               {args.sampling_mode.upper()}  "
          + ("(focused: uncoupled, w_i=1, populations bounded by Casimir + Q)"
             if args.sampling_mode == "focused"
             else "(signed SEO: chi<1 → finite-N populations can exceed [0,1])"))
    print(f"  Hyper policy:                "
          f"{'FROZEN (legacy --freeze_hypers)' if args.freeze_hypers_after_initial_fit else args.refit_hyper_policy}")
    print(f"  lengthscale_prior_weight:    {args.lengthscale_prior_weight:.3g}")
    print(f"  lengthscale_prior_clip:      {args.lengthscale_prior_clip:.3g}")
    if args.lengthscale_prior_weight_per_dim.strip():
        print(f"  lengthscale_prior_weight_per_dim: [{args.lengthscale_prior_weight_per_dim}]")
    print(f"  breathing_anchor_policy:     {args.breathing_anchor_policy!r}")
    print(f"  breathing_anchor_mad_factor: {args.breathing_anchor_mad_factor:.3g}")
    print(f"  breathing_anchor_ewma_beta:  {args.breathing_anchor_ewma_beta:.3g}")
    print(f"  mass, hbar:                  {args.mass:.6g}, {args.hbar:.6g}")
    print(f"  scattering t_c=M|R0|/|P0|:   {args.collision_time:.6g} a.u.")
    print(f"  M/|P0| time-per-bohr:        {args.time_per_bohr:.6g} a.u./bohr")
    print(f"  requested dt upper bound:    {args.dt_requested:.6g}")
    if args.auto_dt:
        print(f"  auto-dt nominal before exact endpoint: {args.dt_auto_nominal:.6g} "
              f"(ref dt={args.auto_dt_ref:g} at P0={args.auto_dt_ref_p0:g}, power={args.auto_dt_power:g})")
        if np.isfinite(args.dt_electronic_ceiling):
            print(f"  electronic-phase dt ceiling: {args.dt_electronic_ceiling:.6g} a.u. "
                  f"({args.steps_per_eperiod:g} steps/period)")
    else:
        print(f"  auto-dt:                     off")
    print(f"  resolved dt:                 {args.dt:.6g}  "
          f"(n_steps={args.n_steps}, total T={args.dt*args.n_steps:.6g})")
    print(banner)

    # ---------------------------------------------------------------
    # 1-2. Initial cloud + GP fit
    # ---------------------------------------------------------------
    # build_initial_state raises if abs_target=True && sampling_mode=focused
    # (abs_target is a signed-SEO variance-reduction trick that has no
    # meaning for the uncoupled-trajectory focused estimator).  Auto-
    # clear abs_target when focused is requested so users don't have to
    # remember to pass both --sampling_mode focused --no_abs_target.
    if args.sampling_mode == "focused" and args.abs_target:
        print(f"[run] --sampling_mode focused: disabling abs_target "
              f"(focused trajectories carry weight=1 by construction; "
              f"abs_target is a signed-SEO variance-reduction trick).")
        args.abs_target = False

    # ------------------------------------------------------------------
    # Sampler-driven GP-config overrides.
    #
    # The LabelInformation contract that `MMSTSampler.sample_focused`
    # attaches to each cloud declares the GP-side configuration that is
    # correct for focused labels:
    #
    #   * recommended_loss = "loocv"
    #         Focused labels y_i = K_focus·W_cl(R_i,P_i) are nearly noise-
    #         less (σ_n pinned at 1e-5 worth of label scale).  MLL has a
    #         classical oversmoothing bias in the noise-free regime: it
    #         maximises a tradeoff between data fit and a log-det-K_y
    #         complexity term that drives ℓ above the natural data
    #         scale (empirically ~1.7× wavepacket width).  LOO-CV
    #         directly minimises prediction error and is immune.
    #         (Confirmed by `test_alpha_sign_faithfulness_focused`:
    #          MLL gives α_neg_L1_frac ≈ 0.14 at t=0; LOO-CV ≈ 5e-13.)
    #
    #   * apply_kkt = False
    #         Focused labels are W_cl·K_focus, not the physical density.
    #         Their 6D integral lives in label units, not physical
    #         units, so KKT moment projection against `normalization=1`
    #         destroys the well-fit α₀.  apply_kkt=False is honoured
    #         inside `_apply_kkt_projection` (early return); the
    #         physical observables come from the cloud Riemann sum
    #         which is Liouville-conserved exactly.
    #
    #   * refit_hyper_policy = "adaptive"
    #         When the cloud spreads beyond the kernel bandwidth (post-
    #         crossing bifurcation), the frozen ℓ produces unphysical
    #         third derivatives between bifurcated lobes — corrupting
    #         the QCLE Q operator.  "adaptive" fires a brief breathing
    #         optimisation only when Var(Z_d)/ℓ_d² > 4 on informative
    #         axes, recovering well-conditioned ℓ without per-step
    #         overhead in the well-conditioned regime.
    #
    # `LabelInformation.recommended_loss` is already honoured inside
    # `GPDensity.pin_lengthscales` (line ~1041 of GP_Density.py) — it
    # overrides `cfg.use_loocv` per the sampler's declaration.  But the
    # refit policy is a per-config property; we override it here.
    # `apply_kkt` is also enforced via the pin contract, independent of
    # any config flag.
    #
    # CLI semantics: this override is silent and unconditional for
    # `--sampling_mode focused`.  If a user has a research reason to
    # combine focused sampling with a non-adaptive policy, they should
    # edit this block — leaving an opt-out flag would invite the
    # exact silent-default regression that caused the present-day
    # P0 = 1.029, P1 = −0.011 disaster log.
    if args.sampling_mode == "focused":
        if args.refit_hyper_policy != "adaptive":
            print(f"[run] --sampling_mode focused: forcing "
                  f"--refit_hyper_policy adaptive "
                  f"(was {args.refit_hyper_policy!r}; the cloud-ratio "
                  f"trigger is required to handle post-crossing "
                  f"bifurcation without GP-derivative blowup).")
            args.refit_hyper_policy = "adaptive"
        _use_loocv_override = True   # LOO-CV; see rationale block above
    else:
        _use_loocv_override = False  # MLL: signed-SEO baseline default

    print(f"[run] Sampling + fitting initial GP  "
          f"(N={args.n_train}, seed={args.seed}, "
          f"sampling_mode={args.sampling_mode!r})...")
    init_state = Simulation.build_initial_state(
        n_train=args.n_train,
        classical_params=GaussianWavePacketParams(
            R0=args.R0, P0=args.P0, sigma_R=args.sigma_R, hbar=args.hbar,
        ),
        mapping_params=MappingInitParams(
            nstates=2, init_state=args.init_state, hbar=args.hbar,
        ),
        gp_config=GPDensityConfig(
            n_opt_steps=args.n_opt_steps,
            # Full geometric fit (2026-07): σ_n joins σ_f and all six
            # lengthscales in the joint NLL/LOOCV optimization, guarded by
            # the validation early stop (patience/min_delta, best-state
            # restore) and the log_sn_floor bound — the previous
            # unconditional pin silently excluded one hyperparameter from
            # every production fit.  Pass --fix_sigma_n to restore pinning.
            fix_sigma_n=args.fix_sigma_n,
            init_log_sigma_n=args.init_log_sigma_n,
            log_sn_floor=args.log_sn_floor,
            reinit_lengthscales=False,
            jitter=1e-6,

            feature_zscore=args.feature_zscore,
            recompute_feature_zscore=args.recompute_feature_zscore,
            interpolate_targets=False,
            validation_fraction=args.validation_fraction,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            l2_regularization=args.l2_regularization,

            constraints_enabled=args.constraints_enabled,
            # `_use_loocv_override` is the run-level baseline; the GP's
            # `pin_lengthscales` will further override per
            # `LabelInformation.recommended_loss` (which currently also
            # declares "loocv" for focused — so they agree, but the
            # pin's choice is authoritative).
            use_loocv=_use_loocv_override,
            refit_hyper_policy=args.refit_hyper_policy,
            refit_opt_steps=args.refit_opt_steps,
            adaptive_fit_rms_target=args.adaptive_fit_rms_target,
            adaptive_cloud_ratio_target=args.adaptive_cloud_ratio_target,
            adaptive_opt_steps=args.adaptive_opt_steps,
            adaptive_cooldown=args.adaptive_cooldown,
            lengthscale_prior_weight=args.lengthscale_prior_weight,
            lengthscale_prior_clip=args.lengthscale_prior_clip,
            lengthscale_prior_weight_per_dim=(
                None if not args.lengthscale_prior_weight_per_dim.strip()
                else np.array([float(x) for x in args.lengthscale_prior_weight_per_dim.split(",")],
                              dtype=np.float64)
            ),
            breathing_anchor_policy=args.breathing_anchor_policy,
            breathing_anchor_mad_factor=args.breathing_anchor_mad_factor,
            breathing_anchor_ewma_beta=args.breathing_anchor_ewma_beta,
        ),
        dynamics=PBMEMIntDynamics(params=PBMEMIntParams(mass=args.mass, hbar=args.hbar)),
        seed=args.seed,
        sampling_mode=args.sampling_mode,
        surrogate=args.surrogate,
        product_g_floor_rel=args.product_g_floor_rel,
        use_density_diff=args.use_density_diff,
        abs_target=args.abs_target,
        abs_cap_quantile=args.abs_cap_quantile,
        omega_clip_quantile=args.omega_clip_quantile,
    )
    print(f"[run]   initial fit done ({time.time()-t0:.1f}s)")

    # Sampling-variance configuration summary
    sd = init_state.sampling_diagnostics or {}
    if sd:
        print(f"[run] sampling: abs_target={'on' if sd.get('abs_target', 0.0) > 0.5 else 'off'}"
              + (f" (cap_q={sd.get('abs_cap_quantile'):.3f})" if sd.get('abs_target', 0.0) > 0.5 else "")
              + ("  ω-clip: off" if not np.isfinite(sd.get('omega_clip_quantile', float('nan')))
                 else f"  ω-clip: q={sd.get('omega_clip_quantile'):.3f},"
                      f" max_raw={sd.get('omega_max_raw'):.3e},"
                      f" max_used={sd.get('omega_max_used'):.3e},"
                      f" n_clipped={int(sd.get('omega_clip_n', 0))},"
                      f" mass_lost={sd.get('omega_clip_mass_frac'):.3e}"))

    from Monodromy import check_mint_jax_consistency
    cc = check_mint_jax_consistency(
        dynamics=init_state.gp.dynamics, dt=args.dt,
    )
    print(f"[run] MInt np/jax consistency: {cc}")

    state_pbme = copy.deepcopy(init_state) if "pbme" in args.run_methods else None
    state_mid  = copy.deepcopy(init_state) if "midpoint" in args.run_methods else None

    # The paired comparison is valid only if both schemes start from the same
    # support realization.  Deep-copying above enforces that contract; the
    # fingerprint is persisted in both run manifests so it is independently
    # auditable rather than asserted only in prose.
    from Reproducibility import array_fingerprint, write_json
    paired_cloud_sha256 = array_fingerprint(init_state.Z)
    comparison_contract = (
        "PBME and MIDPOINT use deep copies of the identical initial support, "
        "labels, weights, and fitted GP"
        if set(args.run_methods) == {"pbme", "midpoint"}
        else
        "Single-method recovery from the deterministic initial support; the "
        "orchestration layer must verify this hash against the companion method"
    )
    paired_metadata = {
        "cli_arguments": vars(args),
        "paired_initial_cloud": True,
        "paired_initial_cloud_sha256": paired_cloud_sha256,
        "comparison_contract": comparison_contract,
    }
    write_json(os.path.join(args.out, "run_manifest.json"), paired_metadata)

    # ---------------------------------------------------------------
    # 3-4. Run both schemes
    # ---------------------------------------------------------------
    common = dict(dt=args.dt, n_steps=args.n_steps,
                  snapshot_every=args.snapshot_every,
                  include_abs_integral=args.include_abs_integral,
                  verbose=verbose, output_dir=args.out,
                  apply_q_clip=args.apply_q_clip,
                  q_clip_frac=args.q_clip_frac,
                  flow_correction_axes=args.flow_correction_axes,
                  flow_correction_grad_floor=args.flow_correction_grad_floor,
                  flow_correction_step_cap=args.flow_correction_step_cap,
                  flow_fraction=args.flow_fraction,
                  weight_scheme=args.weight_scheme,
                  label_scheme=args.label_scheme,
                  freeze_hypers_after_initial_fit=args.freeze_hypers_after_initial_fit,
                  enable_ess_resampling=args.enable_ess_resampling,
                  ess_resample_threshold=args.ess_resample_threshold,
                  ess_resample_cooldown=args.ess_resample_cooldown,
                  ess_resample_max=args.ess_resample_max,
                  use_density_diff=args.use_density_diff)

    pbme_path = os.path.join(args.out, "pbme.npz")
    mid_path = os.path.join(args.out, "midpoint.npz")
    if "pbme" in args.run_methods:
        print(f"\n[run] ==== PBME ====")
        sim_pbme = Simulation(
            DynamicsConfig(scheme="pbme", run_name="pbme", **common),
            state_pbme, run_metadata=paired_metadata,
        )
        sim_pbme.run()
        pbme_path = sim_pbme.save()
        print(f"[run]   saved to {pbme_path}")

    if "midpoint" not in args.run_methods:
        if not args.skip_figures:
            raise ValueError(
                "--run_methods pbme requires --skip_figures because no "
                "MIDPOINT artifact exists for a comparison panel"
            )
        print("\n[run] MIDPOINT skipped by --run_methods.")
        return

    print(f"\n[run] ==== Midpoint (QCLE) ====")
    print(f"[run]   weight_scheme={args.weight_scheme}  label_scheme={args.label_scheme}  "
          f"flow_fraction={args.flow_fraction} (axes={args.flow_correction_axes}, "
          f"grad_floor={args.flow_correction_grad_floor}, step_cap={args.flow_correction_step_cap})")
    if args.flow_fraction == 0.0 and args.label_scheme == "weight":
        print(f"[run]   -> QCLE correction ACTIVE via the correction-weight label "
              f"integrator (weight_scheme='{args.weight_scheme}'): every step, "
              f"Q updates the per-trajectory weight w, the GP refits to w*y, and "
              f"the density shape + all observables carry the correction. "
              f"Only the OPTIONAL flow-displacement channel (flow_fraction=0) and "
              f"the experimental linear b-ODE (label_scheme='weight') are off, so "
              f"fc_* diagnostics are 0 by construction — the label-integrator "
              f"activity is in dw_*/w_*/label_dy_* instead. "
              f"Pass --flow_fraction > 0 or --label_scheme linear to engage the "
              f"alternative channels.")
    sim_mid = Simulation(
        DynamicsConfig(scheme="midpoint", run_name="midpoint", **common),
        state_mid, run_metadata=paired_metadata,
    )
    sim_mid.run()
    mid_path = sim_mid.save()
    print(f"[run]   saved to {mid_path}")

    # ---------------------------------------------------------------
    # 5. Visualization
    # ---------------------------------------------------------------
    if args.skip_figures:
        print(f"\n[run] Figure generation skipped by --skip_figures.")
        paths = {}
    else:
        print(f"\n[run] Generating figures...")
        paths = produce_all_figures_publication(
            pbme_path_no_ext=os.path.join(args.out, "pbme"),
            midpoint_path_no_ext=os.path.join(args.out, "midpoint"),
            out_dir=args.out,
            snapshot_stride=args.panel_every,
        )

    # BUG FIX 2: produce_all_comparison_figures() returns marginal-figure keys
    # with the structure "marg_step{N}_1d_{ax}" and "marg_step{N}_2d_{pair}".
    # The old code used startswith("marg_1d_") / startswith("marg_2d_") which
    # never matched anything, silently making both category lists always empty
    # and misclassifying all marginal figures into time_series_keys.
    # Fix: use substring containment ("_1d_" in k / "_2d_" in k) which is
    # robust to the step-index prefix in the key name.
    time_series_keys = [k for k in paths
                        if not k.startswith("marg_")]
    marg_1d_keys     = [k for k in paths if "_1d_" in k]
    marg_2d_keys     = [k for k in paths if "_2d_" in k]

    print(f"\n  Time-series figures ({len(time_series_keys)}):")
    for k in sorted(time_series_keys):
        print(f"    {k:25s}{os.path.abspath(paths[k])}")

    print(f"\n  1D marginals ({len(marg_1d_keys)}):")
    for k in sorted(marg_1d_keys):
        print(f"    {k:25s}{os.path.abspath(paths[k])}")

    print(f"\n  2D marginals ({len(marg_2d_keys)}):")
    for k in sorted(marg_2d_keys):
        print(f"    {k:25s}{os.path.abspath(paths[k])}")

    print(f"\n{banner}")
    print(f"  DONE in {time.time()-t0:.1f}s")
    print(f"  All files written to: {abs_out}")
    print(f"  Total figures: {len(paths)}")
    print(f"  Data files:    {abs_out}/pbme.npz    {abs_out}/pbme.json")
    print(f"                 {abs_out}/midpoint.npz  {abs_out}/midpoint.json")
    print(banner)

    from FigureCatalog import build_figure_catalog
    caption_md, caption_json = build_figure_catalog(args.out)
    print(f"  Figure caption catalog: {os.path.abspath(caption_md)}")
    print(f"  Figure metadata catalog: {os.path.abspath(caption_json)}")


if __name__ == "__main__":
    main()
