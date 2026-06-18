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
import time

import numpy as np

from .GP_Density import GPDensityConfig
from .Sampling import GaussianWavePacketParams, MappingInitParams
from .Dynamics import DynamicsConfig, Simulation
from .Visualization import produce_all_figures_publication


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_train",        type=int,   default=1000)
    p.add_argument("--n_steps",        type=int,   default=6000,
                   help="Number of time steps (default 5600; at dt=0.5 gives T=2800).")
    p.add_argument("--dt",             type=float, default=0.5,
                   help="Time step dt (default 0.5; smaller → more stable midpoint corrections).")
    p.add_argument("--snapshot_every", type=int,   default=5)
    p.add_argument("--panel_every",    type=int,   default=100,
                   help="Generate density comparison panels every this many saved steps.")
    p.add_argument("--skip_figures", action="store_true",
                   help="Run and save pbme/midpoint outputs but skip figure generation. Useful for smoke tests and batch jobs.")
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--out",            type=str,
                   default="results/run_default")
    p.add_argument("--R0",             type=float, default=-15.0)
    p.add_argument("--P0",             type=float, default= 40.0)
    p.add_argument("--sigma_R",        type=float, default=1.0)
    p.add_argument("--init_state",     type=int,   default=0)
    p.add_argument("--n_opt_steps",    type=int,   default=250)
    # Start in a bona-fide ridge-regression regime with a visible diagonal
    # noise level, then let optimization adapt sigma_n during refits.
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
    # Hyperparameter-freezing policy.  The new production default is
    # "breathing" mode -- sigma_f and sigma_n are hard-pinned at the initial-fit
    # anchor, while lengthscales breathe with the cloud subject to a
    # shrinkage prior.  The legacy hard-freeze path is retained behind
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
    p.add_argument("--refit_hyper_policy", type=str, default="frozen",
                   choices=["frozen", "breathing", "free"],
                   help="Refit policy: frozen (default) pins all hyperparameters at the "
                        "initial-fit values and only re-solves for alpha at each step. "
                        "This is correct for PBME (y frozen, cloud translates) and gives "
                        "stable R² throughout the run. "
                        "breathing and free re-optimise lengthscales at each step, which "
                        "moves ell away from the LOOCV optimum → larger KKT corrections "
                        "→ degrading constrained R². Use breathing/free only when the "
                        "cloud geometry changes dramatically (very long runs past the "
                        "avoided crossing) and fit_r2 has visibly dropped below ~0.95.")
    p.add_argument("--refit_opt_steps", type=int, default=100,
                   help="Adam steps per refit in breathing or free mode.")
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

    # ---------------------------------------------------------------
    # Flow-correction geometric knobs.  The flow correction itself is
    # ALWAYS applied — it is the midpoint corrector that moves
    # trajectories onto QCLE characteristics rather than leaving them
    # on PBME ones.  Only the per-axis restriction, the gradient
    # regulariser, and the step cap are exposed.
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

    # ---------------------------------------------------------------
    # MidpointScheme weight-update variant
    # ---------------------------------------------------------------
    # Both variants update a per-trajectory correction weight w_i via a
    # midpoint refit of the GP.  "midpoint" = explicit Heun (k¹ then k²)
    # on  ̇w = −Q/y.  "cayley"  = symmetric Cayley map on rate
    # σ = −Q/ρ̂, which is 2nd-order time-symmetric and bounded for any
    # real σ.
    p.add_argument("--weight_scheme", choices=["midpoint", "cayley"],
                   default="midpoint",
                   help="QCLE weight-update rule used inside MidpointScheme: "
                        "'midpoint' = explicit-midpoint Heun (default); "
                        "'cayley' = symmetric Cayley (1+Δt/2 σ)/(1−Δt/2 σ).")

    # ---------------------------------------------------------------
    # Density-construction architecture
    # ---------------------------------------------------------------
    # Publication default: full-density single-GP surrogate.  At every
    # step, one GP fits ρ̂(z,t) directly against the live effective labels.
    #
    # Optional experimental mode: density-difference surrogate
    # ρ̂(z,t)=ρ̂₀(Φ_{-t}(z))+δ̂(z,t).  This mode is retained for ablation
    # and diagnostics, but it is not the default production path.
    p.add_argument("--density_mode",
                   type=str, choices=["full", "diff"], default="full",
                   help="Density surrogate architecture. 'full' (default) = "
                        "single full-density GP fitted to the live labels; "
                        "'diff' = optional baseline+correction surrogate used "
                        "for ablation/diagnostics.")

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

    # Translate the CLI mode ('full' / 'diff') into the boolean that
    # downstream modules still expect.
    args.use_density_diff = (args.density_mode == "diff")

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
              f"(optional baseline + correction surrogate)")
    else:
        print(f"  Density mode:                FULL  "
              f"(default production single-GP surrogate)")
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
    print(f"  dt:                          {args.dt:.3g}  "
          f"(n_steps={args.n_steps}, total T={args.dt*args.n_steps:.3g})")
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
            R0=args.R0, P0=args.P0, sigma_R=args.sigma_R,
        ),
        mapping_params=MappingInitParams(
            nstates=2, init_state=args.init_state,
        ),
        gp_config=GPDensityConfig(
            n_opt_steps=args.n_opt_steps,
            fix_sigma_n=True,   # pin σ_n at init_log_sigma_n; prevents MLL drift
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
        seed=args.seed,
        sampling_mode=args.sampling_mode,
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

    from .Monodromy import check_mint_jax_consistency
    cc = check_mint_jax_consistency(
        dynamics=init_state.gp.dynamics, dt=args.dt,
    )
    print(f"[run] MInt np/jax consistency: {cc}")

    state_pbme = copy.deepcopy(init_state)
    state_mid  = copy.deepcopy(init_state)

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
                  weight_scheme=args.weight_scheme,
                  freeze_hypers_after_initial_fit=args.freeze_hypers_after_initial_fit,
                  enable_ess_resampling=args.enable_ess_resampling,
                  ess_resample_threshold=args.ess_resample_threshold,
                  ess_resample_cooldown=args.ess_resample_cooldown,
                  ess_resample_max=args.ess_resample_max,
                  use_density_diff=args.use_density_diff)

    print(f"\n[run] ==== PBME ====")
    sim_pbme = Simulation(
        DynamicsConfig(scheme="pbme", run_name="pbme", **common),
        state_pbme,
    )
    sim_pbme.run()
    pbme_path = sim_pbme.save()
    print(f"[run]   saved to {pbme_path}")

    print(f"\n[run] ==== Midpoint (QCLE) ====")
    sim_mid = Simulation(
        DynamicsConfig(scheme="midpoint", run_name="midpoint", **common),
        state_mid,
    )
    sim_mid.run()
    mid_path = sim_mid.save()
    print(f"[run]   saved to {mid_path}")

    # ---------------------------------------------------------------
    # 5. Visualization
    # ---------------------------------------------------------------
    paths = {}
    if args.skip_figures:
        print(f"\n[run] Skipping figure generation (--skip_figures).")
    else:
        print(f"\n[run] Generating figures...")
        paths = produce_all_figures_publication(
            pbme_path_no_ext=os.path.join(args.out, "pbme"),
            midpoint_path_no_ext=os.path.join(args.out, "midpoint"),
            out_dir=args.out,
            snapshot_stride=args.panel_every,
        )

        # Figure-key classification is robust to step-index prefixes.
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


if __name__ == "__main__":
    main()
