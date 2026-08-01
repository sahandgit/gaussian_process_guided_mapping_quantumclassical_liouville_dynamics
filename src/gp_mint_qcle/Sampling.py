from __future__ import annotations

# --- UTF-8 console safety: prevent UnicodeEncodeError on Windows cp1252 ---
# Banners/diagnostics below print non-ASCII physics notation (α, ρ̂, Δ, →, ħ).
# Reconfigure the console streams to UTF-8 so direct execution of this module
# does not abort under Windows' default cp1252 encoding.  No-op where unsupported.
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
# --------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _as_1d_param(x: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    return arr


def _as_batch(x: ArrayLike, dim: int, name: str) -> FloatArray:
    """
    Convert x to shape (n_samples, dim).

    Accepted inputs:
      - scalar if dim == 1
      - shape (dim,)
      - shape (n_samples, dim)
    """
    arr = np.asarray(x, dtype=np.float64)

    if dim == 1 and arr.ndim == 0:
        return arr.reshape(1, 1)

    if arr.ndim == 1:
        if dim == 1:
            return arr.reshape(-1, 1)
        if arr.shape[0] == dim:
            return arr.reshape(1, dim)

    if arr.ndim == 2 and arr.shape[1] == dim:
        return arr

    raise ValueError(
        f"{name} must have shape ({dim},) or (n_samples, {dim}); got {arr.shape}."
    )


def _require_same_n(a: FloatArray, b: FloatArray, aname: str, bname: str) -> None:
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"{aname} and {bname} must have the same number of samples: "
            f"{a.shape[0]} != {b.shape[0]}"
        )


@dataclass(frozen=True)
class GaussianWavePacketParams:
    """
    Parameters for a product Gaussian nuclear wavepacket.

    Convention for the initial coordinate-space wavefunction:
        psi(R) ~ exp(-(R-R0)^2 / (4 sigma_R^2) + i P0·(R-R0)/hbar)

    Then the Wigner transform is
        W_cl(R, P)
        = prod_j [1 / (pi hbar)]
          exp[-(R_j-R0_j)^2 / (2 sigma_R_j^2)
              -2 sigma_R_j^2 (P_j-P0_j)^2 / hbar^2 ].
    """
    R0: ArrayLike
    P0: ArrayLike
    sigma_R: ArrayLike
    hbar: float = 1.0


class ClassicalGaussianWigner:
    """
    Exact positive Wigner density for a product Gaussian nuclear wavepacket.
    """

    def __init__(self, params: GaussianWavePacketParams) -> None:
        self.hbar = float(params.hbar)
        if self.hbar <= 0.0:
            raise ValueError("hbar must be positive.")

        self.R0 = _as_1d_param(params.R0, "R0")
        self.P0 = _as_1d_param(params.P0, "P0")
        self.sigma_R = _as_1d_param(params.sigma_R, "sigma_R")

        if not (self.R0.shape == self.P0.shape == self.sigma_R.shape):
            raise ValueError("R0, P0, and sigma_R must have the same shape.")

        if np.any(self.sigma_R <= 0.0):
            raise ValueError("All sigma_R values must be positive.")

        self.ndim = self.R0.size
        self.sigma_P = self.hbar / (2.0 * self.sigma_R)
        self._norm = (1.0 / (np.pi * self.hbar)) ** self.ndim

    def sample(
        self,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[FloatArray, FloatArray]:
        rng = np.random.default_rng() if rng is None else rng
        R = rng.normal(loc=self.R0, scale=self.sigma_R, size=(n_samples, self.ndim))
        P = rng.normal(loc=self.P0, scale=self.sigma_P, size=(n_samples, self.ndim))
        return R, P

    def density(self, R: ArrayLike, P: ArrayLike) -> FloatArray:
        Rb = _as_batch(R, self.ndim, "R")
        Pb = _as_batch(P, self.ndim, "P")
        _require_same_n(Rb, Pb, "R", "P")

        dR = Rb - self.R0[None, :]
        dP = Pb - self.P0[None, :]

        exponent = -np.sum(
            (dR ** 2) / (2.0 * self.sigma_R[None, :] ** 2)
            + 2.0 * (self.sigma_R[None, :] ** 2) * (dP ** 2) / (self.hbar ** 2),
            axis=1,
        )
        return self._norm * np.exp(exponent)

    def log_density(self, R: ArrayLike, P: ArrayLike) -> FloatArray:
        Rb = _as_batch(R, self.ndim, "R")
        Pb = _as_batch(P, self.ndim, "P")
        _require_same_n(Rb, Pb, "R", "P")

        dR = Rb - self.R0[None, :]
        dP = Pb - self.P0[None, :]

        return (
            self.ndim * (-np.log(np.pi * self.hbar))
            - np.sum(
                (dR ** 2) / (2.0 * self.sigma_R[None, :] ** 2)
                + 2.0 * (self.sigma_R[None, :] ** 2) * (dP ** 2) / (self.hbar ** 2),
                axis=1,
            )
        )


@dataclass(frozen=True)
class MappingInitParams:
    nstates: int
    init_state: int = 0
    hbar: float = 1.0
    gamma: float = 0.5


@dataclass
class MappingSamples:
    r: FloatArray
    p: FloatArray
    proposal_density: Optional[FloatArray] = None
    target_density: Optional[FloatArray] = None
    weight: Optional[FloatArray] = None


@dataclass
class MMSTSamples:
    R: FloatArray
    P: FloatArray
    r: FloatArray
    p: FloatArray
    proposal_density: Optional[FloatArray] = None
    target_density: Optional[FloatArray] = None
    weight: Optional[FloatArray] = None
    # Per-sampler "label information rank" describing which phase-space axes
    # the y-labels actually constrain.  Used by GPDensity to decide which
    # ARD lengthscales to estimate from data and which to pin at sampler-
    # provided physical anchors.  See `LabelInformation` below.  None for
    # backward compatibility (treated as "all axes informative").
    label_information: Optional["LabelInformation"] = None


@dataclass(frozen=True)
class LabelInformation:
    """
    Per-axis description of what the sampler's y-labels actually constrain.

    Motivation
    ----------
    The GP density surrogate is a 6D ARD-RBF kernel whose ARD lengthscales are
    six free parameters.  Whether each lengthscale can be learned from data
    depends on the SAMPLER:

    * `seo_signed` draws (r,p) from a 6D Gaussian envelope.  Labels
      y_i = q·w_λ vary along every axis ⇒ all six lengthscales are
      data-identifiable.

    * `focused` places (r,p) on the product of two circles (r_α²+p_α² fixed
      per α).  Labels y_i = W_cl(R_i,P_i)·K_focus are LITERALLY CONSTANT
      along the four mapping axes ⇒ four of the six lengthscales receive
      zero data signal.  Soft priors fail; the only correct move is to
      pin those four lengthscales at physically motivated anchor values
      and exclude them from optimization entirely.

    Contract
    --------
    For each axis d ∈ {0,...,D-1}:

      information_rank[d] = 1.0
        Labels constrain this axis; GP learns ℓ_d from data normally.

      information_rank[d] = 0.0
        Labels carry no gradient along this axis; GP pins ℓ_d at
        anchor_lengthscales[d] and excludes it from MLL/LOO-CV gradient.

      anchor_lengthscales[d]
        PHYSICAL-units anchor for axis d.  Required (finite, positive)
        wherever information_rank[d] < 1; ignored where rank == 1
        (use np.nan in that case to make the intent visible).

    The boolean view of these is also exposed via `pinned_mask` and
    `free_mask` for convenience.  In the current implementation rank is
    strictly {0, 1}; the float type is forward-compatible for future
    soft-pinning samplers (e.g. partially-focused / half-Q).

    Factories
    ---------
    * `LabelInformation.all_free()`  — every axis learned from data.
      Use for SEO-signed sampling.

    * `LabelInformation.focused_2state(mapping_params)` — pin all four
      mapping lengthscales at radius_α/2 (the bandwidth that resolves a
      focus circle without smearing it).  Use for focused sampling.
    """
    information_rank:    FloatArray     # (D,) values in [0, 1]
    anchor_lengthscales: FloatArray     # (D,) physical units; NaN where free
    source:              str = "unspecified"
    # Moment names ("normalization" | "trace" | "energy") that are
    # automatically satisfied as Casimir invariants under this sampler's
    # support geometry, and must therefore be EXCLUDED from the KKT
    # projection.  Including a redundant moment makes the Schur complement
    # rank-deficient and KKT returns garbage scaled by the constraint
    # ridge.  Example: focused sampling locks r_α²+p_α² per oscillator,
    # so the trace functional c₀₀+c₁₁ = Σ(rα²+pα²-ℏ)/(2ℏ) is a constant
    # across the training cloud — its KKT row becomes a scalar multiple
    # of the normalization row and the projection collapses.  seo_signed
    # samples fill the full (r,p) envelope, so trace is independent and
    # this tuple is empty.
    redundant_moments: Tuple[str, ...] = ()
    # Whether labels emitted by this sampler are interpretable as the
    # physical density itself (apply_kkt=True), or as a sampler-specific
    # positive proxy (apply_kkt=False).  Background:
    #
    #   * seo_signed: y_i = ρ_phys(z_i) directly.  The 6D integral of the
    #     GP-smoothed labels equals ∫ρ_phys = 1 (up to sampling noise);
    #     KKT enforces this normalization exactly and only nudges α a
    #     tiny amount (correction has O(1e-3) magnitude → R² stays > 0.99).
    #
    #   * focused: y_i = K_focus · W_cl(R_i, P_i).  These are positive
    #     IS-weighted labels — NOT physical density values.  Their 6D
    #     analytic integral has a sampler-dependent prefactor determined
    #     by how the kernel extends off the focus manifold; it does not
    #     equal 1 or K_focus or any other clean physical constant.
    #     Forcing ∫y_GP = (anything physical) via KKT destroys the
    #     well-fit α₀ — the correction has to swing α by orders of
    #     magnitude to satisfy a constraint that is incompatible with
    #     the natural integral of the data.  Physical observables (Psum,
    #     ⟨H⟩, populations) are computed independently via the cloud
    #     Riemann sum Σᵢ ωᵢ yᵢ A(zᵢ), which is conserved exactly by
    #     Liouville and does not depend on the GP integral.
    #
    # When apply_kkt=False, the GP fit is taken as-is (α = α₀ from the
    # unconstrained least-squares solve); KKT is skipped entirely.
    apply_kkt: bool = True
    # Hyperparameter-loss recommendation from the sampler.  Values:
    #   "loocv": leave-one-out cross-validation (L_LOO = mean (αᵢ/[K⁻¹]ᵢᵢ)²)
    #   "mll"  : negative log marginal likelihood
    #   None   : respect GPDensityConfig.use_loocv (legacy / no override)
    #
    # Why this exists: MLL has a well-known oversmoothing bias when the
    # labels are nearly noiseless and well-fittable.  It maximises a
    # tradeoff between data fit and a complexity term that includes the
    # log-determinant of K_y; with σ_n pinned small, MLL can prefer
    # larger lengthscales that explain variance as smooth signal plus
    # model noise, producing ℓ ~ 2× the natural data scale.  LOO-CV
    # directly minimises pointwise prediction error and is immune.
    #
    # Empirically (300 training points, dual Tully focused IC):
    #   * focused + MLL:    ℓ_R=1.73 (vs natural 1.0), fit_r2=0.79, RMS=3.8e-4
    #   * focused + LOO-CV: ℓ_R=0.93 (matches σ_R),    fit_r2≈1,    RMS=5.4e-10
    #   * signed  + MLL:    fit_r2 (post-KKT)=0.986,  RMS=5.7e-4
    #   * signed  + LOO-CV: fit_r2 (post-KKT)=0.9994, RMS=1.2e-4
    # Both factories default to LOO-CV because it is strictly better
    # here.  Users can override via GPDensityConfig.use_loocv when
    # recommended_loss=None, or by editing the factory's recommendation.
    recommended_loss: Optional[str] = "loocv"

    def __post_init__(self) -> None:
        rank = np.asarray(self.information_rank, dtype=np.float64).reshape(-1)
        anch = np.asarray(self.anchor_lengthscales, dtype=np.float64).reshape(-1)
        if rank.shape != anch.shape:
            raise ValueError(
                f"information_rank and anchor_lengthscales must have the same shape; "
                f"got {rank.shape} and {anch.shape}."
            )
        if np.any((rank < 0.0) | (rank > 1.0)):
            raise ValueError("information_rank values must lie in [0, 1].")
        # Pinned axes (rank < 1) require a finite positive anchor.
        pinned = rank < 1.0
        if np.any(pinned):
            if np.any(~np.isfinite(anch[pinned]) | (anch[pinned] <= 0.0)):
                raise ValueError(
                    "anchor_lengthscales must be finite and positive on every axis "
                    "where information_rank < 1."
                )
        # Force the stored arrays to be canonical (immutable view).
        object.__setattr__(self, "information_rank",    rank)
        object.__setattr__(self, "anchor_lengthscales", anch)
        # Validate recommended_loss.
        rl = self.recommended_loss
        if rl is not None and rl not in ("loocv", "mll"):
            raise ValueError(
                f"recommended_loss must be 'loocv', 'mll', or None; got {rl!r}."
            )

    @property
    def D(self) -> int:
        return int(self.information_rank.shape[0])

    @property
    def pinned_mask(self) -> NDArray[np.bool_]:
        """Boolean mask: True where the GP must pin this axis (rank < 1)."""
        return self.information_rank < 1.0

    @property
    def free_mask(self) -> NDArray[np.bool_]:
        return ~self.pinned_mask

    @staticmethod
    def all_free(D: int = 6, source: str = "seo_signed") -> "LabelInformation":
        """All axes learned from data.  Standard for SEO-signed sampling."""
        return LabelInformation(
            information_rank=np.ones(D, dtype=np.float64),
            anchor_lengthscales=np.full(D, np.nan, dtype=np.float64),
            source=source,
        )

    @staticmethod
    def focused_2state(mapping_params: "MappingInitParams",
                       source: str = "focused") -> "LabelInformation":
        """
        Contract for the focused MMST sampler with `nstates == 2`.

        Anchor rationale
        ----------------
        Each mapping oscillator α lives at radius R_α with
            R_active   = sqrt(2 ℏ (1+γ))         (init_state)
            R_inactive = sqrt(2 ℏ γ)             (other state)

        We anchor at ℓ = R/2 — resolves the focus circle of radius R
        without smearing across it.

        Axis ordering: (R, P, r_0, r_1, p_0, p_1) — see `Mint.D = 6`.
        Only the four mapping axes (indices 2..5) are pinned; the
        nuclear axes (R, P, indices 0, 1) remain free for the GP to
        learn from the Gaussian wavepacket marginal.

        With these four lengthscales pinned, the 4-D mapping integral of
        the kernel becomes an analytic constant (depending on γ and ℏ
        only).  The KKT moment constraints (normalization, trace,
        energy) reduce to 2-D constraints on the (R, P) GP, which is
        well-conditioned because (R, P) labels DO carry data signal
        through the W_cl(R, P) factor.
        """
        if mapping_params.nstates != 2:
            raise ValueError(
                f"LabelInformation.focused_2state requires nstates=2; "
                f"got {mapping_params.nstates}."
            )
        hbar  = float(mapping_params.hbar)
        gamma = float(mapping_params.gamma)
        lam   = int(mapping_params.init_state)
        mu    = 1 - lam

        R_active   = np.sqrt(2.0 * hbar * (1.0 + gamma))
        R_inactive = np.sqrt(2.0 * hbar * gamma)

        # Anchor at ℓ = R/2 — resolves the focus circle of radius R
        # without smearing across it.
        ell_active   = 0.5 * R_active
        ell_inactive = 0.5 * R_inactive

        # Axis layout (R, P, r_0, r_1, p_0, p_1):
        rank = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        anch = np.full(6, np.nan, dtype=np.float64)
        # r_α, p_α get the same lengthscale by symmetry (the focus circle
        # is the same in r- and p-direction for each oscillator).
        anch[2 + lam] = ell_active     # r_lambda
        anch[2 + mu]  = ell_inactive   # r_mu
        anch[4 + lam] = ell_active     # p_lambda
        anch[4 + mu]  = ell_inactive   # p_mu

        return LabelInformation(
            information_rank=rank,
            anchor_lengthscales=anch,
            source=source,
            # Trace c₀₀+c₁₁ is a Casimir under focused sampling (see
            # `redundant_moments` docstring above for details).  Declared
            # for forward-compatibility — the actual KKT path is skipped
            # entirely via apply_kkt=False below, so this is belt-and-
            # suspenders.
            redundant_moments=("trace",),
            # ────────────────────────────────────────────────────────────
            # apply_kkt=False is CORRECT — do NOT "fix" this to True.
            # ────────────────────────────────────────────────────────────
            # The focused-sampler labels y_i = W_cl(R_i,P_i)·K_focus are
            # the analytic SEO Wigner density values AT the focus points
            # (K_focus ≡ W_map^λ at the focus manifold by construction),
            # i.e. they ARE ρ_phys(Z_i) at the support points.  With
            # apply_kkt=False the GP solves α₀ = K_y⁻¹ y, which
            # interpolates these labels exactly:  ρ̂(Z_i) = ρ_phys(Z_i).
            # Two consequences, both desirable:
            #
            #   (1) FIT STABILITY.  Pure interpolation gives fit_r² ≈ 1
            #       consistently throughout dynamics — the kernel always
            #       passes through the (frozen, for PBME) labels at the
            #       current support, regardless of how far the cloud has
            #       phase-mixed.  No α-perturbation accumulates.
            #
            #   (2) CORRECT LOCAL QCLE SOURCE.  The midpoint correction
            #       Q[ρ̂](Z_i) (Operator.compute_Q) is a LOCAL third-
            #       derivative operator, LINEAR in α, with NO dependence
            #       on the global integral ∫ρ̂dz.  Because ρ̂ interpolates
            #       ρ_phys near each Z_i, the local shape (and hence its
            #       derivatives) match ρ_phys, so Q[ρ̂](Z_i) ≈ Q[ρ_phys](Z_i).
            #
            # Why apply_kkt=True is WRONG here (learned the hard way):
            # forcing ∫ρ̂dz=1 via a Schur projection moves α₀ → α₀+δα.
            # On focused (measure-zero-manifold) support the kernel-
            # smeared natural integral is ≈3, so δα is O(α₀) and
            #   - it sends fit_r² from ~1 down to ~0.9 at t=0 and then
            #     decays further (to ~0.5) as the cloud spreads and the
            #     required correction grows, AND
            #   - since Q is linear in α, it injects a SPURIOUS
            #     Q[δα] term into the QCLE source — contaminating a
            #     local operator with a global-normalization artifact
            #     that has no physical meaning.
            # The global integral simply does not need to be 1 for any
            # quantity we compute: physical observables come from the
            # cloud Riemann sum Σ ω_i b_i A_i (Liouville-conserved, α-
            # independent), and Q is local.  Skip KKT; use α₀ directly.
            apply_kkt=False,
        )


class SEOMappingWigner:
    """
    Exact SEO-based projected mapping density for a pure initial electronic state |lambda>.

    In the MMST mapping with dimensionless variables (r, p), the pure SEO state's
    mapping Wigner density is

        W_lambda(r, p)
        = (pi hbar)^(-N_s) exp[-(sum_j (r_j^2 + p_j^2))/hbar]
          * ( 2(r_lambda^2 + p_lambda^2)/hbar - 1 ).

    This is exactly the diagonal mapping kernel g_{lambda lambda}(r, p)
    under the present convention.

    Important:
      - W_lambda is an exact target density, but it is sign-changing.
      - Therefore exact sampling is naturally done through a positive Gaussian envelope
        plus a signed polynomial weight.
    """

    def __init__(self, params: MappingInitParams) -> None:
        self.nstates = int(params.nstates)
        self.init_state = int(params.init_state)
        self.hbar = float(params.hbar)

        if self.nstates <= 0:
            raise ValueError("nstates must be positive.")
        if not (0 <= self.init_state < self.nstates):
            raise ValueError("init_state must satisfy 0 <= init_state < nstates.")
        if self.hbar <= 0.0:
            raise ValueError("hbar must be positive.")

        self._norm = (1.0 / (np.pi * self.hbar)) ** self.nstates
        self._coord_std = np.sqrt(self.hbar / 2.0)

    def _as_mapping_batch(
        self,
        r: ArrayLike,
        p: ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        rb = _as_batch(r, self.nstates, "r")
        pb = _as_batch(p, self.nstates, "p")
        _require_same_n(rb, pb, "r", "p")
        return rb, pb

    def gaussian_envelope_density(self, r: ArrayLike, p: ArrayLike) -> FloatArray:
        """
        Positive Gaussian envelope:
            q(r,p) = (pi hbar)^(-N_s) exp[-(r^2 + p^2)/hbar]
        """
        rb, pb = self._as_mapping_batch(r, p)
        radius2 = np.sum(rb * rb + pb * pb, axis=1)
        return self._norm * np.exp(-radius2 / self.hbar)

    def polynomial_factor(self, r: ArrayLike, p: ArrayLike) -> FloatArray:
        """
        Signed polynomial factor:
            w_lambda(r,p) = 2(r_lambda^2 + p_lambda^2)/hbar - 1
        """
        rb, pb = self._as_mapping_batch(r, p)
        active_radius2 = rb[:, self.init_state] ** 2 + pb[:, self.init_state] ** 2
        return 2.0 * active_radius2 / self.hbar - 1.0

    def density(self, r: ArrayLike, p: ArrayLike) -> FloatArray:
        """
        Exact SEO / projected mapping density:
            W_lambda = q * w_lambda
        """
        q = self.gaussian_envelope_density(r, p)
        w = self.polynomial_factor(r, p)
        return q * w

    projected_density = density

    def sample_signed(
        self,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> MappingSamples:
        """
        Exact signed sampling representation of the SEO target.

        We sample from the positive Gaussian envelope q and store the signed weight
        so that the target is recovered as
            W_lambda(r,p) = q(r,p) * w_lambda(r,p).
        """
        rng = np.random.default_rng() if rng is None else rng

        r = rng.normal(
            loc=0.0,
            scale=self._coord_std,
            size=(n_samples, self.nstates),
        )
        p = rng.normal(
            loc=0.0,
            scale=self._coord_std,
            size=(n_samples, self.nstates),
        )

        q = self.gaussian_envelope_density(r, p)
        w = self.polynomial_factor(r, p)
        target = q * w

        return MappingSamples(
            r=r,
            p=p,
            proposal_density=q,
            target_density=target,
            weight=w,
        )

    def sample_signed_abs(
        self,
        n_samples: int,
        rng:        Optional[np.random.Generator] = None,
        *,
        cap_quantile: float = 0.999,
        max_oversample: int = 64,
    ) -> MappingSamples:
        r"""
        Importance sampling from |W_λ(r,p)| (the absolute SEO target) via
        rejection on the polynomial factor.

        Why this is a better proposal
        -----------------------------
        Standard ``sample_signed`` draws (r,p) ~ q (Gaussian envelope) and the
        IS weight is the SIGNED polynomial w_λ.  Cloud Riemann sums then have
        importance ratios |y_i| / q_i = |w_poly_i| that are unbounded as
        |r,p| → ∞ — the variance of ⟨A⟩ ≈ Σ_i ω_i y_i A(z_i) blows up because
        ω_i = 1/(N q_i) is large in tails where |w_poly| is also large.

        Sampling from the absolute target  q · |w_poly|  changes the picture:
        the geometric measure is  ω_i = 1/(N · q_i · |w_poly_i| / Z_abs)
        where Z_abs = ∫ q |w_poly| dz is the normalizer.  Then

            ω_i y_i = ω_i · ρ_0(z_i^0) = sign(w_poly_i) · Z_abs / N

        which is bounded:  every cloud point contributes ±Z_abs/N to the sum,
        and the ESS of the new measure is exactly N.  Variance now comes only
        from the sign cancellation in A(z_i), which is intrinsic — not from
        heavy-tailed importance ratios, which were avoidable.

        Implementation
        --------------
        Acceptance-rejection on the unrestricted Gaussian q.  For each draw
        (r,p) ~ q, accept with probability |w_poly(r,p)| / M, where M is the
        ``cap_quantile`` empirical quantile of |w_poly| under q (default 99.9%).
        Truncating at the cap quantile introduces a small bias (the very tail
        of |w_poly| is dropped) in exchange for a finite acceptance probability.

        The (small) truncation bias is logged via the bookkeeping field
        ``truncation_mass = E_q[|w_poly| · 1{|w_poly|>M}] / E_q[|w_poly|]``,
        estimated on the rejection-stage Gaussian draws.

        Parameters
        ----------
        n_samples : int
            Number of accepted samples to return.
        rng : np.random.Generator, optional.
        cap_quantile : float in (0,1), default 0.999.
            Empirical quantile of |w_poly| under q used as the rejection
            ceiling M.  Higher → less truncation bias, lower → faster.
        max_oversample : int, default 64.
            Safety cap on the proposal-pool inflation factor before raising.

        Returns
        -------
        MappingSamples with:
            r, p              : (n_samples, nstates)
            proposal_density  : q · |w_poly_truncated| / Z_abs  (the SAMPLING
                                density — the new q for downstream geometric
                                measure construction).  In code this is the
                                array that callers should treat as q(z_i^0).
            target_density    : ρ_0(z_i^0) = q · w_poly  (signed, unchanged).
            weight            : sign(w_poly_i) · (Z_abs in front), so that
                                ω_i · y_i = sign(w_poly_i) · Z_abs / N.

        The numerator Z_abs is estimated from the rejection-stage draws and
        printed via the diagnostics dict on samples.diagnostics.
        """
        rng = np.random.default_rng() if rng is None else rng

        # Step 1: empirical M from a moderate-sized Gaussian batch
        n_cal = max(20_000, 8 * n_samples)
        r_cal = rng.normal(0.0, self._coord_std, size=(n_cal, self.nstates))
        p_cal = rng.normal(0.0, self._coord_std, size=(n_cal, self.nstates))
        wabs_cal = np.abs(self.polynomial_factor(r_cal, p_cal))
        M = float(np.quantile(wabs_cal, cap_quantile))

        # Z_abs from the same calibration batch (mean of |w_poly|, then we
        # also need the truncated mean for the actual proposal).
        Z_abs_full      = float(np.mean(wabs_cal))
        truncated_mass  = float(np.mean(np.where(wabs_cal > M, wabs_cal - M, 0.0))) / Z_abs_full
        Z_abs_truncated = float(np.mean(np.minimum(wabs_cal, M)))

        # Step 2: rejection sampling.  Acceptance probability per draw is
        # min(|w_poly|, M)/M ≤ 1.  Effective acceptance ~ Z_abs_truncated/M.
        accept_rate_estimate = Z_abs_truncated / M
        # request batch of size n_samples / accept_rate_estimate * safety
        batch_factor = max(2.0, 1.5 / accept_rate_estimate)
        if batch_factor > max_oversample:
            raise RuntimeError(
                f"sample_signed_abs: estimated acceptance rate "
                f"{accept_rate_estimate:.3e} requires oversample factor "
                f"{batch_factor:.1f}, exceeding cap {max_oversample}.  "
                "Lower `cap_quantile` or increase `max_oversample`."
            )

        accepted_r:    list[FloatArray] = []
        accepted_p:    list[FloatArray] = []
        accepted_sign: list[FloatArray] = []
        accepted_qabs: list[FloatArray] = []
        accepted_qg:   list[FloatArray] = []
        n_drawn = 0
        n_kept  = 0

        while n_kept < n_samples and n_drawn < max_oversample * n_samples:
            batch = max(2 * n_samples, int(batch_factor * (n_samples - n_kept) + 16))
            r_b = rng.normal(0.0, self._coord_std, size=(batch, self.nstates))
            p_b = rng.normal(0.0, self._coord_std, size=(batch, self.nstates))
            w_b = self.polynomial_factor(r_b, p_b)
            wabs_b = np.abs(w_b)
            wabs_clip = np.minimum(wabs_b, M)
            u = rng.uniform(0.0, M, size=batch)
            accept = u < wabs_clip
            n_drawn += batch
            if not np.any(accept):
                continue
            r_a = r_b[accept]
            p_a = p_b[accept]
            sign_a = np.sign(w_b[accept]).astype(np.float64)
            # Gaussian density at the accepted points
            qg_a = self.gaussian_envelope_density(r_a, p_a)
            # New "proposal density" = q · |w_poly| / Z_abs_truncated  (sampling distribution)
            qabs_a = qg_a * wabs_clip[accept] / Z_abs_truncated
            accepted_r.append(r_a)
            accepted_p.append(p_a)
            accepted_sign.append(sign_a)
            accepted_qabs.append(qabs_a)
            accepted_qg.append(qg_a)
            n_kept += int(np.sum(accept))

        if n_kept < n_samples:
            raise RuntimeError(
                f"sample_signed_abs: drew {n_drawn} proposals, accepted only "
                f"{n_kept} of requested {n_samples}.  Try a different cap_quantile."
            )

        r    = np.concatenate(accepted_r)[:n_samples]
        p    = np.concatenate(accepted_p)[:n_samples]
        sign = np.concatenate(accepted_sign)[:n_samples]
        qabs = np.concatenate(accepted_qabs)[:n_samples]
        qg   = np.concatenate(accepted_qg)[:n_samples]

        # Recover the signed target ρ_0 = q · w_poly (this is the LABEL, unchanged).
        # And the IS weight ρ_0/q_sampling = sign · Z_abs_truncated.
        # (because ρ_0 = qg · w_poly = qg · sign · |w_poly| = sign · qg · |w_poly|
        #  and q_sampling = qg · |w_poly| / Z_abs ⇒ ρ_0 / q_sampling = sign · Z_abs.)
        w_poly = self.polynomial_factor(r, p)        # exact signed factor
        rho0   = qg * w_poly                          # ρ_0 at sample point

        return MappingSamples(
            r=r,
            p=p,
            proposal_density=qabs,                    # SAMPLING density |ρ_0|/Z_abs
            target_density=rho0,                      # ρ_0 (signed)
            weight=sign * Z_abs_truncated,            # ρ_0 / q_sampling
        )


class FocusedMappingSampler:
    """
    Focused MMST initial-condition sampler.

    This is a trajectory initializer (the cloud lives on a measure-zero
    submanifold of full mapping phase space — a product of circles, one
    per oscillator).  It is NOT a smooth Lebesgue density on full mapping
    phase space, but the EXACT SEO Wigner density evaluated at any focus
    point is well-defined and **strictly positive** (see "Why labels are
    strictly positive" below).

    Occupation convention:
        n_j = (r_j^2 + p_j^2) / (2 hbar) - gamma

    To focus onto initial electronic state lambda:
        n_lambda = 1,
        n_j = 0  for j != lambda

    hence
        r_j^2 + p_j^2 = 2 hbar (n_j + gamma),
    so the radii are
        active state:   sqrt(2 hbar (1 + gamma))
        inactive state: sqrt(2 hbar gamma)

    Why labels are strictly positive
    --------------------------------
    The full SEO Wigner density factorizes as a (positive) Gaussian
    envelope times a sign-changing polynomial factor:

        W_lambda(r,p) = q(r,p) * w_lambda(r,p)
        q(r,p)       = (pi hbar)^(-N_s) exp[-(r^2+p^2)/hbar]   > 0
        w_lambda     = 2(r_lambda^2 + p_lambda^2)/hbar - 1

    At the focused active radius, r_lambda^2 + p_lambda^2 = 2 hbar(1+gamma),
    so

        w_focus = 2 * 2 hbar(1+gamma) / hbar - 1 = 4(1+gamma) - 1 = 4 gamma + 3

    For gamma >= 0 this is at least 3 — strictly positive — and *constant*
    over the entire focus manifold (it depends only on the radius, not on
    the phase angles).  The Gaussian envelope is likewise constant at
    focus because each oscillator's radius is fixed.  Therefore the SEO
    target evaluated at any focus point equals a fixed positive constant
    K_focus that depends only on (nstates, hbar, gamma):

        K_focus = (pi hbar)^(-N_s) * exp[-2 (1 + N_s gamma)] * (4 gamma + 3).

    The full-phase-space target rho_0(z) = W_cl(R,P) * W_map(r,p)
    evaluated at focus is W_cl(R,P) * K_focus — a strictly positive
    function of the nuclear (R,P) only.  The IS-weight target/proposal
    is identically 1 (we are "perfectly importance-sampling" the SEO
    density on the focus manifold), so cloud Riemann sums reduce to the
    canonical focused estimator <A>(0) = (1/N) Sum_i A(z_i^0).

    Sampling-side outputs
    ---------------------
    To plug into the same Dynamics interface as the signed-SEO sampler,
    the focused sampler returns:
        proposal_density = K_focus * ones(N)   (mapping-only, positive)
        target_density   = K_focus * ones(N)   (mapping-only, positive)
        weight           = ones(N)             (target/proposal = 1)
    `MMSTSampler.sample_focused` multiplies the mapping-side densities
    by W_cl(R,P) so the GP labels become y_i = W_cl(R_i,P_i) * K_focus.
    """

    def __init__(self, params: MappingInitParams) -> None:
        self.nstates = int(params.nstates)
        self.init_state = int(params.init_state)
        self.hbar = float(params.hbar)
        self.gamma = float(params.gamma)

        if self.nstates <= 0:
            raise ValueError("nstates must be positive.")
        if not (0 <= self.init_state < self.nstates):
            raise ValueError("init_state must satisfy 0 <= init_state < nstates.")
        if self.hbar <= 0.0:
            raise ValueError("hbar must be positive.")
        if self.gamma < 0.0:
            raise ValueError("gamma must be nonnegative.")

        self._radii = np.sqrt(2.0 * self.hbar * self.gamma) * np.ones(self.nstates)
        self._radii[self.init_state] = np.sqrt(2.0 * self.hbar * (1.0 + self.gamma))

        # Pre-compute the constant K_focus = q_map_focus * w_focus.  This is
        # the value of the MAPPING-only SEO target at every focus point.
        # active radius² + (N_s - 1)·inactive radius²
        #   = 2 hbar (1 + gamma) + (N_s - 1)·2 hbar gamma
        #   = 2 hbar (1 + N_s gamma)
        # so q_map_focus = (pi hbar)^(-N_s) · exp[-2(1 + N_s gamma)]
        self._q_map_focus = (np.pi * self.hbar) ** (-self.nstates) * np.exp(
            -2.0 * (1.0 + self.nstates * self.gamma)
        )
        # w_focus = 2 (r_lambda^2 + p_lambda^2)/hbar - 1
        #         = 2 · 2 hbar (1 + gamma) / hbar - 1 = 4 gamma + 3 > 0
        self._w_focus = 4.0 * self.gamma + 3.0
        self._K_focus = float(self._q_map_focus * self._w_focus)

    @property
    def mapping_target_at_focus(self) -> float:
        """Constant value of the mapping-only SEO target on the focus manifold (>0)."""
        return self._K_focus

    def sample(
        self,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
        random_angles: bool = True,
    ) -> MappingSamples:
        rng = np.random.default_rng() if rng is None else rng

        if random_angles:
            theta = rng.uniform(0.0, 2.0 * np.pi, size=(n_samples, self.nstates))
        else:
            theta = np.zeros((n_samples, self.nstates), dtype=np.float64)

        r = self._radii[None, :] * np.cos(theta)
        p = self._radii[None, :] * np.sin(theta)

        # On the focus manifold, the SEO mapping target is the positive
        # constant K_focus and the IS weight target/proposal = 1.  We carry
        # all three fields so downstream code (GP labels, geometric measure,
        # signed_expectation) works without branching on sampler type.
        proposal = np.full(n_samples, self._K_focus, dtype=np.float64)
        target   = np.full(n_samples, self._K_focus, dtype=np.float64)
        weight   = np.ones(n_samples, dtype=np.float64)

        return MappingSamples(
            r=r,
            p=p,
            proposal_density=proposal,
            target_density=target,
            weight=weight,
        )

    def density(self, r: ArrayLike, p: ArrayLike) -> FloatArray:
        raise NotImplementedError(
            "Focused initial conditions are singular and should be treated as a sampler, "
            "not as a smooth density. If you want a continuous proposal, regularize them later."
        )


class MMSTInitialDensity:
    """
    Exact initial MMST density built from

        rho_0(R,P,r,p) = W_cl(R,P) * W_map(r,p),

    where
      - W_cl is the exact Wigner transform of the nuclear Gaussian wavepacket,
      - W_map is the exact SEO / projected mapping density for the chosen initial state.
    """

    def __init__(
        self,
        classical: ClassicalGaussianWigner,
        mapping: SEOMappingWigner,
    ) -> None:
        self.classical = classical
        self.mapping = mapping

    def density(
        self,
        R: ArrayLike,
        P: ArrayLike,
        r: ArrayLike,
        p: ArrayLike,
    ) -> FloatArray:
        rho_cl = self.classical.density(R, P)
        rho_map = self.mapping.density(r, p)

        if rho_cl.shape[0] != rho_map.shape[0]:
            raise ValueError(
                "Classical and mapping batches must have the same number of samples."
            )
        return rho_cl * rho_map

    projected_density = density

    def proposal_density_for_signed_seo_sampling(
        self,
        R: ArrayLike,
        P: ArrayLike,
        r: ArrayLike,
        p: ArrayLike,
    ) -> FloatArray:
        """
        Natural positive proposal associated with the exact SEO target:

            q_full(R,P,r,p) = W_cl(R,P) * q_map(r,p),

        where q_map is the positive Gaussian envelope of the SEO mapping density.
        """
        rho_cl = self.classical.density(R, P)
        q_map = self.mapping.gaussian_envelope_density(r, p)

        if rho_cl.shape[0] != q_map.shape[0]:
            raise ValueError(
                "Classical and mapping batches must have the same number of samples."
            )
        return rho_cl * q_map

    def importance_weight(
        self,
        r: ArrayLike,
        p: ArrayLike,
    ) -> FloatArray:
        """
        Since the classical part is sampled exactly and positively,
        the full importance weight reduces to the mapping polynomial factor:
            w_full = W_map / q_map = 2(r_lambda^2 + p_lambda^2)/hbar - 1.
        """
        return self.mapping.polynomial_factor(r, p)

    def sample_signed_seo(
        self,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> MMSTSamples:
        """
        Exact signed sampling of the full MMST initial density.

        Sampling strategy:
          - classical part: sample exactly from W_cl
          - mapping part: sample from the positive Gaussian envelope q_map
          - recover the target with the signed weight
        """
        rng = np.random.default_rng() if rng is None else rng

        R, P = self.classical.sample(n_samples, rng=rng)
        map_samples = self.mapping.sample_signed(n_samples, rng=rng)

        proposal = self.classical.density(R, P) * map_samples.proposal_density
        target = self.classical.density(R, P) * map_samples.target_density
        weight = map_samples.weight

        return MMSTSamples(
            R=R,
            P=P,
            r=map_samples.r,
            p=map_samples.p,
            proposal_density=proposal,
            target_density=target,
            weight=weight,
            label_information=LabelInformation.all_free(D=6, source="seo_signed"),
        )

    def sample_signed_seo_abs(
        self,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
        *,
        cap_quantile: float = 0.999,
    ) -> MMSTSamples:
        """
        Importance sampling from the absolute target |ρ_0| via the abs-target
        mapping sampler.  The classical packet is positive, so the absolute
        full density factorizes as

            |ρ_0(R,P,r,p)| = W_cl(R,P) · |W_map(r,p)|.

        The classical part is still sampled from W_cl exactly; the mapping
        part is sampled from |W_map| / Z_abs by ``sample_signed_abs``.

        See ``SEOMappingWigner.sample_signed_abs`` for the variance argument.
        """
        rng = np.random.default_rng() if rng is None else rng

        R, P = self.classical.sample(n_samples, rng=rng)
        map_samples = self.mapping.sample_signed_abs(
            n_samples, rng=rng, cap_quantile=cap_quantile
        )

        # Sampling density (the new "q") = W_cl · |W_map|/Z_abs
        proposal = self.classical.density(R, P) * map_samples.proposal_density
        # Target ρ_0 = W_cl · W_map (signed)
        target = self.classical.density(R, P) * map_samples.target_density
        # IS ratio ρ_0 / proposal = sign(w_poly) · Z_abs   (mapping factor only;
        # the classical factor cancels because both classical-side densities
        # are W_cl).
        weight = map_samples.weight

        return MMSTSamples(
            R=R,
            P=P,
            r=map_samples.r,
            p=map_samples.p,
            proposal_density=proposal,
            target_density=target,
            weight=weight,
            label_information=LabelInformation.all_free(D=6, source="seo_signed_abs"),
        )




@dataclass
class SamplingHealthReport:
    """
    Sampling-only health audit for one iid candidate cloud.

    The goal is to rank iid candidate clouds before any downstream fit or
    propagation.  All statistics are computed from the sampled cloud itself:

      * classical packet quality: means / variances of (R, P),
      * electronic quality: weighted diabatic populations and trace,
      * jackknife stability of the above,
      * signed-weight quality (ESS / cancellation ratio) for SEO sampling.
    """
    mode: str
    n_samples: int
    score: float
    target_populations: FloatArray
    population_estimate: FloatArray
    population_jackknife_se: FloatArray
    trace_estimate: float
    trace_jackknife_se: float
    R_mean_estimate: FloatArray
    R_mean_jackknife_se: FloatArray
    P_mean_estimate: FloatArray
    P_mean_jackknife_se: FloatArray
    R_var_estimate: FloatArray
    R_var_jackknife_se: FloatArray
    P_var_estimate: FloatArray
    P_var_jackknife_se: FloatArray
    signed_ess: float
    cancellation_ratio: float


@dataclass
class HealthySampleSelection:
    """Result of iid candidate generation + jackknife-based health selection.

    When `healthiest=False` (non-competitive sampling), this dataclass is
    still returned by the sampler when `return_selection=True`, with
    `report=None`, `candidate_reports=[]`, and `best_index=0`.  This
    unifies the return type so callers never need to branch on whether
    `raw` is an `MMSTSamples` or a `HealthySampleSelection`.
    """
    samples: MMSTSamples
    report: Optional["SamplingHealthReport"]   # None when healthiest=False
    candidate_reports: List["SamplingHealthReport"]
    best_index: int


def _weighted_mean_vec(values: ArrayLike, weights: Optional[ArrayLike] = None) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if weights is None:
        return np.mean(arr, axis=0)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if arr.shape[0] != w.shape[0]:
        raise ValueError("values and weights must have the same number of samples.")
    denom = float(np.sum(w))
    if abs(denom) <= 1.0e-15:
        raise ValueError("Sum of weights is too close to zero for a stable weighted mean.")
    return np.sum(arr * w[:, None], axis=0) / denom


def _weighted_second_central_moment(values: ArrayLike, weights: Optional[ArrayLike] = None) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    mu = _weighted_mean_vec(arr, weights)
    centered2 = (arr - mu[None, :]) ** 2
    return _weighted_mean_vec(centered2, weights)


def _sampling_effective_sample_size(weights: Optional[ArrayLike], n_samples: int) -> float:
    if weights is None:
        return float(n_samples)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    denom = float(np.sum(w * w))
    if denom <= 1.0e-30:
        return 0.0
    numer = float(np.sum(w)) ** 2
    return numer / denom


def _sampling_cancellation_ratio(weights: Optional[ArrayLike], n_samples: int) -> float:
    if weights is None:
        return 1.0
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    denom = float(np.sum(np.abs(w)))
    if denom <= 1.0e-30:
        return 0.0
    return abs(float(np.sum(w))) / denom


def _jackknife_blocks(n_samples: int, n_blocks: int) -> List[np.ndarray]:
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least 2.")
    if n_blocks > n_samples:
        raise ValueError("n_blocks cannot exceed n_samples.")
    indices = np.arange(n_samples, dtype=np.int64)
    return [np.asarray(b, dtype=np.int64) for b in np.array_split(indices, n_blocks)]


def _jackknife_se_from_replicates(replicates: List[FloatArray | float]) -> FloatArray | float:
    arr = np.asarray(replicates, dtype=np.float64)
    mean = np.mean(arr, axis=0)
    n_rep = arr.shape[0]
    se = np.sqrt((n_rep - 1.0) / n_rep * np.sum((arr - mean) ** 2, axis=0))
    if np.ndim(se) == 0:
        return float(se)
    return np.asarray(se, dtype=np.float64)


def _selection_score(
    *,
    target_populations: FloatArray,
    population_estimate: FloatArray,
    population_se: FloatArray,
    trace_estimate: float,
    trace_se: float,
    R_mean_estimate: FloatArray,
    R_mean_se: FloatArray,
    P_mean_estimate: FloatArray,
    P_mean_se: FloatArray,
    R_var_estimate: FloatArray,
    R_var_se: FloatArray,
    P_var_estimate: FloatArray,
    P_var_se: FloatArray,
    R0: FloatArray,
    P0: FloatArray,
    sigma_R: FloatArray,
    sigma_P: FloatArray,
    signed_ess: float,
    cancellation_ratio: float,
    n_samples: int,
) -> float:
    eps = 1.0e-14
    pop_res = np.mean((population_estimate - target_populations) ** 2)
    pop_se_pen = np.mean(population_se ** 2)
    trace_res = (trace_estimate - 1.0) ** 2
    trace_se_pen = trace_se ** 2

    R_mean_res = np.mean(((R_mean_estimate - R0) / (sigma_R + eps)) ** 2)
    P_mean_res = np.mean(((P_mean_estimate - P0) / (sigma_P + eps)) ** 2)
    R_var_res = np.mean(((R_var_estimate - sigma_R ** 2) / (sigma_R ** 2 + eps)) ** 2)
    P_var_res = np.mean(((P_var_estimate - sigma_P ** 2) / (sigma_P ** 2 + eps)) ** 2)

    R_mean_se_pen = np.mean((R_mean_se / (sigma_R + eps)) ** 2)
    P_mean_se_pen = np.mean((P_mean_se / (sigma_P + eps)) ** 2)
    R_var_se_pen = np.mean((R_var_se / (sigma_R ** 2 + eps)) ** 2)
    P_var_se_pen = np.mean((P_var_se / (sigma_P ** 2 + eps)) ** 2)

    ess_frac = min(1.0, max(0.0, signed_ess / float(max(1, n_samples))))
    ess_pen = (1.0 - ess_frac) ** 2
    cancel_pen = (1.0 - max(0.0, min(1.0, cancellation_ratio))) ** 2

    score = (
        4.0 * pop_res
        + 2.0 * trace_res
        + 1.0 * pop_se_pen
        + 0.5 * trace_se_pen
        + 1.0 * R_mean_res
        + 1.0 * P_mean_res
        + 0.75 * R_var_res
        + 0.75 * P_var_res
        + 0.25 * R_mean_se_pen
        + 0.25 * P_mean_se_pen
        + 0.25 * R_var_se_pen
        + 0.25 * P_var_se_pen
        + 1.0 * ess_pen
        + 0.5 * cancel_pen
    )
    return float(score)

class MMSTSampler:
    """
    Convenience façade that collects the classical and mapping samplers.

    In addition to the raw samplers, this class can generate multiple iid
    candidate clouds and rank them with a jackknife-based health criterion so
    downstream code can propagate the healthiest initial cloud independently of
    the GP fit / constraint choice.
    """

    def __init__(
        self,
        classical_params: GaussianWavePacketParams,
        mapping_params: MappingInitParams,
    ) -> None:
        self.classical = ClassicalGaussianWigner(classical_params)
        self.seo_mapping = SEOMappingWigner(mapping_params)
        self.focused_mapping = FocusedMappingSampler(mapping_params)
        self.initial_density = MMSTInitialDensity(self.classical, self.seo_mapping)
        self._classical_params = classical_params
        self._mapping_params = mapping_params

    def sample_seo_signed(
        self,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
        *,
        healthiest: bool = False,
        n_candidates: int = 8,
        jackknife_blocks: int = 8,
        return_selection: bool = False,
        abs_target: bool = False,
        abs_cap_quantile: float = 0.999,
    ) -> "MMSTSamples | HealthySampleSelection":
        """
        Sample from the signed SEO target.

        Parameters
        ----------
        abs_target : bool, default False.
            If True, sample from the ABSOLUTE target |ρ_0| via rejection on
            the polynomial factor (see SEOMappingWigner.sample_signed_abs).
            The geometric measure ω_i = 1/(N q(z_i^0)) is then bounded above
            by 1/(N · q_min · |w_min|) instead of growing as 1/q in the
            tails — bounding the per-step variance of cloud Riemann sums.
        abs_cap_quantile : float in (0,1), default 0.999.
            Truncation quantile passed to the abs sampler when
            abs_target=True.
        """
        if abs_target and healthiest:
            raise NotImplementedError(
                "abs_target=True is not yet supported with healthiest selection."
            )
        if abs_target:
            samples = self.initial_density.sample_signed_seo_abs(
                n_samples=n_samples, rng=rng, cap_quantile=abs_cap_quantile
            )
            if return_selection:
                return HealthySampleSelection(
                    samples=samples, report=None, candidate_reports=[], best_index=0
                )
            return samples
        if not healthiest:
            samples = self.initial_density.sample_signed_seo(n_samples=n_samples, rng=rng)
            if return_selection:
                return HealthySampleSelection(
                    samples=samples, report=None, candidate_reports=[], best_index=0
                )
            return samples
        selection = self.sample_healthiest(
            n_samples=n_samples,
            mode="seo_signed",
            rng=rng,
            n_candidates=n_candidates,
            jackknife_blocks=jackknife_blocks,
        )
        return selection if return_selection else selection.samples

    def sample_focused(
        self,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
        random_angles: bool = True,
        *,
        healthiest: bool = False,
        n_candidates: int = 8,
        jackknife_blocks: int = 8,
        return_selection: bool = False,
    ) -> "MMSTSamples | HealthySampleSelection":
        """
        Practical focused initializer with strictly positive labels:

          - classical part sampled exactly from the Gaussian Wigner packet
          - mapping part placed on the focused MMST circles
          - target / proposal labels carried as the EXACT SEO density value
            at each focus point, which is strictly positive:
                target_density_i = W_cl(R_i, P_i) * K_focus
            with K_focus = (pi hbar)^(-N_s) · exp[-2(1+N_s gamma)] · (4 gamma + 3) > 0.
          - IS weights are unity by construction (proposal = target on the
            focus manifold), so cloud Riemann sums reduce to the canonical
            focused estimator  <A>(0) = (1/N) Sum_i A(z_i^0).

        Note
        ----
        This is still an initializer rather than an exact signed sampling of
        the full SEO target on (R,P,r,p) — the focus manifold is measure-zero
        in mapping phase space — but the *labels* used by the GP surrogate
        are the analytic Lebesgue-defined SEO Wigner values at the focus
        points, which the GP smoothly interpolates between.
        """
        if not healthiest:
            rng = np.random.default_rng() if rng is None else rng
            R, P = self.classical.sample(n_samples, rng=rng)
            map_samples = self.focused_mapping.sample(
                n_samples=n_samples,
                rng=rng,
                random_angles=random_angles,
            )
            rho_cl = self.classical.density(R, P)
            # Focused mode: trajectories are sampled FROM the target itself,
            # so there is no separate proposal function — proposal ≡ target.
            # We carry both fields as the same data (and weight ≡ 1) rather
            # than None, because the IS normalisation
            #     ω_i = 1/(N · q(z_i^0)) = 1/(N · ρ_0(z_i^0))
            # is what makes ω·y(0) = 1/N at t=0 (the canonical focused
            # Monte Carlo weight).  Setting ``proposal_density = None`` and
            # ``ω = 1/N`` instead would produce ω·y ~ ρ_0/N which scales
            # the KDE / observables by an extra factor of ρ_0 ≈ K_focus·W_cl
            # ≈ 1e-3 — exactly the "scaled incorrectly" pathology the
            # Visualization module warns about in focused-mode runs.
            #
            # The "no separate proposal" semantics are honoured by storing
            # `proposal_density` as a literal copy of `target_density`:
            # anyone inspecting the snapshot sees they hold the same array.
            target   = rho_cl * map_samples.target_density
            samples = MMSTSamples(
                R=R, P=P, r=map_samples.r, p=map_samples.p,
                proposal_density=target.copy(),    # ≡ target_density in focused
                target_density=target,
                weight=np.ones(n_samples, dtype=np.float64),  # ρ/q = 1
                label_information=LabelInformation.focused_2state(
                    self._mapping_params, source="focused"),
            )
            if return_selection:
                return HealthySampleSelection(
                    samples=samples, report=None, candidate_reports=[], best_index=0
                )
            return samples
        selection = self.sample_healthiest(
            n_samples=n_samples,
            mode="focused",
            rng=rng,
            n_candidates=n_candidates,
            jackknife_blocks=jackknife_blocks,
            random_angles=random_angles,
        )
        return selection if return_selection else selection.samples

    def _draw_iid_candidate(
        self,
        *,
        mode: str,
        n_samples: int,
        rng: np.random.Generator,
        random_angles: bool = True,
    ) -> MMSTSamples:
        if mode == "seo_signed":
            return self.initial_density.sample_signed_seo(n_samples=n_samples, rng=rng)
        if mode == "focused":
            R, P = self.classical.sample(n_samples, rng=rng)
            map_samples = self.focused_mapping.sample(
                n_samples=n_samples, rng=rng, random_angles=random_angles
            )
            rho_cl = self.classical.density(R, P)
            # Focused mode: proposal ≡ target (trajectories sampled from
            # the target itself); see sample_focused() for the full
            # mathematical rationale of the redundant storage.
            target = rho_cl * map_samples.target_density
            return MMSTSamples(
                R=R, P=P, r=map_samples.r, p=map_samples.p,
                proposal_density=target.copy(),
                target_density=target,
                weight=np.ones(n_samples, dtype=np.float64),
                label_information=LabelInformation.focused_2state(
                    self._mapping_params, source="focused"),
            )
        raise ValueError("mode must be 'seo_signed' or 'focused'.")

    def _health_summary(
        self,
        samples: MMSTSamples,
        *,
        mode: str,
        idx: Optional[np.ndarray] = None,
    ) -> Dict[str, FloatArray | float]:
        if idx is None:
            idx = np.arange(samples.R.shape[0], dtype=np.int64)

        R = np.asarray(samples.R[idx], dtype=np.float64)
        P = np.asarray(samples.P[idx], dtype=np.float64)
        r = np.asarray(samples.r[idx], dtype=np.float64)
        p = np.asarray(samples.p[idx], dtype=np.float64)
        weights = None
        if mode == "seo_signed":
            if samples.weight is None:
                raise ValueError("seo_signed health audit requires signed weights.")
            weights = np.asarray(samples.weight[idx], dtype=np.float64)
        elif mode == "focused":
            weights = np.ones(idx.size, dtype=np.float64)
        else:
            raise ValueError("mode must be 'seo_signed' or 'focused'.")

        pops = _diabatic_population_estimators(r, p, hbar=self.seo_mapping.hbar)
        pop_est = _weighted_mean_vec(pops, weights)
        trace_est = float(np.sum(pop_est))

        R_mean = np.mean(R, axis=0)
        P_mean = np.mean(P, axis=0)
        R_var = np.var(R, axis=0)
        P_var = np.var(P, axis=0)

        target_pop = np.zeros(self.seo_mapping.nstates, dtype=np.float64)
        target_pop[self.seo_mapping.init_state] = 1.0

        return {
            "target_populations": target_pop,
            "population_estimate": np.asarray(pop_est, dtype=np.float64),
            "trace_estimate": trace_est,
            "R_mean_estimate": np.asarray(R_mean, dtype=np.float64),
            "P_mean_estimate": np.asarray(P_mean, dtype=np.float64),
            "R_var_estimate": np.asarray(R_var, dtype=np.float64),
            "P_var_estimate": np.asarray(P_var, dtype=np.float64),
            "signed_ess": _sampling_effective_sample_size(weights, idx.size),
            "cancellation_ratio": _sampling_cancellation_ratio(weights, idx.size),
        }

    def evaluate_sampling_health(
        self,
        samples: MMSTSamples,
        *,
        mode: str,
        jackknife_blocks: int = 8,
    ) -> SamplingHealthReport:
        n_samples = int(samples.R.shape[0])
        blocks = _jackknife_blocks(n_samples, min(jackknife_blocks, n_samples))
        full = self._health_summary(samples, mode=mode)

        loo_summaries = []
        all_idx = np.arange(n_samples, dtype=np.int64)
        for block in blocks:
            keep = np.ones(n_samples, dtype=bool)
            keep[block] = False
            idx = all_idx[keep]
            loo_summaries.append(self._health_summary(samples, mode=mode, idx=idx))

        pop_se = _jackknife_se_from_replicates([s["population_estimate"] for s in loo_summaries])
        trace_se = _jackknife_se_from_replicates([s["trace_estimate"] for s in loo_summaries])
        R_mean_se = _jackknife_se_from_replicates([s["R_mean_estimate"] for s in loo_summaries])
        P_mean_se = _jackknife_se_from_replicates([s["P_mean_estimate"] for s in loo_summaries])
        R_var_se = _jackknife_se_from_replicates([s["R_var_estimate"] for s in loo_summaries])
        P_var_se = _jackknife_se_from_replicates([s["P_var_estimate"] for s in loo_summaries])

        score = _selection_score(
            target_populations=np.asarray(full["target_populations"], dtype=np.float64),
            population_estimate=np.asarray(full["population_estimate"], dtype=np.float64),
            population_se=np.asarray(pop_se, dtype=np.float64),
            trace_estimate=float(full["trace_estimate"]),
            trace_se=float(trace_se),
            R_mean_estimate=np.asarray(full["R_mean_estimate"], dtype=np.float64),
            R_mean_se=np.asarray(R_mean_se, dtype=np.float64),
            P_mean_estimate=np.asarray(full["P_mean_estimate"], dtype=np.float64),
            P_mean_se=np.asarray(P_mean_se, dtype=np.float64),
            R_var_estimate=np.asarray(full["R_var_estimate"], dtype=np.float64),
            R_var_se=np.asarray(R_var_se, dtype=np.float64),
            P_var_estimate=np.asarray(full["P_var_estimate"], dtype=np.float64),
            P_var_se=np.asarray(P_var_se, dtype=np.float64),
            R0=self.classical.R0,
            P0=self.classical.P0,
            sigma_R=self.classical.sigma_R,
            sigma_P=self.classical.sigma_P,
            signed_ess=float(full["signed_ess"]),
            cancellation_ratio=float(full["cancellation_ratio"]),
            n_samples=n_samples,
        )

        return SamplingHealthReport(
            mode=mode,
            n_samples=n_samples,
            score=score,
            target_populations=np.asarray(full["target_populations"], dtype=np.float64),
            population_estimate=np.asarray(full["population_estimate"], dtype=np.float64),
            population_jackknife_se=np.asarray(pop_se, dtype=np.float64),
            trace_estimate=float(full["trace_estimate"]),
            trace_jackknife_se=float(trace_se),
            R_mean_estimate=np.asarray(full["R_mean_estimate"], dtype=np.float64),
            R_mean_jackknife_se=np.asarray(R_mean_se, dtype=np.float64),
            P_mean_estimate=np.asarray(full["P_mean_estimate"], dtype=np.float64),
            P_mean_jackknife_se=np.asarray(P_mean_se, dtype=np.float64),
            R_var_estimate=np.asarray(full["R_var_estimate"], dtype=np.float64),
            R_var_jackknife_se=np.asarray(R_var_se, dtype=np.float64),
            P_var_estimate=np.asarray(full["P_var_estimate"], dtype=np.float64),
            P_var_jackknife_se=np.asarray(P_var_se, dtype=np.float64),
            signed_ess=float(full["signed_ess"]),
            cancellation_ratio=float(full["cancellation_ratio"]),
        )

    def sample_healthiest(
        self,
        *,
        n_samples: int,
        mode: str = "seo_signed",
        rng: Optional[np.random.Generator] = None,
        n_candidates: int = 8,
        jackknife_blocks: int = 8,
        random_angles: bool = True,
    ) -> HealthySampleSelection:
        """
        Generate multiple iid candidate clouds and select the healthiest one.

        The selection is purely a property of the sampled cloud, independent of
        any downstream GP fit or constraint choice. For each candidate cloud m,
        we compute the weighted diabatic population estimates

            P_hat_alpha^(m) = [sum_i w_i c_{alpha alpha}(z_i)] / [sum_i w_i],

        with

            c_{alpha alpha}(r,p) = [r_alpha^2 + p_alpha^2 - hbar] / (2 hbar),

        and compare them to the exact initial-state targets

            P_lambda = 1,
            P_{alpha != lambda} = 0,
            sum_alpha P_alpha = 1.

        We also monitor the classical packet quality using the iid sample means
        and variances

            E[R_j] = R0_j,
            Var(R_j) = sigma_R_j^2,
            E[P_j] = P0_j,
            Var(P_j) = sigma_P_j^2 = hbar^2 / (4 sigma_R_j^2),

        and estimate the stability of these statistics with a delete-block
        jackknife. The returned score is smaller for healthier candidate clouds.
        """
        if n_candidates <= 0:
            raise ValueError("n_candidates must be positive.")
        rng = np.random.default_rng() if rng is None else rng

        candidate_samples: List[MMSTSamples] = []
        candidate_reports: List[SamplingHealthReport] = []

        for _ in range(n_candidates):
            cand = self._draw_iid_candidate(
                mode=mode, n_samples=n_samples, rng=rng, random_angles=random_angles
            )
            rep = self.evaluate_sampling_health(
                cand, mode=mode, jackknife_blocks=jackknife_blocks
            )
            candidate_samples.append(cand)
            candidate_reports.append(rep)

        best_index = int(np.argmin([rep.score for rep in candidate_reports]))
        return HealthySampleSelection(
            samples=candidate_samples[best_index],
            report=candidate_reports[best_index],
            candidate_reports=candidate_reports,
            best_index=best_index,
        )


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_classical_gaussian_wigner() -> None:
    print_header("TEST 1: Classical Gaussian Wigner")

    params = GaussianWavePacketParams(
        R0=[-8.0],
        P0=[30.0],
        sigma_R=[1.2],
        hbar=1.0,
    )
    classical = ClassicalGaussianWigner(params)

    rng = np.random.default_rng(1234)
    n = 200000
    R, P = classical.sample(n, rng=rng)

    print("ndim =", classical.ndim)
    print("R shape =", R.shape)
    print("P shape =", P.shape)

    print("\nExpected means:")
    print("R0 =", classical.R0)
    print("P0 =", classical.P0)

    print("\nSample means:")
    print("mean(R) =", np.mean(R, axis=0))
    print("mean(P) =", np.mean(P, axis=0))

    print("\nExpected variances:")
    print("Var(R) =", classical.sigma_R**2)
    print("Var(P) =", classical.sigma_P**2)

    print("\nSample variances:")
    print("Var(R) =", np.var(R, axis=0))
    print("Var(P) =", np.var(P, axis=0))

    dens = classical.density(R[:5], P[:5])
    logdens = classical.log_density(R[:5], P[:5])

    print("\nFirst 5 densities =", dens)
    print("First 5 log densities =", logdens)
    print("exp(log_density) agreement max error =",
          np.max(np.abs(np.exp(logdens) - dens)))


def test_seo_mapping_density_and_signed_sampling() -> None:
    print_header("TEST 2: SEO mapping density and signed sampling")

    params = MappingInitParams(
        nstates=2,
        init_state=0,
        hbar=1.0,
        gamma=0.5,
    )
    seo = SEOMappingWigner(params)

    rng = np.random.default_rng(1234)
    n = 200000
    samples = seo.sample_signed(n, rng=rng)

    r = samples.r
    p = samples.p
    q = samples.proposal_density
    rho = samples.target_density
    w = samples.weight

    print("nstates =", seo.nstates)
    print("init_state =", seo.init_state)
    print("r shape =", r.shape)
    print("p shape =", p.shape)

    print("\nEnvelope / target / weight consistency:")
    recon = q * w
    print("max |rho - q*w| =", np.max(np.abs(rho - recon)))

    print("\nStatistics of mapping variables:")
    print("mean(r) =", np.mean(r, axis=0))
    print("mean(p) =", np.mean(p, axis=0))
    print("var(r)  =", np.var(r, axis=0))
    print("var(p)  =", np.var(p, axis=0))

    active_radius2 = r[:, seo.init_state]**2 + p[:, seo.init_state]**2
    print("\nActive-state radius^2 statistics:")
    print("mean(active radius^2) =", np.mean(active_radius2))
    print("expected ~ hbar =", seo.hbar)

    print("\nSigned weight statistics:")
    print("min(w)  =", np.min(w))
    print("max(w)  =", np.max(w))
    print("mean(w) =", np.mean(w))
    print("fraction negative target density =",
          np.mean(rho < 0.0))

    print("\nFirst 5 q =", q[:5])
    print("First 5 w =", w[:5])
    print("First 5 rho =", rho[:5])


def test_focused_mapping_sampler() -> None:
    print_header("TEST 3: Focused mapping sampler")

    params = MappingInitParams(
        nstates=2,
        init_state=1,
        hbar=1.0,
        gamma=0.5,
    )
    focused = FocusedMappingSampler(params)

    rng = np.random.default_rng(1234)
    n = 10000
    samples = focused.sample(n, rng=rng, random_angles=True)

    r = samples.r
    p = samples.p

    radii2 = r**2 + p**2

    expected_radii2 = 2.0 * focused.hbar * focused.gamma * np.ones(focused.nstates)
    expected_radii2[focused.init_state] = 2.0 * focused.hbar * (1.0 + focused.gamma)

    print("nstates =", focused.nstates)
    print("init_state =", focused.init_state)
    print("gamma =", focused.gamma)

    print("\nExpected radii^2 per state =", expected_radii2)
    print("Sample mean radii^2 per state =", np.mean(radii2, axis=0))
    print("Sample std radii^2 per state =", np.std(radii2, axis=0))

    print("\nMax absolute deviation from expected radii^2:")
    print(np.max(np.abs(radii2 - expected_radii2[None, :])))

    # ------------------------------------------------------------------
    # Strictly-positive-density contract
    # ------------------------------------------------------------------
    # The focused sampler must populate target_density, proposal_density,
    # and weight so the Dynamics.build_initial_state path can consume it
    # directly.  Verify the analytic closed form:
    #     K_focus = (pi hbar)^(-N_s) · exp[-2(1+N_s gamma)] · (4 gamma + 3)
    K_expected = (np.pi * focused.hbar) ** (-focused.nstates) * np.exp(
        -2.0 * (1.0 + focused.nstates * focused.gamma)
    ) * (4.0 * focused.gamma + 3.0)

    print("\nStrictly-positive label contract:")
    print(f"  K_focus (analytic)  = {K_expected:.12e}")
    print(f"  K_focus (sampler)   = {focused.mapping_target_at_focus:.12e}")
    assert samples.target_density is not None, "target_density must be populated."
    assert samples.proposal_density is not None, "proposal_density must be populated."
    assert samples.weight is not None, "weight must be populated."
    print(f"  min(target_density) = {float(np.min(samples.target_density)):.12e}")
    print(f"  max(target_density) = {float(np.max(samples.target_density)):.12e}")
    print(f"  min(weight)         = {float(np.min(samples.weight)):.12e}")
    print(f"  max(weight)         = {float(np.max(samples.weight)):.12e}")
    if float(np.min(samples.target_density)) <= 0.0:
        raise AssertionError("Focused target_density must be strictly positive.")
    if not np.allclose(samples.target_density, samples.proposal_density):
        raise AssertionError("Focused proposal_density must equal target_density (weight=1 convention).")
    if not np.allclose(samples.weight, 1.0):
        raise AssertionError("Focused weights must be identically 1.")
    if abs(float(np.mean(samples.target_density)) - K_expected) > 1e-12:
        raise AssertionError("Focused target_density does not match analytic K_focus.")

    print("\nFirst 5 focused samples r:")
    print(r[:5])
    print("First 5 focused samples p:")
    print(p[:5])


def test_full_mmst_density_factorization() -> None:
    print_header("TEST 4: Full MMST density factorization")

    classical_params = GaussianWavePacketParams(
        R0=[-8.0],
        P0=[30.0],
        sigma_R=[1.0],
        hbar=1.0,
    )
    mapping_params = MappingInitParams(
        nstates=2,
        init_state=0,
        hbar=1.0,
        gamma=0.5,
    )

    classical = ClassicalGaussianWigner(classical_params)
    seo = SEOMappingWigner(mapping_params)
    mmst = MMSTInitialDensity(classical, seo)

    rng = np.random.default_rng(1234)
    n = 5000

    R, P = classical.sample(n, rng=rng)
    map_samples = seo.sample_signed(n, rng=rng)
    r, p = map_samples.r, map_samples.p

    rho_cl = classical.density(R, P)
    rho_map = seo.density(r, p)
    rho_full = mmst.density(R, P, r, p)

    print("rho_cl shape   =", rho_cl.shape)
    print("rho_map shape  =", rho_map.shape)
    print("rho_full shape =", rho_full.shape)

    print("\nFactorization check:")
    print("max |rho_full - rho_cl*rho_map| =",
          np.max(np.abs(rho_full - rho_cl * rho_map)))

    print("\nSign statistics:")
    print("fraction rho_map < 0 =", np.mean(rho_map < 0.0))
    print("fraction rho_full < 0 =", np.mean(rho_full < 0.0))

    print("\nFirst 5 rho_cl   =", rho_cl[:5])
    print("First 5 rho_map  =", rho_map[:5])
    print("First 5 rho_full =", rho_full[:5])


def test_full_signed_mmst_sampling() -> None:
    print_header("TEST 5: Full signed MMST sampling")

    classical_params = GaussianWavePacketParams(
        R0=[-15.0],
        P0=[30.0],
        sigma_R=[0.8],
        hbar=1.0,
    )
    mapping_params = MappingInitParams(
        nstates=2,
        init_state=0,
        hbar=1.0,
        gamma=0.5,
    )

    sampler = MMSTSampler(classical_params, mapping_params)

    rng = np.random.default_rng(1234)
    n = 100000
    samples = sampler.sample_seo_signed(n, rng=rng)

    R, P, r, p = samples.R, samples.P, samples.r, samples.p
    q = samples.proposal_density
    rho = samples.target_density
    w = samples.weight

    print("R shape =", R.shape)
    print("P shape =", P.shape)
    print("r shape =", r.shape)
    print("p shape =", p.shape)

    print("\nProposal / target / weight consistency:")
    print("max |rho - q*w| =", np.max(np.abs(rho - q * w)))

    print("\nClassical sample means:")
    print("mean(R) =", np.mean(R, axis=0))
    print("mean(P) =", np.mean(P, axis=0))

    print("\nSigned weight stats:")
    print("min(w)  =", np.min(w))
    print("max(w)  =", np.max(w))
    print("mean(w) =", np.mean(w))
    print("std(w)  =", np.std(w))

    print("\nDensity stats:")
    print("min(target_density) =", np.min(rho))
    print("max(target_density) =", np.max(rho))
    print("mean(target_density) =", np.mean(rho))
    print("fraction negative target_density =", np.mean(rho < 0.0))

    print("\nFirst 5 proposal densities =", q[:5])
    print("First 5 target densities   =", rho[:5])
    print("First 5 weights            =", w[:5])


def test_focused_vs_signed_seo_not_same_object() -> None:
    print_header("TEST 6: Focused sampler is not the same as exact SEO signed density")

    classical_params = GaussianWavePacketParams(
        R0=[-8.0],
        P0=[20.0],
        sigma_R=[1.0],
        hbar=1.0,
    )
    mapping_params = MappingInitParams(
        nstates=2,
        init_state=0,
        hbar=1.0,
        gamma=0.5,
    )

    sampler = MMSTSampler(classical_params, mapping_params)
    rng = np.random.default_rng(1234)

    focused = sampler.sample_focused(5, rng=rng, random_angles=True)
    signed = sampler.sample_seo_signed(5, rng=rng)

    print("Focused mapping samples (r, p):")
    for i in range(5):
        print(f"  sample {i}: r={focused.r[i]}, p={focused.p[i]}")

    print("\nSigned SEO mapping samples (r, p, weight):")
    for i in range(5):
        print(f"  sample {i}: r={signed.r[i]}, p={signed.p[i]}, w={signed.weight[i]}")

    print("\nConclusion:")
    print("Focused sampling gives constrained-radius initial conditions.")
    print("Signed SEO sampling gives Gaussian-envelope samples with signed weights.")
    print("These are different objects and should not be mixed.")



def _weighted_mean(values: ArrayLike, weights: Optional[ArrayLike] = None) -> float:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if weights is None:
        return float(np.mean(vals))
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if vals.shape[0] != w.shape[0]:
        raise ValueError("values and weights must have the same length.")
    denom = float(np.sum(w))
    if abs(denom) <= 1.0e-15:
        raise ValueError("Sum of weights is too close to zero for a stable estimator.")
    return float(np.sum(w * vals) / denom)


def _diabatic_population_estimators(
    r: ArrayLike,
    p: ArrayLike,
    hbar: float = 1.0,
) -> FloatArray:
    r_arr = np.asarray(r, dtype=np.float64)
    p_arr = np.asarray(p, dtype=np.float64)
    if r_arr.shape != p_arr.shape:
        raise ValueError("r and p must have the same shape.")
    if r_arr.ndim != 2:
        raise ValueError("r and p must have shape (n_samples, nstates).")
    if hbar <= 0.0:
        raise ValueError("hbar must be positive.")
    return 0.5 * (r_arr * r_arr + p_arr * p_arr - hbar) / hbar


def _diabatic_coherence_estimators(
    r: ArrayLike,
    p: ArrayLike,
    hbar: float = 1.0,
) -> FloatArray:
    """
    Per-trajectory MMST coherence estimators σ_λμ for λ < μ.

    With c_λ = (r_λ + i p_λ)/√(2ℏ), the standard MMST off-diagonal estimator
    is ρ_λμ = c_λ c_μ* (the convention shared by QCLE).  Component-wise:

        Re σ_λμ = (r_λ r_μ + p_λ p_μ) / (2ℏ)
        Im σ_λμ = (p_λ r_μ − r_λ p_μ) / (2ℏ)

    Returns an array of shape (n_samples, nstates, nstates, 2) where the
    last axis is [real, imag].  Diagonal entries are zero (use
    `_diabatic_population_estimators` for populations).
    """
    r_arr = np.asarray(r, dtype=np.float64)
    p_arr = np.asarray(p, dtype=np.float64)
    if r_arr.shape != p_arr.shape:
        raise ValueError("r and p must have the same shape.")
    if r_arr.ndim != 2:
        raise ValueError("r and p must have shape (n_samples, nstates).")
    if hbar <= 0.0:
        raise ValueError("hbar must be positive.")

    n_s, n_states = r_arr.shape
    out = np.zeros((n_s, n_states, n_states, 2), dtype=np.float64)
    inv2h = 1.0 / (2.0 * hbar)
    # Vectorize over (λ, μ) pairs with λ ≠ μ
    for lam in range(n_states):
        for mu in range(n_states):
            if lam == mu:
                continue
            out[:, lam, mu, 0] = (r_arr[:, lam] * r_arr[:, mu]
                                  + p_arr[:, lam] * p_arr[:, mu]) * inv2h
            out[:, lam, mu, 1] = (p_arr[:, lam] * r_arr[:, mu]
                                  - r_arr[:, lam] * p_arr[:, mu]) * inv2h
    return out


def pbme_density_matrix(
    r: ArrayLike,
    p: ArrayLike,
    weights: Optional[ArrayLike] = None,
    hbar: float = 1.0,
    *,
    normalize: bool = True,
    return_aux: bool = False,
) -> "FloatArray | tuple[FloatArray, dict]":
    """
    Canonical PBME density-matrix estimator from MMST mapping samples.

    Computes the full diabatic density matrix
        ρ_λμ = ⟨w · σ_λμ(r,p)⟩  /  N_norm
    using the importance-sampling weights `weights`, where σ_λλ are the
    standard SEO population estimators and σ_λμ (λ≠μ) are the MMST
    coherence estimators (see `_diabatic_population_estimators` and
    `_diabatic_coherence_estimators`).

    Normalization
    -------------
    PBME-MInt dynamics conserve the mapping radius along the flow, so
    Tr(ρ̂)(t) is a conserved quantity equal to its t=0 value.  In exact
    sampling Tr(ρ̂)(0) = 1 by construction; with finite N the SEO signed-
    weight estimator has variance from the polynomial factor 2|c|²/ℏ−1,
    so Tr(ρ̂)(0) deviates from 1 by O(1/√N) — purely sampling bias, not
    physics.  The trace deviation is then *frozen* into all subsequent
    times because PBME conserves it.

    With `normalize=True` (default) we divide by the *observed* trace
    Tr_obs = Σ_λ ⟨w·σ_λλ⟩, so the returned density matrix satisfies
    Tr(ρ̂) = 1 to machine precision regardless of N.  This is the right
    thing to do for comparing PBME to SE/QCLE (which trivially have
    Tr=1) and for interpreting populations as probabilities.

    With `normalize=False` the raw self-normalized IS estimator is
    returned, useful as a *diagnostic* of sampling quality.

    Choice of N_norm
    ----------------
    `normalize=True`     → divide by Tr_obs (trace-conserving, the right
                            choice for PBME observables in physics units).
    `normalize=False`    → divide by Σ w (raw self-normalized IS, exposes
                            sampling bias on the trace).
    `normalize="N"`      → divide by N (proper unbiased IS estimator,
                            asymptotically equivalent but higher variance
                            at finite N; rarely useful in practice).

    Parameters
    ----------
    r, p : (n_samples, n_states) arrays of mapping coordinates
    weights : (n_samples,) array of signed IS weights; if None, uniform
    hbar : ℏ in atomic units
    normalize : bool or 'N' — see above
    return_aux : if True also return diagnostics dict with raw trace,
                 effective sample size, cancellation ratio.

    Returns
    -------
    rho : complex array of shape (n_states, n_states), Hermitian
    aux (only if return_aux): dict with
        'trace_raw'  : Tr(ρ̂) before any normalization (raw IS)
        'sum_w'      : Σ w_i
        'ess'        : effective sample size
        'cancel'     : (Σ w)² / (Σ |w|)²  (1.0 = no cancellation)
    """
    r_arr = np.asarray(r, dtype=np.float64)
    p_arr = np.asarray(p, dtype=np.float64)
    if r_arr.ndim != 2 or r_arr.shape != p_arr.shape:
        raise ValueError("r and p must have shape (n_samples, n_states).")
    n_s, n_states = r_arr.shape

    if weights is None:
        w = np.ones(n_s, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != n_s:
            raise ValueError("weights must have shape (n_samples,).")

    sum_w = float(np.sum(w))
    if abs(sum_w) <= 1.0e-15:
        raise ValueError(
            "Sum of weights is near zero — sampling cloud is degenerate "
            "(possibly all-cancelling signed weights at small N)."
        )

    # Per-trajectory diagonal estimators σ_λλ
    pops = _diabatic_population_estimators(r_arr, p_arr, hbar=hbar)
    # Per-trajectory off-diagonal estimators σ_λμ
    cohs = _diabatic_coherence_estimators(r_arr, p_arr, hbar=hbar)

    # Self-normalized IS aggregator: ⟨wO⟩/⟨w⟩
    rho = np.zeros((n_states, n_states), dtype=np.complex128)
    for lam in range(n_states):
        rho[lam, lam] = float(np.sum(w * pops[:, lam])) / sum_w
        for mu in range(lam + 1, n_states):
            re = float(np.sum(w * cohs[:, lam, mu, 0])) / sum_w
            im = float(np.sum(w * cohs[:, lam, mu, 1])) / sum_w
            rho[lam, mu] = re + 1j * im
            rho[mu, lam] = re - 1j * im   # Hermitian

    trace_raw = float(np.real(np.trace(rho)))

    if normalize is True:
        # Trace-conserving normalization (right answer for physics)
        if abs(trace_raw) <= 1.0e-15:
            raise RuntimeError(
                f"Observed PBME trace ≈ 0 ({trace_raw:.2e}); cannot normalize. "
                f"Increase n_samples or check sampler."
            )
        rho = rho / trace_raw
    elif normalize is False:
        pass    # raw self-normalized IS
    elif normalize == "N":
        # Proper unbiased IS estimator: divide by N instead of Σw.
        # Equivalent under exact sampling; exposes a different bias flavour.
        rho = rho * (sum_w / n_s)
    else:
        raise ValueError(f"normalize must be True, False, or 'N'; got {normalize!r}")

    if return_aux:
        ess    = float(sum_w**2 / max(np.sum(w * w), 1e-30))
        sum_aw = float(np.sum(np.abs(w)))
        cancel = float(sum_w**2 / max(sum_aw**2, 1e-30))
        return rho, dict(
            trace_raw=trace_raw,
            sum_w=sum_w,
            ess=ess,
            cancel=cancel,
            n_samples=n_s,
        )
    return rho


def pbme_diabatic_populations(
    r: ArrayLike,
    p: ArrayLike,
    weights: Optional[ArrayLike] = None,
    hbar: float = 1.0,
    *,
    normalize: bool = True,
) -> FloatArray:
    """
    Convenience wrapper: trace-normalized diabatic populations only.
    Equivalent to `np.real(np.diag(pbme_density_matrix(...)))`.
    """
    rho = pbme_density_matrix(r, p, weights=weights, hbar=hbar, normalize=normalize)
    return np.real(np.diag(rho)).astype(np.float64)


def test_short_pbme_mint_dynamics_on_samples(
    sampling_mode: str = "seo_signed",
    n_samples: int = 4000,
    dt: float = 0.25,
    n_steps: int = 1000,
    seed: int = 1234,
) -> None:
    """
    Short-time PBME/MInt diagnostic using the samples produced by this module.

    What this test checks
    ---------------------
    1. Sample initial conditions using either:
         - the exact signed SEO/MMST initial density (`sampling_mode="seo_signed"`), or
         - focused MMST initial conditions (`sampling_mode="focused"`).
    2. Pack those samples into z = (R, P, r0, r1, p0, p1).
    3. Propagate the full cloud with PBME-MInt for a short time.
    4. Monitor:
         - weighted ensemble energy,
         - weighted diabatic populations P_0, P_1,
         - weighted trace P_0 + P_1,
         - maximum single-trajectory energy drift.

    For `seo_signed`, fixed signed weights from the initial proposal cloud are used:
        <A(t)> = [sum_i w_i A(z_i(t))] / [sum_i w_i].
    For `focused`, all weights are 1.

    Notes
    -----
    - The PBME-MInt map should preserve the Hamiltonian to high accuracy.
    - The mapping radius is preserved by the frozen-R rotation, so the trace
      estimator is expected to remain constant up to numerical error.
    - For focused initial conditions with gamma = 1/2 and two states, the
      initial diabatic populations are exactly (1, 0) or (0, 1) trajectory-wise.
    """
    from .Mint import PBMEMIntDynamics, PBMEMIntParams, pack_z
    from .Models import TullyModel, TullyParams

    rng = np.random.default_rng(seed)

    classical_params = GaussianWavePacketParams(
        R0=[-15.0],
        P0=[30.0],
        sigma_R=[1.0],
        hbar=1.0,
    )
    mapping_params = MappingInitParams(
        nstates=2,
        init_state=0,
        hbar=1.0,
        gamma=0.5,
    )
    sampler = MMSTSampler(classical_params, mapping_params)

    if sampling_mode == "seo_signed":
        samples = sampler.sample_seo_signed(n_samples, rng=rng)
        weights = np.asarray(samples.weight, dtype=np.float64)
    elif sampling_mode == "focused":
        samples = sampler.sample_focused(n_samples, rng=rng, random_angles=True)
        weights = np.ones(n_samples, dtype=np.float64)
    else:
        raise ValueError("sampling_mode must be 'seo_signed' or 'focused'.")

    z0 = pack_z(samples.R, samples.P, samples.r, samples.p)

    model = TullyModel(TullyParams.defaults("dual"))
    dyn = PBMEMIntDynamics(model=model, params=PBMEMIntParams(mass=2000.0, hbar=1.0))
    traj = dyn.propagate(z0, dt=dt, n_steps=n_steps)

    energy_hist = []
    pop0_hist = []
    pop1_hist = []
    trace_hist = []
    max_single_energy_drift_hist = []

    E0 = np.asarray(dyn.energy(traj[0]), dtype=np.float64).reshape(-1)

    for k in range(n_steps + 1):
        zk = np.asarray(traj[k], dtype=np.float64)
        Ek = np.asarray(dyn.energy(zk), dtype=np.float64).reshape(-1)

        r = zk[:, 2:4]
        p = zk[:, 4:6]
        pops = _diabatic_population_estimators(r, p, hbar=1.0)

        P0 = _weighted_mean(pops[:, 0], weights)
        P1 = _weighted_mean(pops[:, 1], weights)
        tr = P0 + P1
        Ebar = _weighted_mean(Ek, weights)

        energy_hist.append(Ebar)
        pop0_hist.append(P0)
        pop1_hist.append(P1)
        trace_hist.append(tr)
        max_single_energy_drift_hist.append(float(np.max(np.abs(Ek - E0))))

    energy_hist = np.asarray(energy_hist, dtype=np.float64)
    pop0_hist = np.asarray(pop0_hist, dtype=np.float64)
    pop1_hist = np.asarray(pop1_hist, dtype=np.float64)
    trace_hist = np.asarray(trace_hist, dtype=np.float64)
    max_single_energy_drift_hist = np.asarray(max_single_energy_drift_hist, dtype=np.float64)

    dE_mean = energy_hist - energy_hist[0]
    dtrace = trace_hist - trace_hist[0]

    print_header(f"TEST 7: Short-time PBME/MInt dynamics on {sampling_mode} samples")
    print(f"n_samples = {n_samples}, dt = {dt}, n_steps = {n_steps}, seed = {seed}")
    print(f"sampling_mode = {sampling_mode!r}")
    print(f"sum(weights) = {np.sum(weights):.12e}")
    print()
    print("step,t,E_mean,dE_mean,P0,P1,trace,max_single_traj_|ΔE|")
    for k in range(n_steps + 1):
        print(
            f"{k},{k*dt:.6f},"
            f"{energy_hist[k]:.12e},{dE_mean[k]:.12e},"
            f"{pop0_hist[k]:.12e},{pop1_hist[k]:.12e},{trace_hist[k]:.12e},"
            f"{max_single_energy_drift_hist[k]:.12e}"
        )

    print("\nSummary:")
    print(f"  initial weighted energy = {energy_hist[0]:.12e}")
    print(f"  final   weighted energy = {energy_hist[-1]:.12e}")
    print(f"  max |Δ weighted energy| = {np.max(np.abs(dE_mean)):.3e}")
    print(f"  initial populations     = ({pop0_hist[0]:.12e}, {pop1_hist[0]:.12e})")
    print(f"  final   populations     = ({pop0_hist[-1]:.12e}, {pop1_hist[-1]:.12e})")
    print(f"  initial trace           = {trace_hist[0]:.12e}")
    print(f"  final   trace           = {trace_hist[-1]:.12e}")
    print(f"  max |Δ trace|           = {np.max(np.abs(dtrace)):.3e}")
    print(f"  max single-trajectory |ΔE| over window = {np.max(max_single_energy_drift_hist):.3e}")

    # Mild scientific sanity checks:
    #   - energy conservation should be excellent for PBME-MInt,
    #   - trace should stay constant under the PBME mapping rotation.
    if np.max(np.abs(dE_mean)) > 1.0e-9:
        raise AssertionError(
            "Weighted ensemble energy drift is larger than expected for a short PBME-MInt run."
        )
    if np.max(np.abs(dtrace)) > 1.0e-10:
        raise AssertionError(
            "Weighted trace drift is larger than expected for a short PBME-MInt run."
        )

    if sampling_mode == "seo_signed":
        init_target = np.array([1.0, 0.0], dtype=np.float64)
        init_pop = np.array([pop0_hist[0], pop1_hist[0]], dtype=np.float64)
        if np.max(np.abs(init_pop - init_target)) > 7.5e-2:
            raise AssertionError(
                "Initial signed-SEO weighted populations are too far from the target state."
            )
        if abs(trace_hist[0] - 1.0) > 7.5e-2:
            raise AssertionError(
                "Initial signed-SEO weighted trace is too far from 1."
            )

    if sampling_mode == "focused":
        init_target = np.array([1.0, 0.0], dtype=np.float64)
        init_pop = np.array([pop0_hist[0], pop1_hist[0]], dtype=np.float64)
        if np.max(np.abs(init_pop - init_target)) > 1.0e-12:
            raise AssertionError(
                "Focused initial populations are not exactly the target occupations."
            )
        if abs(trace_hist[0] - 1.0) > 1.0e-12:
            raise AssertionError(
                "Focused initial trace is not exactly 1."
            )


def test_healthiest_sampling_selection() -> None:
    print_header("TEST 8: Healthiest iid sample selection")

    classical_params = GaussianWavePacketParams(
        R0=[-15.0],
        P0=[30.0],
        sigma_R=[1.0],
        hbar=1.0,
    )
    mapping_params = MappingInitParams(
        nstates=2,
        init_state=0,
        hbar=1.0,
        gamma=0.5,
    )
    sampler = MMSTSampler(classical_params, mapping_params)
    rng = np.random.default_rng(2025)

    selection = sampler.sample_healthiest(
        n_samples=2000,
        mode="seo_signed",
        rng=rng,
        n_candidates=6,
        jackknife_blocks=8,
    )

    print(f"best candidate index = {selection.best_index}")
    for i, rep in enumerate(selection.candidate_reports):
        print(
            f"  cand {i}: score={rep.score:.6e}  "
            f"ESS={rep.signed_ess:.3f}  cancel={rep.cancellation_ratio:.3e}  "
            f"P={rep.population_estimate}  trace={rep.trace_estimate:.6e}"
        )

    best = selection.report
    print("\nChosen candidate summary:")
    print("  populations          =", best.population_estimate)
    print("  jackknife pop se     =", best.population_jackknife_se)
    print("  trace                =", best.trace_estimate)
    print("  trace jackknife se   =", best.trace_jackknife_se)
    print("  R_mean               =", best.R_mean_estimate)
    print("  P_mean               =", best.P_mean_estimate)
    print("  R_var                =", best.R_var_estimate)
    print("  P_var                =", best.P_var_estimate)
    print("  signed ESS           =", best.signed_ess)
    print("  cancellation ratio   =", best.cancellation_ratio)



if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    test_classical_gaussian_wigner()
    test_seo_mapping_density_and_signed_sampling()
    test_focused_mapping_sampler()
    test_full_mmst_density_factorization()
    test_full_signed_mmst_sampling()
    test_focused_vs_signed_seo_not_same_object()
    test_short_pbme_mint_dynamics_on_samples(sampling_mode="seo_signed")
    test_short_pbme_mint_dynamics_on_samples(sampling_mode="focused")
    test_healthiest_sampling_selection()