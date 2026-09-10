"""
Path-2 white-dwarf SED model (``--model wd``).

Runs four nested-sampling fits in sequence:

    DA × MIST,  DA × PARSEC,  DB × MIST,  DB × PARSEC

Free parameters (all uniform priors):
    Teff_wd  (K)        effective temperature
    logg_wd  (dex)      surface gravity
    Av       (mag)      F99 R_V=3.1 visual extinction
    parallax (mas)      parallax > 0 ↔ distance = 1000/parallax pc

Derived quantities (not free parameters):
    M_WD    interpolated from Bergeron grid at (Teff, logg)
    M_i     inverted from M_WD via Cummings IFMR
    t_cool  WD cooling age from the grid

Age gate (optional):
    When *system_age_yr* is given (> 0), samples where the IFMR-implied
    progenitor did **not** finish the MS before *system_age_yr* are assigned
    log-likelihood ``-inf`` (hard prior cut, not a soft penalty).

Likelihood:
    Gaussian on observed − predicted magnitudes for detection rows:
        ln L = −0.5 Σ [(m_obs − m_pred)² / σ²  + ln(2π σ²)]
    Upper-limit rows (flag = 1): ignored in this release (conservative).

Summary JSON keys per combination:
    atm_type, ifmr, logevidence, m_wd_median, m_wd_lo, m_wd_hi,
    m_i_median, extrap_mass_frac, extrap_mass (bool flag), teff_median,
    logg_median, av_median, parallax_median.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from darkhunter_sed.bergeron_wd import AtmType, BergeronGrid
from darkhunter_sed.cummings_ifmr import CummingsIFMR, IFMRVariant
from darkhunter_sed.phot_sed_io import FLAG_DETECTION, PhotRow

# Parameter order (locked for this module).
_PARAM_NAMES: tuple[str, ...] = ("teff_wd", "logg_wd", "av", "parallax_mas")
# Extra parameters appended when --ir-bb is active.
_BB_PARAM_NAMES: tuple[str, ...] = ("log10_T_bb", "log10_L_bb")
_AB_ZEROPOINT_ERG_HZ: float = 3.631e-20   # AB zero-point F_nu in erg/s/cm²/Hz
_C_AA_PER_S: float = 2.99792458e18        # speed of light in Å/s


# ---------------------------------------------------------------------------
# Prior bounds dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class WDPriorBounds:
    """
    Uniform prior hypercube for WD free parameters.

    Parameters
    ----------
    teff_wd :
        (lo, hi) in K.  Defaults allow the full Bergeron grid.
    logg_wd :
        (lo, hi) in dex.  Grid range is 7.0–9.0.
    av :
        (lo, hi) visual extinction in mag.
    parallax_mas :
        (lo, hi) parallax in mas (both > 0).
    """

    teff_wd: tuple[float, float] = (1500.0, 150_000.0)
    logg_wd: tuple[float, float] = (7.0, 9.0)
    av: tuple[float, float] = (0.0, 5.0)
    parallax_mas: tuple[float, float] = (0.1, 100.0)

    def as_list(self) -> list[tuple[float, float]]:
        return [self.teff_wd, self.logg_wd, self.av, self.parallax_mas]

    def prior_transform(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map unit-hypercube sample *u* → physical parameters."""
        bounds = self.as_list()
        result = np.empty(len(bounds))
        for i, (lo, hi) in enumerate(bounds):
            result[i] = lo + u[i] * (hi - lo)
        return result


@dataclass(frozen=True, slots=True)
class BBPriorBounds:
    """
    Uniform prior bounds for the optional IR blackbody component.

    Parameters
    ----------
    log10_t_bb :
        ``(lo, hi)`` for log₁₀(T_bb / K).  Default covers 100 K – 31 623 K.
    log10_l_bb :
        ``(lo, hi)`` for log₁₀(L_bb / L☉).  Default covers 10⁻⁶ – 10⁴ L☉.
    """

    log10_t_bb: tuple[float, float] = (2.0, 4.5)
    log10_l_bb: tuple[float, float] = (-6.0, 4.0)

    def as_list(self) -> list[tuple[float, float]]:
        """Return bounds in ``_BB_PARAM_NAMES`` order."""
        return [self.log10_t_bb, self.log10_l_bb]


# ---------------------------------------------------------------------------
# WDParams dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class WDParams:
    """
    One set of WD free parameters.

    Parameters
    ----------
    teff_wd :
        Effective temperature (K).
    logg_wd :
        Surface gravity log g (dex).
    av :
        Visual extinction (mag); F99 R_V=3.1.
    parallax_mas :
        Parallax (mas); must be > 0.
    """

    teff_wd: float
    logg_wd: float
    av: float
    parallax_mas: float

    @property
    def distance_pc(self) -> float:
        return 1000.0 / self.parallax_mas


# ---------------------------------------------------------------------------
# WDFitResult dataclass
# ---------------------------------------------------------------------------
@dataclass
class WDFitResult:
    """
    Nested-sampling result for one (atm_type, ifmr) combination.

    Parameters
    ----------
    atm_type :
        ``"DA"`` or ``"DB"``.
    ifmr :
        ``"MIST"`` or ``"PARSEC"``.
    logevidence :
        Dynesty log-evidence (lnZ).
    samples :
        Shape ``(n_samples, 4)``; columns in ``_PARAM_NAMES`` order.
    weights :
        Posterior weights (importance-weighted); shape ``(n_samples,)``.
    m_wd_samples :
        WD mass (M_sun) for each sample.
    m_i_samples :
        Progenitor initial mass (M_sun) for each sample; nan for out-of-IFMR.
    m_i_unc_ifmr_samples :
        1-σ IFMR systematic uncertainty on Mi for each sample (M_sun).
        Propagated from Cummings+2018 coefficient uncertainties:
        σ_Mi = (1/slope) √[(σ_slope × Mi)² + σ_intercept²].
        nan where Mi is outside the IFMR range.
    extrap_mask :
        Boolean array; True where M_WD > 1.3 M_sun or logg > 9.0.
    t_cool_yr_samples :
        WD cooling age in years for each posterior sample, from the Bergeron
        grid interpolator.  Used by the diagnostics corner plot.
    """

    atm_type: AtmType
    ifmr: IFMRVariant
    logevidence: float
    samples: NDArray[np.float64]
    weights: NDArray[np.float64]
    m_wd_samples: NDArray[np.float64]
    m_i_samples: NDArray[np.float64]
    m_i_unc_ifmr_samples: NDArray[np.float64]
    extrap_mask: NDArray[np.bool_]
    t_cool_yr_samples: NDArray[np.float64]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary dict for this combination."""
        w = self.weights / self.weights.sum()
        m_wd = self.m_wd_samples

        def _quantile(arr: NDArray[np.float64], q: float) -> float:
            idx = np.argsort(arr)
            cdf = np.cumsum(w[idx])
            return float(arr[idx[np.searchsorted(cdf, q)]])

        m_wd_med = _quantile(m_wd, 0.50)
        m_wd_lo  = _quantile(m_wd, 0.16)
        m_wd_hi  = _quantile(m_wd, 0.84)

        mi_arr = self.m_i_samples
        unc_arr = self.m_i_unc_ifmr_samples
        valid = ~np.isnan(mi_arr)
        m_i_med = float(np.median(mi_arr[valid])) if valid.any() else float("nan")

        # Weighted-median IFMR systematic on Mi across valid samples.
        valid_unc = valid & ~np.isnan(unc_arr)
        m_i_ifmr_unc_med = (
            float(np.median(unc_arr[valid_unc])) if valid_unc.any() else float("nan")
        )

        extrap_frac = float(np.sum(self.extrap_mask * w))

        def _par_median(col: int) -> float:
            return _quantile(self.samples[:, col], 0.50)

        return {
            "atm_type": self.atm_type,
            "ifmr": self.ifmr,
            "logevidence": float(self.logevidence),
            "m_wd_median": m_wd_med,
            "m_wd_lo": m_wd_lo,
            "m_wd_hi": m_wd_hi,
            "m_i_median": m_i_med,
            "m_i_ifmr_unc_median": m_i_ifmr_unc_med,
            "extrap_mass_frac": extrap_frac,
            "extrap_mass": bool(extrap_frac > 0.01),
            "teff_median": _par_median(0),
            "logg_median": _par_median(1),
            "av_median": _par_median(2),
            "parallax_median": _par_median(3),
        }


# ---------------------------------------------------------------------------
# Likelihood builder
# ---------------------------------------------------------------------------
def _bb_apparent_mag_ab(
    eff_wave_aa: float,
    t_bb_k: float,
    l_bb_lsun: float,
    distance_pc: float,
) -> float:
    """
    AB apparent magnitude of a blackbody at a single effective wavelength.

    Parameters
    ----------
    eff_wave_aa :
        Filter effective wavelength (Å).
    t_bb_k :
        Blackbody temperature (K; ``> 0``).
    l_bb_lsun :
        Bolometric luminosity (L☉; ``> 0``).
    distance_pc :
        Heliocentric distance (pc; ``> 0``).

    Returns
    -------
    float
        AB magnitude.  Returns ``99.0`` when flux is zero or non-finite.

    Limits
    ------
    Uses the AB zero-point F_ν₀ = 3.631 × 10⁻²⁰ erg/s/cm²/Hz.
    Assumes Bergeron magnitudes are in the AB system.
    """
    from darkhunter_sed.phoenix_grid import bb_flux_flam_at_earth

    wave = np.array([float(eff_wave_aa)], dtype=np.float64)
    f_lam = bb_flux_flam_at_earth(wave, t_bb_k, l_bb_lsun, distance_pc)[0]
    if not np.isfinite(f_lam) or f_lam <= 0.0:
        return 99.0
    f_nu = f_lam * eff_wave_aa**2 / _C_AA_PER_S
    if not np.isfinite(f_nu) or f_nu <= 0.0:
        return 99.0
    return float(-2.5 * math.log10(f_nu / _AB_ZEROPOINT_ERG_HZ))


def _make_loglike(
    grid: BergeronGrid,
    ifmr: CummingsIFMR,
    obs_rows: list[PhotRow],
    bounds: WDPriorBounds,
    system_age_yr: float,
    *,
    bb_bounds: BBPriorBounds | None = None,
) -> tuple[Any, Any]:
    """
    Build (log_likelihood, prior_transform) callables for dynesty.

    Parameters
    ----------
    grid :
        Bergeron atmosphere grid (DA or DB).
    ifmr :
        Cummings IFMR instance.
    obs_rows :
        Detection-only PhotRow list (flag == 0).
    bounds :
        Prior bounds for the 4 WD parameters.
    system_age_yr :
        System age in years; 0.0 → age gate disabled.
    bb_bounds :
        When provided, append ``log10_T_bb`` and ``log10_L_bb`` as free
        parameters (indices 4 and 5).  The BB FLAM is added in AB-mag flux
        space at each band's effective wavelength before extinction.

    Returns
    -------
    (log_likelihood, prior_transform)
        Both accept / return 1-D numpy arrays of length 4 (or 6 with BB).
    """
    # Extract Gaia parallax measurement before filtering to Bergeron bands.
    # Without this, parallax is a free parameter with no data constraint and
    # the sampler pushes it to the prior ceiling (100 mas = 10 pc), giving a
    # completely degenerate (and wrong) posterior.
    _PLX_BAND = "Gaia_parallax"
    _gaia_plx_rows = [r for r in obs_rows if r.band == _PLX_BAND]
    _gaia_plx: float | None = float(_gaia_plx_rows[0].mag) if _gaia_plx_rows else None
    _gaia_plx_err: float | None = float(_gaia_plx_rows[0].err) if _gaia_plx_rows else None
    if _gaia_plx is None:
        print("  [wd] WARNING: no Gaia_parallax row found; parallax will be unconstrained.")

    obs_bands = [r.band for r in obs_rows]
    obs_mags  = np.array([r.mag for r in obs_rows], dtype=np.float64)
    obs_errs  = np.array([r.err for r in obs_rows], dtype=np.float64)

    # Pre-filter to only bands actually in the Bergeron grid.
    avail = set(grid.available_bands)
    keep = [i for i, b in enumerate(obs_bands) if b in avail]
    if not keep:
        raise ValueError(
            "No observed bands overlap with the Bergeron grid.  "
            f"Available: {sorted(avail)}.  Observed: {obs_bands}"
        )
    band_subset = [obs_bands[i] for i in keep]
    mag_vec  = obs_mags[keep]
    err_vec  = obs_errs[keep]
    inv_var  = 1.0 / (err_vec ** 2)
    norm_sum = float(np.sum(np.log(2.0 * math.pi * err_vec ** 2)))

    use_age_gate = system_age_yr > 0.0
    use_bb = bb_bounds is not None
    eff_waves: dict[str, float] = grid.band_eff_waves if use_bb else {}
    ext_ratios: dict[str, float] = grid._ext_ratios if use_bb else {}
    _NEG_INF = -1e300

    all_bounds = bounds.as_list() + (bb_bounds.as_list() if bb_bounds is not None else [])

    def log_likelihood(params: NDArray[np.float64]) -> float:
        teff, logg, av, plx = params[0], params[1], params[2], params[3]
        if plx <= 0.0:
            return _NEG_INF
        dist_pc = 1000.0 / plx

        if use_bb:
            log10_t_bb = float(params[4])
            log10_l_bb = float(params[5])
            t_bb_k = 10.0 ** log10_t_bb
            l_bb_lsun = 10.0 ** log10_l_bb
            # Get unextincted WD apparent mags (a_v=0 → dist_mod only).
            result = grid.synth_phot(teff, logg, 0.0, dist_pc, bands=band_subset)
            if use_age_gate:
                if not ifmr.age_gate_ok(result["m_wd"], system_age_yr):
                    return _NEG_INF
            pred_mags_list: list[float] = []
            for b in band_subset:
                m_wd_unext = float(result["mags"][b])
                m_bb_unext = _bb_apparent_mag_ab(eff_waves[b], t_bb_k, l_bb_lsun, dist_pc)
                # Combine WD + BB in flux space (both unextincted), then add extinction.
                f_total = 10.0 ** (-m_wd_unext / 2.5) + 10.0 ** (-m_bb_unext / 2.5)
                m_combined_unext = -2.5 * math.log10(f_total) if f_total > 0.0 else 99.0
                ext_mag = float(av) * ext_ratios.get(b, 0.0)
                pred_mags_list.append(m_combined_unext + ext_mag)
            pred_mags = np.array(pred_mags_list, dtype=np.float64)
        else:
            result = grid.synth_phot(teff, logg, av, dist_pc, bands=band_subset)
            if use_age_gate:
                if not ifmr.age_gate_ok(result["m_wd"], system_age_yr):
                    return _NEG_INF
            pred_mags = np.array([result["mags"][b] for b in band_subset])

        resid = mag_vec - pred_mags
        lnl = -0.5 * (float(np.dot(resid ** 2, inv_var)) + norm_sum)
        # Gaussian parallax constraint from Gaia.
        if _gaia_plx is not None and _gaia_plx_err is not None and _gaia_plx_err > 0:
            lnl += -0.5 * ((plx - _gaia_plx) / _gaia_plx_err) ** 2
        return lnl

    def prior_transform(u: NDArray[np.float64]) -> NDArray[np.float64]:
        result = np.empty(len(all_bounds))
        for i, (lo, hi) in enumerate(all_bounds):
            result[i] = lo + u[i] * (hi - lo)
        return result

    return log_likelihood, prior_transform


# ---------------------------------------------------------------------------
# run_wd_fit
# ---------------------------------------------------------------------------
def run_wd_fit(
    phot_rows: list[PhotRow],
    *,
    wd_dir: Path,
    atm_types: tuple[AtmType, ...] = ("DA", "DB"),
    ifmr_variants: tuple[IFMRVariant, ...] = ("MIST", "PARSEC"),
    system_age_yr: float = 0.0,
    prior_bounds: WDPriorBounds | None = None,
    bb_bounds: BBPriorBounds | None = None,
    nlive: int = 200,
    maxiter: int | None = None,
    seed: int = 42,
    outdir: Path | None = None,
) -> list[WDFitResult]:
    """
    Run nested-sampling WD fits for all (atm_type × ifmr) combinations.

    Parameters
    ----------
    phot_rows :
        Photometry rows (``PhotRow`` list).  Upper-limit rows are ignored.
    wd_dir :
        Directory containing ``Table_DA`` and ``Table_DB``.
    atm_types :
        Atmosphere models to include.  Default: both DA and DB.
    ifmr_variants :
        IFMR variants to include.  Default: both MIST and PARSEC.
    system_age_yr :
        System age in years for IFMR age gate.  ``0.0`` → disabled.
    prior_bounds :
        Uniform prior bounds; default ``WDPriorBounds()``.
    bb_bounds :
        When provided, activate the IR blackbody component and use these prior
        bounds for ``log10_T_bb`` and ``log10_L_bb``.  Default ``None`` → no BB.
    nlive :
        Dynesty number of live points.
    maxiter :
        Optional dynesty iteration cap (for fast tests).
    seed :
        RNG seed passed to dynesty.
    outdir :
        If given, write per-combination JSON summaries and results NPZ here.

    Returns
    -------
    list[WDFitResult]
        One result per (atm_type, ifmr) pair, in (atm_types × ifmr_variants)
        order.

    Limits
    ------
    Requires ``dynesty`` ≥ 2.1.  Each run is independent (no warm-starting).
    """
    import dynesty  # deferred import — not always present

    if prior_bounds is None:
        prior_bounds = WDPriorBounds()

    det_rows = [r for r in phot_rows if r.flag == FLAG_DETECTION]
    if not det_rows:
        raise ValueError("run_wd_fit: no detection rows in photometry.")

    ndim = 4 + (2 if bb_bounds is not None else 0)

    results: list[WDFitResult] = []

    for atm in atm_types:
        grid = BergeronGrid.from_dir(wd_dir, atm)
        for variant in ifmr_variants:
            ifmr = CummingsIFMR(variant)
            log_like, ptform = _make_loglike(
                grid, ifmr, det_rows, prior_bounds, system_age_yr,
                bb_bounds=bb_bounds,
            )

            rng = np.random.default_rng(seed)
            sampler = dynesty.NestedSampler(
                log_like,
                ptform,
                ndim=ndim,
                nlive=nlive,
                rstate=rng,
            )
            run_kwargs: dict = {}
            if maxiter is not None:
                run_kwargs["maxiter"] = maxiter
            sampler.run_nested(**run_kwargs)

            dres = sampler.results
            samples = dres.samples  # (n, 4)
            weights = np.exp(dres.logwt - dres.logz[-1])
            logevidence = float(dres.logz[-1])

            # Derive M_WD, M_i, IFMR σ_Mi, and t_cool for each posterior sample.
            n_samples = len(samples)
            m_wd_arr       = np.empty(n_samples)
            m_i_arr        = np.full(n_samples, float("nan"))
            m_i_unc_arr    = np.full(n_samples, float("nan"))
            extrap_arr     = np.zeros(n_samples, dtype=bool)
            t_cool_yr_arr  = np.empty(n_samples)

            for j in range(n_samples):
                teff, logg, av, plx = samples[j]
                dist_pc = 1000.0 / max(plx, 1e-6)
                sr = grid.synth_phot(teff, logg, av, dist_pc, bands=None)
                mw = sr["m_wd"]
                m_wd_arr[j]       = mw
                mi, sigma_mi      = ifmr.initial_mass_unc(mw)
                m_i_arr[j]        = mi
                m_i_unc_arr[j]    = sigma_mi
                extrap_arr[j]     = sr["extrap_mass"]
                t_cool_yr_arr[j]  = sr["cooling_age_yr"]

            result = WDFitResult(
                atm_type=atm,
                ifmr=variant,
                logevidence=logevidence,
                samples=samples,
                weights=weights,
                m_wd_samples=m_wd_arr,
                m_i_samples=m_i_arr,
                m_i_unc_ifmr_samples=m_i_unc_arr,
                extrap_mask=extrap_arr,
                t_cool_yr_samples=t_cool_yr_arr,
            )
            results.append(result)

            if outdir is not None:
                outdir.mkdir(parents=True, exist_ok=True)
                stem = f"wd_{atm}_{variant}"
                np.savez(
                    outdir / f"{stem}_samples.npz",
                    samples=samples,
                    weights=weights,
                    m_wd=m_wd_arr,
                    m_i=m_i_arr,
                    m_i_unc_ifmr=m_i_unc_arr,
                    t_cool_yr=t_cool_yr_arr,
                )
                summ = result.summary()
                (outdir / f"{stem}_summary.json").write_text(
                    json.dumps(summ, indent=2)
                )

    return results
