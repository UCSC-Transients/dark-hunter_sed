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
def _make_loglike(
    grid: BergeronGrid,
    ifmr: CummingsIFMR,
    obs_rows: list[PhotRow],
    bounds: WDPriorBounds,
    system_age_yr: float,
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
        Prior bounds.
    system_age_yr :
        System age in years; 0.0 → age gate disabled.

    Returns
    -------
    (log_likelihood, prior_transform)
        Both accept / return 1-D numpy arrays of length 4.
    """
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
    _NEG_INF = -1e300

    def log_likelihood(params: NDArray[np.float64]) -> float:
        teff, logg, av, plx = params
        if plx <= 0.0:
            return _NEG_INF
        dist_pc = 1000.0 / plx
        result = grid.synth_phot(teff, logg, av, dist_pc, bands=band_subset)
        if use_age_gate:
            m_wd = result["m_wd"]
            if not ifmr.age_gate_ok(m_wd, system_age_yr):
                return _NEG_INF
        pred_mags = np.array([result["mags"][b] for b in band_subset])
        resid = mag_vec - pred_mags
        return -0.5 * (float(np.dot(resid ** 2, inv_var)) + norm_sum)

    def prior_transform(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return bounds.prior_transform(u)

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

    results: list[WDFitResult] = []

    for atm in atm_types:
        grid = BergeronGrid.from_dir(wd_dir, atm)
        for variant in ifmr_variants:
            ifmr = CummingsIFMR(variant)
            log_like, ptform = _make_loglike(
                grid, ifmr, det_rows, prior_bounds, system_age_yr
            )

            rng = np.random.default_rng(seed)
            sampler = dynesty.NestedSampler(
                log_like,
                ptform,
                ndim=4,
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

            # Derive M_WD, M_i, and IFMR systematic σ_Mi for each posterior sample.
            n_samples = len(samples)
            m_wd_arr     = np.empty(n_samples)
            m_i_arr      = np.full(n_samples, float("nan"))
            m_i_unc_arr  = np.full(n_samples, float("nan"))
            extrap_arr   = np.zeros(n_samples, dtype=bool)

            for j in range(n_samples):
                teff, logg, av, plx = samples[j]
                dist_pc = 1000.0 / max(plx, 1e-6)
                sr = grid.synth_phot(teff, logg, av, dist_pc, bands=None)
                mw = sr["m_wd"]
                m_wd_arr[j]    = mw
                mi, sigma_mi   = ifmr.initial_mass_unc(mw)
                m_i_arr[j]     = mi
                m_i_unc_arr[j] = sigma_mi
                extrap_arr[j]  = sr["extrap_mass"]

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
                )
                summ = result.summary()
                (outdir / f"{stem}_summary.json").write_text(
                    json.dumps(summ, indent=2)
                )

    return results
