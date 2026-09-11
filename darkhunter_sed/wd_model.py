"""
Path-2 white-dwarf + MS-companion SED model (``--model wd``).

Runs four nested-sampling fits in sequence, one per (atm_type, IFMR variant)
combination:

    DA × MIST,  DA × PARSEC,  DB × MIST,  DB × PARSEC

Each fit combines a Bergeron DA/DB white-dwarf atmosphere (this module's
``BergeronGrid``) with a coeval MIST/PHOENIX MS companion, tying the WD's
Cummings+2018 IFMR-implied progenitor lifetime plus cooling age to the
companion's age. See ``WD_STAR_PARAM_NAMES`` for the 8 free parameters and
``run_wd_plus_star_fit`` for the fit driver.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from darkhunter_sed.bergeron_wd import AtmType, BergeronGrid
from darkhunter_sed.cummings_ifmr import CummingsIFMR, IFMRVariant, ms_lifetime_yr
from darkhunter_sed.phot_sed_io import FLAG_DETECTION, PhotRow


# ---------------------------------------------------------------------------
# WD + MS companion 2-component fit
# ---------------------------------------------------------------------------

WD_STAR_PARAM_NAMES: tuple[str, ...] = (
    "EEP1", "M1", "FeH", "Av", "parallax", "Teff_WD", "logg_WD", "sigma_int"
)
_WDS_EEP_IDX  = 0
_WDS_M1_IDX   = 1
_WDS_FEH_IDX  = 2
_WDS_AV_IDX   = 3
_WDS_PLX_IDX  = 4
_WDS_TEFF_IDX = 5
_WDS_LOGG_IDX = 6
_WDS_SIG_IDX  = 7

_WDS_AFE_FIXED = 0.0          # aFe fixed at solar for WD+star fit
_WDS_PHOT_ERR_FLOOR = 0.01    # minimum photometric error (mag)
_WDS_LOGLIKE_FLOOR  = -1e5
_WDS_SIGMA_INT_SCALE = 0.05   # exponential prior scale for sigma_int (mag)
_WDS_PLX_SIGMA_CLIP  = 5.0    # flat parallax prior width in Gaia σ


@dataclass(frozen=True, slots=True)
class WDStarPriorBounds:
    """
    Uniform prior hypercube for the 8-D WD+MS companion fit.

    Parameters
    ----------
    eep1 :
        EEP of the MS companion (202 = ZAMS, 605 ~ near RGB tip).
    m1 :
        Initial mass of the MS companion (M_sun).
    feh :
        [Fe/H] (shared between star and WD progenitor).
    av :
        Visual extinction (mag, F99 R_V=3.1).
    parallax_mas :
        Parallax (mas); narrowed at runtime around the Gaia measurement.
    teff_wd :
        WD effective temperature (K).
    logg_wd :
        WD surface gravity (dex).
    sigma_int :
        Intrinsic SED scatter (mag); sampled with an exponential prior.
    """

    eep1:         tuple[float, float] = (202.0, 605.0)
    m1:           tuple[float, float] = (0.5, 2.0)
    feh:          tuple[float, float] = (-2.0, 0.5)
    av:           tuple[float, float] = (0.0, 5.0)
    parallax_mas: tuple[float, float] = (0.1, 100.0)
    teff_wd:      tuple[float, float] = (1500.0, 150_000.0)
    logg_wd:      tuple[float, float] = (7.0, 9.0)
    sigma_int:    tuple[float, float] = (0.0, 0.5)

    def as_list(self) -> list[tuple[float, float]]:
        return [
            self.eep1, self.m1, self.feh, self.av, self.parallax_mas,
            self.teff_wd, self.logg_wd, self.sigma_int,
        ]


def _make_wdstar_loglike(
    grid: BergeronGrid,
    ifmr: CummingsIFMR,
    obs_rows: list[PhotRow],
    bounds: WDStarPriorBounds,
    mist_predictor: Any,
    phoenix_grid: Any,
    bandpasses: Any,
) -> tuple[Any, Any]:
    """
    Build (log_likelihood, prior_transform) for the WD+MS companion fit.

    The combined SED flux is star (PHOENIX/MISTy) + WD (Bergeron).
    The age constraint requires t_MS(IFMR⁻¹(M_WD)) + t_cool ≤ companion_age.
    """
    from darkhunter_sed.filters_synphot import BAND_REGISTRY
    from darkhunter_sed.misty_iso import evaluate_mist
    from darkhunter_sed.phoenix_grid import phoenix_synth_phot

    _GAIA_CONSTRAINT_BANDS = frozenset(
        {"Gaia_parallax", "Gaia_Teff", "Gaia_logg", "Gaia_MH"}
    )

    # Gaia parallax extracted before band filtering.
    _gaia_plx_rows = [r for r in obs_rows if r.band == "Gaia_parallax"]
    _gaia_plx: float | None = float(_gaia_plx_rows[0].mag) if _gaia_plx_rows else None
    _gaia_plx_err: float | None = (
        float(_gaia_plx_rows[0].err) if _gaia_plx_rows else None
    )
    if _gaia_plx is None:
        print("  [wd+star] WARNING: no Gaia_parallax row; parallax unconstrained.")

    # Detection-only photometry rows, excluding Gaia pseudo-bands.
    phot_rows = [
        r for r in obs_rows
        if r.flag == FLAG_DETECTION and r.band not in _GAIA_CONSTRAINT_BANDS
    ]

    avail_bergeron = set(grid.available_bands)
    avail_phoenix  = set(BAND_REGISTRY.keys())
    phoenix_bands  = [r.band for r in phot_rows if r.band in avail_phoenix]
    bergeron_bands = [r.band for r in phot_rows if r.band in avail_bergeron]
    all_phot_bands = sorted({r.band for r in phot_rows})

    # Set PHOENIX wavelength grid once for efficiency.
    if phoenix_grid is not None and bandpasses is not None:
        from darkhunter_sed.filters_synphot import bandpass_wavelength_grid
        phoenix_grid.set_photometry_wavelengths(bandpass_wavelength_grid(bandpasses))

    # Narrow parallax prior around Gaia measurement.
    plx_lo, plx_hi = bounds.parallax_mas
    if _gaia_plx is not None and _gaia_plx_err is not None and _gaia_plx_err > 0:
        plx_lo = max(plx_lo, _gaia_plx - _WDS_PLX_SIGMA_CLIP * _gaia_plx_err)
        plx_hi = min(plx_hi, _gaia_plx + _WDS_PLX_SIGMA_CLIP * _gaia_plx_err)

    bound_list = list(bounds.as_list())
    bound_list[_WDS_PLX_IDX] = (plx_lo, plx_hi)
    sig_hi = float(bound_list[_WDS_SIG_IDX][1])

    def log_likelihood(params: NDArray[np.float64]) -> float:
        eep1    = float(params[_WDS_EEP_IDX])
        m1      = float(params[_WDS_M1_IDX])
        feh     = float(params[_WDS_FEH_IDX])
        av      = float(params[_WDS_AV_IDX])
        plx     = float(params[_WDS_PLX_IDX])
        teff_wd = float(params[_WDS_TEFF_IDX])
        logg_wd = float(params[_WDS_LOGG_IDX])
        sig     = float(params[_WDS_SIG_IDX])

        if plx <= 0.0:
            return _WDS_LOGLIKE_FLOOR
        dist_pc = 1000.0 / plx

        # MS companion via MIST.
        try:
            mist = evaluate_mist(
                eep1, m1, feh, _WDS_AFE_FIXED, predictor=mist_predictor
            )
        except Exception:
            return _WDS_LOGLIKE_FLOOR
        companion_age_yr = mist.age_gyr * 1.0e9

        # PHOENIX magnitudes for the companion.
        star_mags: dict[str, float] = {}
        if phoenix_bands and phoenix_grid is not None:
            try:
                raw = phoenix_synth_phot(
                    phoenix_grid,
                    mist.teff_k,
                    mist.logg,
                    feh,
                    _WDS_AFE_FIXED,
                    av,
                    phoenix_bands,
                    radius_cm=mist.radius_cm,
                    distance_pc=dist_pc,
                    systems=("ab",),
                    bandpasses=bandpasses,
                )
                for b in phoenix_bands:
                    m = float(raw[b]["ab"])
                    if math.isfinite(m):
                        star_mags[b] = m
            except Exception:
                return _WDS_LOGLIKE_FLOOR

        # Bergeron magnitudes for the WD.
        try:
            wd_result = grid.synth_phot(
                teff_wd, logg_wd, av, dist_pc, bands=bergeron_bands
            )
        except Exception:
            return _WDS_LOGLIKE_FLOOR
        wd_mags: dict[str, float] = {}
        for b in bergeron_bands:
            m = float(wd_result["mags"][b])
            if math.isfinite(m) and m < 90.0:
                wd_mags[b] = m
        m_wd     = wd_result["m_wd"]
        t_cool_yr = wd_result["cooling_age_yr"]

        # Age constraint: WD progenitor MS lifetime + cooling ≤ companion age.
        mi = ifmr.initial_mass(m_wd)
        if not math.isnan(mi):
            if ms_lifetime_yr(mi) + t_cool_yr > companion_age_yr:
                return _WDS_LOGLIKE_FLOOR

        # Combine star + WD fluxes per band.
        pred_mags: dict[str, float] = {}
        for b in all_phot_bands:
            m_s = star_mags.get(b)
            m_w = wd_mags.get(b)
            if m_s is not None and m_w is not None:
                f = 10.0 ** (-m_s / 2.5) + 10.0 ** (-m_w / 2.5)
                pred_mags[b] = -2.5 * math.log10(f) if f > 0.0 else 99.0
            elif m_s is not None:
                pred_mags[b] = m_s
            elif m_w is not None:
                pred_mags[b] = m_w

        if not pred_mags:
            return _WDS_LOGLIKE_FLOOR

        # Gaussian photometry log-likelihood with sigma_int.
        lnl = 0.0
        sig_sq = sig * sig
        for row in phot_rows:
            if row.band not in pred_mags:
                continue
            m_pred = pred_mags[row.band]
            m_obs  = float(row.mag)
            m_err  = float(row.err) if row.err else 0.05
            err_eff = math.sqrt(
                m_err * m_err + _WDS_PHOT_ERR_FLOOR * _WDS_PHOT_ERR_FLOOR + sig_sq
            )
            lnl += -0.5 * ((m_obs - m_pred) / err_eff) ** 2 - math.log(
                math.sqrt(2.0 * math.pi) * err_eff
            )

        # Gaia parallax Gaussian constraint.
        if _gaia_plx is not None and _gaia_plx_err is not None and _gaia_plx_err > 0:
            lnl += -0.5 * ((plx - _gaia_plx) / _gaia_plx_err) ** 2

        if not math.isfinite(lnl) or lnl < _WDS_LOGLIKE_FLOOR:
            return _WDS_LOGLIKE_FLOOR
        return lnl

    def prior_transform(u: NDArray[np.float64]) -> NDArray[np.float64]:
        theta = np.empty(len(bound_list))
        for i, (lo, hi) in enumerate(bound_list):
            theta[i] = lo + u[i] * (hi - lo)
        # sigma_int: exponential prior truncated at sig_hi.
        u_sig = float(u[_WDS_SIG_IDX])
        trunc = 1.0 - math.exp(-sig_hi / _WDS_SIGMA_INT_SCALE)
        theta[_WDS_SIG_IDX] = -_WDS_SIGMA_INT_SCALE * math.log(
            max(1.0 - u_sig * trunc, 1e-300)
        )
        return theta

    return log_likelihood, prior_transform


def _wdstar_summary(
    atm_type: str,
    ifmr: str,
    gaia_id: str,
    logevidence: float,
    samples: NDArray[np.float64],
    weights: NDArray[np.float64],
    m_wd_samples: NDArray[np.float64],
    m_i_samples: NDArray[np.float64],
) -> dict[str, Any]:
    """JSON-serializable summary for one WD+star fit combination."""
    w = weights / weights.sum()

    def _wq(arr: NDArray[np.float64], q: float) -> float:
        idx = np.argsort(arr)
        cdf = np.cumsum(w[idx])
        return float(arr[idx[np.searchsorted(cdf, q)]])

    d: dict[str, Any] = {
        "model": "wd+star",
        "atm_type": atm_type,
        "ifmr": ifmr,
        "gaia_id": gaia_id,
        "logevidence": float(logevidence),
    }
    for i, name in enumerate(WD_STAR_PARAM_NAMES):
        col = samples[:, i]
        d[f"{name}_median"] = _wq(col, 0.50)
        d[f"{name}_lo"]     = _wq(col, 0.16)
        d[f"{name}_hi"]     = _wq(col, 0.84)

    valid_wd = np.isfinite(m_wd_samples)
    if valid_wd.any():
        mw = m_wd_samples[valid_wd]
        ww = w[valid_wd]
        ww /= ww.sum()
        def _wq2(arr, q):
            idx = np.argsort(arr)
            cdf = np.cumsum(ww[idx])
            return float(arr[idx[np.searchsorted(cdf, q)]])
        d["m_wd_median"] = _wq2(mw, 0.50)
        d["m_wd_lo"]     = _wq2(mw, 0.16)
        d["m_wd_hi"]     = _wq2(mw, 0.84)
    else:
        d["m_wd_median"] = float("nan")

    valid_mi = np.isfinite(m_i_samples)
    d["m_i_median"] = (
        float(np.median(m_i_samples[valid_mi])) if valid_mi.any() else float("nan")
    )
    return d


def run_wd_plus_star_fit(
    phot_rows: list[PhotRow],
    *,
    wd_dir: Path,
    mist_predictor: Any,
    phoenix_grid: Any,
    bandpasses: Any = None,
    gaia_id: str = "",
    atm_types: tuple[AtmType, ...] = ("DA", "DB"),
    ifmr_variants: tuple[IFMRVariant, ...] = ("MIST", "PARSEC"),
    prior_bounds: WDStarPriorBounds | None = None,
    nlive: int = 200,
    maxiter: int | None = None,
    seed: int = 42,
    dlogz: float = 0.5,
    outdir: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Run nested-sampling WD+MS companion fits for all (atm_type × IFMR) pairs.

    Parameters
    ----------
    phot_rows :
        Photometry rows (PhotRow list); upper-limit rows are ignored.
    wd_dir :
        Directory containing Bergeron ``Table_DA`` and ``Table_DB``.
    mist_predictor :
        MISTy predictor callable (from :func:`~darkhunter_sed.misty_iso.load_misty_predictor`).
    phoenix_grid :
        :class:`~darkhunter_sed.phoenix_grid.PhoenixGrid` instance.
    bandpasses :
        Synphot bandpass mapping ``{band: bandpass}``; ``None`` → loaded lazily.
    gaia_id :
        Gaia source id for summary JSON and output filenames.
    atm_types, ifmr_variants :
        Atmosphere models and IFMR variants to run.
    prior_bounds :
        Prior hypercube; default :class:`WDStarPriorBounds`.
    nlive, maxiter, seed, dlogz :
        dynesty settings.
    outdir :
        When given, write per-combination NPZ and JSON summaries here.

    Returns
    -------
    list[dict]
        One dict per (atm_type, ifmr) pair with keys:
        ``atm_type, ifmr, logevidence, samples, weights,
        m_wd_samples, m_i_samples, t_cool_yr_samples, param_names, gaia_id``.
    """
    import dynesty

    if prior_bounds is None:
        prior_bounds = WDStarPriorBounds()

    det_rows = [r for r in phot_rows if r.flag == FLAG_DETECTION]
    if not det_rows:
        raise ValueError("run_wd_plus_star_fit: no detection rows in photometry.")

    ndim = len(WD_STAR_PARAM_NAMES)
    results: list[dict[str, Any]] = []

    for atm in atm_types:
        grid = BergeronGrid.from_dir(wd_dir, atm)
        for variant in ifmr_variants:
            ifmr_obj = CummingsIFMR(variant)
            log_like, ptform = _make_wdstar_loglike(
                grid,
                ifmr_obj,
                det_rows,
                prior_bounds,
                mist_predictor=mist_predictor,
                phoenix_grid=phoenix_grid,
                bandpasses=bandpasses,
            )

            rng = np.random.default_rng(seed)
            sampler = dynesty.NestedSampler(
                log_like, ptform, ndim=ndim, nlive=nlive, rstate=rng
            )
            run_kw: dict = {"dlogz": dlogz}
            if maxiter is not None:
                run_kw["maxiter"] = maxiter
            sampler.run_nested(**run_kw)

            dres     = sampler.results
            samples  = dres.samples
            weights  = np.exp(dres.logwt - dres.logz[-1])
            lnz      = float(dres.logz[-1])

            # Derive M_WD, M_i, t_cool for each posterior sample.
            n_samp        = len(samples)
            m_wd_arr      = np.full(n_samp, float("nan"))
            m_i_arr       = np.full(n_samp, float("nan"))
            t_cool_yr_arr = np.full(n_samp, float("nan"))

            for j in range(n_samp):
                teff_wd = float(samples[j, _WDS_TEFF_IDX])
                logg_wd = float(samples[j, _WDS_LOGG_IDX])
                av_j    = float(samples[j, _WDS_AV_IDX])
                plx_j   = float(samples[j, _WDS_PLX_IDX])
                dist_j  = 1000.0 / max(plx_j, 1e-6)
                try:
                    sr = grid.synth_phot(teff_wd, logg_wd, av_j, dist_j, bands=None)
                    m_wd_arr[j]      = sr["m_wd"]
                    m_i_arr[j]       = ifmr_obj.initial_mass(m_wd_arr[j])
                    t_cool_yr_arr[j] = sr["cooling_age_yr"]
                except Exception:
                    pass

            res: dict[str, Any] = {
                "atm_type":        atm,
                "ifmr":            variant,
                "logevidence":     lnz,
                "samples":         samples,
                "weights":         weights,
                "m_wd_samples":    m_wd_arr,
                "m_i_samples":     m_i_arr,
                "t_cool_yr_samples": t_cool_yr_arr,
                "param_names":     WD_STAR_PARAM_NAMES,
                "gaia_id":         gaia_id,
            }
            results.append(res)

            if outdir is not None:
                outdir.mkdir(parents=True, exist_ok=True)
                stem = f"wdstar_{atm}_{variant}"
                np.savez(
                    outdir / f"{stem}_samples.npz",
                    samples=samples,
                    weights=weights,
                    m_wd=m_wd_arr,
                    m_i=m_i_arr,
                    t_cool_yr=t_cool_yr_arr,
                    param_names=np.array(WD_STAR_PARAM_NAMES),
                )
                summ = _wdstar_summary(
                    atm, variant, gaia_id, lnz, samples, weights,
                    m_wd_arr, m_i_arr,
                )
                (outdir / f"{stem}_summary.json").write_text(
                    json.dumps(summ, indent=2)
                )

    return results
