"""
Path-2 dynesty nested sampling for photometry SED models.

Supports **1-star** and **2-star coeval** models.  Reports **lnZ** from dynesty
and **BIC** from the maximum-likelihood sample. Writes under ``output/phot_sed/``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from darkhunter_sed.config import phot_sed_dir
from darkhunter_sed.misty_iso import (
    DEFAULT_EEP_BOUNDS,
    DEFAULT_MASS_BOUNDS,
    MistPredictFn,
)
from darkhunter_sed.phot_sed_io import FLAG_DETECTION, FLAG_UPPER_LIMIT, PhotRow
from darkhunter_sed.phot_sed_models import (
    ONE_STAR_PARAM_NAMES,
    TWO_STAR_PARAM_NAMES,
    OneStarPrediction,
    SynthPhotFn,
    SynthPhot2StarFn,
    TwoStarPrediction,
    predict_1star_phot,
    predict_2star_phot,
)
from darkhunter_sed.phoenix_grid import PhoenixGrid

# Minimal uniform priors for Issue 4 (full phot_sed_priors later).
DEFAULT_FEH_BOUNDS: tuple[float, float] = (-2.0, 0.5)
DEFAULT_AFE_BOUNDS: tuple[float, float] = (-0.2, 0.6)
DEFAULT_AV_BOUNDS: tuple[float, float] = (0.0, 5.0)
DEFAULT_PARALLAX_BOUNDS: tuple[float, float] = (0.1, 100.0)  # mas
DEFAULT_SIGMA_INT_BOUNDS: tuple[float, float] = (0.0, 0.5)  # intrinsic scatter, mag


@dataclass(frozen=True, slots=True)
class OneStarPriorBounds:
    """
    Uniform prior hypercube for 1-star parameters.

    Parameters
    ----------
    eep, mass, feh, a_v, parallax_mas :
        Inclusive ``(lo, hi)`` bounds in :data:`ONE_STAR_PARAM_NAMES` order.
        aFe is fixed at 0.0 (PHOENIX alpha grid is sparse for non-zero alpha).
    """

    eep: tuple[float, float] = DEFAULT_EEP_BOUNDS
    mass: tuple[float, float] = DEFAULT_MASS_BOUNDS
    feh: tuple[float, float] = DEFAULT_FEH_BOUNDS
    a_v: tuple[float, float] = DEFAULT_AV_BOUNDS
    parallax_mas: tuple[float, float] = DEFAULT_PARALLAX_BOUNDS
    sigma_int: tuple[float, float] = DEFAULT_SIGMA_INT_BOUNDS

    def as_list(self) -> list[tuple[float, float]]:
        """Return bounds in dynesty parameter order (physical params + sigma_int)."""
        return [self.eep, self.mass, self.feh, self.a_v, self.parallax_mas, self.sigma_int]


@dataclass(frozen=True, slots=True)
class FitResult1Star:
    """
    1-star dynesty summary.

    Parameters
    ----------
    param_names :
        Free-parameter labels.
    samples :
        Equally weighted posterior samples ``(n_samples, ndim)``.
    logl :
        Log-likelihood at each sample (same order as ``samples`` before reweight;
        for max-L we use the raw nested-sampling ``logl`` array).
    logz :
        Bayesian evidence ``ln Z``.
    logz_err :
        Approximate uncertainty on ``ln Z``.
    bic :
        BIC from the maximum-likelihood nested-sampling point.
    ln_l_max :
        Max log-likelihood among nested-sampling live/dead points.
    best_theta :
        Parameter vector at ``ln_l_max``.
    n_data :
        Number of photometry rows used in the likelihood.
    n_free :
        Number of free parameters (``ndim``).
    nlive :
        Nested-sampling live points.
    """

    param_names: tuple[str, ...]
    samples: NDArray[np.float64]
    logl: NDArray[np.float64]
    logz: float
    logz_err: float
    bic: float
    ln_l_max: float
    best_theta: NDArray[np.float64]
    n_data: int
    n_free: int
    nlive: int


def bic_from_max_likelihood(*, ln_l_max: float, n_free: int, n_data: int) -> float:
    """
    Bayesian Information Criterion from the maximum-likelihood sample.

    Parameters
    ----------
    ln_l_max :
        Natural log of the maximum likelihood.
    n_free :
        Number of free parameters ``k``.
    n_data :
        Number of data points ``n`` (photometry rows in the likelihood).

    Returns
    -------
    float
        ``BIC = k * ln(n) - 2 * ln(L_max)``.

    Limits
    ------
    Requires ``n_data >= 1``. Uses natural log consistently with ``ln_l_max``.
    """
    if n_data < 1:
        raise ValueError("n_data must be >= 1 for BIC")
    if n_free < 1:
        raise ValueError("n_free must be >= 1")
    return float(n_free) * math.log(float(n_data)) - 2.0 * float(ln_l_max)


def photometry_loglike(
    pred_mags: Mapping[str, float],
    rows: Sequence[PhotRow],
    *,
    sigma_int: float = 0.0,
) -> float:
    """
    Gaussian photometry log-likelihood with 3σ upper-limit treatment.

    Parameters
    ----------
    pred_mags :
        Model magnitudes keyed by band name.
    rows :
        Observed :class:`PhotRow` list (detections and/or ULs).
    sigma_int :
        Intrinsic scatter (mag) added in quadrature to each band's effective
        error alongside the fixed :data:`_PHOT_ERR_FLOOR`.  Sampled as a free
        parameter by dynesty; pass ``0.0`` for direct evaluation.

    Returns
    -------
    float
        Sum of per-band contributions (natural log).

    Notes
    -----
    Effective error per band:
    ``err_eff = sqrt(err_phot^2 + _PHOT_ERR_FLOOR^2 + sigma_int^2)``.

    - Detections (``flag=0``): ``-0.5*((obs-mod)/err_eff)^2 - ln(err_eff*sqrt(2π))``.
    - Upper limits (``flag=1``): penalty only when model is brighter than the limit.
    - Sentinel model (``m_mod >= 90``): photospheric model predicts no flux;
      detection may indicate a non-photospheric source (WD companion, accretion).
      Skipped for the 1-star model — the 2-star / WD model handles it.
    """
    ln_norm = math.log(math.sqrt(2.0 * math.pi))
    sigma_int_sq = float(sigma_int) ** 2
    total = 0.0
    for row in rows:
        m_mod = float(pred_mags[row.band])
        err_raw = float(row.err)
        if err_raw <= 0.0 or not math.isfinite(err_raw):
            raise ValueError(f"Invalid err for band {row.band}")
        if not math.isfinite(m_mod):
            m_mod = 99.0
        err = math.sqrt(err_raw**2 + _PHOT_ERR_FLOOR**2 + sigma_int_sq)
        if row.flag == FLAG_DETECTION:
            if m_mod >= 90.0:
                # Photospheric model predicts no flux; detection may be a WD
                # companion or accretion.  The 1-star model cannot explain this
                # band so we skip it here; the 2-star / WD model handles it.
                continue
            resid = (float(row.mag) - m_mod) / err
            total += -0.5 * resid * resid - math.log(err) - ln_norm
        elif row.flag == FLAG_UPPER_LIMIT:
            # Brighter than limit → penalize; fainter → no contribution.
            if m_mod < float(row.mag):
                resid = (float(row.mag) - m_mod) / err
                total += -0.5 * resid * resid - math.log(err) - ln_norm
        else:
            raise ValueError(f"Unsupported flag={row.flag} for band {row.band}")
    return float(total)


def _unit_cube_to_bounds(
    u: NDArray[np.floating],
    bounds: Sequence[tuple[float, float]],
) -> NDArray[np.float64]:
    """Map ``u ∈ [0,1]^d`` to the hyper-rectangle ``bounds`` (vectorized)."""
    uu = np.asarray(u, dtype=np.float64).ravel()
    if uu.size != len(bounds):
        raise ValueError(f"unit cube length {uu.size} != ndim {len(bounds)}")
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    return lo + uu * (hi - lo)


def load_bandpasses_for_bands(
    bands: Sequence[str],
    *,
    cdbs_root: Path | str | None = None,
) -> dict[str, object]:
    """
    Load synphot ``SpectralElement``s once for a set of Path-2 band names.

    Parameters
    ----------
    bands :
        Registry band names (duplicates ignored; order preserved by first seen).
    cdbs_root :
        Optional CDBS root for :func:`darkhunter_sed.filters_synphot.load_bandpass`.

    Returns
    -------
    dict[str, object]
        ``{band: SpectralElement}``.

    Limits
    ------
    Raises on unknown bands. Call once per fit and pass into the likelihood.
    """
    from darkhunter_sed.filters_synphot import load_bandpass

    out: dict[str, object] = {}
    for band in bands:
        if band in out:
            continue
        out[band] = load_bandpass(band, cdbs_root=cdbs_root)
    return out


_PLX_BAND = "Gaia_parallax"
_PLX_PARAM_IDX = 4  # parallax_mas is the 5th element of the free-param vector (0-indexed)
_PLX_SIGMA_CLIP = 5.0  # tight flat prior spans ± this many σ around observed plx
_TEFF_BAND = "Gaia_Teff"
_LOGG_BAND = "Gaia_logg"
_MH_BAND = "Gaia_MH"
_FEH_PARAM_IDX = 2  # feh is the 3rd element of OneStarParams (0-indexed)
_GAIA_CONSTRAINT_BANDS = frozenset({_PLX_BAND, _TEFF_BAND, _LOGG_BAND, _MH_BAND})

_PHOT_ERR_FLOOR: float = 0.02  # mag — systematic floor (PHOENIX model + zero-point)
_SIGMA_INT_IDX: int = len(ONE_STAR_PARAM_NAMES)  # index of sigma_int in the 7-D theta
# Finite floor returned when the model fails (off-grid MIST/PHOENIX, NaN outputs, or
# physically impossible age).  Must be finite so dynesty always has a non-(-inf) loglstar.
_LOGLIKE_FLOOR: float = -1e100
_UNIVERSE_AGE_GYR: float = 13.8  # stellar age hard upper limit (physical, not photometric)
# Gaia GSP-Phot Teff/logg are photometrically derived (G/BP/RP), so using them as
# likelihood constraints while also fitting Gaia photometry is partially circular.
# Inflate their errors by this factor to down-weight them accordingly.
_GAIA_PHOT_CONSTRAINT_ERR_SCALE: float = 10.0


def fit_1star_dynesty(
    rows: Sequence[PhotRow],
    *,
    mist_predictor: MistPredictFn,
    phoenix_grid: PhoenixGrid | None = None,
    synth_phot: SynthPhotFn | None = None,
    bounds: OneStarPriorBounds | None = None,
    nlive: int = 100,
    maxiter: int | None = None,
    seed: int | None = 42,
    bandpasses: Mapping[str, object] | None = None,
    mag_system: str = "ab",
    dlogz: float = 1.0,
    sample: str = "auto",
    bound: str = "multi",
    nworkers: int = 1,
    jit_warmup: bool = True,
    spec_stride: int = 1,
    print_progress: bool = False,
) -> FitResult1Star:
    """
    Run dynesty nested sampling on the 1-star Path-2 model.

    Parameters
    ----------
    rows :
        Photometry rows (at least one detection recommended).
    mist_predictor :
        Injectable MISTy ``getMIST`` (required so CI can mock).
    phoenix_grid, synth_phot :
        Forward model; one of them required (see :func:`predict_1star_phot`).
    bounds :
        Uniform prior bounds; defaults are minimal Issue-4 uniforms.
    nlive, maxiter, seed, dlogz :
        Dynesty controls. ``maxiter`` stops early (useful in tests).
        ``dlogz`` default 1.0 is appropriate for quick production fits;
        tighten to 0.5 or lower for precise evidence comparison.
    sample :
        Dynesty proposal method (``'auto'``, ``'unif'``, ``'rwalk'``,
        ``'rslice'``, etc.). ``'auto'`` selects ``'unif'`` for ndim < 10.
    bound :
        Dynesty bounding method (``'multi'``, ``'single'``, etc.).
        ``'single'`` is faster for unimodal 6-D posteriors.
    nworkers :
        Number of parallel worker processes for likelihood evaluations
        (default 1 = serial). Values > 1 spawn a ``multiprocessing.Pool``.
        Each worker independently initialises MISTy + PHOENIX; for short
        fits the startup cost can outweigh the gain — benchmark first.
    jit_warmup :
        If ``True`` (default), call ``mist_predictor`` once with a nominal
        point before constructing the ``NestedSampler``.  This forces JAX
        JIT compilation to happen outside the dynesty timing, saving
        10–30 s on the first live-point evaluation.
    spec_stride :
        Subsample the bandpass wavelength grid by this factor before passing
        it to :meth:`PhoenixGrid.set_photometry_wavelengths`.  ``1`` (default)
        uses every point.  Values of 4–10 reduce per-call interpolation and
        trapz cost proportionally with minimal accuracy loss for broad-band
        photometry — useful for a fast exploratory run that can seed priors
        for a subsequent full-resolution run (``spec_stride=1``).
    print_progress :
        Forward dynesty's ``print_progress`` flag.  When ``True``, dynesty
        prints a one-line status (lnZ, remaining work, ncall) to stdout every
        ~1000 iterations so you can monitor long fits.
    bandpasses, mag_system :
        Photometry system / injectable thruputs. When ``bandpasses`` is
        ``None`` and a real ``phoenix_grid`` path is used, bandpasses are
        loaded once via :func:`load_bandpasses_for_bands`. When a
        ``phoenix_grid`` is used, its photometry λ grid is set once from the
        bandpass waveset union (Issue #26).

    Returns
    -------
    FitResult1Star
        Posterior samples, ``lnZ``, and BIC from max-L.

    Limits
    ------
    Uniform priors only (Path-2 ``phot_sed_priors`` not wired yet). Failed
    forward-model evaluations return ``-inf`` likelihood. Fit path uses AB
    magnitudes only (``systems=("ab",)``).
    ``nworkers > 1`` requires the likelihood closure to be picklable;
    JAX JIT functions satisfy this on recent JAX releases but Phoenix grid
    initialisation happens once per worker process.
    """
    import multiprocessing

    from dynesty import NestedSampler

    if not rows:
        raise ValueError("rows must be non-empty")

    # Separate Gaia constraint rows (parallax, Teff, logg, MH) from photometric rows.
    plx_rows = [r for r in rows if r.band == _PLX_BAND]
    teff_rows = [r for r in rows if r.band == _TEFF_BAND]
    logg_rows = [r for r in rows if r.band == _LOGG_BAND]
    mh_rows = [r for r in rows if r.band == _MH_BAND]
    phot_rows = [r for r in rows if r.band not in _GAIA_CONSTRAINT_BANDS]
    if not phot_rows:
        raise ValueError("rows contains only Gaia constraints; need photometric bands too")

    # Tighten parallax prior only — Gaia parallax is astrometric (independent).
    # GSP-Phot MH/Teff/logg are photometrically derived, so using them to shrink
    # the prior would bias results; they appear only as down-weighted likelihood terms.
    prior = bounds if bounds is not None else OneStarPriorBounds()
    if bounds is None and plx_rows:
        plx_obs = plx_rows[0].mag
        plx_err = plx_rows[0].err
        lo = max(DEFAULT_PARALLAX_BOUNDS[0], plx_obs - _PLX_SIGMA_CLIP * plx_err)
        hi = min(DEFAULT_PARALLAX_BOUNDS[1], plx_obs + _PLX_SIGMA_CLIP * plx_err)
        prior = OneStarPriorBounds(parallax_mas=(lo, hi))

    bound_list = prior.as_list()
    bands = [r.band for r in phot_rows]
    ndim = len(ONE_STAR_PARAM_NAMES) + 1  # physical params + sigma_int

    bps = bandpasses
    if bps is None and synth_phot is None:
        bps = load_bandpasses_for_bands(bands)
    if phoenix_grid is not None and bps is not None:
        from darkhunter_sed.filters_synphot import bandpass_wavelength_grid

        wave_grid = bandpass_wavelength_grid(bps)
        if spec_stride > 1:
            wave_grid = wave_grid[::int(spec_stride)]
        phoenix_grid.set_photometry_wavelengths(wave_grid)

    # Warm up JAX JIT before dynesty allocates live points.  The first call
    # to a jax.jit-wrapped function triggers trace+compile (~10-30 s for
    # MISTy LinNet); firing it here keeps that cost outside the sampler.
    if jit_warmup:
        _warmup_bounds = prior.as_list()
        _mid = np.array([0.5 * (lo + hi) for lo, hi in _warmup_bounds], dtype=np.float64)
        try:
            predict_1star_phot(
                _mid[:_SIGMA_INT_IDX],  # physical params only (no sigma_int)
                bands,
                mist_predictor=mist_predictor,
                phoenix_grid=phoenix_grid,
                synth_phot=synth_phot,
                systems=("ab",),
                bandpasses=bps,
                mag_system=mag_system,
            )
        except Exception:
            pass

    def prior_transform(u: NDArray[np.floating]) -> NDArray[np.float64]:
        return _unit_cube_to_bounds(u, bound_list)

    def loglike(theta: NDArray[np.floating]) -> float:
        sigma_int = float(theta[_SIGMA_INT_IDX])
        try:
            pred = predict_1star_phot(
                theta[:_SIGMA_INT_IDX],  # physical params only
                bands,
                mist_predictor=mist_predictor,
                phoenix_grid=phoenix_grid,
                synth_phot=synth_phot,
                systems=("ab",),
                bandpasses=bps,
                mag_system=mag_system,
            )
        except Exception:
            return _LOGLIKE_FLOOR
        # Guard against NaN from off-grid MIST predictions.
        if not (math.isfinite(pred.mist.teff_k) and math.isfinite(pred.mist.logg)):
            return _LOGLIKE_FLOOR
        # Physically impossible: star older than the universe.
        if pred.mist.age_gyr > _UNIVERSE_AGE_GYR:
            return _LOGLIKE_FLOOR
        lnl = photometry_loglike(pred.mags, phot_rows, sigma_int=sigma_int)
        if not math.isfinite(lnl):
            return _LOGLIKE_FLOOR
        # Gaia parallax Gaussian constraint: theta[5] is parallax_mas directly.
        for pr in plx_rows:
            resid = (float(theta[_PLX_PARAM_IDX]) - pr.mag) / pr.err
            lnl += -0.5 * resid * resid
        # Gaia GSP-Phot MH: photometrically derived, errors inflated like Teff/logg.
        for mr in mh_rows:
            resid = (float(theta[_FEH_PARAM_IDX]) - mr.mag) / (mr.err * _GAIA_PHOT_CONSTRAINT_ERR_SCALE)
            lnl += -0.5 * resid * resid
        # Gaia GSP-Phot Teff/logg: photometrically derived, so partially circular
        # with our own photometric fit.  Errors inflated by _GAIA_PHOT_CONSTRAINT_ERR_SCALE
        # to down-weight these terms while still keeping them as weak regularisers.
        for tr in teff_rows:
            resid = (pred.mist.teff_k - tr.mag) / (tr.err * _GAIA_PHOT_CONSTRAINT_ERR_SCALE)
            lnl += -0.5 * resid * resid
        for lr in logg_rows:
            resid = (pred.mist.logg - lr.mag) / (lr.err * _GAIA_PHOT_CONSTRAINT_ERR_SCALE)
            lnl += -0.5 * resid * resid
        return lnl if math.isfinite(lnl) else _LOGLIKE_FLOOR

    pool: Any = None
    queue_size: int | None = None
    if nworkers > 1:
        pool = multiprocessing.Pool(int(nworkers))
        queue_size = int(nworkers)

    try:
        sampler = NestedSampler(
            loglike,
            prior_transform,
            ndim,
            nlive=int(nlive),
            sample=sample,
            bound=bound,
            rstate=np.random.default_rng(seed) if seed is not None else None,
            pool=pool,
            queue_size=queue_size,
        )
        run_kw: dict[str, Any] = {
            "print_progress": bool(print_progress),
            "dlogz": float(dlogz),
        }
        if maxiter is not None:
            run_kw["maxiter"] = int(maxiter)
        sampler.run_nested(**run_kw)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    res = sampler.results

    logl = np.asarray(res.logl, dtype=np.float64)
    samples_u = np.asarray(res.samples, dtype=np.float64)
    imax = int(np.argmax(logl))
    ln_l_max = float(logl[imax])
    best_theta = samples_u[imax].copy()
    n_data = len(rows)
    bic = bic_from_max_likelihood(ln_l_max=ln_l_max, n_free=ndim, n_data=n_data)

    # Equal-weight posterior samples for export.
    try:
        from dynesty.utils import resample_equal

        weights = np.exp(np.asarray(res.logwt, dtype=np.float64) - float(res.logz[-1]))
        eq = np.asarray(resample_equal(samples_u, weights), dtype=np.float64)
    except Exception:
        eq = samples_u

    logz = float(res.logz[-1])
    logz_err = float(res.logzerr[-1]) if getattr(res, "logzerr", None) is not None else float("nan")

    return FitResult1Star(
        param_names=ONE_STAR_PARAM_NAMES + ("sigma_int",),
        samples=eq,
        logl=logl,
        logz=logz,
        logz_err=logz_err,
        bic=bic,
        ln_l_max=ln_l_max,
        best_theta=best_theta,
        n_data=n_data,
        n_free=ndim,
        nlive=int(nlive),
    )


def write_1star_outputs(
    result: FitResult1Star,
    *,
    gaia_id: str,
    out_dir: Path | str | None = None,
    best_pred: OneStarPrediction | None = None,
) -> dict[str, Path]:
    """
    Write 1-star fit products under ``output/phot_sed/``.

    Parameters
    ----------
    result :
        :class:`FitResult1Star` from :func:`fit_1star_dynesty`.
    gaia_id :
        Object id string (used in filenames).
    out_dir :
        Override root; default :func:`phot_sed_dir`.
    best_pred :
        Optional max-L forward prediction for the summary JSON.

    Returns
    -------
    dict[str, Path]
        Paths for ``summary_json`` and ``samples_npz``.

    Limits
    ------
    Creates ``out_dir`` if missing. Does not write plots (Issue 9).
    """
    root = Path(out_dir).expanduser().resolve() if out_dir is not None else phot_sed_dir()
    root.mkdir(parents=True, exist_ok=True)
    stem = f"Gaia_DR3_{gaia_id}_1star"
    summary_path = root / f"{stem}_summary.json"
    samples_path = root / f"{stem}_samples.npz"

    summary: dict[str, Any] = {
        "schema_version": 1,
        "model": "1star",
        "gaia_id": str(gaia_id),
        "param_names": list(result.param_names),
        "logz": result.logz,
        "logz_err": result.logz_err,
        "bic": result.bic,
        "ln_l_max": result.ln_l_max,
        "best_theta": {n: float(v) for n, v in zip(result.param_names, result.best_theta)},
        "n_data": result.n_data,
        "n_free": result.n_free,
        "nlive": result.nlive,
        "n_samples": int(result.samples.shape[0]),
    }
    if best_pred is not None:
        summary["best_mags"] = {k: float(v) for k, v in best_pred.mags.items()}
        summary["best_mist"] = {
            "teff_k": best_pred.mist.teff_k,
            "logg": best_pred.mist.logg,
            "radius_rsun": best_pred.mist.radius_rsun,
            "age_gyr": best_pred.mist.age_gyr,
            "log_l": best_pred.mist.log_l,
            "mass_current": best_pred.mist.mass_current,
        }
        summary["distance_pc"] = best_pred.distance_pc

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        samples_path,
        samples=result.samples,
        logl=result.logl,
        param_names=np.asarray(result.param_names),
        best_theta=result.best_theta,
        logz=np.asarray([result.logz, result.logz_err], dtype=np.float64),
        bic=np.asarray([result.bic], dtype=np.float64),
    )
    return {"summary_json": summary_path, "samples_npz": samples_path}


def run_1star_fit(
    rows: Sequence[PhotRow],
    *,
    gaia_id: str,
    mist_predictor: MistPredictFn,
    phoenix_grid: PhoenixGrid | None = None,
    synth_phot: SynthPhotFn | None = None,
    bounds: OneStarPriorBounds | None = None,
    out_dir: Path | str | None = None,
    nlive: int = 100,
    maxiter: int | None = None,
    seed: int | None = 42,
    bandpasses: Mapping[str, object] | None = None,
    dlogz: float = 1.0,
    sample: str = "auto",
    bound: str = "multi",
    nworkers: int = 1,
    jit_warmup: bool = True,
    spec_stride: int = 1,
    print_progress: bool = False,
) -> tuple[FitResult1Star, dict[str, Path]]:
    """
    Fit 1-star + write ``output/phot_sed/`` products.

    Parameters
    ----------
    dlogz, sample, bound, nworkers, jit_warmup, spec_stride, print_progress :
        Forwarded to :func:`fit_1star_dynesty`; see that function for details.

    See also :func:`fit_1star_dynesty` and :func:`write_1star_outputs`.
    """
    result = fit_1star_dynesty(
        rows,
        mist_predictor=mist_predictor,
        phoenix_grid=phoenix_grid,
        synth_phot=synth_phot,
        bounds=bounds,
        nlive=nlive,
        maxiter=maxiter,
        seed=seed,
        bandpasses=bandpasses,
        dlogz=dlogz,
        sample=sample,
        bound=bound,
        nworkers=nworkers,
        jit_warmup=jit_warmup,
        spec_stride=spec_stride,
        print_progress=print_progress,
    )
    bands = [r.band for r in rows if r.band not in _GAIA_CONSTRAINT_BANDS]
    bps = bandpasses
    if bps is None and synth_phot is None:
        bps = load_bandpasses_for_bands(bands)
    best_pred = predict_1star_phot(
        result.best_theta[:_SIGMA_INT_IDX],  # physical params only (no sigma_int)
        bands,
        mist_predictor=mist_predictor,
        phoenix_grid=phoenix_grid,
        synth_phot=synth_phot,
        systems=("ab",),
        bandpasses=bps,
    )
    paths = write_1star_outputs(
        result, gaia_id=gaia_id, out_dir=out_dir, best_pred=best_pred
    )
    return result, paths


# ---------------------------------------------------------------------------
# 2-star coeval fit (Issue #31)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TwoStarPriorBounds:
    """
    Uniform prior hypercube for 2-star coeval parameters.

    Parameters
    ----------
    eep1, mass1, mass2, feh, afe, a_v, parallax_mas :
        Inclusive ``(lo, hi)`` bounds in :data:`TWO_STAR_PARAM_NAMES` order.

    Notes
    -----
    ``mass2`` prior is independent uniform; the constraint ``M₁ ≥ M₂`` is
    enforced inside the likelihood (returns ``−∞`` when violated) so dynesty's
    live-point distribution correctly respects the constraint without a
    non-rectangular prior transform.
    """

    eep1: tuple[float, float] = DEFAULT_EEP_BOUNDS
    mass1: tuple[float, float] = DEFAULT_MASS_BOUNDS
    mass2: tuple[float, float] = DEFAULT_MASS_BOUNDS
    feh: tuple[float, float] = DEFAULT_FEH_BOUNDS
    afe: tuple[float, float] = DEFAULT_AFE_BOUNDS
    a_v: tuple[float, float] = DEFAULT_AV_BOUNDS
    parallax_mas: tuple[float, float] = DEFAULT_PARALLAX_BOUNDS

    def as_list(self) -> list[tuple[float, float]]:
        """Return bounds in dynesty parameter order."""
        return [
            self.eep1, self.mass1, self.mass2,
            self.feh, self.afe, self.a_v, self.parallax_mas,
        ]


@dataclass(frozen=True, slots=True)
class FitResult2Star:
    """
    2-star dynesty summary (same structure as :class:`FitResult1Star`).

    Parameters
    ----------
    param_names :
        Free-parameter labels (7 for 2-star).
    samples :
        Equally weighted posterior samples ``(n_samples, 7)``.
    logl :
        Log-likelihood at each nested-sampling dead point.
    logz, logz_err :
        Bayesian evidence ``ln Z`` and its uncertainty.
    bic :
        BIC from the maximum-likelihood nested-sampling point.
    ln_l_max :
        Max log-likelihood among nested-sampling live/dead points.
    best_theta :
        Parameter vector at ``ln_l_max``.
    n_data :
        Number of photometry rows used in the likelihood.
    n_free :
        Number of free parameters (7).
    nlive :
        Nested-sampling live points.
    """

    param_names: tuple[str, ...]
    samples: NDArray[np.float64]
    logl: NDArray[np.float64]
    logz: float
    logz_err: float
    bic: float
    ln_l_max: float
    best_theta: NDArray[np.float64]
    n_data: int
    n_free: int
    nlive: int


def fit_2star_dynesty(
    rows: Sequence[PhotRow],
    *,
    mist_predictor: MistPredictFn,
    phoenix_grid: PhoenixGrid | None = None,
    synth_2star: SynthPhot2StarFn | None = None,
    bounds: TwoStarPriorBounds | None = None,
    nlive: int = 100,
    maxiter: int | None = None,
    seed: int | None = 42,
    bandpasses: Mapping[str, object] | None = None,
    mag_system: str = "ab",
    dlogz: float = 1.0,
    sample: str = "auto",
    bound: str = "multi",
    nworkers: int = 1,
    jit_warmup: bool = True,
    eep2_xtol: float = 0.5,
    spec_stride: int = 1,
    print_progress: bool = False,
) -> FitResult2Star:
    """
    Run dynesty nested sampling on the 2-star coeval Path-2 model.

    Parameters
    ----------
    rows :
        Photometry rows (at least one detection recommended).
    mist_predictor :
        Injectable MISTy ``getMIST`` (required so CI can mock).
    phoenix_grid, synth_2star :
        Forward model; one of them required.
    bounds :
        Uniform prior bounds; default is :class:`TwoStarPriorBounds`.
    nlive, maxiter, seed, dlogz, sample, bound, nworkers, jit_warmup,
    spec_stride, print_progress :
        Dynesty / performance controls; see :func:`fit_1star_dynesty`.
    eep2_xtol :
        EEP tolerance for the coeval solver (passed to
        :func:`~darkhunter_sed.phot_sed_models.solve_eep2_for_age_match`).
        Default 0.5 EEP ≈ sub-0.1 Gyr age residual for typical tracks.

    Returns
    -------
    FitResult2Star
        Posterior samples, ``lnZ``, and BIC from max-L.

    Limits
    ------
    ``M₁ ≥ M₂`` enforced as a ``−∞`` likelihood gate (not a prior boundary).
    No coeval EEP₂ solution → ``−∞`` likelihood (logged, not raised, inside
    dynesty). Uniform priors only.
    """
    import multiprocessing

    from dynesty import NestedSampler

    if not rows:
        raise ValueError("rows must be non-empty")
    prior = bounds if bounds is not None else TwoStarPriorBounds()
    bound_list = prior.as_list()
    bands = [r.band for r in rows]
    ndim = len(TWO_STAR_PARAM_NAMES)

    bps = bandpasses
    if bps is None and synth_2star is None:
        bps = load_bandpasses_for_bands(bands)
    if phoenix_grid is not None and bps is not None:
        from darkhunter_sed.filters_synphot import bandpass_wavelength_grid

        wave_grid = bandpass_wavelength_grid(bps)
        if spec_stride > 1:
            wave_grid = wave_grid[::int(spec_stride)]
        phoenix_grid.set_photometry_wavelengths(wave_grid)

    # JIT warm-up: drive MISTy compilation before sampler allocates live points.
    if jit_warmup:
        _mid = np.array([0.5 * (lo + hi) for lo, hi in bound_list], dtype=np.float64)
        # Ensure mass2 <= mass1 for the warm-up point.
        _mid[2] = min(_mid[2], _mid[1])
        try:
            predict_2star_phot(
                _mid,
                bands,
                mist_predictor=mist_predictor,
                phoenix_grid=phoenix_grid,
                synth_2star=synth_2star,
                systems=("ab",),
                bandpasses=bps,
                mag_system=mag_system,
                eep2_xtol=eep2_xtol,
            )
        except Exception:
            pass

    def prior_transform(u: NDArray[np.floating]) -> NDArray[np.float64]:
        return _unit_cube_to_bounds(u, bound_list)

    def loglike(theta: NDArray[np.floating]) -> float:
        try:
            pred = predict_2star_phot(
                theta,
                bands,
                mist_predictor=mist_predictor,
                phoenix_grid=phoenix_grid,
                synth_2star=synth_2star,
                systems=("ab",),
                bandpasses=bps,
                mag_system=mag_system,
                eep2_xtol=eep2_xtol,
            )
        except Exception:
            return _LOGLIKE_FLOOR
        lnl2 = photometry_loglike(pred.mags, rows)
        return lnl2 if math.isfinite(lnl2) else _LOGLIKE_FLOOR

    pool: Any = None
    queue_size: int | None = None
    if nworkers > 1:
        pool = multiprocessing.Pool(int(nworkers))
        queue_size = int(nworkers)

    try:
        sampler = NestedSampler(
            loglike,
            prior_transform,
            ndim,
            nlive=int(nlive),
            sample=sample,
            bound=bound,
            rstate=np.random.default_rng(seed) if seed is not None else None,
            pool=pool,
            queue_size=queue_size,
        )
        run_kw: dict[str, Any] = {
            "print_progress": bool(print_progress),
            "dlogz": float(dlogz),
        }
        if maxiter is not None:
            run_kw["maxiter"] = int(maxiter)
        sampler.run_nested(**run_kw)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    res = sampler.results
    logl = np.asarray(res.logl, dtype=np.float64)
    samples_u = np.asarray(res.samples, dtype=np.float64)
    imax = int(np.argmax(logl))
    ln_l_max = float(logl[imax])
    best_theta = samples_u[imax].copy()
    n_data = len(rows)
    bic = bic_from_max_likelihood(ln_l_max=ln_l_max, n_free=ndim, n_data=n_data)

    try:
        from dynesty.utils import resample_equal

        weights = np.exp(np.asarray(res.logwt, dtype=np.float64) - float(res.logz[-1]))
        eq = np.asarray(resample_equal(samples_u, weights), dtype=np.float64)
    except Exception:
        eq = samples_u

    logz = float(res.logz[-1])
    logz_err = (
        float(res.logzerr[-1]) if getattr(res, "logzerr", None) is not None else float("nan")
    )

    return FitResult2Star(
        param_names=TWO_STAR_PARAM_NAMES,
        samples=eq,
        logl=logl,
        logz=logz,
        logz_err=logz_err,
        bic=bic,
        ln_l_max=ln_l_max,
        best_theta=best_theta,
        n_data=n_data,
        n_free=ndim,
        nlive=int(nlive),
    )


def write_2star_outputs(
    result: FitResult2Star,
    *,
    gaia_id: str,
    out_dir: Path | str | None = None,
    best_pred: TwoStarPrediction | None = None,
) -> dict[str, Path]:
    """
    Write 2-star fit products under ``output/phot_sed/``.

    Parameters
    ----------
    result :
        :class:`FitResult2Star` from :func:`fit_2star_dynesty`.
    gaia_id :
        Object id string (used in filenames).
    out_dir :
        Override root; default :func:`~darkhunter_sed.config.phot_sed_dir`.
    best_pred :
        Optional max-L forward prediction for the summary JSON.

    Returns
    -------
    dict[str, Path]
        Paths for ``summary_json`` and ``samples_npz``.
    """
    root = Path(out_dir).expanduser().resolve() if out_dir is not None else phot_sed_dir()
    root.mkdir(parents=True, exist_ok=True)
    stem = f"Gaia_DR3_{gaia_id}_2star"
    summary_path = root / f"{stem}_summary.json"
    samples_path = root / f"{stem}_samples.npz"

    summary: dict[str, Any] = {
        "schema_version": 1,
        "model": "2star",
        "gaia_id": str(gaia_id),
        "param_names": list(result.param_names),
        "logz": result.logz,
        "logz_err": result.logz_err,
        "bic": result.bic,
        "ln_l_max": result.ln_l_max,
        "best_theta": {n: float(v) for n, v in zip(result.param_names, result.best_theta)},
        "n_data": result.n_data,
        "n_free": result.n_free,
        "nlive": result.nlive,
        "n_samples": int(result.samples.shape[0]),
    }
    if best_pred is not None:
        summary["best_mags"] = {k: float(v) for k, v in best_pred.mags.items()}
        summary["eep2_solved"] = best_pred.eep2
        summary["distance_pc"] = best_pred.distance_pc
        for star_idx, mist in enumerate((best_pred.mist1, best_pred.mist2), start=1):
            summary[f"best_mist{star_idx}"] = {
                "teff_k": mist.teff_k,
                "logg": mist.logg,
                "radius_rsun": mist.radius_rsun,
                "age_gyr": mist.age_gyr,
                "log_l": mist.log_l,
                "mass_current": mist.mass_current,
            }

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        samples_path,
        samples=result.samples,
        logl=result.logl,
        param_names=np.asarray(result.param_names),
        best_theta=result.best_theta,
        logz=np.asarray([result.logz, result.logz_err], dtype=np.float64),
        bic=np.asarray([result.bic], dtype=np.float64),
    )
    return {"summary_json": summary_path, "samples_npz": samples_path}


def run_2star_fit(
    rows: Sequence[PhotRow],
    *,
    gaia_id: str,
    mist_predictor: MistPredictFn,
    phoenix_grid: PhoenixGrid | None = None,
    synth_2star: SynthPhot2StarFn | None = None,
    bounds: TwoStarPriorBounds | None = None,
    out_dir: Path | str | None = None,
    nlive: int = 100,
    maxiter: int | None = None,
    seed: int | None = 42,
    bandpasses: Mapping[str, object] | None = None,
    dlogz: float = 1.0,
    sample: str = "auto",
    bound: str = "multi",
    nworkers: int = 1,
    jit_warmup: bool = True,
    eep2_xtol: float = 0.5,
    spec_stride: int = 1,
    print_progress: bool = False,
) -> tuple[FitResult2Star, dict[str, Path]]:
    """
    Fit 2-star coeval + write ``output/phot_sed/`` products.

    Parameters
    ----------
    eep2_xtol, spec_stride, print_progress :
        Forwarded to :func:`fit_2star_dynesty`.

    See also :func:`fit_2star_dynesty` and :func:`write_2star_outputs`.
    """
    result = fit_2star_dynesty(
        rows,
        mist_predictor=mist_predictor,
        phoenix_grid=phoenix_grid,
        synth_2star=synth_2star,
        bounds=bounds,
        nlive=nlive,
        maxiter=maxiter,
        seed=seed,
        bandpasses=bandpasses,
        dlogz=dlogz,
        sample=sample,
        bound=bound,
        nworkers=nworkers,
        jit_warmup=jit_warmup,
        eep2_xtol=eep2_xtol,
        spec_stride=spec_stride,
        print_progress=print_progress,
    )
    bands = [r.band for r in rows]
    bps = bandpasses
    if bps is None and synth_2star is None:
        bps = load_bandpasses_for_bands(bands)
    best_pred = predict_2star_phot(
        result.best_theta,
        bands,
        mist_predictor=mist_predictor,
        phoenix_grid=phoenix_grid,
        synth_2star=synth_2star,
        systems=("ab",),
        bandpasses=bps,
        eep2_xtol=eep2_xtol,
    )
    paths = write_2star_outputs(
        result, gaia_id=gaia_id, out_dir=out_dir, best_pred=best_pred
    )
    return result, paths
