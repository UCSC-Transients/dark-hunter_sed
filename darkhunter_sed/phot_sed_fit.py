"""
Path-2 dynesty nested sampling for photometry SED models.

This issue: **1-star** only. Reports **lnZ** from dynesty and **BIC** from the
maximum-likelihood sample. Writes under ``output/phot_sed/``.
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
    OneStarPrediction,
    SynthPhotFn,
    predict_1star_phot,
)
from darkhunter_sed.phoenix_grid import PhoenixGrid

# Minimal uniform priors for Issue 4 (full phot_sed_priors later).
DEFAULT_FEH_BOUNDS: tuple[float, float] = (-2.0, 0.5)
DEFAULT_AFE_BOUNDS: tuple[float, float] = (-0.2, 0.6)
DEFAULT_AV_BOUNDS: tuple[float, float] = (0.0, 5.0)
DEFAULT_PARALLAX_BOUNDS: tuple[float, float] = (0.1, 100.0)  # mas


@dataclass(frozen=True, slots=True)
class OneStarPriorBounds:
    """
    Uniform prior hypercube for 1-star parameters.

    Parameters
    ----------
    eep, mass, feh, afe, a_v, parallax_mas :
        Inclusive ``(lo, hi)`` bounds in :data:`ONE_STAR_PARAM_NAMES` order.
    """

    eep: tuple[float, float] = DEFAULT_EEP_BOUNDS
    mass: tuple[float, float] = DEFAULT_MASS_BOUNDS
    feh: tuple[float, float] = DEFAULT_FEH_BOUNDS
    afe: tuple[float, float] = DEFAULT_AFE_BOUNDS
    a_v: tuple[float, float] = DEFAULT_AV_BOUNDS
    parallax_mas: tuple[float, float] = DEFAULT_PARALLAX_BOUNDS

    def as_list(self) -> list[tuple[float, float]]:
        """Return bounds in dynesty parameter order."""
        return [self.eep, self.mass, self.feh, self.afe, self.a_v, self.parallax_mas]


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
) -> float:
    """
    Gaussian photometry log-likelihood with 3σ upper-limit treatment.

    Parameters
    ----------
    pred_mags :
        Model magnitudes keyed by band name.
    rows :
        Observed :class:`PhotRow` list (detections and/or ULs).

    Returns
    -------
    float
        Sum of per-band contributions (natural log).

    Limits
    ------
    - Detections (``flag=0``): ``-0.5 * ((m_obs - m_mod)/err)^2 - ln(err*sqrt(2π))``.
    - Upper limits (``flag=1``): if ``m_mod < m_ul`` (model brighter than limit),
      apply the same Gaussian penalty vs ``m_ul``; otherwise contribute ``0``.
    - Missing model bands raise ``KeyError``.
    """
    ln_norm = math.log(math.sqrt(2.0 * math.pi))
    total = 0.0
    for row in rows:
        m_mod = float(pred_mags[row.band])
        err = float(row.err)
        if err <= 0.0 or not math.isfinite(err):
            raise ValueError(f"Invalid err for band {row.band}")
        if row.flag == FLAG_DETECTION:
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
    dlogz: float = 0.5,
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
    bandpasses, mag_system :
        Photometry system / injectable thruputs.

    Returns
    -------
    FitResult1Star
        Posterior samples, ``lnZ``, and BIC from max-L.

    Limits
    ------
    Uniform priors only (Path-2 ``phot_sed_priors`` not wired yet). Failed
    forward-model evaluations return ``-inf`` likelihood.
    """
    from dynesty import NestedSampler

    if not rows:
        raise ValueError("rows must be non-empty")
    prior = bounds if bounds is not None else OneStarPriorBounds()
    bound_list = prior.as_list()
    bands = [r.band for r in rows]
    ndim = len(ONE_STAR_PARAM_NAMES)

    def prior_transform(u: NDArray[np.floating]) -> NDArray[np.float64]:
        return _unit_cube_to_bounds(u, bound_list)

    def loglike(theta: NDArray[np.floating]) -> float:
        try:
            pred = predict_1star_phot(
                theta,
                bands,
                mist_predictor=mist_predictor,
                phoenix_grid=phoenix_grid,
                synth_phot=synth_phot,
                bandpasses=bandpasses,
                mag_system=mag_system,
            )
        except Exception:
            return -np.inf
        return photometry_loglike(pred.mags, rows)

    sampler = NestedSampler(
        loglike,
        prior_transform,
        ndim,
        nlive=int(nlive),
        rstate=np.random.default_rng(seed) if seed is not None else None,
    )
    run_kw: dict[str, Any] = {"print_progress": False, "dlogz": float(dlogz)}
    if maxiter is not None:
        run_kw["maxiter"] = int(maxiter)
    sampler.run_nested(**run_kw)
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
        param_names=ONE_STAR_PARAM_NAMES,
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
) -> tuple[FitResult1Star, dict[str, Path]]:
    """
    Fit 1-star + write ``output/phot_sed/`` products.

    See :func:`fit_1star_dynesty` and :func:`write_1star_outputs`.
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
    )
    bands = [r.band for r in rows]
    best_pred = predict_1star_phot(
        result.best_theta,
        bands,
        mist_predictor=mist_predictor,
        phoenix_grid=phoenix_grid,
        synth_phot=synth_phot,
        bandpasses=bandpasses,
    )
    paths = write_1star_outputs(
        result, gaia_id=gaia_id, out_dir=out_dir, best_pred=best_pred
    )
    return result, paths
