"""Gaia-informed SVI starting values (init only; priors stay in fit.py)."""

from __future__ import annotations

import logging
import math
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EEP = 300.0
DEFAULT_MASS = 1.0
DEFAULT_FEH = 0.0
TEFF_WARN_DELTA_K = 200.0


def _finite_float(val, default: float = float("nan")) -> float:
    try:
        x = float(val)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _load_gen_mist(mist_nn_path: str):
    from jax import jit
    from misty.predict import GenModJax as GenMIST

    gm = GenMIST.modpred(nnpath=mist_nn_path, nntype="LinNet", normed=True)
    return jit(gm.getMIST)


def eep_from_age_on_grid(
    gen_mist_fn: Callable,
    age_gyr: float,
    mass: float,
    feh: float,
    afe: float,
    eep_grid: np.ndarray,
) -> float:
    """
    Interpolate EEP at fixed (mass, feh, afe) for target age (Gyr).

    MIST ``log(Age)`` is log10(years); ages along the grid are converted to Gyr
    like ``MISTy/misty/utils/mkiso.py``.
    """
    ages_gyr: list[float] = []
    for eep in eep_grid:
        pred = gen_mist_fn(eep=float(eep), mass=float(mass), feh=float(feh), afe=float(afe), verbose=False)
        log_age = float(pred["log(Age)"])
        ages_gyr.append(10.0 ** (log_age - 9.0))
    ages = np.asarray(ages_gyr, dtype=float)
    order = np.argsort(ages)
    ages_s = ages[order]
    eep_s = np.asarray(eep_grid, dtype=float)[order]
    valid = np.isfinite(ages_s)
    if valid.sum() < 2:
        return float("nan")
    return float(np.interp(float(age_gyr), ages_s[valid], eep_s[valid], left=np.nan, right=np.nan))


def guess_eep_from_age(
    mist_nn_path: str,
    age_gyr: float,
    mass: float,
    feh: float,
    afe: float,
    *,
    eep_bounds: tuple[float, float] = (250.0, 500.0),
    n_eep: int = 50,
    gen_mist_fn: Callable | None = None,
) -> float:
    """Map Gaia FLAME age (Gyr) to EEP via MIST; clip to ``eep_bounds``."""
    age = _finite_float(age_gyr)
    if not math.isfinite(age) or age <= 0.0:
        return DEFAULT_EEP
    lo, hi = float(eep_bounds[0]), float(eep_bounds[1])
    grid = np.linspace(lo, hi, max(int(n_eep), 4))
    fn = gen_mist_fn if gen_mist_fn is not None else _load_gen_mist(mist_nn_path)
    eep = eep_from_age_on_grid(fn, age, mass, feh, afe, grid)
    if not math.isfinite(eep):
        logger.warning("Could not interpolate EEP from age=%.4f Gyr; using EEP=%s", age, DEFAULT_EEP)
        return DEFAULT_EEP
    return float(np.clip(eep, lo, hi))


def mist_teff_kelvin(
    mist_nn_path: str,
    eep: float,
    mass: float,
    feh: float,
    afe: float,
    *,
    gen_mist_fn: Callable | None = None,
) -> float:
    fn = gen_mist_fn if gen_mist_fn is not None else _load_gen_mist(mist_nn_path)
    pred = fn(eep=float(eep), mass=float(mass), feh=float(feh), afe=float(afe), verbose=False)
    return float(10.0 ** float(pred["log(Teff)"]))


def build_ums_initpars(
    fit_data: dict,
    mist_nn_path: str,
    *,
    dist_init: float,
    fix_photjitter: bool,
    fix_specjitter: bool,
    dospec: bool,
    nspec: int,
) -> dict[str, float]:
    """
    Starting values for uberMS UMS ``init_to_value``.

    Gaia GSP-Phot / FLAME inform init only; sampling priors are set separately in fit.py.
    """
    feh = _finite_float(fit_data.get("[Fe/H]"), DEFAULT_FEH)
    if not math.isfinite(feh):
        feh = DEFAULT_FEH

    mass = _finite_float(fit_data.get("Mass"), DEFAULT_MASS)
    if not math.isfinite(mass) or mass <= 0.0:
        mass = DEFAULT_MASS

    afe = _finite_float(fit_data.get("[a/Fe]"), -0.2)

    age_gyr = _finite_float(fit_data.get("Age_Gyr"))
    eep = guess_eep_from_age(mist_nn_path, age_gyr, mass, feh, afe)

    gaia_teff = _finite_float(fit_data.get("Teff"))
    if math.isfinite(gaia_teff):
        try:
            mist_teff = mist_teff_kelvin(mist_nn_path, eep, mass, feh, afe)
            delta = abs(mist_teff - gaia_teff)
            logger.info(
                "UMS init: EEP=%.2f Mass=%.3f [Fe/H]=%.3f | MIST Teff=%.0f K vs Gaia Teff=%.0f K (Δ=%.0f K)",
                eep,
                mass,
                feh,
                mist_teff,
                gaia_teff,
                delta,
            )
            if delta > TEFF_WARN_DELTA_K:
                logger.warning(
                    "MIST Teff at init differs from Gaia by %.0f K (>%.0f); "
                    "priors are unchanged — SVI may need more steps",
                    delta,
                    TEFF_WARN_DELTA_K,
                )
        except Exception as exc:
            logger.warning("Could not evaluate MIST Teff at init: %s", exc)

    initpars: dict[str, float] = {
        "EEP": eep,
        "initial_Mass": mass,
        "initial_[Fe/H]": feh,
        "initial_[a/Fe]": afe,
        "dist": float(dist_init),
        "Av": 0.1,
        "vmic": 1.0,
        "vstar": 8.0,
        "photjitter": 1e-4 if not fix_photjitter else 0.0,
    }

    if dospec and nspec > 0:
        sj0 = 1e-4 if not fix_specjitter else 0.0
        for ii in range(nspec):
            initpars[f"lsf_{ii}"] = 60000.0
            initpars[f"specjitter_{ii}"] = sj0
            initpars[f"pc0_{ii}"] = 1.0
            initpars[f"pc1_{ii}"] = 0.0
            initpars[f"pc2_{ii}"] = 0.0
            initpars[f"pc3_{ii}"] = 0.0

    return initpars
