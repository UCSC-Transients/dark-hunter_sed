"""
MISTy isochrone / track wrapper for Path-2 photometry SED fits.

Loads ``mistNN`` weights under ``STELLAR_ROOT`` / ``DARKHUNTER_SED_MODELS_DIR``
(via :func:`darkhunter_sed.models.model_paths`) and evaluates
``getMIST(EEP, mass, [Fe/H], [α/Fe])`` → Teff, log g, radius, age, etc.

Limits
------
- Requires an editable MISTy install on ``sys.path`` (``STELLAR_ROOT/MISTy``).
- Default NN file: ``mistNN/mistyNN_2.3_v256_v0.h5`` (LinNet, normed).
- Valid EEP span for this NN is approximately ``[1, 808]``; mass / chem bounds
  are those of the training grid (callers should keep priors inside the grid).
- JAX Metal on some macOS setups is broken; this module sets
  ``JAX_PLATFORMS=cpu`` before importing JAX if unset.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Prefer CPU before any JAX import (Mac Metal backend is often unusable).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Solar radius (IAU 2015 nominal) in cm — convert MISTy ``log(R)`` [R☉] → cm.
R_SUN_CM: float = 6.957e10

# Path-2 / UMS convention for this mistNN.
DEFAULT_EEP_BOUNDS: tuple[float, float] = (1.0, 808.0)
DEFAULT_MASS_BOUNDS: tuple[float, float] = (0.3, 3.0)

MistPredictFn = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class MistPoint:
    """
    One MISTy evaluation at ``(EEP, M_init, [Fe/H], [α/Fe])``.

    Parameters
    ----------
    eep :
        Equivalent evolutionary point (input).
    mass_init :
        Initial mass in solar masses (input).
    feh :
        Initial ``[Fe/H]`` (input).
    afe :
        Initial ``[α/Fe]`` (input).
    teff_k :
        Effective temperature in Kelvin (``10**log(Teff)``).
    logg :
        Surface gravity ``log10(g/[cm s^-2])``.
    radius_rsun :
        Stellar radius in solar radii (``10**log(R)``).
    age_gyr :
        Age in Gyr from ``log(Age)`` (log10 years).
    log_l :
        ``log10(L/L☉)``.
    mass_current :
        Current stellar mass (M☉) from the NN (may differ slightly from init).
    """

    eep: float
    mass_init: float
    feh: float
    afe: float
    teff_k: float
    logg: float
    radius_rsun: float
    age_gyr: float
    log_l: float
    mass_current: float

    @property
    def radius_cm(self) -> float:
        """Radius in cm for geometric dilution with PHOENIX surface fluxes."""
        return float(self.radius_rsun) * R_SUN_CM


def resolve_mist_nn_path(explicit: Path | str | None = None) -> Path:
    """
    Resolve the mistNN HDF5 path.

    Parameters
    ----------
    explicit :
        Optional override. When ``None``, uses :func:`darkhunter_sed.models.model_paths`
        (``STELLAR_ROOT`` / ``DARKHUNTER_SED_MODELS_DIR``).

    Returns
    -------
    Path
        Absolute path to ``mistyNN_*.h5``.

    Limits
    ------
    Does not verify the file exists; :func:`load_misty_predictor` raises on open.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    from darkhunter_sed.models import model_paths

    _, _, mist = model_paths()
    return Path(mist).expanduser().resolve()


def load_misty_predictor(
    mist_nn_path: Path | str | None = None,
    *,
    nntype: str = "LinNet",
    normed: bool = True,
    applyspot: bool = False,
    jit: bool = True,
) -> MistPredictFn:
    """
    Build a callable ``getMIST`` from mistNN weights.

    Parameters
    ----------
    mist_nn_path :
        HDF5 weights; default from :func:`resolve_mist_nn_path`.
    nntype, normed, applyspot :
        Forwarded to ``misty.predict.GenModJax.modpred``.
    jit :
        If ``True``, wrap with ``jax.jit`` (faster repeated calls).

    Returns
    -------
    MistPredictFn
        ``fn(eep=…, mass=…, feh=…, afe=…, verbose=False) → mapping``.

    Limits
    ------
    Imports MISTy + JAX. Prefer injecting a mock callable in unit tests so CI
    does not need mistNN weights or a live MISTy install.
    """
    from darkhunter_sed.models import init_stellar_stack

    init_stellar_stack()
    from jax import jit as jax_jit
    from misty.predict import GenModJax as GenMIST

    path = resolve_mist_nn_path(mist_nn_path)
    if not path.is_file():
        raise FileNotFoundError(f"mistNN weights not found: {path}")
    gm = GenMIST.modpred(
        nnpath=str(path),
        nntype=nntype,
        normed=normed,
        applyspot=applyspot,
    )
    if jit:
        return jax_jit(gm.getMIST)
    return gm.getMIST


def _as_float(pred: Mapping[str, Any], key: str) -> float:
    return float(np.asarray(pred[key]).item())


def evaluate_mist(
    eep: float,
    mass: float,
    feh: float,
    afe: float,
    *,
    predictor: MistPredictFn | None = None,
    mist_nn_path: Path | str | None = None,
) -> MistPoint:
    """
    Evaluate MISTy at one ``(EEP, M, [Fe/H], [α/Fe])`` point.

    Parameters
    ----------
    eep, mass, feh, afe :
        Evolutionary / chemical inputs (see :class:`MistPoint`).
    predictor :
        Optional injectable ``getMIST`` (tests). When ``None``, loads mistNN.
    mist_nn_path :
        Used only when ``predictor`` is ``None``.

    Returns
    -------
    MistPoint
        Derived Teff, log g, radius, age, luminosity.

    Limits
    ------
    Does not clip inputs to the training grid; out-of-grid EEP/mass/chem may
    return nonsense or NaNs. Age uses ``age_Gyr = 10**(log(Age) - 9)``.
    """
    fn = predictor if predictor is not None else load_misty_predictor(mist_nn_path)
    pred = fn(eep=float(eep), mass=float(mass), feh=float(feh), afe=float(afe), verbose=False)
    log_age = _as_float(pred, "log(Age)")
    log_r = _as_float(pred, "log(R)")
    log_teff = _as_float(pred, "log(Teff)")
    return MistPoint(
        eep=float(eep),
        mass_init=float(mass),
        feh=float(feh),
        afe=float(afe),
        teff_k=float(10.0**log_teff),
        logg=_as_float(pred, "log(g)"),
        radius_rsun=float(10.0**log_r),
        age_gyr=float(10.0 ** (log_age - 9.0)),
        log_l=_as_float(pred, "log(L)"),
        mass_current=_as_float(pred, "Mass"),
    )


def evaluate_mist_vectorized(
    eep: NDArray[np.floating],
    mass: NDArray[np.floating],
    feh: NDArray[np.floating],
    afe: NDArray[np.floating],
    *,
    predictor: MistPredictFn | None = None,
    mist_nn_path: Path | str | None = None,
) -> list[MistPoint]:
    """
    Evaluate MISTy at many points (elementwise; not a batched NN call).

    Parameters
    ----------
    eep, mass, feh, afe :
        Equal-length 1-D arrays.
    predictor, mist_nn_path :
        Same as :func:`evaluate_mist`.

    Returns
    -------
    list[MistPoint]
        One point per input row.

    Limits
    ------
    Loops in Python over samples. Prefer a single shared ``predictor`` for
    dynesty likelihoods rather than reloading mistNN each call.
    """
    e = np.asarray(eep, dtype=np.float64).ravel()
    m = np.asarray(mass, dtype=np.float64).ravel()
    f = np.asarray(feh, dtype=np.float64).ravel()
    a = np.asarray(afe, dtype=np.float64).ravel()
    if not (e.size == m.size == f.size == a.size):
        raise ValueError("eep, mass, feh, afe must have the same length")
    fn = predictor if predictor is not None else load_misty_predictor(mist_nn_path)
    return [
        evaluate_mist(float(e[i]), float(m[i]), float(f[i]), float(a[i]), predictor=fn)
        for i in range(e.size)
    ]
