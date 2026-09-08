"""
Fitzpatrick (1999) extinction at R_V=3.1 for Path-2 photometry SEDs.

Uses the Astropy-affiliated ``dust_extinction`` F99 model and its
``extinguish`` API (same pattern as the Astropy Learn reddening tutorial).
Apply to the spectrum / system SED **before** synphot magnitude synthesis.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from numpy.typing import NDArray

# Locked Path-2 reddening law.
DEFAULT_R_V = 3.1

# F99 validity: 0.3 <= x <= 10.0 µm^-1 → λ ∈ [1000, 33333.3] Å.
_F99_WAVE_AA_MIN = 1.0e4 / 10.0
_F99_WAVE_AA_MAX = 1.0e4 / 0.3


def f99_model(*, r_v: float = DEFAULT_R_V):
    """
    Construct a Fitzpatrick99 extinction model.

    Parameters
    ----------
    r_v :
        Total-to-selective extinction. Path-2 default is ``3.1``.

    Returns
    -------
    dust_extinction.parameter_averages.F99
        Model instance (callable ``A_λ/A_V``; also has ``extinguish``).

    Limits
    ------
    Requires ``dust_extinction``. ``r_v`` must be within the F99 supported
    range (typically 2–6).
    """
    from dust_extinction.parameter_averages import F99

    if float(r_v) <= 0.0:
        raise ValueError(f"r_v must be positive; got {r_v}")
    return F99(Rv=float(r_v))


def f99_alambda_over_av(
    wave_aa: NDArray[np.floating],
    *,
    r_v: float = DEFAULT_R_V,
) -> NDArray[np.float64]:
    """
    Evaluate Fitzpatrick99 ``A_λ / A_V`` on a wavelength grid.

    Parameters
    ----------
    wave_aa :
        Wavelengths in Angstroms (1-D).
    r_v :
        Total-to-selective extinction. Path-2 default is ``3.1``.

    Returns
    -------
    NDArray[np.float64]
        ``A_λ / A_V`` at each wavelength (same shape as ``wave_aa``).

    Limits
    ------
    F99 is defined for ``1000 Å ≤ λ ≤ 33333.3 Å``. Outside that window this
    function **holds the edge** value (wavelengths clipped before calling the
    model). It does not invent UV/IR wings beyond F99.
    """
    wave = np.asarray(wave_aa, dtype=np.float64)
    if wave.ndim != 1 or wave.size == 0:
        raise ValueError("wave_aa must be a non-empty 1-D array")
    if not np.all(np.isfinite(wave)):
        raise ValueError("wave_aa must be finite")

    wave_clip = np.clip(wave, _F99_WAVE_AA_MIN, _F99_WAVE_AA_MAX)
    ext = f99_model(r_v=r_v)
    al_av = np.asarray(ext(wave_clip * u.AA), dtype=np.float64)
    if al_av.shape != wave.shape:
        raise RuntimeError("F99 returned unexpected shape")
    return al_av


def apply_f99_extinction(
    wave_aa: NDArray[np.floating],
    flux: NDArray[np.floating],
    a_v: float,
    *,
    r_v: float = DEFAULT_R_V,
) -> NDArray[np.float64]:
    """
    Redden a spectrum with Fitzpatrick99 via ``F99.extinguish``.

    Parameters
    ----------
    wave_aa :
        Wavelengths in Angstroms (1-D).
    flux :
        Flux density on ``wave_aa`` (any linear density unit; attenuation is
        multiplicative). Must match ``wave_aa`` shape.
    a_v :
        Visual extinction in magnitudes (``A_V ≥ 0``). ``a_v=0`` returns a
        copy of ``flux``.
    r_v :
        Total-to-selective extinction (Path-2 default ``3.1``).

    Returns
    -------
    NDArray[np.float64]
        Extincted flux: ``flux * F99.extinguish(λ, Av=a_v)``.

    Limits
    ------
    Same F99 wavelength window as :func:`f99_alambda_over_av` (edge hold
    outside). Does not add emission; negative ``a_v`` is rejected. Vectorized
    over the wavelength axis only (single spectrum).
    """
    wave = np.asarray(wave_aa, dtype=np.float64)
    flux_arr = np.asarray(flux, dtype=np.float64)
    if wave.shape != flux_arr.shape or wave.ndim != 1:
        raise ValueError("wave_aa and flux must be 1-D arrays of equal length")
    if not np.isfinite(a_v):
        raise ValueError("a_v must be finite")
    if a_v < 0.0:
        raise ValueError(f"a_v must be >= 0; got {a_v}")
    if a_v == 0.0:
        return flux_arr.copy()

    wave_clip = np.clip(wave, _F99_WAVE_AA_MIN, _F99_WAVE_AA_MAX)
    ext = f99_model(r_v=r_v)
    thru = np.asarray(
        ext.extinguish(wave_clip * u.AA, Av=float(a_v)),
        dtype=np.float64,
    )
    return flux_arr * thru
