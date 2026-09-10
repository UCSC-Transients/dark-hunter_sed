"""
Path-2 dust and astrometric priors for photometry SED fits (dynesty).

Extinction law throughout: **Fitzpatrick (1999) F99, R_V = 3.1**.
Av_SF is the CSFD (Chiang 2023) / S&F (Schlafly & Finkbeiner 2011)
line-of-sight A_V at R_V = 3.1, used as the hard upper bound on A_V.

Prior modes (controlled by :func:`build_prior_spec`):

* **Default** — Av: Uniform[0, Av_SF]; parallax: flat ±5σ clip (Gaia
  parallax enters as a likelihood term in phot_sed_fit).
* ``--prior-dust`` — Av: Truncated-Normal centred on 3D dust map mean/σ,
  hard-capped at Av_SF.
* ``--prior-gaia-vac`` — parallax: Gaussian prior via Gaia astrometry
  (Gaia parallax likelihood term removed to avoid double-counting).
* ``--prior-uberms PATH`` — [Fe/H] and/or mass: Gaussian priors from a
  uberMS-format spectral-fit JSON; **1-star model only** (ignored + warning
  for 2-star/wd).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Fitzpatrick 1999 R_V used throughout Path-2 SED extinction model.
# Consistent with _A_G_PER_AV = 0.789 in phot_sed_fit.py (Gaia G, F99 R_V=3.1).
# Do NOT use for uberMS/UMS spectral fits (those use R_V=3.32 in dust_prior.py).
R_V_F99: float = 3.1

# Default Av upper bound when CSFD is unavailable (deg-only fallback).
_AV_FALLBACK_HI: float = 5.0
# Parallax flat-box half-width in units of observed error.
_PLX_SIGMA_CLIP: float = 5.0
# Hard parallax bounds (mas).
_PLX_BOUNDS: tuple[float, float] = (0.1, 100.0)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AvPriorSpec:
    """A_V prior specification for one dynesty prior_transform slot.

    Parameters
    ----------
    kind :
        ``"flat"`` — Uniform[av_lo, av_hi]; ``"gaussian"`` — Truncated-Normal.
    av_lo :
        Always 0.0.
    av_hi :
        Hard upper bound = Av_SF (CSFD LOS at R_V=3.1 F99).
    av_mean, av_sigma :
        Centre and (inflated) σ of the Truncated-Normal; ``None`` for ``"flat"``.
    map_used :
        Provenance string for logging (e.g. ``"bayestar2019"``).
    """

    kind: str
    av_lo: float
    av_hi: float
    av_mean: float | None = None
    av_sigma: float | None = None
    map_used: str = ""


@dataclass(frozen=True, slots=True)
class PlxPriorSpec:
    """Parallax prior specification.

    Parameters
    ----------
    kind :
        ``"flat"`` — Uniform on [plx_lo, plx_hi];
        ``"gaussian"`` — Truncated-Normal(plx_obs, plx_err) on [plx_lo, plx_hi].
    plx_lo, plx_hi :
        Hard bounds (mas); derived from ±:data:`_PLX_SIGMA_CLIP` × plx_err.
    plx_obs, plx_err :
        Observed Gaia parallax (mas) and its 1-σ error (mas).
    """

    kind: str
    plx_lo: float
    plx_hi: float
    plx_obs: float
    plx_err: float


@dataclass(frozen=True, slots=True)
class UberMSPriorSpec:
    """Spectral-fit (uberMS) priors for 1-star model [Fe/H] and/or mass.

    Parameters
    ----------
    feh_mean, feh_sigma :
        Gaussian centre and σ for [Fe/H]; ``None`` if absent from the JSON.
    mass_mean, mass_sigma :
        Gaussian centre and σ for stellar mass (M☉); ``None`` if absent.
    """

    feh_mean: float | None = None
    feh_sigma: float | None = None
    mass_mean: float | None = None
    mass_sigma: float | None = None


@dataclass(frozen=True, slots=True)
class PhotSedPriorSpec:
    """Full Path-2 prior specification for one dynesty fit.

    Parameters
    ----------
    av :
        A_V prior (flat or Gaussian with CSFD cap).
    plx :
        Parallax prior (flat or Gaussian).
    uberms :
        Spectral-fit priors for 1-star feh/mass; ``None`` for 2-star/wd.
    plx_in_prior :
        ``True`` when ``kind="gaussian"`` so the caller knows to **skip** the
        parallax Gaussian likelihood term (which would otherwise double-count
        the Gaia astrometry).
    """

    av: AvPriorSpec
    plx: PlxPriorSpec
    uberms: UberMSPriorSpec | None = None
    plx_in_prior: bool = False


# ---------------------------------------------------------------------------
# CSFD Av_SF query
# ---------------------------------------------------------------------------


def get_av_sf(
    ra_deg: float,
    dec_deg: float,
    *,
    backend: Any | None = None,
) -> float:
    """CSFD (Chiang 2023) / S&F LOS A_V at R_V = 3.1 (F99).

    Parameters
    ----------
    ra_deg, dec_deg :
        ICRS coordinates (degrees).
    backend :
        Injectable :class:`~darkhunter_sed.dust_prior.DustQueryBackend` for
        tests; ``None`` uses the live :class:`~darkhunter_sed.dust_prior._DustmapsBackend`.

    Returns
    -------
    float
        A_V (mag) at R_V = 3.1 F99; ``math.nan`` on query failure.

    Notes
    -----
    CSFD returns E(B-V); converted here as A_V = R_V_F99 * E(B-V).
    Returned value is the LOS integral (all distance), used as a hard cap.
    """
    from darkhunter_sed.dust_prior import _DustmapsBackend, galactic_lb

    l_deg, b_deg = galactic_lb(ra_deg, dec_deg)
    be: Any = backend if backend is not None else _DustmapsBackend()
    # query_csfd returns A_V already at R_V=3.32 (dust_prior.py convention).
    # We need A_V at R_V=3.1: fetch E(B-V) directly via the CSFD object.
    # The _DustmapsBackend.query_csfd converts ebv → av with RV_DUST=3.32, so
    # we back-convert to E(B-V) and re-apply R_V_F99.
    from darkhunter_sed.dust_prior import RV_DUST  # 3.32

    av_3p32 = be.query_csfd(l_deg, b_deg)
    if av_3p32 is None or not math.isfinite(av_3p32) or av_3p32 < 0:
        return math.nan
    ebv = av_3p32 / RV_DUST
    return R_V_F99 * ebv


# ---------------------------------------------------------------------------
# Per-parameter prior spec builders
# ---------------------------------------------------------------------------


def build_av_prior_spec(
    ra_deg: float,
    dec_deg: float,
    d_pc: float | None,
    *,
    prior_dust: bool = False,
    av_sf: float | None = None,
    backend: Any | None = None,
    av_sigma_inflate: float = 2.0,
    av_hi_fallback: float = _AV_FALLBACK_HI,
) -> AvPriorSpec:
    """Build A_V prior specification for one Path-2 fit.

    Parameters
    ----------
    ra_deg, dec_deg :
        ICRS coordinates for CSFD / 3D dust map queries.
    d_pc :
        Photometric distance estimate (pc) for 3D map lookup; ``None`` skips
        3D queries and falls back to flat CSFD.
    prior_dust :
        ``False`` (default) — flat Uniform[0, Av_SF];
        ``True`` — Truncated-Normal from 3D map with Av_SF cap.
    av_sf :
        Override the CSFD Av_SF value (useful in tests; computed when ``None``).
    av_sigma_inflate :
        3D map σ inflation factor (default 2×).
    av_hi_fallback :
        Upper bound used when CSFD query fails (default 5.0 mag).

    Returns
    -------
    AvPriorSpec
        ``kind="flat"`` or ``kind="gaussian"`` with hard cap ``av_hi = Av_SF``.

    Notes
    -----
    Extinction law: F99, R_V = 3.1 throughout (see :data:`R_V_F99`).
    """
    if av_sf is None:
        av_sf_val = get_av_sf(ra_deg, dec_deg, backend=backend)
        av_sf_val = av_sf_val if math.isfinite(av_sf_val) and av_sf_val > 0 else av_hi_fallback
    else:
        av_sf_val = float(av_sf)
        if not math.isfinite(av_sf_val) or av_sf_val <= 0:
            av_sf_val = av_hi_fallback

    if not prior_dust:
        return AvPriorSpec(kind="flat", av_lo=0.0, av_hi=av_sf_val, map_used="csfd_flat")

    # Gaussian: query 3D maps in priority order, cap at av_sf.
    from darkhunter_sed.dust_prior import (
        _DustmapsBackend,
        galactic_lb,
        in_bayestar_footprint,
        in_chen_footprint,
        in_edenhofer_distance_range,
    )

    l_deg, b_deg = galactic_lb(ra_deg, dec_deg)
    be: Any = backend if backend is not None else _DustmapsBackend()

    pair: tuple[float, float] | None = None
    map_name = ""

    if d_pc is not None and math.isfinite(d_pc) and d_pc > 0:
        if in_bayestar_footprint(dec_deg):
            pair = be.query_bayestar(l_deg, b_deg, d_pc)
            map_name = "bayestar2019"
        if pair is None and in_edenhofer_distance_range(d_pc):
            pair = be.query_edenhofer(l_deg, b_deg, d_pc)
            map_name = "edenhofer2023"
        if pair is None and in_chen_footprint(l_deg, b_deg):
            pair = be.query_chen_3d(l_deg, b_deg, d_pc)
            map_name = "chen2014"

    if pair is not None:
        av_mean_raw, av_sigma_raw = pair
        if math.isfinite(av_mean_raw) and av_mean_raw >= 0:
            av_mean = max(0.0, float(av_mean_raw))
            av_sigma = float(av_sigma_raw) * av_sigma_inflate
            # Hard cap: gaussian truncated at Av_SF
            av_hi = min(av_sf_val, av_mean + 5.0 * av_sigma)
            av_hi = max(av_hi, av_mean)  # hi must exceed mean
            return AvPriorSpec(
                kind="gaussian",
                av_lo=0.0,
                av_hi=av_hi,
                av_mean=av_mean,
                av_sigma=av_sigma,
                map_used=map_name,
            )

    logger.warning(
        "No 3D dust map result for (%.2f, %.2f) d_pc=%s; falling back to flat [0, %.3f]",
        ra_deg,
        dec_deg,
        d_pc,
        av_sf_val,
    )
    return AvPriorSpec(kind="flat", av_lo=0.0, av_hi=av_sf_val, map_used="csfd_flat_fallback")


def build_plx_prior_spec(
    plx_obs: float,
    plx_err: float,
    *,
    prior_gaia_vac: bool = False,
    plx_bounds: tuple[float, float] = _PLX_BOUNDS,
    sigma_clip: float = _PLX_SIGMA_CLIP,
) -> PlxPriorSpec:
    """Build parallax prior specification.

    Parameters
    ----------
    plx_obs, plx_err :
        Observed Gaia parallax (mas) and 1-σ uncertainty (mas).
    prior_gaia_vac :
        ``False`` (default) — flat ±:data:`_PLX_SIGMA_CLIP` σ box; Gaia parallax
        enters as an extra Gaussian likelihood term in phot_sed_fit.
        ``True`` — Gaussian Truncated-Normal prior; Gaia parallax **not** added as
        a separate likelihood term (``PhotSedPriorSpec.plx_in_prior = True``).
    plx_bounds :
        Hard global bounds (mas).
    sigma_clip :
        Width of the flat box (units of plx_err).

    Returns
    -------
    PlxPriorSpec
    """
    lo = max(plx_bounds[0], plx_obs - sigma_clip * plx_err)
    hi = min(plx_bounds[1], plx_obs + sigma_clip * plx_err)
    kind = "gaussian" if prior_gaia_vac else "flat"
    return PlxPriorSpec(kind=kind, plx_lo=lo, plx_hi=hi, plx_obs=plx_obs, plx_err=plx_err)


# ---------------------------------------------------------------------------
# uberMS prior loader
# ---------------------------------------------------------------------------


def load_uberms_prior_spec(path: Path | str) -> UberMSPriorSpec:
    """Load spectral-fit priors from a uberMS-format star-summary JSON.

    Parameters
    ----------
    path :
        Path to a uberMS star-summary JSON (or any dict with ``feh``,
        ``feh_sigma``, ``mass``, ``mass_sigma`` keys).

    Returns
    -------
    UberMSPriorSpec
        Finite ``feh_mean``/``feh_sigma`` and/or ``mass_mean``/``mass_sigma``.
        Non-finite, missing, or non-positive-σ values are set to ``None``.

    Limits
    ------
    Only [Fe/H] and mass are extracted; Teff/logg are not used as Path-2 priors
    here (they are weakly constrained by Gaia GSP-Phot in the likelihood).
    """
    data: dict = json.loads(Path(path).read_text(encoding="utf-8"))

    def _finite_float(val: Any) -> float | None:
        if val is None:
            return None
        try:
            v = float(val)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    feh_mean = _finite_float(data.get("feh"))
    feh_sigma_raw = _finite_float(data.get("feh_sigma"))
    feh_sigma = feh_sigma_raw if feh_sigma_raw is not None and feh_sigma_raw > 0 else None

    mass_mean = _finite_float(data.get("mass"))
    mass_sigma_raw = _finite_float(data.get("mass_sigma"))
    mass_sigma = mass_sigma_raw if mass_sigma_raw is not None and mass_sigma_raw > 0 else None

    return UberMSPriorSpec(
        feh_mean=feh_mean,
        feh_sigma=feh_sigma,
        mass_mean=mass_mean,
        mass_sigma=mass_sigma,
    )


# ---------------------------------------------------------------------------
# Top-level prior spec builder
# ---------------------------------------------------------------------------


def build_prior_spec(
    ra_deg: float,
    dec_deg: float,
    plx_obs: float,
    plx_err: float,
    *,
    prior_dust: bool = False,
    prior_gaia_vac: bool = False,
    uberms_path: Path | str | None = None,
    model: str = "1star",
    d_pc: float | None = None,
    backend: Any | None = None,
) -> PhotSedPriorSpec:
    """Build the full Path-2 prior specification for one fit.

    Parameters
    ----------
    ra_deg, dec_deg :
        ICRS coordinates (degrees); used for CSFD / 3D map queries.
    plx_obs, plx_err :
        Gaia parallax (mas) and 1-σ uncertainty (mas).
    prior_dust :
        Enable 3D Gaussian Av prior with CSFD hard cap.
    prior_gaia_vac :
        Enable Gaussian parallax prior (removes likelihood parallax term).
    uberms_path :
        Path to a uberMS spectral-fit JSON; applied **only** to 1-star model.
        A warning is logged and the spec is set to ``None`` for other models.
    model :
        ``"1star"``, ``"2star"``, or ``"wd"``; gates uberMS application.
    d_pc :
        Distance estimate (pc) for 3D map depth; derived from ``plx_obs`` when
        ``None`` and plx_obs > 0.
    backend :
        Injectable dust backend for tests.

    Returns
    -------
    PhotSedPriorSpec
    """
    if d_pc is None and math.isfinite(plx_obs) and plx_obs > 0:
        d_pc = 1000.0 / plx_obs

    av = build_av_prior_spec(
        ra_deg,
        dec_deg,
        d_pc,
        prior_dust=prior_dust,
        backend=backend,
    )
    plx = build_plx_prior_spec(plx_obs, plx_err, prior_gaia_vac=prior_gaia_vac)

    uberms: UberMSPriorSpec | None = None
    if uberms_path is not None:
        if model != "1star":
            logger.warning(
                "--prior-uberms is only applied to the 1-star model; "
                "ignored for model=%r",
                model,
            )
        else:
            uberms = load_uberms_prior_spec(uberms_path)

    return PhotSedPriorSpec(
        av=av,
        plx=plx,
        uberms=uberms,
        plx_in_prior=prior_gaia_vac,
    )


# ---------------------------------------------------------------------------
# Inverse-CDF helpers (used by phot_sed_fit.prior_transform)
# ---------------------------------------------------------------------------


def unit_to_av(u: float, spec: AvPriorSpec) -> float:
    """Map unit-hypercube ``u ∈ [0, 1]`` to an A_V sample.

    Parameters
    ----------
    u :
        Uniform sample from dynesty.
    spec :
        :class:`AvPriorSpec` with ``kind="flat"`` or ``kind="gaussian"``.

    Returns
    -------
    float
        A_V in magnitudes, guaranteed in ``[spec.av_lo, spec.av_hi]``.

    Notes
    -----
    ``kind="flat"``: linear scaling.
    ``kind="gaussian"``: inverse CDF of Truncated-Normal(mean, sigma) on [lo, hi]
    via :func:`scipy.stats.truncnorm.ppf`.  Falls back to flat when scipy is
    absent or the truncnorm call fails.
    """
    if spec.kind == "flat" or spec.av_mean is None or spec.av_sigma is None:
        return spec.av_lo + u * (spec.av_hi - spec.av_lo)
    return _trunc_normal_icdf(u, spec.av_mean, spec.av_sigma, spec.av_lo, spec.av_hi)


def unit_to_plx(u: float, spec: PlxPriorSpec) -> float:
    """Map unit-hypercube ``u ∈ [0, 1]`` to a parallax sample (mas).

    Parameters
    ----------
    u :
        Uniform sample from dynesty.
    spec :
        :class:`PlxPriorSpec` with ``kind="flat"`` or ``kind="gaussian"``.

    Returns
    -------
    float
        Parallax in mas, guaranteed in ``[spec.plx_lo, spec.plx_hi]``.
    """
    if spec.kind == "flat":
        return spec.plx_lo + u * (spec.plx_hi - spec.plx_lo)
    return _trunc_normal_icdf(u, spec.plx_obs, spec.plx_err, spec.plx_lo, spec.plx_hi)


def unit_to_gaussian_param(
    u: float,
    mu: float,
    sigma: float,
    lo: float,
    hi: float,
) -> float:
    """Map ``u ∈ [0, 1]`` to a Truncated-Normal(mu, sigma) on [lo, hi].

    Parameters
    ----------
    u :
        Dynesty unit-hypercube value.
    mu, sigma :
        Gaussian centre and scale.
    lo, hi :
        Hard truncation bounds.

    Returns
    -------
    float
        Sample clipped to ``[lo, hi]``.
    """
    return _trunc_normal_icdf(u, mu, sigma, lo, hi)


def _trunc_normal_icdf(u: float, mu: float, sigma: float, lo: float, hi: float) -> float:
    """Inverse CDF of TruncatedNormal(mu, sigma, lo, hi).

    Falls back to linear rescaling when scipy is unavailable or σ ≤ 0.
    """
    sigma = float(sigma)
    if sigma <= 0:
        return float(lo) + float(u) * (float(hi) - float(lo))
    try:
        from scipy.stats import truncnorm

        a = (float(lo) - float(mu)) / sigma
        b = (float(hi) - float(mu)) / sigma
        val = float(truncnorm.ppf(float(u), a, b, loc=float(mu), scale=sigma))
        # Clamp to hard bounds (PPF can produce tiny float rounding outside).
        return float(min(max(val, float(lo)), float(hi)))
    except Exception:
        return float(lo) + float(u) * (float(hi) - float(lo))
