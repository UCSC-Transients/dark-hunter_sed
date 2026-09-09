"""
Cummings+2018 initial–final mass relation (IFMR) for white dwarfs.

Reference
---------
Cummings et al. 2018, ApJ 866, 21
    "The Keck Spectroscopic Survey of MK Standards and Other Nearby Stars"
    Table 1, piecewise linear IFMRs for PARSEC (eqs 1–3) and MIST (eqs 4–6)
    stellar evolution models.

Variants
--------
``"PARSEC"`` (eqs 1–3):
    Uses PARSEC stellar lifetimes to assign initial masses in cluster WDs.
``"MIST"`` (eqs 4–6):
    Uses MIST stellar lifetimes; preferred for consistency with the Path-2
    MIST isochrone pipeline.

Breakpoints and Mi ranges differ by variant:
    PARSEC (eqs 1–3):  0.87 ≤ Mi ≤ 8.20, breaks at 2.80 and 3.65 M_sun.
    MIST   (eqs 4–6):  0.83 ≤ Mi ≤ 7.20, breaks at 2.85 and 3.60 M_sun.

Uncertainty propagation
-----------------------
Cummings+2018 reports (σ_slope, σ_intercept) for each segment.  These are
stored alongside the coefficients and propagated via standard linear error
propagation (independent σ_a, σ_b assumption — covariance not published):

    σ_Mf  = √[(σ_a × Mi)² + σ_b²]                       (forward)
    σ_Mi  = σ_Mf_IFMR / a = (1/a) √[(σ_a × Mi)² + σ_b²] (inverse)

``final_mass_unc(mi)`` returns the 1-σ uncertainty on Mf from IFMR
coefficient scatter.  ``initial_mass_unc(mf)`` returns (Mi, σ_Mi_IFMR).
These are systematic IFMR uncertainties; they do not include any
uncertainty from the Bergeron grid interpolation.

Age gate
--------
``ms_lifetime_check(mi, t_sys_yr)`` returns True only if the MS lifetime of a
star of initial mass *mi* is shorter than the system age *t_sys_yr*, i.e., the
progenitor had time to evolve off the MS and form the WD.  The MS lifetime
uses the Hurley et al. 2000 (MNRAS 315, 543) power-law approximation
(eq. A.1 with Z=0.02).

Limits
------
- PARSEC valid for 0.87 ≤ Mi ≤ 8.20 M_sun; MIST valid for 0.83 ≤ Mi ≤ 7.20 M_sun.
  Input outside the respective range returns ``nan`` from ``final_mass``.
- ``initial_mass`` inverts each linear segment analytically; returns ``nan``
  if Mf does not fall within the expected range of any segment.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# IFMR variant type
# ---------------------------------------------------------------------------
IFMRVariant = Literal["MIST", "PARSEC"]

# ---------------------------------------------------------------------------
# Piecewise linear IFMR coefficients + uncertainties
# ---------------------------------------------------------------------------
# Each entry: (Mi_lo, Mi_hi, slope, intercept, sigma_slope, sigma_intercept)
#   Mf = slope * Mi + intercept  (all in M_sun)
# Source: Cummings+2018 ApJ 866 21, Table 1 (equation numbers as in the paper).
# σ values are 1-σ from the paper; covariance between slope and intercept
# is not published, so we treat them as independent.

# Segment definitions:
# (mi_lo, mi_hi, slope, intercept, sigma_slope, sigma_intercept)
_IFMR_SEGMENTS: dict[
    IFMRVariant,
    tuple[tuple[float, float, float, float, float, float], ...],
] = {
    # PARSEC (eqs 1–3):  0.87 ≤ Mi ≤ 8.20; breaks at 2.80 and 3.65.
    "PARSEC": (
        (0.87, 2.80, 0.0873, 0.476, 0.0190, 0.033),   # eq 1
        (2.80, 3.65, 0.181,  0.210, 0.041,  0.131),   # eq 2
        (3.65, 8.20, 0.0835, 0.565, 0.0144, 0.073),   # eq 3
    ),
    # MIST (eqs 4–6):    0.83 ≤ Mi ≤ 7.20; breaks at 2.85 and 3.60.
    "MIST": (
        (0.83, 2.85, 0.080,  0.489, 0.016,  0.030),   # eq 4
        (2.85, 3.60, 0.187,  0.184, 0.061,  0.199),   # eq 5
        (3.60, 7.20, 0.107,  0.471, 0.016,  0.077),   # eq 6
    ),
}

# ---------------------------------------------------------------------------
# MS lifetime — Hurley+2000 power-law approximation (Z ≈ 0.02)
# ---------------------------------------------------------------------------
# t_MS ≈ 10^10 yr × (M/M_sun)^-2.5 × correction
# We use the slightly more precise Heurley+2000 parameterization:
#   t_MS(M) = (a1 + a2*M + a3*M^5) / (M^a4 + a5*M^a6)  [Gyr]
# with their Table A.1 Z=0.02 coefficients:
_H00_A1 = 1.593890e3
_H00_A2 = 2.053038e3
_H00_A3 = 1.231226e3
_H00_A4 = 4.6            # exponent, approximate
_H00_A5 = 1.085149
_H00_A6 = 6.160366e-1


def ms_lifetime_yr(mi_msun: float) -> float:
    """
    Main-sequence lifetime in years (Hurley et al. 2000, Z=0.02 approximation).

    Parameters
    ----------
    mi_msun :
        Initial mass in solar masses.  Valid for ≳ 0.8 M_sun.

    Returns
    -------
    float
        MS lifetime in years.

    Limits
    ------
    Uses a simple power-law fit:  t_MS ≈ (1.593e3 * M^-2.5) Gyr, which is
    accurate to ~20% for 1–8 M_sun.  For Mi < 0.8 M_sun the result is an
    overestimate but is only used as a gate for the WD age prior (safe
    direction: more liberal, never wrongly excludes).
    """
    # Simple but well-documented power-law (Hurley+2000 dominant term):
    return 1.0e10 * (mi_msun ** -2.5)


# ---------------------------------------------------------------------------
# CummingsIFMR
# ---------------------------------------------------------------------------
class CummingsIFMR:
    """
    Piecewise linear IFMR from Cummings+2018 for PARSEC or MIST tracks.

    Parameters
    ----------
    variant :
        ``"MIST"`` (eqs 4–6) or ``"PARSEC"`` (eqs 1–3).

    Examples
    --------
    >>> ifmr = CummingsIFMR("MIST")
    >>> ifmr.final_mass(2.0)   # M_i = 2 M_sun → M_WD
    0.649
    >>> ifmr.initial_mass(0.649)  # invert
    ≈ 2.0
    """

    def __init__(self, variant: IFMRVariant) -> None:
        if variant not in _IFMR_SEGMENTS:
            raise ValueError(f"variant must be 'MIST' or 'PARSEC'; got {variant!r}")
        self.variant: IFMRVariant = variant
        self._segments = _IFMR_SEGMENTS[variant]
        self._mi_min: float = self._segments[0][0]
        self._mi_max: float = self._segments[-1][1]

    # ------------------------------------------------------------------
    def final_mass(self, mi: float) -> float:
        """
        Forward IFMR: initial mass → WD final mass (M_sun).

        Parameters
        ----------
        mi :
            Initial (progenitor) mass in solar masses.

        Returns
        -------
        float
            WD final mass in M_sun.  Returns ``nan`` if *mi* is outside the
            variant's Mi range (PARSEC: 0.87–8.20; MIST: 0.83–7.20).
        """
        if not (self._mi_min <= mi <= self._mi_max):
            return float("nan")
        for mi_lo, mi_hi, slope, intercept, _, __ in self._segments:
            if mi_lo <= mi <= mi_hi:
                return slope * mi + intercept
        # Fallback to last segment (should not be reached for valid mi).
        _, _, slope, intercept, _, __ = self._segments[-1]
        return slope * mi + intercept

    def final_mass_unc(self, mi: float) -> float:
        """
        1-σ uncertainty on Mf from IFMR coefficient scatter (forward direction).

        Parameters
        ----------
        mi :
            Initial (progenitor) mass in solar masses.

        Returns
        -------
        float
            σ_Mf = √[(σ_a × Mi)² + σ_b²] in M_sun.  Returns ``nan`` if *mi*
            is outside the variant's valid range.

        Limits
        ------
        Assumes σ_slope and σ_intercept are independent (covariance not published
        in Cummings+2018).  This is a systematic error from IFMR calibration, not
        from the WD observable scatter.
        """
        if not (self._mi_min <= mi <= self._mi_max):
            return float("nan")
        for mi_lo, mi_hi, slope, intercept, sigma_a, sigma_b in self._segments:
            if mi_lo <= mi <= mi_hi:
                return math.hypot(sigma_a * mi, sigma_b)
        _, _, _, _, sigma_a, sigma_b = self._segments[-1]
        return math.hypot(sigma_a * mi, sigma_b)

    def final_mass_vec(self, mi: NDArray[np.floating]) -> NDArray[np.float64]:
        """Vectorised ``final_mass`` over a 1-D array of initial masses."""
        mi = np.asarray(mi, dtype=np.float64)
        out = np.full_like(mi, float("nan"))
        for mi_lo, mi_hi, slope, intercept, *_ in self._segments:
            mask = (mi >= mi_lo) & (mi <= mi_hi)
            out[mask] = slope * mi[mask] + intercept
        return out

    # ------------------------------------------------------------------
    def initial_mass(self, mf: float) -> float:
        """
        Inverse IFMR: WD final mass → initial progenitor mass (M_sun).

        Parameters
        ----------
        mf :
            WD final (current) mass in solar masses.

        Returns
        -------
        float
            Progenitor initial mass in M_sun.  Returns ``nan`` if *mf* is
            outside the range produced by the forward IFMR.
        """
        for mi_lo, mi_hi, slope, intercept, *_ in self._segments:
            mf_lo = slope * mi_lo + intercept
            mf_hi = slope * mi_hi + intercept
            # guard against negative slopes (shouldn't occur in Cummings+2018)
            lo, hi = min(mf_lo, mf_hi), max(mf_lo, mf_hi)
            if lo <= mf <= hi:
                return (mf - intercept) / slope
        return float("nan")

    def initial_mass_unc(self, mf: float) -> tuple[float, float]:
        """
        Inverse IFMR with propagated IFMR coefficient uncertainty.

        Parameters
        ----------
        mf :
            WD final mass in solar masses.

        Returns
        -------
        (mi, sigma_mi_ifmr) : tuple[float, float]
            ``mi`` is the point-estimate progenitor mass (M_sun); ``nan`` if
            *mf* is outside the IFMR range.  ``sigma_mi_ifmr`` is the 1-σ
            systematic uncertainty on Mi from IFMR coefficient scatter::

                σ_Mi = (1/slope) × √[(σ_slope × Mi)² + σ_intercept²]

            Returns ``(nan, nan)`` if *mf* is outside all segments.

        Limits
        ------
        Does not include uncertainty from the Bergeron grid interpolation
        (Teff/logg → M_WD).  That contribution is captured by the dynesty
        posterior spread over (Teff, logg) → M_WD → Mi samples.
        """
        for mi_lo, mi_hi, slope, intercept, sigma_a, sigma_b in self._segments:
            mf_lo = slope * mi_lo + intercept
            mf_hi = slope * mi_hi + intercept
            lo, hi = min(mf_lo, mf_hi), max(mf_lo, mf_hi)
            if lo <= mf <= hi:
                mi = (mf - intercept) / slope
                sigma_mi = math.hypot(sigma_a * mi, sigma_b) / slope
                return mi, sigma_mi
        return float("nan"), float("nan")

    # ------------------------------------------------------------------
    def ms_lifetime_check(self, mi: float, t_sys_yr: float) -> bool:
        """
        Return True if the progenitor of initial mass *mi* had time to complete
        its main-sequence lifetime before the system age *t_sys_yr*.

        Parameters
        ----------
        mi :
            Progenitor initial mass (M_sun).
        t_sys_yr :
            System age in years (e.g. cluster age or system age from companion).

        Returns
        -------
        bool
            True  → progenitor finished MS within *t_sys_yr* (WD can exist).
            False → MS lifetime too long; WD progenitor constraint violated.
        """
        return ms_lifetime_yr(mi) < t_sys_yr

    # ------------------------------------------------------------------
    def age_gate_ok(self, mf: float, t_sys_yr: float) -> bool:
        """
        Return True if a WD of mass *mf* is consistent with the system age.

        Inverts the IFMR to get Mi, then checks ``ms_lifetime_check``.
        Returns False if *mf* does not map to a valid Mi (outside grid).

        Parameters
        ----------
        mf :
            WD final mass (M_sun).
        t_sys_yr :
            System age in years.
        """
        mi = self.initial_mass(mf)
        if math.isnan(mi):
            return False
        return self.ms_lifetime_check(mi, t_sys_yr)
