"""
Bergeron DA/DB white-dwarf atmosphere grid — parser, interpolator, and
distance+extinction synthesizer for Path-2 photometry SED (``--model wd``).

Data source
-----------
Bergeron atmosphere grids (Bergeron, Wesemael & Beauchamp 1995; Holberg &
Bergeron 2006; Kowalski & Saumon 2006; Bergeron et al. 2011; updated 2021):

    /path/to/stellar/wd/Table_DA  – pure-hydrogen (DA) global grid
    /path/to/stellar/wd/Table_DB  – pure-helium  (DB) global grid

Each file has two header lines, then rows of::

    Teff  log g  M/Mo  Mbol  BC  <44 filter columns>  Age

The grid covers five log g values (7.0, 7.5, 8.0, 8.5, 9.0) at 61 Teff
points (DA) or 72 Teff points (DB).  Magnitudes are **absolute** (at 10 pc).

Extrapolation
-------------
The grid max mass is 1.3 M_sun (log g ≈ 9.0 → M ~ 1.24 M_sun at high Teff).
When the interpolated M_WD at a requested (Teff, log g) exceeds 1.3 M_sun the
``BergeronGrid.synth_phot`` call sets ``extrap_mass=True`` in the returned
dict.  The interpolation itself proceeds (extrapolated in log g above 9.0).

Band-name mapping
-----------------
Bergeron column names are mapped to the project's canonical ``*_phot.fits``
band-name strings::

    G3 / G3_BP / G3_RP → GaiaDR3_G / GaiaDR3_BP / GaiaDR3_RP
    G2 / G2_BP / G2_RP → GaiaDR2_G / GaiaDR2_BP / GaiaDR2_RP
    FUV / NUV          → GALEX_FUV / GALEX_NUV
    W1 / W2 / W3 / W4  → WISE_W1 / WISE_W2 / WISE_W3 / WISE_W4
    J / H / Ks         → 2MASS_J / 2MASS_H / 2MASS_Ks  (first triplet)
    u/g/r/i/z (SDSS)   → SDSS_u / SDSS_g / SDSS_r / SDSS_i / SDSS_z
    g/r/i/z/y (PS1)    → PS_g / PS_r / PS_i / PS_z / PS_y

Extinction
----------
F99 extinction at effective wavelength λ_eff (Å) of each filter, tabulated
here.  Magnitude increment: Δm = A_V × (A_λ / A_V)(λ_eff).  Wavelengths
outside the F99 range (< 1000 Å or > 33333 Å) are clamped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from darkhunter_sed.extinction_f99 import (
    DEFAULT_R_V,
    _F99_WAVE_AA_MAX,
    _F99_WAVE_AA_MIN,
    f99_alambda_over_av,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
AtmType = Literal["DA", "DB"]

# ---------------------------------------------------------------------------
# Band-name registry
# ---------------------------------------------------------------------------
# (column_index_in_global_table, project_band_name, effective_wavelength_Å)
# Column indices are 0-based from the full data array (after skipping 2 header lines).
# Table_DA col layout (44 cols):
#   0:Teff  1:logg  2:M/Mo  3:Mbol  4:BC
#   5:U 6:B 7:V 8:R 9:I                        (Johnson)
#   10:J 11:H 12:Ks 13:Y  14:J 15:H 16:K       (2MASS/CIT; J,H,Ks are 2MASS)
#   17:W1 18:W2 19:W3 20:W4                     (WISE)
#   21:S3.6 22:S4.5 23:S5.8 24:S8.0             (Spitzer – not mapped)
#   25:u 26:g 27:r 28:i 29:z                    (SDSS)
#   30:g 31:r 32:i 33:z 34:y                    (PS1)
#   35:G2 36:G2_BP 37:G2_RP                     (Gaia DR2)
#   38:G3 39:G3_BP 40:G3_RP                     (Gaia DR3)
#   41:FUV 42:NUV                               (GALEX)
#   43:Age
_BAND_REGISTRY_DA: tuple[tuple[int, str, float], ...] = (
    ( 5, "Johnson_U",    3650.0),
    ( 6, "Johnson_B",    4380.0),
    ( 7, "Johnson_V",    5450.0),
    ( 8, "Johnson_R",    6400.0),
    ( 9, "Johnson_I",    8060.0),
    (10, "2MASS_J",     12200.0),
    (11, "2MASS_H",     16300.0),
    (12, "2MASS_Ks",    21900.0),
    (17, "WISE_W1",     33500.0),
    (18, "WISE_W2",     46000.0),
    (19, "WISE_W3",    115608.0),
    (20, "WISE_W4",    220883.0),
    (25, "SDSS_u",      3550.0),
    (26, "SDSS_g",      4770.0),
    (27, "SDSS_r",      6230.0),
    (28, "SDSS_i",      7630.0),
    (29, "SDSS_z",      9130.0),
    (30, "PS_g",        4810.0),
    (31, "PS_r",        6170.0),
    (32, "PS_i",        7520.0),
    (33, "PS_z",        8660.0),
    (34, "PS_y",        9620.0),
    (35, "GaiaDR2_G",   6230.0),
    (36, "GaiaDR2_BP",  5050.0),
    (37, "GaiaDR2_RP",  7730.0),
    (38, "GaiaDR3_G",   6230.0),
    (39, "GaiaDR3_BP",  5050.0),
    (40, "GaiaDR3_RP",  7730.0),
    (41, "GALEX_FUV",   1530.0),
    (42, "GALEX_NUV",   2310.0),
)

# Table_DB has the same column layout as Table_DA — confirmed by inspection.
_BAND_REGISTRY_DB = _BAND_REGISTRY_DA

# Pre-compute extinction ratios A_lambda/A_V for each band at the clamped λ_eff.
def _make_ext_ratios(
    registry: tuple[tuple[int, str, float], ...],
) -> dict[str, float]:
    """Return {band_name: A_lambda/A_V} for every entry in *registry*."""
    waves = np.array(
        [max(_F99_WAVE_AA_MIN, min(_F99_WAVE_AA_MAX, entry[2])) for entry in registry],
        dtype=np.float64,
    )
    ratios = f99_alambda_over_av(waves)
    return {entry[1]: float(r) for entry, r in zip(registry, ratios)}


# ---------------------------------------------------------------------------
# Grid parsing helpers
# ---------------------------------------------------------------------------
_LOGG_GRID: NDArray[np.float64] = np.array([7.0, 7.5, 8.0, 8.5, 9.0])
_LOGG_MAX_BERGERON = 9.0
_MASS_MAX_BERGERON = 1.3  # M_sun; above this, synth_phot sets extrap_mass=True


def _parse_global_table(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Parse a Bergeron global atmosphere table (Table_DA or Table_DB).

    Parameters
    ----------
    path :
        Absolute path to the table file.

    Returns
    -------
    teff_grid : shape (n_teff,)
        Sorted unique Teff values from the table (K).
    data : shape (n_logg, n_teff, n_col)
        All numeric columns except Teff and log g.  Axis 0 indexes log g in
        ascending order matching ``_LOGG_GRID``; axis 1 indexes Teff in
        ascending order; axis 2 indexes the remaining columns (M/Mo at col 0,
        then magnitudes, Age last).

    Limits
    ------
    Requires exactly the five log g values in ``_LOGG_GRID``.  Raises
    ``ValueError`` if the file structure does not match.
    """
    raw = np.loadtxt(path, skiprows=2)
    logg_vals = raw[:, 1]
    teff_vals = raw[:, 0]

    unique_logg = np.unique(logg_vals)
    if unique_logg.shape != _LOGG_GRID.shape or not np.allclose(unique_logg, _LOGG_GRID):
        raise ValueError(
            f"{path}: expected log g values {_LOGG_GRID.tolist()}, "
            f"got {unique_logg.tolist()}"
        )

    # Build Teff grid from one log g slice (all slices share the same Teff grid).
    mask0 = np.abs(logg_vals - _LOGG_GRID[0]) < 1e-4
    teff_grid = np.sort(teff_vals[mask0])
    n_teff = len(teff_grid)
    n_col = raw.shape[1] - 2  # subtract Teff and log g columns

    data = np.empty((len(_LOGG_GRID), n_teff, n_col), dtype=np.float64)
    for i_g, lg in enumerate(_LOGG_GRID):
        mask = np.abs(logg_vals - lg) < 1e-4
        rows = raw[mask]
        teff_order = np.argsort(rows[:, 0])
        rows = rows[teff_order]
        if len(rows) != n_teff:
            raise ValueError(
                f"{path}: log g={lg} has {len(rows)} Teff points, expected {n_teff}"
            )
        # columns 2..end (exclude Teff and log g)
        data[i_g] = rows[:, 2:]

    return teff_grid, data


# ---------------------------------------------------------------------------
# BergeronGrid
# ---------------------------------------------------------------------------
class BergeronGrid:
    """
    2-D interpolating wrapper for a Bergeron DA or DB atmosphere table.

    Parameters
    ----------
    atm_type :
        ``"DA"`` (pure H) or ``"DB"`` (pure He).
    teff_grid :
        1-D array of Teff values (K) from the parsed table.
    data :
        Array of shape ``(n_logg, n_teff, n_col)`` from ``_parse_global_table``.
        Column 0 is M/Mo; the subsequent columns correspond to ``_BAND_REGISTRY_*``
        column indices (minus the first two for Teff/logg).  Last column is Age.
    """

    def __init__(
        self,
        atm_type: AtmType,
        teff_grid: NDArray[np.float64],
        data: NDArray[np.float64],
    ) -> None:
        self.atm_type: AtmType = atm_type
        self._teff_grid = teff_grid
        self._logg_grid = _LOGG_GRID
        self._data = data  # (n_logg, n_teff, n_col)

        registry = _BAND_REGISTRY_DA if atm_type == "DA" else _BAND_REGISTRY_DB
        self._registry = registry
        self._ext_ratios: dict[str, float] = _make_ext_ratios(registry)

        # Build one interpolator per output column.
        # Column index in *data* (axis 2) is col_idx - 2 from the raw table because
        # we stripped Teff (col 0) and log g (col 1) when building data.
        self._interps: dict[str, RegularGridInterpolator] = {}
        for col_idx, band_name, _ in registry:
            data_col = col_idx - 2  # offset for stripped Teff/logg
            arr = data[:, :, data_col]  # (n_logg, n_teff)
            self._interps[band_name] = RegularGridInterpolator(
                (self._logg_grid, teff_grid),
                arr,
                method="linear",
                bounds_error=False,
                fill_value=None,  # allow extrapolation
            )

        # Interpolator for M/Mo (col_idx=2 in raw → data col 0).
        self._mass_interp = RegularGridInterpolator(
            (self._logg_grid, teff_grid),
            data[:, :, 0],  # M/Mo column
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        # Interpolator for cooling age (last column in data).
        self._age_interp = RegularGridInterpolator(
            (self._logg_grid, teff_grid),
            data[:, :, -1],  # Age column (yr)
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_dir(cls, wd_dir: Path, atm_type: AtmType) -> "BergeronGrid":
        """
        Load a ``BergeronGrid`` from the Bergeron WD data directory.

        Parameters
        ----------
        wd_dir :
            Directory containing ``Table_DA`` and ``Table_DB``.
        atm_type :
            ``"DA"`` or ``"DB"``.

        Limits
        ------
        The directory must contain the expected table file.
        """
        fname = "Table_DA" if atm_type == "DA" else "Table_DB"
        path = wd_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Bergeron table not found: {path}")
        teff_grid, data = _parse_global_table(path)
        return cls(atm_type, teff_grid, data)

    # ------------------------------------------------------------------
    def m_wd(self, teff_k: float, logg: float) -> float:
        """
        Interpolate WD mass (M_sun) at (Teff, log g).

        Parameters
        ----------
        teff_k :
            Effective temperature (K); must be within the table's Teff range.
        logg :
            Surface gravity log g (dex).

        Returns
        -------
        float
            Interpolated M_WD in solar masses.
        """
        pt = np.array([[logg, teff_k]])
        return float(self._mass_interp(pt)[0])

    def cooling_age_yr(self, teff_k: float, logg: float) -> float:
        """Interpolated WD cooling age in years."""
        pt = np.array([[logg, teff_k]])
        return float(self._age_interp(pt)[0])

    # ------------------------------------------------------------------
    def synth_phot(
        self,
        teff_k: float,
        logg: float,
        a_v: float,
        distance_pc: float,
        bands: list[str] | None = None,
    ) -> dict:
        """
        Synthesize apparent magnitudes for a WD at (Teff, log g, A_V, d).

        Parameters
        ----------
        teff_k :
            Effective temperature (K).
        logg :
            Surface gravity log g (dex).
        a_v :
            Visual extinction A_V (mag); F99 R_V=3.1.
        distance_pc :
            Distance in parsecs.
        bands :
            Subset of band names to synthesize (default: all in registry).

        Returns
        -------
        dict with keys:
            ``"mags"``  : {band_name: apparent_magnitude}
            ``"m_wd"``  : interpolated WD mass (M_sun)
            ``"extrap_mass"`` : True when M_WD > 1.3 M_sun or log g > 9.0
            ``"cooling_age_yr"`` : interpolated cooling age (yr)

        Limits
        ------
        The returned magnitudes assume F99 extinction at each filter's
        tabulated effective wavelength (single-wavelength approximation).
        This is valid for WDs where the SED is smooth across the filter.
        """
        dist_mod = 5.0 * np.log10(distance_pc / 10.0)
        pt = np.array([[logg, teff_k]])

        m_wd_val = float(self._mass_interp(pt)[0])
        extrap = (m_wd_val > _MASS_MAX_BERGERON) or (logg > _LOGG_MAX_BERGERON)

        target_bands = set(bands) if bands is not None else {e[1] for e in self._registry}
        mags: dict[str, float] = {}
        for _, band_name, _ in self._registry:
            if band_name not in target_bands:
                continue
            m_abs = float(self._interps[band_name](pt)[0])
            ext_mag = a_v * self._ext_ratios[band_name]
            mags[band_name] = m_abs + dist_mod + ext_mag

        age_yr = float(self._age_interp(pt)[0])

        return {
            "mags": mags,
            "m_wd": m_wd_val,
            "extrap_mass": extrap,
            "cooling_age_yr": age_yr,
        }

    # ------------------------------------------------------------------
    @property
    def teff_min(self) -> float:
        """Minimum Teff in the grid (K)."""
        return float(self._teff_grid[0])

    @property
    def teff_max(self) -> float:
        """Maximum Teff in the grid (K)."""
        return float(self._teff_grid[-1])

    @property
    def available_bands(self) -> list[str]:
        """Sorted list of band names in this grid."""
        return sorted(e[1] for e in self._registry)

    @property
    def band_eff_waves(self) -> dict[str, float]:
        """Effective wavelengths (Å) for each band in this grid."""
        return {e[1]: float(e[2]) for e in self._registry}
