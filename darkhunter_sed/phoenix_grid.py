"""
PHOENIX ACES HiRes spectrum grid loader / interpolator (Path 2).

Layout expected under ``PHOENIX_DIR`` (default ``/Users/rfoley/phoenix/HiResFITS``)::

    WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
    PHOENIX-ACES-AGSS-COND-2011/
        Z{±M/H}/
        Z{±M/H}.Alpha={±α}/
            lte{Teff:05d}-{logg:.2f}{±M/H}[.Alpha={±α}].PHOENIX-...-HiRes.fits

Flux files store surface flux density in ``erg/s/cm^2/cm`` (per cm of
wavelength). This module converts to FLAM (``erg/s/cm^2/Å``) for synphot.

Path-2 SED order (absolute)
---------------------------
Dilute each luminous component to Earth → **sum** fluxes on a common λ grid →
apply F99 (R_V=3.1) to the **combined** SED → **then** synthesize photometry
(:func:`synthesize_mags`). Multi-star / WD / IR-BB models must use the same
combine → redden → synth sequence (never stack per-component magnitudes).

Photometry fits call :meth:`PhoenixGrid.set_photometry_wavelengths` with the
union of bandpass wavesets so dynesty likelihoods blend/redden on that short
grid rather than full HiRes (~1.57M λ).
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from darkhunter_sed.config import phoenix_dir
from darkhunter_sed.extinction_f99 import (
    DEFAULT_R_V,
    apply_f99_extinction,
    f99_alambda_over_av,
)

FluxLoader = Callable[[Path], NDArray[np.floating]]

# Shared wavelength product shipped with the HiRes tree.
_WAVE_NAME = "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
_GRID_SUBDIR = "PHOENIX-ACES-AGSS-COND-2011"

# Max cached resampled-photometry flux arrays in :meth:`PhoenixGrid._load_for_sed`.
# 4096 entries × ~25 KB each ≈ 100 MB — large enough that a 200-live-point dynesty
# run never thrashes the cache.
_DEFAULT_FLUX_CACHE_SIZE = 4096

# Max cached full HiRes flux arrays in :meth:`PhoenixGrid._load`.
# Each HiRes array is 1.57 M points × 8 B ≈ 12.5 MB; 32 entries ≈ 400 MB.
# Only the 8 interpolation corners of the current call need to be hot, so 32
# is more than enough and avoids the ~51 GB explosion that 4096 would cause.
_DEFAULT_HIRES_CACHE_SIZE = 32

# Directory: Z-0.0 or Z-0.0.Alpha=+0.20
_DIR_RE = re.compile(
    r"^Z(?P<mh>[+-]?\d+\.\d+)(?:\.Alpha=(?P<alpha>[+-]?\d+\.\d+))?$"
)
# File: lte05000-4.50-0.0.PHOENIX-... or lte05000-4.50-0.0.Alpha=+0.20.PHOENIX-...
_FILE_RE = re.compile(
    r"^lte(?P<teff>\d{5})-(?P<logg>\d+\.\d+)(?P<mh>[+-]\d+\.\d+)"
    r"(?:\.Alpha=(?P<alpha>[+-]?\d+\.\d+))?"
    r"\.PHOENIX-ACES-AGSS-COND-2011-HiRes\.fits$"
)

# erg/s/cm^2/cm → erg/s/cm^2/Å  (1 cm = 1e8 Å)
_CM_TO_AA = 1.0e8
_PC_TO_CM = 3.0856775814913673e18


@dataclass(frozen=True, slots=True)
class PhoenixPoint:
    """
    One on-disk PHOENIX HiRes model.

    Parameters
    ----------
    teff_k :
        Effective temperature in Kelvin.
    logg :
        Surface gravity ``log10(g/[cm s^-2])``.
    mh :
        Metallicity ``[M/H]`` / ``[Fe/H]`` (grid label).
    alpha :
        Alpha enhancement ``[α/Fe]`` (``0.0`` when the directory has no Alpha tag).
    path :
        Absolute path to the spectrum FITS primary HDU.
    """

    teff_k: float
    logg: float
    mh: float
    alpha: float
    path: Path


def _fmt_mh_dir(mh: float) -> str:
    """Format metallicity for a ``Z…`` directory name (matches on-disk labels)."""
    if abs(mh) < 1e-12:
        return "Z-0.0"
    return f"Z{mh:+.1f}"


def _fmt_alpha_tag(alpha: float) -> str:
    """Format ``.Alpha=±0.20`` segment (empty string for α≈0)."""
    if abs(alpha) < 1e-12:
        return ""
    return f".Alpha={alpha:+.2f}"


def _fmt_mh_file(mh: float) -> str:
    """Metallicity token inside the ``lte…`` filename."""
    if abs(mh) < 1e-12:
        return "-0.0"
    return f"{mh:+.1f}"


def phoenix_model_path(
    root: Path | str,
    teff_k: float,
    logg: float,
    mh: float,
    alpha: float = 0.0,
) -> Path:
    """
    Build the canonical on-disk path for a grid point (may not exist).

    Parameters
    ----------
    root :
        HiResFITS root (``PHOENIX_DIR``).
    teff_k, logg, mh, alpha :
        Atmospheric parameters (Teff rounded to nearest integer Kelvin for the
        filename).

    Returns
    -------
    Path
        Expected spectrum FITS path under ``PHOENIX-ACES-AGSS-COND-2011/``.
    """
    root = Path(root)
    teff_i = int(round(float(teff_k)))
    dirname = _fmt_mh_dir(float(mh)) + _fmt_alpha_tag(float(alpha))
    fname = (
        f"lte{teff_i:05d}-{float(logg):.2f}{_fmt_mh_file(float(mh))}"
        f"{_fmt_alpha_tag(float(alpha))}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    )
    return root / _GRID_SUBDIR / dirname / fname


def load_phoenix_wavelength(
    root: Path | str | None = None,
) -> NDArray[np.float64]:
    """
    Load the shared PHOENIX HiRes wavelength array (Å).

    Parameters
    ----------
    root :
        HiResFITS root, or ``None`` → :func:`~darkhunter_sed.config.phoenix_dir`.

    Returns
    -------
    NDArray[np.float64]
        1-D wavelengths in Angstroms.

    Limits
    ------
    Expects ``WAVE_PHOENIX-ACES-AGSS-COND-2011.fits`` primary HDU. Full file is
    ~1.5e6 samples; tests should inject a tiny array instead of calling this.
    """
    root_p = phoenix_dir(root)
    path = root_p / _WAVE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"PHOENIX wavelength file missing: {path}")
    data = fits.getdata(path, memmap=True)
    wave = np.asarray(data, dtype=np.float64).reshape(-1)
    if wave.size < 2:
        raise ValueError(f"Wavelength array too short in {path}")
    return wave


def load_phoenix_flux(
    path: Path | str,
    *,
    to_flam: bool = True,
    stride: int = 1,
) -> NDArray[np.float64]:
    """
    Load a single PHOENIX HiRes flux array.

    Parameters
    ----------
    path :
        Spectrum FITS path.
    to_flam :
        If True (default), convert ``erg/s/cm^2/cm`` → ``erg/s/cm^2/Å`` by
        dividing by ``1e8``. If False, return native surface flux.
    stride :
        Subsample the native array by this factor (``data[::stride]``)
        *before* casting to float64, so the copy is only ``1/stride`` the
        size. Default ``1`` (no subsampling).

    Returns
    -------
    NDArray[np.float64]
        1-D surface flux density.

    Limits
    ------
    Reads primary HDU only via a memory-mapped file (``fits.open(...,
    memmap=True)``) rather than ``fits.getdata`` — measured ~3-8x faster per
    file for this data (see issue #65); avoids astropy's extra validation
    overhead on top of the raw read. Does not apply radius/distance scaling
    or extinction.
    """
    path = Path(path)
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        if stride > 1:
            data = data[::stride]
        flux = np.asarray(data, dtype=np.float64).reshape(-1)
    if to_flam:
        flux = flux / _CM_TO_AA
    return flux


def scale_surface_to_earth(
    flux_surface: NDArray[np.floating],
    *,
    radius_cm: float,
    distance_pc: float,
) -> NDArray[np.float64]:
    """
    Convert stellar surface flux density to Earth-frame flux density.

    Parameters
    ----------
    flux_surface :
        Surface flux density (e.g. FLAM at R_star).
    radius_cm :
        Stellar radius in cm (``> 0``).
    distance_pc :
        Heliocentric distance in parsecs (``> 0``).

    Returns
    -------
    NDArray[np.float64]
        ``flux_surface * (R / d)^2``.

    Limits
    ------
    Geometric dilution only; no limb darkening or ISM (use F99 separately).
    """
    if radius_cm <= 0.0 or distance_pc <= 0.0:
        raise ValueError("radius_cm and distance_pc must be positive")
    d_cm = float(distance_pc) * _PC_TO_CM
    geom = (float(radius_cm) / d_cm) ** 2
    return np.asarray(flux_surface, dtype=np.float64) * geom


def _brackets(
    axis: Sequence[float],
    value: float,
    *,
    allow_nearest: bool = False,
) -> tuple[float, float]:
    """
    Return ``(lo, hi)`` grid values bracketing ``value``.

    If ``value`` lies on a node, ``lo == hi == value``. Outside the axis range,
    raises unless ``allow_nearest`` (then both edges are the nearest endpoint).
    """
    arr = np.asarray(axis, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Empty axis")
    if value < arr[0] or value > arr[-1]:
        if not allow_nearest:
            raise ValueError(
                f"Value {value} outside grid axis [{arr[0]}, {arr[-1]}]"
            )
        nearest = float(arr[0] if value < arr[0] else arr[-1])
        return nearest, nearest
    # Exact node
    exact = np.where(np.isclose(arr, value, rtol=0.0, atol=1e-9))[0]
    if exact.size:
        v = float(arr[int(exact[0])])
        return v, v
    hi_idx = int(np.searchsorted(arr, value, side="left"))
    lo = float(arr[hi_idx - 1])
    hi = float(arr[hi_idx])
    return lo, hi


def _norm_weight(lo: float, hi: float, value: float) -> float:
    """Fractional weight toward ``hi`` on ``[lo, hi]`` (0 at lo, 1 at hi)."""
    if lo == hi:
        return 0.0
    return (float(value) - lo) / (hi - lo)


def _multilinear(
    corners: Mapping[tuple[float, ...], NDArray[np.floating]],
    axes_lo_hi: Sequence[tuple[float, float]],
    query: Sequence[float],
) -> NDArray[np.float64]:
    """
    Multilinear interpolation on a hypercube of corner fluxes.

    ``corners`` keys are tuples of axis values (each lo or hi). Missing corners
    raise ``KeyError`` with a clear message (caller should reduce dimensionality).
    """
    n = len(query)
    if n != len(axes_lo_hi):
        raise ValueError("query / axes dimension mismatch")
    weights = [_norm_weight(lo, hi, q) for (lo, hi), q in zip(axes_lo_hi, query)]

    # Iterate all 2^n corners
    acc: NDArray[np.float64] | None = None
    for mask in range(1 << n):
        key_vals: list[float] = []
        w = 1.0
        for i in range(n):
            lo, hi = axes_lo_hi[i]
            use_hi = bool(mask & (1 << i))
            key_vals.append(hi if use_hi else lo)
            t = weights[i]
            w *= t if use_hi else (1.0 - t)
        key = tuple(key_vals)
        if key not in corners:
            raise KeyError(key)
        piece = w * np.asarray(corners[key], dtype=np.float64)
        acc = piece if acc is None else acc + piece
    assert acc is not None
    return acc


class PhoenixGrid:
    """
    Indexed PHOENIX HiRes library with multi-linear parameter interpolation.

    Parameters
    ----------
    root :
        HiResFITS root, or ``None`` → :func:`~darkhunter_sed.config.phoenix_dir`.
    wavelength :
        Optional injected wavelength array (Å). When set, the shared WAVE FITS
        is **not** read (use in CI with a tiny mock grid).
    points :
        Optional injected :class:`PhoenixPoint` list. When set, the on-disk tree
        is **not** scanned (CI). Each point's ``path`` must be loadable by
        :func:`load_phoenix_flux` unless ``flux_loader`` is overridden.
    flux_loader :
        Optional ``callable(path) -> flux`` for tests (bypass FITS I/O).
    to_flam :
        Convert native ``/cm`` flux to FLAM when loading.
    hires_stride :
        Subsample the native HiRes grid (0.01 Å spacing) by this factor at
        load time, before any interpolation — e.g. ``8`` keeps every 8th
        pixel (0.08 Å spacing). Applied once to :attr:`wavelength` at
        construction and to every corner's raw flux array in :meth:`_load`,
        so the two always stay shape-consistent. Default ``1`` (no
        subsampling) is fully backward compatible. Broadband filters are
        hundreds to thousands of Å wide, so even large strides lose no
        fidelity for photometric synthesis; this only shrinks the array that
        every cache miss must load from disk and interpolate against.

    Limits
    ------
    - Alpha enhancement is used **only** when Alpha-tagged directories exist in the
      index; requesting α≠0 with no Alpha coverage raises.
    - Interpolation requires a complete hypercube of corner models in the active
      dimensions; sparse α coverage may force α to the nearest available node
      after Teff/logg/[Fe/H] interpolation at that α, or raise if no α slice
      exists near the query.
    - Outside Teff/logg/[Fe/H] bounds → ``ValueError`` (no extrapolation).
    - Full ~42GB tree is never required for unit tests (inject ``wavelength`` +
      ``points`` + ``flux_loader``).
    - Sorted axis lists per α (Teff, logg, [M/H]) are pre-computed at init time
      so :meth:`_interp_3d_at_alpha` is O(log k) for binary search rather than
      O(N) for repeated set/sort over all 14 k+ grid points.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        wavelength: NDArray[np.floating] | None = None,
        points: Sequence[PhoenixPoint] | None = None,
        flux_loader: FluxLoader | None = None,
        to_flam: bool = True,
        flux_cache_size: int = _DEFAULT_FLUX_CACHE_SIZE,
        hires_cache_size: int = _DEFAULT_HIRES_CACHE_SIZE,
        hires_stride: int = 1,
    ) -> None:
        self.root = phoenix_dir(root)
        self.to_flam = bool(to_flam)
        self._flux_loader = flux_loader
        self._hires_cache_size = max(int(hires_cache_size), 0)
        self._flux_cache_size = max(int(flux_cache_size), 0)
        self._hires_stride = max(int(hires_stride), 1)
        self._flux_cache: OrderedDict[Path, NDArray[np.float64]] = OrderedDict()
        # Optional photometry λ grid: dynesty evaluates SED here, not on HiRes.
        self._phot_wave: NDArray[np.float64] | None = None
        self._phot_flux_cache: OrderedDict[Path, NDArray[np.float64]] = OrderedDict()
        self._phot_alav: dict[float, NDArray[np.float64]] = {}
        if wavelength is not None:
            self.wavelength = np.asarray(wavelength, dtype=np.float64).reshape(-1)
        else:
            self.wavelength = load_phoenix_wavelength(self.root)
        if self._hires_stride > 1:
            self.wavelength = self.wavelength[:: self._hires_stride]
        if points is not None:
            self._points = list(points)
        else:
            self._points = self._scan_disk()
        if not self._points:
            raise FileNotFoundError(
                f"No PHOENIX HiRes models under {self.root / _GRID_SUBDIR}"
            )
        self._index: dict[tuple[float, float, float, float], PhoenixPoint] = {}
        for p in self._points:
            self._index[_key(p.teff_k, p.logg, p.mh, p.alpha)] = p

        # Pre-compute sorted (Teff, logg, [M/H]) axis lists for each α slice.
        # _interp_3d_at_alpha previously rebuilt these via O(N) Python loops on
        # every likelihood call (N ≈ 14 000 PhoenixPoints, 6–8 ms / call).
        # Pre-computing once at init saves 30–50 s over a full dynesty run.
        self._alpha_teffs: dict[float, list[float]] = {}
        self._alpha_loggs: dict[float, list[float]] = {}
        self._alpha_mhs: dict[float, list[float]] = {}
        for _a in self.available_alphas():
            _pts_a = self._points_at_alpha(_a)
            self._alpha_teffs[_a] = sorted({p.teff_k for p in _pts_a})
            self._alpha_loggs[_a] = sorted({p.logg for p in _pts_a})
            self._alpha_mhs[_a] = sorted({p.mh for p in _pts_a})

    def _scan_disk(self) -> list[PhoenixPoint]:
        grid_root = self.root / _GRID_SUBDIR
        if not grid_root.is_dir():
            raise FileNotFoundError(f"PHOENIX grid directory missing: {grid_root}")
        out: list[PhoenixPoint] = []
        for d in sorted(grid_root.iterdir()):
            if not d.is_dir():
                continue
            dm = _DIR_RE.match(d.name)
            if dm is None:
                continue
            dir_mh = float(dm.group("mh"))
            dir_alpha = float(dm.group("alpha") or 0.0)
            for f in d.glob("lte*.fits"):
                fm = _FILE_RE.match(f.name)
                if fm is None:
                    continue
                teff = float(fm.group("teff"))
                logg = float(fm.group("logg"))
                mh = float(fm.group("mh"))
                alpha = float(fm.group("alpha") or 0.0)
                # Prefer filename tags; fall back to directory if inconsistent.
                if abs(mh - dir_mh) > 1e-6:
                    mh = dir_mh
                if abs(alpha - dir_alpha) > 1e-6:
                    alpha = dir_alpha
                out.append(
                    PhoenixPoint(teff, logg, mh, alpha, f.resolve())
                )
        return out

    @property
    def points(self) -> tuple[PhoenixPoint, ...]:
        """All indexed models."""
        return tuple(self._points)

    def available_alphas(self) -> tuple[float, ...]:
        """Sorted unique ``[α/Fe]`` values present in the index."""
        return tuple(sorted({p.alpha for p in self._points}))

    def available_mh(self) -> tuple[float, ...]:
        """Sorted unique ``[M/H]`` values present in the index."""
        return tuple(sorted({p.mh for p in self._points}))

    def _load(self, point: PhoenixPoint) -> NDArray[np.float64]:
        """
        Load surface FLAM for one grid point (LRU-cached).

        Parameters
        ----------
        point :
            Indexed :class:`PhoenixPoint`.

        Returns
        -------
        NDArray[np.float64]
            Flux on :attr:`wavelength` (same length).

        Limits
        ------
        Cache size is :attr:`_hires_cache_size` (default 32; ~400 MB). Injected
        ``flux_loader`` results are also cached by ``point.path``.
        """
        key = point.path
        cached = self._flux_cache.get(key)
        if cached is not None:
            self._flux_cache.move_to_end(key)
            return cached
        if self._flux_loader is not None:
            flux = np.asarray(self._flux_loader(point.path), dtype=np.float64)
            if self._hires_stride > 1:
                flux = flux[:: self._hires_stride]
        else:
            # Stride applied inside load_phoenix_flux, before the float64
            # cast, so the copy is 1/stride the size (see issue #65).
            flux = load_phoenix_flux(
                point.path, to_flam=self.to_flam, stride=self._hires_stride
            )
        if flux.shape != self.wavelength.shape:
            raise ValueError(
                f"Flux length {flux.size} != wavelength {self.wavelength.size} "
                f"for {point.path}"
            )
        if self._hires_cache_size > 0:
            self._flux_cache[key] = flux
            self._flux_cache.move_to_end(key)
            while len(self._flux_cache) > self._hires_cache_size:
                self._flux_cache.popitem(last=False)
        return flux

    def set_photometry_wavelengths(
        self,
        wave_aa: NDArray[np.floating] | None,
    ) -> NDArray[np.float64] | None:
        """
        Restrict Path-2 photometry SED evaluation to ``wave_aa`` (Å).

        Parameters
        ----------
        wave_aa :
            Sorted photometry wavelength grid (typically the unique union of
            bandpass wavesets), or ``None`` to clear and use full HiRes again.

        Returns
        -------
        NDArray[np.float64] | None
            The active photometry grid (copy), or ``None`` if cleared.

        Limits
        ------
        Corner HiRes spectra are still loaded once, then resampled onto this
        grid and cached. Dynesty likelihoods should call this once per fit.
        """
        if wave_aa is None:
            self._phot_wave = None
            self._phot_flux_cache.clear()
            self._phot_alav.clear()
            return None
        wave = np.asarray(wave_aa, dtype=np.float64).reshape(-1)
        if wave.size < 2:
            raise ValueError("photometry wavelength grid needs ≥2 samples")
        if not np.all(np.diff(wave) > 0.0):
            wave = np.unique(wave)
        if (
            self._phot_wave is not None
            and self._phot_wave.shape == wave.shape
            and np.allclose(self._phot_wave, wave, rtol=0.0, atol=1e-9)
        ):
            return self._phot_wave
        self._phot_wave = wave.copy()
        self._phot_flux_cache.clear()
        self._phot_alav.clear()
        return self._phot_wave

    def _load_for_sed(self, point: PhoenixPoint) -> NDArray[np.float64]:
        """HiRes load, or resampled onto :attr:`_phot_wave` when set."""
        if self._phot_wave is None:
            return self._load(point)
        key = point.path
        cached = self._phot_flux_cache.get(key)
        if cached is not None:
            self._phot_flux_cache.move_to_end(key)
            return cached
        hi = self._load(point)
        flux = np.interp(self._phot_wave, self.wavelength, hi)
        if self._flux_cache_size > 0:
            self._phot_flux_cache[key] = flux
            self._phot_flux_cache.move_to_end(key)
            while len(self._phot_flux_cache) > self._flux_cache_size:
                self._phot_flux_cache.popitem(last=False)
        return flux

    def _phot_alav_curve(self, r_v: float) -> NDArray[np.float64] | None:
        """Cached ``A_λ/A_V`` on the photometry grid, or ``None`` if HiRes mode."""
        if self._phot_wave is None:
            return None
        key = float(r_v)
        cached = self._phot_alav.get(key)
        if cached is not None:
            return cached
        al = f99_alambda_over_av(self._phot_wave, r_v=r_v)
        self._phot_alav[key] = al
        return al

    def nearest_point(
        self,
        teff_k: float,
        logg: float,
        mh: float,
        alpha: float = 0.0,
    ) -> PhoenixPoint:
        """
        Return the indexed model minimizing a scaled parameter distance.

        Parameters
        ----------
        teff_k, logg, mh, alpha :
            Query atmosphere. If ``alpha`` coverage is absent, only α=0 models
            are considered when ``alpha≈0``; otherwise raises.

        Returns
        -------
        PhoenixPoint
            Nearest on-disk model.
        """
        alphas = self.available_alphas()
        if abs(alpha) > 1e-12 and not any(abs(a - alpha) < 1e-9 for a in alphas):
            # Allow nearest alpha only among existing α values.
            if not any(abs(a) > 1e-12 for a in alphas):
                raise ValueError(
                    f"[α/Fe]={alpha} requested but no Alpha-tagged directories exist "
                    f"under {self.root}"
                )
        best: PhoenixPoint | None = None
        best_d = np.inf
        for p in self._points:
            # Prefer matching alpha slice when possible.
            d = (
                ((p.teff_k - teff_k) / 250.0) ** 2
                + ((p.logg - logg) / 0.25) ** 2
                + ((p.mh - mh) / 0.25) ** 2
                + ((p.alpha - alpha) / 0.1) ** 2
            )
            if d < best_d:
                best_d = d
                best = p
        assert best is not None
        return best

    def spectrum(
        self,
        teff_k: float,
        logg: float,
        mh: float,
        alpha: float = 0.0,
        *,
        interpolate: bool = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Return ``(wavelength_AA, flux_FLAM_surface)`` at the requested parameters.

        Parameters
        ----------
        teff_k, logg, mh, alpha :
            Atmospheric parameters. ``mh`` is the grid ``[M/H]`` / ``[Fe/H]``.
        interpolate :
            If True (default), multi-linear interpolate among bracketing grid
            nodes. If False, return the nearest indexed model flux.

        Returns
        -------
        wave, flux :
            Shared wavelength (Å) and surface flux (FLAM if ``to_flam``).

        Limits
        ------
        No radius/distance or extinction. Extrapolation outside Teff/logg/[Fe/H]
        bounds is rejected. Partial α grids: interpolates in α only when both
        bracketing α slices have the needed Teff/logg/[Fe/H] corners; otherwise
        uses the nearest available α after 3-D interp on that slice.
        """
        if not interpolate:
            pt = self.nearest_point(teff_k, logg, mh, alpha)
            return self.wavelength.copy(), self._load(pt)

        return self.wavelength.copy(), self._interpolate_flux(
            float(teff_k), float(logg), float(mh), float(alpha)
        )

    def _points_at_alpha(self, alpha: float) -> list[PhoenixPoint]:
        return [p for p in self._points if abs(p.alpha - alpha) < 1e-9]

    def _interpolate_flux(
        self,
        teff_k: float,
        logg: float,
        mh: float,
        alpha: float,
    ) -> NDArray[np.float64]:
        alphas = list(self.available_alphas())
        has_alpha = any(abs(a) > 1e-12 for a in alphas)

        if abs(alpha) > 1e-12 and not has_alpha:
            raise ValueError(
                f"[α/Fe]={alpha} requested but no Alpha-tagged directories exist "
                f"under {self.root}"
            )

        # Solar-α (or grid with only α=0): 3-D interp at α=0.
        if abs(alpha) < 1e-12:
            a_use = 0.0 if any(abs(a) < 1e-12 for a in alphas) else alphas[0]
            return self._interp_3d_at_alpha(teff_k, logg, mh, a_use)

        a_lo, a_hi = _brackets(alphas, alpha, allow_nearest=False)
        if a_lo == a_hi:
            return self._interp_3d_at_alpha(teff_k, logg, mh, a_lo)

        try:
            f_lo = self._interp_3d_at_alpha(teff_k, logg, mh, a_lo)
            f_hi = self._interp_3d_at_alpha(teff_k, logg, mh, a_hi)
        except (ValueError, KeyError) as exc:
            # Sparse α: nearest α slice with a complete 3-D neighborhood.
            nearest_a = min(alphas, key=lambda a: abs(a - alpha))
            try:
                return self._interp_3d_at_alpha(teff_k, logg, mh, nearest_a)
            except Exception as exc2:
                raise ValueError(
                    f"Cannot interpolate at [α/Fe]={alpha} near Teff={teff_k}, "
                    f"logg={logg}, [M/H]={mh}: {exc2}"
                ) from exc

        w = _norm_weight(a_lo, a_hi, alpha)
        return (1.0 - w) * f_lo + w * f_hi

    def _interp_3d_at_alpha(
        self,
        teff_k: float,
        logg: float,
        mh: float,
        alpha: float,
    ) -> NDArray[np.float64]:
        # Use pre-computed axis lists (O(1) lookup) instead of rebuilding
        # sorted sets from all grid points on every call (was O(N), N≈14k).
        if alpha not in self._alpha_teffs:
            raise ValueError(f"No models at [α/Fe]={alpha}")

        teffs = self._alpha_teffs[alpha]
        loggs = self._alpha_loggs[alpha]
        mhs = self._alpha_mhs[alpha]

        t_lo, t_hi = _brackets(teffs, teff_k)
        g_lo, g_hi = _brackets(loggs, logg)
        m_lo, m_hi = _brackets(mhs, mh)

        corners: dict[tuple[float, ...], NDArray[np.float64]] = {}
        for t in (t_lo, t_hi):
            for g in (g_lo, g_hi):
                for m in (m_lo, m_hi):
                    key4 = _key(t, g, m, alpha)
                    if key4 not in self._index:
                        raise KeyError(
                            f"Missing PHOENIX corner Teff={t}, logg={g}, "
                            f"[M/H]={m}, [α/Fe]={alpha}"
                        )
                    corners[(t, g, m)] = self._load_for_sed(self._index[key4])

        return _multilinear(
            corners,
            ((t_lo, t_hi), (g_lo, g_hi), (m_lo, m_hi)),
            (teff_k, logg, mh),
        )

    def extincted_spectrum(
        self,
        teff_k: float,
        logg: float,
        mh: float,
        alpha: float,
        a_v: float,
        *,
        radius_cm: float | None = None,
        distance_pc: float | None = None,
        r_v: float = DEFAULT_R_V,
        interpolate: bool = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Interpolate PHOENIX, optionally dilute to Earth, then apply F99.

        Parameters
        ----------
        teff_k, logg, mh, alpha :
            Atmosphere parameters.
        a_v :
            Visual extinction (mag). Applied **after** geometric dilution.
        radius_cm, distance_pc :
            If both given, scale surface → Earth flux. If both ``None``, leave
            surface flux (tests / absolute scaling deferred). Mixing one of
            each is an error.
        r_v :
            Extinction R_V (Path-2 default 3.1).
        interpolate :
            Passed to :meth:`spectrum`.

        Returns
        -------
        wave, flux :
            Wavelength (Å) and extincted flux density. When
            :meth:`set_photometry_wavelengths` is active, ``wave`` is that
            photometry grid (not full HiRes).

        Limits
        ------
        Path-2 photometry fits should set a bandpass λ grid so dynesty does not
        blend 1.57M-point arrays every call.
        """
        if self._phot_wave is not None:
            if not interpolate:
                pt = self.nearest_point(teff_k, logg, mh, alpha)
                flux = self._load_for_sed(pt).copy()
            else:
                flux = self._interpolate_flux(
                    float(teff_k), float(logg), float(mh), float(alpha)
                )
            wave = self._phot_wave
        else:
            wave, flux = self.spectrum(
                teff_k, logg, mh, alpha, interpolate=interpolate
            )
        if (radius_cm is None) ^ (distance_pc is None):
            raise ValueError("Provide both radius_cm and distance_pc, or neither")
        if radius_cm is not None and distance_pc is not None:
            flux = scale_surface_to_earth(
                flux, radius_cm=radius_cm, distance_pc=distance_pc
            )
        alav = self._phot_alav_curve(r_v)
        flux = apply_f99_extinction(
            wave, flux, a_v, r_v=r_v, alambda_over_av=alav
        )
        return wave, flux


def _key(teff: float, logg: float, mh: float, alpha: float) -> tuple[float, float, float, float]:
    """Round parameters to stable dict keys."""
    return (
        round(float(teff), 3),
        round(float(logg), 5),
        round(float(mh), 5),
        round(float(alpha), 5),
    )



def _abmag_from_flam_on_bandpass(
    wave_src: NDArray[np.float64],
    flux_flam: NDArray[np.float64],
    wave_bp: NDArray[np.float64],
    throughput: NDArray[np.float64],
) -> float:
    """
    Photon-weighted AB magnitude matching synphot ``effstim('abmag')``.

    Parameters
    ----------
    wave_src, flux_flam :
        Source SED (Å, FLAM), already reddened.
    wave_bp, throughput :
        Bandpass wavelength (Å) and dimensionless throughput on ``wave_bp``.

    Returns
    -------
    float
        AB magnitude.

    Limits
    ------
    Uses ``m = -2.5 log10(∫ f_ν S dν/ν / ∫ S dν/ν) - 48.60`` with
    ``f_ν = f_λ λ²/c`` (c in Å/s). Samples outside ``wave_src`` get zero flux.
    """
    # Speed of light in Å/s for F_λ (erg/s/cm²/Å) → F_ν (erg/s/cm²/Hz).
    c_aa = 2.99792458e18
    flux_bp = np.interp(wave_bp, wave_src, flux_flam)
    outside = (wave_bp < wave_src[0]) | (wave_bp > wave_src[-1])
    if np.any(outside):
        flux_bp = flux_bp.copy()
        flux_bp[outside] = 0.0
    fnu = flux_bp * (wave_bp * wave_bp) / c_aa
    nu = c_aa / wave_bp
    order = np.argsort(nu)
    nu_s = nu[order]
    wgt = throughput[order] / nu_s
    num = float(np.trapz(fnu[order] * wgt, nu_s))
    den = float(np.trapz(wgt, nu_s))
    if den <= 0.0 or not np.isfinite(den):
        return float("nan")
    if num <= 0.0 or not np.isfinite(num):
        # Zero or negative integrated flux (e.g. cool star in UV bandpass):
        # return a very faint sentinel magnitude rather than NaN so the
        # Gaussian likelihood returns a large but finite penalty and dynesty
        # can still follow the gradient toward hotter/brighter models.
        return 99.0
    return float(-2.5 * np.log10(num / den) - 48.60)


def synthesize_mags(
    wave_aa: NDArray[np.floating],
    flux_flam: NDArray[np.floating],
    bands: Sequence[str],
    *,
    systems: Sequence[str] = ("ab",),
    bandpasses: Mapping[str, object] | None = None,
    cdbs_root: Path | str | None = None,
) -> dict[str, dict[str, float]]:
    """
    Synthesize AB and/or Vega magnitudes from a **post-reddening** SED.

    Parameters
    ----------
    wave_aa :
        Wavelengths in Angstroms (common system grid).
    flux_flam :
        Flux density in FLAM (``erg/s/cm^2/Å``). Must already be geometrically
        diluted and **F99-reddened** (or be the sum of diluted components after
        a single F99 on the combined SED). This function does **not** apply
        extinction.
    bands :
        Registry band names (e.g. ``PS_g``, ``GaiaDR3_G``).
    systems :
        Subset of ``{"ab", "vega"}`` (case-insensitive). Default AB only.
    bandpasses :
        Optional mapping ``band → synphot.SpectralElement``. When omitted,
        loads via :func:`darkhunter_sed.filters_synphot.load_bandpass`.
    cdbs_root :
        Optional CDBS root forwarded to ``load_bandpass``.

    Returns
    -------
    dict
        ``{band: {"ab": float, "vega": float}}`` for requested systems
        (missing system keys omitted).

    Limits
    ------
    Path-2 order: dilute → **sum components** → F99 on the full SED → **then**
    this function. Multi-star/WD/BB must not synthesize per component and stack
    magnitudes.

    Performance: AB uses a vectorized photon-weighted integral on each
    bandpass ``waveset`` (no synphot ``Observation`` on the HiRes grid).
    Vega (if requested) still uses synphot on the bandpass-native source only.
    Broadband filters do not need R~500000 sampling inside the integral.

    Samples outside the source λ range are set to 0 (taper-like) so WISE_W1/W2
    still integrate with missing long-λ flux — examine later; may need a
    PHOENIX/SED red extension.
    """
    from synphot import SpectralElement
    from astropy import units as u

    wave = np.asarray(wave_aa, dtype=np.float64)
    flux = np.asarray(flux_flam, dtype=np.float64)
    if wave.shape != flux.shape or wave.ndim != 1:
        raise ValueError("wave_aa and flux_flam must be equal-length 1-D arrays")
    if wave.size < 2:
        raise ValueError("Need ≥2 wavelength samples")
    if not np.all(np.diff(wave) > 0.0):
        order = np.argsort(wave)
        wave = wave[order]
        flux = flux[order]

    sys_set = {s.lower() for s in systems}
    unknown = sys_set - {"ab", "vega"}
    if unknown:
        raise ValueError(f"Unknown magnitude systems: {sorted(unknown)}")

    need_vega = "vega" in sys_set
    if need_vega:
        from synphot import Observation, SourceSpectrum
        from synphot.models import Empirical1D
        from synphot.units import FLAM
        import warnings

        vega_spec = SourceSpectrum.from_vega()

    out: dict[str, dict[str, float]] = {}

    for band in bands:
        if bandpasses is not None and band in bandpasses:
            bp = bandpasses[band]
        else:
            from darkhunter_sed.filters_synphot import load_bandpass

            bp = load_bandpass(band, cdbs_root=cdbs_root)
        if not isinstance(bp, SpectralElement):
            raise TypeError(f"Bandpass for {band} is not a SpectralElement")
        bp_waves = bp.waveset
        if bp_waves is None or len(bp_waves) < 2:
            raise ValueError(f"Bandpass {band} has no usable waveset")
        wave_bp = np.asarray(bp_waves.to(u.AA).value, dtype=np.float64)
        thru = np.asarray(bp(bp_waves).value, dtype=np.float64)
        entry: dict[str, float] = {}
        if "ab" in sys_set:
            entry["ab"] = _abmag_from_flam_on_bandpass(wave, flux, wave_bp, thru)
        if need_vega:
            flux_bp = np.interp(wave_bp, wave, flux)
            outside = (wave_bp < wave[0]) | (wave_bp > wave[-1])
            if np.any(outside):
                flux_bp = flux_bp.copy()
                flux_bp[outside] = 0.0
            source = SourceSpectrum(
                Empirical1D,
                points=wave_bp * u.AA,
                lookup_table=flux_bp * FLAM,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Source spectrum is tapered",
                    module="synphot.observation",
                )
                obs = Observation(source, bp, force="taper")
            entry["vega"] = float(obs.effstim("vegamag", vegaspec=vega_spec).value)
        out[band] = entry
    return out


# Physical constants (CGS) for blackbody calculation.
_H_CGS: float = 6.62607015e-27    # erg·s
_C_CGS: float = 2.99792458e10     # cm/s
_K_CGS: float = 1.380649e-16      # erg/K
_SIGMA_SB_CGS: float = 5.670374419e-5  # erg/s/cm²/K⁴
_L_SUN_ERG: float = 3.828e33      # erg/s


def bb_flux_flam_at_earth(
    wave_aa: NDArray[np.float64],
    t_bb_k: float,
    l_bb_lsun: float,
    distance_pc: float,
) -> NDArray[np.float64]:
    """
    Planck blackbody FLAM (erg/s/cm²/Å) at Earth for a given luminosity and temperature.

    The BB radius is derived from Stefan-Boltzmann: R_bb = √(L_bb / 4π σ T_bb⁴).
    Earth-frame F_λ = π B_λ(T) × (R_bb / d)² = L_bb × B_λ(T) / (4 σ T_bb⁴ d²).

    Parameters
    ----------
    wave_aa :
        Wavelengths in Angstroms (must be positive).
    t_bb_k :
        Blackbody temperature in Kelvin (``> 0``).
    l_bb_lsun :
        Bolometric luminosity in solar units (``> 0``).
    distance_pc :
        Heliocentric distance in parsecs (``> 0``).

    Returns
    -------
    NDArray[np.float64]
        FLAM in erg/s/cm²/Å on ``wave_aa``; same shape as input.

    Limits
    ------
    No extinction applied here — apply F99 to the combined stellar+BB SED after summing.
    Returns zeros (not NaN) for wavelengths where the Planck exponential overflows.
    """
    wave_aa = np.asarray(wave_aa, dtype=np.float64)
    wave_cm = wave_aa * 1.0e-8  # Å → cm
    hc_lkT = _H_CGS * _C_CGS / (wave_cm * _K_CGS * float(t_bb_k))
    # B_lambda in erg/s/cm²/sr/cm; use expm1 for numerical stability.
    # np.expm1 saturates to -1 for very large negative args and overflows for very large positive;
    # clip the exponent to avoid division-by-zero or inf in the denominator.
    hc_lkT_clipped = np.clip(hc_lkT, 0.0, 709.0)  # exp(709) ≈ DBL_MAX
    planck_per_cm = 2.0 * _H_CGS * _C_CGS**2 / wave_cm**5 / np.expm1(hc_lkT_clipped)
    planck_flam = planck_per_cm * 1.0e-8  # erg/s/cm²/sr/Å (divide by 1 cm / 1e8 Å)
    # F_lambda = L_bb × B_λ(T) / (4 σ T_bb⁴ d²)
    l_bb_erg = float(l_bb_lsun) * _L_SUN_ERG
    d_cm = float(distance_pc) * _PC_TO_CM
    factor = l_bb_erg / (4.0 * _SIGMA_SB_CGS * float(t_bb_k) ** 4 * d_cm**2)
    return planck_flam * factor


def phoenix_synth_phot(
    grid: PhoenixGrid,
    teff_k: float,
    logg: float,
    mh: float,
    alpha: float,
    a_v: float,
    bands: Sequence[str],
    *,
    radius_cm: float | None = None,
    distance_pc: float | None = None,
    systems: Sequence[str] = ("ab",),
    bandpasses: Mapping[str, object] | None = None,
    r_v: float = DEFAULT_R_V,
    bb_t_k: float | None = None,
    bb_l_lsun: float | None = None,
) -> dict[str, dict[str, float]]:
    """
    End-to-end Path-2 forward photometry: PHOENIX → dilute → [+BB] → F99 → synth mags.

    Parameters
    ----------
    grid :
        :class:`PhoenixGrid` instance (real or mocked).
    teff_k, logg, mh, alpha, a_v :
        Atmosphere + extinction (F99 applied to this single-star SED).
    bands :
        Registry band names.
    radius_cm, distance_pc :
        Optional geometric dilution (both or neither).
    systems, bandpasses, r_v :
        Forwarded to :func:`synthesize_mags` / F99.
    bb_t_k, bb_l_lsun :
        Optional IR blackbody temperature (K) and luminosity (L☉).
        When both are provided, BB FLAM is added to the stellar FLAM **before**
        F99 extinction, consistent with Path-2 SED order.
        Requires ``distance_pc`` to be set.

    Returns
    -------
    dict
        Per-band AB/Vega magnitudes.

    Limits
    ------
    1-star convenience wrapper. Multi-component models must sum diluted
    spectra, call F99 once on the sum, then :func:`synthesize_mags`.
    Same as :meth:`PhoenixGrid.extincted_spectrum` and :func:`synthesize_mags`.

    Sets :meth:`PhoenixGrid.set_photometry_wavelengths` from the bandpass
    waveset union so interp+F99 run on that grid (not full HiRes).
    """
    from darkhunter_sed.filters_synphot import (
        bandpass_wavelength_grid,
        load_bandpass,
    )

    bps: Mapping[str, object]
    if bandpasses is not None and grid._phot_wave is not None:
        # Photometry λ-grid already set by the caller (e.g. fit_1star_dynesty).
        # Skip the O(bandpasses) bandpass_wavelength_grid + np.allclose check
        # that was otherwise paid on every dynesty likelihood evaluation.
        bps = bandpasses
    else:
        bps = bandpasses if bandpasses is not None else {b: load_bandpass(b) for b in bands}
        grid.set_photometry_wavelengths(bandpass_wavelength_grid(bps))
    wave, flux = grid.extincted_spectrum(
        teff_k,
        logg,
        mh,
        alpha,
        0.0,  # apply F99 after summing with BB (if any)
        radius_cm=radius_cm,
        distance_pc=distance_pc,
        r_v=r_v,
    )
    if bb_t_k is not None and bb_l_lsun is not None:
        if distance_pc is None:
            raise ValueError("distance_pc is required for BB component")
        flux = flux + bb_flux_flam_at_earth(wave, bb_t_k, bb_l_lsun, distance_pc)
    alav = grid._phot_alav_curve(r_v)
    flux = apply_f99_extinction(wave, flux, float(a_v), r_v=r_v, alambda_over_av=alav)
    return synthesize_mags(
        wave,
        flux,
        bands,
        systems=systems,
        bandpasses=bps,
    )


def phoenix_synth_phot_2star(
    grid: PhoenixGrid,
    mist1: object,
    mist2: object,
    mh: float,
    alpha: float,
    a_v: float,
    bands: Sequence[str],
    *,
    distance_pc: float,
    systems: Sequence[str] = ("ab",),
    bandpasses: Mapping[str, object] | None = None,
    r_v: float = DEFAULT_R_V,
    bb_t_k: float | None = None,
    bb_l_lsun: float | None = None,
) -> dict[str, dict[str, float]]:
    """
    2-star coeval Path-2 forward photometry: dilute both stars → sum → [+BB] → F99 → mags.

    Parameters
    ----------
    grid :
        :class:`PhoenixGrid` instance with photometry wavelengths set (or will
        be set from ``bandpasses``).
    mist1, mist2 :
        :class:`~darkhunter_sed.misty_iso.MistPoint` for primary and secondary.
        Each must expose ``teff_k``, ``logg``, and ``radius_cm``.
    mh, alpha :
        Shared metallicity ``[M/H]`` and alpha enhancement ``[α/Fe]``.
    a_v :
        Shared visual extinction (mag); applied **once** to the summed SED.
    bands :
        Registry band names.
    distance_pc :
        Shared heliocentric distance in pc (``> 0``).
    systems, bandpasses, r_v :
        Forwarded to :func:`synthesize_mags` / F99.
    bb_t_k, bb_l_lsun :
        Optional IR blackbody temperature (K) and luminosity (L☉).
        When both are provided, BB FLAM is added to the combined stellar SED
        **before** F99 extinction.

    Returns
    -------
    dict
        ``{band: {"ab": float}}`` (and "vega" if requested) for the combined SED.

    Limits
    ------
    Path-2 multi-component order: dilute star 1 (no extinction) + dilute star 2
    (no extinction) [+ BB] → **sum** on the bandpass-λ grid → F99 applied once to the
    combined SED → synthesize mags.  The ``extincted_spectrum(a_v=0)`` fast path
    avoids double extinction.  Both stars share the same ``mh``, ``alpha``, and
    photometry wavelength grid.
    """
    from darkhunter_sed.filters_synphot import (
        bandpass_wavelength_grid,
        load_bandpass,
    )

    bps: Mapping[str, object]
    if bandpasses is not None and grid._phot_wave is not None:
        bps = bandpasses
    else:
        bps = bandpasses if bandpasses is not None else {b: load_bandpass(b) for b in bands}
        grid.set_photometry_wavelengths(bandpass_wavelength_grid(bps))

    # Star 1: interpolate on photometry grid, dilute to Earth; a_v=0 is a no-op
    # in apply_f99_extinction so this is efficient.
    wave, flux1 = grid.extincted_spectrum(
        float(mist1.teff_k),  # type: ignore[attr-defined]
        float(mist1.logg),    # type: ignore[attr-defined]
        float(mh),
        float(alpha),
        0.0,
        radius_cm=float(mist1.radius_cm),    # type: ignore[attr-defined]
        distance_pc=float(distance_pc),
        r_v=r_v,
    )
    # Star 2: same grid, same distance, different atmosphere
    _, flux2 = grid.extincted_spectrum(
        float(mist2.teff_k),  # type: ignore[attr-defined]
        float(mist2.logg),    # type: ignore[attr-defined]
        float(mh),
        float(alpha),
        0.0,
        radius_cm=float(mist2.radius_cm),    # type: ignore[attr-defined]
        distance_pc=float(distance_pc),
        r_v=r_v,
    )
    # Sum diluted spectra (and optional BB), then apply F99 once to the combined SED.
    combined = flux1 + flux2
    if bb_t_k is not None and bb_l_lsun is not None:
        combined = combined + bb_flux_flam_at_earth(wave, bb_t_k, bb_l_lsun, float(distance_pc))
    alav = grid._phot_alav_curve(r_v)
    combined = apply_f99_extinction(wave, combined, float(a_v), r_v=r_v, alambda_over_av=alav)

    return synthesize_mags(wave, combined, bands, systems=systems, bandpasses=bps)
