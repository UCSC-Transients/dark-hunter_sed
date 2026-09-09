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

# Max cached HiRes flux arrays in :meth:`PhoenixGrid._load` (dynesty reuse).
_DEFAULT_FLUX_CACHE_SIZE = 128

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
    data = fits.getdata(path)
    wave = np.asarray(data, dtype=np.float64).reshape(-1)
    if wave.size < 2:
        raise ValueError(f"Wavelength array too short in {path}")
    return wave


def load_phoenix_flux(
    path: Path | str,
    *,
    to_flam: bool = True,
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

    Returns
    -------
    NDArray[np.float64]
        1-D surface flux density.

    Limits
    ------
    Reads primary HDU only. Does not apply radius/distance scaling or
    extinction.
    """
    path = Path(path)
    data = fits.getdata(path)
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
    ) -> None:
        self.root = phoenix_dir(root)
        self.to_flam = bool(to_flam)
        self._flux_loader = flux_loader
        self._flux_cache_size = max(int(flux_cache_size), 0)
        self._flux_cache: OrderedDict[Path, NDArray[np.float64]] = OrderedDict()
        # Optional photometry λ grid: dynesty evaluates SED here, not on HiRes.
        self._phot_wave: NDArray[np.float64] | None = None
        self._phot_flux_cache: OrderedDict[Path, NDArray[np.float64]] = OrderedDict()
        self._phot_alav: dict[float, NDArray[np.float64]] = {}
        if wavelength is not None:
            self.wavelength = np.asarray(wavelength, dtype=np.float64).reshape(-1)
        else:
            self.wavelength = load_phoenix_wavelength(self.root)
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
        Cache size is :attr:`_flux_cache_size` (default 128). Injected
        ``flux_loader`` results are also cached by ``point.path``.
        """
        key = point.path
        cached = self._flux_cache.get(key)
        if cached is not None:
            self._flux_cache.move_to_end(key)
            return cached
        if self._flux_loader is not None:
            flux = np.asarray(self._flux_loader(point.path), dtype=np.float64)
        else:
            flux = load_phoenix_flux(point.path, to_flam=self.to_flam)
        if flux.shape != self.wavelength.shape:
            raise ValueError(
                f"Flux length {flux.size} != wavelength {self.wavelength.size} "
                f"for {point.path}"
            )
        if self._flux_cache_size > 0:
            self._flux_cache[key] = flux
            self._flux_cache.move_to_end(key)
            while len(self._flux_cache) > self._flux_cache_size:
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
        pts = self._points_at_alpha(alpha)
        if not pts:
            raise ValueError(f"No models at [α/Fe]={alpha}")

        teffs = sorted({p.teff_k for p in pts})
        loggs = sorted({p.logg for p in pts})
        mhs = sorted({p.mh for p in pts})

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
    if den <= 0.0 or num <= 0.0 or not np.isfinite(num) or not np.isfinite(den):
        return float("nan")
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
) -> dict[str, dict[str, float]]:
    """
    End-to-end Path-2 forward photometry: PHOENIX → dilute → F99 → synth mags.

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
    if bandpasses is not None:
        bps = bandpasses
    else:
        bps = {b: load_bandpass(b) for b in bands}
    grid.set_photometry_wavelengths(bandpass_wavelength_grid(bps))
    wave, flux = grid.extincted_spectrum(
        teff_k,
        logg,
        mh,
        alpha,
        a_v,
        radius_cm=radius_cm,
        distance_pc=distance_pc,
        r_v=r_v,
    )
    return synthesize_mags(
        wave,
        flux,
        bands,
        systems=systems,
        bandpasses=bps,
    )
