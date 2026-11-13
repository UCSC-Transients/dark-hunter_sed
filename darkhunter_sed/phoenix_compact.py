"""
Compact (rebinned) PHOENIX ACES grid: flux-conserving log-λ rebinning + FITS I/O.

The HiRes ACES archive (~1.57M λ per spectrum, ~6 MB per file) is reduced to a
constant-resolution log-λ grid (default R = λ/Δλ = 2000, ~9.4k bins over
~500–55000 Å, ~45 KB per file) that is sufficient for Path-2 broadband
synthesis. Used by ``scripts/build_phoenix_compact.py`` (builder) and read back
through :class:`darkhunter_sed.phoenix_grid.PhoenixGrid` (layout auto-detect).

Compact layout (same subdirectories as HiRes)::

    WAVE_PHOENIX-ACES-AGSS-COND-2011-R{R}.fits      # primary: bin centers (Å, vacuum)
                                                     # EDGES HDU: bin edges (Å)
    PHOENIX-ACES-AGSS-COND-2011/Z-0.0/
        lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-R{R}.fits   # float32 F_λ

Compact spectra keep the native units (``erg/s/cm^2/cm``) and every ``PHX*``
header card of the source, plus ``CMP*`` provenance cards.
"""

from __future__ import annotations

import html
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from astropy import units as u
from astropy.io import fits
from numpy.typing import NDArray
from synphot import SpectralElement
from synphot.models import Empirical1D

from darkhunter_sed.phoenix_grid import (
    COMPACT_EDGES_EXTNAME,
    PhoenixFileTags,
    compact_flux_name,
    compact_wave_name,
    load_compact_edges,
    parse_phoenix_filename,
    synthesize_mags,
)

DEFAULT_RESOLUTION = 2000
DEFAULT_HIRES_URL = "https://ftp.gwdg.de/pub/misc/phoenix/HiResFITS/"
GRID_SUBDIR = "PHOENIX-ACES-AGSS-COND-2011"
HIRES_WAVE_NAME = "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
NATIVE_FLUX_UNIT = "erg/s/cm^2/cm"
COMPACT_METHOD = "flux-conserving bin mean: exact integral of linear F_lam"
COMPACT_FORMAT_VERSION = 1

# Nominal HiRes archive properties; used only for --dry-run size estimates.
NOMINAL_HIRES_N_WAVE = 1_569_128
NOMINAL_HIRES_WAVE_RANGE_AA = (500.0, 55000.0)

_FITS_BLOCK = 2880
_HREF_RE = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


def log_lambda_edges(
    wave_min: float,
    wave_max: float,
    resolution: float,
) -> NDArray[np.float64]:
    """
    Constant-resolution (uniform in ln λ) bin edges inside ``[wave_min, wave_max]``.

    Parameters
    ----------
    wave_min, wave_max :
        Wavelength range in Å (``0 < wave_min < wave_max``); typically the
        first/last native HiRes sample so no bin extends past coverage.
    resolution :
        Resolving power ``R``; consecutive edges satisfy
        ``ln(e[k+1]/e[k]) = 1/R`` (so ``λ/Δλ ≈ R`` per bin).

    Returns
    -------
    NDArray[np.float64]
        Strictly increasing edges, ``edges[0] == wave_min`` and
        ``edges[-1] <= wave_max`` (the red remainder < one bin is dropped).

    Limits
    ------
    Raises ``ValueError`` if the range holds less than one bin.
    """
    lo = float(wave_min)
    hi = float(wave_max)
    r = float(resolution)
    if not (0.0 < lo < hi) or not np.isfinite(hi):
        raise ValueError(f"Need 0 < wave_min < wave_max, got {lo}, {hi}")
    if not (r > 0.0 and np.isfinite(r)):
        raise ValueError(f"resolution must be positive, got {resolution}")
    n_bins = int(np.floor(np.log(hi / lo) * r))
    if n_bins < 1:
        raise ValueError(f"Range [{lo}, {hi}] Å holds no bin at R={r}")
    edges = lo * np.exp(np.arange(n_bins + 1, dtype=np.float64) / r)
    edges[0] = lo
    edges[-1] = min(float(edges[-1]), hi)
    return edges


def bin_centers(edges: NDArray[np.floating]) -> NDArray[np.float64]:
    """Geometric-mean bin centers ``sqrt(e[k] e[k+1])`` (Å) for log-λ edges."""
    e = np.asarray(edges, dtype=np.float64)
    return np.sqrt(e[:-1] * e[1:])


def _linear_cumulative_integral(
    wave: NDArray[np.float64],
    flux: NDArray[np.float64],
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """∫_{wave[0]}^{x} F dλ for piecewise-linear ``F`` on ``wave`` (vectorized)."""
    seg = np.diff(wave)
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (flux[1:] + flux[:-1]) * seg)))
    idx = np.clip(np.searchsorted(wave, x, side="right") - 1, 0, wave.size - 2)
    dx = x - wave[idx]
    slope = (flux[idx + 1] - flux[idx]) / seg[idx]
    return cum[idx] + flux[idx] * dx + 0.5 * slope * dx * dx


def rebin_flux_conserving(
    wave: NDArray[np.floating],
    flux: NDArray[np.floating],
    edges: NDArray[np.floating],
) -> NDArray[np.float64]:
    """
    Flux-conserving rebinning: mean of ``F_λ`` over each output bin.

    Parameters
    ----------
    wave :
        Strictly increasing native wavelengths (Å), any (non-uniform) spacing.
    flux :
        Native flux density on ``wave`` (any per-wavelength unit).
    edges :
        Strictly increasing output bin edges (Å) inside ``[wave[0], wave[-1]]``.

    Returns
    -------
    NDArray[np.float64]
        ``(1/Δ_k) ∫_{e_k}^{e_{k+1}} F dλ`` for each bin, same unit as ``flux``.

    Limits
    ------
    ``F`` is treated as piecewise linear between native samples (the same
    model as trapezoidal integration), so ``Σ mean_k Δ_k`` equals the
    trapezoidal integral of the native spectrum over ``[e_0, e_N]`` to
    rounding. Edges outside native coverage raise (no extrapolation).
    """
    w = np.asarray(wave, dtype=np.float64).reshape(-1)
    f = np.asarray(flux, dtype=np.float64).reshape(-1)
    e = np.asarray(edges, dtype=np.float64).reshape(-1)
    if w.shape != f.shape or w.size < 2:
        raise ValueError("wave and flux must be equal-length 1-D arrays (≥2 samples)")
    if e.size < 2 or not np.all(np.diff(e) > 0.0):
        raise ValueError("edges must be strictly increasing with ≥2 values")
    if not np.all(np.diff(w) > 0.0):
        raise ValueError("native wave must be strictly increasing")
    if e[0] < w[0] or e[-1] > w[-1]:
        raise ValueError(
            f"edges [{e[0]}, {e[-1]}] extend outside native coverage [{w[0]}, {w[-1]}]"
        )
    integral = _linear_cumulative_integral(w, f, e)
    return np.diff(integral) / np.diff(e)


def _atomic_writeto(hdul: fits.HDUList, path: Path) -> None:
    """Write ``hdul`` with checksums to a temp file next to ``path``, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    os.close(fd)
    try:
        hdul.writeto(tmp, overwrite=True, checksum=True)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_compact_wave(
    path: Path | str,
    edges: NDArray[np.floating],
    *,
    resolution: int,
    source_wave: str,
) -> None:
    """
    Write the shared compact wavelength file (centers + ``EDGES`` HDU).

    Parameters
    ----------
    path :
        Output path, normally ``dest / compact_wave_name(resolution)``.
    edges :
        Bin edges (Å, vacuum) from :func:`log_lambda_edges`.
    resolution :
        Resolving power ``R`` recorded in ``CMPRES``.
    source_wave :
        Provenance label (HiRes WAVE filename or URL).
    """
    e = np.asarray(edges, dtype=np.float64)
    primary = fits.PrimaryHDU(bin_centers(e))
    hdr = primary.header
    hdr["WAVEUNIT"] = ("Angstrom", "bin centers, vacuum")
    hdr["AIRORVAC"] = ("vac", "vacuum wavelengths (as HiRes)")
    hdr["CMPRES"] = (int(resolution), "constant resolving power R; dln(lam)=1/R")
    hdr["CMPNBIN"] = (int(e.size - 1), "number of bins")
    hdr["CMPWMIN"] = (float(e[0]), "[Angstrom] first bin edge")
    hdr["CMPWMAX"] = (float(e[-1]), "[Angstrom] last bin edge")
    hdr["CMPCENT"] = ("geometric", "centers = sqrt(e_k * e_k+1)")
    hdr["CMPMETH"] = COMPACT_METHOD
    hdr["CMPSRC"] = str(source_wave)
    hdr["CMPVER"] = COMPACT_FORMAT_VERSION
    edges_hdu = fits.ImageHDU(e, name=COMPACT_EDGES_EXTNAME)
    edges_hdu.header["WAVEUNIT"] = ("Angstrom", "bin edges, vacuum")
    _atomic_writeto(fits.HDUList([primary, edges_hdu]), Path(path))


def read_compact_wave(path: Path | str) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """
    Read a compact wavelength file.

    Returns
    -------
    centers, edges, resolution :
        Bin centers (Å), bin edges (Å, length ``n+1``) and ``CMPRES``.
    """
    with fits.open(path, memmap=False) as hdul:
        centers = np.asarray(hdul[0].data, dtype=np.float64).reshape(-1)
        resolution = int(hdul[0].header["CMPRES"])
    edges = load_compact_edges(path)
    if edges.size != centers.size + 1:
        raise ValueError(f"{path}: {edges.size} edges for {centers.size} centers")
    return centers, edges, resolution


def compact_flux_header(
    src_header: fits.Header,
    *,
    resolution: int,
    source_name: str,
    n_native: int,
    n_bins: int,
) -> fits.Header:
    """
    Header for a compact spectrum: source ``PHX*`` cards, unit, provenance.

    Parameters
    ----------
    src_header :
        Primary header of the HiRes spectrum.
    resolution, n_native, n_bins :
        Recorded as ``CMPRES``, ``CMPNNAT``, ``CMPNBIN``.
    source_name :
        Source filename or URL (``CMPSRC``).

    Returns
    -------
    astropy.io.fits.Header
        Cards to merge into the compact primary HDU (structural cards excluded).
    """
    out = fits.Header()
    for card in src_header.cards:
        if card.keyword.startswith("PHX"):
            out[card.keyword] = (card.value, card.comment)
    out["BUNIT"] = (str(src_header.get("BUNIT", NATIVE_FLUX_UNIT)), "native PHOENIX unit")
    out["CMPRES"] = (int(resolution), "constant resolving power R; dln(lam)=1/R")
    out["CMPNBIN"] = (int(n_bins), "number of compact bins")
    out["CMPNNAT"] = (int(n_native), "number of native HiRes samples")
    out["CMPWAVE"] = (compact_wave_name(resolution), "shared wavelength file")
    out["CMPMETH"] = COMPACT_METHOD
    out["CMPSRC"] = str(source_name)
    out["CMPVER"] = COMPACT_FORMAT_VERSION
    return out


def write_compact_flux(
    path: Path | str,
    flux_binned: NDArray[np.floating],
    header: fits.Header,
) -> None:
    """Atomically write a float32 compact spectrum with checksums."""
    hdu = fits.PrimaryHDU(np.asarray(flux_binned, dtype=np.float32))
    hdu.header.extend(header, update=True)
    _atomic_writeto(fits.HDUList([hdu]), Path(path))


def compact_flux_is_valid(path: Path | str, *, n_bins: int, resolution: int) -> bool:
    """
    True if ``path`` is a complete compact spectrum for this grid (resume check).

    Parameters
    ----------
    path :
        Candidate compact spectrum.
    n_bins, resolution :
        Expected length and ``CMPRES``.

    Limits
    ------
    Verifies FITS ``CHECKSUM``/``DATASUM``, length, ``CMPRES`` and finiteness;
    any read error counts as invalid (the model is rebuilt).
    """
    p = Path(path)
    if not p.is_file():
        return False
    try:
        with fits.open(p, memmap=False) as hdul:
            hdu = hdul[0]
            if hdu.verify_checksum() != 1 or hdu.verify_datasum() != 1:
                return False
            if int(hdu.header.get("CMPRES", -1)) != int(resolution):
                return False
            data = hdu.data
            if data is None or data.shape != (int(n_bins),):
                return False
            return bool(np.all(np.isfinite(data)))
    except (OSError, ValueError, TypeError, KeyError):
        return False


def build_compact_from_hires(
    src_path: Path | str,
    dest_path: Path | str,
    *,
    native_wave: NDArray[np.float64],
    edges: NDArray[np.float64],
    resolution: int,
    source_name: str | None = None,
) -> NDArray[np.float64]:
    """
    Rebin one HiRes spectrum file and write its compact counterpart.

    Parameters
    ----------
    src_path :
        HiRes spectrum FITS (local file; may be a temp download).
    dest_path :
        Compact output path.
    native_wave :
        HiRes wavelength array (Å), same length as the flux.
    edges :
        Compact bin edges (Å).
    resolution :
        Resolving power ``R`` (provenance + resume check).
    source_name :
        Provenance label; default ``src_path.name``.

    Returns
    -------
    NDArray[np.float64]
        Rebinned flux before float32 storage (native units).
    """
    src = Path(src_path)
    with fits.open(src, memmap=False) as hdul:
        flux = np.asarray(hdul[0].data, dtype=np.float64).reshape(-1)
        src_header = hdul[0].header.copy()
    if flux.size != native_wave.size:
        raise ValueError(
            f"{src.name}: flux length {flux.size} != native wavelength {native_wave.size}"
        )
    if not np.all(np.isfinite(flux)):
        raise ValueError(f"{src.name}: non-finite flux values")
    binned = rebin_flux_conserving(native_wave, flux, edges)
    header = compact_flux_header(
        src_header,
        resolution=resolution,
        source_name=source_name or src.name,
        n_native=flux.size,
        n_bins=binned.size,
    )
    write_compact_flux(dest_path, binned, header)
    return binned


@dataclass(frozen=True, slots=True)
class ListingEntry:
    """One entry of an HTTP directory listing (URL-decoded name)."""

    name: str
    is_dir: bool


def parse_directory_listing(page: str) -> list[ListingEntry]:
    """
    Extract child entries from an Apache/nginx-style HTML directory index.

    Parameters
    ----------
    page :
        HTML text of the listing.

    Returns
    -------
    list[ListingEntry]
        Unique entries in page order. Names are HTML-unescaped and URL-decoded
        (``%2B`` → ``+``, ``%3D`` → ``=``); directories keep no trailing slash.

    Limits
    ------
    Skips parent links, absolute/query links and nested paths; only direct
    children are returned.
    """
    seen: set[str] = set()
    out: list[ListingEntry] = []
    for href in _HREF_RE.findall(page):
        raw = html.unescape(href)
        if raw.startswith(("/", "?", "#", "..")) or "://" in raw:
            continue
        is_dir = raw.endswith("/")
        name = unquote(raw.rstrip("/"))
        if not name or "/" in name or name in seen:
            continue
        seen.add(name)
        out.append(ListingEntry(name=name, is_dir=is_dir))
    return out


def tags_in_range(
    tags: PhoenixFileTags,
    *,
    teff_min: float | None = None,
    teff_max: float | None = None,
    logg_min: float | None = None,
    logg_max: float | None = None,
) -> bool:
    """True if ``tags`` lies inside the inclusive Teff/logg filter box (``None`` = open)."""
    if teff_min is not None and tags.teff_k < teff_min:
        return False
    if teff_max is not None and tags.teff_k > teff_max:
        return False
    if logg_min is not None and tags.logg < logg_min:
        return False
    if logg_max is not None and tags.logg > logg_max:
        return False
    return True


@dataclass(frozen=True, slots=True)
class CompactJob:
    """
    One HiRes → compact conversion.

    Parameters
    ----------
    subgrid :
        Subgrid directory name (``Z-0.0``, ``Z-0.5.Alpha=+0.40``).
    hires_name :
        URL-decoded HiRes filename.
    tags :
        Parsed atmosphere labels.
    compact_path :
        Destination compact spectrum path.
    """

    subgrid: str
    hires_name: str
    tags: PhoenixFileTags
    compact_path: Path


def make_jobs(
    dest: Path | str,
    subgrid: str,
    hires_names: Iterable[str],
    *,
    resolution: int,
    teff_min: float | None = None,
    teff_max: float | None = None,
    logg_min: float | None = None,
    logg_max: float | None = None,
) -> list[CompactJob]:
    """
    Build conversion jobs for HiRes filenames in one subgrid, applying filters.

    Parameters
    ----------
    dest :
        Compact grid root.
    subgrid :
        Subgrid directory name.
    hires_names :
        Candidate filenames (non-HiRes or unparsable names are ignored).
    resolution :
        Compact resolving power (selects the output suffix).
    teff_min, teff_max, logg_min, logg_max :
        Inclusive filters; ``None`` leaves that side open.

    Returns
    -------
    list[CompactJob]
        Sorted by (Teff, logg) for deterministic ordering.
    """
    out: list[CompactJob] = []
    sub_dir = Path(dest) / GRID_SUBDIR / subgrid
    for name in hires_names:
        tags = parse_phoenix_filename(name)
        if tags is None or tags.resolution is not None:
            continue
        if not tags_in_range(
            tags, teff_min=teff_min, teff_max=teff_max, logg_min=logg_min, logg_max=logg_max
        ):
            continue
        out.append(
            CompactJob(
                subgrid=subgrid,
                hires_name=name,
                tags=tags,
                compact_path=sub_dir / compact_flux_name(name, resolution),
            )
        )
    out.sort(key=lambda j: (j.tags.teff_k, j.tags.logg, j.hires_name))
    return out


def split_done(
    jobs: Sequence[CompactJob],
    *,
    n_bins: int | None,
    resolution: int,
) -> tuple[list[CompactJob], list[CompactJob]]:
    """
    Partition jobs into ``(todo, done)`` using :func:`compact_flux_is_valid`.

    Parameters
    ----------
    jobs :
        Planned conversions.
    n_bins :
        Expected compact length; ``None`` (no compact WAVE file yet) marks
        every job as to-do.
    resolution :
        Expected ``CMPRES``.
    """
    if n_bins is None:
        return list(jobs), []
    todo: list[CompactJob] = []
    done: list[CompactJob] = []
    for job in jobs:
        ok = compact_flux_is_valid(job.compact_path, n_bins=n_bins, resolution=resolution)
        (done if ok else todo).append(job)
    return todo, done


def fits_image_bytes(n_values: int, *, bytes_per_value: int, header_blocks: int = 3) -> int:
    """Approximate on-disk size of a 1-D FITS image (header + padded data)."""
    data_blocks = -(-int(n_values) * int(bytes_per_value) // _FITS_BLOCK)
    return (int(header_blocks) + data_blocks) * _FITS_BLOCK


def expected_n_bins(
    resolution: int,
    wave_range: tuple[float, float] = NOMINAL_HIRES_WAVE_RANGE_AA,
) -> int:
    """Number of compact bins for ``resolution`` over ``wave_range`` (Å)."""
    return int(log_lambda_edges(wave_range[0], wave_range[1], resolution).size - 1)


@dataclass(frozen=True, slots=True)
class BandComparison:
    """
    AB magnitudes of one model in one band from four synthesis routes.

    Attributes
    ----------
    reference :
        Native HiRes with the throughput resampled onto the native grid
        (exact integral of the native spectrum; the truth).
    compact_exact :
        Compact spectrum through the same resampled throughput — isolates the
        rebinning error controlled by ``R``.
    compact_pipeline :
        Compact spectrum through the stock bandpass waveset (what Path-2 fits do).
    native_pipeline :
        HiRes through the stock bandpass waveset (what HiRes fits did; point
        sampling of a line-rich spectrum on the waveset).
    """

    reference: float
    compact_exact: float
    compact_pipeline: float
    native_pipeline: float


def _resampled_bandpass(bp: SpectralElement, native_wave: NDArray[np.float64]) -> SpectralElement:
    """Bandpass sampled on its own knots ∪ native samples inside its waveset."""
    knots = np.asarray(bp.waveset.to(u.AA).value, dtype=np.float64)
    inside = native_wave[(native_wave >= knots[0]) & (native_wave <= knots[-1])]
    fine = np.union1d(knots, inside)
    thru = np.asarray(bp(fine * u.AA).value, dtype=np.float64)
    return SpectralElement(Empirical1D, points=fine * u.AA, lookup_table=thru)


def compare_band_mags(
    native_wave: NDArray[np.floating],
    native_flam: NDArray[np.floating],
    compact_wave: NDArray[np.floating],
    compact_flam: NDArray[np.floating],
    bandpasses: Mapping[str, SpectralElement],
) -> dict[str, BandComparison]:
    """
    Compare synthetic AB magnitudes of a HiRes spectrum and its compact version.

    Parameters
    ----------
    native_wave, native_flam :
        HiRes wavelengths (Å) and FLAM.
    compact_wave, compact_flam :
        Compact bin centers (Å) and FLAM (as stored, i.e. float32-rounded).
    bandpasses :
        ``band → SpectralElement`` (e.g. every registered Path-2 band).

    Returns
    -------
    dict[str, BandComparison]
        Per-band magnitudes from :func:`~darkhunter_sed.phoenix_grid.synthesize_mags`.

    Limits
    ------
    Units cancel in the differences, so surface FLAM (no dilution) is fine.
    Bands with no flux return the ``99`` sentinel from ``synthesize_mags``.
    """
    nw = np.asarray(native_wave, dtype=np.float64)
    nf = np.asarray(native_flam, dtype=np.float64)
    cw = np.asarray(compact_wave, dtype=np.float64)
    cf = np.asarray(compact_flam, dtype=np.float64)
    out: dict[str, BandComparison] = {}
    for band, bp in bandpasses.items():
        fine = {band: _resampled_bandpass(bp, nw)}
        stock = {band: bp}
        out[band] = BandComparison(
            reference=synthesize_mags(nw, nf, [band], bandpasses=fine)[band]["ab"],
            compact_exact=synthesize_mags(cw, cf, [band], bandpasses=fine)[band]["ab"],
            compact_pipeline=synthesize_mags(cw, cf, [band], bandpasses=stock)[band]["ab"],
            native_pipeline=synthesize_mags(nw, nf, [band], bandpasses=stock)[band]["ab"],
        )
    return out
