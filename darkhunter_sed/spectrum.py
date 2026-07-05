"""Spectrum preparation for uberMS using dark-hunter_rv continuum normalization."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from darkhunter_rv import continuum, io_utils
from darkhunter_rv.blaze import BlazeCalibration, strong_lines_in_span

from darkhunter_sed.config import (
    SPECTRUM_ORDER_END_DEV_TOL,
    SPECTRUM_ORDER_END_MIN_PIXELS,
    blaze_calibration_path,
)
from darkhunter_sed.region_picker import (
    OrderRegions,
    build_manual_base_mask,
    fit_model_from_regions,
    load_regions_json,
    order_blaze_model_from_regions,
    order_regions_from_doc,
    poly_order_from_regions,
)
from darkhunter_sed.stellar_data import import_airtovacuum

logger = logging.getLogger(__name__)

airtovacuum = import_airtovacuum()

_DEFAULT_WAVE_RANGE = (5150.0, 5300.0)
_EPOCH_NUM_RE = re.compile(r"epoch_(\d+)", re.I)


def load_blaze_calibration(path: Path | None = None) -> BlazeCalibration | None:
    p = path or blaze_calibration_path()
    if not p.is_file():
        logger.warning("Blaze calibration not found at %s; falling back to spline continuum", p)
        return None
    return BlazeCalibration.load(p)


def _trim_normalized_order_ends(
    wavelength: np.ndarray,
    flux: np.ndarray,
    eflux: np.ndarray,
    *,
    dev_tol: float = SPECTRUM_ORDER_END_DEV_TOL,
    min_pixels: int = SPECTRUM_ORDER_END_MIN_PIXELS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Peel noisy blaze-normalized pixels from both order ends.

    Overlap wings often rise to ~1.05 after per-order normalization. Trim from each
    end while ``|flux - 1| > dev_tol``; stop once a pixel is within tolerance. Keep at
    least ``min_pixels``; if trimming would drop below that, return the order untrimmed.
    """
    w = np.asarray(wavelength, float)
    f = np.asarray(flux, float)
    e = np.asarray(eflux, float)
    n = w.size
    if n <= int(min_pixels):
        return w, f, e

    dev = np.abs(f - 1.0)
    lo = 0
    while lo < n and (not np.isfinite(f[lo]) or dev[lo] > dev_tol):
        lo += 1
    hi = n - 1
    while hi > lo and (not np.isfinite(f[hi]) or dev[hi] > dev_tol):
        hi -= 1

    if hi - lo + 1 < int(min_pixels):
        logger.debug("order-end trim would leave <%d px; keeping full order", min_pixels)
        return w, f, e
    sl = slice(lo, hi + 1)
    return w[sl], f[sl], e[sl]


def _coalesce_segments(
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    dedup_tol: float = 5e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge overlapping order segments to a 1D spectrum.

    Duplicate wavelengths (within ``dedup_tol``) collapse via inverse-variance
    weighting (``w = 1/eflux**2``): low-error pixels dominate the merged flux and the
    combined error is ``sqrt(1/sum(w))``.
    """
    waves_list: list[float] = []
    flux_list: list[float] = []
    e_list: list[float] = []
    for w, f, e in segments:
        waves_list.extend(np.asarray(w, float).tolist())
        flux_list.extend(np.asarray(f, float).tolist())
        e_list.extend(np.maximum(np.asarray(e, float), 1e-99).tolist())

    if not waves_list:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    waves = np.array(waves_list, dtype=float)
    fluxes = np.array(flux_list, dtype=float)
    efluxes = np.array(e_list, dtype=float)

    order_idx = np.argsort(waves)
    waves = waves[order_idx]
    fluxes = fluxes[order_idx]
    efluxes = efluxes[order_idx]

    w_out: list[float] = []
    f_out: list[float] = []
    e_out: list[float] = []
    n = len(waves)
    i = 0
    while i < n:
        j = i + 1
        w0 = waves[i]
        while j < n and waves[j] - w0 <= dedup_tol:
            j += 1
        sl = slice(i, j)
        ww = waves[sl]
        ff = fluxes[sl]
        ee = efluxes[sl]
        iv = 1.0 / (ee**2)
        w_mean = float(np.average(ww, weights=iv))
        f_mean = float(np.average(ff, weights=iv))
        e_comb = float(np.sqrt(1.0 / np.sum(iv)))
        w_out.append(w_mean)
        f_out.append(f_mean)
        e_out.append(e_comb)
        i = j

    return (
        np.array(w_out, dtype=float),
        np.array(f_out, dtype=float),
        np.array(e_out, dtype=float),
    )


def normalize_order_chunk(
    wavelength: np.ndarray,
    flux: np.ndarray,
    eflux: np.ndarray,
    *,
    echelle_order: int,
    blaze_calibration: BlazeCalibration | None,
    continuum_mode: str = "sinc_blaze_only",
    order_regions: OrderRegions | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RV pipeline continuum + CR despike on one echelle order."""
    mode = continuum_mode
    w = np.asarray(wavelength, dtype=float)
    f = np.asarray(flux, dtype=float)
    fallback = None
    if blaze_calibration is not None:
        fallback = blaze_calibration.model_for_order(int(echelle_order))
    blaze_model = order_blaze_model_from_regions(
        order_regions,
        int(echelle_order),
        w,
        fallback,
    )
    if blaze_model is None and mode.startswith("sinc_blaze"):
        mode = "spline"
        logger.debug("Order %s: no blaze model; spline continuum", echelle_order)

    mask_kwargs: dict = {}
    fit_kwargs: dict = {}
    if order_regions is not None and mode.startswith("sinc_blaze") and blaze_model is not None:
        rests = strong_lines_in_span(float(np.min(w)), float(np.max(w)))
        if order_regions.get("continuum_regions") or order_regions.get("line_regions"):
            bundle = build_manual_base_mask(
                w,
                f,
                continuum_regions=order_regions.get("continuum_regions"),
                line_regions=order_regions.get("line_regions"),
                rest_lines=rests if rests else None,
                half_width_angstrom=blaze_model.line_mask_half_width_angstrom,
            )
            mask_kwargs = {
                "base_mask": bundle.base_mask,
                "fixed_line_mask": bundle.fixed_line_mask,
                "fixed_cont_mask": bundle.fixed_cont_mask,
                "cr_mask": bundle.cr_mask,
            }
        fit_kwargs = {
            "fit_model": fit_model_from_regions(order_regions),
            "poly_order": poly_order_from_regions(order_regions),
        }

    nw, nf, ne = continuum.fit_continuum(
        wavelength,
        flux,
        eflux,
        continuum_mode=mode,
        blaze_model=blaze_model,
        echelle_order=int(echelle_order),
        **mask_kwargs,
        **fit_kwargs,
    )
    return continuum.despike_normalized_pre_ccf(nw, nf, ne)


def process_apf_txt_spectrum(
    txt_path: str | Path,
    *,
    wave_range: tuple[float, float] = _DEFAULT_WAVE_RANGE,
    blaze_calibration: BlazeCalibration | None = None,
    continuum_mode: str = "sinc_blaze_only",
    dedup_tol: float = 5e-3,
    regions_json: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """
    Read APF Gaia export, normalize per order (RV blaze + CR rejection), coalesce to 1D.

    Returns uberMS-ready dict with ``obs_wave``, ``obs_flux``, ``obs_eflux`` (vacuum, median-norm).
    """
    txt_path = Path(txt_path)
    _header, spectrum_data = io_utils.read_spectrum(str(txt_path))
    if blaze_calibration is None:
        blaze_calibration = load_blaze_calibration()

    regions_doc: dict | None = None
    if regions_json is not None:
        regions_doc = load_regions_json(regions_json)
        logger.info("Using manual regions from %s", regions_json)

    wave_min, wave_max = wave_range
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for order_num in sorted(spectrum_data.keys()):
        order = spectrum_data[order_num]
        w = np.asarray(order["wavelength"], dtype=float)
        if not np.any((w >= wave_min) & (w <= wave_max)):
            continue
        f = np.asarray(order["flux"], dtype=float)
        e = np.asarray(order["eflux"], dtype=float)
        order_regions = (
            order_regions_from_doc(regions_doc, int(order_num)) if regions_doc else None
        )
        nw, nf, ne = normalize_order_chunk(
            w,
            f,
            e,
            echelle_order=int(order_num),
            blaze_calibration=blaze_calibration,
            continuum_mode=continuum_mode,
            order_regions=order_regions,
        )
        m = (nw >= wave_min) & (nw <= wave_max)
        if np.any(m):
            tw, tf, te = _trim_normalized_order_ends(nw[m], nf[m], ne[m])
            segments.append((tw, tf, te))

    waves, fluxes, efluxes = _coalesce_segments(segments, dedup_tol=dedup_tol)
    if waves.size == 0:
        raise ValueError(f"No flux in {wave_min}–{wave_max} Å after normalization: {txt_path}")

    waves = airtovacuum(waves)
    medflux = float(np.nanmedian(fluxes))
    if not np.isfinite(medflux) or medflux == 0.0:
        raise ValueError(f"Invalid median flux after normalization: {txt_path}")
    fluxes = fluxes / medflux
    efluxes = efluxes / medflux

    return {"obs_wave": waves, "obs_flux": fluxes, "obs_eflux": efluxes}


def epoch_number_from_path(path: str | Path) -> int | None:
    m = _EPOCH_NUM_RE.search(Path(path).name)
    return int(m.group(1)) if m else None


def sort_spectrum_paths(paths: list[Path]) -> list[Path]:
    def key(p: Path) -> tuple:
        ep = epoch_number_from_path(p)
        if ep is not None:
            return (0, ep, p.name.lower())
        return (1, p.name.lower())

    return sorted(paths, key=key)


def txt_to_uberms_fits(
    txt_path: str | Path,
    fits_path: str | Path,
    **kwargs,
) -> Table:
    """Convert one epoch .txt to FITS table (wave, flux, eflux) for caching."""
    spec = process_apf_txt_spectrum(txt_path, **kwargs)
    tbl = Table(
        {
            "wave": spec["obs_wave"],
            "flux": spec["obs_flux"],
            "eflux": spec["obs_eflux"],
        }
    )
    fits_path = Path(fits_path)
    fits_path.parent.mkdir(parents=True, exist_ok=True)
    tbl.write(fits_path, overwrite=True)
    return tbl


def process_spectrum_fits(
    fits_path: str | Path,
    wave_range: tuple[float, float] = _DEFAULT_WAVE_RANGE,
) -> dict[str, np.ndarray]:
    """Read pre-converted FITS and clip/re-normalize (no RV continuum re-run)."""
    spec = Table.read(fits_path, format="fits")
    wmin, wmax = wave_range
    m = (spec["wave"] > wmin) & (spec["wave"] < wmax)
    spec = spec[m]
    spec["wave"] = airtovacuum(spec["wave"])
    medflux = float(np.nanmedian(spec["flux"]))
    spec["flux"] /= medflux
    spec["eflux"] /= medflux
    return {
        "obs_wave": np.asarray(spec["wave"], dtype=float),
        "obs_flux": np.asarray(spec["flux"], dtype=float),
        "obs_eflux": np.asarray(spec["eflux"], dtype=float),
    }
