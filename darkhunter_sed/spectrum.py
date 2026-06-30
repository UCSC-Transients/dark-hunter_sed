"""Spectrum preparation for uberMS using dark-hunter_rv continuum normalization."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from darkhunter_rv import continuum, io_utils
from darkhunter_rv.blaze import BlazeCalibration

from darkhunter_sed.config import blaze_calibration_path
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


def _coalesce_segments(
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    dedup_tol: float = 5e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RV pipeline continuum + CR despike on one echelle order."""
    mode = continuum_mode
    blaze_model = None
    if blaze_calibration is not None:
        blaze_model = blaze_calibration.model_for_order(int(echelle_order))
    if blaze_model is None and mode.startswith("sinc_blaze"):
        mode = "spline"
        logger.debug("Order %s: no blaze model; spline continuum", echelle_order)

    nw, nf, ne = continuum.fit_continuum(
        wavelength,
        flux,
        eflux,
        continuum_mode=mode,
        blaze_model=blaze_model,
        echelle_order=int(echelle_order),
    )
    return continuum.despike_normalized_pre_ccf(nw, nf, ne)


def process_apf_txt_spectrum(
    txt_path: str | Path,
    *,
    wave_range: tuple[float, float] = _DEFAULT_WAVE_RANGE,
    blaze_calibration: BlazeCalibration | None = None,
    continuum_mode: str = "sinc_blaze_only",
    dedup_tol: float = 5e-3,
) -> dict[str, np.ndarray]:
    """
    Read APF Gaia export, normalize per order (RV blaze + CR rejection), coalesce to 1D.

    Returns uberMS-ready dict with ``obs_wave``, ``obs_flux``, ``obs_eflux`` (vacuum, median-norm).
    """
    txt_path = Path(txt_path)
    _header, spectrum_data = io_utils.read_spectrum(str(txt_path))
    if blaze_calibration is None:
        blaze_calibration = load_blaze_calibration()

    wave_min, wave_max = wave_range
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for order_num in sorted(spectrum_data.keys()):
        order = spectrum_data[order_num]
        w = np.asarray(order["wavelength"], dtype=float)
        if not np.any((w >= wave_min) & (w <= wave_max)):
            continue
        f = np.asarray(order["flux"], dtype=float)
        e = np.asarray(order["eflux"], dtype=float)
        nw, nf, ne = normalize_order_chunk(
            w,
            f,
            e,
            echelle_order=int(order_num),
            blaze_calibration=blaze_calibration,
            continuum_mode=continuum_mode,
        )
        m = (nw >= wave_min) & (nw <= wave_max)
        if np.any(m):
            segments.append((nw[m], nf[m], ne[m]))

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
