"""
Path-2 photometry FITS I/O: ``band``, ``mag``, ``err``, ``flag``.

Flag convention (locked for photometry SED Path 2)
--------------------------------------------------
- ``flag=0``: detection (``mag`` / ``err`` as usual).
- ``flag=1``: 3σ upper limit; ``mag`` is the limit magnitude (plot as hat at limit).
- Legacy files without a ``flag`` column are treated as all detections (``flag=0``).

Limits
------
- Band names are stored as FITS ``32A`` (truncate/pad at write if longer).
- ``err`` must be finite and ``> 0`` for both detections and UL rows (I/O validity);
  likelihood treatment of UL ``err`` is defined in the Path-2 fitter, not here.
- Only ``flag`` values ``0`` and ``1`` are written; unknown flags on load are kept as
  integers but Path-1 detection dict builders treat non-zero as non-detections.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np
from astropy.io import fits
from astropy.table import Table

FLAG_DETECTION: int = 0
FLAG_UPPER_LIMIT: int = 1

_BAND_FMT = "32A"
_FLAG_FMT = "I"  # int16


@dataclass(frozen=True)
class PhotRow:
    """
    One photometry table row.

    Parameters
    ----------
    band :
        Filter / system name (e.g. ``GaiaDR3_G``, ``PS_g``).
    mag :
        AB/Vega magnitude for a detection, or 3σ UL magnitude when ``flag=1``.
    err :
        Magnitude uncertainty; must be finite and ``> 0`` for I/O validity.
    flag :
        ``0`` detection, ``1`` 3σ upper limit.
    """

    band: str
    mag: float
    err: float
    flag: int = FLAG_DETECTION


PhotTuple = tuple[str, float, float] | tuple[str, float, float, int]


def _as_band_str(band: object) -> str:
    if isinstance(band, bytes):
        return band.decode("ascii", errors="replace").strip()
    return str(band).strip()


def normalize_phot_rows(
    photometry: Iterable[PhotTuple | PhotRow | Mapping[str, object]],
) -> list[PhotRow]:
    """
    Coerce gather-style tuples / mappings into :class:`PhotRow` list.

    Accepts ``(band, mag, err)`` (flag defaults to detection), ``(band, mag, err, flag)``,
    :class:`PhotRow`, or mappings with keys ``band``, ``mag``, ``err``, optional ``flag``.

    Parameters
    ----------
    photometry :
        Iterable of row-like objects from catalog gather or callers.

    Returns
    -------
    list[PhotRow]
        Validated rows (finite mag/err, ``err > 0``, flag in ``{0, 1}``).

    Raises
    ------
    ValueError
        If a row cannot be parsed or fails the finite / positive-err / flag checks.
    """
    rows: list[PhotRow] = []
    for i, item in enumerate(photometry):
        if isinstance(item, PhotRow):
            row = item
        elif isinstance(item, Mapping):
            flag_raw = item.get("flag", FLAG_DETECTION)
            row = PhotRow(
                band=_as_band_str(item["band"]),
                mag=float(item["mag"]),  # type: ignore[arg-type]
                err=float(item["err"]),  # type: ignore[arg-type]
                flag=int(flag_raw) if flag_raw is not None else FLAG_DETECTION,
            )
        else:
            if len(item) == 3:
                band, mag, err = item  # type: ignore[misc]
                flag = FLAG_DETECTION
            elif len(item) == 4:
                band, mag, err, flag = item  # type: ignore[misc]
            else:
                raise ValueError(
                    f"photometry row {i}: expected 3 or 4 fields, got {len(item)}"
                )
            row = PhotRow(
                band=_as_band_str(band),
                mag=float(mag),
                err=float(err),
                flag=int(flag),
            )
        if not row.band:
            raise ValueError(f"photometry row {i}: empty band name")
        if not np.isfinite(row.mag) or not np.isfinite(row.err) or row.err <= 0.0:
            raise ValueError(
                f"photometry row {i} ({row.band}): mag/err must be finite with err > 0 "
                f"(got mag={row.mag}, err={row.err})"
            )
        if row.flag not in (FLAG_DETECTION, FLAG_UPPER_LIMIT):
            raise ValueError(
                f"photometry row {i} ({row.band}): flag must be "
                f"{FLAG_DETECTION} or {FLAG_UPPER_LIMIT}, got {row.flag}"
            )
        rows.append(row)
    return rows


def write_photometry_fits(
    path: str | os.PathLike[str],
    photometry: Iterable[PhotTuple | PhotRow | Mapping[str, object]],
    *,
    overwrite: bool = True,
) -> Path:
    """
    Write ``{band, mag, err, flag}`` binary table HDU to ``path``.

    Parameters
    ----------
    path :
        Output ``*_phot.fits`` path (parent dirs created as needed).
    photometry :
        Rows accepted by :func:`normalize_phot_rows`. Empty list writes an empty table.
    overwrite :
        Passed to ``writeto`` (default True).

    Returns
    -------
    Path
        Resolved output path.
    """
    rows = normalize_phot_rows(photometry)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = len(rows)
    bands = np.array([r.band.encode("ascii", errors="replace")[:32] for r in rows], dtype="S32")
    mags = np.array([r.mag for r in rows], dtype=np.float64)
    errs = np.array([r.err for r in rows], dtype=np.float64)
    flags = np.array([r.flag for r in rows], dtype=np.int16)

    hdu_primary = fits.PrimaryHDU()
    hdu_table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="band", format=_BAND_FMT, array=bands if n else np.array([], dtype="S32")),
            fits.Column(name="mag", format="D", array=mags if n else np.array([], dtype=np.float64)),
            fits.Column(name="err", format="D", array=errs if n else np.array([], dtype=np.float64)),
            fits.Column(name="flag", format=_FLAG_FMT, array=flags if n else np.array([], dtype=np.int16)),
        ]
    )
    fits.HDUList([hdu_primary, hdu_table]).writeto(out, overwrite=overwrite)
    return out.resolve()


def _flag_column_or_zeros(phottab: Table, n: int) -> np.ndarray:
    """
    Return int16 flag vector; missing column → all detections (0).

    Parameters
    ----------
    phottab :
        Astropy table from FITS HDU 1.
    n :
        Number of rows (must match table length).
    """
    if "flag" not in phottab.colnames:
        return np.zeros(n, dtype=np.int16)
    raw = np.asarray(phottab["flag"])
    # Vectorized: masked → 0 (detection); else int
    if np.ma.isMaskedArray(raw):
        data = np.ma.filled(raw, FLAG_DETECTION)
    else:
        data = raw
    return np.asarray(data, dtype=np.int16).reshape(-1)[:n]


def read_photometry_fits(path: str | os.PathLike[str]) -> list[PhotRow]:
    """
    Load Path-2 photometry rows from ``*_phot.fits``.

    Parameters
    ----------
    path :
        Existing photometry FITS path.

    Returns
    -------
    list[PhotRow]
        Rows with ``flag`` defaulting to ``0`` when the column is absent (legacy files).
        Invalid mag/err rows are skipped (NaN or ``err <= 0``), matching Path-1 loaders.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If required columns ``band``, ``mag``, ``err`` are missing.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Photometry file not found: {p}")
    phottab = Table.read(p, format="fits", hdu=1)
    for col in ("band", "mag", "err"):
        if col not in phottab.colnames:
            raise ValueError(f"{p}: missing required column {col!r}")

    n = len(phottab)
    flags = _flag_column_or_zeros(phottab, n)
    bands = phottab["band"]
    mags = np.asarray(phottab["mag"], dtype=np.float64)
    errs = np.asarray(phottab["err"], dtype=np.float64)

    out: list[PhotRow] = []
    for i in range(n):
        mag = float(mags[i])
        err = float(errs[i])
        if not np.isfinite(mag) or not np.isfinite(err) or err <= 0.0:
            continue
        flag_i = int(flags[i])
        if flag_i not in (FLAG_DETECTION, FLAG_UPPER_LIMIT):
            # Preserve unknown flags as detections for Path-1 safety only if 0;
            # non-zero non-1 still excluded from detection-only builders.
            pass
        out.append(
            PhotRow(
                band=_as_band_str(bands[i]),
                mag=mag,
                err=err,
                flag=flag_i,
            )
        )
    return out


def rows_to_detection_phot_dict(
    rows: Sequence[PhotRow],
    *,
    standardize_band_name: Callable[[str], str] | None = None,
) -> tuple[dict[str, list[float]], list[str]]:
    """
    Build Path-1 style ``{band: [mag, err]}`` from detection rows only (``flag==0``).

    Parameters
    ----------
    rows :
        Loaded :class:`PhotRow` values.
    standardize_band_name :
        Optional callable ``str -> str`` (defaults to
        :func:`darkhunter_sed.stellar_data.standardize_band_name`).

    Returns
    -------
    phot, phot_filtarr
        Detection-only dict and preferred-order band list (via stellar_data helpers).
    """
    from darkhunter_sed import stellar_data

    std = standardize_band_name or stellar_data.standardize_band_name
    phot: dict[str, list[float]] = {}
    for row in rows:
        if row.flag != FLAG_DETECTION:
            continue
        pb = std(row.band)
        if pb not in phot:
            phot[pb] = [row.mag, row.err]
    return phot, stellar_data.ordered_phot_filtarr_from_dict(phot)


def split_detections_and_upper_limits(
    rows: Sequence[PhotRow],
) -> tuple[list[PhotRow], list[PhotRow]]:
    """
    Partition rows into detections (``flag=0``) and 3σ upper limits (``flag=1``).

    Parameters
    ----------
    rows :
        Loaded photometry rows.

    Returns
    -------
    detections, upper_limits
        Two lists preserving input order within each class. Other flag values are
        omitted from both lists.
    """
    dets = [r for r in rows if r.flag == FLAG_DETECTION]
    uls = [r for r in rows if r.flag == FLAG_UPPER_LIMIT]
    return dets, uls
