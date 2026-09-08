"""
synphot bandpass registry and PS1 / Swift UVOT thruput conversion (Path 2).

ASCII throughput curves under ``data/`` are converted to CDBS-style FITS
(``WAVELENGTH`` in Å, ``THROUGHPUT`` transmission) and installed into
``$PYSYN_CDBS/.../comp/nonhst/``. Band names match ``*_phot.fits`` stems
(``PS_*``, ``Swift_*``).

Limits
------
- Wavelengths in source ASCII are assumed to be Angstroms.
- Negative throughput samples are clamped to 0 (seen in Swift white).
- Swift UVOT **catalog gather** is out of scope; filters/registry only.
- Stock CDBS bands (Gaia/GALEX/SDSS/2MASS/WISE W1–W2) resolve only when
  ``PYSYN_CDBS`` points at a tree that already contains those thruputs.
- **WISE_W3 / WISE_W4 are not registered** for Path-2: PHOENIX HiRes ends
  near 5.5 μm while W3/W4 thruputs extend to tens of μm (see module comments
  near the WISE stock entries).
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from darkhunter_sed.config import REPO_ROOT

# Default local CDBS root used on this workstation (document; override via env).
DEFAULT_PYSYN_CDBS = Path("/Users/rfoley/pysynphot/trds")

# Repo-relative ASCII sources and generated FITS products.
DATA_DIR = REPO_ROOT / "data"
THRUPUT_DIR = DATA_DIR / "thruputs"


@dataclass(frozen=True, slots=True)
class BandSpec:
    """
    Registry entry for one ``*_phot.fits`` band name.

    Parameters
    ----------
    band :
        Canonical name written in photometry FITS (e.g. ``PS_g``, ``Swift_UVW1``).
    nonhst_filename :
        Basename under ``comp/nonhst/`` (and under ``data/thruputs/`` for
        repo-generated PS1/Swift files).
    ascii_source :
        Optional repo-relative ASCII curve (PS1/Swift only). ``None`` means
        stock CDBS thruput (must already exist under ``PYSYN_CDBS``).
    descrip :
        Primary-header DESCRIP string for generated FITS.
    """

    band: str
    nonhst_filename: str
    ascii_source: str | None
    descrip: str


# ASCII stem → band / FITS naming for conversion.
_PS1_ASCII: Mapping[str, tuple[str, str]] = {
    "g": ("PS_g", "ps1_g_001_syn.fits"),
    "r": ("PS_r", "ps1_r_001_syn.fits"),
    "i": ("PS_i", "ps1_i_001_syn.fits"),
    "z": ("PS_z", "ps1_z_001_syn.fits"),
    "y": ("PS_y", "ps1_y_001_syn.fits"),
    "w": ("PS_w", "ps1_w_001_syn.fits"),
}

_SWIFT_ASCII: Mapping[str, tuple[str, str]] = {
    "U": ("Swift_U", "swift_u_001_syn.fits"),
    "B": ("Swift_B", "swift_b_001_syn.fits"),
    "V": ("Swift_V", "swift_v_001_syn.fits"),
    "UVW1": ("Swift_UVW1", "swift_uvw1_001_syn.fits"),
    "UVW2": ("Swift_UVW2", "swift_uvw2_001_syn.fits"),
    "UVM2": ("Swift_UVM2", "swift_uvm2_001_syn.fits"),
    "white": ("Swift_white", "swift_white_001_syn.fits"),
}


def _build_band_registry() -> dict[str, BandSpec]:
    """Assemble Path-2 band registry (stock CDBS + PS1/Swift conversions)."""
    registry: dict[str, BandSpec] = {}

    stock: list[tuple[str, str, str]] = [
        ("GaiaDR3_G", "gaia_g_001_syn.fits", "Gaia G (stock CDBS)"),
        ("GaiaDR3_BP", "gaia_gbp_001_syn.fits", "Gaia BP (stock CDBS)"),
        ("GaiaDR3_RP", "gaia_grp_001_syn.fits", "Gaia RP (stock CDBS)"),
        ("GALEX_FUV", "galex_fuv_001_syn.fits", "GALEX FUV (stock CDBS)"),
        ("GALEX_NUV", "galex_nuv_001_syn.fits", "GALEX NUV (stock CDBS)"),
        ("SDSS_u", "sdss_u_005_syn.fits", "SDSS u (stock CDBS)"),
        ("SDSS_g", "sdss_g_005_syn.fits", "SDSS g (stock CDBS)"),
        ("SDSS_r", "sdss_r_005_syn.fits", "SDSS r (stock CDBS)"),
        ("SDSS_i", "sdss_i_005_syn.fits", "SDSS i (stock CDBS)"),
        ("SDSS_z", "sdss_z_005_syn.fits", "SDSS z (stock CDBS)"),
        ("2MASS_J", "2mass_j_001_syn.fits", "2MASS J (stock CDBS)"),
        ("2MASS_H", "2mass_h_001_syn.fits", "2MASS H (stock CDBS)"),
        ("2MASS_Ks", "2mass_ks_001_syn.fits", "2MASS Ks (stock CDBS)"),
        # WISE vs PHOENIX HiRes (~500–55_000 Å / ≤~5.5 μm):
        # - W1/W2 CDBS thruputs extend redward of the PHOENIX WAVE file
        #   (~65_000 / ~80_000 Å). Kept for Path-2 photometry, but mid-IR
        #   integrals are incomplete (synphot taper). Later: examine missing
        #   long-λ flux and possibly build a PHOENIX/SED red extension.
        # - W3/W4 thruputs go to ~285_000 Å — drop from Path-2 registry until
        #   such an extension exists (do not synthesize these bands).
        ("WISE_W1", "wise_w1_001_syn.fits", "WISE W1 (stock CDBS; partial vs PHOENIX)"),
        ("WISE_W2", "wise_w2_001_syn.fits", "WISE W2 (stock CDBS; partial vs PHOENIX)"),
        # ("WISE_W3", "wise_w3_001_syn.fits", ...),  # excluded: far beyond PHOENIX
        # ("WISE_W4", "wise_w4_001_syn.fits", ...),  # excluded: far beyond PHOENIX
    ]
    for band, fname, descrip in stock:
        registry[band] = BandSpec(band, fname, None, descrip)

    for filt, (band, fname) in _PS1_ASCII.items():
        registry[band] = BandSpec(
            band,
            fname,
            f"PAN-STARRS_PS1.{filt}.dat",
            f"Pan-STARRS PS1 {filt} thruput (repo ASCII)",
        )
    for filt, (band, fname) in _SWIFT_ASCII.items():
        registry[band] = BandSpec(
            band,
            fname,
            f"Swift_UVOT.{filt}_trn.dat",
            f"Swift UVOT {filt} thruput (repo ASCII)",
        )
    return registry


BAND_REGISTRY: dict[str, BandSpec] = _build_band_registry()

# Bands produced from repo ASCII (always testable without stock CDBS).
CONVERTED_BANDS: tuple[str, ...] = tuple(
    sorted(
        b.band
        for b in BAND_REGISTRY.values()
        if b.ascii_source is not None
    )
)


def pysyn_cdbs_root(explicit: Path | str | None = None) -> Path:
    """
    Resolve ``PYSYN_CDBS`` root directory.

    Parameters
    ----------
    explicit :
        Optional override path. When ``None``, reads ``PYSYN_CDBS`` from the
        environment, else falls back to :data:`DEFAULT_PYSYN_CDBS`.

    Returns
    -------
    Path
        Absolute CDBS root (typically ``.../pysynphot/trds``).

    Limits
    ------
    Does not verify that ``comp/nonhst`` exists; callers that need the tree
    should call :func:`nonhst_dir`.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("PYSYN_CDBS")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_PYSYN_CDBS.expanduser().resolve()


def nonhst_dir(cdbs_root: Path | str | None = None) -> Path:
    """
    Return ``comp/nonhst`` under a CDBS tree.

    Parameters
    ----------
    cdbs_root :
        CDBS root or ``None`` to use :func:`pysyn_cdbs_root`.

    Returns
    -------
    Path
        ``{cdbs}/grp/redcat/trds/comp/nonhst`` when that layout exists,
        else ``{cdbs}/comp/nonhst`` when present, else the grp/redcat path
        (created by install).

    Limits
    ------
    Supports both ``PYSYN_CDBS=.../trds`` (common) and a root that already
    contains ``comp/nonhst`` directly.
    """
    root = pysyn_cdbs_root(cdbs_root)
    nested = root / "grp" / "redcat" / "trds" / "comp" / "nonhst"
    flat = root / "comp" / "nonhst"
    if nested.is_dir():
        return nested
    if flat.is_dir():
        return flat
    return nested


def read_ascii_thruput(path: Path | str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Load a two-column ASCII thruput curve.

    Parameters
    ----------
    path :
        File with wavelength (Å) and throughput columns; ``#`` comments OK.

    Returns
    -------
    wave_aa, thruput :
        1-D float64 arrays, sorted by wavelength. Throughput is clamped to
        ``>= 0``.

    Limits
    ------
    Requires ≥2 finite rows; wavelengths must be strictly increasing after sort.
    Does not convert nm→Å (sources already in Å).
    """
    path = Path(path)
    waves: list[float] = []
    thrus: list[float] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith(";"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            w = float(parts[0])
            t = float(parts[1])
        except ValueError:
            continue
        if not np.isfinite(w) or not np.isfinite(t):
            continue
        waves.append(w)
        thrus.append(max(t, 0.0))
    if len(waves) < 2:
        raise ValueError(f"Need ≥2 thruput samples in {path}")
    wave = np.asarray(waves, dtype=np.float64)
    thru = np.asarray(thrus, dtype=np.float64)
    order = np.argsort(wave, kind="mergesort")
    wave = wave[order]
    thru = thru[order]
    if np.any(np.diff(wave) <= 0):
        # Collapse exact duplicates by keeping max thruput at that λ.
        uniq_w: list[float] = []
        uniq_t: list[float] = []
        for w, t in zip(wave, thru, strict=True):
            if uniq_w and w == uniq_w[-1]:
                uniq_t[-1] = max(uniq_t[-1], float(t))
            else:
                uniq_w.append(float(w))
                uniq_t.append(float(t))
        wave = np.asarray(uniq_w, dtype=np.float64)
        thru = np.asarray(uniq_t, dtype=np.float64)
    if wave.size < 2 or np.any(np.diff(wave) <= 0):
        raise ValueError(f"Wavelengths not strictly increasing in {path}")
    return wave, thru


def write_thruput_fits(
    path: Path | str,
    wave_aa: NDArray[np.floating],
    thruput: NDArray[np.floating],
    *,
    compname: str,
    descrip: str,
    overwrite: bool = True,
) -> Path:
    """
    Write a synphot/CDBS-compatible thruput FITS file.

    Parameters
    ----------
    path :
        Output ``.fits`` path.
    wave_aa :
        Wavelengths in Angstroms.
    thruput :
        Dimensionless transmission (≥0).
    compname :
        ``COMPNAME`` header value (basename without ``_syn.fits``).
    descrip :
        ``DESCRIP`` primary-header string.
    overwrite :
        Replace existing file when True.

    Returns
    -------
    Path
        Written file path.

    Limits
    ------
    Column layout matches stock nonhst thruputs: ``WAVELENGTH`` (ANGSTROMS),
    ``THROUGHPUT`` (TRANSMISSION). Does not register graph/component tables.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wave = np.asarray(wave_aa, dtype=np.float64)
    thru = np.asarray(thruput, dtype=np.float64)
    if wave.shape != thru.shape or wave.ndim != 1:
        raise ValueError("wave_aa and thruput must be 1-D and same length")
    if wave.size < 2:
        raise ValueError("Need ≥2 thruput samples")
    thru = np.maximum(thru, 0.0)

    pri = fits.PrimaryHDU()
    pri.header["DBTABLE"] = "CRTHROUGHPUT"
    pri.header["INSTRUME"] = "NONHST"
    pri.header["PEDIGREE"] = "MODEL"
    pri.header["DESCRIP"] = descrip[:68]
    pri.header["COMPNAME"] = compname[:18]
    pri.header["COMMENT"] = "Converted by darkhunter_sed.filters_synphot"

    col_w = fits.Column(
        name="WAVELENGTH",
        format="1D",
        unit="ANGSTROMS",
        array=wave,
        disp="F10.6",
    )
    col_t = fits.Column(
        name="THROUGHPUT",
        format="1D",
        unit="TRANSMISSION",
        array=thru,
        disp="G12.5",
    )
    tab = fits.BinTableHDU.from_columns([col_w, col_t])
    fits.HDUList([pri, tab]).writeto(path, overwrite=overwrite)
    return path.resolve()


def convert_ascii_file_to_fits(
    ascii_path: Path | str,
    fits_path: Path | str,
    *,
    band: str,
    descrip: str | None = None,
    overwrite: bool = True,
) -> Path:
    """
    Convert one ASCII thruput curve to synphot FITS.

    Parameters
    ----------
    ascii_path :
        Two-column source (λ Å, throughput).
    fits_path :
        Destination FITS path.
    band :
        Registry / ``*_phot.fits`` band name (for COMPNAME hint).
    descrip :
        Optional DESCRIP; default mentions ``band``.
    overwrite :
        Replace existing FITS when True.

    Returns
    -------
    Path
        Written FITS path.
    """
    wave, thru = read_ascii_thruput(ascii_path)
    fits_path = Path(fits_path)
    comp = fits_path.name.replace("_syn.fits", "").replace(".fits", "")
    return write_thruput_fits(
        fits_path,
        wave,
        thru,
        compname=comp,
        descrip=descrip or f"Thruput for band {band}",
        overwrite=overwrite,
    )


def convert_repo_ascii_thruputs(
    *,
    out_dir: Path | str | None = None,
    overwrite: bool = True,
) -> dict[str, Path]:
    """
    Convert all repo PS1/Swift ASCII curves into FITS under ``data/thruputs/``.

    Parameters
    ----------
    out_dir :
        Output directory; default :data:`THRUPUT_DIR`.
    overwrite :
        Replace existing FITS when True.

    Returns
    -------
    dict
        Mapping band name → written FITS path.

    Limits
    ------
    Only processes :data:`BAND_REGISTRY` entries with ``ascii_source`` set.
    """
    out = Path(out_dir) if out_dir is not None else THRUPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for band in CONVERTED_BANDS:
        spec = BAND_REGISTRY[band]
        assert spec.ascii_source is not None
        ascii_path = DATA_DIR / spec.ascii_source
        if not ascii_path.is_file():
            raise FileNotFoundError(f"Missing ASCII thruput: {ascii_path}")
        dest = out / spec.nonhst_filename
        written[band] = convert_ascii_file_to_fits(
            ascii_path,
            dest,
            band=band,
            descrip=spec.descrip,
            overwrite=overwrite,
        )
    return written


def install_thruputs_to_cdbs(
    *,
    cdbs_root: Path | str | None = None,
    source_dir: Path | str | None = None,
    bands: Iterable[str] | None = None,
    convert_if_missing: bool = True,
) -> dict[str, Path]:
    """
    Copy PS1/Swift thruput FITS into ``$PYSYN_CDBS`` ``comp/nonhst/``.

    Parameters
    ----------
    cdbs_root :
        CDBS root override; default :func:`pysyn_cdbs_root`.
    source_dir :
        Directory of generated FITS; default :data:`THRUPUT_DIR`.
    bands :
        Subset of :data:`CONVERTED_BANDS`; default all converted bands.
    convert_if_missing :
        Run :func:`convert_repo_ascii_thruputs` when a source FITS is absent.

    Returns
    -------
    dict
        Mapping band → installed path under nonhst.

    Limits
    ------
    Idempotent overwrite of matching basenames only. Does not modify stock
    CDBS thruputs or graph tables. Swift gather remains unimplemented.
    """
    src_root = Path(source_dir) if source_dir is not None else THRUPUT_DIR
    dest_dir = nonhst_dir(cdbs_root)
    dest_dir.mkdir(parents=True, exist_ok=True)
    wanted = list(bands) if bands is not None else list(CONVERTED_BANDS)
    installed: dict[str, Path] = {}
    for band in wanted:
        if band not in BAND_REGISTRY:
            raise KeyError(f"Unknown band {band!r}")
        spec = BAND_REGISTRY[band]
        if spec.ascii_source is None:
            raise ValueError(f"{band} is stock CDBS; nothing to install")
        src = src_root / spec.nonhst_filename
        if not src.is_file():
            if not convert_if_missing:
                raise FileNotFoundError(src)
            convert_repo_ascii_thruputs(out_dir=src_root, overwrite=True)
        if not src.is_file():
            raise FileNotFoundError(src)
        dest = dest_dir / spec.nonhst_filename
        shutil.copy2(src, dest)
        installed[band] = dest.resolve()
    return installed


def thruput_path_for_band(
    band: str,
    *,
    cdbs_root: Path | str | None = None,
    allow_repo_fallback: bool = True,
) -> Path:
    """
    Resolve the thruput FITS path for a registry band.

    Parameters
    ----------
    band :
        Canonical ``*_phot.fits`` name.
    cdbs_root :
        Optional CDBS root.
    allow_repo_fallback :
        For converted bands, fall back to ``data/thruputs/`` (converting from
        ASCII if needed) when the file is absent from CDBS.

    Returns
    -------
    Path
        Existing FITS path.

    Raises
    ------
    KeyError
        Unknown band.
    FileNotFoundError
        Thruput missing and fallback disabled / impossible.
    """
    if band not in BAND_REGISTRY:
        raise KeyError(
            f"Unknown band {band!r}; known={sorted(BAND_REGISTRY)}"
        )
    spec = BAND_REGISTRY[band]
    cdbs_file = nonhst_dir(cdbs_root) / spec.nonhst_filename
    if cdbs_file.is_file():
        return cdbs_file.resolve()
    if spec.ascii_source is not None and allow_repo_fallback:
        repo_file = THRUPUT_DIR / spec.nonhst_filename
        if not repo_file.is_file():
            convert_repo_ascii_thruputs(overwrite=True)
        if repo_file.is_file():
            return repo_file.resolve()
        ascii_path = DATA_DIR / spec.ascii_source
        if ascii_path.is_file():
            return convert_ascii_file_to_fits(
                ascii_path,
                repo_file,
                band=band,
                descrip=spec.descrip,
            )
    raise FileNotFoundError(
        f"Thruput for {band} not found at {cdbs_file}"
        + (" (repo fallback failed)" if allow_repo_fallback else "")
    )


def load_bandpass(band: str, *, cdbs_root: Path | str | None = None):
    """
    Load a synphot ``SpectralElement`` for a registry band.

    Parameters
    ----------
    band :
        Canonical ``*_phot.fits`` name (``PS_*``, ``Swift_*``, or stock).
    cdbs_root :
        Optional CDBS root for path resolution.

    Returns
    -------
    synphot.spectrum.SpectralElement
        Empirical bandpass.

    Limits
    ------
    Requires the ``synphot`` package. Stock bands need ``PYSYN_CDBS`` (or
    ``cdbs_root``) populated; converted PS1/Swift bands can load from repo
    ``data/thruputs/``.
    """
    from synphot import SpectralElement

    path = thruput_path_for_band(band, cdbs_root=cdbs_root)
    return SpectralElement.from_file(str(path))


def registered_bands(*, converted_only: bool = False) -> tuple[str, ...]:
    """
    List registry band names.

    Parameters
    ----------
    converted_only :
        If True, return only PS1/Swift bands built from repo ASCII.

    Returns
    -------
    tuple of str
        Sorted band names.
    """
    if converted_only:
        return CONVERTED_BANDS
    return tuple(sorted(BAND_REGISTRY))


def _cli(argv: list[str] | None = None) -> int:
    """
    CLI: convert ASCII thruputs and/or install into ``PYSYN_CDBS``.

    Parameters
    ----------
    argv :
        Argument list without program name; ``None`` → ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code (0 success).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert data/PAN-STARRS_PS1.*.dat and data/Swift_UVOT.*_trn.dat "
            "to synphot FITS and install into $PYSYN_CDBS comp/nonhst/"
        )
    )
    parser.add_argument(
        "--cdbs",
        type=Path,
        default=None,
        help="PYSYN_CDBS root (default: $PYSYN_CDBS or workstation default)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"FITS output dir (default: {THRUPUT_DIR})",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Write data/thruputs FITS only; do not copy into CDBS",
    )
    parser.add_argument(
        "--install-only",
        action="store_true",
        help="Copy existing thruputs into CDBS (convert if missing)",
    )
    args = parser.parse_args(argv)

    if args.convert_only and args.install_only:
        parser.error("Use only one of --convert-only / --install-only")

    if args.install_only:
        installed = install_thruputs_to_cdbs(cdbs_root=args.cdbs)
        for band, path in sorted(installed.items()):
            print(f"installed {band} -> {path}")
        return 0

    written = convert_repo_ascii_thruputs(out_dir=args.out_dir, overwrite=True)
    for band, path in sorted(written.items()):
        print(f"wrote {band} -> {path}")
    if not args.convert_only:
        installed = install_thruputs_to_cdbs(
            cdbs_root=args.cdbs,
            source_dir=args.out_dir or THRUPUT_DIR,
            convert_if_missing=False,
        )
        for band, path in sorted(installed.items()):
            print(f"installed {band} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
