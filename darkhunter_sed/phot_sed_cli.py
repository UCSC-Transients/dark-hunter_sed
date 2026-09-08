"""
CLI entry point ``darkhunter-sed-phot`` for Path-2 photometry SED fits.

This issue supports ``--model 1star`` only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from darkhunter_sed.config import phot_sed_dir, photometry_dir
from darkhunter_sed.misty_iso import load_misty_predictor, resolve_mist_nn_path
from darkhunter_sed.phot_sed_fit import OneStarPriorBounds, run_1star_fit
from darkhunter_sed.phot_sed_io import read_photometry_fits
from darkhunter_sed.phoenix_grid import PhoenixGrid


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="darkhunter-sed-phot",
        description="Path-2 photometry SED fit (dynesty; PHOENIX×MISTy).",
    )
    p.add_argument("gaia_id", help="Gaia source id (digits) or Gaia_DR3_* stem")
    p.add_argument(
        "--model",
        default="1star",
        choices=("1star",),
        help="SED model (only 1star in this release)",
    )
    p.add_argument(
        "--phot",
        type=Path,
        default=None,
        help="Path to *_phot.fits (default: output/photometry/<id>_phot.fits, same as gather)",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: output/phot_sed/)",
    )
    p.add_argument("--nlive", type=int, default=200, help="dynesty live points")
    p.add_argument(
        "--maxiter",
        type=int,
        default=None,
        help="Optional dynesty maxiter (tests / quick runs)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for dynesty")
    p.add_argument(
        "--mist-nn",
        type=Path,
        default=None,
        help="Override mistNN HDF5 path (default: STELLAR_ROOT mistNN)",
    )
    p.add_argument(
        "--phoenix-dir",
        type=Path,
        default=None,
        help="Override PHOENIX_DIR HiResFITS root",
    )
    return p


def _normalize_gaia_id(raw: str) -> str:
    s = str(raw).strip()
    prefix = "Gaia_DR3_"
    if s.startswith(prefix):
        s = s[len(prefix) :]
    if s.endswith("_phot.fits"):
        s = s[: -len("_phot.fits")]
        if s.startswith(prefix):
            s = s[len(prefix) :]
    return s


def default_phot_fits_path(gaia_id: str, phot_dir: Path | None = None) -> Path:
    """
    Resolve the default ``*_phot.fits`` path for a Gaia source id.

    Parameters
    ----------
    gaia_id :
        Bare source id digits (after :func:`_normalize_gaia_id`).
    phot_dir :
        Photometry directory; default :func:`photometry_dir`.

    Returns
    -------
    Path
        ``{phot_dir}/{gaia_id}_phot.fits`` — same stem as
        :func:`darkhunter_sed.photometry_gather.save_photometry_to_fits`.

    Limits
    ------
    Does not check that the file exists. Does **not** use a ``Gaia_DR3_``
    filename prefix (gather never wrote that form).
    """
    root = phot_dir if phot_dir is not None else photometry_dir()
    return Path(root).expanduser().resolve() / f"{gaia_id}_phot.fits"


def main(argv: list[str] | None = None) -> int:
    """
    Run ``darkhunter-sed-phot``.

    Parameters
    ----------
    argv :
        CLI args excluding program name; ``None`` → ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code (0 success).

    Limits
    ------
    Only ``--model 1star``. Requires live MISTy mistNN + PHOENIX grid on disk
    for production runs (unit tests inject mocks and do not call this path).
    """
    args = _build_parser().parse_args(argv)
    if args.model != "1star":
        print(f"Unsupported model: {args.model}", file=sys.stderr)
        return 2

    gaia_id = _normalize_gaia_id(args.gaia_id)
    phot_path = (
        Path(args.phot).expanduser().resolve()
        if args.phot is not None
        else default_phot_fits_path(gaia_id)
    )
    if not phot_path.is_file():
        print(f"Photometry FITS not found: {phot_path}", file=sys.stderr)
        return 1

    rows = read_photometry_fits(phot_path)
    # Path-2 registry omits WISE_W3/W4 (beyond PHOENIX HiRes); drop if present
    # in legacy ``*_phot.fits``. W1/W2 kept but still truncated vs PHOENIX —
    # see filters_synphot / phoenix_grid synthesize_mags notes.
    from darkhunter_sed.filters_synphot import BAND_REGISTRY

    _skip = {"WISE_W3", "WISE_W4"}
    rows = [r for r in rows if r.band not in _skip and r.band in BAND_REGISTRY]
    if not rows:
        print(
            f"No Path-2-registered photometry bands in {phot_path}",
            file=sys.stderr,
        )
        return 1

    mist_nn = resolve_mist_nn_path(args.mist_nn)
    predictor = load_misty_predictor(mist_nn)
    grid = PhoenixGrid(root=args.phoenix_dir)

    out_dir = args.outdir if args.outdir is not None else phot_sed_dir()
    result, paths = run_1star_fit(
        rows,
        gaia_id=gaia_id,
        mist_predictor=predictor,
        phoenix_grid=grid,
        bounds=OneStarPriorBounds(),
        out_dir=out_dir,
        nlive=int(args.nlive),
        maxiter=args.maxiter,
        seed=int(args.seed),
    )
    print(
        f"1star fit gaia_id={gaia_id}  lnZ={result.logz:.3f}±{result.logz_err:.3f}  "
        f"BIC={result.bic:.3f}  lnL_max={result.ln_l_max:.3f}"
    )
    print(f"summary: {paths['summary_json']}")
    print(f"samples: {paths['samples_npz']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
