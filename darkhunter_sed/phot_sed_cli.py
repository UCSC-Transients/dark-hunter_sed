"""
CLI entry point ``darkhunter-sed-phot`` for Path-2 photometry SED fits.

Supported models: ``1star``, ``2star``, ``wd``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from darkhunter_sed.config import phot_sed_dir, photometry_dir
from darkhunter_sed.misty_iso import load_misty_predictor, resolve_mist_nn_path
from darkhunter_sed.phot_sed_fit import (
    OneStarPriorBounds,
    TwoStarPriorBounds,
    load_bandpasses_for_bands,
    run_1star_fit,
    run_2star_fit,
)
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
        choices=("1star", "2star", "wd"),
        help=(
            "SED model: 1star (single star), 2star (coeval binary), "
            "or wd (Bergeron DA/DB × Cummings IFMR)"
        ),
    )
    # WD-model-specific arguments.
    p.add_argument(
        "--wd-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing Bergeron Table_DA and Table_DB "
            "(default: $STELLAR_ROOT/wd or ~/stellar/wd)"
        ),
    )
    p.add_argument(
        "--system-age",
        type=float,
        default=0.0,
        metavar="GYR",
        help=(
            "System age in Gyr for Cummings IFMR progenitor age gate "
            "(default 0.0 = gate disabled)"
        ),
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
    p.add_argument(
        "--nlive",
        type=int,
        default=100,
        help="dynesty live points (default 100; increase for tighter evidence)",
    )
    p.add_argument(
        "--dlogz",
        type=float,
        default=1.0,
        help="dynesty stopping criterion dlogZ (default 1.0; tighten to 0.5 for precise lnZ)",
    )
    p.add_argument(
        "--maxiter",
        type=int,
        default=None,
        help="Optional dynesty maxiter (tests / quick runs)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for dynesty")
    p.add_argument(
        "--sample",
        default="auto",
        help="dynesty proposal method: auto, unif, rwalk, rslice, hslice (default auto)",
    )
    p.add_argument(
        "--bound",
        default="multi",
        help="dynesty bounding method: multi, single, balls, cubes, none (default multi)",
    )
    p.add_argument(
        "--nworkers",
        type=int,
        default=1,
        help=(
            "Parallel worker processes for likelihood evaluations (default 1 = serial). "
            "Workers > 1 spawn a multiprocessing.Pool; each re-initialises MISTy+PHOENIX "
            "so benchmark before enabling on short fits."
        ),
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Subsample the bandpass wavelength grid by N before PHOENIX integration "
            "(default 1 = full resolution). Values 4–10 give ~4–10× speedup with "
            "negligible accuracy loss for broad-band photometry — useful for a fast "
            "exploratory run; rerun at stride=1 for final results."
        ),
    )
    p.add_argument(
        "--print-progress",
        action="store_true",
        default=False,
        help="Print dynesty progress (lnZ, ncall, remaining work) to stdout ~every 1000 iter.",
    )
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


def _resolve_wd_dir(cli_path: Path | None) -> Path | None:
    """Return the Bergeron WD data directory, trying CLI arg, env, then ~/.stellar/wd."""
    import os

    if cli_path is not None:
        p = Path(cli_path).expanduser().resolve()
        return p if p.is_dir() else None
    stellar_root = os.environ.get("STELLAR_ROOT")
    if stellar_root:
        p = Path(stellar_root) / "wd"
        if p.is_dir():
            return p
    p = Path.home() / "stellar" / "wd"
    return p if p.is_dir() else None


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

    gaia_id = _normalize_gaia_id(args.gaia_id)
    phot_path = (
        Path(args.phot).expanduser().resolve()
        if args.phot is not None
        else default_phot_fits_path(gaia_id)
    )
    if not phot_path.is_file():
        print(f"Photometry FITS not found: {phot_path}", file=sys.stderr)
        return 1

    all_rows = read_photometry_fits(phot_path)

    if args.model == "wd":
        # WD model: pass all rows; wd_model.py filters to Bergeron-available bands.
        rows = all_rows
        mist_nn = predictor = grid_phx = bandpasses = common_kw = None  # unused
    else:
        # PHOENIX models: filter to synphot-registered bands only.
        from darkhunter_sed.filters_synphot import BAND_REGISTRY
        from darkhunter_sed.phot_sed_fit import _GAIA_CONSTRAINT_BANDS

        _skip = {"WISE_W3", "WISE_W4"}
        rows = [
            r for r in all_rows
            if r.band not in _skip
            and (r.band in BAND_REGISTRY or r.band in _GAIA_CONSTRAINT_BANDS)
        ]
        if not any(r.band in BAND_REGISTRY for r in rows):
            print(
                f"No Path-2-registered photometry bands in {phot_path}",
                file=sys.stderr,
            )
            return 1

        mist_nn = resolve_mist_nn_path(args.mist_nn)
        predictor = load_misty_predictor(mist_nn)
        grid_phx = PhoenixGrid(root=args.phoenix_dir)
        bandpasses = load_bandpasses_for_bands(
            [r.band for r in rows if r.band not in _GAIA_CONSTRAINT_BANDS]
        )
        out_dir = args.outdir if args.outdir is not None else phot_sed_dir()

        common_kw = dict(
            gaia_id=gaia_id,
            mist_predictor=predictor,
            phoenix_grid=grid_phx,
            out_dir=out_dir,
            nlive=int(args.nlive),
            dlogz=float(args.dlogz),
            maxiter=args.maxiter,
            seed=int(args.seed),
            sample=args.sample,
            bound=args.bound,
            nworkers=int(args.nworkers),
            bandpasses=bandpasses,
            spec_stride=int(args.stride),
            print_progress=bool(args.print_progress),
        )

    if args.model in ("1star", "2star"):
        if args.model == "1star":
            result, paths = run_1star_fit(rows, bounds=OneStarPriorBounds(), **common_kw)
        else:
            result, paths = run_2star_fit(rows, bounds=TwoStarPriorBounds(), **common_kw)
        print(
            f"{args.model} fit gaia_id={gaia_id}  lnZ={result.logz:.3f}±{result.logz_err:.3f}  "
            f"BIC={result.bic:.3f}  lnL_max={result.ln_l_max:.3f}"
        )
        print(f"summary: {paths['summary_json']}")
        print(f"samples: {paths['samples_npz']}")

    elif args.model == "wd":
        from darkhunter_sed.wd_model import WDPriorBounds, run_wd_fit

        wd_dir = _resolve_wd_dir(args.wd_dir)
        if wd_dir is None:
            print(
                "Cannot locate Bergeron WD tables. "
                "Set --wd-dir or STELLAR_ROOT env variable.",
                file=sys.stderr,
            )
            return 1

        out_dir = args.outdir if args.outdir is not None else phot_sed_dir()
        wd_out = Path(out_dir) / gaia_id / "wd"
        system_age_yr = float(args.system_age) * 1e9

        wd_results = run_wd_fit(
            rows,
            wd_dir=wd_dir,
            system_age_yr=system_age_yr,
            prior_bounds=WDPriorBounds(),
            nlive=int(args.nlive),
            maxiter=args.maxiter,
            seed=int(args.seed),
            outdir=wd_out,
        )
        for res in wd_results:
            s = res.summary()
            print(
                f"wd fit  atm={s['atm_type']}  ifmr={s['ifmr']}  "
                f"lnZ={s['logevidence']:.2f}  "
                f"M_WD={s['m_wd_median']:.3f}+{s['m_wd_hi']-s['m_wd_median']:.3f}"
                f"-{s['m_wd_median']-s['m_wd_lo']:.3f}  "
                f"extrap={s['extrap_mass']}"
            )
        print(f"wd outputs: {wd_out}")

    else:
        print(f"Unsupported model: {args.model}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
