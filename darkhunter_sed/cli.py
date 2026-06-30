#!/usr/bin/env python3
"""CLI for single-star SED fitting."""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

from darkhunter_rv.summary_paths import discover_primary_epoch_files

from darkhunter_sed import data, fit, models, posterior
from darkhunter_sed.config import photometry_dir, spec_root


def _expand_paths(pattern: str) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(pattern))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit stellar SED with uberMS (UMS primary + UTP secondary).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m darkhunter_sed.cli 1702370142434513152 \\
    --spec-glob 'data/Gaia_DR3_1702370142434513152_epoch_*.txt'

  python -m darkhunter_sed.cli 1702370142434513152 --from-spec-root

RV priors: per-epoch vrad_i normal priors come from dark-hunter_rv summary
[PIPELINE RESULTS] (mask CCF RV), with inflated errors (default 2x, floor 2 km/s).
Gaia priors default to summary [GAIA METADATA]; use --force-redownload to re-query TAP.
        """,
    )
    parser.add_argument("gaia_id", help="Gaia DR3 source_id")
    parser.add_argument(
        "--spec-glob",
        "-f",
        default=None,
        help="Glob for epoch spectra (.txt or .fits)",
    )
    parser.add_argument(
        "--from-spec-root",
        action="store_true",
        help="Discover epoch .txt files under SPEC_ROOT for this gaia_id",
    )
    parser.add_argument(
        "--photometry-dir",
        "-D",
        type=Path,
        default=None,
        help=f"Directory with {{gaia_id}}_phot.fits (default: {photometry_dir()})",
    )
    parser.add_argument("--ums-only", action="store_true", help="Skip UTP fit")
    parser.add_argument("--utp-only", action="store_true", help="Skip UMS fit")
    parser.add_argument("--from-fits", action="store_true", help="Input spectra are pre-converted FITS")
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Re-query Gaia TAP for stellar priors instead of RV summary cache",
    )
    parser.add_argument(
        "--vrad-err-inflate",
        type=float,
        default=2.0,
        help="Multiply formal RV errors for normal vrad_i priors (default 2)",
    )
    parser.add_argument(
        "--vrad-err-floor",
        type=float,
        default=2.0,
        help="Minimum sigma (km/s) for vrad_i normal priors (default 2)",
    )
    parser.add_argument(
        "--vrad-prior",
        choices=("normal", "uniform", "fixed"),
        default="normal",
        help="Prior type for per-epoch vrad_i (default normal from RV pipeline)",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--phot-outlier-sigma",
        type=float,
        default=3.0,
        help="Drop photometric bands deviating from blackbody SED (0=off)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    models.init_stellar_stack()
    _, phot_nn, _ = models.model_paths()

    spec_paths: list[Path] = []
    if args.from_spec_root:
        spec_paths = discover_primary_epoch_files(spec_root(), args.gaia_id)
    elif args.spec_glob:
        spec_paths = _expand_paths(args.spec_glob)
    else:
        parser.error("Provide --spec-glob or --from-spec-root")

    if not spec_paths:
        print("No spectrum files found.", file=sys.stderr)
        return 1

    phot_dir = args.photometry_dir or photometry_dir()
    fit_data = data.getdata(
        args.gaia_id,
        spectrum_paths=spec_paths,
        photometry_dir=phot_dir,
        phot_nn=phot_nn,
        force_redownload=args.force_redownload,
        phot_outlier_sigma=args.phot_outlier_sigma if args.phot_outlier_sigma > 0 else None,
        from_fits=args.from_fits,
    )

    do_ums = not args.utp_only
    do_utp = not args.ums_only
    if do_ums:
        try:
            fit.preflight_ums(len(fit_data.get("spec", [])), args.gaia_id)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            if not do_utp:
                return 1
            do_ums = False

    results = fit.run_both(
        args.gaia_id,
        fit_data,
        do_ums=do_ums,
        do_utp=do_utp,
        progressbar=not args.no_progress,
        vrad_prior=args.vrad_prior,
        vrad_err_inflate=args.vrad_err_inflate,
        vrad_err_floor_kms=args.vrad_err_floor,
    )
    json_path = posterior.write_sed_summary(
        args.gaia_id,
        ums_samples=results.get("ums"),
        utp_samples=results.get("utp"),
        extra={"n_epoch_spectra": len(spec_paths)},
    )
    for kind, path in results.items():
        print(f"{kind.upper()} samples: {path}")
    print(f"SED summary: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
