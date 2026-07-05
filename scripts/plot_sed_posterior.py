#!/usr/bin/env python3
"""Regenerate ppUMS-style posterior PDF(s) from existing sample FITS.

Rebuilds ``fit_data`` with the same spectrum prep as the fit (so this must be
re-run after any ingest change, e.g. order-end trimming), then draws the
two-page UMS/UTP posterior plot.

Example:
    bash scripts/run_local.sh scripts/plot_sed_posterior.py 77413727493690112 \\
        --from-spec-root \\
        --ums-samples output/samples/Gaia_DR3_77413727493690112_ums.fits \\
        --utp-samples output/samples/Gaia_DR3_77413727493690112_utp.fits \\
        --sed-summary output/sed_summaries/Gaia_DR3_77413727493690112_sed_summary.json \\
        --epoch 0
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

from darkhunter_rv.summary_paths import discover_primary_epoch_files

from darkhunter_sed import data, models, plotting_ppums, posterior
from darkhunter_sed.config import photometry_dir, spec_root


def _expand_paths(pattern: str) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(pattern))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gaia_id", help="Gaia DR3 source_id")
    parser.add_argument("--spec-glob", "-f", default=None, help="Glob for epoch spectra")
    parser.add_argument("--from-spec-root", action="store_true", help="Discover epoch files under SPEC_ROOT")
    parser.add_argument("--from-fits", action="store_true", help="Input spectra are pre-converted FITS")
    parser.add_argument(
        "--photometry-dir",
        "-D",
        type=Path,
        default=photometry_dir(),
        help=f"Directory with {{gaia_id}}_phot.fits (default: {photometry_dir()})",
    )
    parser.add_argument("--ums-samples", type=Path, default=None, help="UMS sample FITS")
    parser.add_argument("--utp-samples", type=Path, default=None, help="UTP sample FITS")
    parser.add_argument("--sed-summary", type=Path, default=None, help="SED summary JSON for header extras")
    parser.add_argument("--epoch", type=int, default=0, help="Spectrum epoch index (0-based)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.ums_samples is None and args.utp_samples is None:
        parser.error("Provide at least one of --ums-samples / --utp-samples")

    if args.from_spec_root:
        spec_paths = discover_primary_epoch_files(spec_root(), args.gaia_id)
    elif args.spec_glob:
        spec_paths = _expand_paths(args.spec_glob)
    else:
        parser.error("Provide --spec-glob or --from-spec-root")
    if not spec_paths:
        parser.error("No spectrum files found")

    _, phot_nn, _ = models.model_paths()
    fit_data = data.getdata(
        args.gaia_id,
        spectrum_paths=spec_paths,
        photometry_dir=args.photometry_dir,
        phot_nn=phot_nn,
        from_fits=args.from_fits,
    )

    sed_summary = posterior.read_sed_summary(args.sed_summary) if args.sed_summary else None

    if args.ums_samples is not None:
        pdf = plotting_ppums.plot_ums_posterior(
            args.gaia_id,
            fit_data,
            args.ums_samples,
            fit_type="ums",
            sed_summary=sed_summary,
            epoch_index=args.epoch,
        )
        print(f"UMS plot: {pdf}")
    if args.utp_samples is not None:
        pdf = plotting_ppums.plot_utp_posterior(
            args.gaia_id,
            fit_data,
            args.utp_samples,
            sed_summary=sed_summary,
            epoch_index=args.epoch,
        )
        print(f"UTP plot: {pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
