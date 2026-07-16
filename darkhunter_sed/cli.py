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
        default=photometry_dir(),
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
    parser.add_argument(
        "--flexible-continuum",
        action="store_true",
        help="Free UMS per-epoch pc0–pc3 and LSF (UTP-style; not recommended for APF)",
    )
    parser.add_argument(
        "--spec-err-scale",
        type=float,
        default=1.0,
        help="Multiply spectral flux errors in likelihood (default 1)",
    )
    parser.add_argument(
        "--phot-err-scale",
        type=float,
        default=1.0,
        help="Multiply photometric errors in likelihood (>1 weakens phot constraint)",
    )
    flex = parser.add_argument_group("flexible continuum (requires --flexible-continuum)")
    flex.add_argument(
        "--pc0-min",
        type=float,
        default=0.95,
        help="Lower bound for pc0_i when --flexible-continuum (default 0.95)",
    )
    flex.add_argument(
        "--pc0-max",
        type=float,
        default=1.05,
        help="Upper bound for pc0_i when --flexible-continuum (default 1.05)",
    )
    flex.add_argument(
        "--pc-poly-abs-max",
        type=float,
        default=0.1,
        help="Absolute bound for pc1_i..pc3_i when --flexible-continuum (default 0.1)",
    )
    flex.add_argument(
        "--specjitter-max",
        type=float,
        default=1e-2,
        help="Upper bound for specjitter_i when --flexible-continuum (default 0.01)",
    )
    flex.add_argument(
        "--photjitter-max",
        type=float,
        default=1e-2,
        help="Upper bound for photometric jitter when --flexible-continuum (default 0.01)",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--phot-outlier-sigma",
        type=float,
        default=3.0,
        help="Drop photometric bands deviating from blackbody SED (0=off)",
    )
    parser.add_argument(
        "--phot-err-floor",
        type=float,
        default=0.02,
        help="Minimum mag uncertainty per phot band before SVI (default 0.02)",
    )
    parser.add_argument(
        "--gaia-phot-err-floor",
        type=float,
        default=None,
        help="Gaia DR3 mag error floor (default: same as --phot-err-floor)",
    )
    parser.add_argument(
        "--no-dust-av-prior",
        action="store_true",
        help="Use legacy Av=0 tnormal prior instead of dustmaps chain",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write diagnostic PDF(s) to output/plots after fit",
    )
    parser.add_argument(
        "--plot-epoch",
        type=int,
        default=0,
        help="Spectrum epoch index for --plot (0-based, default 0)",
    )
    parser.add_argument(
        "--plot-simple",
        action="store_true",
        help="Use the lightweight plotting.plot_fit_diagnostics fallback instead of the ppUMS-style PDF",
    )
    parser.add_argument(
        "--regions-json",
        type=Path,
        default=None,
        help="Regions/blaze JSON from pick_spectrum_regions.py (default: auto per-star in output/masks/)",
    )
    parser.add_argument(
        "--no-auto-regions",
        action="store_true",
        help="Do not auto-resolve per-star regions JSON; use calibrated blaze only",
    )
    parser.add_argument(
        "--no-auto-gather-phot",
        action="store_true",
        help="Do not query catalogs when photometry FITS is missing",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.verbose:
        for name in ("jax", "jax._src", "absl"):
            logging.getLogger(name).setLevel(logging.WARNING)

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
        phot_err_floor_mag=args.phot_err_floor,
        gaia_phot_err_floor_mag=args.gaia_phot_err_floor,
        from_fits=args.from_fits,
        regions_json=args.regions_json,
        auto_resolve_regions=not args.no_auto_regions,
        auto_gather_photometry=not args.no_auto_gather_phot,
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
        dust_av_prior=not args.no_dust_av_prior,
        rigid_continuum=not args.flexible_continuum,
        spec_err_scale=args.spec_err_scale,
        phot_err_scale=args.phot_err_scale,
        pc0_min=args.pc0_min,
        pc0_max=args.pc0_max,
        pc_poly_abs_max=args.pc_poly_abs_max,
        specjitter_max=args.specjitter_max,
        photjitter_max=args.photjitter_max,
    )
    extra: dict = {
        "n_epoch_spectra": len(spec_paths),
        "ums_continuum_policy": "fixed_pc_1_0_0_0" if not args.flexible_continuum else "flexible_pc",
    }
    if fit_data.get("regions_json"):
        extra["regions_json"] = fit_data["regions_json"]
    if fit_data.get("av_prior"):
        extra["av_prior"] = fit_data["av_prior"]
    if "ums" in results:
        try:
            from darkhunter_sed.continuum_diagnostics import diagnose_continuum_from_samples

            extra["continuum_diagnostics"] = diagnose_continuum_from_samples(
                fit_data,
                str(results["ums"]),
                epoch_index=args.plot_epoch,
                fixed_pc=not args.flexible_continuum,
            ).metrics
        except Exception as exc:
            logging.getLogger(__name__).warning("Continuum diagnostics skipped: %s", exc)
    json_path = posterior.write_sed_summary(
        args.gaia_id,
        ums_samples=results.get("ums"),
        utp_samples=results.get("utp"),
        extra=extra,
    )
    for kind, path in results.items():
        print(f"{kind.upper()} samples: {path}")
    print(f"SED summary: {json_path}")

    if "ums" in results:
        try:
            from darkhunter_sed.push_m1 import push_m1_for_gaia_id

            push_result = push_m1_for_gaia_id(args.gaia_id)
            if push_result.get("ok"):
                print(
                    f"push_m1: M1={push_result['m1_msun']:.5f} "
                    f"summary_updated={push_result['summary_updated']} "
                    f"csv_updated={push_result['csv_updated']}"
                )
            else:
                print(f"push_m1 skipped: {push_result.get('reason')}", file=sys.stderr)
        except Exception as exc:
            print(f"push_m1 failed: {exc}", file=sys.stderr)

    if args.plot:
        sed_summary = posterior.read_sed_summary(json_path)
        if args.plot_simple:
            from darkhunter_sed import plotting

            for kind in ("ums", "utp"):
                if kind in results:
                    pdf = plotting.plot_fit_diagnostics(
                        args.gaia_id,
                        fit_data,
                        results[kind],
                        fit_type=kind,
                        epoch_index=args.plot_epoch,
                    )
                    print(f"{kind.upper()} plot: {pdf}")
        else:
            from darkhunter_sed import plotting_ppums

            if "ums" in results:
                pdf = plotting_ppums.plot_ums_posterior(
                    args.gaia_id,
                    fit_data,
                    results["ums"],
                    fit_type="ums",
                    sed_summary=sed_summary,
                    epoch_index=args.plot_epoch,
                )
                print(f"UMS plot: {pdf}")
            if "utp" in results:
                pdf = plotting_ppums.plot_utp_posterior(
                    args.gaia_id,
                    fit_data,
                    results["utp"],
                    sed_summary=sed_summary,
                    epoch_index=args.plot_epoch,
                )
                print(f"UTP plot: {pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
