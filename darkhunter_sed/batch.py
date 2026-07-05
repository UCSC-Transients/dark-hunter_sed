"""Batch SED fitting with incremental --update skip logic."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from darkhunter_rv.summary_paths import (
    discover_primary_epoch_files,
    discover_spec_gaia_ids,
    discover_summary_path,
)

from darkhunter_sed import data, fit, models, posterior
from darkhunter_sed.config import (
    photometry_dir,
    rv_output_dir,
    samples_dir,
    sed_summaries_dir,
    spec_root,
)
from darkhunter_sed.region_picker import resolve_regions_json_for_star
from darkhunter_sed.stellar_data import photometry_fits_path

logger = logging.getLogger(__name__)


@dataclass
class BatchStats:
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


def _mtime(path: Path | None) -> float:
    if path is None or not path.is_file():
        return 0.0
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def ums_samples_path(gaia_id: str) -> Path:
    return samples_dir() / f"Gaia_DR3_{gaia_id}_ums.fits"


def utp_samples_path(gaia_id: str) -> Path:
    return samples_dir() / f"Gaia_DR3_{gaia_id}_utp.fits"


def input_paths_for_star(
    gaia_id: str,
    *,
    spec_root_path: Path,
    phot_dir: Path,
    rv_out: Path,
) -> tuple[list[Path], Path | None, Path | None]:
    spec_paths = discover_primary_epoch_files(spec_root_path, gaia_id)
    summary = discover_summary_path(rv_out, gaia_id)
    phot = Path(photometry_fits_path(gaia_id, phot_dir))
    return spec_paths, summary, phot if phot.is_file() else None


def newest_input_mtime(
    spec_paths: list[Path],
    summary: Path | None,
    phot: Path | None,
    regions_json: Path | None = None,
) -> float:
    mtimes = [_mtime(p) for p in spec_paths]
    mtimes.append(_mtime(summary))
    mtimes.append(_mtime(phot))
    mtimes.append(_mtime(regions_json))
    return max(mtimes) if mtimes else 0.0


def newest_output_mtime(gaia_id: str) -> float:
    paths = [
        posterior.sed_summary_path(gaia_id),
        ums_samples_path(gaia_id),
        utp_samples_path(gaia_id),
    ]
    return max(_mtime(p) for p in paths)


def needs_update(
    gaia_id: str,
    *,
    spec_root_path: Path,
    phot_dir: Path,
    rv_out: Path,
    force: bool = False,
    regions_json: Path | None = None,
    auto_resolve_regions: bool = True,
    auto_gather_photometry: bool = True,
) -> tuple[bool, str]:
    if force:
        return True, "force"

    spec_paths, summary, phot = input_paths_for_star(
        gaia_id, spec_root_path=spec_root_path, phot_dir=phot_dir, rv_out=rv_out
    )
    if len(spec_paths) < 2:
        return False, f"need >=2 epoch spectra (have {len(spec_paths)})"
    if phot is None:
        if auto_gather_photometry:
            return True, "missing photometry (will gather)"
        return False, "missing photometry FITS"

    resolved_regions = resolve_regions_json_for_star(
        gaia_id,
        regions_json,
        auto_resolve=auto_resolve_regions,
    )
    in_m = newest_input_mtime(spec_paths, summary, phot, resolved_regions)
    out_m = newest_output_mtime(gaia_id)
    json_path = posterior.sed_summary_path(gaia_id)

    if out_m == 0.0 or not json_path.is_file():
        return True, "no prior sed_summary"

    if in_m > out_m:
        return True, "inputs newer than outputs"
    return False, "up to date"


def fit_one_star(
    gaia_id: str,
    *,
    spec_root_path: Path | None = None,
    phot_dir: Path | None = None,
    rv_out: Path | None = None,
    do_ums: bool = True,
    do_utp: bool = True,
    force_redownload: bool = False,
    progressbar: bool = False,
    vrad_prior: str = "normal",
    vrad_err_inflate: float = 2.0,
    vrad_err_floor_kms: float = 2.0,
    phot_outlier_sigma: float | None = 3.0,
    regions_json: Path | None = None,
    auto_resolve_regions: bool = True,
    auto_gather_photometry: bool = True,
) -> dict[str, Path]:
    """Run SED fit + write sed_summary.json for one Gaia source."""
    spec_root_path = spec_root_path or spec_root()
    phot_dir = phot_dir or photometry_dir()
    rv_out = rv_out or rv_output_dir()

    models.init_stellar_stack()
    _, phot_nn, _ = models.model_paths()

    spec_paths, summary, _phot = input_paths_for_star(
        gaia_id, spec_root_path=spec_root_path, phot_dir=phot_dir, rv_out=rv_out
    )
    if not spec_paths:
        raise FileNotFoundError(f"No spectra under {spec_root_path} for {gaia_id}")

    fit_data = data.getdata(
        gaia_id,
        spectrum_paths=spec_paths,
        photometry_dir=phot_dir,
        phot_nn=phot_nn,
        summary_path=summary,
        force_redownload=force_redownload,
        phot_outlier_sigma=phot_outlier_sigma,
        regions_json=regions_json,
        auto_resolve_regions=auto_resolve_regions,
        auto_gather_photometry=auto_gather_photometry,
    )

    run_ums = do_ums
    if do_ums:
        try:
            fit.preflight_ums(len(fit_data.get("spec", [])), gaia_id)
        except ValueError:
            if not do_utp:
                raise
            run_ums = False

    sample_paths = fit.run_both(
        gaia_id,
        fit_data,
        do_ums=run_ums,
        do_utp=do_utp,
        progressbar=progressbar,
        vrad_prior=vrad_prior,
        vrad_err_inflate=vrad_err_inflate,
        vrad_err_floor_kms=vrad_err_floor_kms,
    )

    phot_path = Path(photometry_fits_path(gaia_id, phot_dir))
    extra: dict = {
        "n_epoch_spectra": len(spec_paths),
        "rv_summary": str(summary.resolve()) if summary else None,
        "photometry_fits": str(phot_path.resolve()) if phot_path.is_file() else None,
    }
    if fit_data.get("regions_json"):
        extra["regions_json"] = fit_data["regions_json"]

    json_path = posterior.write_sed_summary(
        gaia_id,
        ums_samples=sample_paths.get("ums"),
        utp_samples=sample_paths.get("utp"),
        extra=extra,
    )
    sample_paths["sed_summary"] = json_path
    return sample_paths


def run_batch(
    gaia_ids: list[str] | None = None,
    *,
    spec_root_path: Path | None = None,
    update: bool = False,
    force: bool = False,
    limit: int | None = None,
    **fit_kwargs,
) -> BatchStats:
    spec_root_path = spec_root_path or spec_root()
    phot_dir = photometry_dir()
    rv_out = rv_output_dir()
    sed_summaries_dir().mkdir(parents=True, exist_ok=True)
    samples_dir().mkdir(parents=True, exist_ok=True)

    if gaia_ids is None:
        gaia_ids = sorted(discover_spec_gaia_ids(spec_root_path))

    stats = BatchStats()
    for i, gid in enumerate(gaia_ids):
        if limit is not None and stats.processed >= limit:
            break

        if update and not force:
            ok, reason = needs_update(
                gid,
                spec_root_path=spec_root_path,
                phot_dir=phot_dir,
                rv_out=rv_out,
                force=False,
                regions_json=fit_kwargs.get("regions_json"),
                auto_resolve_regions=fit_kwargs.get("auto_resolve_regions", True),
                auto_gather_photometry=fit_kwargs.get("auto_gather_photometry", True),
            )
            if not ok:
                logger.info("skip %s (%s)", gid, reason)
                stats.skipped += 1
                continue

        logger.info("fitting %s (%d/%d)", gid, i + 1, len(gaia_ids))
        try:
            paths = fit_one_star(
                gid,
                spec_root_path=spec_root_path,
                phot_dir=phot_dir,
                rv_out=rv_out,
                **fit_kwargs,
            )
            stats.processed += 1
            logger.info(
                "done %s ums=%s utp=%s json=%s",
                gid,
                paths.get("ums"),
                paths.get("utp"),
                paths.get("sed_summary"),
            )
        except Exception as exc:
            stats.failed += 1
            msg = f"{gid}: {exc}"
            stats.errors.append(msg)
            logger.error("failed %s", msg, exc_info=True)

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch SED fits for Gaia stars with spectra under SPEC_ROOT."
    )
    parser.add_argument(
        "--gaia-id",
        action="append",
        dest="gaia_ids",
        help="Process only these source_ids (repeatable)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Skip stars whose inputs are not newer than sed_summary + samples",
    )
    parser.add_argument("--force", action="store_true", help="Refit even when up to date")
    parser.add_argument("--ums-only", action="store_true")
    parser.add_argument("--utp-only", action="store_true")
    parser.add_argument("--force-redownload", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--spec-root",
        type=Path,
        default=None,
        help=f"Spectrum tree (default {spec_root()})",
    )
    parser.add_argument(
        "--regions-json",
        type=Path,
        default=None,
        help="Shared regions/blaze JSON for all stars in this batch (overrides per-star auto)",
    )
    parser.add_argument(
        "--no-auto-regions",
        action="store_true",
        help="Do not auto-resolve per-star regions JSON from output/masks/",
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

    stats = run_batch(
        args.gaia_ids,
        spec_root_path=args.spec_root,
        update=args.update or args.force,
        force=args.force,
        limit=args.limit,
        do_ums=not args.utp_only,
        do_utp=not args.ums_only,
        force_redownload=args.force_redownload,
        progressbar=not args.no_progress,
        regions_json=args.regions_json,
        auto_resolve_regions=not args.no_auto_regions,
        auto_gather_photometry=not args.no_auto_gather_phot,
    )

    print(
        f"batch complete: processed={stats.processed} skipped={stats.skipped} failed={stats.failed}"
    )
    for err in stats.errors:
        print(f"  ERROR {err}", file=sys.stderr)
    return 1 if stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
