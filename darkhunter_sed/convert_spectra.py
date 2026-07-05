#!/usr/bin/env python3
"""Convert Gaia/APF epoch .txt to uberMS-ready FITS using RV continuum normalization."""

from __future__ import annotations

import argparse
import glob
import logging
import re
import sys
from pathlib import Path

from darkhunter_sed import spectrum
from darkhunter_sed.region_picker import resolve_regions_json_for_star

logger = logging.getLogger(__name__)


def _epoch_sort_key(p: Path) -> tuple:
    m = re.search(r"epoch_(\d+)", p.name, re.I)
    if m:
        return (0, int(m.group(1)), p.name.lower())
    return (1, p.name.lower(), p.name)


def _gaia_id_from_paths(paths: list[Path]) -> str | None:
    for p in paths:
        m = re.search(r"Gaia_DR3_(\d+)_", p.name)
        if m:
            return m.group(1)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Gaia multi-order .txt spectra to FITS (RV blaze + CR normalization)."
    )
    parser.add_argument("--source-id", required=True, help="Gaia source_id")
    parser.add_argument("--glob", "-g", dest="glob_pattern", required=True, help="Input .txt glob")
    parser.add_argument("--out-dir", "-o", type=Path, required=True, help="Output directory for FITS")
    parser.add_argument(
        "--regions-json",
        type=Path,
        default=None,
        help="Regions/blaze JSON (default: auto per-star in output/masks/)",
    )
    parser.add_argument(
        "--no-auto-regions",
        action="store_true",
        help="Do not auto-resolve per-star regions JSON",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    paths = sorted((Path(p) for p in glob.glob(args.glob_pattern)), key=_epoch_sort_key)
    if not paths:
        print(f"No files match {args.glob_pattern!r}", file=sys.stderr)
        return 1

    gaia_id = str(args.source_id).strip()
    resolved_regions = resolve_regions_json_for_star(
        gaia_id,
        args.regions_json,
        auto_resolve=not args.no_auto_regions,
    )
    if resolved_regions is not None:
        logger.info("Using regions JSON %s", resolved_regions)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    blaze = spectrum.load_blaze_calibration()
    regions_kw = {"regions_json": str(resolved_regions)} if resolved_regions else {}

    for txt in paths:
        out = args.out_dir / (txt.stem + ".fits")
        spectrum.txt_to_uberms_fits(txt, out, blaze_calibration=blaze, **regions_kw)
        print(f"Wrote {out}")

    if len(paths) < 2:
        print(
            "Note: uberMS UMS (dva) requires at least 2 epoch spectra.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
