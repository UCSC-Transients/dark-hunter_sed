#!/usr/bin/env python3
"""Convert Gaia/APF epoch .txt to uberMS-ready FITS using RV continuum normalization."""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

from darkhunter_sed import spectrum


def _epoch_sort_key(p: Path) -> tuple:
    m = re.search(r"epoch_(\d+)", p.name, re.I)
    if m:
        return (0, int(m.group(1)), p.name.lower())
    return (1, p.name.lower(), p.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Gaia multi-order .txt spectra to FITS (RV blaze + CR normalization)."
    )
    parser.add_argument("--source-id", required=True, help="Gaia source_id")
    parser.add_argument("--glob", "-g", dest="glob_pattern", required=True, help="Input .txt glob")
    parser.add_argument("--out-dir", "-o", type=Path, required=True, help="Output directory for FITS")
    args = parser.parse_args(argv)

    paths = sorted((Path(p) for p in glob.glob(args.glob_pattern)), key=_epoch_sort_key)
    if not paths:
        print(f"No files match {args.glob_pattern!r}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    blaze = spectrum.load_blaze_calibration()

    for txt in paths:
        out = args.out_dir / (txt.stem + ".fits")
        spectrum.txt_to_uberms_fits(txt, out, blaze_calibration=blaze)
        print(f"Wrote {out}")

    if len(paths) < 2:
        print(
            "Note: uberMS UMS (dva) requires at least 2 epoch spectra.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
