"""Push uberMS M1 (m1_msun) into RV summary.txt and website data.csv."""

from __future__ import annotations

import argparse
import csv
import fcntl
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from darkhunter_sed.config import rv_output_dir, sed_summaries_dir
from darkhunter_sed.posterior import (
    m1_msun_from_summary,
    read_sed_summary,
    sed_summary_path,
)

logger = logging.getLogger(__name__)

_SED_M1_KEYS = ("M1", "m1_msun", "M1_p16", "M1_p84")
_DEFAULT_DATA_CSV = Path("/var/www/html/darkhunter/rv/tables/data.csv")


def _gaia_id_from_sed_summary_name(path: Path) -> str | None:
    m = re.match(r"Gaia_DR3_(\d+)_sed_summary\.json$", path.name)
    return m.group(1) if m else None


def m1_block_from_doc(doc: dict[str, Any]) -> dict[str, float] | None:
    """Return median/p16/p84 for luminous M1, or None if missing."""
    block = doc.get("m1_msun")
    if not isinstance(block, dict):
        fits = doc.get("fits") or {}
        ums = fits.get("ums") or {}
        block = ums.get("m1_msun") or (ums.get("parameters") or {}).get("initial_Mass")
    if not isinstance(block, dict):
        return None
    try:
        med = float(block["median"])
        p16 = float(block.get("p16", med))
        p84 = float(block.get("p84", med))
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(med) or med <= 0.0:
        return None
    out = {"median": med}
    if math.isfinite(p16):
        out["p16"] = p16
    if math.isfinite(p84):
        out["p84"] = p84
    return out


def patch_summary_m1(summary_path: Path, m1: dict[str, float]) -> bool:
    """
    Insert/update M1 keys in ``[GAIA METADATA]``.

    Returns True if the file was modified.
    """
    if not summary_path.is_file():
        logger.warning("summary missing: %s", summary_path)
        return False

    text = summary_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    meta_start = None
    meta_end = None
    for i, line in enumerate(lines):
        if line.strip() == "[GAIA METADATA]":
            meta_start = i
            continue
        if meta_start is not None and meta_end is None:
            s = line.strip()
            if s.startswith("[") and s.endswith("]") and s != "[GAIA METADATA]":
                meta_end = i
                break
    if meta_start is None:
        logger.warning("no [GAIA METADATA] in %s", summary_path)
        return False
    if meta_end is None:
        meta_end = len(lines)

    new_vals = {
        "M1": f"{m1['median']:.6f}",
        "m1_msun": f"{m1['median']:.6f}",
    }
    if "p16" in m1:
        new_vals["M1_p16"] = f"{m1['p16']:.6f}"
    if "p84" in m1:
        new_vals["M1_p84"] = f"{m1['p84']:.6f}"

    body = lines[meta_start + 1 : meta_end]
    kept: list[str] = []
    seen: set[str] = set()
    changed = False
    for line in body:
        raw = line.strip()
        if not raw or raw.startswith("#"):
            kept.append(line)
            continue
        if ":" not in raw:
            kept.append(line)
            continue
        key, _ = raw.split(":", 1)
        key = key.strip()
        if key in new_vals:
            seen.add(key)
            new_line = f"{key}: {new_vals[key]}\n"
            if line != new_line and line.rstrip("\n") + "\n" != new_line:
                changed = True
            kept.append(new_line)
        else:
            kept.append(line if line.endswith("\n") else line + "\n")

    for key, val in new_vals.items():
        if key not in seen:
            kept.append(f"{key}: {val}\n")
            changed = True

    if not changed:
        # Still rewrite if keys were missing from seen but values matched — already handled.
        # Detect no-op when all keys present with same values.
        return False

    out_lines = lines[: meta_start + 1] + kept + lines[meta_end:]
    summary_path.write_text("".join(out_lines), encoding="utf-8")
    return True


def _gaia_id_from_row(gaia_cell: str) -> str:
    m = re.search(r"(\d{8,})", gaia_cell or "")
    return m.group(1) if m else ""


def patch_data_csv_m1(data_csv: Path, gaia_id: str, m1_msun: float) -> bool:
    """
    Set ``M1 (Msun)`` for ``gaia_id`` in website ``data.csv``.

    Uses an exclusive flock on the CSV path. Returns True if a row was updated.
    """
    if not data_csv.is_file():
        logger.warning("data.csv missing: %s", data_csv)
        return False

    lock_path = data_csv.with_suffix(data_csv.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            with data_csv.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
            if not rows:
                logger.warning("data.csv empty: %s", data_csv)
                return False
            hdr = rows[0]
            if "M1 (Msun)" not in hdr or "GAIA NAME" not in hdr:
                logger.warning("data.csv missing M1/GAIA NAME columns: %s", data_csv)
                return False
            m1_i = hdr.index("M1 (Msun)")
            gaia_i = hdr.index("GAIA NAME")
            target = str(gaia_id).strip()
            updated = False
            for r in rows[1:]:
                if not r:
                    continue
                while len(r) <= max(m1_i, gaia_i):
                    r.append("")
                sid = _gaia_id_from_row(r[gaia_i])
                if sid != target:
                    continue
                new_val = f"{float(m1_msun):.5f}"
                if r[m1_i] != new_val:
                    r[m1_i] = new_val
                    updated = True
                break
            else:
                logger.warning("gaia id %s not found in %s", target, data_csv)
                return False
            if updated:
                with data_csv.open("w", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerows(rows)
            return updated
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def push_m1_for_gaia_id(
    gaia_id: str,
    *,
    summaries_dir: Path | None = None,
    rv_out: Path | None = None,
    data_csv: Path | None = None,
    sed_summary: Path | None = None,
) -> dict[str, Any]:
    """
    Load sed_summary for ``gaia_id`` and push M1 into summary + data.csv.

    Returns a status dict with keys: ok, m1_msun, summary_updated, csv_updated, reason.
    """
    gid = str(gaia_id).strip()
    path = Path(sed_summary) if sed_summary is not None else sed_summary_path(gid, summaries_dir)
    if not path.is_file():
        return {
            "ok": False,
            "gaia_id": gid,
            "m1_msun": None,
            "summary_updated": False,
            "csv_updated": False,
            "reason": f"sed_summary missing: {path}",
        }
    doc = read_sed_summary(path)
    block = m1_block_from_doc(doc)
    if block is None:
        med = m1_msun_from_summary(doc)
        if med is None:
            return {
                "ok": False,
                "gaia_id": gid,
                "m1_msun": None,
                "summary_updated": False,
                "csv_updated": False,
                "reason": "no m1_msun in sed_summary",
            }
        block = {"median": med}

    rv_out = rv_out or rv_output_dir()
    summary_path = rv_out / f"Gaia_DR3_{gid}_summary.txt"
    data_csv = data_csv or Path(os.environ.get("DATA_CSV", str(_DEFAULT_DATA_CSV)))

    summ_ok = patch_summary_m1(summary_path, block)
    csv_ok = patch_data_csv_m1(data_csv, gid, block["median"])
    return {
        "ok": True,
        "gaia_id": gid,
        "m1_msun": block["median"],
        "summary_updated": summ_ok,
        "csv_updated": csv_ok,
        "summary_path": str(summary_path),
        "data_csv": str(data_csv),
        "reason": "pushed",
    }


def discover_sed_summary_gaia_ids(summaries_dir: Path | None = None) -> list[str]:
    root = summaries_dir or sed_summaries_dir()
    if not root.is_dir():
        return []
    ids: list[str] = []
    for p in sorted(root.glob("Gaia_DR3_*_sed_summary.json")):
        gid = _gaia_id_from_sed_summary_name(p)
        if gid:
            ids.append(gid)
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Push uberMS M1 from sed_summary.json into RV summary.txt and website data.csv."
    )
    parser.add_argument(
        "gaia_ids",
        nargs="*",
        help="Gaia source id(s); omit with --all to scan sed_summaries",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Push for every Gaia_DR3_*_sed_summary.json under sed_summaries",
    )
    parser.add_argument(
        "--summaries-dir",
        type=Path,
        default=None,
        help=f"sed_summaries dir (default {sed_summaries_dir()})",
    )
    parser.add_argument(
        "--rv-output-dir",
        type=Path,
        default=None,
        help=f"RV output with summaries (default {rv_output_dir()})",
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=None,
        help=f"Website tables/data.csv (default env DATA_CSV or {_DEFAULT_DATA_CSV})",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    ids = list(args.gaia_ids)
    if args.all:
        ids = discover_sed_summary_gaia_ids(args.summaries_dir)
    if not ids:
        print("No Gaia ids to process (pass ids or --all)", file=sys.stderr)
        return 2

    n_ok = 0
    n_fail = 0
    for gid in ids:
        result = push_m1_for_gaia_id(
            gid,
            summaries_dir=args.summaries_dir,
            rv_out=args.rv_output_dir,
            data_csv=args.data_csv,
        )
        if result["ok"]:
            n_ok += 1
            logger.info(
                "pushed M1=%.5f for %s summary=%s csv=%s",
                result["m1_msun"],
                gid,
                result["summary_updated"],
                result["csv_updated"],
            )
        else:
            n_fail += 1
            logger.warning("skip %s: %s", gid, result["reason"])

    print(f"push_m1 complete: ok={n_ok} failed={n_fail}")
    return 1 if n_fail and not n_ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
