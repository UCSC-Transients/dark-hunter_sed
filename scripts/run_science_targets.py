#!/usr/bin/env python3
"""
Diagnostic SED fits for three science programs.

Three science programs
----------------------
1. uberMS (Foley et al.)
   Gaia IDs: 4219507576765009536, 77413727493690112
   Models:   1-star only
   Goal:     Compare photometric SED fit to existing spectroscopic summaries.

2. El-Badry+2026 two-temperature composites
   Gaia IDs: 5336217383170465792, 5611909581558507648
   Models:   1-star + 2-star
   Goal:     Check whether 2-star model is preferred; recover component parameters.

3. Yamaguchi+2024 ultramassive WDs
   Gaia IDs: 6475655404885617920, 843829411442724864,
             5033197892724532736, 2692960678029100800
   Models:   1-star + WD (DA/DB × MIST/PARSEC)
   Goal:     Recover M_WD and M_i; compare 1-star vs WD model evidence.

Usage
-----
    python3 scripts/run_science_targets.py [--nlive 200] [--outdir output/phot_sed]

Photometry gather
-----------------
For each target the script resolves the photometry FITS in this order:
  1. output/photometry/<id>_phot.fits  (canonical project location)
  2. Any directory in _PHOT_SEARCH_DIRS  (e.g. ~/stellar/gaia)
     → file is copied into output/photometry/ automatically
  3. Live catalog query via darkhunter_sed.photometry_gather
     → gathered file is written to output/photometry/
Pass --no-gather to skip step 3 and fail instead of querying the network.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repository root — resolve the installed main repo from the worktree location.
#
# The script lives in a worktree alongside the main repo:
#   .../seds/dark-hunter_sed/           ← main repo (has updated phot_sed_cli)
#   .../seds/dark-hunter_sed-wt-*/      ← this worktree
#
# Subprocess calls must use _MAIN_REPO as cwd so Python finds the main repo's
# darkhunter_sed package instead of the worktree's stale copy.
# ---------------------------------------------------------------------------
_WORKTREE_ROOT = Path(__file__).resolve().parent.parent
_MAIN_REPO = _WORKTREE_ROOT.parent / "dark-hunter_sed"
if not (_MAIN_REPO / "darkhunter_sed" / "__init__.py").exists():
    # Fallback: worktree IS the main repo (or non-standard layout).
    _MAIN_REPO = _WORKTREE_ROOT

# Canonical photometry directory lives in the main repo; the worktree shares it.
_PHOT_DIR = _MAIN_REPO / "output" / "photometry"
_SPEC_SUM_DIR = _MAIN_REPO / "output" / "sed_summaries"
_WD_DIR_DEFAULT = Path("~/stellar/wd").expanduser()

# Ordered list of directories searched for pre-existing photometry FITS files
# when the canonical output/photometry/ location is empty.
_PHOT_SEARCH_DIRS: list[Path] = [
    Path("~/stellar/gaia").expanduser(),
    Path("~/darkhunter/seds/output/photometry").expanduser(),
    _WORKTREE_ROOT / "output" / "photometry",  # worktree may have copies
]

# ---------------------------------------------------------------------------
# Science program definitions
# ---------------------------------------------------------------------------
# Each entry: (program_label, gaia_id, [models])
# Models: "1star", "2star", "wd"
_SCIENCE_TARGETS: list[tuple[str, str, list[str]]] = [
    # --- uberMS ---
    ("uberMS",      "4219507576765009536",  ["1star"]),
    ("uberMS",      "77413727493690112",    ["1star"]),
    # --- El-Badry+2026 two-temperature ---
    ("ElBadry2026", "5336217383170465792",  ["1star", "2star"]),
    ("ElBadry2026", "5611909581558507648",  ["1star", "2star"]),
    # --- Yamaguchi+2024 ultramassive WDs ---
    ("Yamaguchi2024", "6475655404885617920", ["1star", "wd"]),
    ("Yamaguchi2024", "843829411442724864",  ["1star", "wd"]),
    ("Yamaguchi2024", "5033197892724532736", ["1star", "wd"]),
    ("Yamaguchi2024", "2692960678029100800", ["1star", "wd"]),
]

# Keys used for the uberMS spectroscopic cross-check (phot_sed 1-star summary)
_COMPARE_KEYS = {
    "Teff_K":      lambda s: s.get("best_mist", {}).get("teff_k"),
    "distance_pc": lambda s: s.get("distance_pc"),
    "Av_mag":      lambda s: (s.get("best_theta") or {}).get("Av"),
    "logz":        lambda s: s.get("logz"),
    "bic":         lambda s: s.get("bic"),
}

# Keys from the spectroscopic summary (for uberMS targets with existing fits)
def _spec_values(spec: dict[str, Any]) -> dict[str, Any | None]:
    ums = (spec.get("fits") or {}).get("ums") or {}
    params = ums.get("parameters") or {}
    return {
        "Teff_K":      (params.get("Teff") or {}).get("median"),
        "distance_pc": (ums.get("dist_pc") or {}).get("median"),
        "Av_mag":      (params.get("Av") or {}).get("median"),
        "m1_msun":     (ums.get("m1_msun") or {}).get("median"),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _canonical_phot_path(gaia_id: str) -> Path:
    """Canonical location: output/photometry/<id>_phot.fits."""
    return _PHOT_DIR / f"{gaia_id}_phot.fits"


def _ensure_phot(gaia_id: str, *, auto_gather: bool = True) -> Path | None:
    """
    Ensure a photometry FITS exists in output/photometry/ and return its path.

    Resolution order:
      1. output/photometry/<id>_phot.fits  (already in place)
      2. Any _PHOT_SEARCH_DIRS entry       (copy into output/photometry/)
      3. Live catalog gather               (network; skipped if auto_gather=False)

    Returns None if the file cannot be found or gathered.
    """
    dest = _canonical_phot_path(gaia_id)
    if dest.is_file():
        return dest

    # Search fallback directories.
    for search_dir in _PHOT_SEARCH_DIRS:
        candidate = search_dir / f"{gaia_id}_phot.fits"
        if candidate.is_file():
            _PHOT_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, dest)
            print(f"  [phot] Copied from {candidate}")
            return dest

    # Live gather from catalogs.
    if not auto_gather:
        print(f"  [phot] Not found; --no-gather set — skipping {gaia_id}")
        return None

    print(f"  [phot] Querying catalogs for {gaia_id} …")
    try:
        from darkhunter_sed.photometry_gather import gather_photometry_for_star
        _PHOT_DIR.mkdir(parents=True, exist_ok=True)
        path = gather_photometry_for_star(gaia_id, outdir=_PHOT_DIR)
        print(f"  [phot] Gathered → {path}")
        return Path(path)
    except Exception as exc:
        print(f"  [phot] Gather failed: {exc}")
        return None


def _spec_sum_path(gaia_id: str) -> Path:
    return _SPEC_SUM_DIR / f"Gaia_DR3_{gaia_id}_sed_summary.json"


def _run_fit(
    gaia_id: str,
    model: str,
    *,
    outdir: Path,
    nlive: int,
    wd_dir: Path | None,
    dlogz: float,
    phot_path: Path | None = None,
    stride: int = 1,
    print_progress: bool = True,
) -> tuple[int, Path | None]:
    """
    Run a single fit via the CLI.  Returns (returncode, summary_json_path_or_None).
    """
    cmd = [
        sys.executable, "-m", "darkhunter_sed.phot_sed_cli",
        gaia_id,
        "--model", model,
        "--nlive", str(nlive),
        "--dlogz", str(dlogz),
        "--outdir", str(outdir),
        "--stride", str(stride),
    ]
    if print_progress:
        cmd.append("--print-progress")
    if phot_path is not None:
        cmd += ["--phot", str(phot_path)]
    if model == "wd":
        wd = wd_dir or _WD_DIR_DEFAULT
        cmd += ["--wd-dir", str(wd)]

    print(f"\n  CMD: {' '.join(cmd)}")
    # Run from the main repo so Python finds the correct darkhunter_sed package.
    result = subprocess.run(cmd, capture_output=False, cwd=_MAIN_REPO)
    rc = result.returncode

    # Locate the summary JSON written by the CLI.
    if rc == 0:
        if model == "wd":
            # WD summaries written as <outdir>/<gaia_id>/wd/wd_<atm>_<ifmr>_summary.json
            wd_out = outdir / gaia_id / "wd"
            summaries = sorted(wd_out.glob("wd_*_summary.json"))
            return rc, summaries[0] if summaries else None
        else:
            stem = f"Gaia_DR3_{gaia_id}_{model}"
            candidate = outdir / f"{stem}_summary.json"
            return rc, candidate if candidate.is_file() else None
    return rc, None


def _fmt(val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4g}"
    return str(val)


# ---------------------------------------------------------------------------
# Cross-check: uberMS phot vs spec summary
# ---------------------------------------------------------------------------
def _crosscheck_uberms(gaia_id: str, phot_summary: dict[str, Any]) -> None:
    spec_path = _spec_sum_path(gaia_id)
    if not spec_path.is_file():
        print(f"  [cross-check] No spectroscopic summary at {spec_path}; skipping.")
        return
    spec = json.loads(spec_path.read_text())
    sv = _spec_values(spec)

    print(f"\n  Cross-check vs spectroscopic summary ({spec_path.name}):")
    header = f"    {'Parameter':<16}  {'Phot-SED':>12}  {'Spec (UMS)':>12}  {'Delta':>12}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for key, extractor in _COMPARE_KEYS.items():
        phot_val = extractor(phot_summary)
        spec_val = sv.get(key)
        if phot_val is not None and spec_val is not None:
            delta = f"{phot_val - spec_val:+.4g}"
        else:
            delta = "—"
        print(f"    {key:<16}  {_fmt(phot_val):>12}  {_fmt(spec_val):>12}  {delta:>12}")

    # Also print spec-only keys not in phot summary
    for key in ("m1_msun",):
        val = sv.get(key)
        if val is not None:
            print(f"    {key:<16}  {'(spec only)':>12}  {_fmt(val):>12}  {'—':>12}")


# ---------------------------------------------------------------------------
# WD diagnostic: print all four combination summaries
# ---------------------------------------------------------------------------
def _report_wd(gaia_id: str, outdir: Path) -> None:
    wd_out = outdir / gaia_id / "wd"
    summaries = sorted(wd_out.glob("wd_*_summary.json"))
    if not summaries:
        print("  [wd] No WD summary files found.")
        return

    print(f"\n  WD fit results ({len(summaries)} combinations):")
    print(f"    {'Combo':<14}  {'lnZ':>8}  {'M_WD':>8}  {'M_i':>8}  {'Teff':>7}  {'logg':>6}  {'Av':>6}  {'extrap':>7}")
    print("    " + "-" * 80)
    for p in summaries:
        s = json.loads(p.read_text())
        combo = f"{s.get('atm_type','?')}/{s.get('ifmr','?')}"
        print(
            f"    {combo:<14}  "
            f"{_fmt(s.get('logevidence')):>8}  "
            f"{_fmt(s.get('m_wd_median')):>8}  "
            f"{_fmt(s.get('m_i_median')):>8}  "
            f"{_fmt(s.get('teff_median')):>7}  "
            f"{_fmt(s.get('logg_median')):>6}  "
            f"{_fmt(s.get('av_median')):>6}  "
            f"{str(s.get('extrap_mass','')):>7}"
        )
        unc = s.get("m_i_ifmr_unc_median")
        if unc is not None:
            print(f"      σ_Mi (IFMR sys): {_fmt(unc)} M_sun")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Diagnostic SED fits for three science programs."
    )
    ap.add_argument("--nlive",  type=int,   default=200,
                    help="Dynesty live points (default 200)")
    ap.add_argument("--dlogz",  type=float, default=0.5,
                    help="Dynesty dlogz stopping criterion (default 0.5)")
    ap.add_argument("--outdir", type=Path,
                    default=_MAIN_REPO / "output" / "phot_sed",
                    help="Output directory for phot_sed products")
    ap.add_argument("--wd-dir", type=Path, default=None,
                    help="Bergeron WD table directory (default ~/stellar/wd)")
    ap.add_argument("--stride", type=int, default=1, metavar="N",
                    help=(
                        "Subsample bandpass wavelength grid by N for PHOENIX integration "
                        "(default 1 = full; 4–10 gives ~4–10× speedup, same accuracy for "
                        "broad-band photometry). Use --stride 8 for a fast exploratory run."
                    ))
    ap.add_argument("--no-gather", action="store_true",
                    help="Skip live catalog gather; fail instead if phot FITS is absent")
    args = ap.parse_args(argv)

    outdir: Path = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    wd_dir: Path | None = (
        args.wd_dir.expanduser().resolve() if args.wd_dir else None
    )
    auto_gather: bool = not args.no_gather

    print("=" * 72)
    print("  Diagnostic SED fits — science targets")
    print(f"  outdir      : {outdir}")
    print(f"  nlive       : {args.nlive}  dlogz : {args.dlogz}  stride : {args.stride}")
    print(f"  auto-gather : {auto_gather}")
    print("=" * 72)

    results: list[dict[str, Any]] = []  # accumulate per-run status

    # --- Run fits ---
    for prog, gaia_id, models in _SCIENCE_TARGETS:
        print("\n" + "=" * 72)
        print(f"  Program : {prog}")
        print(f"  Gaia ID : {gaia_id}")
        print(f"  Models  : {', '.join(models)}")
        print("=" * 72)

        phot = _ensure_phot(gaia_id, auto_gather=auto_gather)
        if phot is None:
            print(f"  Phot    : UNAVAILABLE — skipping")
            results.append({
                "prog": prog, "gaia_id": gaia_id, "status": "SKIPPED (no phot)",
            })
            continue
        print(f"  Phot    : {phot}")

        for model in models:
            print(f"\n  --- {model} fit ---")
            rc, sum_path = _run_fit(
                gaia_id, model,
                outdir=outdir,
                nlive=args.nlive,
                wd_dir=wd_dir,
                dlogz=args.dlogz,
                phot_path=phot,
                stride=args.stride,
                print_progress=True,
            )
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            row: dict[str, Any] = {
                "prog": prog, "gaia_id": gaia_id, "model": model, "status": status,
            }
            if rc == 0 and sum_path is not None and sum_path.is_file():
                summary = json.loads(sum_path.read_text())
                row["summary"] = summary

                # Program-specific diagnostics
                if prog == "uberMS" and model == "1star":
                    _crosscheck_uberms(gaia_id, summary)
                elif model == "wd":
                    _report_wd(gaia_id, outdir)

            results.append(row)

    # --- Final summary table ---
    print("\n\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  {'Program':<14}  {'Gaia ID':<22}  {'Model':<8}  {'Status'}")
    print("  " + "-" * 60)
    for row in results:
        print(
            f"  {row.get('prog',''):<14}  "
            f"{row.get('gaia_id',''):<22}  "
            f"{row.get('model',''):<8}  "
            f"{row.get('status','')}"
        )

    n_ok = sum(1 for r in results if r.get("status") == "OK")
    n_tot = len(results)
    print(f"\n  {n_ok}/{n_tot} fits succeeded.")
    return 0 if n_ok == n_tot else 1


if __name__ == "__main__":
    sys.exit(main())
