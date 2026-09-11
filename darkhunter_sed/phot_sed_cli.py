"""
CLI entry point ``darkhunter-sed-phot`` for Path-2 photometry SED fits.

Supported models: ``1star``, ``2star``, ``wd``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from darkhunter_sed.config import phot_sed_dir, photometry_dir
from darkhunter_sed.misty_iso import load_misty_predictor, resolve_mist_nn_path
from darkhunter_sed.phot_sed_fit import (
    BBPriorBounds,
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
        "--ir-bb",
        action="store_true",
        default=False,
        help=(
            "Add an IR blackbody component (T_bb, L_bb) to all models. "
            "The BB is placed at the system distance, summed with stellar flux "
            "before the shared F99 R_V=3.1 extinction is applied."
        ),
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
    p.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help=(
            "Save diagnostic plots after the fit: SED panel, WD corner (wd model only), "
            "and model-comparison bar chart.  Requires matplotlib >= 3.8."
        ),
    )
    # Prior mode flags (Issue #48)
    p.add_argument(
        "--prior-dust",
        action="store_true",
        default=False,
        help=(
            "Use a 3D dust-map Gaussian A_V prior (Bayestar/Edenhofer/Chen) with a "
            "hard cap at Av_SF (CSFD/S&F LOS, R_V=3.1 F99).  Default: flat Uniform[0, Av_SF]."
        ),
    )
    p.add_argument(
        "--prior-gaia-vac",
        action="store_true",
        default=False,
        help=(
            "Use a Gaussian parallax prior centred on the Gaia astrometric value "
            "(removes the parallax likelihood constraint to avoid double-counting)."
        ),
    )
    p.add_argument(
        "--prior-uberms",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to a uberMS spectral-fit JSON with 'feh'/'feh_sigma' and/or "
            "'mass'/'mass_sigma'.  Applied as Gaussian priors to the 1-star model only; "
            "silently ignored (with a warning) for 2-star and wd."
        ),
    )
    p.add_argument(
        "--ra-deg",
        type=float,
        default=None,
        metavar="DEG",
        help=(
            "Target right ascension (degrees, ICRS) for CSFD Av_SF query.  "
            "When omitted, derived from Gaia TAP query using the source id."
        ),
    )
    p.add_argument(
        "--dec-deg",
        type=float,
        default=None,
        metavar="DEG",
        help=(
            "Target declination (degrees, ICRS) for CSFD Av_SF query.  "
            "When omitted, derived from Gaia TAP query using the source id."
        ),
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


def _resolve_ra_dec(
    gaia_id: str,
    *,
    ra_deg: float | None,
    dec_deg: float | None,
) -> tuple[float | None, float | None]:
    """Resolve RA/Dec from CLI args or Gaia TAP query.

    Falls back to ``(None, None)`` on failure (caller uses av_hi_fallback).
    """
    import math

    if ra_deg is not None and dec_deg is not None:
        if math.isfinite(ra_deg) and math.isfinite(dec_deg):
            return ra_deg, dec_deg
    try:
        from darkhunter_sed.stellar_data import query_gaia_stellar_priors

        priors = query_gaia_stellar_priors(gaia_id)
        ra = priors.get("RA")
        dec = priors.get("Dec")
        if ra is not None and dec is not None and math.isfinite(float(ra)) and math.isfinite(float(dec)):
            return float(ra), float(dec)
    except Exception as exc:
        import sys

        print(f"Warning: could not resolve RA/Dec for {gaia_id}: {exc}", file=sys.stderr)
    return None, None


def _build_prior_spec_for_rows(
    gaia_id: str,
    rows: list,
    *,
    model: str,
    prior_dust: bool,
    prior_gaia_vac: bool,
    uberms_path: Path | None,
    ra_deg: float | None,
    dec_deg: float | None,
):
    """Build PhotSedPriorSpec from CLI args and Gaia parallax row.

    Returns ``None`` when no parallax row is found and no prior flags are set
    (keeps backward-compatible behaviour for callers that pass ``bounds``).
    """
    import math

    from darkhunter_sed.phot_sed_fit import _PLX_BAND
    from darkhunter_sed.phot_sed_priors import build_prior_spec

    plx_rows = [r for r in rows if r.band == _PLX_BAND]
    if not plx_rows:
        if not (prior_dust or prior_gaia_vac or uberms_path):
            return None
        # No parallax row — use fallback defaults.
        plx_obs, plx_err = 5.0, 0.5  # ~200 pc placeholder; CSFD still applies
    else:
        plx_obs = float(plx_rows[0].mag)
        plx_err = float(plx_rows[0].err)

    ra, dec = _resolve_ra_dec(gaia_id, ra_deg=ra_deg, dec_deg=dec_deg)
    if ra is None or dec is None:
        # Cannot query CSFD; use fallback Av_SF
        if not (prior_dust or prior_gaia_vac or uberms_path):
            return None
        ra, dec = 0.0, 0.0  # galactic l,b will be arbitrary; av_hi_fallback kicks in

    d_pc = 1000.0 / plx_obs if math.isfinite(plx_obs) and plx_obs > 0 else None

    return build_prior_spec(
        ra,
        dec,
        plx_obs,
        plx_err,
        prior_dust=prior_dust,
        prior_gaia_vac=prior_gaia_vac,
        uberms_path=uberms_path,
        model=model,
        d_pc=d_pc,
    )


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


def _resolve_bergeron_bands(wd_dir_arg: Path | None) -> frozenset[str]:
    """Return Bergeron-available band names, or empty set if WD dir not found."""
    wd_dir = _resolve_wd_dir(wd_dir_arg)
    if wd_dir is None:
        return frozenset()
    try:
        from darkhunter_sed.bergeron_wd import BergeronGrid
        grid = BergeronGrid.from_dir(wd_dir, "DA")
        return frozenset(grid.available_bands)
    except Exception:
        return frozenset()


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
        # WD+star model: needs MIST+PHOENIX for the MS companion component.
        from darkhunter_sed.filters_synphot import BAND_REGISTRY
        from darkhunter_sed.phot_sed_fit import _GAIA_CONSTRAINT_BANDS

        _skip = {"WISE_W3", "WISE_W4"}
        rows = [
            r for r in all_rows
            if r.band not in _skip
            and (
                r.band in BAND_REGISTRY
                or r.band in _GAIA_CONSTRAINT_BANDS
                # Bergeron-only bands (e.g. GALEX) not in BAND_REGISTRY:
                # pass them through so the Bergeron component can use them.
                or r.band in _resolve_bergeron_bands(args.wd_dir)
            )
        ]
        if not any(r.band in BAND_REGISTRY or r.band in _GAIA_CONSTRAINT_BANDS for r in rows):
            print(
                f"No registered photometry bands in {phot_path}", file=sys.stderr
            )
            return 1
        mist_nn  = resolve_mist_nn_path(args.mist_nn)
        predictor = load_misty_predictor(mist_nn)
        grid_phx  = PhoenixGrid(root=args.phoenix_dir)
        bandpasses = load_bandpasses_for_bands(
            [r.band for r in rows
             if r.band not in _GAIA_CONSTRAINT_BANDS and r.band in BAND_REGISTRY]
        )
        common_kw = None  # WD path has its own keyword handling below
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
        prior_spec = _build_prior_spec_for_rows(
            gaia_id,
            rows,
            model=args.model,
            prior_dust=bool(args.prior_dust),
            prior_gaia_vac=bool(args.prior_gaia_vac),
            uberms_path=args.prior_uberms,
            ra_deg=args.ra_deg,
            dec_deg=args.dec_deg,
        )
        bb_bounds = BBPriorBounds() if args.ir_bb else None
        if args.model == "1star":
            result, paths = run_1star_fit(rows, bounds=None, prior_spec=prior_spec, bb_bounds=bb_bounds, **common_kw)
        else:
            result, paths = run_2star_fit(rows, bounds=TwoStarPriorBounds(), prior_spec=prior_spec, bb_bounds=bb_bounds, **common_kw)
        print(
            f"{args.model} fit gaia_id={gaia_id}  lnZ={result.logz:.3f}±{result.logz_err:.3f}  "
            f"BIC={result.bic:.3f}  lnL_max={result.ln_l_max:.3f}"
        )
        print(f"summary: {paths['summary_json']}")
        print(f"samples: {paths['samples_npz']}")
        if args.plot:
            _run_phoenix_diagnostics(
                rows=rows,
                result=result,
                gaia_id=gaia_id,
                model=args.model,
                predictor=predictor,
                grid_phx=grid_phx,
                bandpasses=bandpasses,
                out_dir=Path(out_dir),
            )

    elif args.model == "wd":
        from darkhunter_sed.wd_model import WDStarPriorBounds, run_wd_plus_star_fit

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

        wd_results = run_wd_plus_star_fit(
            rows,
            wd_dir=wd_dir,
            mist_predictor=predictor,
            phoenix_grid=grid_phx,
            bandpasses=bandpasses,
            gaia_id=gaia_id,
            nlive=int(args.nlive),
            maxiter=args.maxiter,
            seed=int(args.seed),
            dlogz=float(args.dlogz),
            outdir=wd_out,
        )
        for res in wd_results:
            print(
                f"wd+star fit  atm={res['atm_type']}  ifmr={res['ifmr']}  "
                f"lnZ={res['logevidence']:.2f}  "
                f"M_WD_med={float(np.nanmedian(res['m_wd_samples'])):.3f}"
            )
        print(f"wd+star outputs: {wd_out}")

    else:
        print(f"Unsupported model: {args.model}", file=sys.stderr)
        return 2

    return 0


def _run_phoenix_diagnostics(
    rows: list,
    result: object,
    gaia_id: str,
    model: str,
    predictor: object,
    grid_phx: object,
    bandpasses: object,
    out_dir: Path,
) -> None:
    """
    Compute best-fit predictions and save diagnostic plots for a PHOENIX fit.

    Parameters
    ----------
    rows :
        Observed :class:`~darkhunter_sed.phot_sed_io.PhotRow` list.
    result :
        :class:`~darkhunter_sed.phot_sed_fit.FitResult1Star` or
        :class:`~darkhunter_sed.phot_sed_fit.FitResult2Star`.
    gaia_id :
        Source id for filenames and titles.
    model :
        ``"1star"`` or ``"2star"``.
    predictor, grid_phx, bandpasses :
        MISTy predictor, PHOENIX grid, and synphot bandpasses.
    out_dir :
        Directory for output figures.

    Limits
    ------
    Fails silently if ``matplotlib`` is unavailable or PHOENIX/MIST model
    evaluation raises an exception (prints a warning to stderr).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print(
            "WARNING: --plot requires matplotlib >= 3.8; skipping diagnostics.",
            file=sys.stderr,
        )
        return

    from darkhunter_sed.phot_sed_diagnostics import save_all_diagnostics
    from darkhunter_sed.phot_sed_fit import _GAIA_CONSTRAINT_BANDS, _SIGMA_INT_IDX
    from darkhunter_sed.phot_sed_models import predict_1star_phot, OneStarParams

    phot_bands = [r.band for r in rows if r.band not in _GAIA_CONSTRAINT_BANDS]

    model_preds: dict[str, dict[str, float]] = {}

    try:
        if model == "1star":
            pred = predict_1star_phot(
                result.best_theta[:_SIGMA_INT_IDX],
                phot_bands,
                mist_predictor=predictor,
                phoenix_grid=grid_phx,
                bandpasses=bandpasses,
            )
            model_preds["1-star"] = pred.mags
        else:
            # 2-star: combined + per-component
            from darkhunter_sed.phot_sed_models import (
                TwoStarParams,
                predict_2star_phot,
                solve_eep2_for_age_match,
            )
            from darkhunter_sed.misty_iso import evaluate_mist

            pred_combined = predict_2star_phot(
                result.best_theta,
                phot_bands,
                mist_predictor=predictor,
                phoenix_grid=grid_phx,
                bandpasses=bandpasses,
            )
            model_preds["2-star sum"] = pred_combined.mags

            p = TwoStarParams.from_array(result.best_theta)
            mist1 = evaluate_mist(p.eep1, p.mass1, p.feh, p.afe, predictor=predictor)
            eep2 = solve_eep2_for_age_match(
                mist1.age_gyr, p.mass2, p.feh, p.afe, predictor=predictor
            )
            if eep2 is not None:
                star1_p = OneStarParams(
                    eep=p.eep1, mass=p.mass1, feh=p.feh, afe=p.afe,
                    a_v=p.a_v, parallax_mas=p.parallax_mas,
                )
                star2_p = OneStarParams(
                    eep=eep2, mass=p.mass2, feh=p.feh, afe=p.afe,
                    a_v=p.a_v, parallax_mas=p.parallax_mas,
                )
                pred_s1 = predict_1star_phot(
                    star1_p, phot_bands,
                    mist_predictor=predictor, phoenix_grid=grid_phx, bandpasses=bandpasses,
                )
                pred_s2 = predict_1star_phot(
                    star2_p, phot_bands,
                    mist_predictor=predictor, phoenix_grid=grid_phx, bandpasses=bandpasses,
                )
                model_preds["star1"] = pred_s1.mags
                model_preds["star2"] = pred_s2.mags
    except Exception as exc:
        print(f"WARNING: best-fit prediction failed ({exc}); skipping diagnostics.", file=sys.stderr)
        return

    model_scores: dict[str, dict] = {
        model if model == "1star" else "2-star sum": {
            "logz": result.logz,
            "bic": result.bic,
        }
    }

    diag_dir = out_dir / gaia_id / "diagnostics"
    diag_paths = save_all_diagnostics(
        rows=rows,
        model_preds=model_preds,
        model_scores=model_scores,
        gaia_id=gaia_id,
        out_dir=diag_dir,
    )
    for key, p in diag_paths.items():
        print(f"diagnostic [{key}]: {p}")



if __name__ == "__main__":
    raise SystemExit(main())
