#!/usr/bin/env python3
"""Continuum residual diagnostics for UMS (fixed pc* + sinc_blaze ingest)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from darkhunter_rv.summary_paths import discover_primary_epoch_files

from darkhunter_sed import data, models
from darkhunter_sed.config import photometry_dir, samples_dir, spec_root
from darkhunter_sed.continuum_diagnostics import diagnose_continuum_from_samples


def _pick_interactive_backend() -> str:
    import matplotlib

    for backend in ("MacOSX", "TkAgg", "Qt5Agg"):
        try:
            matplotlib.use(backend, force=True)
            return backend
        except ImportError:
            continue
    return matplotlib.get_backend()


def _show_continuum_plot(
    gaia_id: str,
    diagnostic,
) -> None:
    import matplotlib.pyplot as plt

    m = diagnostic.metrics
    wave = diagnostic.obs_wave
    flux = diagnostic.obs_flux
    model = diagnostic.model_flux
    ratio = np.divide(
        flux,
        np.maximum(model, 1e-99),
        out=np.full_like(flux, np.nan),
        where=np.isfinite(model),
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax_spec, ax_ratio = axes
    ax_spec.step(wave, flux, color="k", lw=0.8, where="mid", label="data")
    ax_spec.plot(wave, model, color="C0", lw=1.0, label="model")
    ax_spec.set_ylabel("Normalized flux")
    ax_spec.legend(loc="upper right", fontsize=9)
    ax_spec.set_title(
        f"Gaia DR3 {gaia_id} epoch {m['epoch_index'] + 1} ({m['pc_policy']}); "
        f"Teff={m['Teff_median']:.0f} K"
    )

    ax_ratio.plot(wave, ratio, color="C2", lw=0.6)
    ax_ratio.axhline(1.0, color="k", ls="--", lw=0.5)
    ax_ratio.set_xlabel(r"Wavelength [$\AA$]")
    ax_ratio.set_ylabel("data / model")
    ax_ratio.set_title(
        f"continuum_rms={m['continuum_rms']:.4f}  "
        f"full_rms={m['full_rms']:.4f}  "
        f"median_ratio={m['median_ratio']:.4f}"
    )
    fig.tight_layout()
    plt.show()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gaia_id", help="Gaia DR3 source_id")
    parser.add_argument("--from-spec-root", action="store_true")
    parser.add_argument("--photometry-dir", "-D", type=Path, default=None)
    parser.add_argument("--epoch", type=int, default=0, help="0-based spectrum epoch")
    parser.add_argument(
        "--samples",
        type=Path,
        default=None,
        help="UMS samples FITS (default: output/samples/Gaia_DR3_<id>_ums.fits)",
    )
    parser.add_argument(
        "--use-posterior-pc",
        action="store_true",
        help="Use posterior median pc* instead of fixed 1,0,0,0",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Open interactive matplotlib window (full-resolution data vs model)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    gaia_id = str(args.gaia_id).strip()
    _, phot_nn, _ = models.model_paths()
    spec_paths = None
    if args.from_spec_root:
        spec_paths = discover_primary_epoch_files(spec_root(), gaia_id)

    fit_data = data.getdata(
        gaia_id,
        spectrum_paths=spec_paths,
        photometry_dir=args.photometry_dir or photometry_dir(),
        phot_nn=phot_nn,
        phot_outlier_sigma=3.0,
    )

    samples_path = args.samples or samples_dir() / f"Gaia_DR3_{gaia_id}_ums.fits"
    if not samples_path.is_file():
        print(f"Samples not found: {samples_path}", file=sys.stderr)
        return 1

    diagnostic = diagnose_continuum_from_samples(
        fit_data,
        str(samples_path),
        epoch_index=args.epoch,
        fixed_pc=not args.use_posterior_pc,
    )
    print(json.dumps(diagnostic.metrics, indent=2))

    if args.plot:
        backend = _pick_interactive_backend()
        logging.info("matplotlib backend=%s", backend)
        _show_continuum_plot(gaia_id, diagnostic)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
