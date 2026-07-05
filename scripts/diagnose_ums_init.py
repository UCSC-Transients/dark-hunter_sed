#!/usr/bin/env python3
"""Diagnose uberMS UMS NumPyro initialization for a single star."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from darkhunter_sed import data, fit, models


def _print_versions() -> None:
    try:
        import jax
        import numpyro

        print(f"jax {jax.__version__} backend={jax.default_backend()}")
        print(f"numpyro {numpyro.__version__}")
    except ImportError as exc:
        print(f"JAX/NumPyro import failed: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gaia_id", help="Gaia DR3 source_id")
    parser.add_argument("--from-spec-root", action="store_true")
    parser.add_argument("--photometry-dir", "-D", type=Path, default=None)
    parser.add_argument("--probe-svi", action="store_true", help="Run 1-step UMS SVI init probe (loads NNs)")
    parser.add_argument("--no-dust-av-prior", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    _print_versions()

    from darkhunter_rv.summary_paths import discover_primary_epoch_files
    from darkhunter_sed.config import photometry_dir, spec_root

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
    )

    print("\n=== fit_data (stellar + parallax) ===")
    for key in ("Teff", "[Fe/H]", "Mass", "Age_Gyr", "Flags_FLAME", "parallax"):
        if key in fit_data:
            print(f"  {key}: {fit_data[key]}")

    indict, spec_nn_list, phot_nn_path, mist_nn = fit.build_ums_indict(
        gaia_id,
        fit_data,
        progressbar=False,
        dust_av_prior=not args.no_dust_av_prior,
    )

    print("\n=== initpars ===")
    for key in sorted(indict["initpars"]):
        print(f"  {key} = {indict['initpars'][key]}")

    print("\n=== sampled priors (unchanged by Gaia init) ===")
    for key in ("EEP", "initial_Mass", "initial_[Fe/H]", "dist", "Av"):
        if key in indict["priors"]:
            print(f"  {key}: {indict['priors'][key]}")

    if args.probe_svi:
        print("\n=== UMS SVI init probe ===")
        ok, msg = fit.probe_ums_init(
            indict,
            spec_nn_list=spec_nn_list,
            phot_nn=phot_nn_path,
            mist_nn=mist_nn,
        )
        print(msg)
        if not ok:
            return 1

    if args.verbose:
        print("\n=== full indict (json) ===")
        slim = {
            "outfile": indict.get("outfile"),
            "initpars": indict.get("initpars"),
            "priors": indict.get("priors"),
            "svi": indict.get("svi"),
        }
        print(json.dumps(slim, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
