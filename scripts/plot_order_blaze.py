#!/usr/bin/env python3
"""
Per-order blaze diagnostic: raw flux + sinc² blaze, normalized spectrum, data/model.

Uses APF export (.txt), calibrated blaze JSON, and Payne spec model (no modpoly pc terms).

Batch all orders with manual regions::

  scripts/run_local.sh scripts/plot_order_blaze.py GAIA_ID \\
    --from-spec-root --epoch 0 --all-orders --no-show \\
    --regions-json output/masks/regions_Gaia_DR3_....json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

from darkhunter_rv import io_utils
from darkhunter_rv.blaze import (
    FIT_MODEL_SINC_POLY,
    OrderBlazeModel,
    fit_order_blaze_from_profiles,
    normalize_order_sinc_blaze_only,
    strong_lines_in_span,
)
from darkhunter_rv import config as rv_config
from darkhunter_rv.summary_paths import discover_primary_epoch_files

from darkhunter_sed import data, models, spectrum
from darkhunter_sed.config import photometry_dir, samples_dir, spec_root
from darkhunter_sed.region_picker import (
    ManualMaskBundle,
    OrderRegions,
    fit_model_from_regions,
    fit_order_from_regions,
    load_regions_json,
    order_blaze_model_from_regions,
    order_has_stored_blaze,
    order_regions_from_doc,
    poly_order_from_regions,
    resolve_regions_json_for_star,
    resolve_order_regions,
    shade_mask_gaps,
)
from darkhunter_sed.stellar_data import (
    ensure_stellar_sys_path,
    import_airtovacuum,
    import_payne_genmod,
    resolve_spec_nn_path,
)


def _pick_interactive_backend() -> str:
    import matplotlib

    for backend in ("MacOSX", "TkAgg", "Qt5Agg"):
        try:
            matplotlib.use(backend, force=True)
            return backend
        except ImportError:
            continue
    return matplotlib.get_backend()


def _orders_with_blaze(spectrum_data: dict, blaze_cal) -> list[int]:
    """Echelle orders present in the spectrum that have a calibrated blaze model."""
    out: list[int] = []
    for order_num in sorted(spectrum_data.keys()):
        if blaze_cal is not None and blaze_cal.model_for_order(int(order_num)) is None:
            continue
        out.append(int(order_num))
    return out


def _orders_in_range(
    spectrum_data: dict,
    wave_min: float,
    wave_max: float,
    blaze_cal,
) -> list[int]:
    out: list[int] = []
    for order_num in sorted(spectrum_data.keys()):
        if blaze_cal is not None and blaze_cal.model_for_order(int(order_num)) is None:
            continue
        w = np.asarray(spectrum_data[order_num]["wavelength"], dtype=float)
        if np.any((w >= wave_min) & (w <= wave_max)):
            out.append(int(order_num))
    return out


def _pick_order(
    spectrum_data: dict,
    wave_min: float,
    wave_max: float,
    blaze_cal,
    order: int | None,
) -> int:
    candidates = _orders_in_range(spectrum_data, wave_min, wave_max, blaze_cal)
    if not candidates:
        raise ValueError(f"No echelle orders with blaze model in [{wave_min}, {wave_max}] Å")
    if order is not None:
        if int(order) not in candidates:
            raise ValueError(f"Order {order} not available; choose from {candidates}")
        return int(order)
    best = candidates[0]
    best_n = 0
    for o in candidates:
        w = np.asarray(spectrum_data[o]["wavelength"], dtype=float)
        n = int(np.sum((w >= wave_min) & (w <= wave_max)))
        if n > best_n:
            best_n = n
            best = o
    return best


def _shade_mask_gaps(ax, wavelength: np.ndarray, mask: np.ndarray, *, color: str, alpha: float) -> None:
    """Shade wavelength intervals where mask is False."""
    shade_mask_gaps(ax, wavelength, mask, color=color, alpha=alpha, zorder=1)


def _collect_order_profiles(
    spec_paths: list[Path],
    echelle_order: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    profiles: list[tuple[np.ndarray, np.ndarray]] = []
    for spec_path in spec_paths:
        try:
            _hdr, spec_data = io_utils.read_spectrum(str(spec_path))
        except OSError:
            continue
        if echelle_order not in spec_data:
            continue
        order = spec_data[echelle_order]
        w = np.asarray(order["wavelength"], dtype=float)
        f = np.asarray(order["flux"], dtype=float)
        if w.size < 20 or not np.any(np.isfinite(f) & (f > 0)):
            continue
        profiles.append((w, f))
    return profiles


def _normalize_order_sinc_blaze(
    wavelength: np.ndarray,
    flux: np.ndarray,
    eflux: np.ndarray,
    blaze_model: OrderBlazeModel,
    *,
    mask_bundle: ManualMaskBundle | None = None,
    order_regions: OrderRegions | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-order sinc_blaze_only normalization (matches SED ingest)."""
    fit_model = fit_model_from_regions(order_regions)
    poly_order = poly_order_from_regions(order_regions)
    if mask_bundle is None:
        return normalize_order_sinc_blaze_only(
            wavelength,
            flux,
            eflux,
            blaze_model,
            fit_model=fit_model,
            poly_order=poly_order,
        )
    return normalize_order_sinc_blaze_only(
        wavelength,
        flux,
        eflux,
        blaze_model,
        base_mask=mask_bundle.base_mask,
        fixed_line_mask=mask_bundle.fixed_line_mask,
        fixed_cont_mask=mask_bundle.fixed_cont_mask,
        cr_mask=mask_bundle.cr_mask,
        fit_model=fit_model,
        poly_order=poly_order,
    )


def _shade_manual_regions(
    ax,
    wavelength: np.ndarray,
    bundle: ManualMaskBundle,
    order_regions: OrderRegions,
) -> None:
    """Overlay manual continuum (green), line (red), and CR rejects (gray)."""
    w = np.asarray(wavelength, float)
    cont_spans, line_spans = resolve_order_regions(
        order_regions.get("continuum_regions", []),
        order_regions.get("line_regions", []),
    )
    for lo, hi in cont_spans:
        ax.axvspan(lo, hi, color="green", alpha=0.12, lw=0, zorder=1)
    for lo, hi in line_spans:
        ax.axvspan(lo, hi, color="red", alpha=0.15, lw=0, zorder=1)
    _shade_mask_gaps(ax, w, bundle.cr_mask, color="0.5", alpha=0.1)


def _model_pars_from_samples(
    samples_path: Path,
    spec_i: int,
    *,
    vrad_kms: float,
    lsf: float = 60000.0,
) -> list[float]:
    samples = Table.read(samples_path, format="fits")
    bfdict = {k: float(np.median(samples[k])) for k in samples.colnames}
    vrad_key = f"vrad_{spec_i}"
    if vrad_key in bfdict and np.isfinite(bfdict[vrad_key]):
        vrad = float(bfdict[vrad_key])
    elif np.isfinite(vrad_kms):
        vrad = float(vrad_kms)
    else:
        vrad = 0.0
    return [
        bfdict["Teff"],
        bfdict["log(g)"],
        bfdict["[Fe/H]"],
        bfdict["[a/Fe]"],
        vrad,
        bfdict.get("vstar", 8.0),
        bfdict.get("vmic", 1.0),
        lsf,
    ]


def plot_order_blaze_diagnostic(
    gaia_id: str,
    spec_path: Path,
    *,
    echelle_order: int,
    wave_min: float = 5150.0,
    wave_max: float = 5300.0,
    samples_path: Path | None = None,
    epoch_index: int = 0,
    vrad_kms: float = float("nan"),
    refit_profiles: list[tuple[np.ndarray, np.ndarray]] | None = None,
    mask_profiles: list[tuple[np.ndarray, np.ndarray]] | None = None,
    order_regions: OrderRegions | None = None,
    show_mask_stages: bool = False,
    save_path: Path | None = None,
    show: bool = True,
    blaze_only: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    blaze_cal = spectrum.load_blaze_calibration()
    if blaze_cal is None:
        raise RuntimeError("Blaze calibration not found; set DARKHUNTER_BLAZE_CALIBRATION or RV config")

    _header, spectrum_data = io_utils.read_spectrum(str(spec_path))
    if echelle_order not in spectrum_data:
        raise ValueError(f"Order {echelle_order} not in {spec_path.name}")

    order = spectrum_data[echelle_order]
    wave_air = np.asarray(order["wavelength"], dtype=float)
    flux_raw = np.asarray(order["flux"], dtype=float)
    eflux_raw = np.asarray(order["eflux"], dtype=float)

    cal_model = blaze_cal.model_for_order(int(echelle_order))
    blaze_model = cal_model
    blaze_label = "calibrated sinc²"
    has_stored_blaze = order_has_stored_blaze(order_regions)
    if has_stored_blaze:
        blaze_model = order_blaze_model_from_regions(
            order_regions,
            int(echelle_order),
            wave_air,
            cal_model,
        )
        blaze_label = "stored sinc² (picker)"
    elif refit_profiles is not None and len(refit_profiles) >= 3:
        wmins = [float(np.min(w)) for w, _f in refit_profiles]
        wmaxs = [float(np.max(w)) for w, _f in refit_profiles]
        rests = strong_lines_in_span(float(np.min(wmins)), float(np.max(wmaxs)))
        refit = fit_order_blaze_from_profiles(
            refit_profiles,
            echelle_order,
            rest_lines=rests if rests else [],
            line_mask_half_width=(
                cal_model.line_mask_half_width_angstrom if cal_model is not None else 22.0
            ),
        )
        if refit is not None:
            blaze_model = refit
            blaze_label = f"refit sinc² (N={refit.n_spectra_fit})"
    if blaze_model is None:
        raise ValueError(f"No blaze model for echelle order {echelle_order}")

    fit_preview = fit_order_from_regions(
        wave_air,
        flux_raw,
        order_regions,
        blaze_model,
        echelle_order=int(echelle_order),
    )
    if fit_preview is None:
        raise ValueError(f"Shared blaze fit failed for echelle order {echelle_order}")

    if fit_preview.fit_model == FIT_MODEL_SINC_POLY:
        blaze_label = f"{blaze_label} ×poly"

    in_win = (wave_air >= wave_min) & (wave_air <= wave_max)
    if not np.any(in_win):
        raise ValueError(f"Order {echelle_order} has no pixels in [{wave_min}, {wave_max}] Å")

    w_air = wave_air[in_win]
    f_raw = flux_raw[in_win]
    e_raw = eflux_raw[in_win]

    blaze_overlay = fit_preview.envelope[in_win]
    cont_on_epoch = fit_preview.continuum_mask[in_win]
    mask_bundle = fit_preview.mask_bundle
    stage_counts = list(fit_preview.stage_counts)
    final_thr = float(getattr(rv_config, "BLAZE_ITERATIVE_THRESHOLDS", (0.92,))[-1])
    if stage_counts:
        final_thr = stage_counts[-1][0]
    if order_regions is not None and (
        order_regions.get("continuum_regions") or order_regions.get("line_regions")
    ):
        mask_note = "manual regions JSON"
        if fit_preview.fit_model == FIT_MODEL_SINC_POLY:
            mask_note += " (sinc²×poly)"
    else:
        mask_note = "shared blaze"

    w_norm, f_norm, _e_norm = _normalize_order_sinc_blaze(
        wave_air,
        flux_raw,
        eflux_raw,
        blaze_model,
        mask_bundle=mask_bundle,
        order_regions=order_regions,
    )
    plot_win = (w_norm >= wave_min) & (w_norm <= wave_max)
    w_norm = w_norm[plot_win]
    f_norm = f_norm[plot_win]

    ax_ratio = None
    if blaze_only:
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        ax_raw, ax_norm = axes
    else:
        if samples_path is None or not samples_path.is_file():
            raise FileNotFoundError(f"Samples not found: {samples_path}")

        airtovacuum = import_airtovacuum()
        wave_vac = airtovacuum(w_norm)

        ensure_stellar_sys_path()
        spec_nn = resolve_spec_nn_path()
        gm = import_payne_genmod()()
        gm._initspecnn(nnpath=spec_nn, NNtype="LinNet")
        pars = _model_pars_from_samples(
            samples_path,
            epoch_index,
            vrad_kms=vrad_kms,
        )
        _, model_flux = gm.genspec(pars, outwave=wave_vac, modpoly=False)

        ratio = np.divide(
            f_norm,
            np.maximum(model_flux, 1e-99),
            out=np.full_like(f_norm, np.nan),
            where=np.isfinite(model_flux),
        )

        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
        ax_raw, ax_norm, ax_ratio = axes

    if mask_bundle is not None and order_regions is not None:
        clipped_bundle = ManualMaskBundle(
            base_mask=mask_bundle.base_mask[in_win],
            fixed_line_mask=mask_bundle.fixed_line_mask[in_win],
            fixed_cont_mask=mask_bundle.fixed_cont_mask[in_win],
            cr_mask=mask_bundle.cr_mask[in_win],
        )
        _shade_manual_regions(ax_raw, w_air, clipped_bundle, order_regions)
    _shade_mask_gaps(ax_raw, w_air, cont_on_epoch, color="C3", alpha=0.12)
    ax_raw.plot(w_air, f_raw, color="k", lw=0.8, label="raw flux", zorder=3)
    ax_raw.scatter(
        w_air[cont_on_epoch],
        f_raw[cont_on_epoch],
        s=8,
        c="C2",
        alpha=0.55,
        label="continuum fit pixels",
        zorder=4,
    )
    ax_raw.plot(
        w_air,
        blaze_overlay,
        color="C1",
        lw=1.2,
        ls="--",
        label=blaze_label,
        zorder=3,
    )
    ax_raw.set_ylabel("Counts (raw)")
    n_cont = int(np.sum(cont_on_epoch))
    ax_raw.set_title(
        f"Gaia {gaia_id} epoch {epoch_index + 1} order {echelle_order} — "
        f"raw + blaze ({n_cont}/{w_air.size} continuum px @ thr={final_thr:.2f}, {mask_note})"
    )
    ax_raw.legend(loc="upper right", fontsize=8)
    if show_mask_stages and stage_counts:
        stage_txt = "mask stages: " + ", ".join(f"{t:.2f}:{n}" for t, n in stage_counts)
        ax_raw.text(
            0.02,
            0.02,
            stage_txt,
            transform=ax_raw.transAxes,
            fontsize=7,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    ax_norm.step(w_norm, f_norm, color="k", lw=0.8, where="mid", label="blaze-normalized")
    ax_norm.set_ylabel("Normalized flux")
    ax_norm.set_title("Per-order sinc_blaze_only (median norm)")
    ax_norm.set_ylim(0.8, 1.1)
    ax_norm.axhline(1.0, color="k", ls="--", lw=0.5)
    ax_norm.legend(loc="upper right", fontsize=8)

    if ax_ratio is not None:
        ax_ratio.plot(wave_vac, ratio, color="C2", lw=0.6)
        ax_ratio.axhline(1.0, color="k", ls="--", lw=0.5)
        ax_ratio.set_xlabel(r"Wavelength [$\AA$] (vacuum on norm/ratio panels)")
        ax_ratio.set_ylabel("data / model")
        ax_ratio.set_title(
            f"Payne spec model (no modpoly); Teff={pars[0]:.0f} K vrad={pars[4]:.2f} km/s"
        )
    else:
        ax_norm.set_xlabel(r"Wavelength [$\AA$] (air)")

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logging.info("Wrote %s", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gaia_id", help="Gaia DR3 source_id")
    parser.add_argument("--from-spec-root", action="store_true")
    parser.add_argument("--spec-path", type=Path, default=None, help="Explicit epoch .txt path")
    parser.add_argument("--epoch", type=int, default=0, help="0-based epoch if using --from-spec-root")
    parser.add_argument("--order", type=int, default=None, help="Echelle order number (default: best in window)")
    parser.add_argument(
        "--all-orders",
        action="store_true",
        help="Plot every echelle order with a blaze model (uses full order wavelength span)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for --all-orders PDFs (default: output/plots/order_blaze_<gaia_id>)",
    )
    parser.add_argument("--wave-min", type=float, default=5150.0)
    parser.add_argument("--wave-max", type=float, default=5300.0)
    parser.add_argument("--photometry-dir", "-D", type=Path, default=None)
    parser.add_argument(
        "--samples",
        type=Path,
        default=None,
        help="UMS samples FITS for stellar params + vrad (default: output/samples/..._ums.fits)",
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help=(
            "Replace shared OrderBlazeModel with multi-epoch fit_order_blaze_from_profiles; "
            "per-epoch amplitude scale is still computed on the displayed epoch"
        ),
    )
    parser.add_argument(
        "--show-mask-stages",
        action="store_true",
        help="Annotate iterative mask pixel counts per threshold on panel 1",
    )
    parser.add_argument(
        "--regions-json",
        type=Path,
        default=None,
        help="Manual continuum/line regions JSON from pick_spectrum_regions.py (default: auto per-star)",
    )
    parser.add_argument(
        "--no-auto-regions",
        action="store_true",
        help="Do not auto-resolve per-star regions JSON from output/masks/",
    )
    parser.add_argument(
        "--blaze-only",
        action="store_true",
        help="Two-panel blaze diagnostic only (no UMS samples / Payne model panel)",
    )
    parser.add_argument("--save", type=Path, default=None, help="Optional PDF/PNG output path")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive window (use with --save)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    gaia_id = str(args.gaia_id).strip()
    if args.spec_path is not None:
        spec_path = Path(args.spec_path)
    elif args.from_spec_root:
        paths = spectrum.sort_spectrum_paths(
            discover_primary_epoch_files(spec_root(), gaia_id)
        )
        if not paths:
            print("No spectrum files found.", file=sys.stderr)
            return 1
        spec_path = paths[int(np.clip(args.epoch, 0, len(paths) - 1))]
    else:
        parser.error("Provide --spec-path or --from-spec-root")

    samples_path: Path | None = None
    if not args.blaze_only:
        samples_path = args.samples or samples_dir() / f"Gaia_DR3_{gaia_id}_ums.fits"
        if not samples_path.is_file():
            print(f"Samples not found: {samples_path}", file=sys.stderr)
            print("Use --blaze-only for raw+normalized panels without UMS samples.", file=sys.stderr)
            return 1

    blaze_cal = spectrum.load_blaze_calibration()
    _header, spectrum_data = io_utils.read_spectrum(str(spec_path))

    regions_doc: dict | None = None
    resolved_regions = resolve_regions_json_for_star(
        gaia_id,
        args.regions_json,
        auto_resolve=not args.no_auto_regions,
    )
    if resolved_regions is not None:
        regions_doc = load_regions_json(resolved_regions)
        logging.info("Loaded regions from %s", resolved_regions)

    vrad_kms = float("nan")
    try:
        _, phot_nn, _ = models.model_paths()
        fit_data = data.getdata(
            gaia_id,
            spectrum_paths=[spec_path],
            photometry_dir=args.photometry_dir or photometry_dir(),
            phot_nn=phot_nn,
            phot_outlier_sigma=3.0,
        )
        rv_epochs = fit_data.get("rv_epochs") or []
        if rv_epochs and args.epoch < len(rv_epochs):
            vrad_kms = float(rv_epochs[args.epoch].rv_kms)
    except Exception as exc:
        logging.warning("Could not load RV for epoch: %s", exc)

    if args.all_orders:
        if args.order is not None:
            parser.error("Use --all-orders or --order, not both")
        order_list = _orders_with_blaze(spectrum_data, blaze_cal)
        if not order_list:
            print("No orders with blaze models found.", file=sys.stderr)
            return 1
        out_dir = args.output_dir or (
            Path("output/plots") / f"order_blaze_{gaia_id}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        all_paths: list[Path] | None = None
        if args.from_spec_root:
            all_paths = spectrum.sort_spectrum_paths(
                discover_primary_epoch_files(spec_root(), gaia_id)
            )
        failures: list[str] = []
        for order_num in order_list:
            w = np.asarray(spectrum_data[order_num]["wavelength"], dtype=float)
            wave_min = float(np.min(w))
            wave_max = float(np.max(w))
            refit_profiles: list[tuple[np.ndarray, np.ndarray]] | None = None
            mask_profiles: list[tuple[np.ndarray, np.ndarray]] | None = None
            if all_paths is not None:
                mask_profiles = _collect_order_profiles(all_paths, order_num)
                if args.refit:
                    refit_profiles = mask_profiles
            save_path = out_dir / f"o{order_num:02d}.pdf"
            logging.info("Order %s -> %s", order_num, save_path)
            order_regions = (
                order_regions_from_doc(regions_doc, order_num) if regions_doc else None
            )
            try:
                plot_order_blaze_diagnostic(
                    gaia_id,
                    spec_path,
                    echelle_order=order_num,
                    wave_min=wave_min,
                    wave_max=wave_max,
                    samples_path=samples_path,
                    epoch_index=args.epoch,
                    vrad_kms=vrad_kms,
                    refit_profiles=refit_profiles,
                    mask_profiles=mask_profiles if regions_doc is None else None,
                    order_regions=order_regions,
                    show_mask_stages=args.show_mask_stages,
                    save_path=save_path,
                    show=False,
                    blaze_only=args.blaze_only,
                )
            except Exception as exc:
                logging.error("Order %s failed: %s", order_num, exc)
                failures.append(f"o{order_num}: {exc}")
        logging.info("Wrote %d plots to %s", len(order_list) - len(failures), out_dir)
        if failures:
            print("Failures:", file=sys.stderr)
            for line in failures:
                print(f"  {line}", file=sys.stderr)
            return 1
        return 0

    order_num = _pick_order(
        spectrum_data,
        args.wave_min,
        args.wave_max,
        blaze_cal,
        args.order,
    )
    logging.info("Using echelle order %s from %s", order_num, spec_path.name)

    if not args.no_show:
        _pick_interactive_backend()

    refit_profiles: list[tuple[np.ndarray, np.ndarray]] | None = None
    mask_profiles: list[tuple[np.ndarray, np.ndarray]] | None = None
    if args.from_spec_root:
        all_paths = spectrum.sort_spectrum_paths(
            discover_primary_epoch_files(spec_root(), gaia_id)
        )
        mask_profiles = _collect_order_profiles(all_paths, order_num)
        if args.refit:
            refit_profiles = mask_profiles
            logging.info(
                "Refit using %d epoch profiles on order %s",
                len(refit_profiles),
                order_num,
            )
        elif mask_profiles:
            logging.info(
                "Continuum mask from median of %d epochs on order %s",
                len(mask_profiles),
                order_num,
            )

    order_regions = (
        order_regions_from_doc(regions_doc, order_num) if regions_doc else None
    )

    plot_order_blaze_diagnostic(
        gaia_id,
        spec_path,
        echelle_order=order_num,
        wave_min=args.wave_min,
        wave_max=args.wave_max,
        samples_path=samples_path,
        epoch_index=args.epoch,
        vrad_kms=vrad_kms,
        refit_profiles=refit_profiles,
        mask_profiles=mask_profiles if regions_doc is None else None,
        order_regions=order_regions,
        show_mask_stages=args.show_mask_stages,
        save_path=args.save,
        show=not args.no_show,
        blaze_only=args.blaze_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
