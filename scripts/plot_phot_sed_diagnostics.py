#!/usr/bin/env python3
"""
Plot diagnostics for phot_sed fit outputs.

Produces per fit:
  1. Corner plot of posterior samples
  2. Observed vs. model SED with residual panel

Supported stem types:
  output/phot_sed/Gaia_DR3_<id>_1star      1-star Phoenix fit
  output/phot_sed/Gaia_DR3_<id>_2star      2-star Phoenix fit
  output/phot_sed/<id>/wd/wd_<ATM>_<IFMR>    WD-only Bergeron fit
  output/phot_sed/<id>/wd/wdstar_<ATM>_<IFMR> WD+companion fit

Usage:
    /opt/local/bin/python3 scripts/plot_phot_sed_diagnostics.py \\
        output/phot_sed/Gaia_DR3_4219507576765009536_1star

    # All 1-star fits:
    /opt/local/bin/python3 scripts/plot_phot_sed_diagnostics.py \\
        output/phot_sed/Gaia_DR3_*_1star

    # WD fits in a gaia-id subdirectory:
    /opt/local/bin/python3 scripts/plot_phot_sed_diagnostics.py \\
        --wd-dir ~/stellar/wd \\
        output/phot_sed/6475655404885617920/wd/wd_DA_MIST \\
        output/phot_sed/6475655404885617920/wd/wdstar_DA_MIST

The script accepts glob patterns or explicit stems (with or without
_samples.npz / _summary.json suffix).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from darkhunter_sed.phot_sed_io import read_photometry_fits, FLAG_UPPER_LIMIT

# Approximate effective wavelength in microns (blue → red)
_BAND_WAV = {
    "GALEX_FUV": 0.154, "GALEX_NUV": 0.227,
    "SDSS_u": 0.354, "SDSS_g": 0.477, "SDSS_r": 0.623, "SDSS_i": 0.763, "SDSS_z": 0.913,
    "PS_g": 0.481, "PS_r": 0.617, "PS_i": 0.752, "PS_z": 0.866, "PS_y": 0.963,
    "DECam_u": 0.357,
    "GaiaDR3_BP": 0.532, "GaiaDR3_G": 0.673, "GaiaDR3_RP": 0.797,
    "2MASS_J": 1.235, "2MASS_H": 1.662, "2MASS_Ks": 2.159,
    "WISE_W1": 3.353, "WISE_W2": 4.603, "WISE_W3": 11.56, "WISE_W4": 22.09,
}
_BAND_ORDER = sorted(_BAND_WAV, key=_BAND_WAV.__getitem__)

_PHOT_ERR_FLOOR = 0.02  # added in quadrature for residual computation

_WD_PARAM_NAMES = ["teff_wd", "logg_wd", "Av", "parallax"]


# ---------------------------------------------------------------------------
# NPZ type detection
# ---------------------------------------------------------------------------

def _is_wdonly(d: np.lib.npyio.NpzFile) -> bool:
    """WD-only Bergeron fit: has m_wd but no stored param_names."""
    return "m_wd" in d.files and "param_names" not in d.files


def _is_wdstar(d: np.lib.npyio.NpzFile) -> bool:
    """WD+companion fit: has m_wd AND stored param_names (8-param)."""
    return "m_wd" in d.files and "param_names" in d.files


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load(stem: Path) -> tuple[dict, np.ndarray, np.ndarray | None, list[str], np.ndarray | None]:
    """Return (summary, samples, logl_or_None, param_names, weights_or_None)."""
    npz_path = stem.parent / (stem.name + "_samples.npz")
    json_path = stem.parent / (stem.name + "_summary.json")
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not json_path.is_file():
        raise FileNotFoundError(json_path)
    d = np.load(npz_path)
    summary = json.loads(json_path.read_text())

    if _is_wdstar(d):
        return _load_wdstar(d, summary)
    if _is_wdonly(d):
        return _load_wdonly(d, summary)
    # 1-star / 2-star Phoenix fit
    return summary, d["samples"], d["logl"], list(d["param_names"]), None


def _load_wdonly(d: np.lib.npyio.NpzFile, summary: dict):
    """Load WD-only Bergeron posterior: 4 params + M_WD + M_i."""
    raw = np.asarray(d["samples"])          # (N, 4)
    m_wd = np.asarray(d["m_wd"]).reshape(-1, 1)
    m_i  = np.asarray(d["m_i"]).reshape(-1, 1)
    weights = np.asarray(d["weights"])
    full = np.hstack([raw, m_wd, m_i])
    mask = np.all(np.isfinite(full), axis=1)
    frac_finite = mask.sum() / max(len(mask), 1)
    if frac_finite < 0.5:
        samples = np.hstack([raw, m_wd])
        weights = np.asarray(d["weights"])
        param_names = _WD_PARAM_NAMES + ["M_WD"]
        if frac_finite > 0.0:
            print(f"  WD: only {frac_finite:.0%} of samples have finite M_i "
                  f"(IFMR in range); M_i excluded from corner.")
    else:
        samples = full[mask]
        weights = weights[mask]
        param_names = _WD_PARAM_NAMES + ["M_WD", "M_i"]
    return summary, samples, None, param_names, weights


def _load_wdstar(d: np.lib.npyio.NpzFile, summary: dict):
    """Load WD+companion posterior: 8 physical params + M_WD + M_i."""
    raw = np.asarray(d["samples"])          # (N, 8)
    m_wd = np.asarray(d["m_wd"]).reshape(-1, 1)
    m_i  = np.asarray(d["m_i"]).reshape(-1, 1)
    weights = np.asarray(d["weights"])
    param_names_base = list(d["param_names"])
    full = np.hstack([raw, m_wd, m_i])
    mask = np.all(np.isfinite(full), axis=1)
    frac_finite = mask.sum() / max(len(mask), 1)
    if frac_finite < 0.5:
        samples = np.hstack([raw, m_wd])
        weights = np.asarray(d["weights"])
        param_names = param_names_base + ["M_WD"]
        if frac_finite > 0.0:
            print(f"  WD+star: only {frac_finite:.0%} of samples have finite M_i; "
                  "M_i excluded from corner.")
    else:
        samples = full[mask]
        weights = weights[mask]
        param_names = param_names_base + ["M_WD", "M_i"]
    return summary, samples, None, param_names, weights


# ---------------------------------------------------------------------------
# Helper: photometry observation lookup
# ---------------------------------------------------------------------------

def _get_obs(phot_path: Path | None, bands: set[str]):
    """Return (detections, upper_limits) as lists of PhotRow for the given bands."""
    if not phot_path or not phot_path.is_file():
        return [], []
    rows = [r for r in read_photometry_fits(phot_path) if r.band in bands and r.mag is not None]
    dets = [r for r in rows if r.flag != FLAG_UPPER_LIMIT]
    uls  = [r for r in rows if r.flag == FLAG_UPPER_LIMIT]
    return dets, uls


# ---------------------------------------------------------------------------
# Corner plot
# ---------------------------------------------------------------------------

def _plot_corner(samples: np.ndarray, param_names: list[str],
                 summary: dict, pdf_path: Path,
                 weights: np.ndarray | None = None) -> None:
    corner_kw: dict = dict(
        labels=param_names[:],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 9},
    )
    if weights is not None:
        corner_kw["weights"] = weights
    fig = corner.corner(samples, **corner_kw)

    gaia_id = summary.get("gaia_id", "")
    model = summary.get("model", summary.get("atm_type", ""))
    ifmr = summary.get("ifmr", "")
    title_model = f"{model}/{ifmr}" if ifmr else model
    if "logevidence" in summary:
        lnz_str = f"ln Z = {summary['logevidence']:.2f}"
    else:
        logz = summary.get("logz", 0.0)
        logz_err = summary.get("logz_err", 0.0)
        lnz_str = f"ln Z = {logz:.2f} ± {logz_err:.2f}"
    fig.suptitle(f"Gaia DR3 {gaia_id}  [{title_model}]  {lnz_str}", fontsize=10)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  corner → {pdf_path}")


# ---------------------------------------------------------------------------
# SED plot helpers
# ---------------------------------------------------------------------------

def _add_ul_markers(ax, obs_uls, wav_for_band):
    """Draw downward-triangle upper-limit markers with short downward line."""
    for r in obs_uls:
        w = wav_for_band(r.band)
        if w is None:
            continue
        lim_mag = float(r.mag)
        # Triangle marker at the limit
        ax.plot(w, lim_mag, "v", color="gray", ms=7, zorder=3, mec="gray", mfc="none")
        # Short line extending toward fainter (larger mag = lower on inverted y)
        ax.annotate("", xy=(w, lim_mag + 0.4), xytext=(w, lim_mag),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=1))


def _plot_sed_panels(ax_sed, ax_res, best_mags: dict[str, float],
                     dets, uls, title: str) -> None:
    """Fill a prepared SED axes + optional residual axes."""
    wav_for_band = lambda b: _BAND_WAV.get(b)

    # --- model curve ---
    bands_sorted = sorted(best_mags, key=lambda b: _BAND_WAV.get(b, 99))
    wav_m = [_BAND_WAV.get(b) for b in bands_sorted]
    mag_m = [best_mags[b] for b in bands_sorted]
    wav_m_f = [w for w, m in zip(wav_m, mag_m) if w is not None]
    mag_m_f = [m for w, m in zip(wav_m, mag_m) if w is not None]
    ax_sed.plot(wav_m_f, mag_m_f, "b-", lw=1, alpha=0.6, label="Model (best-fit)")
    ax_sed.plot(wav_m_f, mag_m_f, "bD", ms=4)

    # --- detections ---
    for r in dets:
        w = wav_for_band(r.band)
        if w is None:
            continue
        ax_sed.errorbar(w, float(r.mag),
                        yerr=float(r.err) if r.err else _PHOT_ERR_FLOOR,
                        fmt="ko", ms=5, capsize=3, zorder=3)

    # --- upper limits ---
    _add_ul_markers(ax_sed, uls, wav_for_band)

    ax_sed.set_xscale("log")
    ax_sed.invert_yaxis()
    ax_sed.set_ylabel("AB mag")
    ax_sed.set_title(title, fontsize=10)
    handles, labels = ax_sed.get_legend_handles_labels()
    if handles:
        ax_sed.legend(fontsize=8)
    ax_sed.grid(True, alpha=0.3)

    # --- residuals ---
    if ax_res is None:
        return
    pulls = []
    wav_det = []
    for r in dets:
        w = wav_for_band(r.band)
        if w is None or r.band not in best_mags:
            continue
        err = float(r.err) if r.err else _PHOT_ERR_FLOOR
        sigma_eff = np.hypot(err, _PHOT_ERR_FLOOR)
        pull = (float(r.mag) - best_mags[r.band]) / sigma_eff
        wav_det.append(w)
        pulls.append(pull)

    if pulls:
        ax_res.scatter(wav_det, pulls, color="k", s=18, zorder=3)
        ax_res.axhline(0, color="b", lw=1, alpha=0.6)
        for lv in (-3, 3):
            ax_res.axhline(lv, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax_res.set_ylabel("Pull (σ)")
        ax_res.set_ylim(-5, 5)
    else:
        ax_res.text(0.5, 0.5, "no detections", transform=ax_res.transAxes,
                    ha="center", va="center", color="gray", fontsize=9)
    ax_res.set_xscale("log")
    ax_res.set_xlabel("Wavelength (μm)")
    ax_res.grid(True, alpha=0.3)


def _make_sed_figure(has_residuals: bool):
    """Return (fig, ax_sed, ax_res_or_None)."""
    if has_residuals:
        fig = plt.figure(figsize=(7, 5), layout="constrained")
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05, figure=fig)
        ax_sed = fig.add_subplot(gs[0])
        ax_res = fig.add_subplot(gs[1], sharex=ax_sed)
        plt.setp(ax_sed.get_xticklabels(), visible=False)
        return fig, ax_sed, ax_res
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        return fig, ax, None


# ---------------------------------------------------------------------------
# 1-star Phoenix SED
# ---------------------------------------------------------------------------

def _plot_sed(summary: dict, phot_path: Path | None, pdf_path: Path) -> None:
    best_mags: dict[str, float] = summary.get("best_mags", {})
    if not best_mags:
        print("  SED: no best_mags in summary — skipping")
        return

    dets, uls = _get_obs(phot_path, set(best_mags))
    has_res = bool(dets)
    fig, ax_sed, ax_res = _make_sed_figure(has_res)

    gaia_id = summary.get("gaia_id", "")
    model = summary.get("model", "")
    best = summary.get("best_mist", {})
    teff = best.get("teff_k")
    dist = summary.get("distance_pc")
    title = f"Gaia DR3 {gaia_id}  [{model}]"
    if teff:
        title += f"   Teff={teff:.0f} K"
    if dist:
        title += f"   d={dist:.0f} pc"

    _plot_sed_panels(ax_sed, ax_res, best_mags, dets, uls, title)

    if ax_res is None:
        fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  SED  → {pdf_path}")


# ---------------------------------------------------------------------------
# 2-star Phoenix SED (combined + per-component)
# ---------------------------------------------------------------------------

_LOGLIKE_FLOOR = -1e5  # dynesty converges here for failed fits
_LOGZ_FAIL_THRESHOLD = -1000.0  # logz below this means fit collapsed to floor


def _load_phoenix_stack(mist_nn: Path | None, phoenix_dir: Path | None):
    """Return (predictor, grid_phx) or raise ImportError / FileNotFoundError."""
    from darkhunter_sed.misty_iso import load_misty_predictor, resolve_mist_nn_path
    from darkhunter_sed.phoenix_grid import PhoenixGrid
    predictor = load_misty_predictor(resolve_mist_nn_path(mist_nn))
    grid_phx = PhoenixGrid(root=phoenix_dir)
    return predictor, grid_phx


def _plot_sed_2star(summary: dict, phot_path: Path | None, pdf_path: Path,
                    mist_nn: Path | None = None,
                    phoenix_dir: Path | None = None) -> None:
    logz = summary.get("logz", _LOGLIKE_FLOOR)
    if logz < _LOGZ_FAIL_THRESHOLD:
        print(f"  2-star SED: fit failed (logz={logz:.0f}) — skipping")
        return

    # Best-theta order matches param_names stored in summary
    param_names = summary.get("param_names", [])
    best_theta_dict = summary.get("best_theta", {})
    if not best_theta_dict or len(param_names) != 7:
        print("  2-star SED: missing best_theta or param_names — skipping")
        return

    best_theta = np.array([best_theta_dict[n] for n in param_names])

    # We need phot bands from the observations file
    if not phot_path or not phot_path.is_file():
        print(f"  2-star SED: phot file not found ({phot_path}) — skipping")
        return

    try:
        from darkhunter_sed.phot_sed_fit import _GAIA_CONSTRAINT_BANDS, load_bandpasses_for_bands
        from darkhunter_sed.filters_synphot import BAND_REGISTRY
        from darkhunter_sed.phot_sed_models import (
            TwoStarParams, OneStarParams,
            predict_2star_phot, predict_1star_phot, solve_eep2_for_age_match,
        )
        from darkhunter_sed.misty_iso import evaluate_mist
    except ImportError as exc:
        print(f"  2-star SED: import error ({exc}) — skipping")
        return

    all_rows = read_photometry_fits(phot_path)
    phot_bands = [
        r.band for r in all_rows
        if r.band not in _GAIA_CONSTRAINT_BANDS and r.band in BAND_REGISTRY
    ]
    if not phot_bands:
        print("  2-star SED: no registered photometry bands found — skipping")
        return

    try:
        predictor, grid_phx = _load_phoenix_stack(mist_nn, phoenix_dir)
        bandpasses = load_bandpasses_for_bands(phot_bands)
    except Exception as exc:
        print(f"  2-star SED: could not load MIST/PHOENIX ({exc}) — skipping")
        return

    try:
        pred_combined = predict_2star_phot(
            best_theta, phot_bands,
            mist_predictor=predictor, phoenix_grid=grid_phx, bandpasses=bandpasses,
        )
        mags_combined = pred_combined.mags

        p = TwoStarParams.from_array(best_theta)
        mist1 = evaluate_mist(p.eep1, p.mass1, p.feh, p.afe, predictor=predictor)
        eep2 = solve_eep2_for_age_match(
            mist1.age_gyr, p.mass2, p.feh, p.afe, predictor=predictor
        )

        mags_s1 = mags_s2 = None
        if eep2 is not None:
            s1 = OneStarParams(eep=p.eep1, mass=p.mass1, feh=p.feh, afe=p.afe,
                               a_v=p.a_v, parallax_mas=p.parallax_mas)
            s2 = OneStarParams(eep=eep2,   mass=p.mass2, feh=p.feh, afe=p.afe,
                               a_v=p.a_v, parallax_mas=p.parallax_mas)
            mags_s1 = predict_1star_phot(
                s1, phot_bands,
                mist_predictor=predictor, phoenix_grid=grid_phx, bandpasses=bandpasses,
            ).mags
            mags_s2 = predict_1star_phot(
                s2, phot_bands,
                mist_predictor=predictor, phoenix_grid=grid_phx, bandpasses=bandpasses,
            ).mags
    except Exception as exc:
        print(f"  2-star SED: model evaluation failed ({exc}) — skipping")
        return

    # --- figure ---
    all_model_bands = set(mags_combined)
    dets, uls = _get_obs(phot_path, all_model_bands)
    has_res = bool(dets)
    fig, ax_sed, ax_res = _make_sed_figure(has_res)

    wav_for_band = lambda b: _BAND_WAV.get(b)

    def _plot_curve(mags, color, ls, lw, label, ms=4, marker="D"):
        bands_s = sorted(mags, key=lambda b: _BAND_WAV.get(b, 99))
        wx = [_BAND_WAV[b] for b in bands_s if b in _BAND_WAV]
        my = [mags[b] for b in bands_s if b in _BAND_WAV]
        ax_sed.plot(wx, my, color=color, ls=ls, lw=lw, alpha=0.7, label=label)
        ax_sed.plot(wx, my, color=color, marker=marker, ms=ms, ls="none")

    # Individual components (dashed)
    if mags_s1 is not None:
        _plot_curve(mags_s1, color="C1", ls="--", lw=1, label=f"Star 1 (Teff={pred_combined.mist1.teff_k:.0f} K)")
    if mags_s2 is not None:
        _plot_curve(mags_s2, color="C2", ls="--", lw=1, label=f"Star 2 (Teff={pred_combined.mist2.teff_k:.0f} K)")
    # Combined (solid, on top)
    _plot_curve(mags_combined, color="C0", ls="-", lw=1.5, label="Combined (best-fit)", ms=5, marker="D")

    # Detections and ULs
    for r in dets:
        w = wav_for_band(r.band)
        if w is None:
            continue
        ax_sed.errorbar(w, float(r.mag),
                        yerr=float(r.err) if r.err else _PHOT_ERR_FLOOR,
                        fmt="ko", ms=5, capsize=3, zorder=3)
    _add_ul_markers(ax_sed, uls, wav_for_band)

    ax_sed.set_xscale("log")
    ax_sed.invert_yaxis()
    ax_sed.set_ylabel("AB mag")
    gaia_id = summary.get("gaia_id", "")
    dist_pc = pred_combined.distance_pc
    title = (f"Gaia DR3 {gaia_id}  [2-star]  "
             f"Teff₁={pred_combined.mist1.teff_k:.0f} K  "
             f"Teff₂={pred_combined.mist2.teff_k:.0f} K  "
             f"d={dist_pc:.0f} pc")
    ax_sed.set_title(title, fontsize=9)
    handles, labels = ax_sed.get_legend_handles_labels()
    if handles:
        ax_sed.legend(fontsize=8)
    ax_sed.grid(True, alpha=0.3)

    # Residuals
    if ax_res is not None:
        pulls, wav_det = [], []
        for r in dets:
            w = wav_for_band(r.band)
            if w is None or r.band not in mags_combined:
                continue
            err = float(r.err) if r.err else _PHOT_ERR_FLOOR
            pull = (float(r.mag) - mags_combined[r.band]) / np.hypot(err, _PHOT_ERR_FLOOR)
            wav_det.append(w)
            pulls.append(pull)
        if pulls:
            ax_res.scatter(wav_det, pulls, color="k", s=18, zorder=3)
            ax_res.axhline(0, color="C0", lw=1, alpha=0.6)
            for lv in (-3, 3):
                ax_res.axhline(lv, color="gray", lw=0.8, ls="--", alpha=0.6)
            ax_res.set_ylabel("Pull (σ)")
            ax_res.set_ylim(-5, 5)
        ax_res.set_xscale("log")
        ax_res.set_xlabel("Wavelength (μm)")
        ax_res.grid(True, alpha=0.3)

    if ax_res is None:
        fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  SED  → {pdf_path}")


# ---------------------------------------------------------------------------
# WD-only Bergeron SED
# ---------------------------------------------------------------------------

def _plot_sed_wd(summary: dict, phot_path: Path | None, pdf_path: Path,
                 wd_dir: Path | None) -> None:
    if wd_dir is None:
        print("  WD SED: --wd-dir not set — skipping")
        return

    atm_type = summary.get("atm_type", "DA")
    teff   = summary.get("teff_median")
    logg   = summary.get("logg_median")
    av     = summary.get("av_median")
    plx    = summary.get("parallax_median")
    if any(v is None for v in (teff, logg, av, plx)):
        print("  WD SED: missing median parameters in summary — skipping")
        return
    dist_pc = 1000.0 / plx

    try:
        from darkhunter_sed.bergeron_wd import BergeronGrid
        grid = BergeronGrid.from_dir(wd_dir, atm_type)
        result = grid.synth_phot(teff, logg, av, dist_pc)
    except Exception as exc:
        print(f"  WD SED: BergeronGrid error ({exc}) — skipping")
        return

    best_mags: dict[str, float] = result["mags"]
    dets, uls = _get_obs(phot_path, set(best_mags))
    has_res = bool(dets)
    fig, ax_sed, ax_res = _make_sed_figure(has_res)

    gaia_id = summary.get("gaia_id", "")
    ifmr = summary.get("ifmr", "")
    title = f"Gaia DR3 {gaia_id}  [WD-{atm_type}/{ifmr}]  Teff={teff:.0f} K  d={dist_pc:.0f} pc"
    _plot_sed_panels(ax_sed, ax_res, best_mags, dets, uls, title)

    if ax_res is None:
        fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  SED  → {pdf_path}")


# ---------------------------------------------------------------------------
# WD+companion SED (WD component only via Bergeron; star needs PHOENIX)
# ---------------------------------------------------------------------------

def _plot_sed_wdstar(summary: dict, phot_path: Path | None, pdf_path: Path,
                     wd_dir: Path | None) -> None:
    if wd_dir is None:
        print("  WD+star SED: --wd-dir not set — skipping")
        return

    atm_type = summary.get("atm_type", "DA")
    teff_wd = summary.get("Teff_WD_median")
    logg_wd = summary.get("logg_WD_median")
    av      = summary.get("Av_median")
    plx     = summary.get("parallax_median")
    if any(v is None for v in (teff_wd, logg_wd, av, plx)):
        print("  WD+star SED: missing median parameters in summary — skipping")
        return
    dist_pc = 1000.0 / plx

    try:
        from darkhunter_sed.bergeron_wd import BergeronGrid
        grid = BergeronGrid.from_dir(wd_dir, atm_type)
        result = grid.synth_phot(teff_wd, logg_wd, av, dist_pc)
    except Exception as exc:
        print(f"  WD+star SED: BergeronGrid error ({exc}) — skipping")
        return

    best_mags: dict[str, float] = result["mags"]
    dets, uls = _get_obs(phot_path, set(best_mags))
    has_res = bool(dets)
    fig, ax_sed, ax_res = _make_sed_figure(has_res)

    gaia_id = summary.get("gaia_id", "")
    ifmr = summary.get("ifmr", "")
    m_wd = summary.get("m_wd_median", float("nan"))
    title = (f"Gaia DR3 {gaia_id}  [WD+star-{atm_type}/{ifmr}]  "
             f"Teff_WD={teff_wd:.0f} K  M_WD={m_wd:.2f} M☉\n"
             f"(WD component only; companion star requires PHOENIX model)")
    _plot_sed_panels(ax_sed, ax_res, best_mags, dets, uls, title)

    if ax_res is None:
        fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  SED  → {pdf_path}")


# ---------------------------------------------------------------------------
# Stem / path resolution
# ---------------------------------------------------------------------------

def _resolve_stems(paths: list[str]) -> list[Path]:
    stems = []
    for p in paths:
        p = Path(p)
        for suffix in ("_samples.npz", "_summary.json"):
            if p.name.endswith(suffix):
                p = p.parent / p.name[: -len(suffix)]
        stems.append(p)
    return stems


def _get_gaia_id(summary: dict, stem: Path) -> str:
    gaia_id = summary.get("gaia_id", "")
    if not gaia_id:
        for part in stem.parts[::-1]:
            if part.isdigit():
                gaia_id = part
                break
    return gaia_id


def _model_type(summary: dict, d: np.lib.npyio.NpzFile) -> str:
    """Return one of '1star', '2star', 'wdonly', 'wdstar'."""
    if _is_wdstar(d):
        return "wdstar"
    if _is_wdonly(d):
        return "wdonly"
    return summary.get("model", "1star")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="+", metavar="STEM",
                    help="Fit stem(s): output/phot_sed/Gaia_DR3_<id>_<model> or "
                         "output/phot_sed/<id>/wd/wd_<ATM>_<IFMR>")
    ap.add_argument("--phot-dir", type=Path,
                    default=Path("output/photometry"),
                    help="Directory with <id>_phot.fits (default: output/photometry)")
    ap.add_argument("--wd-dir", type=Path, default=None,
                    help="Bergeron WD grid directory (required for WD/WD+star SED plots; "
                         "falls back to STELLAR_ROOT/wd or ~/stellar/wd)")
    ap.add_argument("--mist-nn", type=Path, default=None,
                    help="MistNN HDF5 path (required for 2-star SEDs; "
                         "falls back to STELLAR_ROOT resolution)")
    ap.add_argument("--phoenix-dir", type=Path, default=None,
                    help="PHOENIX HiResFITS root (required for 2-star SEDs)")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Where to write PDFs (default: same dir as the samples)")
    args = ap.parse_args(argv)

    # Resolve WD dir from env/home if not supplied
    wd_dir = args.wd_dir
    if wd_dir is None:
        import os
        for candidate in [
            os.environ.get("STELLAR_ROOT", None) and Path(os.environ["STELLAR_ROOT"]) / "wd",
            Path.home() / "stellar" / "wd",
        ]:
            if candidate and Path(candidate).is_dir():
                wd_dir = Path(candidate)
                break

    stems = _resolve_stems(args.stems)
    n_ok = 0
    for stem in stems:
        print(f"\n{stem.name}")
        npz_path = stem.parent / (stem.name + "_samples.npz")
        if not npz_path.is_file():
            print(f"  MISSING: {npz_path}")
            continue
        try:
            d_probe = np.load(npz_path)
            summary, samples, logl, param_names, weights = _load(stem)
        except FileNotFoundError as e:
            print(f"  MISSING: {e}")
            continue
        except Exception as e:
            print(f"  ERROR loading {stem.name}: {e}")
            continue

        gaia_id = _get_gaia_id(summary, stem)
        mtype = _model_type(summary, d_probe)
        outdir = args.outdir or stem.parent
        outdir.mkdir(parents=True, exist_ok=True)
        phot_path = args.phot_dir / f"{gaia_id}_phot.fits"

        # Corner plot
        _plot_corner(samples, param_names, summary,
                     outdir / (stem.name + "_corner.pdf"),
                     weights=weights)

        # SED plot (routed by model type)
        sed_pdf = outdir / (stem.name + "_sed.pdf")
        if mtype == "1star":
            _plot_sed(summary, phot_path, sed_pdf)
        elif mtype == "2star":
            _plot_sed_2star(summary, phot_path, sed_pdf,
                            mist_nn=args.mist_nn, phoenix_dir=args.phoenix_dir)
        elif mtype == "wdonly":
            _plot_sed_wd(summary, phot_path, sed_pdf, wd_dir)
        elif mtype == "wdstar":
            _plot_sed_wdstar(summary, phot_path, sed_pdf, wd_dir)

        n_ok += 1

    print(f"\n{n_ok}/{len(stems)} fits plotted.")
    return 0 if n_ok == len(stems) else 1


if __name__ == "__main__":
    sys.exit(main())
