"""
Path-2 photometry SED diagnostic plots.

Two publication-quality figures produced after any ``darkhunter-sed-phot``
run when ``--plot`` is requested:

1. **SED panel** — observed photometry with upper-limit hats, per-model
   best-fit synth-phot curves, and a residual sub-panel.
2. **Model-comparison bar chart** — ΔlnZ and BIC grouped bars for all
   models; IFMR variants shown by bar hatching.

Limits
------
Requires ``matplotlib >= 3.8``.  Always call ``matplotlib.use("Agg")`` (or
any non-interactive backend) before importing this module in test contexts.
No fit is re-run; all functions accept precomputed data arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from darkhunter_sed.phot_sed_io import FLAG_DETECTION, FLAG_UPPER_LIMIT, PhotRow

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Band effective wavelengths (Angstroms)
# ---------------------------------------------------------------------------
# Used to place synth-phot points and model curves along the wavelength axis.
# Values from filter pivot/effective wavelengths; Bergeron registry for common bands.
BAND_EFF_WAVE_AA: dict[str, float] = {
    # GALEX
    "GALEX_FUV": 1530.0,
    "GALEX_NUV": 2310.0,
    # Swift UVOT (approximate pivot wavelengths)
    "Swift_UVW2": 1928.0,
    "Swift_UVM2": 2246.0,
    "Swift_UVW1": 2600.0,
    "Swift_U":    3465.0,
    "Swift_B":    4392.0,
    "Swift_V":    5468.0,
    # SDSS
    "SDSS_u": 3550.0,
    "SDSS_g": 4770.0,
    "SDSS_r": 6230.0,
    "SDSS_i": 7630.0,
    "SDSS_z": 9130.0,
    # Johnson-Cousins
    "Johnson_U": 3650.0,
    "Johnson_B": 4380.0,
    "Johnson_V": 5450.0,
    "Johnson_R": 6400.0,
    "Johnson_I": 8060.0,
    # Pan-STARRS
    "PS_g": 4810.0,
    "PS_r": 6170.0,
    "PS_i": 7520.0,
    "PS_z": 8660.0,
    "PS_y": 9620.0,
    # DECam
    "DECam_u": 3560.0,
    "DECam_g": 4770.0,
    "DECam_r": 6415.0,
    "DECam_i": 7835.0,
    "DECam_z": 9260.0,
    "DECam_Y": 9884.0,
    # Gaia DR2
    "GaiaDR2_G":  6230.0,
    "GaiaDR2_BP": 5050.0,
    "GaiaDR2_RP": 7730.0,
    # Gaia DR3
    "GaiaDR3_G":  6230.0,
    "GaiaDR3_BP": 5050.0,
    "GaiaDR3_RP": 7730.0,
    # 2MASS
    "2MASS_J":  12200.0,
    "2MASS_H":  16300.0,
    "2MASS_Ks": 21900.0,
    # WISE
    "WISE_W1": 33500.0,
    "WISE_W2": 46000.0,
    "WISE_W3": 115608.0,
    "WISE_W4": 220883.0,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MODEL_COLORS: dict[str, str] = {
    "1-star":       "#1f77b4",
    "2-star sum":   "#ff7f0e",
    "star1":        "#ff7f0e",
    "star2":        "#2ca02c",
    "WD-DA-MIST":   "#9467bd",
    "WD-DA-PARSEC": "#8c564b",
    "WD-DB-MIST":   "#e377c2",
    "WD-DB-PARSEC": "#7f7f7f",
}
_MODEL_HATCHES: dict[str, str] = {
    "WD-DA-PARSEC": "//",
    "WD-DB-PARSEC": "//",
}
# Group labels for grouping bars in the comparison chart.
_MODEL_GROUP: dict[str, str] = {
    "1-star":       "1-star",
    "2-star sum":   "2-star",
    "WD-DA-MIST":   "WD-DA",
    "WD-DA-PARSEC": "WD-DA",
    "WD-DB-MIST":   "WD-DB",
    "WD-DB-PARSEC": "WD-DB",
}


def _model_color(label: str) -> str:
    """Return a stable color for a model label."""
    return _MODEL_COLORS.get(label, "#333333")


def _wave_um(band: str) -> float | None:
    """
    Effective wavelength in microns for *band*, or ``None`` if unknown.

    Parameters
    ----------
    band :
        Band name as used in ``PhotRow.band`` (e.g. ``"GaiaDR3_G"``).
    """
    aa = BAND_EFF_WAVE_AA.get(band)
    return None if aa is None else aa * 1e-4


def _sorted_bands_by_wave(
    bands: list[str],
) -> list[str]:
    """Return *bands* sorted by effective wavelength (unknown bands last)."""
    known = [(b, BAND_EFF_WAVE_AA[b]) for b in bands if b in BAND_EFF_WAVE_AA]
    unknown = [b for b in bands if b not in BAND_EFF_WAVE_AA]
    known.sort(key=lambda t: t[1])
    return [b for b, _ in known] + unknown


# ---------------------------------------------------------------------------
# plot_sed_panel
# ---------------------------------------------------------------------------

def plot_sed_panel(
    rows: list[PhotRow],
    model_preds: dict[str, dict[str, float]],
    *,
    gaia_id: str = "",
    ul_arrow_frac: float = 0.05,
    outpath: Path | None = None,
) -> "Figure":
    """
    SED panel: observed photometry + model synth-phot curves + residuals.

    Parameters
    ----------
    rows :
        Observed photometry rows (detections and upper limits).
    model_preds :
        ``{model_label: {band: pred_mag}}`` mapping.  Each value must cover
        at least the detection bands present in *rows*.  Labels such as
        ``"1-star"``, ``"star1"``, ``"star2"``, ``"2-star sum"``,
        ``"WD-DA-MIST"`` receive canonical colors; other labels use grey.
    gaia_id :
        Source id string; used in the figure title.
    ul_arrow_frac :
        Upper-limit arrow length as a fraction of the displayed magnitude
        range.  Default 0.05 (5 %).
    outpath :
        If given, save the figure to this path and return it.

    Returns
    -------
    matplotlib.figure.Figure

    Limits
    ------
    Only bands present in :data:`BAND_EFF_WAVE_AA` are plotted.  Bands with
    no known effective wavelength are silently skipped.
    Residuals are computed against the first entry in *model_preds*.
    If *model_preds* is empty the residual panel is left blank.
    """
    import matplotlib.pyplot as plt

    det_rows = [r for r in rows if r.flag == FLAG_DETECTION and _wave_um(r.band) is not None]
    ul_rows  = [r for r in rows if r.flag == FLAG_UPPER_LIMIT and _wave_um(r.band) is not None]

    fig, (ax_sed, ax_res) = plt.subplots(
        2, 1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.05)

    # --- SED panel ---
    # Determine y range from data first; then handle arrows.
    all_mags = [r.mag for r in det_rows] + [r.mag for r in ul_rows]
    if not all_mags:
        y_lo, y_hi = 15.0, 25.0
    else:
        span = max(all_mags) - min(all_mags)
        pad = max(span * 0.15, 1.0)
        y_lo = min(all_mags) - pad
        y_hi = max(all_mags) + pad
    arrow_len = ul_arrow_frac * (y_hi - y_lo)

    # Detections
    if det_rows:
        waves = np.array([_wave_um(r.band) for r in det_rows], dtype=np.float64)
        mags  = np.array([r.mag for r in det_rows], dtype=np.float64)
        errs  = np.array([r.err for r in det_rows], dtype=np.float64)
        ax_sed.errorbar(
            waves, mags, yerr=errs,
            fmt="o", color="k", ms=5, zorder=5,
            label="observed",
        )

    # Upper limits — downward arrow with hat (matplotlib annotate)
    for r in ul_rows:
        wv = _wave_um(r.band)
        ax_sed.annotate(
            "",
            xy=(wv, r.mag + arrow_len),
            xytext=(wv, r.mag),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
        )
        ax_sed.plot(wv, r.mag, "v", color="#555555", ms=6, zorder=4)

    # Model curves
    for label, preds in model_preds.items():
        model_bands = _sorted_bands_by_wave(
            [b for b in preds if _wave_um(b) is not None]
        )
        if not model_bands:
            continue
        xs = np.array([_wave_um(b) for b in model_bands], dtype=np.float64)
        ys = np.array([preds[b] for b in model_bands], dtype=np.float64)
        color = _model_color(label)
        ls = "--" if label in ("star1", "star2") else "-"
        ax_sed.plot(xs, ys, ls, color=color, lw=1.5, label=label, zorder=3)
        ax_sed.scatter(xs, ys, marker="s", color=color, s=20, zorder=4)

    ax_sed.set_xscale("log")
    ax_sed.invert_yaxis()
    ax_sed.set_ylabel("AB mag")
    ax_sed.set_ylim(y_hi + pad * 0.2, y_lo - pad * 0.2)
    handles, labels = ax_sed.get_legend_handles_labels()
    if handles:
        ax_sed.legend(handles, labels, fontsize=7, ncol=2)
    title = f"SED  {gaia_id}" if gaia_id else "SED"
    ax_sed.set_title(title, fontsize=9)

    # --- Residual panel ---
    # Use the first model_preds entry as the reference model.
    if model_preds and det_rows:
        ref_label, ref_preds = next(iter(model_preds.items()))
        res_rows = [r for r in det_rows if r.band in ref_preds]
        if res_rows:
            from darkhunter_sed.phot_sed_fit import _PHOT_ERR_FLOOR
            rw = np.array([_wave_um(r.band) for r in res_rows], dtype=np.float64)
            rm = np.array([r.mag for r in res_rows], dtype=np.float64)
            re = np.array([r.err for r in res_rows], dtype=np.float64)
            rp = np.array([ref_preds[r.band] for r in res_rows], dtype=np.float64)
            err_eff = np.sqrt(re**2 + _PHOT_ERR_FLOOR**2)
            pull = (rm - rp) / err_eff
            ax_res.axhline(0, color="#999999", lw=0.8, ls="--")
            ax_res.scatter(rw, pull, color=_model_color(ref_label), s=25, zorder=4)
            ax_res.axhline(2, color="#cccccc", lw=0.6, ls=":")
            ax_res.axhline(-2, color="#cccccc", lw=0.6, ls=":")
    ax_res.set_xlabel("Wavelength (μm)")
    ax_res.set_ylabel("Pull (σ)")
    ax_res.set_xlim(ax_sed.get_xlim())

    _format_wave_axis(ax_res)

    if outpath is not None:
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
    return fig


def _format_wave_axis(ax: "Axes") -> None:
    """Apply human-readable log-spaced tick labels in microns."""
    import matplotlib.ticker as ticker

    ticks_um = [0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0, 10.0, 25.0]
    ax.set_xticks(ticks_um)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


# ---------------------------------------------------------------------------
# plot_model_comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(
    model_scores: dict[str, dict[str, float | None]],
    *,
    gaia_id: str = "",
    outpath: Path | None = None,
) -> "Figure":
    """
    Grouped bar chart: ΔlnZ and BIC for each model variant.

    Parameters
    ----------
    model_scores :
        ``{model_label: {"logz": float, "bic": float | None}}``
        Keys should be from:
        ``"1-star"``, ``"2-star sum"``,
        ``"WD-DA-MIST"``, ``"WD-DA-PARSEC"``,
        ``"WD-DB-MIST"``, ``"WD-DB-PARSEC"``.
        ``"bic"`` may be ``None`` (WD models have no BIC).
    gaia_id :
        Source id string; used in the figure title.
    outpath :
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure

    Limits
    ------
    ΔlnZ = lnZ − lnZ_best; positive = more evidence relative to worst model.
    BIC panel is blank for models with ``bic=None``.
    Bar fill color follows :data:`_MODEL_COLORS`; PARSEC variants are hatched
    with ``//``.
    """
    import matplotlib.pyplot as plt

    labels = list(model_scores.keys())
    logz_vals = np.array([model_scores[l]["logz"] for l in labels], dtype=np.float64)
    bic_vals: list[float | None] = [model_scores[l].get("bic") for l in labels]

    best_logz = np.max(logz_vals[np.isfinite(logz_vals)]) if logz_vals.size else 0.0
    delta_logz = logz_vals - best_logz

    x = np.arange(len(labels))
    width = 0.6

    n_panels = 2
    fig, (ax_logz, ax_bic) = plt.subplots(1, n_panels, figsize=(max(6, len(labels) * 1.2), 4))

    for ax, vals, ylabel, title_str, flip in [
        (ax_logz, delta_logz, "ΔlnZ", "Bayesian evidence (ΔlnZ)", False),
        (ax_bic, None, "BIC", "BIC (lower = better)", True),
    ]:
        colors = [_model_color(l) for l in labels]
        hatches = [_MODEL_HATCHES.get(l, "") for l in labels]

        if ax is ax_logz:
            bar_vals = np.where(np.isfinite(delta_logz), delta_logz, 0.0)
        else:
            bar_vals = np.array(
                [v if v is not None else 0.0 for v in bic_vals], dtype=np.float64
            )
            valid_bic = np.array(
                [v is not None for v in bic_vals], dtype=bool
            )

        bars = ax.bar(x, bar_vals, width, color=colors, edgecolor="k", linewidth=0.5)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        if ax is ax_bic:
            for i, (bar, vb) in enumerate(zip(bars, bic_vals)):
                if vb is None:
                    bar.set_visible(False)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title_str, fontsize=8)
        ax.axhline(0, color="#999999", lw=0.8, ls="--")

    title = f"Model comparison  {gaia_id}" if gaia_id else "Model comparison"
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Convenience: save all diagnostics for a complete run
# ---------------------------------------------------------------------------

def save_all_diagnostics(
    rows: list[PhotRow],
    model_preds: dict[str, dict[str, float]],
    model_scores: dict[str, dict[str, float | None]],
    *,
    gaia_id: str = "",
    out_dir: Path | str,
) -> dict[str, Path]:
    """
    Save all diagnostic figures for one target.

    Parameters
    ----------
    rows :
        Observed photometry rows.
    model_preds :
        ``{label: {band: pred_mag}}`` passed to :func:`plot_sed_panel`.
    model_scores :
        ``{label: {"logz": float, "bic": float | None}}`` passed to
        :func:`plot_model_comparison`.
    gaia_id :
        Source id; used in titles and output filenames.
    out_dir :
        Directory for output files (created if absent).

    Returns
    -------
    dict[str, Path]
        Mapping ``{"sed": path, "comparison": path}`` for the saved figure
        files.

    Limits
    ------
    Figures are saved as PDF (lossless vector format); the filename stem
    encodes the target id.
    """
    import matplotlib
    matplotlib.use("Agg")  # ensure non-interactive backend for file output

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = gaia_id.replace(" ", "_") if gaia_id else "target"
    paths: dict[str, Path] = {}

    sed_path = out / f"{stem}_sed.pdf"
    plot_sed_panel(rows, model_preds, gaia_id=gaia_id, outpath=sed_path)
    paths["sed"] = sed_path

    cmp_path = out / f"{stem}_model_comparison.pdf"
    plot_model_comparison(model_scores, gaia_id=gaia_id, outpath=cmp_path)
    paths["comparison"] = cmp_path

    return paths
