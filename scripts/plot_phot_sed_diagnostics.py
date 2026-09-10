#!/usr/bin/env python3
"""
Plot diagnostics for phot_sed fit outputs.

Produces two pages per fit:
  1. Corner plot of posterior samples (EEP, M, FeH, Av, parallax, sigma_int)
  2. Observed vs. model SED (best-fit magnitudes vs. measured photometry)

Usage:
    /opt/local/bin/python3 scripts/plot_phot_sed_diagnostics.py \\
        output/phot_sed/Gaia_DR3_4219507576765009536_1star

    # All 1-star fits:
    /opt/local/bin/python3 scripts/plot_phot_sed_diagnostics.py \\
        output/phot_sed/Gaia_DR3_*_1star

The script accepts glob patterns or explicit stems (with or without _samples.npz suffix).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from darkhunter_sed.phot_sed_io import read_photometry_fits

# Band display order for the SED plot (blue → red)
_BAND_WAV = {  # approximate effective wavelength in micron for x-axis
    "GALEX_FUV": 0.154, "GALEX_NUV": 0.227,
    "SDSS_u": 0.354, "SDSS_g": 0.477, "SDSS_r": 0.623, "SDSS_i": 0.763, "SDSS_z": 0.913,
    "PS_g": 0.481, "PS_r": 0.617, "PS_i": 0.752, "PS_z": 0.866, "PS_y": 0.963,
    "DECam_u": 0.357,
    "GaiaDR3_BP": 0.532, "GaiaDR3_G": 0.673, "GaiaDR3_RP": 0.797,
    "2MASS_J": 1.235, "2MASS_H": 1.662, "2MASS_Ks": 2.159,
    "WISE_W1": 3.353, "WISE_W2": 4.603, "WISE_W3": 11.56, "WISE_W4": 22.09,
}
_BAND_LABEL = {
    "GALEX_FUV": "FUV", "GALEX_NUV": "NUV",
    "SDSS_u": "u", "SDSS_g": "g", "SDSS_r": "r", "SDSS_i": "i", "SDSS_z": "z",
    "PS_g": "PS_g", "PS_r": "PS_r", "PS_i": "PS_i", "PS_z": "PS_z", "PS_y": "PS_y",
    "DECam_u": "DECam_u",
    "GaiaDR3_BP": "G$_{BP}$", "GaiaDR3_G": "G", "GaiaDR3_RP": "G$_{RP}$",
    "2MASS_J": "J", "2MASS_H": "H", "2MASS_Ks": "Ks",
    "WISE_W1": "W1", "WISE_W2": "W2", "WISE_W3": "W3", "WISE_W4": "W4",
}
_BAND_ORDER = sorted(_BAND_WAV, key=_BAND_WAV.__getitem__)


_WD_PARAM_NAMES = ["teff_wd", "logg_wd", "Av", "parallax"]


def _is_wd(d: np.lib.npyio.NpzFile) -> bool:
    return "m_wd" in d.files


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
    if _is_wd(d):
        # WD fit: samples are (N, 4) = (teff, logg, Av, parallax);
        # derived M_WD and M_i are stored separately.
        raw = np.asarray(d["samples"])
        m_wd = np.asarray(d["m_wd"]).reshape(-1, 1)
        m_i  = np.asarray(d["m_i"]).reshape(-1, 1)
        weights = np.asarray(d["weights"])
        # M_i is NaN for samples where IFMR is out of range (M_WD too high).
        # Keep a full set (raw + M_WD) and a finite subset that includes M_i.
        full = np.hstack([raw, m_wd, m_i])
        mask = np.all(np.isfinite(full), axis=1)
        frac_finite = mask.sum() / max(len(mask), 1)
        if frac_finite < 0.5:
            # Too few finite M_i (IFMR out of range for most samples): drop M_i
            # from the corner to avoid degenerate 2D histograms.
            samples = np.hstack([raw, m_wd])
            weights = np.asarray(d["weights"])  # restore full weights
            param_names = _WD_PARAM_NAMES + ["M_WD"]
            if frac_finite > 0.0:
                print(f"  WD: only {frac_finite:.0%} of samples have finite M_i "
                      f"(IFMR in range); M_i excluded from corner.")
        else:
            samples = full[mask]
            weights = weights[mask]
            param_names = _WD_PARAM_NAMES + ["M_WD", "M_i"]
        return summary, samples, None, param_names, weights
    else:
        return summary, d["samples"], d["logl"], list(d["param_names"]), None


def _plot_corner(samples: np.ndarray, param_names: list[str],
                 summary: dict, pdf_path: Path,
                 weights: np.ndarray | None = None) -> None:
    labels = param_names[:]
    corner_kw: dict = dict(
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 9},
    )
    if weights is not None:
        corner_kw["weights"] = weights
    fig = corner.corner(samples, **corner_kw)
    # Title: WD uses logevidence; 1-star/2-star uses logz.
    gaia_id = summary.get("gaia_id", "")
    model = summary.get("model", summary.get("atm_type", ""))
    if "logevidence" in summary:
        logz = summary["logevidence"]
        lnz_str = f"ln Z = {logz:.2f}"
    else:
        logz = summary.get("logz", 0.0)
        logz_err = summary.get("logz_err", 0.0)
        lnz_str = f"ln Z = {logz:.2f} ± {logz_err:.2f}"
    ifmr = summary.get("ifmr", "")
    title_model = f"{model}/{ifmr}" if ifmr else model
    fig.suptitle(
        f"Gaia DR3 {gaia_id}  [{title_model}]  {lnz_str}",
        fontsize=10,
    )
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  corner → {pdf_path}")


def _plot_sed(summary: dict, phot_path: Path | None, pdf_path: Path) -> None:
    best_mags: dict[str, float] = summary.get("best_mags", {})
    if not best_mags:
        print("  SED: no best_mags in summary — skipping")
        return

    obs_rows = []
    if phot_path and phot_path.is_file():
        obs_rows = [
            r for r in read_photometry_fits(phot_path)
            if r.band in best_mags and r.mag is not None
        ]

    fig, ax = plt.subplots(figsize=(7, 4))

    # Model SED
    bands_sorted = sorted(best_mags, key=lambda b: _BAND_WAV.get(b, 99))
    wav_model = [_BAND_WAV.get(b, None) for b in bands_sorted]
    mag_model = [best_mags[b] for b in bands_sorted]
    wav_model_f = [w for w, m in zip(wav_model, mag_model) if w is not None]
    mag_model_f = [m for w, m in zip(wav_model, mag_model) if w is not None]
    ax.plot(wav_model_f, mag_model_f, "b-", lw=1, alpha=0.6, label="Model (best-fit)")
    ax.plot(wav_model_f, mag_model_f, "bD", ms=4)

    # Observed photometry
    if obs_rows:
        wavs = [_BAND_WAV.get(r.band, None) for r in obs_rows]
        mags = [float(r.mag) for r in obs_rows]
        errs = [float(r.err) if r.err is not None else 0.05 for r in obs_rows]
        # upper limits
        for r, w in zip(obs_rows, wavs):
            if w is None:
                continue
            if getattr(r, "flag", None) == 2:  # FLAG_UPPER_LIMIT
                ax.annotate("", xy=(w, float(r.mag)), xytext=(w, float(r.mag) - 0.5),
                            arrowprops=dict(arrowstyle="-|>", color="gray"))
            else:
                ax.errorbar(w, float(r.mag),
                            yerr=float(r.err) if r.err else 0.05,
                            fmt="ko", ms=5, capsize=3, zorder=3)

    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Wavelength (μm)")
    ax.set_ylabel("AB mag")
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
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  SED  → {pdf_path}")


def _resolve_stems(paths: list[str]) -> list[Path]:
    stems = []
    for p in paths:
        p = Path(p)
        # strip suffixes added by glob
        for suffix in ("_samples.npz", "_summary.json"):
            if p.name.endswith(suffix):
                p = p.parent / p.name[: -len(suffix)]
        stems.append(p)
    return stems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="+", metavar="STEM",
                    help="Fit stem(s): output/phot_sed/Gaia_DR3_<id>_<model>")
    ap.add_argument("--phot-dir", type=Path,
                    default=Path("output/photometry"),
                    help="Directory with <id>_phot.fits (default: output/photometry)")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Where to write PDFs (default: same dir as the samples)")
    args = ap.parse_args(argv)

    stems = _resolve_stems(args.stems)
    n_ok = 0
    for stem in stems:
        print(f"\n{stem.name}")
        try:
            summary, samples, logl, param_names, weights = _load(stem)
        except FileNotFoundError as e:
            print(f"  MISSING: {e}")
            continue

        # WD fits store gaia_id in the parent directory name, not the summary.
        gaia_id = summary.get("gaia_id", "")
        if not gaia_id:
            # Walk up to find a numeric directory name (the Gaia ID).
            for part in stem.parts[::-1]:
                if part.isdigit():
                    gaia_id = part
                    break

        outdir = args.outdir or stem.parent
        outdir.mkdir(parents=True, exist_ok=True)

        _plot_corner(samples, param_names, summary,
                     outdir / (stem.name + "_corner.pdf"),
                     weights=weights)

        phot_path = args.phot_dir / f"{gaia_id}_phot.fits"
        _plot_sed(summary, phot_path,
                  outdir / (stem.name + "_sed.pdf"))
        n_ok += 1

    print(f"\n{n_ok}/{len(stems)} fits plotted.")
    return 0 if n_ok == len(stems) else 1


if __name__ == "__main__":
    sys.exit(main())
