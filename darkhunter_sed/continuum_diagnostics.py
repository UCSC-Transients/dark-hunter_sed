"""Wavelength-binned continuum residual metrics for UMS spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.table import Table

from darkhunter_sed.stellar_data import (
    ensure_stellar_sys_path,
    import_payne_genmod,
    resolve_spec_nn_path,
)


def _fixed_pc_pars(
    bfdict: dict[str, tuple[float, float]],
    spec_i: int,
    *,
    lsf: float = 60000.0,
) -> list[float]:
    return [
        bfdict["Teff"][0],
        bfdict["log(g)"][0],
        bfdict["[Fe/H]"][0],
        bfdict["[a/Fe]"][0],
        bfdict.get(f"vrad_{spec_i}", bfdict.get("vrad", (0.0,)))[0],
        bfdict["vstar"][0],
        bfdict["vmic"][0],
        lsf,
        1.0,
        0.0,
        0.0,
        0.0,
    ]


def continuum_fractional_residual(
    obs_wave: np.ndarray,
    obs_flux: np.ndarray,
    model_flux: np.ndarray,
    *,
    n_bins: int = 40,
    line_mask_sigma: float = 3.0,
) -> dict[str, Any]:
    """
    Binned (model - data) / data with optional masking of deep absorption pixels.

    ``continuum_rms`` uses pixels where local flux is within ``line_mask_sigma``
    MAD of a running median (broadband continuum regions).
    """
    wave = np.asarray(obs_wave, dtype=float)
    data = np.asarray(obs_flux, dtype=float)
    model = np.asarray(model_flux, dtype=float)
    ok = np.isfinite(wave) & np.isfinite(data) & np.isfinite(model) & (data > 0)
    wave, data, model = wave[ok], data[ok], model[ok]
    if wave.size < 10:
        nan = float("nan")
        return {
            "continuum_rms": nan,
            "full_rms": nan,
            "median_ratio": nan,
            "n_pixels": int(wave.size),
            "n_continuum_pixels": 0,
        }

    frac = (model - data) / data
    full_rms = float(np.sqrt(np.mean(frac**2)))

    order = np.argsort(wave)
    wave_s = wave[order]
    data_s = data[order]
    model_s = model[order]
    frac_s = frac[order]

    win = max(15, wave_s.size // 50)
    if win % 2 == 0:
        win += 1
    from scipy.ndimage import median_filter

    med = median_filter(data_s, size=win, mode="nearest")
    resid = data_s - med
    mad = float(np.median(np.abs(resid - np.median(resid))))
    sigma = max(1.4826 * mad, 1e-6)
    continuum_mask = np.abs(resid) <= line_mask_sigma * sigma

    if np.any(continuum_mask):
        continuum_rms = float(np.sqrt(np.mean(frac_s[continuum_mask] ** 2)))
    else:
        continuum_rms = full_rms

    edges = np.linspace(wave_s.min(), wave_s.max(), n_bins + 1)
    bin_centers: list[float] = []
    bin_med_frac: list[float] = []
    for i in range(n_bins):
        m = (wave_s >= edges[i]) & (wave_s < edges[i + 1])
        if np.any(m):
            bin_centers.append(float(0.5 * (edges[i] + edges[i + 1])))
            bin_med_frac.append(float(np.median(frac_s[m])))

    return {
        "continuum_rms": continuum_rms,
        "full_rms": full_rms,
        "median_ratio": float(np.median(model / data)),
        "n_pixels": int(wave.size),
        "n_continuum_pixels": int(np.sum(continuum_mask)),
        "binned_wave_A": bin_centers,
        "binned_median_frac": bin_med_frac,
    }


@dataclass(frozen=True)
class ContinuumDiagnostic:
    """Scalar metrics plus full-resolution spectrum arrays for plotting."""

    metrics: dict[str, Any]
    obs_wave: np.ndarray
    obs_flux: np.ndarray
    model_flux: np.ndarray


def diagnose_continuum_from_samples(
    fit_data: dict,
    samples_path: str,
    *,
    epoch_index: int = 0,
    fixed_pc: bool = True,
) -> ContinuumDiagnostic:
    """Forward-model spectrum with fixed pc=1,0,0,0 or posterior pc medians."""
    ensure_stellar_sys_path()
    spec_nn = resolve_spec_nn_path()
    GenMod = import_payne_genmod()
    gm = GenMod()
    gm._initspecnn(nnpath=spec_nn, NNtype="LinNet")
    genspec = gm.genspec

    samples = Table.read(samples_path, format="fits")
    bfdict = {k: (float(np.median(samples[k])), float(np.std(samples[k]))) for k in samples.colnames}

    spec_list = fit_data.get("spec") or []
    if not spec_list:
        raise ValueError("fit_data has no spectra")
    spec_i = int(np.clip(epoch_index, 0, len(spec_list) - 1))
    sp = spec_list[spec_i]
    wave = np.asarray(sp["obs_wave"], dtype=float)
    flux = np.asarray(sp["obs_flux"], dtype=float)

    if fixed_pc:
        pars = _fixed_pc_pars(bfdict, spec_i)
        pc_policy = "fixed_1_0_0_0"
    else:
        from darkhunter_sed.plotting import _spec_pars_from_bfdict

        pars = _spec_pars_from_bfdict(bfdict, spec_i, colnames=list(samples.colnames))
        pc_policy = "posterior_median_pc"

    _, mod = genspec(pars, outwave=wave, modpoly=True)
    model_flux = np.asarray(mod, dtype=float)
    metrics = continuum_fractional_residual(wave, flux, model_flux)
    metrics["epoch_index"] = spec_i
    metrics["pc_policy"] = pc_policy
    metrics["Teff_median"] = bfdict.get("Teff", (float("nan"),))[0]
    return ContinuumDiagnostic(
        metrics=metrics,
        obs_wave=wave,
        obs_flux=flux,
        model_flux=model_flux,
    )
