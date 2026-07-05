"""Diagnostic PDF plots using the same spectrum/phot data as the fit."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages

from darkhunter_sed.config import plots_dir
from darkhunter_sed.stellar_data import (
    ensure_stellar_sys_path,
    import_payne_genmod,
    normalize_phot_nn_dir,
    resolve_models_dir,
    resolve_spec_nn_path,
)

logger = logging.getLogger(__name__)


def _calcbf(xarr) -> tuple[float, float]:
    a = np.asarray(xarr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(np.median(a)), float(np.std(a))


def _spec_pars_from_sample(
    sample,
    spec_i: int,
    *,
    use_sample_vrad: bool = True,
) -> list[float]:
    colnames = list(sample.colnames) if hasattr(sample, "colnames") else []
    vrad_key = f"vrad_{spec_i}"
    lsf_key = f"lsf_{spec_i}"
    if use_sample_vrad and vrad_key in colnames:
        vrad = float(sample[vrad_key])
    elif "vrad" in colnames:
        vrad = float(sample["vrad"])
    else:
        vrad = 0.0
    if lsf_key in colnames:
        lsf = float(sample[lsf_key])
    elif "lsf" in colnames:
        lsf = float(sample["lsf"])
    else:
        lsf = 60000.0
    pars: list[float] = [
        float(sample["Teff"]),
        float(sample["log(g)"]),
        float(sample["[Fe/H]"]),
        float(sample["[a/Fe]"]),
        vrad,
        float(sample["vstar"]),
        float(sample["vmic"]),
        lsf,
    ]
    pc_keys = sorted(
        [k for k in colnames if k.startswith("pc") and k.endswith(f"_{spec_i}")],
        key=lambda k: int(k.split("_")[0][2:]),
    )
    for k in pc_keys:
        pars.append(float(sample[k]))
    if not pc_keys:
        for k in ("pc0", "pc1", "pc2", "pc3"):
            if k in colnames:
                pars.append(float(sample[k]))
    return pars


def _pc_terms_fixed_in_samples(samples: Table, spec_i: int) -> bool:
    key = f"pc0_{spec_i}"
    if key not in samples.colnames:
        return False
    return float(np.std(samples[key])) < 1e-5


def _spec_pars_from_bfdict(
    bfdict: dict[str, tuple[float, float]],
    spec_i: int,
    *,
    colnames: list[str] | None = None,
    force_fixed_pc: bool = False,
) -> list[float]:
    def med(key: str, default: float = 0.0) -> float:
        if key in bfdict:
            return float(bfdict[key][0])
        return default

    vrad_key = f"vrad_{spec_i}"
    lsf_key = f"lsf_{spec_i}"
    vrad = med(vrad_key, med("vrad", 0.0))
    lsf = 60000.0 if force_fixed_pc else med(lsf_key, med("lsf", 60000.0))
    pars: list[float] = [
        med("Teff"),
        med("log(g)"),
        med("[Fe/H]"),
        med("[a/Fe]"),
        vrad,
        med("vstar"),
        med("vmic", 1.0),
        lsf,
    ]
    if force_fixed_pc:
        pars.extend([1.0, 0.0, 0.0, 0.0])
        return pars
    if colnames:
        pc_keys = sorted(
            [k for k in colnames if k.startswith("pc") and k.endswith(f"_{spec_i}")],
            key=lambda k: int(k.split("_")[0][2:]),
        )
        for k in pc_keys:
            pars.append(med(k))
        if not pc_keys:
            for k in ("pc0", "pc1", "pc2", "pc3"):
                if k in bfdict or k in colnames:
                    pars.append(med(k))
    else:
        for pci in range(4):
            k = f"pc{pci}_{spec_i}"
            if k in bfdict:
                pars.append(med(k))
    return pars


def plot_fit_diagnostics(
    gaia_id: str,
    fit_data: dict,
    samples_path: str | Path,
    *,
    fit_type: str = "ums",
    epoch_index: int = 0,
    out_path: str | Path | None = None,
    n_posterior_draws: int = 30,
) -> Path:
    """
    Write spectrum + photometry diagnostic PDF aligned with ``fit_data`` spectra.

    Uses Payne ``genspec`` / ``genphot`` with per-epoch ``vrad_i`` from posterior.
    """
    ensure_stellar_sys_path()
    root = resolve_models_dir()
    spec_nn = resolve_spec_nn_path()
    phot_nn = normalize_phot_nn_dir(root / "photNN")
    GenMod = import_payne_genmod()
    gm = GenMod()
    gm._initspecnn(nnpath=spec_nn, NNtype="LinNet")
    gm._initphotnn(list(fit_data["phot_filtarr"]), nnpath=phot_nn)
    genspec = gm.genspec
    genphot = gm.genphot

    samples_path = Path(samples_path)
    samples = Table.read(samples_path, format="fits")
    bfdict = {k: _calcbf(samples[k]) for k in samples.colnames}

    spec_list = fit_data.get("spec") or []
    if not spec_list:
        raise ValueError("fit_data has no spectra for plotting")
    spec_i = int(np.clip(epoch_index, 0, len(spec_list) - 1))
    sp = spec_list[spec_i]
    obs_wave = np.asarray(sp["obs_wave"], dtype=float)
    obs_flux = np.asarray(sp["obs_flux"], dtype=float)
    obs_eflux = np.maximum(np.asarray(sp["obs_eflux"], dtype=float), 1e-99)

    colnames = list(samples.colnames)
    pcs_fixed = fit_type.lower() == "ums" and _pc_terms_fixed_in_samples(samples, spec_i)
    bf_pars = _spec_pars_from_bfdict(
        bfdict, spec_i, colnames=colnames, force_fixed_pc=pcs_fixed
    )
    _, bf_flux = genspec(bf_pars, outwave=obs_wave, modpoly=True)

    draw_idx = np.linspace(0, len(samples) - 1, min(n_posterior_draws, len(samples)), dtype=int)
    spec_draws: list[np.ndarray] = []
    for irow in draw_idx:
        if pcs_fixed:
            row_bfdict = {k: (float(samples[k][int(irow)]), 0.0) for k in samples.colnames}
            pars = _spec_pars_from_bfdict(
                row_bfdict, spec_i, colnames=colnames, force_fixed_pc=True
            )
        else:
            pars = _spec_pars_from_sample(samples[int(irow)], spec_i, use_sample_vrad=True)
        _, mod = genspec(pars, outwave=obs_wave, modpoly=True)
        if np.all(np.isfinite(mod)):
            spec_draws.append(np.asarray(mod, dtype=float))

    phot_pars_bf = [
        bfdict["Teff"][0],
        bfdict["log(g)"][0],
        bfdict["[Fe/H]"][0],
        bfdict["[a/Fe]"][0],
        bfdict["log(R)"][0],
        bfdict["dist"][0],
        bfdict["Av"][0],
        3.1,
    ]
    bf_phot = genphot(phot_pars_bf)

    if out_path is None:
        out_path = plots_dir() / f"Gaia_DR3_{gaia_id}_{fit_type.lower()}.pdf"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax_spec, ax_res = axes
        ax_spec.step(obs_wave, obs_flux, color="k", lw=1.0, where="mid", label="data")
        for mod in spec_draws:
            ax_spec.plot(obs_wave, mod, color="C3", alpha=0.15, lw=0.5)
        ax_spec.plot(obs_wave, bf_flux, color="C0", lw=1.2, label="best fit (posterior median)")
        pc_note = "pc*=1,0,0,0 fixed" if pcs_fixed else "pc* from posterior"
        ax_spec.set_title(
            f"Gaia {gaia_id} epoch {spec_i + 1} ({fit_type.upper()}; {pc_note})"
        )
        ax_spec.set_ylabel("Flux")
        ax_spec.legend(loc="upper right", fontsize=8)

        resid = (bf_flux - obs_flux) / obs_eflux
        ax_res.axhline(0, color="k", lw=0.5)
        ax_res.plot(obs_wave, resid, color="C0", lw=0.8)
        ax_res.set_ylim(-5, 5)
        ax_res.set_xlabel(r"Wavelength [$\AA$]")
        ax_res.set_ylabel(r"$\chi$")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        ratio = np.divide(
            obs_flux,
            np.maximum(bf_flux, 1e-99),
            out=np.full_like(obs_flux, np.nan),
            where=np.isfinite(bf_flux),
        )
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(obs_wave, ratio, color="C2", lw=0.8)
        ax.axhline(1.0, color="k", lw=0.5, ls="--")
        ax.set_xlabel(r"Wavelength [$\AA$]")
        ax.set_ylabel("data / model")
        ax.set_title(f"Continuum offset ({pc_note})")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        bands = list(fit_data["phot_filtarr"])
        obs_m = [fit_data["phot"][b][0] for b in bands]
        obs_e = [fit_data["phot"][b][1] for b in bands]
        mod_m = [bf_phot[b] for b in bands]
        x = np.arange(len(bands))
        ax.errorbar(x, obs_m, yerr=obs_e, fmt="ko", label="data")
        ax.scatter(x, mod_m, c="C0", marker="s", label="model (posterior median)", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mag")
        ax.set_title("Photometry (best fit)")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        teff = bfdict.get("Teff", (float("nan"), float("nan")))
        logg = bfdict.get("log(g)", (float("nan"), float("nan")))
        av = bfdict.get("Av", (float("nan"), float("nan")))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        lines = [
            f"Gaia DR3 {gaia_id}",
            f"Fit: {fit_type.upper()}",
            f"Epoch plotted: {spec_i + 1} / {len(spec_list)}",
            f"Teff = {teff[0]:.1f} +/- {teff[1]:.1f} K",
            f"log g = {logg[0]:.3f} +/- {logg[1]:.3f}",
            f"A_V = {av[0]:.4f} +/- {av[1]:.4f}",
        ]
        av_meta = fit_data.get("av_prior")
        if isinstance(av_meta, dict):
            lines.append(f"Av prior map: {av_meta.get('map_used', '?')} ({av_meta.get('prior_kind', '')})")
        ax.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=10, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

    logger.info("Wrote diagnostic PDF %s", out_path)
    return out_path
