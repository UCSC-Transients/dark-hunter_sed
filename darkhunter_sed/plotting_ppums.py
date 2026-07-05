"""ppUMS-style two-page posterior PDF (spectrum/phot/f_lambda + corner + Kiel/HRD).

Page 1: full spectrum + chi residuals, two zoom windows, photometry mags,
f_lambda SED, and a parameter text block. Page 2: corner plot with Kiel
(log g vs Teff) and HRD (log L vs Teff) insets overlaid with MIST tracks.

Driven by a uberMS sample FITS, the ``fit_data`` dict (same spectrum prep as
the fit), and an optional ``sed_summary.json`` for extra header lines.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from scipy import constants
from scipy.stats import gaussian_kde

from darkhunter_sed.config import plots_dir, photometry_dir
from darkhunter_sed.stellar_data import (
    ensure_stellar_sys_path,
    import_payne_genmod,
    load_photometry_fits,
    normalize_phot_nn_dir,
    resolve_models_dir,
    resolve_spec_nn_path,
)

logger = logging.getLogger(__name__)

SPEEDOFLIGHT = constants.c / 1000.0
LSUN = 3.846e33
PC_CM = 3.085677581467192e18
JANSKY_CGS = 1e-23

ZOOM_WINDOWS = ((5175.0, 5200.0), (5250.0, 5300.0))
FLUX_XLIM_DEFAULT = (0.25, 6.0)


def _mag_to_f_lambda_cgs(mag: float, lambda_angstrom: float, zeropt_jy: float) -> float:
    flux_jy = zeropt_jy * 10.0 ** (mag / -2.5)
    return flux_jy * JANSKY_CGS * (SPEEDOFLIGHT / ((lambda_angstrom * 1e-8) ** 2.0))


def _flux_plot_xlim(wavelength_um: np.ndarray) -> tuple[float, float]:
    """Log-x limits for f_lambda panel: at least FLUX_XLIM_DEFAULT, widened for all phot."""
    xmin, xmax = FLUX_XLIM_DEFAULT
    if wavelength_um.size:
        wmin = float(np.min(wavelength_um))
        wmax = float(np.max(wavelength_um))
        xmin = min(xmin, wmin * 0.85)
        xmax = max(xmax, wmax * 1.15)
    return xmin, xmax


def _stats_float_array(xarr) -> np.ndarray:
    x = np.ma.asarray(xarr, dtype=np.float64)
    if np.ma.is_masked(x):
        x = np.ma.filled(x, np.nan)
    return np.array(x, dtype=np.float64, copy=True)


def _finite_only(xarr) -> np.ndarray:
    a = _stats_float_array(xarr)
    return a[np.isfinite(a)]


def _paired_finite_rows(table, colnames: tuple[str, ...]):
    arrs = [_stats_float_array(table[c]) for c in colnames if c in table.colnames]
    if len(arrs) != len(colnames):
        return tuple()
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return tuple(a[m] for a in arrs)


def _calcbf(xarr) -> tuple[float, float]:
    a = _finite_only(xarr)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(np.median(a)), float(np.std(a))


def _quantile(xarr, qs) -> np.ndarray:
    a = _finite_only(xarr)
    if a.size == 0:
        return np.full(len(qs), np.nan)
    return np.percentile(a, [100.0 * q for q in qs])


def _spec_pars(sample_row, spec_i: int, colnames: list[str], *, force_fixed_pc: bool) -> list[float]:
    def val(key: str, default: float = 0.0) -> float:
        if key in colnames:
            return float(sample_row[key])
        return default

    vrad = val(f"vrad_{spec_i}", val("vrad", 0.0))
    lsf = 60000.0 if force_fixed_pc else val(f"lsf_{spec_i}", val("lsf", 60000.0))
    pars = [
        val("Teff"),
        val("log(g)"),
        val("[Fe/H]"),
        val("[a/Fe]"),
        vrad,
        val("vstar"),
        val("vmic", 1.0),
        lsf,
    ]
    if force_fixed_pc:
        pars.extend([1.0, 0.0, 0.0, 0.0])
        return pars
    pc_keys = sorted(
        [k for k in colnames if k.startswith("pc") and k.endswith(f"_{spec_i}")],
        key=lambda k: int(k.split("_")[0][2:]),
    )
    for k in pc_keys:
        pars.append(float(sample_row[k]))
    return pars


def _pc_fixed(samples: Table, spec_i: int) -> bool:
    key = f"pc0_{spec_i}"
    if key not in samples.colnames:
        return False
    return float(np.std(_finite_only(samples[key]))) < 1e-5


def _mkspec(ax_spec, ax_resid, specdata, model_draws, bf_model, *, waverange=None, labely=True):
    w, f, e = specdata
    if waverange is not None:
        cond = (w >= waverange[0] - 10.0) & (w <= waverange[1] + 10.0)
    else:
        cond = np.ones(w.size, dtype=bool)
        waverange = (float(w.min()), float(w.max()))
    ow, of, oe = w[cond], f[cond], e[cond]

    ax_spec.step(ow, of, ls="-", lw=1.0, c="k", zorder=0)
    ax_spec.yaxis.set_minor_locator(AutoMinorLocator())
    for mod in model_draws:
        ax_spec.plot(ow, mod[cond], ls="-", lw=0.5, c="C3", alpha=0.2, zorder=-1)
        if ax_resid is not None:
            ax_resid.plot(ow, (mod[cond] - of) / oe, ls="-", lw=0.1, c="C3", alpha=0.2, zorder=-1)
    if bf_model is not None:
        ax_spec.plot(ow, bf_model[cond], ls="-", lw=0.8, c="C0", alpha=0.85, zorder=1)

    ax_spec.set_xlim(*waverange)
    if labely:
        ax_spec.set_ylabel("Flux")
    if ax_resid is not None:
        ax_resid.axhline(0.0, ls="-", lw=0.5, c="k", alpha=0.85)
        ax_resid.axhline(-1.0, ls=":", lw=0.5, c="k", alpha=0.85)
        ax_resid.axhline(1.0, ls=":", lw=0.5, c="k", alpha=0.85)
        ax_resid.set_xlim(*waverange)
        ax_resid.set_ylim(-5, 5)
        ax_resid.set_xlabel(r"Wavelength [$\AA$]")
        if labely:
            ax_resid.set_ylabel(r"$\chi$")
        ax_spec.set_xticklabels([])


def _mkphot(ax_phot, ax_flux, photdata, bf_phot, bfdict, phot_draws, *, phot_all=None):
    from uberMS.utils import ccm_curve, photsys, star_basis

    phot_all = photdata if phot_all is None else phot_all
    WAVE_d = photsys.photsys()
    bands_fit = [b for b in WAVE_d if b in photdata]
    bands_all = [b for b in WAVE_d if b in phot_all]
    WAVE = {b: WAVE_d[b][0] for b in bands_all}
    zeropts = {b: WAVE_d[b][2] for b in bands_all}
    fitsym = {b: WAVE_d[b][-2] for b in bands_all}
    fitcol = {b: WAVE_d[b][-1] for b in bands_all}
    fcurves = photsys.filtercurves()

    SB = star_basis.StarBasis(
        libname=str(resolve_models_dir() / "specNN/c3k_v1.3.sed_r500.h5"),
        use_params=["logt", "logg", "feh", "afe"],
        n_neighbors=1,
        verbose=False,
    )
    teff = bfdict["Teff"][0]
    logg = bfdict["log(g)"][0]
    feh = max(bfdict["[Fe/H]"][0], -0.75)
    afe = bfdict["[a/Fe]"][0]
    spec_w, spec_f, _ = SB.get_star_spectrum(logt=np.log10(teff), logg=logg, feh=feh, afe=afe)
    to_cgs = LSUN / (4.0 * np.pi * (PC_CM * bfdict["dist"][0]) ** 2)
    spec_f = spec_f * SB.normalize(logr=bfdict["log(R)"][0]) * to_cgs
    spec_f = spec_f * (SPEEDOFLIGHT / ((spec_w * 1e-8) ** 2.0))
    spec_f = np.nan_to_num(spec_f)
    keep = spec_f > 1e-32
    spec_w, spec_f = spec_w[keep], spec_f[keep]
    extratio = ccm_curve.ccm_curve(spec_w / 10.0, bfdict["Av"][0] / 3.1)
    if ax_flux is not None:
        ax_flux.plot(spec_w / 1e4, np.log10(spec_f / extratio), ls="-", lw=0.5, c="C0", zorder=-1)

    obswave_fit = np.array([WAVE[b] for b in bands_fit])
    obsmag = np.array([photdata[b][0] for b in bands_fit])
    obsmagerr = np.array([photdata[b][1] for b in bands_fit])
    modmag_fit = np.array([bf_phot[b] for b in bands_fit])

    obswave_all = np.array([WAVE[b] for b in bands_all])
    obsflux_all = np.array(
        [_mag_to_f_lambda_cgs(phot_all[b][0], WAVE[b], zeropts[b]) for b in bands_all]
    )
    modflux_fit = {
        b: _mag_to_f_lambda_cgs(bf_phot[b], WAVE[b], zeropts[b])
        for b in bands_fit
        if b in bf_phot
    }

    minf, maxf = np.inf, -np.inf
    if ax_flux is not None:
        for b, lam, fx in zip(bands_all, obswave_all, obsflux_all):
            if not np.isfinite(fx) or np.log10(fx) <= -30.0:
                continue
            s = fitsym[b]
            logfx = np.log10(fx)
            ax_flux.scatter(lam / 1e4, logfx, marker=s, c="k", zorder=1, s=10)
            minf = min(minf, logfx)
            maxf = max(maxf, logfx)
            if b in modflux_fit:
                logmo = np.log10(modflux_fit[b])
                ax_flux.scatter(lam / 1e4, logmo, marker=s, c="C0", zorder=0, s=20)
                minf = min(minf, logmo)
                maxf = max(maxf, logmo)

    if ax_phot is not None:
        for lam, m, me, mo, s in zip(obswave_fit, obsmag, obsmagerr, modmag_fit, [fitsym[b] for b in bands_fit]):
            ax_phot.errorbar(lam / 1e4, m, yerr=me, ls="", marker=",", c="k", zorder=-1)
            ax_phot.scatter(lam / 1e4, m, marker=s, c="k", zorder=-1, s=10)
            ax_phot.scatter(lam / 1e4, mo, marker=s, c="C0", zorder=0, s=10)
        for lam, s, draws in zip(
            obswave_fit,
            [fitsym[b] for b in bands_fit],
            [phot_draws[b] for b in bands_fit],
        ):
            ax_phot.scatter([lam / 1e4 for _ in draws], draws, marker="d", c="C3", ec="none", zorder=-1, s=20, alpha=0.1)

    xlo, xhi = _flux_plot_xlim(obswave_all / 1e4)

    if ax_flux is not None and np.isfinite(minf) and np.isfinite(maxf):
        ax_flux.set_ylim(minf - 0.25, maxf + 0.25)
        ylim = ax_flux.get_ylim()
        for b in bands_all:
            if b not in fcurves:
                continue
            fc_i = fcurves[b]
            trans = 0.15 * fc_i["trans"] * (ylim[1] - ylim[0]) + ylim[0]
            ax_flux.plot(fc_i["wave"] / 1e4, trans, ls="-", lw=1.0, c=fitcol[b], alpha=1.0)
        ax_flux.set_ylim(minf - 0.25, maxf + 0.25)
        ax_flux.set_xlim(xlo, xhi)
        ax_flux.set_xscale("log")
        ax_flux.set_xticks([0.3, 0.5, 0.7, 1.0, 3, 5])
        ax_flux.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax_flux.set_ylabel(r"log(F$_{\lambda}$)")
        ax_flux.set_xlabel(r"$\lambda$ [$\mu$m]")
        ax_flux.yaxis.tick_right()
        ax_flux.yaxis.set_label_position("right")

    if ax_phot is not None:
        ax_phot.set_xlim(xlo, xhi)
        ax_phot.set_xscale("log")
        ax_phot.set_ylim(ax_phot.get_ylim()[::-1])
        ax_phot.set_xticks([0.3, 0.5, 0.7, 1, 3, 5])
        ax_phot.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax_phot.set_ylabel("Mag.")
        ax_phot.yaxis.tick_right()
        ax_phot.yaxis.set_label_position("right")


def _mkkiel(ax_kiel, ax_hrd, samples):
    rows = _paired_finite_rows(samples, ("Teff", "log(g)", "log(L)"))
    if not rows or rows[0].size < 5:
        for ax in (ax_kiel, ax_hrd):
            if ax is not None:
                ax.text(0.5, 0.5, "Too few finite samples", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.set_axis_off()
        return
    teff, logg, logL = rows
    Tbf = _quantile(teff, [0.0001, 0.16, 0.5, 0.84, 0.9999])
    Gbf = _quantile(logg, [0.0001, 0.16, 0.5, 0.84, 0.9999])
    Lbf = _quantile(logL, [0.0001, 0.16, 0.5, 0.84, 0.9999])

    for ax, yv, ybf, ylab in ((ax_kiel, logg, Gbf, "log(g)"), (ax_hrd, logL, Lbf, "log(L)")):
        if ax is None:
            continue
        try:
            kde = gaussian_kde(np.vstack([teff, yv]))
            xg = np.linspace(Tbf[0] - 50.0, Tbf[-1] + 50.0, 100)
            yg = np.linspace(ybf[0] - 0.1, ybf[-1] + 0.1, 100)
            X, Y = np.meshgrid(xg, yg)
            Z = kde.evaluate(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            Z[Z < 1e-5] = np.nan
            ax.imshow(Z, origin="lower", aspect="auto", extent=[xg[0], xg[-1], yg[0], yg[-1]], cmap="BrBG", alpha=0.75)
        except Exception as exc:
            logger.debug("KDE failed for %s: %s", ylab, exc)
        ax.scatter(Tbf[2], ybf[2], marker="*", c="C3", s=25, zorder=5)
        ax.set_xlabel(r"$T_{eff}$")
        ax.set_ylabel(ylab)

    _overlay_mist_tracks(ax_kiel, ax_hrd, samples, Tbf, Gbf, Lbf)


def _overlay_mist_tracks(ax_kiel, ax_hrd, samples, Tbf, Gbf, Lbf):
    needed = ("initial_Mass", "initial_[Fe/H]", "initial_[a/Fe]", "EEP")
    if not all(c in samples.colnames for c in needed):
        _set_kiel_limits(ax_kiel, ax_hrd, Tbf, Gbf, Lbf)
        return
    try:
        from jax import jit
        from misty.predict import GenModJax as GenMIST
    except ImportError:
        _set_kiel_limits(ax_kiel, ax_hrd, Tbf, Gbf, Lbf)
        return

    mistNN = str(resolve_models_dir() / "mistNN/mistyNN_2.3_v256_v0.h5")
    try:
        GMIST = GenMIST.modpred(nnpath=mistNN, nntype="LinNet", normed=True, applyspot=False)
        genfn = jit(GMIST.getMIST)
    except Exception as exc:
        logger.debug("MIST init failed: %s", exc)
        _set_kiel_limits(ax_kiel, ax_hrd, Tbf, Gbf, Lbf)
        return

    eep = _finite_only(samples["EEP"])
    EEPbf = int(np.clip(np.median(eep), 1, 900)) if eep.size else 300
    cmap = ListedColormap(["coral", "steelblue", "forestgreen", "plum"])
    norm = BoundaryNorm([1, 203, 405, 606, 808], cmap.N)
    n_draw = min(25, len(samples))
    idx = np.random.choice(len(samples), size=n_draw, replace=False)
    eep_range = np.arange(EEPbf - 200, EEPbf + 200, 1)
    for irow in idx:
        row = samples[int(irow)]
        Ts, Gs, Ls = [], [], []
        for eep_i in eep_range:
            d = genfn(eep=eep_i, mass=row["initial_Mass"], feh=row["initial_[Fe/H]"], afe=row["initial_[a/Fe]"], verbose=False)
            Ts.append(10.0 ** d["log(Teff)"])
            Gs.append(d["log(g)"])
            Ls.append(d["log(L)"])
        for ax, ys in ((ax_kiel, Gs), (ax_hrd, Ls)):
            if ax is None:
                continue
            pts = np.array([Ts, ys]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=cmap, norm=norm)
            lc.set_array(eep_range)
            lc.set_linewidth(2)
            lc.set_alpha(0.25)
            lc.set_zorder(-1)
            ax.add_collection(lc)
    _set_kiel_limits(ax_kiel, ax_hrd, Tbf, Gbf, Lbf)


def _set_kiel_limits(ax_kiel, ax_hrd, Tbf, Gbf, Lbf):
    Trange = [min(Tbf[-1] + 500.0, 8500.0), max(Tbf[0] - 500.0, 3500.0)]
    if ax_kiel is not None:
        ax_kiel.set_xlim(Trange)
        ax_kiel.set_ylim([min(Gbf[-1] + 0.25, 5.5), max(Gbf[0] - 0.25, 0.0)])
    if ax_hrd is not None:
        ax_hrd.set_xlim(Trange)
        ax_hrd.set_ylim([max(Lbf[-1] - 0.25, -3.0), min(Lbf[0] + 0.25, 3.0)])


def _corner_param_list(samples: Table, spec_i: int) -> list[str]:
    cn = samples.colnames
    pcterms = sorted(
        [k for k in cn if k.startswith("pc") and k.endswith(f"_{spec_i}")],
        key=lambda k: int(k.split("_")[0][2:]),
    )
    pltpars = ["Teff", "log(g)", "[Fe/H]", "[a/Fe]"]
    if f"vrad_{spec_i}" in cn or "vrad" in cn:
        pltpars.append("vrad")
    pltpars.append("vstar")
    if "vmic" in cn:
        pltpars.append("vmic")
    if f"lsf_{spec_i}" in cn or "lsf" in cn:
        pltpars.append("lsf")
    pltpars += pcterms
    if "Av" in cn:
        pltpars += ["dist", "log(R)", "Av"]
    for p in ("EEP", "log(Age)", "Mass", "log(L)"):
        if p in cn:
            pltpars.append(p)
    # drop fixed / missing
    out = []
    for p in pltpars:
        if p not in cn:
            continue
        w = _finite_only(samples[p])
        if w.size and w.min() != w.max():
            out.append(p)
    return out


def _prep_corner_aliases(samples: Table, bfdict: dict, spec_i: int) -> None:
    cn = samples.colnames
    if f"vrad_{spec_i}" in cn and "vrad" not in cn:
        samples["vrad"] = np.asarray(samples[f"vrad_{spec_i}"])
        bfdict["vrad"] = _calcbf(samples["vrad"])
    if f"lsf_{spec_i}" in cn and "lsf" not in cn:
        samples["lsf"] = np.asarray(samples[f"lsf_{spec_i}"])
        bfdict["lsf"] = _calcbf(samples["lsf"])


def _draw_corner(fig, samples, bfdict, pltpars):
    parind = np.arange(len(pltpars))
    nbins = 35
    gs = gridspec.GridSpec(len(pltpars), len(pltpars))
    gs.update(wspace=0.05, hspace=0.05)
    for kk in itertools.product(pltpars, pltpars):
        i0 = parind[np.array(pltpars) == kk[0]][0]
        i1 = parind[np.array(pltpars) == kk[1]][0]
        ax = fig.add_subplot(gs[i0, i1])
        if i0 < i1:
            ax.set_axis_off()
            continue
        xr = [bfdict[kk[0]][0] - 5.0 * bfdict[kk[0]][1], bfdict[kk[0]][0] + 5.0 * bfdict[kk[0]][1]]
        yr = [bfdict[kk[1]][0] - 5.0 * bfdict[kk[1]][1], bfdict[kk[1]][0] + 5.0 * bfdict[kk[1]][1]]
        if kk[0] == kk[1]:
            ax.hist(_finite_only(samples[kk[0]]), bins=nbins, histtype="step", lw=1.5, density=True, range=xr)
            ax.set_yticks([])
            ax.text(0.5, 1.1, "{0:.3g} +/- {1:.3g}".format(*bfdict[kk[0]]), ha="center", va="center", transform=ax.transAxes, fontsize=5)
        else:
            try:
                ax.hist2d(_finite_only(samples[kk[1]]), _finite_only(samples[kk[0]]), bins=nbins, cmap="Blues", range=[yr, xr])
            except Exception:
                pass
            ax.set_xlim(*yr)
            ax.set_ylim(*xr)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_fontsize(6)
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(6)
        if i1 != 0 or (i0 == 0):
            ax.set_yticks([])
        elif kk[0] != pltpars[0]:
            ax.set_ylabel(kk[0], fontsize=7)
        if i0 != len(pltpars) - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel(kk[1], fontsize=7)


def _param_text(gaia_id, bfdict, specdata, sed_summary, spec_i) -> str:
    snr = float(np.median(specdata[1] / np.maximum(specdata[2], 1e-99)))
    s = f"Gaia DR3 {gaia_id}\nSNR = {snr:.2f}\n"
    def line(key, label, fmt="{0:.3f} +/- {1:.3f}"):
        if key in bfdict and np.isfinite(bfdict[key][0]):
            return label + " = " + fmt.format(*bfdict[key]) + "\n"
        return ""
    s += line("Teff", r"T$_{eff}$", "{0:.0f} +/- {1:.0f} K")
    s += line("log(g)", "log(g)")
    s += line("[Fe/H]", "[Fe/H]")
    s += line("[a/Fe]", "[a/Fe]")
    s += line(f"vrad_{spec_i}", r"V$_{rad}$", "{0:.3f} +/- {1:.3f} km/s")
    s += line("vstar", r"V$_{\bigstar}$", "{0:.2f} +/- {1:.2f} km/s")
    s += line("vmic", r"V$_{mic}$", "{0:.2f} +/- {1:.2f} km/s")
    s += line("log(R)", "log(R)")
    if "dist" in bfdict and np.isfinite(bfdict["dist"][0]):
        s += "Dist = {0:.3f} +/- {1:.3f} kpc\n".format(bfdict["dist"][0] / 1000.0, bfdict["dist"][1] / 1000.0)
    s += line("Av", r"A$_{V}$")
    s += line("initial_Mass", "Mass")
    s += line("Age", "Age", "{0:.3f} +/- {1:.3f} Gyr")
    if isinstance(sed_summary, dict):
        m1 = sed_summary.get("m1_msun")
        if isinstance(m1, dict) and m1.get("median") is not None:
            s += "M1 = {0:.3f} Msun\n".format(float(m1["median"]))
        av = sed_summary.get("av_prior")
        if isinstance(av, dict):
            s += "Av prior: {0} ({1})\n".format(av.get("map_used", "?"), av.get("prior_kind", ""))
    return s


def plot_ums_posterior(
    gaia_id: str,
    fit_data: dict,
    samples_path: str | Path,
    *,
    fit_type: str = "ums",
    sed_summary: dict[str, Any] | None = None,
    epoch_index: int = 0,
    out_path: str | Path | None = None,
    n_posterior_draws: int = 30,
) -> Path:
    """Write two-page ppUMS-style posterior PDF from a sample FITS + ``fit_data``."""
    ensure_stellar_sys_path()
    root = resolve_models_dir()
    GM = import_payne_genmod()()
    GM._initspecnn(nnpath=resolve_spec_nn_path(), NNtype="LinNet")
    GM._initphotnn(list(fit_data["phot_filtarr"]), nnpath=normalize_phot_nn_dir(root / "photNN"))
    genspec = GM.genspec
    genphot = GM.genphot

    samples = Table.read(Path(samples_path), format="fits")
    if "vmic" not in samples.colnames:
        samples["vmic"] = np.ones(len(samples), dtype=float)
    bfdict = {k: _calcbf(samples[k]) for k in samples.colnames}

    spec_list = fit_data.get("spec") or []
    if not spec_list:
        raise ValueError("fit_data has no spectra for plotting")
    spec_i = int(np.clip(epoch_index, 0, len(spec_list) - 1))
    sp = spec_list[spec_i]
    specdata = (
        np.asarray(sp["obs_wave"], float),
        np.asarray(sp["obs_flux"], float),
        np.maximum(np.asarray(sp["obs_eflux"], float), 1e-99),
    )

    colnames = list(samples.colnames)
    pcs_fixed = fit_type.lower() == "ums" and _pc_fixed(samples, spec_i)
    modpoly = not pcs_fixed  # rigid UMS: no pc correction

    bf_pars = _spec_pars(bfdict_row := {k: bfdict[k][0] for k in colnames}, spec_i, colnames, force_fixed_pc=pcs_fixed)
    _, bf_model = genspec(bf_pars, outwave=specdata[0], modpoly=modpoly)

    draw_idx = np.linspace(0, len(samples) - 1, min(n_posterior_draws, len(samples)), dtype=int)
    model_draws = []
    phot_draws = {b: [] for b in fit_data["phot_filtarr"]}
    for irow in draw_idx:
        row = samples[int(irow)]
        pars = _spec_pars(row, spec_i, colnames, force_fixed_pc=pcs_fixed)
        _, mod = genspec(pars, outwave=specdata[0], modpoly=modpoly)
        if np.all(np.isfinite(mod)):
            model_draws.append(np.asarray(mod, float))
        pp = genphot([float(row["Teff"]), float(row["log(g)"]), float(row["[Fe/H]"]), float(row["[a/Fe]"]), float(row["log(R)"]), float(row["dist"]), float(row["Av"]), 3.1])
        for b in phot_draws:
            phot_draws[b].append(pp[b])

    bf_phot = genphot([bfdict["Teff"][0], bfdict["log(g)"][0], bfdict["[Fe/H]"][0], bfdict["[a/Fe]"][0], bfdict["log(R)"][0], bfdict["dist"][0], bfdict["Av"][0], 3.1])

    phot_all = fit_data.get("phot_all")
    if phot_all is None:
        try:
            phot_all, _ = load_photometry_fits(gaia_id, photometry_dir())
        except FileNotFoundError:
            phot_all = fit_data["phot"]

    if out_path is None:
        out_path = plots_dir() / f"Gaia_DR3_{gaia_id}_{fit_type.lower()}.pdf"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        # ---- Page 1: spectrum / zooms / phot / f_lambda / params ----
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        gs = gridspec.GridSpec(6, 6, figure=fig)
        ax_spec = fig.add_subplot(gs[:3, :-2])
        ax_resid = fig.add_subplot(gs[3:4, :-2])
        ax_z1 = fig.add_subplot(gs[4:, :2])
        ax_z2 = fig.add_subplot(gs[4:, 2:4])
        ax_phot = fig.add_subplot(gs[:2, -2:])
        ax_flux = fig.add_subplot(gs[2:4, -2:])

        try:
            _mkphot(ax_phot, ax_flux, fit_data["phot"], bf_phot, bfdict, phot_draws, phot_all=phot_all)
        except Exception as exc:
            logger.warning("phot/f_lambda panel failed: %s", exc)
            ax_phot.text(0.5, 0.5, "phot panel failed", ha="center", transform=ax_phot.transAxes, fontsize=8)

        _mkspec(ax_spec, ax_resid, specdata, model_draws, bf_model, waverange=None)
        _mkspec(ax_z1, None, specdata, model_draws, bf_model, waverange=ZOOM_WINDOWS[0])
        _mkspec(ax_z2, None, specdata, model_draws, bf_model, waverange=ZOOM_WINDOWS[1], labely=False)
        ax_spec.set_title(f"Gaia {gaia_id} epoch {spec_i + 1} ({fit_type.upper()}; {'pc fixed 1,0,0,0' if pcs_fixed else 'pc free'})")

        fig.text(0.70, 0.02, _param_text(gaia_id, bfdict, specdata, sed_summary, spec_i), fontsize=8)
        fig.align_labels()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 2: corner + Kiel + HRD ----
        _prep_corner_aliases(samples, bfdict, spec_i)
        pltpars = _corner_param_list(samples, spec_i)
        fig = plt.figure(figsize=(15, 15))
        ax_kiel = fig.add_axes([0.6, 0.75, 0.22, 0.22])
        ax_hrd = fig.add_axes([0.6, 0.5, 0.22, 0.22])
        _mkkiel(ax_kiel, ax_hrd, samples)
        if pltpars:
            _draw_corner(fig, samples, bfdict, pltpars)
        else:
            fig.text(0.5, 0.08, "Corner skipped: all candidate params fixed.", ha="center", fontsize=10)
        fig.align_labels()
        pdf.savefig(fig)
        plt.close(fig)

    logger.info("Wrote ppUMS-style PDF %s", out_path)
    return out_path


def plot_utp_posterior(gaia_id, fit_data, samples_path, **kwargs) -> Path:
    """UTP posterior PDF (same layout; pc terms free)."""
    kwargs.setdefault("fit_type", "utp")
    return plot_ums_posterior(gaia_id, fit_data, samples_path, **kwargs)
