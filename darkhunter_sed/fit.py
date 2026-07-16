"""uberMS SVI drivers for UMS (primary) and UTP (secondary)."""

from __future__ import annotations

import inspect
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np

from darkhunter_sed import data, models, priors
from darkhunter_sed.config import samples_dir
from darkhunter_sed.dust_prior import AvPriorResult, build_av_prior_from_fit_data
from darkhunter_sed.initguess import build_ums_initpars
from darkhunter_sed.stellar_data import sanitize_xla_flags_env

logger = logging.getLogger(__name__)

SB1_RV_SPAN_WARN_KMS = 10.0


def _ensure_cpu_jax_on_macos() -> None:
    """uberMS SVI is validated on CPU; avoid experimental Metal unless user overrides."""
    if sys.platform == "darwin" and not os.environ.get("JAX_PLATFORMS"):
        os.environ["JAX_PLATFORMS"] = "cpu"


def _log_jax_backend() -> None:
    try:
        import jax

        backend = jax.default_backend()
        logger.info("JAX %s backend=%s", jax.__version__, backend)
        if backend != "cpu":
            logger.warning(
                "JAX backend is %r (expected 'cpu' for uberMS SVI). "
                "Set JAX_PLATFORMS=cpu or use conda Python.",
                backend,
            )
    except ImportError:
        logger.warning("JAX not importable before UMS SVI")


def _apply_av_prior(
    fit_data: dict,
    initpars: dict,
    priors_dict: dict,
    *,
    dust_av_prior: bool,
) -> AvPriorResult | None:
    if not dust_av_prior:
        return None
    av_result = build_av_prior_from_fit_data(fit_data, use_dustmaps=True)
    priors_dict["Av"] = av_result.prior
    initpars["Av"] = av_result.init_av
    fit_data["av_prior"] = av_result.to_metadata()
    logger.info(
        "Av prior from %s (%s): loc=%.4f scale=%.4f [%.4f, %.4f]",
        av_result.map_used,
        av_result.prior_kind,
        av_result.prior[1][0],
        av_result.prior[1][1],
        av_result.prior[1][2],
        av_result.prior[1][3],
    )
    return av_result


def _samples_outfile(gaia_id: str, runtype: str, output_name: str | None = None) -> Path:
    samples_dir().mkdir(parents=True, exist_ok=True)
    if output_name:
        return samples_dir() / output_name
    suffix = "ums" if runtype.upper() == "UMS" else "utp"
    return samples_dir() / f"Gaia_DR3_{gaia_id}_{suffix}.fits"


def preflight_ums(nspec: int, gaia_id: str) -> None:
    """
    Validate UMS spectrum count.

    uberMS.dva models support nspec>=1. Zero spectra is still invalid when dospec.
    """
    if nspec < 1:
        raise ValueError(
            f"UMS requires at least 1 spectrum for {gaia_id}; got {nspec}."
        )


def _dist_prior_ms_tp(
    data: dict,
    *,
    omit_parallax_likelihood: bool,
    dist_uniform_min_pc: float | None,
    dist_uniform_max_pc: float | None,
) -> tuple[float, list]:
    pl0, pl1 = data["parallax"]
    if omit_parallax_likelihood:
        dlo = float(dist_uniform_min_pc) if dist_uniform_min_pc is not None else 50.0
        dhi = float(dist_uniform_max_pc) if dist_uniform_max_pc is not None else 5000.0
        if dlo >= dhi:
            raise ValueError("dist_uniform_min_pc must be < dist_uniform_max_pc")
        return math.sqrt(dlo * dhi), ["uniform", [dlo, dhi]]
    dist_init = 1000.0 / pl0
    return dist_init, [
        "normal",
        [dist_init, (dist_init - 1000.0 / (pl0 + pl1)) * 5.0],
    ]


def _dist_prior_tp(
    data: dict,
    *,
    omit_parallax_likelihood: bool,
    dist_uniform_min_pc: float | None,
    dist_uniform_max_pc: float | None,
) -> tuple[float, list]:
    pl0, pl1 = data["parallax"]
    if omit_parallax_likelihood:
        dlo = float(dist_uniform_min_pc) if dist_uniform_min_pc is not None else 50.0
        dhi = float(dist_uniform_max_pc) if dist_uniform_max_pc is not None else 5000.0
        if dlo >= dhi:
            raise ValueError("dist_uniform_min_pc must be < dist_uniform_max_pc")
        return math.sqrt(dlo * dhi), ["uniform", [dlo, dhi]]
    dist_init = 1000.0 / pl0
    return dist_init, [
        "normal",
        [dist_init, (dist_init - 1000.0 / (pl0 + pl1)) * 10.0],
    ]


def _svi_guide_key(svi_guide: str) -> str:
    g = (svi_guide or "normal").strip().lower()
    if g in ("flow", "bnaf", "normalizing flow", "normalizing_flow"):
        return "Normalizing Flow"
    return "Normal"


def _apply_rv_priors_to_indict(
    indict: dict,
    initpars: dict,
    rv_epochs: list[priors.RvEpoch],
    *,
    vrad_prior: str = "normal",
    vrad_err_inflate: float = 2.0,
    vrad_err_floor_kms: float = 2.0,
    vrad_uniform_half_span: float = 80.0,
) -> None:
    v_init, v_priors = priors.build_vrad_init_and_priors(
        rv_epochs,
        prior_kind=vrad_prior,
        err_inflate=vrad_err_inflate,
        err_floor_kms=vrad_err_floor_kms,
        uniform_half_span=vrad_uniform_half_span,
    )
    for key, val in v_init.items():
        initpars[key] = val
    for key, val in v_priors.items():
        indict["priors"][key] = val


def _apply_per_epoch_spec_priors(
    priors_dict: dict,
    nspec: int,
    *,
    fix_specjitter: bool,
    rigid_continuum: bool = True,
    pc0_min: float = 0.95,
    pc0_max: float = 1.05,
    pc_poly_abs_max: float = 0.1,
    specjitter_max: float = 1e-2,
) -> None:
    """
    Per-epoch spectral nuisance priors (LSF, continuum polynomial, spec jitter).

    Default (``rigid_continuum=True``) matches legacy uberMS UMS: fixed pc0–pc3 at
    1,0,0,0 and LSF at 60000. Continuum is handled in sinc_blaze ingest; modpoly
    pc terms were calibrated on a different spectrograph and must not be fit on APF.
    """
    for ii in range(nspec):
        if rigid_continuum:
            priors_dict[f"lsf_{ii}"] = ["fixed", 60000.0]
            for pc in ("pc0", "pc1", "pc2", "pc3"):
                priors_dict[f"{pc}_{ii}"] = ["fixed", 1.0 if pc == "pc0" else 0.0]
        else:
            priors_dict[f"lsf_{ii}"] = ["tnormal", [60000.0, 500.0, 50000.0, 65000.0]]
            for pc, bounds in (
                ("pc0", [pc0_min, pc0_max]),
                ("pc1", [-pc_poly_abs_max, pc_poly_abs_max]),
                ("pc2", [-pc_poly_abs_max, pc_poly_abs_max]),
                ("pc3", [-pc_poly_abs_max, pc_poly_abs_max]),
            ):
                priors_dict[f"{pc}_{ii}"] = ["uniform", bounds]
        priors_dict[f"specjitter_{ii}"] = (
            ["fixed", 0.0] if fix_specjitter else ["uniform", [1e-6, specjitter_max]]
        )


def _warn_large_rv_span(rv_epochs: list[priors.RvEpoch]) -> None:
    rvs = [float(ep.rv_kms) for ep in rv_epochs if math.isfinite(ep.rv_kms)]
    if len(rvs) < 2:
        return
    span = max(rvs) - min(rvs)
    if span > SB1_RV_SPAN_WARN_KMS:
        logger.warning(
            "RV span across epochs is %.1f km/s (>%.0f); binary orbital motion may stress "
            "tight vrad_i normal priors. Consider --vrad-prior uniform or higher --vrad-err-inflate.",
            span,
            SB1_RV_SPAN_WARN_KMS,
        )


def build_ums_indict(
    gaia_id: str,
    fit_data: dict,
    *,
    output_name: str | None = None,
    dospec: bool = True,
    dophot: bool = True,
    progressbar: bool = True,
    svi_guide: str = "normal",
    fix_vstar: bool = False,
    fix_photjitter: bool = False,
    fix_specjitter: bool = False,
    vrad_prior: str = "normal",
    vrad_err_inflate: float = 2.0,
    vrad_err_floor_kms: float = 2.0,
    omit_parallax_likelihood: bool = False,
    dist_uniform_min_pc: float | None = None,
    dist_uniform_max_pc: float | None = None,
    dust_av_prior: bool = True,
    mist_nn: str | None = None,
    rigid_continuum: bool = True,
    spec_err_scale: float = 1.0,
    phot_err_scale: float = 1.0,
    pc0_min: float = 0.95,
    pc0_max: float = 1.05,
    pc_poly_abs_max: float = 0.1,
    specjitter_max: float = 1e-2,
    photjitter_max: float = 1e-2,
) -> tuple[dict, list[str] | None, str, str]:
    """
    Build uberMS UMS ``indict`` and return (indict, spec_nn_list, phot_nn, mist_nn).

    Gaia values in ``fit_data`` inform ``initpars`` only; ``priors`` match legacy uberMS.
    """
    _spec_nn, phot_nn, mist_nn_path = models.model_paths()
    if mist_nn is not None:
        mist_nn_path = mist_nn

    fit_data = data.apply_likelihood_error_scales(
        fit_data,
        spec_err_scale=spec_err_scale,
        phot_err_scale=phot_err_scale,
    )
    if not rigid_continuum:
        if pc0_min >= pc0_max:
            raise ValueError("pc0_min must be < pc0_max")
        if pc_poly_abs_max <= 0.0:
            raise ValueError("pc_poly_abs_max must be > 0")
    if specjitter_max <= 0.0 or photjitter_max <= 0.0:
        raise ValueError("specjitter_max and photjitter_max must be > 0")
    if dophot and "parallax" in fit_data:
        plx, plx_err = fit_data["parallax"]
        logger.info(
            "Distance prior parallax: %.5f +/- %.5f mas (Gaia query COALESCE nss_2b, nss_acc, gaia_source)",
            plx,
            plx_err,
        )
    av_meta = fit_data.get("av_prior")
    if isinstance(av_meta, dict):
        logger.info(
            "Av prior dust map=%s R_V=%s; phot forward model uses R_V=3.1",
            av_meta.get("map_used", "?"),
            av_meta.get("rv", "?"),
        )

    indict: dict = {}
    outfile = _samples_outfile(gaia_id, "UMS", output_name)
    indict["outfile"] = str(outfile)

    indict["data"] = {}
    nspec = 0
    if dospec and "spec" in fit_data:
        indict["data"]["spec"] = fit_data["spec"]
        nspec = len(indict["data"]["spec"])
        preflight_ums(nspec, gaia_id)
    if dophot:
        indict["data"]["phot"] = fit_data["phot"]
        indict["data"]["parallax"] = fit_data["parallax"]
        indict["data"]["filtarr"] = list(fit_data["phot_filtarr"])

    spec_nn_list = [str(_spec_nn) for _ in range(nspec)] if nspec > 0 else None

    dist_init, dist_prior = _dist_prior_ms_tp(
        fit_data,
        omit_parallax_likelihood=omit_parallax_likelihood,
        dist_uniform_min_pc=dist_uniform_min_pc,
        dist_uniform_max_pc=dist_uniform_max_pc,
    )
    if omit_parallax_likelihood:
        indict["omit_parallax_likelihood"] = True

    initpars = build_ums_initpars(
        fit_data,
        mist_nn_path,
        dist_init=dist_init,
        fix_photjitter=fix_photjitter,
        fix_specjitter=fix_specjitter,
        dospec=dospec,
        nspec=nspec,
    )

    indict["initpars"] = initpars
    indict["priors"] = {
        "EEP": ["uniform", [250, 500]],
        "initial_Mass": ["IMF", {"mass_le": 0.3, "mass_ue": 1.25}],
        "initial_[Fe/H]": ["uniform", [-1.6, 0.5]],
        "initial_[a/Fe]": ["fixed", fit_data["[a/Fe]"]],
        "vstar": ["fixed", 8.0] if fix_vstar else ["uniform", [0.0, 25.0]],
        "vmic": ["fixed", 1.0],
        "Av": ["tnormal", [0.0, 0.1, 0.0, 0.5]],
        "dist": dist_prior,
        "photjitter": ["fixed", 0.0] if fix_photjitter else ["uniform", [1e-6, photjitter_max]],
    }
    _apply_av_prior(fit_data, initpars, indict["priors"], dust_av_prior=dust_av_prior)

    if dospec and nspec > 0:
        logger.info(
            "UMS spectral nuisance: rigid_continuum=%s (pc0..pc3 %s)",
            rigid_continuum,
            "fixed 1,0,0,0" if rigid_continuum else "free",
        )
        _apply_per_epoch_spec_priors(
            indict["priors"],
            nspec,
            fix_specjitter=fix_specjitter,
            rigid_continuum=rigid_continuum,
            pc0_min=pc0_min,
            pc0_max=pc0_max,
            pc_poly_abs_max=pc_poly_abs_max,
            specjitter_max=specjitter_max,
        )

        rv_epochs = fit_data.get("rv_epochs") or []
        if len(rv_epochs) != nspec:
            logger.warning(
                "RV epoch count (%s) != spectrum count (%s); using default vrad priors for missing",
                len(rv_epochs),
                nspec,
            )
            rv_epochs = (rv_epochs + [priors.RvEpoch("", i, float("nan"), float("nan"), float("nan")) for i in range(nspec)])[:nspec]
        _warn_large_rv_span(rv_epochs)
        _apply_rv_priors_to_indict(
            indict,
            initpars,
            rv_epochs,
            vrad_prior=vrad_prior,
            vrad_err_inflate=vrad_err_inflate,
            vrad_err_floor_kms=vrad_err_floor_kms,
        )

    indict["svi"] = {
        "steps": 5000,
        "opt_tol": 1e-6,
        "start_tol": 1e-2,
        "progress_bar": progressbar,
        "post_resample": 4000,
        "guide": _svi_guide_key(svi_guide),
    }

    return indict, spec_nn_list, phot_nn, mist_nn_path


def probe_ums_init(
    indict: dict,
    *,
    spec_nn_list: list[str] | None,
    phot_nn: str,
    mist_nn: str,
) -> tuple[bool, str]:
    """
    Run a minimal UMS SVI step to test NumPyro init (loads NNs).

    Returns (success, message).
    """
    from uberMS.dva import runSVI

    sanitize_xla_flags_env()
    probe = dict(indict)
    probe["svi"] = {**indict.get("svi", {}), "steps": 1, "progress_bar": False, "post_resample": 1}
    svi = runSVI.sviMS(
        specNN=spec_nn_list,
        photNN=phot_nn,
        mistNN=mist_nn,
        verbose=False,
        usegrad=False,
    )
    try:
        svi.run(probe)
        return True, "UMS SVI init succeeded (1-step probe)"
    except RuntimeError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def run_ums(
    gaia_id: str,
    fit_data: dict,
    *,
    output_name: str | None = None,
    dospec: bool = True,
    dophot: bool = True,
    progressbar: bool = True,
    svi_guide: str = "normal",
    fix_vstar: bool = False,
    fix_photjitter: bool = False,
    fix_specjitter: bool = False,
    vrad_prior: str = "normal",
    vrad_err_inflate: float = 2.0,
    vrad_err_floor_kms: float = 2.0,
    omit_parallax_likelihood: bool = False,
    dist_uniform_min_pc: float | None = None,
    dist_uniform_max_pc: float | None = None,
    dust_av_prior: bool = True,
    rigid_continuum: bool = True,
    spec_err_scale: float = 1.0,
    phot_err_scale: float = 1.0,
    pc0_min: float = 0.95,
    pc0_max: float = 1.05,
    pc_poly_abs_max: float = 0.1,
    specjitter_max: float = 1e-2,
    photjitter_max: float = 1e-2,
) -> Path:
    _ensure_cpu_jax_on_macos()
    _log_jax_backend()
    from uberMS.dva import runSVI

    sanitize_xla_flags_env()

    indict, spec_nn_list, phot_nn, mist_nn = build_ums_indict(
        gaia_id,
        fit_data,
        output_name=output_name,
        dospec=dospec,
        dophot=dophot,
        progressbar=progressbar,
        svi_guide=svi_guide,
        fix_vstar=fix_vstar,
        fix_photjitter=fix_photjitter,
        fix_specjitter=fix_specjitter,
        vrad_prior=vrad_prior,
        vrad_err_inflate=vrad_err_inflate,
        vrad_err_floor_kms=vrad_err_floor_kms,
        omit_parallax_likelihood=omit_parallax_likelihood,
        dist_uniform_min_pc=dist_uniform_min_pc,
        dist_uniform_max_pc=dist_uniform_max_pc,
        dust_av_prior=dust_av_prior,
        rigid_continuum=rigid_continuum,
        spec_err_scale=spec_err_scale,
        phot_err_scale=phot_err_scale,
        pc0_min=pc0_min,
        pc0_max=pc0_max,
        pc_poly_abs_max=pc_poly_abs_max,
        specjitter_max=specjitter_max,
        photjitter_max=photjitter_max,
    )

    outfile = Path(indict["outfile"])
    logger.info("Running UMS SVI → %s", outfile)
    svi = runSVI.sviMS(
        specNN=spec_nn_list,
        photNN=phot_nn,
        mistNN=mist_nn,
        verbose=True,
        usegrad=False,
    )
    svi.run(indict)
    return outfile


def run_utp(
    gaia_id: str,
    fit_data: dict,
    *,
    output_name: str | None = None,
    dospec: bool = True,
    dophot: bool = True,
    progressbar: bool = True,
    vrad_prior: str = "normal",
    vrad_err_inflate: float = 2.0,
    vrad_err_floor_kms: float = 2.0,
    omit_parallax_likelihood: bool = False,
    dist_uniform_min_pc: float | None = None,
    dist_uniform_max_pc: float | None = None,
    dust_av_prior: bool = True,
    spec_err_scale: float = 1.0,
    phot_err_scale: float = 1.0,
) -> Path:
    _ensure_cpu_jax_on_macos()
    _log_jax_backend()
    from uberMS.dva import runSVI

    sanitize_xla_flags_env()

    spec_nn, phot_nn, _mist = models.model_paths()
    fit_data = data.apply_likelihood_error_scales(
        fit_data,
        spec_err_scale=spec_err_scale,
        phot_err_scale=phot_err_scale,
    )

    indict: dict = {}
    outfile = _samples_outfile(gaia_id, "UTP", output_name)
    indict["outfile"] = str(outfile)

    indict["data"] = {}
    nspec = 0
    if dospec and "spec" in fit_data:
        indict["data"]["spec"] = fit_data["spec"]
        nspec = len(indict["data"]["spec"])
    if dophot:
        indict["data"]["phot"] = fit_data["phot"]
        indict["data"]["parallax"] = fit_data["parallax"]
        indict["data"]["filtarr"] = list(fit_data["phot_filtarr"])

    spec_nn_list = [spec_nn for _ in range(nspec)] if nspec > 0 else []

    dist_init, dist_prior = _dist_prior_tp(
        fit_data,
        omit_parallax_likelihood=omit_parallax_likelihood,
        dist_uniform_min_pc=dist_uniform_min_pc,
        dist_uniform_max_pc=dist_uniform_max_pc,
    )
    if omit_parallax_likelihood:
        indict["omit_parallax_likelihood"] = True

    initpars = {
        "Teff": fit_data["Teff"],
        "[Fe/H]": fit_data["[Fe/H]"],
        "[a/Fe]": fit_data["[a/Fe]"],
        "log(g)": fit_data["log(g)"],
        "log(R)": fit_data["log(R)"],
        "dist": dist_init,
        "Av": 0.1,
        "vmic": 1.0,
        "vstar": 2.0,
        "photjitter": 1e-5,
    }

    for ii in range(nspec):
        initpars[f"lsf_{ii}"] = 60000.0
        initpars[f"specjitter_{ii}"] = 1e-5
        initpars[f"pc0_{ii}"] = 1.0
        initpars[f"pc1_{ii}"] = 0.0
        initpars[f"pc2_{ii}"] = 0.0
        initpars[f"pc3_{ii}"] = 0.0

    indict["initpars"] = initpars
    indict["priors"] = {
        "Teff": ["uniform", [5500.0, 7000.0]],
        "log(g)": ["uniform", [2.0, 5.5]],
        "[Fe/H]": ["uniform", [-1.5, 0.5]],
        "[a/Fe]": ["uniform", [-0.9, 0.6]],
        "log(R)": ["uniform", [-1, 1]],
        "vstar": ["uniform", [0.0, 25.0]],
        "vmic": ["uniform", [0.5, 3.0]],
        "Av": ["tnormal", [0.0, 0.01, 0.0, 0.5]],
        "dist": dist_prior,
        "photjitter": ["fixed", 0.0],
    }
    _apply_av_prior(fit_data, initpars, indict["priors"], dust_av_prior=dust_av_prior)

    _apply_per_epoch_spec_priors(
        indict["priors"],
        nspec,
        fix_specjitter=False,
        rigid_continuum=False,
        pc0_min=0.95,
        pc0_max=1.05,
        pc_poly_abs_max=0.1,
        specjitter_max=1e-2,
    )

    rv_epochs = fit_data.get("rv_epochs") or []
    if nspec > 0:
        if len(rv_epochs) != nspec:
            rv_epochs = (rv_epochs + [priors.RvEpoch("", i, float("nan"), float("nan"), float("nan")) for i in range(nspec)])[:nspec]
        _apply_rv_priors_to_indict(
            indict,
            initpars,
            rv_epochs,
            vrad_prior=vrad_prior,
            vrad_err_inflate=vrad_err_inflate,
            vrad_err_floor_kms=vrad_err_floor_kms,
        )

    indict["svi"] = {
        "steps": 10000,
        "opt_tol": 1e-6,
        "start_tol": 1e-2,
        "progress_bar": progressbar,
        "post_resample": 30000,
        "guide": "Normal",
    }

    logger.info("Running UTP SVI → %s", outfile)
    svi = runSVI.sviTP(specNN=spec_nn_list, photNN=phot_nn, verbose=True)
    svi.run(indict)
    return outfile


def _kwargs_for(func, kwargs: dict) -> dict:
    """Pass only keyword arguments accepted by ``func`` (excludes gaia_id, fit_data)."""
    params = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def run_both(
    gaia_id: str,
    fit_data: dict,
    *,
    do_ums: bool = True,
    do_utp: bool = True,
    **kwargs,
) -> dict[str, Path]:
    """Run UMS (primary) and UTP (secondary); returns paths to sample FITS files."""
    results: dict[str, Path] = {}
    if do_ums:
        results["ums"] = run_ums(gaia_id, fit_data, **_kwargs_for(run_ums, kwargs))
    if do_utp:
        results["utp"] = run_utp(gaia_id, fit_data, **_kwargs_for(run_utp, kwargs))
    return results
