"""uberMS SVI drivers for UMS (primary) and UTP (secondary)."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from darkhunter_sed import data, models, priors
from darkhunter_sed.config import samples_dir

logger = logging.getLogger(__name__)


def _samples_outfile(gaia_id: str, runtype: str, output_name: str | None = None) -> Path:
    samples_dir().mkdir(parents=True, exist_ok=True)
    if output_name:
        return samples_dir() / output_name
    suffix = "ums" if runtype.upper() == "UMS" else "utp"
    return samples_dir() / f"Gaia_DR3_{gaia_id}_{suffix}.fits"


def preflight_ums(nspec: int, gaia_id: str) -> None:
    if nspec < 2:
        raise ValueError(
            f"UMS (uberMS.dva.sviMS) requires at least 2 spectra for {gaia_id}; got {nspec}. "
            "Collect another epoch or run UTP only."
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
) -> Path:
    from uberMS.dva import runSVI

    spec_nn, phot_nn, mist_nn = models.model_paths()
    fit_data = data.apply_likelihood_error_scales(fit_data)

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

    spec_nn_list = [spec_nn for _ in range(nspec)] if nspec > 0 else None

    dist_init, dist_prior = _dist_prior_ms_tp(
        fit_data,
        omit_parallax_likelihood=omit_parallax_likelihood,
        dist_uniform_min_pc=dist_uniform_min_pc,
        dist_uniform_max_pc=dist_uniform_max_pc,
    )
    if omit_parallax_likelihood:
        indict["omit_parallax_likelihood"] = True

    initpars = {
        "EEP": 300.0,
        "initial_Mass": fit_data["Mass"],
        "initial_[Fe/H]": 0.0,
        "initial_[a/Fe]": fit_data["[a/Fe]"],
        "dist": dist_init,
        "Av": 0.1,
        "vmic": 1.0,
        "vstar": 8.0,
        "photjitter": 1e-5 if not fix_photjitter else 0.0,
    }

    if dospec and nspec > 0:
        sj0 = 1e-5 if not fix_specjitter else 0.0
        for ii in range(nspec):
            initpars[f"lsf_{ii}"] = 60000.0
            initpars[f"specjitter_{ii}"] = sj0
            initpars[f"pc0_{ii}"] = 1.0
            initpars[f"pc1_{ii}"] = 0.0
            initpars[f"pc2_{ii}"] = 0.0
            initpars[f"pc3_{ii}"] = 0.0

    indict["initpars"] = initpars
    indict["priors"] = {
        "EEP": ["uniform", [250, 350]],
        "initial_Mass": ["IMF", {"mass_le": 0.3, "mass_ue": 1.25}],
        "initial_[Fe/H]": ["uniform", [-1.6, 0.5]],
        "initial_[a/Fe]": ["fixed", fit_data["[a/Fe]"]],
        "vstar": ["fixed", 8.0] if fix_vstar else ["uniform", [0.0, 25.0]],
        "vmic": ["fixed", 1.0],
        "Av": ["tnormal", [0.0, 0.1, 0.0, 0.5]],
        "dist": dist_prior,
        "photjitter": ["fixed", 0.0] if fix_photjitter else ["uniform", [1e-6, 1e-2]],
    }

    if dospec and nspec > 0:
        for ii in range(nspec):
            indict["priors"][f"lsf_{ii}"] = ["fixed", 60000.0]
            for pc in ("pc0", "pc1", "pc2", "pc3"):
                indict["priors"][f"{pc}_{ii}"] = ["fixed", 1.0 if pc == "pc0" else 0.0]
            indict["priors"][f"specjitter_{ii}"] = (
                ["fixed", 0.0] if fix_specjitter else ["uniform", [1e-6, 1e-2]]
            )

        rv_epochs = fit_data.get("rv_epochs") or []
        if len(rv_epochs) != nspec:
            logger.warning(
                "RV epoch count (%s) != spectrum count (%s); using default vrad priors for missing",
                len(rv_epochs),
                nspec,
            )
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
        "steps": 5000,
        "opt_tol": 1e-6,
        "start_tol": 1e-2,
        "progress_bar": progressbar,
        "post_resample": 4000,
        "guide": _svi_guide_key(svi_guide),
    }

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
) -> Path:
    from uberMS.dva import runSVI

    spec_nn, phot_nn, _mist = models.model_paths()
    fit_data = data.apply_likelihood_error_scales(fit_data)

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

    for ii in range(nspec):
        indict["priors"][f"lsf_{ii}"] = ["tnormal", [60000.0, 500.0, 50000.0, 65000.0]]
        for pc, bounds in (
            ("pc0", [0.95, 1.05]),
            ("pc1", [-0.1, 0.1]),
            ("pc2", [-0.1, 0.1]),
            ("pc3", [-0.1, 0.1]),
        ):
            indict["priors"][f"{pc}_{ii}"] = ["uniform", bounds]
        indict["priors"][f"specjitter_{ii}"] = ["uniform", [1e-6, 1e-2]]

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
        results["ums"] = run_ums(gaia_id, fit_data, **kwargs)
    if do_utp:
        results["utp"] = run_utp(gaia_id, fit_data, **kwargs)
    return results
