"""Assemble photometry, stellar priors, and spectra for uberMS."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np

from darkhunter_sed import priors, spectrum
from darkhunter_sed.config import rv_output_dir
from darkhunter_sed.region_picker import resolve_regions_json_for_star
from darkhunter_sed.stellar_data import (
    iterative_photometry_outlier_rejection,
    load_photometry_fits,
    normalize_phot_nn_dir,
    require_phot_nn_for_bands,
    restrict_photometry_to_phot_nn_bands,
)

logger = logging.getLogger(__name__)

_GAIA_DR3_BANDS = frozenset({"GaiaDR3_G", "GaiaDR3_BP", "GaiaDR3_RP"})


def _finalize_parallax_in_out(
    out: dict,
    *,
    parallax_mas: float | None = None,
    parallax_error_mas: float | None = None,
    parallax_err_floor_mas: float | None = None,
    parallax_err_mult: float = 1.0,
    parallax_distance_factor: float = 1.0,
) -> None:
    pl0, pl1 = out["parallax"]
    if parallax_mas is not None:
        pl0 = float(parallax_mas)
    else:
        fac = float(parallax_distance_factor)
        if fac <= 0.0:
            raise ValueError("parallax_distance_factor must be positive")
        if fac != 1.0:
            pl0 *= fac
    if parallax_error_mas is not None:
        pl1 = float(parallax_error_mas)
    else:
        pl1 *= float(parallax_err_mult)
        if parallax_err_floor_mas is not None:
            pl1 = max(pl1, float(parallax_err_floor_mas))
    if pl1 <= 0.0:
        raise ValueError(f"Parallax uncertainty must be positive (got {pl1})")
    out["parallax"] = [pl0, pl1]


def apply_photometry_error_floors(
    phot: dict,
    *,
    default_floor_mag: float = 0.02,
    gaia_floor_mag: float | None = None,
) -> dict:
    """
    Raise catalog magnitude uncertainties to usable floors before SVI.

    Gaia DR3 formal errors (~0.0002 mag) are far too small for broadband SED
    fitting and force ``photjitter`` to the prior ceiling without improving the
    model. Floors apply per band after any outlier rejection.
    """
    floor = float(default_floor_mag)
    if floor <= 0.0:
        return phot
    g_floor = floor if gaia_floor_mag is None else float(gaia_floor_mag)
    if g_floor <= 0.0:
        raise ValueError("gaia_floor_mag must be positive when set")
    out: dict = {}
    for band, (mag, err) in phot.items():
        band_floor = g_floor if band in _GAIA_DR3_BANDS else floor
        out[band] = [float(mag), max(float(err), band_floor)]
    return out


def apply_likelihood_error_scales(
    data: dict,
    *,
    spec_err_scale: float = 1.0,
    phot_err_scale: float = 1.0,
) -> dict:
    se = float(spec_err_scale)
    pe = float(phot_err_scale)
    if se <= 0.0 or pe <= 0.0:
        raise ValueError("spec_err_scale and phot_err_scale must be positive")
    if se == 1.0 and pe == 1.0:
        return data
    out = dict(data)
    if "spec" in out and se != 1.0:
        out["spec"] = [
            {**sp, "obs_eflux": np.asarray(sp["obs_eflux"], dtype=float) * se}
            for sp in out["spec"]
        ]
    if "phot" in out and pe != 1.0:
        out["phot"] = {
            band: [float(mag), float(err) * pe] for band, (mag, err) in out["phot"].items()
        }
    return out


def load_spectra_from_paths(
    paths: list[Path],
    *,
    from_fits: bool = False,
    regions_json: str | Path | None = None,
) -> list[dict]:
    specs: list[dict] = []
    for p in spectrum.sort_spectrum_paths(paths):
        if from_fits or p.suffix.lower() in (".fits", ".fit"):
            specs.append(spectrum.process_spectrum_fits(p))
        else:
            kwargs: dict = {}
            if regions_json is not None:
                kwargs["regions_json"] = regions_json
            specs.append(spectrum.process_apf_txt_spectrum(p, **kwargs))
    return specs


def getdata(
    gaia_id: str,
    *,
    spectrum_paths: list[Path] | None = None,
    photometry_dir: str | Path | None = None,
    phot_nn: str,
    require_phot_nn: bool = True,
    phot_outlier_sigma: float | None = None,
    phot_outlier_min_kept: int = 3,
    phot_outlier_method: str = "blackbody",
    phot_err_floor_mag: float = 0.02,
    gaia_phot_err_floor_mag: float | None = None,
    parallax_mas: float | None = None,
    parallax_error_mas: float | None = None,
    parallax_err_floor_mas: float | None = None,
    parallax_err_mult: float = 1.0,
    parallax_distance_factor: float = 1.0,
    force_redownload: bool = False,
    summary_path: Path | None = None,
    from_fits: bool = False,
    regions_json: str | Path | None = None,
    auto_resolve_regions: bool = True,
    auto_gather_photometry: bool = True,
) -> dict:
    """Build uberMS input data dict (phot, spec, stellar priors, RV epoch list)."""
    from darkhunter_sed.photometry_gather import ensure_photometry_fits

    ensure_photometry_fits(
        gaia_id,
        photometry_dir,
        auto_gather=auto_gather_photometry,
    )
    phot, phot_filtarr = load_photometry_fits(gaia_id, photometry_dir)

    resolved_regions = resolve_regions_json_for_star(
        gaia_id,
        regions_json,
        auto_resolve=auto_resolve_regions,
    )
    if resolved_regions is not None:
        logger.info("Using regions JSON %s for Gaia %s", resolved_regions, gaia_id)

    out = priors.load_stellar_priors(
        gaia_id,
        summary_path=summary_path,
        force_redownload=force_redownload,
    )
    out["phot"] = phot
    out["phot_filtarr"] = phot_filtarr
    if resolved_regions is not None:
        out["regions_json"] = str(resolved_regions)

    # APF reduced spectra are already barycentric-corrected; no Vhelio shift applied.
    out["Vhelio"] = None

    rv_epochs: list[priors.RvEpoch] = []
    summary = summary_path or priors.resolve_summary_for_star(gaia_id, rv_output_dir())
    if summary is not None and summary.is_file():
        rv_epochs = priors.parse_rv_epochs_from_summary(summary)

    if spectrum_paths:
        sorted_paths = spectrum.sort_spectrum_paths(list(spectrum_paths))
        regions_arg = str(resolved_regions) if resolved_regions is not None else None
        out["spec"] = load_spectra_from_paths(
            sorted_paths,
            from_fits=from_fits,
            regions_json=regions_arg,
        )
        out["spectrum_paths"] = sorted_paths
        if rv_epochs:
            out["rv_epochs"] = priors.match_rv_epochs_to_spectrum_paths(rv_epochs, sorted_paths)
        else:
            out["rv_epochs"] = []

    if require_phot_nn:
        phot, phot_filtarr = restrict_photometry_to_phot_nn_bands(phot_nn, phot, phot_filtarr)
        if phot_outlier_sigma is not None and float(phot_outlier_sigma) > 0:
            phot, phot_filtarr, dropped = iterative_photometry_outlier_rejection(
                phot,
                phot_filtarr,
                sigma=float(phot_outlier_sigma),
                min_kept=int(phot_outlier_min_kept),
                method=str(phot_outlier_method),
            )
            if dropped:
                warnings.warn(
                    f"Photometry outlier rejection dropped bands: {dropped}",
                    UserWarning,
                    stacklevel=2,
                )
        phot = apply_photometry_error_floors(
            phot,
            default_floor_mag=phot_err_floor_mag,
            gaia_floor_mag=gaia_phot_err_floor_mag,
        )
        out["phot"] = phot
        out["phot_filtarr"] = phot_filtarr
        require_phot_nn_for_bands(phot_nn, phot_filtarr)

    _finalize_parallax_in_out(
        out,
        parallax_mas=parallax_mas,
        parallax_error_mas=parallax_error_mas,
        parallax_err_floor_mas=parallax_err_floor_mas,
        parallax_err_mult=parallax_err_mult,
        parallax_distance_factor=parallax_distance_factor,
    )
    return out
