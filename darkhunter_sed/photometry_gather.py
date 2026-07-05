import argparse
import logging
import math
import os
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astroquery.gaia import Gaia

from darkhunter_sed import stellar_data
from astroquery.ipac.irsa import Irsa
from astroquery.sdss import SDSS
from astroquery.mast import Catalogs
from astroquery.vizier import Vizier
import astropy.units as u
import astropy.coordinates as coord

# PS1 mean-object astrometry / stack epochs cluster roughly 2010–2013 (survey 2010–2014).
# Gaia positions are at ``ref_epoch`` (often ~2016). For Vizier cone search, propagating
# to ~2011 matches the PS1 observation era better than J2000.0 (which is only equinox).
PS1_VIZIER_DEFAULT_EPOCH_JY = 2011.0

logger = logging.getLogger(__name__)


def good_number_checker(number):
    if number is None:
        return False
    try:
        if hasattr(number, "mask") and np.ma.getmask(number):
            return False
    except Exception:
        pass
    return (
        isinstance(number, (int, float, np.floating))
        and math.isfinite(float(number))
        and not math.isnan(float(number))
    )


def mag_error(flux, err):
    return 2.5 * err / flux / np.log(10)


def _gaia_scalar(row, col):
    """Single-row Gaia / Vizier value -> float or NaN (handles masked columns)."""
    if col not in row.colnames:
        return float("nan")
    v = row[col]
    if v is np.ma.masked:
        return float("nan")
    if hasattr(v, "mask") and np.ma.getmask(v):
        return float("nan")
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _ps1_archive_mag_pair(mag, err, default_err=0.05):
    """
    Pan-STARRS magnitudes in ``gaiadr2.panstarrs1_original_valid`` use sentinels
    (e.g. -999) when undefined; errors may be null in the archive.
    """
    if mag is None or (isinstance(mag, float) and not math.isfinite(mag)):
        return None
    try:
        m = float(mag)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(m) or m < -90.0 or m > 90.0:
        return None
    e = None
    if err is not None and err is not np.ma.masked:
        if not (hasattr(err, "mask") and np.ma.getmask(err)):
            try:
                e = float(err)
            except (TypeError, ValueError):
                e = None
    if e is None or not math.isfinite(e) or e <= 0.0:
        e = float(default_err)
    return (m, e)


def _append_ps1_from_gaia_archive(row, photometry):
    """
    Prefer PS1 mean PSF AB mags from ``gaiadr2.panstarrs1_original_valid`` via
    ``gaiadr3.panstarrs1_best_neighbour`` (Gaia DR3 cross-match).
    """
    bands = [
        ("PS_g", "g_mean_psf_mag", "g_mean_psf_mag_error"),
        ("PS_r", "r_mean_psf_mag", "r_mean_psf_mag_error"),
        ("PS_i", "i_mean_psf_mag", "i_mean_psf_mag_error"),
        ("PS_z", "z_mean_psf_mag", "z_mean_psf_mag_error"),
        ("PS_y", "y_mean_psf_mag", "y_mean_psf_mag_error"),
    ]
    n = 0
    for out_name, c_mag, c_err in bands:
        m = _gaia_scalar(row, c_mag)
        err_cell = row[c_err] if c_err in row.colnames else None
        pair = _ps1_archive_mag_pair(m if math.isfinite(m) else None, err_cell)
        if pair is None:
            continue
        photometry.append((out_name, pair[0], pair[1]))
        n += 1
    return n


def skycoord_ps1_cone_search(row, target_epoch_jyear=PS1_VIZIER_DEFAULT_EPOCH_JY):
    """
    Sky position for a Vizier cone search against PS1 (II/349). Catalogue entries use
    ICRS equinox J2000, but mean coordinates are at each object's observation/stack
    epoch (typically ~2010–2013). Gaia positions are at ``ref_epoch`` (DR3: often
    ~2016). Propagate proper motion from Gaia's epoch to ``target_epoch_jyear``
    (default :data:`PS1_VIZIER_DEFAULT_EPOCH_JY`, ~2011) so cone matches align with
    where PS1 data were taken; use 2000.0 only if you intentionally want J2000.0.

    Radial velocity is omitted when unknown (tangential propagation only).
    """
    ra = _gaia_scalar(row, "ra")
    dec = _gaia_scalar(row, "dec")
    pmra = _gaia_scalar(row, "pmra")
    pmdec = _gaia_scalar(row, "pmdec")
    ref_ep = _gaia_scalar(row, "ref_epoch")
    if not math.isfinite(ref_ep):
        ref_ep = 2016.0

    if not (math.isfinite(ra) and math.isfinite(dec)):
        return coord.SkyCoord(0.0, 0.0, unit=u.deg, frame="icrs")

    if not (math.isfinite(pmra) and math.isfinite(pmdec)):
        return coord.SkyCoord(ra, dec, unit=u.deg, frame="icrs")

    try:
        t0 = Time(ref_ep, format="jyear")
        t1 = Time(float(target_epoch_jyear), format="jyear")
        sc = coord.SkyCoord(
            ra=ra * u.deg,
            dec=dec * u.deg,
            pm_ra_cosdec=pmra * u.mas / u.yr,
            pm_dec=pmdec * u.mas / u.yr,
            obstime=t0,
            frame="icrs",
        )
        return sc.apply_space_motion(new_obstime=t1)
    except Exception:
        return coord.SkyCoord(ra, dec, unit=u.deg, frame="icrs")


def query_catalogs(source_id, radius=3, ps1_vizier_epoch_jyear=PS1_VIZIER_DEFAULT_EPOCH_JY):
    """
    Cross-match photometry: Gaia DR3 G/BP/RP, external catalogs, and Pan-STARRS.

    Pan-STARRS: when ``gaiadr3.panstarrs1_best_neighbour`` joins to
    ``gaiadr2.panstarrs1_original_valid``, use archive PSF AB mags from that row.
    Otherwise query Vizier II/349 at the Gaia position propagated from
    ``ref_epoch`` to ``ps1_vizier_epoch_jyear`` (default ~2011, PS1-era).
    """
    radius = coord.Angle(radius, "arcsec")
    photometry = []

    # Two-step Gaia queries: a single JOIN to panstarrs1_original_valid often hits
    # archive timeouts; neighbour + PS1 row is a fast indexed lookup by source_id.
    job = Gaia.launch_job(
        f"""
        SELECT TOP 10 ra, dec, ref_epoch, pmra, pmdec,
               phot_g_mean_mag, phot_g_mean_flux, phot_g_mean_flux_error,
               phot_bp_mean_mag, phot_bp_mean_flux, phot_bp_mean_flux_error,
               phot_rp_mean_mag, phot_rp_mean_flux, phot_rp_mean_flux_error
        FROM gaiadr3.gaia_source
        WHERE source_id = {source_id}
        """
    )
    gaia_data = job.get_results()
    if len(gaia_data) == 0:
        print(photometry)
        return photometry

    row0 = gaia_data[0]

    job_ps = Gaia.launch_job(
        f"""
        SELECT TOP 10 ps.g_mean_psf_mag, ps.g_mean_psf_mag_error,
               ps.r_mean_psf_mag, ps.r_mean_psf_mag_error,
               ps.i_mean_psf_mag, ps.i_mean_psf_mag_error,
               ps.z_mean_psf_mag, ps.z_mean_psf_mag_error,
               ps.y_mean_psf_mag, ps.y_mean_psf_mag_error
        FROM gaiadr3.panstarrs1_best_neighbour AS nb
        INNER JOIN gaiadr2.panstarrs1_original_valid AS ps
            ON nb.original_ext_source_id = ps.obj_id
        WHERE nb.source_id = {source_id}
        """
    )
    ps_join = job_ps.get_results()
    row_ps1 = ps_join[0] if len(ps_join) else None

    photometry.append(
        (
            "GaiaDR3_G",
            row0["phot_g_mean_mag"],
            mag_error(
                row0["phot_g_mean_flux"],
                row0["phot_g_mean_flux_error"],
            ),
        )
    )
    photometry.append(
        (
            "GaiaDR3_BP",
            row0["phot_bp_mean_mag"],
            mag_error(
                row0["phot_bp_mean_flux"],
                row0["phot_bp_mean_flux_error"],
            ),
        )
    )
    photometry.append(
        (
            "GaiaDR3_RP",
            row0["phot_rp_mean_mag"],
            mag_error(
                row0["phot_rp_mean_flux"],
                row0["phot_rp_mean_flux_error"],
            ),
        )
    )

    ra, dec = float(row0["ra"]), float(row0["dec"])
    position = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame="icrs")

    Irsa.ROW_LIMIT = 1
    tmass_data = Irsa.query_region(position, catalog="fp_psc", radius=radius)
    if len(tmass_data) > 0:
        jm, jme = tmass_data["j_m"][0], tmass_data["j_msigcom"][0]
        hm, hme = tmass_data["h_m"][0], tmass_data["h_msigcom"][0]
        km, kme = tmass_data["k_m"][0], tmass_data["k_msigcom"][0]
        if good_number_checker(jm) and good_number_checker(jme):
            photometry.append(("2MASS_J", float(jm), float(jme)))
        if good_number_checker(hm) and good_number_checker(hme):
            photometry.append(("2MASS_H", float(hm), float(hme)))
        if good_number_checker(km) and good_number_checker(kme):
            photometry.append(("2MASS_Ks", float(km), float(kme)))

    wise_data = Irsa.query_region(position, catalog="allwise_p3as_psd", radius=radius)
    if len(wise_data) > 0:
        photometry.append(("WISE_W1", wise_data["w1mpro"][0], wise_data["w1sigmpro"][0]))
        photometry.append(("WISE_W2", wise_data["w2mpro"][0], wise_data["w2sigmpro"][0]))

    sdss_data = SDSS.query_crossid(
        position,
        photoobj_fields=[
            "modelMag_u",
            "modelMag_g",
            "modelMag_r",
            "modelMag_i",
            "modelMag_z",
            "modelMagErr_u",
            "modelMagErr_g",
            "modelMagErr_r",
            "modelMagErr_i",
            "modelMagErr_z",
        ],
    )
    if sdss_data is not None and len(sdss_data) > 0:
        photometry.append(("SDSS_u", sdss_data["modelMag_u"][0], sdss_data["modelMagErr_u"][0]))
        photometry.append(("SDSS_g", sdss_data["modelMag_g"][0], sdss_data["modelMagErr_g"][0]))
        photometry.append(("SDSS_r", sdss_data["modelMag_r"][0], sdss_data["modelMagErr_r"][0]))
        photometry.append(("SDSS_i", sdss_data["modelMag_i"][0], sdss_data["modelMagErr_i"][0]))
        photometry.append(("SDSS_z", sdss_data["modelMag_z"][0], sdss_data["modelMagErr_z"][0]))

    galex_data = Catalogs.query_region(position, catalog="GALEX", radius=radius)
    if len(galex_data) > 0:
        if good_number_checker(galex_data["fuv_mag"][0]) and good_number_checker(
            galex_data["fuv_magerr"][0]
        ):
            photometry.append(("GALEX_FUV", galex_data["fuv_mag"][0], galex_data["fuv_magerr"][0]))
        if good_number_checker(galex_data["nuv_mag"][0]) and good_number_checker(
            galex_data["nuv_magerr"][0]
        ):
            photometry.append(("GALEX_NUV", galex_data["nuv_mag"][0], galex_data["nuv_magerr"][0]))

    n_ps1_arch = _append_ps1_from_gaia_archive(row_ps1, photometry) if row_ps1 is not None else 0
    if n_ps1_arch > 0:
        print(
            "Pan-STARRS: {} bands from Gaia archive "
            "(gaiadr3.panstarrs1_best_neighbour + gaiadr2.panstarrs1_original_valid).".format(
                n_ps1_arch
            )
        )
    else:
        ps1_search_pos = skycoord_ps1_cone_search(row0, target_epoch_jyear=ps1_vizier_epoch_jyear)
        ref_ep = _gaia_scalar(row0, "ref_epoch")
        if not math.isfinite(ref_ep):
            ref_ep = 2016.0
        print(
            "Pan-STARRS: no archive join row; Vizier II/349 cone search at "
            "RA, Dec propagated from Gaia ref_epoch={:.3f} to J-year {:.3f}.".format(
                ref_ep, float(ps1_vizier_epoch_jyear)
            )
        )
        Vizier.ROW_LIMIT = 1
        ps1_catalog = "II/349/ps1"
        ps1_data = Vizier.query_region(ps1_search_pos, catalog=ps1_catalog, radius=radius)
        if ps1_data is not None and ps1_catalog in ps1_data.keys():
            ps1_table = ps1_data[ps1_catalog]
            if good_number_checker(ps1_table["gmag"][0]) and good_number_checker(
                ps1_table["e_gmag"][0]
            ):
                photometry.append(("PS_g", ps1_table["gmag"][0], ps1_table["e_gmag"][0]))
            if good_number_checker(ps1_table["rmag"][0]) and good_number_checker(
                ps1_table["e_rmag"][0]
            ):
                photometry.append(("PS_r", ps1_table["rmag"][0], ps1_table["e_rmag"][0]))
            if good_number_checker(ps1_table["imag"][0]) and good_number_checker(
                ps1_table["e_imag"][0]
            ):
                photometry.append(("PS_i", ps1_table["imag"][0], ps1_table["e_imag"][0]))
            if good_number_checker(ps1_table["zmag"][0]) and good_number_checker(
                ps1_table["e_zmag"][0]
            ):
                photometry.append(("PS_z", ps1_table["zmag"][0], ps1_table["e_zmag"][0]))
            if good_number_checker(ps1_table["ymag"][0]) and good_number_checker(
                ps1_table["e_ymag"][0]
            ):
                photometry.append(("PS_y", ps1_table["ymag"][0], ps1_table["e_ymag"][0]))

    print(photometry)
    return photometry


def save_photometry_to_fits(source_id, photometry, outdir=None):
    # Wider string column for band names (e.g. GaiaDR3_BP, GALEX_NUV)
    data = np.array(photometry, dtype=[("band", "S32"), ("mag", "d"), ("err", "d")])

    hdu_primary = fits.PrimaryHDU()
    hdu_table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="band", format="32A", array=data["band"]),
            fits.Column(name="mag", format="D", array=data["mag"]),
            fits.Column(name="err", format="D", array=data["err"]),
        ]
    )

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        output_filename = os.path.join(outdir, f"{source_id}_phot.fits")
    else:
        output_filename = f"{source_id}_phot.fits"

    fits.HDUList([hdu_primary, hdu_table]).writeto(output_filename, overwrite=True)
    print(f"Saved: {output_filename}")
    return Path(output_filename)


def gather_photometry_for_star(
    source_id: str,
    *,
    outdir: str | os.PathLike | None = None,
    ps1_radius: float = 3.0,
    ps1_vizier_epoch_jyear: float | None = None,
    phot_outlier_sigma: float | None = None,
    phot_outlier_min_kept: int = 3,
    phot_outlier_model: str = "blackbody",
) -> Path:
    """Query catalogs and write ``{source_id}_phot.fits``; returns output path."""
    photometry = query_catalogs(
        source_id,
        radius=ps1_radius,
        ps1_vizier_epoch_jyear=ps1_vizier_epoch_jyear,
    )
    if phot_outlier_sigma is not None and phot_outlier_sigma > 0:
        phot_d = {b: [float(m), float(e)] for b, m, e in photometry}
        filt = stellar_data.ordered_phot_filtarr_from_dict(phot_d)
        phot_d, filt, dropped = stellar_data.iterative_photometry_outlier_rejection(
            phot_d,
            filt,
            sigma=float(phot_outlier_sigma),
            min_kept=max(3, int(phot_outlier_min_kept)),
            method=str(phot_outlier_model),
        )
        if dropped:
            warnings.warn(
                f"gather_phot: dropped photometry outlier bands: {dropped}",
                UserWarning,
                stacklevel=2,
            )
        photometry = [(b, phot_d[b][0], phot_d[b][1]) for b in filt]
    return save_photometry_to_fits(source_id, photometry, outdir=outdir)


def ensure_photometry_fits(
    gaia_id: str,
    phot_dir: str | os.PathLike | None = None,
    *,
    auto_gather: bool = True,
) -> Path:
    """
    Return path to ``{gaia_id}_phot.fits``, gathering from catalogs when missing.

    When ``auto_gather`` is False and the file is absent, raises FileNotFoundError
    with a hint to run ``python -m darkhunter_sed.photometry_gather``.
    """
    from darkhunter_sed.config import photometry_dir as default_phot_dir

    dir_path = Path(phot_dir) if phot_dir is not None else default_phot_dir()
    path = Path(stellar_data.photometry_fits_path(gaia_id, dir_path))
    if path.is_file():
        return path.resolve()
    if not auto_gather:
        raise FileNotFoundError(
            f"Photometry file not found: {path}\n"
            f"Run: python -m darkhunter_sed.photometry_gather {gaia_id} -d {dir_path}"
        )
    logger.info("Gathering photometry for Gaia %s (network query)", gaia_id)
    return gather_photometry_for_star(gaia_id, outdir=dir_path)


def main():
    parser = argparse.ArgumentParser(description="Gather multi-band photometry for a Gaia source_id.")
    parser.add_argument("source_id", help="Gaia DR3 source_id")
    parser.add_argument(
        "--outdir",
        "-d",
        default=None,
        help="Directory for {source_id}_phot.fits (default: current directory)",
    )
    parser.add_argument(
        "--ps1-radius",
        type=float,
        default=3.0,
        help="Cone radius (arcsec) for Vizier PS1 fallback when Gaia archive join is missing",
    )
    parser.add_argument(
        "--ps1-vizier-epoch",
        type=float,
        default=PS1_VIZIER_DEFAULT_EPOCH_JY,
        help=(
            "Julian year to propagate Gaia astrometry to before Vizier PS1 fallback "
            "(default {:.1f} ≈ PS1 survey mean; try 2000.0 for J2000.0-only matching)."
        ).format(PS1_VIZIER_DEFAULT_EPOCH_JY),
    )
    parser.add_argument(
        "--phot-outlier-sigma",
        type=float,
        default=None,
        help=(
            "If set (>0), drop bands deviating from iterative mag vs log10(λ) fit before "
            "writing FITS (same logic as run_uber.py; typical 3)."
        ),
    )
    parser.add_argument(
        "--phot-outlier-min-kept",
        type=int,
        default=3,
        help="Minimum bands with known λ to keep when using --phot-outlier-sigma.",
    )
    parser.add_argument(
        "--phot-outlier-model",
        choices=("blackbody", "linear"),
        default="blackbody",
        help="Outlier model when --phot-outlier-sigma is set (default blackbody).",
    )
    args = parser.parse_args()
    gather_photometry_for_star(
        args.source_id,
        outdir=args.outdir,
        ps1_radius=args.ps1_radius,
        ps1_vizier_epoch_jyear=args.ps1_vizier_epoch,
        phot_outlier_sigma=args.phot_outlier_sigma,
        phot_outlier_min_kept=args.phot_outlier_min_kept,
        phot_outlier_model=args.phot_outlier_model,
    )


if __name__ == "__main__":
    main()
