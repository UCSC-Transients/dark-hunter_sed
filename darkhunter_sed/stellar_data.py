"""
Load photometry FITS and query Gaia DR3 for priors (parallax with NSS fallbacks).

Used by darkhunter_sed fit driver and post-processing.

Optional :func:`iterative_photometry_outlier_rejection` drops bands that deviate from a
blackbody :math:`F_\\lambda \\propto B_\\lambda(T)` (default) or a legacy mag vs
``log10(λ)`` line, using uberMS ``photsys`` wavelengths and zeropoints.
"""
from __future__ import annotations

import math
import os
import re
import sys
import warnings
from io import BytesIO
from pathlib import Path


def _sanitize_xla_flags_env() -> None:
    """
    Some shells / conda envs set XLA_FLAGS=--xla_cpu_use_thunk_runtime=false for older
    JAX CPU builds. Current XLA (including Apple Metal) aborts on unknown flags:
    ``Unknown flags in XLA_FLAGS: --xla_cpu_use_thunk_runtime=false``. Strip it before
    any library (Payne, uberMS) imports JAX.

    If this token appears anywhere in XLA_FLAGS, drop the whole variable: Metal XLA
    does not accept the flag even when mixed with other tokens, and stripping is easy
    to get wrong across shells (quotes, repeated flags).
    """
    raw = os.environ.get("XLA_FLAGS", "").strip()
    if not raw:
        return
    if "xla_cpu_use_thunk_runtime" in raw.lower():
        os.environ.pop("XLA_FLAGS", None)
        return
    cleaned = re.sub(r"--xla_cpu_use_thunk_runtime(?:=\S*)?", "", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned:
        os.environ["XLA_FLAGS"] = cleaned
    else:
        os.environ.pop("XLA_FLAGS", None)


_sanitize_xla_flags_env()


def _silence_multiprocessing_semaphore_shutdown_noise() -> None:
    """JAX / XLA on macOS often triggers a harmless resource_tracker warning at exit."""
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r"resource_tracker: There appear to be \d+ leaked semaphore objects",
        category=UserWarning,
    )


_silence_multiprocessing_semaphore_shutdown_noise()

import numpy as np
import requests
from astropy.io.votable import parse as votable_parse
from astropy.table import Table
from astroquery.gaia import Gaia

# TAP sync URL for fallback when ESAC ``gea.esac.esa.int`` is degraded (POST + Accept */*).
_DEFAULT_GAIA_FALLBACK_TAP_SYNC = "https://gaia.aip.de/tap/sync"

# Legacy gather_phot labels -> canonical uberMS / photNN names
BAND_ALIASES = {
    "GaiaEDR3_G": "GaiaDR3_G",
    "GaiaEDR3_BP": "GaiaDR3_BP",
    "GaiaEDR3_RP": "GaiaDR3_RP",
    # Pan-STARRS from gather_phot vs uberMS photsys / photNN names
    "PS1_g": "PS_g",
    "PS1_r": "PS_r",
    "PS1_i": "PS_i",
    "PS1_z": "PS_z",
    "PS1_y": "PS_y",
}

PREFERRED_BAND_ORDER = [
    "GaiaDR3_G",
    "GaiaDR3_BP",
    "GaiaDR3_RP",
    "2MASS_J",
    "2MASS_H",
    "2MASS_Ks",
    "WISE_W1",
    "WISE_W2",
    "SDSS_u",
    "SDSS_g",
    "SDSS_r",
    "SDSS_i",
    "SDSS_z",
    "GALEX_FUV",
    "GALEX_NUV",
    "PS_g",
    "PS_r",
    "PS_i",
    "PS_z",
    "PS_y",
]


def repo_root() -> Path:
    """Stellar stack root (uberMS, ThePayne, models)."""
    from darkhunter_sed.config import stellar_root

    return stellar_root()


def ensure_stellar_sys_path(root: Path | None = None) -> Path:
    """
    Insert sibling package dirs (uberMS, MISTy, MINESweeper, ThePayne) onto sys.path.

    The uberMS / Payne jax stack expects the **Payne** package from Phillip Cargile's
    ThePayne tree (folder ``ThePayne`` with subpackage ``Payne/``). That is not the same
    as tingyuansen's ``The_Payne`` repo (package ``The_Payne``).

    Optional: set ``THEPAYNE_ROOT`` or ``PAYNE_PACKAGE_ROOT`` to the directory that
    **contains** the ``Payne`` package (i.e. the parent of ``Payne/``).
    """
    root = root or repo_root()
    for name in ("uberMS", "MISTy", "MINESweeper", "ThePayne"):
        p = root / name
        if p.is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(1, s)
    payne_parent = os.environ.get("THEPAYNE_ROOT") or os.environ.get("PAYNE_PACKAGE_ROOT")
    if payne_parent:
        pp = Path(payne_parent).resolve()
        if (pp / "Payne").is_dir():
            s = str(pp)
            if s not in sys.path:
                sys.path.insert(1, s)
    return root


def import_airtovacuum():
    """
    Wavelength conversion used when reading spectra: prefer Payne.jax (ThePayne),
    else the copy shipped with MINESweeper.
    """
    try:
        from Payne.jax.fitutils import airtovacuum as _fn

        return _fn
    except ImportError as err:
        try:
            from minesweeper.jax.fitutils import airtovacuum as _fn

            return _fn
        except ImportError as err2:
            raise ImportError(
                "Could not import airtovacuum. Install Phillip Cargile's ThePayne "
                "(folder ThePayne with Payne.jax.fitutils), or MINESweeper. "
                "tingyuansen/The_Payne is a different package (The_Payne, not Payne). "
                "Set THEPAYNE_ROOT to the directory containing the Payne package, "
                "or clone ThePayne next to gaia/. "
            ) from err2


def import_payne_genmod():
    """Neural spectral model GenMod: Payne.jax first, else minesweeper.jax."""
    try:
        from Payne.jax.genmod import GenMod

        return GenMod
    except ImportError as err:
        try:
            from minesweeper.jax.genmod import GenMod

            return GenMod
        except ImportError as err2:
            raise ImportError(
                "Could not import GenMod from Payne.jax or minesweeper.jax. "
                "Use ThePayne (Cargile) with package Payne, or set THEPAYNE_ROOT."
            ) from err2


def standardize_band_name(raw_band) -> str:
    if isinstance(raw_band, bytes):
        raw_band = raw_band.decode("ascii", errors="ignore").strip()
    else:
        raw_band = str(raw_band).strip()
    return BAND_ALIASES.get(raw_band, raw_band)


def resolve_models_dir(root: Path | None = None) -> Path:
    """
    Directory that contains ``specNN/``, ``photNN/``, ``mistNN/``.

    Checks ``DARKHUNTER_SED_MODELS_DIR``, ``<stellar>/gaia/models``, then ``<stellar>/models``.
    """
    from darkhunter_sed.config import models_dir

    root = root or repo_root()
    env_models = models_dir()
    candidates = [env_models, root / "gaia" / "models", root / "models"]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    tried = "\n  ".join(str(c.resolve()) for c in candidates)
    raise FileNotFoundError(
        "No models directory found. Tried:\n  "
        f"{tried}\n"
        "Place weights under gaia/models/ or <stellar>/models/ (specNN, photNN, mistNN)."
    )


def resolve_samples_fits_path(samplespath: str) -> str:
    """
    Locate posterior sample FITS: absolute path, cwd, or ./samples/<basename>.
    """
    p = Path(samplespath).expanduser()
    if p.is_file():
        return str(p.resolve())
    name = p.name
    candidates = [
        Path.cwd() / name,
        Path.cwd() / "samples" / name,
        Path.cwd() / samplespath,
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    tried = ", ".join(str(x) for x in candidates)
    raise FileNotFoundError(
        f"Samples FITS not found: {samplespath!r}. Checked: {tried}"
    )


def resolve_spec_nn_path(root: Path | None = None) -> str:
    """
    Path to the R65K Payne spectral NN used by uberMS / ppUMS.

    Prefer ``wvt2.h5`` (same default as ``run_uber.py``); fall back to ``wvt.h5`` if
    only that file is present.
    """
    root = root or repo_root()
    spec_dir = resolve_models_dir(root) / "specNN"
    candidates = [
        spec_dir / "modV0_spec_LinNet_R65K_WL515_530_wvt2.h5",
        spec_dir / "modV0_spec_LinNet_R65K_WL515_530_wvt.h5",
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Spectral NN weights not found under <models>/specNN. Tried:\n  "
        f"{tried}\n"
        "Install the matching Payne model files (see uberMS / ThePayne docs)."
    )


def normalize_phot_nn_dir(nnpath: str | os.PathLike) -> str:
    """
    Return ``photNN`` path with a trailing path separator.

    Payne's JAX ``photANN.ANN`` builds HDF5 paths as
    ``self.nnpath + 'nnMIST_{band}.h5'`` (no slash between). Using
    ``str(pathlib.Path(...) / 'photNN/')`` drops the trailing slash, producing
    ``.../photNNnnMIST_*.h5`` so every load fails and ``FastPayneSEDPredict`` ends
    with an empty ``nnlist`` / ``IndexError``.
    """
    p = Path(nnpath).expanduser().resolve()
    s = os.fspath(p)
    if not s.endswith(os.sep):
        s += os.sep
    return s


def phot_nn_h5_path(nnpath: str | os.PathLike, band: str) -> Path:
    """Payne photometry weights: ``<photNN>/nnMIST_<band>.h5``."""
    return Path(nnpath) / f"nnMIST_{band}.h5"


def list_available_phot_nn_bands(nnpath: str | os.PathLike) -> list[str]:
    """Band strings implied by ``nnMIST_*.h5`` in ``photNN``."""
    root = Path(nnpath)
    if not root.is_dir():
        return []
    out: list[str] = []
    for f in sorted(root.glob("nnMIST_*.h5")):
        stem = f.stem
        if stem.startswith("nnMIST_"):
            out.append(stem[len("nnMIST_") :])
    return out


def restrict_photometry_to_phot_nn_bands(
    nnpath: str | os.PathLike,
    phot: dict,
    phot_filtarr: list,
) -> tuple[dict, list]:
    """
    Keep only bands that have ``nnMIST_<band>.h5`` under ``photNN``.

    Many photometry tables include GALEX, SDSS, etc., while a given Payne photNN
    install only ships a subset (often no GALEX). Dropping unmatched bands avoids
    empty ``nnlist`` / IndexError inside ``FastPayneSEDPredict``.
    """
    root = Path(nnpath)
    avail = set(list_available_phot_nn_bands(nnpath))
    if not root.is_dir() or not avail:
        return phot, phot_filtarr

    kept_filt = [b for b in phot_filtarr if b in avail]
    dropped = [b for b in phot_filtarr if b not in avail]
    if dropped:
        warnings.warn(
            f"No nnMIST_<band>.h5 for {dropped} under {root.resolve()}; "
            "those bands are omitted for Payne photometry (fit/plot use remaining bands only).",
            UserWarning,
            stacklevel=2,
        )
    new_phot = {b: phot[b] for b in kept_filt if b in phot}
    return new_phot, kept_filt


def require_phot_nn_for_bands(nnpath: str | os.PathLike, bands: list | tuple) -> None:
    """
    Ensure Payne photometry NNs exist for each requested band.

    ``FastPayneSEDPredict`` catches load failures and then crashes with
    ``IndexError: list index out of range`` on an empty ``nnlist``; this checks
    up front and reports missing files and which bands exist on disk.
    """
    root = Path(nnpath)
    if not bands:
        raise ValueError(
            "No photometric bands in the photometry table (phot_filtarr is empty). "
            "The Payne phot predictor needs at least one band with finite magnitude "
            "and positive error in the *_phot.fits file."
        )
    if not root.is_dir():
        raise FileNotFoundError(
            f"photNN directory not found: {root.resolve()}. "
            "Expected a folder of nnMIST_<band>.h5 files next to specNN/ under your models tree."
        )
    missing = [b for b in bands if not phot_nn_h5_path(root, b).is_file()]
    if not missing:
        return
    avail = list_available_phot_nn_bands(root)
    if not avail:
        raise FileNotFoundError(
            f"No nnMIST_*.h5 files in {root.resolve()}. "
            "Install photometry neural-net weights (Payne / MIST photNN pack) in this directory."
        )
    show = avail[:50]
    extra = f" … (+{len(avail) - 50} more)" if len(avail) > 50 else ""
    raise FileNotFoundError(
        f"Missing phot NN weights for bands {missing!r}.\n"
        f"Expected files like nnMIST_<band>.h5 under:\n  {root.resolve()}\n"
        f"Bands available in that directory ({len(avail)}): {show}{extra}\n"
        "Band names in *_phot.fits must match these stems (e.g. GaiaDR3_G, PS_g)."
    )


def photometry_fits_path(gaia_id, data_dir: str | os.PathLike | None = None) -> str:
    base = os.getcwd() if data_dir is None else os.fspath(data_dir)
    return os.path.join(base, f"{gaia_id}_phot.fits")


def build_phot_dict_from_table(phottab: Table) -> tuple[dict, list]:
    """
    Build phot dict and ordered phot_filtarr from FITS table rows (band, mag, err).
    Skips invalid mags/errors; first row wins for duplicate band keys after aliasing.
    """
    phot: dict = {}
    for row in phottab:
        pb = standardize_band_name(row["band"])
        mag = float(row["mag"])
        err = float(row["err"])
        if math.isnan(mag) or math.isnan(err) or err <= 0:
            continue
        if pb not in phot:
            phot[pb] = [mag, err]
    phot_filtarr = [b for b in PREFERRED_BAND_ORDER if b in phot]
    extras = sorted(k for k in phot if k not in phot_filtarr)
    phot_filtarr.extend(extras)
    return phot, phot_filtarr


def ordered_phot_filtarr_from_dict(phot: dict) -> list:
    """Band ordering consistent with :func:`build_phot_dict_from_table`."""
    phot_filtarr = [b for b in PREFERRED_BAND_ORDER if b in phot]
    phot_filtarr.extend(sorted(k for k in phot if k not in phot_filtarr))
    return phot_filtarr


def _photsys_effective_wavelength_map() -> dict[str, float]:
    """Band name -> effective wavelength (Å) from uberMS ``photsys``; empty if import fails."""
    ensure_stellar_sys_path()
    try:
        from uberMS.utils import photsys

        return {str(k): float(v[0]) for k, v in photsys.photsys().items()}
    except Exception:
        return {}


def _photsys_zeropoint_jy_map() -> dict[str, float]:
    """Band name -> magnitude zeropoint (Jy at 0 mag), uberMS ``photsys`` index 2."""
    ensure_stellar_sys_path()
    try:
        from uberMS.utils import photsys

        return {str(k): float(v[2]) for k, v in photsys.photsys().items()}
    except Exception:
        return {}


def _mag_to_f_lambda_cgs_ubermss_style(mag: float, lambda_angstrom: float, zeropt_jy: float) -> float:
    """
    Same F_λ conversion as ``ppUMS.mkphot`` / uberMS plotting (``scipy.constants.c / 1000``).
    """
    from scipy import constants as scipy_constants

    speedoflight = scipy_constants.c / 1000.0
    jansky_cgs = 1e-23
    fnu_jy = float(zeropt_jy) * 10.0 ** (float(mag) / -2.5)
    lam = float(lambda_angstrom) * 1e-8
    return fnu_jy * jansky_cgs * (speedoflight / (lam**2))


def _planck_b_lambda_cgs(lambda_cm: np.ndarray, t_kelvin: float) -> np.ndarray:
    """Planck :math:`B_\\lambda` [erg s⁻¹ cm⁻² sr⁻¹ cm⁻¹]; λ in cm."""
    h = 6.62607015e-27
    k = 1.380649e-16
    c = 2.99792458e10
    lam = np.maximum(np.asarray(lambda_cm, dtype=float), 1e-16)
    x = np.clip(h * c / (lam * k * float(t_kelvin)), 1e-10, 700.0)
    b = (2.0 * h * c**2 / lam**5) / np.expm1(x)
    return np.maximum(b, 1e-300)


def _blackbody_grid_fit_log10f(
    lambda_cm: np.ndarray,
    log10_f_obs: np.ndarray,
    wts: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """
    Grid-search temperature; analytic weighted ``log10 K`` for ``F = K * B_λ(T)``.
    Returns ``(T_best, log10_K, pred_log10_f)``.
    """
    t_grid = np.unique(
        np.round(
            np.concatenate(
                [
                    np.linspace(2000.0, 8000.0, 55),
                    np.linspace(8000.0, 20000.0, 45),
                    np.linspace(20000.0, 50000.0, 25),
                ]
            ),
            3,
        )
    )
    best_chi = np.inf
    best_T = 6000.0
    best_logk = 0.0
    best_pred = log10_f_obs.copy()

    for t in t_grid:
        log10b = np.log10(_planck_b_lambda_cgs(lambda_cm, float(t)))
        sw = float(np.sum(wts))
        if sw <= 0.0:
            continue
        log10k = float(np.sum(wts * (log10_f_obs - log10b)) / sw)
        pred = log10k + log10b
        chi = float(np.sum(wts * (log10_f_obs - pred) ** 2))
        if chi < best_chi:
            best_chi = chi
            best_T = float(t)
            best_logk = log10k
            best_pred = pred

    return best_T, best_logk, best_pred


def _iterative_outlier_loop_linear(
    phot: dict,
    kept: list[str],
    waves: dict[str, float],
    *,
    sigma: float,
    max_iters: int,
    min_kept: int,
    floor_intrinsic_mag_err: float,
    dropped: list[str],
) -> list[str]:
    for _ in range(max_iters):
        if len(kept) <= min_kept:
            break
        x = np.array([math.log10(waves[b]) for b in kept], dtype=float)
        y = np.array([float(phot[b][0]) for b in kept], dtype=float)
        err = np.array([max(float(phot[b][1]), 1e-6) for b in kept], dtype=float)
        wts = 1.0 / (err**2)
        coef = np.polyfit(x, y, 1, w=wts)
        pred = coef[0] * x + coef[1]
        res = y - pred
        med = float(np.median(res))
        mad = float(np.median(np.abs(res - med)))
        s_int = 1.4826 * mad if mad > 0.0 else floor_intrinsic_mag_err
        s_int = max(s_int, floor_intrinsic_mag_err)
        comb = np.sqrt(err**2 + s_int**2)
        z = np.abs(res) / comb
        j = int(np.argmax(z))
        if float(z[j]) <= sigma:
            break
        dropped.append(kept.pop(j))
    return kept


def _iterative_outlier_loop_blackbody(
    phot: dict,
    kept: list[str],
    waves: dict[str, float],
    zeropts: dict[str, float],
    *,
    sigma: float,
    max_iters: int,
    min_kept: int,
    floor_intrinsic_log_floor: float,
    dropped: list[str],
) -> list[str]:
    """
    Fit ``F_λ ∝ B_λ(T)`` in log10 flux; σ(log10 F) ≈ 0.4 σ(mag) for ``F ∝ 10^{-m/2.5}``.
    """
    for _ in range(max_iters):
        if len(kept) <= min_kept:
            break
        lam_cm = np.array([waves[b] * 1e-8 for b in kept], dtype=float)
        m = np.array([float(phot[b][0]) for b in kept], dtype=float)
        m_err = np.array([max(float(phot[b][1]), 1e-6) for b in kept], dtype=float)
        f_obs = np.array(
            [_mag_to_f_lambda_cgs_ubermss_style(m[i], waves[kept[i]], zeropts[kept[i]]) for i in range(len(kept))],
            dtype=float,
        )
        log10f = np.log10(np.maximum(f_obs, 1e-300))
        sig_log = 0.4 * m_err
        wts = 1.0 / np.maximum(sig_log**2, 1e-12)
        _t, _lk, pred = _blackbody_grid_fit_log10f(lam_cm, log10f, wts)
        res = log10f - pred
        med = float(np.median(res))
        mad = float(np.median(np.abs(res - med)))
        s_int = 1.4826 * mad if mad > 0.0 else floor_intrinsic_log_floor
        s_int = max(s_int, floor_intrinsic_log_floor)
        comb = np.sqrt(sig_log**2 + s_int**2)
        z = np.abs(res) / comb
        j = int(np.argmax(z))
        if float(z[j]) <= sigma:
            break
        dropped.append(kept.pop(j))
    return kept


def iterative_photometry_outlier_rejection(
    phot: dict,
    phot_filtarr: list,
    *,
    sigma: float = 3.0,
    max_iters: int = 16,
    min_kept: int = 3,
    floor_intrinsic_mag_err: float = 0.03,
    method: str = "blackbody",
) -> tuple[dict, list, list[str]]:
    """
    Iteratively drop bands whose photometry disagrees with a simple SED model.

    **blackbody** (default): convert mags to :math:`F_\\lambda` with uberMS ``photsys``
    zeropoints (same recipe as ``ppUMS.mkphot``), grid-fit :math:`F_\\lambda \\propto B_\\lambda(T)`,
    reject outliers in log10 flux using robust MAD scatter plus magnitude errors.

    **linear**: legacy weighted straight line in magnitude vs :math:`\\log_{10}(\\lambda)`.

    Bands without wavelength (and for BB, without zeropoint) in ``photsys`` are never dropped.
    Requires more than ``min_kept`` participating bands before any drops.
    """
    if sigma <= 0 or not phot or not phot_filtarr:
        return phot, list(phot_filtarr), []

    m = (method or "blackbody").strip().lower()
    if m in ("bb", "planck", "blackbody"):
        m = "blackbody"
    elif m in ("line", "linear", "poly"):
        m = "linear"
    else:
        raise ValueError("iterative_photometry_outlier_rejection: method must be 'blackbody' or 'linear'.")

    waves = _photsys_effective_wavelength_map()
    if not waves:
        warnings.warn(
            "iterative_photometry_outlier_rejection: could not load uberMS photsys; skipping.",
            UserWarning,
            stacklevel=2,
        )
        return phot, list(phot_filtarr), []

    zeropts = _photsys_zeropoint_jy_map() if m == "blackbody" else {}

    if m == "blackbody":
        no_fit = {
            b
            for b in phot_filtarr
            if b in phot and (b not in waves or b not in zeropts or not math.isfinite(zeropts[b]) or zeropts[b] <= 0)
        }
        fit_bands = [b for b in phot_filtarr if b in phot and b not in no_fit]
    else:
        no_fit = {b for b in phot_filtarr if b in phot and b not in waves}
        fit_bands = [b for b in phot_filtarr if b in phot and b in waves]

    if len(fit_bands) <= min_kept:
        return phot, list(phot_filtarr), []

    kept = list(fit_bands)
    dropped: list[str] = []

    if m == "linear":
        kept = _iterative_outlier_loop_linear(
            phot,
            kept,
            waves,
            sigma=sigma,
            max_iters=max_iters,
            min_kept=min_kept,
            floor_intrinsic_mag_err=floor_intrinsic_mag_err,
            dropped=dropped,
        )
    else:
        floor_log = max(0.4 * floor_intrinsic_mag_err, 0.01)
        kept = _iterative_outlier_loop_blackbody(
            phot,
            kept,
            waves,
            zeropts,
            sigma=sigma,
            max_iters=max_iters,
            min_kept=min_kept,
            floor_intrinsic_log_floor=floor_log,
            dropped=dropped,
        )

    kept_fit = set(kept)
    new_filtarr = [b for b in phot_filtarr if b in phot and (b in kept_fit or b in no_fit)]
    new_phot = {b: phot[b] for b in new_filtarr}
    return new_phot, new_filtarr, dropped


def load_photometry_fits(gaia_id, data_dir: str | os.PathLike | None = None) -> tuple[dict, list]:
    path = photometry_fits_path(gaia_id, data_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Photometry file not found: {path}")
    phottab = Table.read(path, format="fits", hdu=1)
    return build_phot_dict_from_table(phottab)


def _float_field(val) -> float:
    if np.ma.is_masked(val):
        return float("nan")
    return float(val)


def _gaia_fallback_tap_enabled() -> bool:
    raw = os.environ.get("STELLAR_GAIA_TAP_FALLBACK")
    if raw is None or not str(raw).strip():
        return True
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def _gaia_fallback_sync_url() -> str:
    return os.environ.get(
        "STELLAR_GAIA_TAP_FALLBACK_URL", _DEFAULT_GAIA_FALLBACK_TAP_SYNC
    ).strip()


def _tap_sync_votable_table(sync_url: str, adql: str, *, timeout: float = 120.0) -> Table:
    """
    Run a synchronous ADQL job on a TAP ``sync`` endpoint (e.g. Gaia@AIP).

    Gaia@AIP rejects ``Accept: text/plain`` (used by astroquery TapConn); use ``*/*``.
    """
    r = requests.post(
        sync_url,
        data={
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "votable",
            "QUERY": adql.strip(),
        },
        headers={"Accept": "*/*"},
        timeout=timeout,
    )
    r.raise_for_status()
    votable = votable_parse(BytesIO(r.content))
    for resource in votable.resources:
        for info in resource.infos:
            if getattr(info, "name", None) == "QUERY_STATUS" and info.value != "OK":
                raise RuntimeError(
                    "TAP QUERY_STATUS={!r}: {}".format(info.value, getattr(info, "content", "") or "")
                )
    return votable.get_first_table().to_table()


def _nss_parallax_pair(table: Table | None) -> tuple[float, float] | None:
    """First row (π, σ_π) if both are finite and positive; else None."""
    if table is None or len(table) == 0:
        return None
    plx = _float_field(table["parallax"][0])
    sig = _float_field(table["parallax_error"][0])
    if math.isnan(plx) or plx <= 0.0 or math.isnan(sig) or sig <= 0.0:
        return None
    return (plx, sig)


def _query_gaia_stellar_priors_fallback(gaia_id: int | str, *, sync_url: str) -> Table:
    """
    Same coalesce order as the ESAC query (NSS 2-body → NSS acceleration → ``gaia_source``),
    using several simple ADQL statements. Mirrors such as Gaia@AIP often lack ``COALESCE`` /
    ``CASE`` in their ADQL translator, so coalescing is done in Python.
    """
    gid = int(gaia_id)
    gs = _tap_sync_votable_table(
        sync_url,
        f"""
        SELECT TOP 10
            teff_gspphot,
            logg_gspphot,
            mh_gspphot,
            parallax,
            parallax_error
        FROM gaiadr3.gaia_source
        WHERE source_id = {gid}
        """,
    )
    if gs is None or len(gs) == 0:
        raise ValueError(
            f"Fallback TAP returned no row in gaiadr3.gaia_source for source_id {gaia_id}."
        )

    teff = gs["teff_gspphot"][0]
    logg = gs["logg_gspphot"][0]
    mh = gs["mh_gspphot"][0]
    g_plx = _float_field(gs["parallax"][0])
    g_err = _float_field(gs["parallax_error"][0])

    nss2 = _tap_sync_votable_table(
        sync_url,
        f"""
        SELECT TOP 10 parallax, parallax_error
        FROM gaiadr3.nss_two_body_orbit
        WHERE source_id = {gid}
        """,
    )
    pair = _nss_parallax_pair(nss2)
    if pair is None:
        nss_acc = _tap_sync_votable_table(
            sync_url,
            f"""
            SELECT TOP 10 parallax, parallax_error
            FROM gaiadr3.nss_acceleration_astro
            WHERE source_id = {gid}
            """,
        )
        pair = _nss_parallax_pair(nss_acc)
    if pair is None:
        pair = (g_plx, g_err)

    final_plx, final_err = pair
    return Table(
        rows=[[teff, logg, mh, final_plx, final_err]],
        names=[
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "final_parallax",
            "final_parallax_error",
        ],
    )


def query_gaia_stellar_priors(gaia_id: int | str) -> dict:
    """
    Stellar priors from Gaia DR3 ``gaia_source`` with NSS parallax fallbacks.

    Parallax is taken in fixed priority order **nss_two_body_orbit →
    nss_acceleration_astro → gaia_source** (ESAC: ``COALESCE`` in one ADQL job).

    If the ESAC TAP service fails (HTTP 400, unknown table on a degraded cluster, etc.),
    and ``STELLAR_GAIA_TAP_FALLBACK`` is not disabled, this function retries using an
    alternate sync TAP URL (default Gaia@AIP), issuing simple ADQL queries and applying
    the same coalesce order in Python. Override the URL with ``STELLAR_GAIA_TAP_FALLBACK_URL``.

    A **positive parallax** and **positive parallax error** after coalescing are
    **required**; otherwise this function raises and the caller must not run uberMS
    without a distance prior.
    """
    # Explicit ``SELECT TOP`` avoids astroquery TAP injecting ``TOP 2000`` right after
    # ``SELECT``, which breaks multiline ADQL on the Gaia archive (unknown table/columns).
    query = f"""
        SELECT TOP 10
            gs.teff_gspphot,
            gs.logg_gspphot,
            gs.mh_gspphot,
            COALESCE(nss_2b.parallax, nss_acc.parallax, gs.parallax) AS final_parallax,
            COALESCE(nss_2b.parallax_error, nss_acc.parallax_error, gs.parallax_error)
                AS final_parallax_error
        FROM gaiadr3.gaia_source AS gs
        LEFT JOIN gaiadr3.nss_two_body_orbit AS nss_2b
            ON gs.source_id = nss_2b.source_id
        LEFT JOIN gaiadr3.nss_acceleration_astro AS nss_acc
            ON gs.source_id = nss_acc.source_id
        WHERE gs.source_id = {gaia_id}
    """
    result = None
    primary_err: BaseException | None = None
    try:
        job = Gaia.launch_job(query)
        result = job.get_results()
    except Exception as err:
        primary_err = err

    if result is None and _gaia_fallback_tap_enabled():
        sync_url = _gaia_fallback_sync_url()
        try:
            warnings.warn(
                "Gaia ESAC TAP query failed for source_id {}; using fallback TAP at {!r}.".format(
                    gaia_id, sync_url
                ),
                UserWarning,
                stacklevel=2,
            )
            result = _query_gaia_stellar_priors_fallback(gaia_id, sync_url=sync_url)
        except Exception as fb_err:
            hint = (
                " If the message mentions an unknown NSS table or HTTP 400, the Gaia TAP "
                "endpoint is often at fault (mirror lag or maintenance), not this ADQL "
                "or the DR3 schema names."
            )
            if primary_err is not None:
                raise RuntimeError(
                    "Gaia archive query failed for source_id {} (ESAC: {}; fallback {!r}: {}).{}".format(
                        gaia_id, primary_err, sync_url, fb_err, hint
                    )
                ) from fb_err
            raise RuntimeError(
                "Gaia archive fallback TAP failed for source_id {}: {}{}".format(
                    gaia_id, fb_err, hint
                )
            ) from fb_err

    if result is None:
        assert primary_err is not None
        hint = (
            " If the message mentions an unknown NSS table or HTTP 400, the Gaia TAP "
            "endpoint is often at fault (mirror lag or maintenance), not this ADQL "
            "or the DR3 schema names."
        )
        raise RuntimeError(
            f"Gaia archive query failed for source_id {gaia_id}: {primary_err}.{hint}"
        ) from primary_err

    if result is None or len(result) == 0:
        raise ValueError(
            f"Gaia DR3 returned no row for source_id {gaia_id}. "
            "Check that the id is a valid ``source_id``."
        )

    out: dict = {}
    out["Teff"] = _float_field(result["teff_gspphot"][0])
    out["log(g)"] = _float_field(result["logg_gspphot"][0])
    out["[Fe/H]"] = _float_field(result["mh_gspphot"][0])
    out["[a/Fe]"] = -0.2
    out["log(R)"] = 0.0
    out["Mass"] = 1.0

    plx = _float_field(result["final_parallax"][0])
    plx_err = _float_field(result["final_parallax_error"][0])
    if math.isnan(plx) or plx <= 0:
        raise ValueError(
            f"No valid, positive parallax for Gaia source_id {gaia_id}. "
            "The archive returned NULL/non-finite/<=0 in "
            "nss_two_body_orbit, nss_acceleration_astro, and gaia_source.parallax."
        )
    if math.isnan(plx_err) or plx_err <= 0:
        raise ValueError(
            f"No valid parallax uncertainty for Gaia source_id {gaia_id} "
            f"(final_parallax={plx} mas but error is missing or <= 0). "
            "uberMS needs [parallax, parallax_error] for the distance prior."
        )
    out["parallax"] = [plx, plx_err]

    if math.isnan(out["Teff"]):
        out["Teff"] = 5500.0
    if math.isnan(out["log(g)"]):
        out["log(g)"] = 0.0
    if math.isnan(out["[Fe/H]"]):
        out["[Fe/H]"] = 0.0

    return out
