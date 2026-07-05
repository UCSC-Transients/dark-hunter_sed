"""Build uberMS priors from dark-hunter_rv star summaries and optional Gaia re-query."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

from darkhunter_rv.gaia_utils import parse_gaia_metadata_from_star_summary
from darkhunter_rv.rv_point_filters import rv_epoch_is_valid
from darkhunter_rv.summary_paths import discover_summary_path

from darkhunter_sed.config import rv_output_dir
from darkhunter_sed.stellar_data import query_gaia_stellar_priors

logger = logging.getLogger(__name__)

_EPOCH_FILE_RE = re.compile(r"Gaia_DR3_\d+_epoch_(\d+)\.txt", re.I)


@dataclass(frozen=True)
class RvEpoch:
    basename: str
    epoch_num: int
    mjd: float
    rv_kms: float
    err_kms: float


def _float_or_nan(val) -> float:
    try:
        x = float(val)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def parse_legacy_file_summary_rv_epochs(summary_path: Path) -> list[RvEpoch]:
    """
    Parse legacy ``# File Summary`` rows (basename MJD RV Err …) when
    ``[PIPELINE RESULTS]`` is absent.
    """
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    if "[PIPELINE RESULTS]" in text:
        return []
    out: list[RvEpoch] = []
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("# file summary"):
            in_block = True
            continue
        if not in_block:
            continue
        if not stripped or stripped.startswith("#"):
            if out and stripped.startswith("#") and "file summary" not in stripped.lower():
                break
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            break
        parts = stripped.split()
        if len(parts) < 4:
            continue
        basename = Path(parts[0]).name
        mjd = _float_or_nan(parts[1])
        rv = _float_or_nan(parts[2])
        err = _float_or_nan(parts[3])
        if not rv_epoch_is_valid(mjd, rv):
            continue
        m = _EPOCH_FILE_RE.match(basename)
        epoch_num = int(m.group(1)) if m else len(out)
        out.append(
            RvEpoch(
                basename=basename,
                epoch_num=epoch_num,
                mjd=mjd,
                rv_kms=rv,
                err_kms=err if math.isfinite(err) and err > 0 else float("nan"),
            )
        )
    out.sort(key=lambda e: e.epoch_num)
    return out


def parse_rv_epochs_from_summary(summary_path: Path) -> list[RvEpoch]:
    """Pipeline results first, else legacy file summary block."""
    epochs = parse_pipeline_rv_epochs(summary_path)
    if epochs:
        return epochs
    return parse_legacy_file_summary_rv_epochs(summary_path)


def parse_pipeline_rv_epochs(summary_path: Path) -> list[RvEpoch]:
    """Parse [PIPELINE RESULTS] rows with finite RV measurements."""
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    if "[PIPELINE RESULTS]" not in text:
        return []
    block = text.split("[PIPELINE RESULTS]", 1)[1]
    out: list[RvEpoch] = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or (line.startswith("[") and line.endswith("]")):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        basename = parts[0]
        mjd = _float_or_nan(parts[1])
        rv = _float_or_nan(parts[2])
        err = _float_or_nan(parts[3])
        if not rv_epoch_is_valid(mjd, rv):
            continue
        m = _EPOCH_FILE_RE.match(basename)
        epoch_num = int(m.group(1)) if m else len(out)
        out.append(
            RvEpoch(
                basename=basename,
                epoch_num=epoch_num,
                mjd=mjd,
                rv_kms=rv,
                err_kms=err if math.isfinite(err) and err > 0 else float("nan"),
            )
        )
    out.sort(key=lambda e: e.epoch_num)
    return out


def match_rv_epochs_to_spectrum_paths(
    rv_epochs: list[RvEpoch],
    spectrum_paths: list[Path],
) -> list[RvEpoch]:
    """
    Align RV epochs to spectrum file order (by epoch number in filename).

    Returns one RvEpoch per spectrum path when possible; missing epochs get NaN RV.
    """
    by_epoch = {e.epoch_num: e for e in rv_epochs}
    matched: list[RvEpoch] = []
    for i, path in enumerate(spectrum_paths):
        m = _EPOCH_FILE_RE.search(path.name)
        ep = int(m.group(1)) if m else i
        if ep in by_epoch:
            matched.append(by_epoch[ep])
        else:
            logger.warning("No pipeline RV for spectrum %s (epoch %s)", path.name, ep)
            matched.append(
                RvEpoch(
                    basename=path.name,
                    epoch_num=ep,
                    mjd=float("nan"),
                    rv_kms=float("nan"),
                    err_kms=float("nan"),
                )
            )
    return matched


def stellar_priors_from_summary_metadata(meta: dict) -> dict:
    """Map [GAIA METADATA] keys to uberMS stellar prior dict."""
    teff = _float_or_nan(meta.get("Teff"))
    logg = _float_or_nan(meta.get("logg"))
    mh = _float_or_nan(meta.get("MH"))
    plx = _float_or_nan(meta.get("Parallax"))
    plx_err = _float_or_nan(meta.get("Parallax_Error"))
    mass_flame = _float_or_nan(meta.get("Mass_FLAME"))
    age_flame = _float_or_nan(meta.get("Age_FLAME"))

    ra = _float_or_nan(meta.get("RA"))
    dec = _float_or_nan(meta.get("Dec"))

    mass_init = mass_flame if math.isfinite(mass_flame) and mass_flame > 0.0 else 1.0

    out: dict = {
        "Teff": teff if math.isfinite(teff) else 5500.0,
        "log(g)": logg if math.isfinite(logg) else 4.0,
        "[Fe/H]": mh if math.isfinite(mh) else 0.0,
        "[a/Fe]": -0.2,
        "log(R)": 0.0,
        "Mass": mass_init,
    }
    if math.isfinite(age_flame) and age_flame > 0.0:
        out["Age_Gyr"] = float(age_flame)
    flags_flame = meta.get("Flags_FLAME")
    if flags_flame is not None and str(flags_flame).strip() not in ("", "None", "nan"):
        out["Flags_FLAME"] = str(flags_flame).strip()
    if math.isfinite(ra):
        out["RA"] = ra
    if math.isfinite(dec):
        out["Dec"] = dec

    if not math.isfinite(plx) or plx <= 0:
        raise ValueError("Summary metadata missing positive Parallax")
    if not math.isfinite(plx_err) or plx_err <= 0:
        raise ValueError("Summary metadata missing positive Parallax_Error")
    out["parallax"] = [plx, plx_err]
    return out


def load_stellar_priors(
    gaia_id: str,
    *,
    summary_path: Path | None = None,
    rv_output: Path | None = None,
    force_redownload: bool = False,
) -> dict:
    """
    Stellar atmosphere + parallax priors.

    Default: read ``[GAIA METADATA]`` from the RV summary. With ``force_redownload``,
    query Gaia TAP via ``stellar_data.query_gaia_stellar_priors``.
    """
    if force_redownload:
        logger.info("Force redownload: querying Gaia for source_id %s", gaia_id)
        return query_gaia_stellar_priors(gaia_id)

    rv_output = rv_output or rv_output_dir()
    summary_path = summary_path or discover_summary_path(rv_output, gaia_id)
    if summary_path is None or not summary_path.is_file():
        logger.warning(
            "No RV summary for %s under %s; querying Gaia TAP",
            gaia_id,
            rv_output,
        )
        return query_gaia_stellar_priors(gaia_id)

    meta = parse_gaia_metadata_from_star_summary(summary_path)
    if meta is None or meta.get("Source_ID") is None:
        logger.warning("Incomplete summary %s; querying Gaia TAP", summary_path)
        return query_gaia_stellar_priors(gaia_id)

    try:
        return stellar_priors_from_summary_metadata(meta)
    except ValueError as exc:
        logger.warning("%s; querying Gaia TAP", exc)
        return query_gaia_stellar_priors(gaia_id)


def build_vrad_init_and_priors(
    rv_epochs: list[RvEpoch],
    *,
    prior_kind: str = "normal",
    err_inflate: float = 2.0,
    err_floor_kms: float = 2.0,
    uniform_half_span: float = 80.0,
) -> tuple[dict[str, float], dict[str, list]]:
    """
    Per-epoch ``vrad_<i>`` init and uberMS prior entries from RV pipeline measurements.

    Default: normal priors centered on mask-CCF RV with inflated uncertainty.
    """
    init: dict[str, float] = {}
    priors: dict[str, list] = {}
    kind = (prior_kind or "normal").strip().lower()

    for ii, ep in enumerate(rv_epochs):
        key = f"vrad_{ii}"
        if math.isfinite(ep.rv_kms):
            center = float(ep.rv_kms)
        else:
            center = 0.0
        init[key] = center

        if kind == "fixed":
            priors[key] = ["fixed", center]
            continue

        sigma = err_floor_kms
        if math.isfinite(ep.err_kms) and ep.err_kms > 0:
            sigma = max(err_floor_kms, float(err_inflate) * float(ep.err_kms))

        if kind == "uniform":
            h = float(uniform_half_span)
            priors[key] = ["uniform", [center - h, center + h]]
        else:
            priors[key] = ["normal", [center, sigma]]

    return init, priors


def resolve_summary_for_star(
    gaia_id: str,
    rv_output: Path | None = None,
) -> Path | None:
    return discover_summary_path(rv_output or rv_output_dir(), gaia_id)
