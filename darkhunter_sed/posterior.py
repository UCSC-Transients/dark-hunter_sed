"""Extract posterior summaries from uberMS sample FITS → sed_summary.json."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table

UMS_PRIMARY_PARAMS = (
    "initial_Mass",
    "EEP",
    "initial_[Fe/H]",
    "initial_[a/Fe]",
    "Teff",
    "log(g)",
    "[Fe/H]",
    "[a/Fe]",
    "Age",
    "log(L)",
    "log(R)",
    "dist",
    "Av",
    "vstar",
    "vmic",
)

UTP_PRIMARY_PARAMS = (
    "Teff",
    "log(g)",
    "[Fe/H]",
    "[a/Fe]",
    "log(R)",
    "dist",
    "Av",
    "vstar",
    "vmic",
)

_VRAD_RE = re.compile(r"^vrad_(\d+)$")


def _finite_only(xarr: np.ndarray) -> np.ndarray:
    a = np.asarray(xarr, dtype=float).ravel()
    return a[np.isfinite(a)]


def percentile_stats(xarr) -> dict[str, float]:
    """Median, std, and 16/84% credible interval from posterior draws."""
    a = _finite_only(xarr)
    if a.size == 0:
        nan = float("nan")
        return {"median": nan, "std": nan, "p16": nan, "p84": nan, "n_finite": 0}
    return {
        "median": float(np.median(a)),
        "std": float(np.std(a)),
        "p16": float(np.percentile(a, 16)),
        "p84": float(np.percentile(a, 84)),
        "n_finite": int(a.size),
    }


def _summarize_column(samples: Table, col: str) -> dict[str, float] | None:
    if col not in samples.colnames:
        return None
    stats = percentile_stats(samples[col])
    if stats["n_finite"] == 0:
        return None
    return stats


def _summarize_vrad_epochs(samples: Table) -> list[dict[str, Any]]:
    epochs: list[tuple[int, str]] = []
    for col in samples.colnames:
        m = _VRAD_RE.match(str(col))
        if m:
            epochs.append((int(m.group(1)), col))
    epochs.sort(key=lambda t: t[0])

    out: list[dict[str, Any]] = []
    for ep_i, col in epochs:
        stats = _summarize_column(samples, col)
        if stats is None:
            continue
        out.append({"epoch_index": ep_i, "column": col, **stats})
    return out


def summarize_samples_fits(
    samples_path: str | Path,
    *,
    fit_type: str,
    primary_params: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Load uberMS posterior FITS and compute per-parameter statistics."""
    samples_path = Path(samples_path)
    if not samples_path.is_file():
        raise FileNotFoundError(samples_path)

    samples = Table.read(samples_path, format="fits")
    fit_type_l = fit_type.strip().lower()
    if primary_params is None:
        primary_params = UMS_PRIMARY_PARAMS if fit_type_l == "ums" else UTP_PRIMARY_PARAMS

    parameters: dict[str, dict[str, float]] = {}
    for col in primary_params:
        stats = _summarize_column(samples, col)
        if stats is not None:
            parameters[col] = stats

    for col in samples.colnames:
        if col in parameters:
            continue
        if _VRAD_RE.match(str(col)) or str(col).startswith(("specjitter_", "lsf_", "pc")):
            stats = _summarize_column(samples, col)
            if stats is not None:
                parameters[col] = stats

    result: dict[str, Any] = {
        "fit_type": fit_type_l,
        "samples_fits": str(samples_path.resolve()),
        "samples_mtime_iso": datetime.fromtimestamp(
            samples_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "n_samples": len(samples),
        "parameters": parameters,
        "vrad_epochs": _summarize_vrad_epochs(samples),
    }

    if fit_type_l == "ums" and "initial_Mass" in parameters:
        result["m1_msun"] = dict(parameters["initial_Mass"])

    if "dist" in parameters:
        # uberMS stores distance in pc (init = 1000/parallax_mas).
        result["dist_pc"] = dict(parameters["dist"])

    return result


def build_sed_summary(
    gaia_id: str,
    *,
    ums_samples: str | Path | None = None,
    utp_samples: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble full per-star JSON document."""
    doc: dict[str, Any] = {
        "gaia_source_id": str(gaia_id),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fits": {},
    }
    if ums_samples is not None:
        doc["fits"]["ums"] = summarize_samples_fits(ums_samples, fit_type="ums")
    if utp_samples is not None:
        doc["fits"]["utp"] = summarize_samples_fits(utp_samples, fit_type="utp")

    if extra:
        doc.update(extra)

    if "fits" in doc and "ums" in doc["fits"] and "m1_msun" in doc["fits"]["ums"]:
        doc["m1_msun"] = doc["fits"]["ums"]["m1_msun"]

    return doc


def sed_summary_path(gaia_id: str, summaries_dir: Path | None = None) -> Path:
    from darkhunter_sed.config import sed_summaries_dir

    root = summaries_dir or sed_summaries_dir()
    return root / f"Gaia_DR3_{gaia_id}_sed_summary.json"


def write_sed_summary(
    gaia_id: str,
    *,
    ums_samples: str | Path | None = None,
    utp_samples: str | Path | None = None,
    out_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    out_path = Path(out_path) if out_path is not None else sed_summary_path(gaia_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = build_sed_summary(
        gaia_id,
        ums_samples=ums_samples,
        utp_samples=utp_samples,
        extra=extra,
    )
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def read_sed_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def m1_msun_from_summary(doc: dict[str, Any]) -> float | None:
    """Best luminous M1 (Msun) from a sed_summary document."""
    for key in ("m1_msun",):
        block = doc.get(key)
        if isinstance(block, dict):
            med = block.get("median")
            if med is not None and math.isfinite(float(med)):
                return float(med)
    fits = doc.get("fits", {})
    ums = fits.get("ums", {})
    block = ums.get("m1_msun") or ums.get("parameters", {}).get("initial_Mass")
    if isinstance(block, dict):
        med = block.get("median")
        if med is not None and math.isfinite(float(med)):
            return float(med)
    return None
