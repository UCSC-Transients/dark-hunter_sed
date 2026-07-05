"""Environment and path configuration for dark-hunter_sed."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Ingest: peel noisy blaze-normalized order ends (overlap wings rising to ~1.05)
# while |flux - 1| exceeds tolerance, keeping at least this many pixels per order.
SPECTRUM_ORDER_END_DEV_TOL = 0.04
SPECTRUM_ORDER_END_MIN_PIXELS = 18


def stellar_root() -> Path:
    """Directory containing uberMS, ThePayne, MISTy (sibling installs)."""
    env = os.environ.get("STELLAR_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # Default: ~/stellar next to darkhunter tree
    candidate = Path.home() / "stellar"
    if candidate.is_dir():
        return candidate
    return (REPO_ROOT.parent.parent / "stellar").resolve()


def output_dir() -> Path:
    return Path(
        os.environ.get("DARKHUNTER_SED_OUTPUT_DIR", REPO_ROOT / "output")
    ).expanduser().resolve()


def samples_dir() -> Path:
    return Path(
        os.environ.get("DARKHUNTER_SED_SAMPLES_DIR", output_dir() / "samples")
    ).expanduser().resolve()


def plots_dir() -> Path:
    return Path(
        os.environ.get("DARKHUNTER_SED_PLOTS_DIR", output_dir() / "plots")
    ).expanduser().resolve()


def photometry_dir() -> Path:
    return Path(
        os.environ.get("DARKHUNTER_SED_PHOTOMETRY_DIR", output_dir() / "photometry")
    ).expanduser().resolve()


def masks_dir() -> Path:
    """Picker regions JSON directory (``regions_Gaia_DR3_<id>_*.json``)."""
    return Path(
        os.environ.get("DARKHUNTER_SED_MASKS_DIR", output_dir() / "masks")
    ).expanduser().resolve()


def models_dir() -> Path | None:
    raw = os.environ.get("DARKHUNTER_SED_MODELS_DIR")
    if raw:
        p = Path(raw).expanduser().resolve()
        return p if p.is_dir() else None
    return None


def rv_output_dir() -> Path:
    """dark-hunter_rv pipeline output (summaries)."""
    return Path(
        os.environ.get(
            "DARKHUNTER_OUTPUT_DIR",
            REPO_ROOT.parent.parent / "rvs" / "dark-hunter_rv" / "output",
        )
    ).expanduser().resolve()


def spec_root() -> Path:
    return Path(
        os.environ.get(
            "SPEC_ROOT",
            REPO_ROOT.parent.parent / "rvs" / "data",
        )
    ).expanduser().resolve()


def blaze_calibration_path() -> Path:
    from darkhunter_rv.config import BLAZE_CALIBRATION_FILE

    env = os.environ.get("DARKHUNTER_BLAZE_CALIBRATION")
    if env:
        return Path(env).expanduser().resolve()
    return BLAZE_CALIBRATION_FILE


def sed_summaries_dir() -> Path:
    return Path(
        os.environ.get("DARKHUNTER_SED_SUMMARIES_DIR", output_dir() / "sed_summaries")
    ).expanduser().resolve()
