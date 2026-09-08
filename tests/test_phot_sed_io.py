"""Unit tests for Path-2 photometry FITS flag / upper-limit I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from darkhunter_sed.phot_sed_io import (
    FLAG_DETECTION,
    FLAG_UPPER_LIMIT,
    PhotRow,
    read_photometry_fits,
    split_detections_and_upper_limits,
    write_photometry_fits,
)
from darkhunter_sed.photometry_gather import save_photometry_to_fits
from darkhunter_sed.stellar_data import build_phot_dict_from_table, load_photometry_fits


def _legacy_three_col_fits(path: Path, rows: list[tuple[str, float, float]]) -> None:
    bands = np.array([b.encode("ascii") for b, _, _ in rows], dtype="S32")
    mags = np.array([m for _, m, _ in rows], dtype=np.float64)
    errs = np.array([e for _, _, e in rows], dtype=np.float64)
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="band", format="32A", array=bands),
            fits.Column(name="mag", format="D", array=mags),
            fits.Column(name="err", format="D", array=errs),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def test_write_read_roundtrip_with_ul(tmp_path: Path) -> None:
    path = tmp_path / "123_phot.fits"
    write_photometry_fits(
        path,
        [
            ("GaiaDR3_G", 12.0, 0.01, FLAG_DETECTION),
            ("GALEX_NUV", 21.5, 0.3, FLAG_UPPER_LIMIT),
            PhotRow("PS_g", 12.2, 0.02, FLAG_DETECTION),
        ],
    )
    rows = read_photometry_fits(path)
    assert len(rows) == 3
    assert rows[0] == PhotRow("GaiaDR3_G", 12.0, 0.01, FLAG_DETECTION)
    assert rows[1].flag == FLAG_UPPER_LIMIT
    assert rows[1].mag == pytest.approx(21.5)
    dets, uls = split_detections_and_upper_limits(rows)
    assert [r.band for r in dets] == ["GaiaDR3_G", "PS_g"]
    assert [r.band for r in uls] == ["GALEX_NUV"]

    with fits.open(path) as hdul:
        names = hdul[1].data.dtype.names
    assert names == ("band", "mag", "err", "flag")


def test_missing_flag_column_defaults_to_detection(tmp_path: Path) -> None:
    path = tmp_path / "legacy_phot.fits"
    _legacy_three_col_fits(
        path,
        [("GaiaDR3_G", 10.0, 0.01), ("PS_r", 9.5, 0.02)],
    )
    rows = read_photometry_fits(path)
    assert all(r.flag == FLAG_DETECTION for r in rows)
    assert [r.band for r in rows] == ["GaiaDR3_G", "PS_r"]

    phot, filt = build_phot_dict_from_table(Table.read(path, format="fits", hdu=1))
    assert set(phot) == {"GaiaDR3_G", "PS_r"}
    assert filt[0] in phot


def test_path1_loader_skips_upper_limits(tmp_path: Path) -> None:
    gid = "999001"
    path = tmp_path / f"{gid}_phot.fits"
    write_photometry_fits(
        path,
        [
            ("GaiaDR3_G", 11.0, 0.01, FLAG_DETECTION),
            ("GALEX_FUV", 22.0, 0.2, FLAG_UPPER_LIMIT),
        ],
    )
    phot, filt = load_photometry_fits(gid, tmp_path)
    assert list(phot) == ["GaiaDR3_G"]
    assert filt == ["GaiaDR3_G"]
    assert "GALEX_FUV" not in phot


def test_gather_save_writes_flag_zero(tmp_path: Path) -> None:
    out = save_photometry_to_fits(
        "4219",
        [("GaiaDR3_G", 13.0, 0.05), ("2MASS_J", 12.0, 0.04)],
        outdir=tmp_path,
    )
    rows = read_photometry_fits(out)
    assert len(rows) == 2
    assert all(r.flag == FLAG_DETECTION for r in rows)
    with fits.open(out) as hdul:
        assert "flag" in hdul[1].columns.names
        assert np.all(hdul[1].data["flag"] == 0)


def test_normalize_rejects_bad_flag_and_err(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="flag must be"):
        write_photometry_fits(tmp_path / "bad.fits", [("PS_g", 12.0, 0.01, 99)])
    with pytest.raises(ValueError, match="err > 0"):
        write_photometry_fits(tmp_path / "bad2.fits", [("PS_g", 12.0, 0.0, 0)])
