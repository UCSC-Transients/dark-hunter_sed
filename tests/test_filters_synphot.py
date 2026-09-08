"""Tests for PS1/Swift synphot thruput conversion and band registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from darkhunter_sed.filters_synphot import (
    BAND_REGISTRY,
    CONVERTED_BANDS,
    DATA_DIR,
    convert_repo_ascii_thruputs,
    install_thruputs_to_cdbs,
    load_bandpass,
    nonhst_dir,
    registered_bands,
    thruput_path_for_band,
)


def test_converted_bands_match_ascii_sources() -> None:
    assert "PS_g" in CONVERTED_BANDS
    assert "PS_w" in CONVERTED_BANDS
    assert "Swift_UVW1" in CONVERTED_BANDS
    assert "Swift_white" in CONVERTED_BANDS
    for band in CONVERTED_BANDS:
        spec = BAND_REGISTRY[band]
        assert spec.ascii_source is not None
        assert (DATA_DIR / spec.ascii_source).is_file()


def test_convert_install_and_load_nonzero(tmp_path: Path) -> None:
    out = tmp_path / "thruputs"
    written = convert_repo_ascii_thruputs(out_dir=out, overwrite=True)
    assert set(written) == set(CONVERTED_BANDS)

    fake_cdbs = tmp_path / "trds"
    installed = install_thruputs_to_cdbs(
        cdbs_root=fake_cdbs,
        source_dir=out,
        convert_if_missing=False,
    )
    assert set(installed) == set(CONVERTED_BANDS)
    nh = nonhst_dir(fake_cdbs)
    assert nh.is_dir()

    for band in CONVERTED_BANDS:
        path = thruput_path_for_band(
            band, cdbs_root=fake_cdbs, allow_repo_fallback=False
        )
        assert path.is_file()
        assert path.parent == nh
        bp = load_bandpass(band, cdbs_root=fake_cdbs)
        waves = bp.waveset
        assert waves is not None and len(waves) >= 2
        thru = np.asarray(bp(waves).value, dtype=float)
        assert np.all(np.isfinite(thru))
        assert float(np.nanmax(thru)) > 0.0


@pytest.mark.parametrize("band", list(CONVERTED_BANDS))
def test_load_each_converted_bandpass_nonzero(band: str, tmp_path: Path) -> None:
    """Load every PS1/Swift band via repo fallback; thruput peak must be > 0."""
    empty_cdbs = tmp_path / "empty_trds"
    empty_cdbs.mkdir()
    bp = load_bandpass(band, cdbs_root=empty_cdbs)
    waves = bp.waveset
    assert waves is not None
    thru = np.asarray(bp(waves).value, dtype=float)
    assert float(np.nanmax(thru)) > 0.0


def test_registered_bands_api() -> None:
    all_bands = registered_bands()
    converted = registered_bands(converted_only=True)
    assert set(converted) == set(CONVERTED_BANDS)
    assert set(converted).issubset(set(all_bands))
    assert "GaiaDR3_G" in all_bands
    assert "PS_g" in all_bands
    assert "Swift_U" in all_bands
    assert "WISE_W1" in all_bands
    assert "WISE_W2" in all_bands
    # Path-2: W3/W4 excluded (beyond PHOENIX HiRes red end).
    assert "WISE_W3" not in all_bands
    assert "WISE_W4" not in all_bands


def test_unknown_band_raises() -> None:
    with pytest.raises(KeyError, match="Unknown band"):
        load_bandpass("NotARealBand_xyz")
