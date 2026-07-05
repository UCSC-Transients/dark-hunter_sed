"""Tests for regions JSON resolution and getdata plumbing."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from darkhunter_sed import data
from darkhunter_sed.region_picker import resolve_regions_json_for_star


def test_resolve_regions_json_explicit_path(tmp_path: Path) -> None:
    regions = tmp_path / "custom_regions.json"
    regions.write_text("{}", encoding="utf-8")
    resolved = resolve_regions_json_for_star("123", regions)
    assert resolved == regions.resolve()


def test_resolve_regions_json_newest_in_masks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("darkhunter_sed.region_picker.masks_dir", lambda: tmp_path)
    gid = "77413727493690112"
    older = tmp_path / f"regions_Gaia_DR3_{gid}_epoch_1.json"
    newer = tmp_path / f"regions_Gaia_DR3_{gid}_epoch_2.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")
    base = time.time()
    os.utime(older, (base - 100, base - 100))
    os.utime(newer, (base, base))

    resolved = resolve_regions_json_for_star(gid)
    assert resolved == newer.resolve()


def test_load_spectra_from_paths_passes_regions_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tmp_path / "Gaia_DR3_99_epoch_1.txt"
    spec.write_text("# Order 1\n5000 1.0\n", encoding="utf-8")
    regions = tmp_path / "regions.json"
    regions.write_text("{}", encoding="utf-8")
    seen: list[str | None] = []

    def fake_process(path: Path, **kwargs: object) -> dict:
        seen.append(kwargs.get("regions_json"))
        return {"path": str(path)}

    monkeypatch.setattr(data.spectrum, "process_apf_txt_spectrum", fake_process)
    out = data.load_spectra_from_paths([spec], regions_json=str(regions))
    assert len(out) == 1
    assert seen == [str(regions)]


def test_getdata_records_resolved_regions_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gid = "2254791692199089536"
    regions = tmp_path / f"regions_Gaia_DR3_{gid}_epoch_1.json"
    regions.write_text("{}", encoding="utf-8")
    phot_path = tmp_path / f"{gid}_phot.fits"
    phot_path.write_bytes(b"phot")

    monkeypatch.setattr(
        "darkhunter_sed.photometry_gather.ensure_photometry_fits",
        lambda gaia_id, phot_dir, auto_gather=True: phot_path.resolve(),
    )
    monkeypatch.setattr(
        data,
        "load_photometry_fits",
        lambda gaia_id, phot_dir=None: ({"GaiaDR3_G": [10.0, 0.02]}, ["GaiaDR3_G"]),
    )
    monkeypatch.setattr(
        data.priors,
        "load_stellar_priors",
        lambda gaia_id, **kwargs: {
            "parallax": [1.0, 0.1],
            "gaia_source_id": gaia_id,
        },
    )
    monkeypatch.setattr(
        data,
        "restrict_photometry_to_phot_nn_bands",
        lambda phot_nn, phot, filt: (phot, filt),
    )
    monkeypatch.setattr(data, "require_phot_nn_for_bands", lambda phot_nn, filt: None)
    monkeypatch.setattr(
        "darkhunter_sed.config.masks_dir",
        lambda: tmp_path,
    )

    out = data.getdata(
        gid,
        phot_nn="dummy",
        require_phot_nn=False,
        regions_json=regions,
        auto_gather_photometry=False,
    )
    assert out["regions_json"] == str(regions.resolve())
