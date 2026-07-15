"""Tests for ensure_photometry_fits auto-gather behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from darkhunter_sed.photometry_gather import (
    PS1_VIZIER_DEFAULT_EPOCH_JY,
    ensure_photometry_fits,
    gather_photometry_for_star,
)


def test_ensure_photometry_returns_existing(tmp_path: Path) -> None:
    gid = "1702370142434513152"
    path = tmp_path / f"{gid}_phot.fits"
    path.write_bytes(b"existing")

    out = ensure_photometry_fits(gid, tmp_path, auto_gather=False)
    assert out == path.resolve()


def test_ensure_photometry_gathers_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gid = "2254791692199089536"

    def fake_gather(gaia_id: str, *, outdir: Path | None = None, **kwargs: object) -> Path:
        assert gaia_id == gid
        out = Path(outdir) / f"{gaia_id}_phot.fits"
        out.write_bytes(b"gathered")
        return out.resolve()

    monkeypatch.setattr(
        "darkhunter_sed.photometry_gather.gather_photometry_for_star",
        fake_gather,
    )

    out = ensure_photometry_fits(gid, tmp_path, auto_gather=True)
    assert out == (tmp_path / f"{gid}_phot.fits").resolve()
    assert out.read_bytes() == b"gathered"


def test_ensure_photometry_raises_without_auto_gather(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Photometry file not found"):
        ensure_photometry_fits("999", tmp_path, auto_gather=False)


def test_gather_none_ps1_epoch_uses_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ensure_photometry_fits path leaves epoch None; must coerce before Vizier print."""
    captured: dict[str, object] = {}

    def fake_query(
        source_id: str,
        radius: float = 3,
        ps1_vizier_epoch_jyear: float | None = None,
    ) -> list[tuple[str, float, float]]:
        captured["epoch"] = ps1_vizier_epoch_jyear
        return [("GaiaDR3_G", 10.0, 0.01)]

    def fake_save(
        source_id: str,
        photometry: list[tuple[str, float, float]],
        outdir: Path | None = None,
    ) -> Path:
        out = Path(outdir) / f"{source_id}_phot.fits"
        out.write_bytes(b"ok")
        return out.resolve()

    monkeypatch.setattr(
        "darkhunter_sed.photometry_gather.query_catalogs", fake_query
    )
    monkeypatch.setattr(
        "darkhunter_sed.photometry_gather.save_photometry_to_fits", fake_save
    )

    out = gather_photometry_for_star(
        "4219507576765009536",
        outdir=tmp_path,
        ps1_vizier_epoch_jyear=None,
    )
    assert out.is_file()
    assert captured["epoch"] == PS1_VIZIER_DEFAULT_EPOCH_JY
