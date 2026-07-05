"""Tests for ensure_photometry_fits auto-gather behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from darkhunter_sed.photometry_gather import ensure_photometry_fits


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
