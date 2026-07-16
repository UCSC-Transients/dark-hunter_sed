"""Tests for batch --update skip logic."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from darkhunter_sed import batch, posterior
from darkhunter_sed import config as sed_config


def test_needs_update_when_no_outputs(tmp_path, monkeypatch):
    gid = "1702370142434513152"
    spec_root_path = tmp_path / "spec"
    phot_dir = tmp_path / "phot"
    rv_out = tmp_path / "rv"
    spec_dir = spec_root_path / f"Gaia_DR3_{gid}"
    spec_dir.mkdir(parents=True)
    phot_dir.mkdir(parents=True)

  # two epoch files
    for ep in (1, 2):
        p = spec_dir / f"Gaia_DR3_{gid}_epoch_{ep}.txt"
        p.write_text("# Order 1\n5000 1.0\n", encoding="utf-8")

    phot = phot_dir / f"{gid}_phot.fits"
    phot.write_bytes(b"\x00")  # placeholder; batch only checks mtime

    monkeypatch.setattr(sed_config, "samples_dir", lambda: tmp_path / "samples")
    monkeypatch.setattr(sed_config, "sed_summaries_dir", lambda: tmp_path / "sed_summaries")

    ok, reason = batch.needs_update(
        gid,
        spec_root_path=spec_root_path,
        phot_dir=phot_dir,
        rv_out=rv_out,
    )
    assert ok is True
    assert reason == "no prior sed_summary"


def test_needs_update_skips_when_outputs_newer(tmp_path, monkeypatch):
    gid = "99"
    spec_root_path = tmp_path / "spec"
    phot_dir = tmp_path / "phot"
    rv_out = tmp_path / "rv"
    summaries = tmp_path / "sed_summaries"
    samples = tmp_path / "samples"
    summaries.mkdir(parents=True)
    samples.mkdir(parents=True)

    spec_dir = spec_root_path / f"Gaia_DR3_{gid}"
    spec_dir.mkdir(parents=True)
    for ep in (1, 2):
        (spec_dir / f"Gaia_DR3_{gid}_epoch_{ep}.txt").write_text("# Order 1\n5000 1.0\n")

    phot = phot_dir / f"{gid}_phot.fits"
    phot.parent.mkdir(parents=True, exist_ok=True)
    phot.write_bytes(b"x")

    json_path = summaries / f"Gaia_DR3_{gid}_sed_summary.json"
    json_path.write_text(json.dumps({"gaia_source_id": gid}), encoding="utf-8")

    ums = samples / f"Gaia_DR3_{gid}_ums.fits"
    ums.write_bytes(b"x")

    # Make outputs newer than inputs
    now = time.time() + 100
    for p in (json_path, ums, phot):
        import os

        os.utime(p, (now, now))
    for ep in (1, 2):
        import os

        os.utime(spec_dir / f"Gaia_DR3_{gid}_epoch_{ep}.txt", (now - 200, now - 200))

    monkeypatch.setattr(sed_config, "samples_dir", lambda: samples)
    monkeypatch.setattr(sed_config, "sed_summaries_dir", lambda: summaries)

    ok, reason = batch.needs_update(
        gid,
        spec_root_path=spec_root_path,
        phot_dir=phot_dir,
        rv_out=rv_out,
    )
    assert ok is False
    assert reason == "up to date"


def test_needs_update_single_epoch_allowed(tmp_path, monkeypatch):
    gid = "42"
    spec_root_path = tmp_path / "spec"
    phot_dir = tmp_path / "phot"
    rv_out = tmp_path / "rv"
    spec_dir = spec_root_path / f"Gaia_DR3_{gid}"
    spec_dir.mkdir(parents=True)
    phot_dir.mkdir(parents=True)

    (spec_dir / f"Gaia_DR3_{gid}_epoch_1.txt").write_text("# Order 1\n5000 1.0\n")
    (phot_dir / f"{gid}_phot.fits").write_bytes(b"x")

    monkeypatch.setattr(sed_config, "samples_dir", lambda: tmp_path / "samples")
    monkeypatch.setattr(sed_config, "sed_summaries_dir", lambda: tmp_path / "sed_summaries")

    ok, reason = batch.needs_update(
        gid,
        spec_root_path=spec_root_path,
        phot_dir=phot_dir,
        rv_out=rv_out,
    )
    assert ok is True
    assert reason == "no prior sed_summary"


def test_needs_update_missing_phot_with_auto_gather(tmp_path, monkeypatch):
    gid = "2254791692199089536"
    spec_root_path = tmp_path / "spec"
    phot_dir = tmp_path / "phot"
    rv_out = tmp_path / "rv"
    spec_dir = spec_root_path / f"Gaia_DR3_{gid}"
    spec_dir.mkdir(parents=True)
    phot_dir.mkdir(parents=True)

    for ep in (1, 2):
        (spec_dir / f"Gaia_DR3_{gid}_epoch_{ep}.txt").write_text("# Order 1\n5000 1.0\n")

    monkeypatch.setattr(sed_config, "samples_dir", lambda: tmp_path / "samples")
    monkeypatch.setattr(sed_config, "sed_summaries_dir", lambda: tmp_path / "sed_summaries")

    ok, reason = batch.needs_update(
        gid,
        spec_root_path=spec_root_path,
        phot_dir=phot_dir,
        rv_out=rv_out,
        auto_gather_photometry=True,
    )
    assert ok is True
    assert reason == "missing photometry (will gather)"

