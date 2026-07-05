"""Tests for legacy RV summary parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from darkhunter_sed import priors


def test_parse_legacy_file_summary(tmp_path: Path):
    summary = tmp_path / "legacy_summary.txt"
    summary.write_text(
        """# File Summary
# Input File | MJD | RV | Err | RMS
Gaia_DR3_99_epoch_1.txt 60532.5 -1.62 0.003 0.3
Gaia_DR3_99_epoch_2.txt 60540.1 -2.10 0.004 0.2
[GAIA METADATA]
Source_ID = 99
""",
        encoding="utf-8",
    )
    epochs = priors.parse_legacy_file_summary_rv_epochs(summary)
    assert len(epochs) == 2
    assert epochs[0].epoch_num == 1
    assert epochs[0].rv_kms == pytest.approx(-1.62)
    assert epochs[1].epoch_num == 2


def test_parse_rv_epochs_prefers_pipeline(tmp_path: Path):
    summary = tmp_path / "both.txt"
    summary.write_text(
        """# File Summary
Gaia_DR3_1_epoch_1.txt 1.0 0.0 0.1
[PIPELINE RESULTS]
Gaia_DR3_1_epoch_1.txt 58000.0 -5.0 1.0
""",
        encoding="utf-8",
    )
    epochs = priors.parse_rv_epochs_from_summary(summary)
    assert len(epochs) == 1
    assert epochs[0].rv_kms == pytest.approx(-5.0)
