"""Tests for push_m1 (summary + data.csv)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from darkhunter_sed import push_m1


def _sed_doc(m1: float = 0.85, p16: float = 0.80, p84: float = 0.90) -> dict:
    return {
        "gaia_source_id": "123",
        "m1_msun": {"median": m1, "p16": p16, "p84": p84},
    }


def test_patch_summary_m1_inserts_keys(tmp_path):
    summ = tmp_path / "Gaia_DR3_123_summary.txt"
    summ.write_text(
        "### STAR SUMMARY: 123 ###\n\n"
        "[GAIA METADATA]\n"
        "Source_ID: 123\n"
        "Teff: 5800.0\n"
        "\n"
        "[PIPELINE RESULTS]\n"
        "# File | MJD\n",
        encoding="utf-8",
    )
    assert push_m1.patch_summary_m1(summ, {"median": 0.91, "p16": 0.88, "p84": 0.95})
    text = summ.read_text(encoding="utf-8")
    assert "M1: 0.910000" in text
    assert "m1_msun: 0.910000" in text
    assert "M1_p16: 0.880000" in text
    assert "M1_p84: 0.950000" in text
    assert text.index("M1:") < text.index("[PIPELINE RESULTS]")


def test_patch_summary_m1_updates_existing(tmp_path):
    summ = tmp_path / "Gaia_DR3_123_summary.txt"
    summ.write_text(
        "[GAIA METADATA]\nSource_ID: 123\nM1: 1.000000\nm1_msun: 1.000000\n\n[PIPELINE RESULTS]\n",
        encoding="utf-8",
    )
    assert push_m1.patch_summary_m1(summ, {"median": 0.77})
    text = summ.read_text(encoding="utf-8")
    assert "M1: 0.770000" in text
    assert "1.000000" not in text


def test_patch_data_csv_m1(tmp_path):
    csv_path = tmp_path / "data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["GAIA NAME", "M1 (Msun)", "M2 (Msun)"])
        w.writerow(["Gaia_DR3_999999999999999999", "1.0", ""])
        w.writerow(["Gaia_DR3_123456789012345678", "0.5", "0.1"])
    assert push_m1.patch_data_csv_m1(csv_path, "123456789012345678", 0.91234)
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[2][1] == "0.91234"
    assert rows[1][1] == "1.0"


def test_push_m1_for_gaia_id(tmp_path, monkeypatch):
    gid = "555555555555555555"
    summaries = tmp_path / "sed_summaries"
    summaries.mkdir()
    sed = summaries / f"Gaia_DR3_{gid}_sed_summary.json"
    sed.write_text(json.dumps(_sed_doc(0.82)), encoding="utf-8")

    rv_out = tmp_path / "rv"
    rv_out.mkdir()
    summ = rv_out / f"Gaia_DR3_{gid}_summary.txt"
    summ.write_text(
        f"[GAIA METADATA]\nSource_ID: {gid}\n\n[PIPELINE RESULTS]\n",
        encoding="utf-8",
    )

    data_csv = tmp_path / "data.csv"
    with data_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["GAIA NAME", "M1 (Msun)"])
        w.writerow([f"Gaia_DR3_{gid}", ""])

    result = push_m1.push_m1_for_gaia_id(
        gid,
        summaries_dir=summaries,
        rv_out=rv_out,
        data_csv=data_csv,
    )
    assert result["ok"] is True
    assert result["m1_msun"] == 0.82
    assert result["summary_updated"] is True
    assert result["csv_updated"] is True
    assert "M1: 0.820000" in summ.read_text(encoding="utf-8")
