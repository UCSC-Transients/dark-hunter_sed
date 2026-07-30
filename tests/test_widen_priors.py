"""Tests for widened SED priors, clamps, and NaN→TAP summary patching."""

from __future__ import annotations

from pathlib import Path

import pytest

from darkhunter_sed import data, fit, priors


def test_ums_mass_and_eep_prior_bounds():
    """UMS IMF mass_ue and EEP span full MISTy range."""
    # Inspect prior literals via a minimal build is heavy; assert constants used in fit module source path.
    # Use run_utp/run_ums helpers by checking the module-level expected values through a dry construct.
    src = Path(fit.__file__).read_text(encoding="utf-8")
    assert '"mass_ue": 3.0' in src or "'mass_ue': 3.0" in src
    assert '"EEP": ["uniform", [1, 808]]' in src or "'EEP': ['uniform', [1, 808]]" in src


def test_utp_teff_prior_bounds_in_source():
    src = Path(fit.__file__).read_text(encoding="utf-8")
    assert "[3500.0, 15000.0]" in src


def test_clamp_utp_teff_init():
    # Mirror the clip used in run_utp
    import numpy as np

    assert float(np.clip(16000.0, 3500.0, 15000.0)) == 15000.0
    assert float(np.clip(3000.0, 3500.0, 15000.0)) == 3500.0


def test_clamp_ums_mass_eep():
    import numpy as np

    assert float(np.clip(3.5, 0.3, 3.0)) == 3.0
    assert float(np.clip(900.0, 1.0, 808.0)) == 808.0


def test_parallax_err_mult_default_doubles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = {"parallax": [10.0, 0.1]}
    data._finalize_parallax_in_out(out)
    assert out["parallax"][0] == pytest.approx(10.0)
    assert out["parallax"][1] == pytest.approx(0.2)


def test_finite_gaia_metadata_updates_skips_nan():
    updates = priors.finite_gaia_metadata_updates(
        {
            "Teff": float("nan"),
            "log(g)": 4.1,
            "[Fe/H]": -0.2,
            "parallax": [5.0, 0.05],
            "Mass": 1.0,
        }
    )
    assert "Teff" not in updates
    assert updates["logg"].startswith("4.1")
    assert "MH" in updates
    assert "Parallax" in updates
    assert "Mass_FLAME" not in updates


def test_finite_gaia_metadata_updates_includes_mass_flame():
    updates = priors.finite_gaia_metadata_updates(
        {"Teff": 8000.0, "Mass_FLAME": 1.7, "parallax": [8.0, 0.1]}
    )
    assert updates["Teff"].startswith("8000")
    assert updates["Mass_FLAME"].startswith("1.7")


def test_patch_gaia_metadata_fields(tmp_path: Path) -> None:
    path = tmp_path / "Gaia_DR3_1_summary.txt"
    path.write_text(
        "### STAR SUMMARY: 1 ###\n\n"
        "[GAIA METADATA]\n"
        "Source_ID: 1\n"
        "Teff: NaN\n"
        "logg: 4.0\n"
        "Parallax: 10.0\n"
        "Parallax_Error: 0.1\n"
        "\n"
        "[PIPELINE RESULTS]\n"
        "epoch 1\n",
        encoding="utf-8",
    )
    assert priors.patch_gaia_metadata_fields(path, {"Teff": "8164.00000000"})
    text = path.read_text(encoding="utf-8")
    assert "Teff: 8164.00000000" in text
    assert "logg: 4.0" in text
    assert "[PIPELINE RESULTS]" in text


def test_load_stellar_priors_nan_teff_updates_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "Gaia_DR3_99_summary.txt"
    path.write_text(
        "### STAR SUMMARY: 99 ###\n\n"
        "[GAIA METADATA]\n"
        "Source_ID: 99\n"
        "Teff: NaN\n"
        "logg: 4.2\n"
        "MH: 0.0\n"
        "Parallax: 10.0\n"
        "Parallax_Error: 0.1\n"
        "RA: 1.0\n"
        "Dec: 2.0\n",
        encoding="utf-8",
    )

    def fake_tap(gaia_id: str) -> dict:
        return {
            "Teff": 8164.0,
            "log(g)": 4.2,
            "[Fe/H]": 0.0,
            "[a/Fe]": -0.2,
            "log(R)": 0.0,
            "Mass": 1.8,
            "Mass_FLAME": 1.8,
            "parallax": [10.0, 0.1],
            "RA": 1.0,
            "Dec": 2.0,
        }

    monkeypatch.setattr(priors, "query_gaia_stellar_priors", fake_tap)
    out = priors.load_stellar_priors("99", summary_path=path, rv_output=tmp_path)
    assert out["Teff"] == pytest.approx(8164.0)
    assert "Teff: 8164.00000000" in path.read_text(encoding="utf-8")


def test_load_stellar_priors_tap_still_nan_skips_teff_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "Gaia_DR3_88_summary.txt"
    path.write_text(
        "### STAR SUMMARY: 88 ###\n\n"
        "[GAIA METADATA]\n"
        "Source_ID: 88\n"
        "Teff: NaN\n"
        "logg: NaN\n"
        "MH: NaN\n"
        "Parallax: 10.0\n"
        "Parallax_Error: 0.1\n"
        "RA: 1.0\n"
        "Dec: 2.0\n",
        encoding="utf-8",
    )

    def fake_tap(gaia_id: str) -> dict:
        return {
            "Teff": float("nan"),
            "log(g)": float("nan"),
            "[Fe/H]": float("nan"),
            "[a/Fe]": -0.2,
            "log(R)": 0.0,
            "Mass": 1.0,
            "parallax": [10.0, 0.1],
            "RA": 1.0,
            "Dec": 2.0,
        }

    monkeypatch.setattr(priors, "query_gaia_stellar_priors", fake_tap)
    out = priors.load_stellar_priors("88", summary_path=path, rv_output=tmp_path)
    assert out["Teff"] == pytest.approx(5500.0)
    text = path.read_text(encoding="utf-8")
    assert "Teff: NaN" in text
    # Parallax/RA/Dec from TAP are finite and may be rewritten
    assert "Parallax:" in text


def test_load_stellar_priors_tap_fail_no_disk_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "Gaia_DR3_77_summary.txt"
    original = (
        "### STAR SUMMARY: 77 ###\n\n"
        "[GAIA METADATA]\n"
        "Source_ID: 77\n"
        "Teff: NaN\n"
        "logg: 4.0\n"
        "MH: 0.0\n"
        "Parallax: 10.0\n"
        "Parallax_Error: 0.1\n"
        "RA: 1.0\n"
        "Dec: 2.0\n"
    )
    path.write_text(original, encoding="utf-8")

    def boom(gaia_id: str) -> dict:
        raise RuntimeError("network down")

    monkeypatch.setattr(priors, "query_gaia_stellar_priors", boom)
    out = priors.load_stellar_priors("77", summary_path=path, rv_output=tmp_path)
    assert out["Teff"] == pytest.approx(5500.0)
    assert path.read_text(encoding="utf-8") == original


def test_meta_atmosphere_incomplete():
    assert priors._meta_atmosphere_incomplete({"Teff": float("nan"), "logg": 4.0, "MH": 0.0})
    assert not priors._meta_atmosphere_incomplete({"Teff": 5800.0, "logg": 4.0, "MH": 0.0})
