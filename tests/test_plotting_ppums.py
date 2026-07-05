"""Smoke test for ppUMS-style posterior plotting with mocked spectral models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from darkhunter_sed import plotting_ppums


def _write_samples(path: Path, *, pc_fixed: bool, n: int = 40) -> None:
    rng = np.random.default_rng(0)
    cols = {
        "Teff": rng.normal(5800, 50, n),
        "log(g)": rng.normal(4.3, 0.05, n),
        "[Fe/H]": rng.normal(-0.2, 0.05, n),
        "[a/Fe]": rng.normal(0.05, 0.02, n),
        "vstar": rng.normal(3.0, 0.2, n),
        "vmic": rng.normal(1.0, 0.05, n),
        "vrad_0": rng.normal(-11.0, 0.1, n),
        "lsf_0": rng.normal(60000.0, 10.0, n),
        "log(R)": rng.normal(0.0, 0.01, n),
        "dist": rng.normal(100.0, 1.0, n),
        "Av": np.abs(rng.normal(0.1, 0.02, n)),
        "log(L)": rng.normal(0.0, 0.02, n),
    }
    if pc_fixed:
        cols["pc0_0"] = np.ones(n)
        cols["pc1_0"] = np.zeros(n)
        cols["pc2_0"] = np.zeros(n)
        cols["pc3_0"] = np.zeros(n)
    else:
        cols["pc0_0"] = rng.normal(1.0, 0.02, n)
        cols["pc1_0"] = rng.normal(0.0, 0.01, n)
        cols["pc2_0"] = rng.normal(0.0, 0.01, n)
        cols["pc3_0"] = rng.normal(0.0, 0.01, n)
    Table(cols).write(path, format="fits", overwrite=True)


class _FakeGM:
    """Records the modpoly flag genspec is called with."""

    modpoly_calls: list[bool] = []

    def _initspecnn(self, **_kw):
        pass

    def _initphotnn(self, *_a, **_kw):
        pass

    def genspec(self, pars, *, outwave, modpoly):
        type(self).modpoly_calls.append(modpoly)
        return outwave, np.ones_like(outwave)

    def genphot(self, pars):
        return {"PS_g": 12.0, "PS_r": 11.5}


@pytest.fixture
def fit_data():
    w = np.linspace(5160.0, 5290.0, 400)
    return {
        "phot": {"PS_g": [12.0, 0.02], "PS_r": [11.5, 0.02]},
        "phot_filtarr": ["PS_g", "PS_r"],
        "spec": [
            {
                "obs_wave": w,
                "obs_flux": np.ones_like(w) + np.random.normal(0, 0.01, w.size),
                "obs_eflux": np.full_like(w, 0.01),
            }
        ],
    }


def _patch(monkeypatch):
    _FakeGM.modpoly_calls = []
    monkeypatch.setattr(plotting_ppums, "ensure_stellar_sys_path", lambda *a, **k: Path("."))
    monkeypatch.setattr(plotting_ppums, "resolve_models_dir", lambda *a, **k: Path("."))
    monkeypatch.setattr(plotting_ppums, "resolve_spec_nn_path", lambda *a, **k: "specNN")
    monkeypatch.setattr(plotting_ppums, "normalize_phot_nn_dir", lambda *a, **k: "photNN")
    monkeypatch.setattr(plotting_ppums, "import_payne_genmod", lambda: _FakeGM)
    # mkphot / mkkiel require real NN + MIST weights; both are guarded, force skip.
    monkeypatch.setattr(plotting_ppums, "_mkphot", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no weights")))
    monkeypatch.setattr(plotting_ppums, "_mkkiel", lambda *a, **k: None)


def test_rigid_ums_uses_modpoly_false(tmp_path, monkeypatch, fit_data):
    _patch(monkeypatch)
    samples = tmp_path / "ums.fits"
    _write_samples(samples, pc_fixed=True)
    out = plotting_ppums.plot_ums_posterior(
        "123", fit_data, samples, fit_type="ums", out_path=tmp_path / "ums.pdf"
    )
    assert out.is_file()
    assert _FakeGM.modpoly_calls, "genspec never called"
    assert all(mp is False for mp in _FakeGM.modpoly_calls)


def test_flexible_pc_uses_modpoly_true(tmp_path, monkeypatch, fit_data):
    _patch(monkeypatch)
    samples = tmp_path / "utp.fits"
    _write_samples(samples, pc_fixed=False)
    out = plotting_ppums.plot_utp_posterior(
        "123", fit_data, samples, out_path=tmp_path / "utp.pdf"
    )
    assert out.is_file()
    assert all(mp is True for mp in _FakeGM.modpoly_calls)


def test_flux_plot_xlim_never_narrower_than_default():
    # In-band phot only: keep legacy 0.25–6 µm window.
    xlo, xhi = plotting_ppums._flux_plot_xlim(np.array([0.5, 2.0, 5.0]))
    assert xlo == plotting_ppums.FLUX_XLIM_DEFAULT[0]
    assert xhi == plotting_ppums.FLUX_XLIM_DEFAULT[1]


def test_flux_plot_xlim_widens_for_uv_phot():
    # GALEX-scale UV should extend left of default xmin.
    xlo, xhi = plotting_ppums._flux_plot_xlim(np.array([0.15, 0.5, 2.0]))
    assert xlo < plotting_ppums.FLUX_XLIM_DEFAULT[0]
    assert xhi >= plotting_ppums.FLUX_XLIM_DEFAULT[1]
