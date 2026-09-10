"""
Smoke tests for Path-2 SED diagnostic plots (Issue #46).

All tests use synthetic data arrays — no real PHOENIX grid, MISTy weights,
Bergeron tables, or display backend required.
``matplotlib.use("Agg")`` is called at module import to guarantee headless
operation in CI.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must precede any other matplotlib import

import numpy as np
import pytest

from darkhunter_sed.phot_sed_io import FLAG_DETECTION, FLAG_UPPER_LIMIT, PhotRow
from darkhunter_sed.phot_sed_diagnostics import (
    BAND_EFF_WAVE_AA,
    _wave_um,
    _sorted_bands_by_wave,
    plot_sed_panel,
    plot_wd_corner,
    plot_model_comparison,
    save_all_diagnostics,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_DET_BANDS = ["GaiaDR3_G", "GaiaDR3_BP", "PS_r", "PS_i", "2MASS_J"]
_UL_BANDS  = ["WISE_W1", "WISE_W2"]


def _make_rows() -> list[PhotRow]:
    """Return a small synthetic photometry row list with detections + upper limits."""
    rows: list[PhotRow] = []
    # Detections: mag decreases slightly with wavelength (red star)
    for i, band in enumerate(_DET_BANDS):
        rows.append(PhotRow(band=band, mag=18.0 + 0.3 * i, err=0.05, flag=FLAG_DETECTION))
    # Upper limits
    for band in _UL_BANDS:
        rows.append(PhotRow(band=band, mag=17.5, err=0.1, flag=FLAG_UPPER_LIMIT))
    return rows


def _make_model_preds(rows: list[PhotRow]) -> dict[str, dict[str, float]]:
    """Return synthetic best-fit predictions that cover all detection bands."""
    preds: dict[str, float] = {}
    for r in rows:
        if r.flag == FLAG_DETECTION:
            preds[r.band] = r.mag + 0.1  # slight offset from truth
    return {"1-star": preds}


def _make_wd_result(
    atm_type: str = "DA",
    ifmr: str = "MIST",
    n_samples: int = 40,
    seed: int = 42,
) -> object:
    """
    Construct a synthetic WDFitResult-like object with all required arrays.

    Returns a simple namespace that matches the WDFitResult field interface
    expected by plot_wd_corner.
    """
    import types

    rng = np.random.default_rng(seed)
    samples = np.column_stack([
        rng.uniform(1.0, 10.0, n_samples),   # parallax_mas
        rng.uniform(0.0, 1.0, n_samples),    # Av
        rng.uniform(5000.0, 20000.0, n_samples),  # teff_wd (unused in corner)
        rng.uniform(7.5, 9.0, n_samples),    # logg_wd  (unused in corner)
    ])
    # Reorder to match _PARAM_NAMES = ("teff_wd", "logg_wd", "av", "parallax_mas")
    samples_ordered = np.column_stack([
        samples[:, 2],  # teff
        samples[:, 3],  # logg
        samples[:, 1],  # av
        samples[:, 0],  # parallax
    ])
    weights = rng.dirichlet(np.ones(n_samples))
    m_wd = rng.uniform(0.5, 1.0, n_samples)
    m_i  = rng.uniform(1.5, 4.0, n_samples)
    t_cool_yr = rng.uniform(1e8, 5e9, n_samples)

    res = types.SimpleNamespace(
        atm_type=atm_type,
        ifmr=ifmr,
        logevidence=-42.0,
        samples=samples_ordered,
        weights=weights,
        m_wd_samples=m_wd,
        m_i_samples=m_i,
        m_i_unc_ifmr_samples=rng.uniform(0.1, 0.5, n_samples),
        extrap_mask=np.zeros(n_samples, dtype=bool),
        t_cool_yr_samples=t_cool_yr,
    )
    return res


def _make_prior_bounds() -> object:
    """Return a minimal WDPriorBounds-compatible namespace."""
    import types
    return types.SimpleNamespace(
        parallax_mas=(0.1, 100.0),
        av=(0.0, 5.0),
    )


# ---------------------------------------------------------------------------
# BAND_EFF_WAVE_AA
# ---------------------------------------------------------------------------

def test_band_eff_wave_aa_known_bands() -> None:
    """Core bands must have entries with positive wavelengths."""
    for band in ["GaiaDR3_G", "GaiaDR3_BP", "PS_r", "2MASS_J", "WISE_W1"]:
        assert band in BAND_EFF_WAVE_AA
        assert BAND_EFF_WAVE_AA[band] > 0


def test_wave_um_known() -> None:
    wv = _wave_um("GaiaDR3_G")
    assert wv is not None
    assert 0.5 < wv < 0.8  # Gaia G ~ 0.62 µm


def test_wave_um_unknown() -> None:
    assert _wave_um("NONEXISTENT_BAND") is None


def test_sorted_bands_by_wave_order() -> None:
    bands = ["2MASS_J", "GaiaDR3_G", "GALEX_FUV"]
    sorted_b = _sorted_bands_by_wave(bands)
    waves = [BAND_EFF_WAVE_AA[b] for b in sorted_b]
    assert waves == sorted(waves)


# ---------------------------------------------------------------------------
# plot_sed_panel
# ---------------------------------------------------------------------------

def test_plot_sed_panel_returns_figure() -> None:
    """plot_sed_panel must return a Figure without raising."""
    import matplotlib.figure
    rows = _make_rows()
    preds = _make_model_preds(rows)
    fig = plot_sed_panel(rows, preds, gaia_id="test_001")
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_plot_sed_panel_empty_model_preds() -> None:
    """Empty model_preds is allowed; residual panel stays blank."""
    import matplotlib.pyplot as plt
    rows = _make_rows()
    fig = plot_sed_panel(rows, {}, gaia_id="test_empty")
    assert fig is not None
    plt.close(fig)


def test_plot_sed_panel_multi_model() -> None:
    """Multiple model entries must all be drawn without error."""
    import matplotlib.pyplot as plt
    rows = _make_rows()
    preds_base: dict[str, float] = {r.band: r.mag + 0.05 for r in rows if r.flag == FLAG_DETECTION}
    model_preds = {
        "1-star":     preds_base,
        "2-star sum": {b: v - 0.2 for b, v in preds_base.items()},
        "star1":      {b: v + 0.5 for b, v in preds_base.items()},
        "star2":      {b: v + 1.0 for b, v in preds_base.items()},
    }
    fig = plot_sed_panel(rows, model_preds, gaia_id="test_multi")
    assert fig is not None
    plt.close(fig)


def test_plot_sed_panel_saves_file(tmp_path: Path) -> None:
    """outpath causes the figure to be written to disk."""
    import matplotlib.pyplot as plt
    rows = _make_rows()
    preds = _make_model_preds(rows)
    outpath = tmp_path / "sed.pdf"
    plot_sed_panel(rows, preds, outpath=outpath)
    assert outpath.is_file()
    assert outpath.stat().st_size > 0
    plt.close("all")


def test_plot_sed_panel_upper_limits_only() -> None:
    """All-UL rows must not raise."""
    import matplotlib.pyplot as plt
    rows = [PhotRow("WISE_W1", 17.0, 0.1, FLAG_UPPER_LIMIT)]
    fig = plot_sed_panel(rows, {}, gaia_id="ul_only")
    assert fig is not None
    plt.close(fig)


def test_plot_sed_panel_unknown_band_skipped() -> None:
    """Bands absent from BAND_EFF_WAVE_AA are silently skipped."""
    import matplotlib.pyplot as plt
    rows = [
        PhotRow("UNKNOWN_BAND", 18.0, 0.05, FLAG_DETECTION),
        PhotRow("GaiaDR3_G",    18.0, 0.05, FLAG_DETECTION),
    ]
    preds = {"1-star": {"GaiaDR3_G": 18.1}}
    fig = plot_sed_panel(rows, preds, gaia_id="unknown")
    assert fig is not None
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_wd_corner
# ---------------------------------------------------------------------------

def test_plot_wd_corner_single_result() -> None:
    """Single WDFitResult must produce a valid Figure."""
    import matplotlib.pyplot as plt
    import matplotlib.figure
    res = _make_wd_result("DA", "MIST")
    bounds = _make_prior_bounds()
    fig = plot_wd_corner([res], bounds, gaia_id="wd_test")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_wd_corner_four_results() -> None:
    """Four WD variants must all overlay without error."""
    import matplotlib.pyplot as plt
    bounds = _make_prior_bounds()
    results = [
        _make_wd_result("DA", "MIST",   seed=1),
        _make_wd_result("DA", "PARSEC", seed=2),
        _make_wd_result("DB", "MIST",   seed=3),
        _make_wd_result("DB", "PARSEC", seed=4),
    ]
    fig = plot_wd_corner(results, bounds, gaia_id="wd_all")
    assert fig is not None
    plt.close(fig)


def test_plot_wd_corner_saves_file(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt
    res = _make_wd_result()
    bounds = _make_prior_bounds()
    out = tmp_path / "corner.pdf"
    plot_wd_corner([res], bounds, outpath=out)
    assert out.is_file()
    plt.close("all")


def test_plot_wd_corner_nan_mi() -> None:
    """NaN m_i_samples must not raise (IFMR extrapolation)."""
    import matplotlib.pyplot as plt
    res = _make_wd_result()
    res.m_i_samples[:5] = np.nan  # make some samples out-of-IFMR
    bounds = _make_prior_bounds()
    fig = plot_wd_corner([res], bounds)
    assert fig is not None
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_model_comparison
# ---------------------------------------------------------------------------

def test_plot_model_comparison_phoenix_only() -> None:
    """1-star + 2-star-only comparison chart."""
    import matplotlib.pyplot as plt
    import matplotlib.figure
    scores = {
        "1-star":     {"logz": -100.0, "bic": 25.0},
        "2-star sum": {"logz":  -90.0, "bic": 30.0},
    }
    fig = plot_model_comparison(scores, gaia_id="cmp_phx")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_model_comparison_with_wd() -> None:
    """Full set including WD-DA/DB MIST+PARSEC (bic=None for WD)."""
    import matplotlib.pyplot as plt
    scores = {
        "1-star":       {"logz": -120.0, "bic":  50.0},
        "2-star sum":   {"logz": -110.0, "bic":  60.0},
        "WD-DA-MIST":   {"logz":  -95.0, "bic": None},
        "WD-DA-PARSEC": {"logz":  -96.0, "bic": None},
        "WD-DB-MIST":   {"logz": -105.0, "bic": None},
        "WD-DB-PARSEC": {"logz": -106.0, "bic": None},
    }
    fig = plot_model_comparison(scores, gaia_id="cmp_all")
    assert fig is not None
    plt.close(fig)


def test_plot_model_comparison_saves_file(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt
    scores = {"1-star": {"logz": -100.0, "bic": 25.0}}
    out = tmp_path / "cmp.pdf"
    plot_model_comparison(scores, outpath=out)
    assert out.is_file()
    plt.close("all")


def test_plot_model_comparison_single_model() -> None:
    """A single model entry must not raise (ΔlnZ = 0)."""
    import matplotlib.pyplot as plt
    scores = {"WD-DA-MIST": {"logz": -88.0, "bic": None}}
    fig = plot_model_comparison(scores)
    assert fig is not None
    plt.close(fig)


# ---------------------------------------------------------------------------
# save_all_diagnostics
# ---------------------------------------------------------------------------

def test_save_all_diagnostics_phoenix(tmp_path: Path) -> None:
    """save_all_diagnostics must write sed + comparison files for PHOENIX models."""
    rows = _make_rows()
    preds = _make_model_preds(rows)
    scores = {"1-star": {"logz": -100.0, "bic": 25.0}}
    paths = save_all_diagnostics(
        rows=rows,
        model_preds=preds,
        model_scores=scores,
        gaia_id="test_target",
        out_dir=tmp_path,
    )
    assert "sed" in paths
    assert "comparison" in paths
    assert paths["sed"].is_file()
    assert paths["comparison"].is_file()
    assert "corner" not in paths


def test_save_all_diagnostics_wd(tmp_path: Path) -> None:
    """save_all_diagnostics must write corner file when wd_results is provided."""
    rows = _make_rows()
    preds: dict = {}
    scores: dict = {}
    results = [_make_wd_result("DA", "MIST")]
    bounds = _make_prior_bounds()
    paths = save_all_diagnostics(
        rows=rows,
        model_preds=preds,
        model_scores=scores,
        gaia_id="wd_target",
        wd_results=results,
        wd_prior_bounds=bounds,
        out_dir=tmp_path,
    )
    assert "corner" in paths
    assert paths["corner"].is_file()


# ---------------------------------------------------------------------------
# CLI --plot flag smoke test
# ---------------------------------------------------------------------------

def test_cli_plot_flag_present() -> None:
    """The --plot flag must be parseable without error."""
    from darkhunter_sed.phot_sed_cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args(["9999999", "--model", "1star", "--plot"])
    assert args.plot is True


def test_cli_no_plot_flag_default() -> None:
    """--plot should default to False."""
    from darkhunter_sed.phot_sed_cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args(["9999999"])
    assert args.plot is False


# ---------------------------------------------------------------------------
# WDFitResult t_cool_yr_samples field
# ---------------------------------------------------------------------------

def test_wd_fit_result_has_t_cool() -> None:
    """WDFitResult dataclass must have a t_cool_yr_samples field."""
    import dataclasses
    from darkhunter_sed.wd_model import WDFitResult
    field_names = {f.name for f in dataclasses.fields(WDFitResult)}
    assert "t_cool_yr_samples" in field_names
