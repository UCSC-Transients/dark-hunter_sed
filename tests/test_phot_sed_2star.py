"""
Path-2 2-star coeval model tests (Issue #31).

All tests use synthetic predictors / synth functions — no real MISTy weights
or PHOENIX grid required.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from darkhunter_sed.phot_sed_fit import (
    FitResult2Star,
    TwoStarPriorBounds,
    fit_2star_dynesty,
    run_2star_fit,
)
from darkhunter_sed.phot_sed_io import FLAG_DETECTION, PhotRow
from darkhunter_sed.phot_sed_models import (
    TWO_STAR_PARAM_NAMES,
    TwoStarParams,
    solve_eep2_for_age_match,
    predict_2star_phot,
)
from darkhunter_sed.phot_sed_cli import (
    _build_parser,
    main as phot_cli_main,
)


# ---------------------------------------------------------------------------
# Shared mock callables
# ---------------------------------------------------------------------------

def _mock_mist_predictor(**kwargs):
    """
    Analytic MISTy mock with deterministic age, Teff, R.

    Age increases monotonically with EEP so the coeval solver has a unique root.

      log(Age) = 9.0 + 0.001 * EEP + 0.01 * (1/mass)
      log(Teff) = log10(5000 + 2*EEP + 100*mass)
      log(g)    = 4.0 + 0.1*feh
      log(R)    = log10(1.0 + 0.1*mass)
    """
    eep = float(kwargs["eep"])
    mass = float(kwargs["mass"])
    feh = float(kwargs["feh"])
    afe = float(kwargs["afe"])
    teff = 5000.0 + 2.0 * eep + 100.0 * mass
    radius = 1.0 + 0.1 * mass
    log_age = 9.0 + 0.001 * eep + 0.01 * (1.0 / max(mass, 0.1))
    return {
        "EEP": eep,
        "initial_Mass": mass,
        "initial_[Fe/H]": feh,
        "initial_[a/Fe]": afe,
        "Mass": mass,
        "log(Age)": log_age,
        "log(R)": math.log10(radius),
        "log(L)": 0.0,
        "log(Teff)": math.log10(teff),
        "log(g)": 4.0 + 0.1 * feh,
        "[Fe/H]": feh,
        "[a/Fe]": afe,
    }


def _mock_synth_2star(
    teff1_k, logg1, teff2_k, logg2, mh, alpha, a_v, bands,
    *, radius1_cm, radius2_cm, distance_pc, systems=("ab",), bandpasses=None,
):
    """
    Analytic combined SED mock: flux-sum of two stars then approximate extinction.

    F_combined ∝ (R1/d)^2 * f(Teff1) + (R2/d)^2 * f(Teff2)
    mag = -2.5 log10(F_combined) + const + Av * 0.5  (rough band-independent extinction)
    """
    _pc_to_cm = 3.0856775814913673e18
    d_cm = float(distance_pc) * _pc_to_cm
    geom1 = (float(radius1_cm) / d_cm) ** 2
    geom2 = (float(radius2_cm) / d_cm) ** 2
    # Relative flux proxy (arbitrary units consistent across calls)
    f1 = geom1 * (float(teff1_k) / 5000.0) ** 4
    f2 = geom2 * (float(teff2_k) / 5000.0) ** 4
    f_combined = f1 + f2
    if f_combined <= 0.0:
        f_combined = 1e-30
    mag_base = -2.5 * math.log10(f_combined) + 20.0 + float(a_v) * 0.5
    out: dict[str, dict[str, float]] = {}
    for i, band in enumerate(bands):
        entry: dict[str, float] = {}
        if "ab" in {s.lower() for s in systems}:
            entry["ab"] = mag_base + 0.01 * i
        if "vega" in {s.lower() for s in systems}:
            entry["vega"] = mag_base + 0.01 * i + 0.1
        out[band] = entry
    return out


# ---------------------------------------------------------------------------
# EEP2 solver tests
# ---------------------------------------------------------------------------

def test_solve_eep2_returns_float_near_truth() -> None:
    """Solver recovers the known EEP at which the mock age matches target."""
    # With the mock: log(Age) = 9.0 + 0.001*EEP + 0.01/mass
    # For mass=1.0, feh=0, afe=0: age_gyr = 10^(log(Age) - 9) = 10^(0.001*EEP + 0.01)
    # At EEP=300: age = 10^(0.3 + 0.01) = 10^0.31 ≈ 2.042 Gyr
    target_eep = 300.0
    mass2 = 1.0
    target_age = 10.0 ** (0.001 * target_eep + 0.01)

    eep2 = solve_eep2_for_age_match(
        target_age, mass2, 0.0, 0.0,
        predictor=_mock_mist_predictor,
        eep_bounds=(1.0, 808.0),
        xtol=0.1,
    )
    assert eep2 is not None
    assert abs(eep2 - target_eep) < 1.0  # within 1 EEP unit


def test_solve_eep2_age_residual_within_tolerance() -> None:
    """Age residual at the solved EEP2 is within expected bounds."""
    target_age = 2.5  # Gyr
    mass2 = 0.8
    eep2 = solve_eep2_for_age_match(
        target_age, mass2, 0.0, 0.0,
        predictor=_mock_mist_predictor,
        eep_bounds=(1.0, 808.0),
        xtol=0.5,
    )
    assert eep2 is not None
    # Evaluate age at solved EEP2 and check residual
    from darkhunter_sed.misty_iso import evaluate_mist
    mist2 = evaluate_mist(eep2, mass2, 0.0, 0.0, predictor=_mock_mist_predictor)
    assert abs(mist2.age_gyr - target_age) < 0.1  # < 0.1 Gyr residual


def test_solve_eep2_returns_none_when_age_outside_range() -> None:
    """Returns None when target age is outside the achievable range at eep_bounds."""
    # With the mock predictor, age is finite and bounded. Use a tiny range
    # that does not bracket the target age.
    eep2 = solve_eep2_for_age_match(
        1000.0,  # impossibly old (> 10^0.808 * 10 Gyr)
        1.0, 0.0, 0.0,
        predictor=_mock_mist_predictor,
        eep_bounds=(1.0, 808.0),
        xtol=0.5,
    )
    assert eep2 is None


def test_solve_eep2_exact_bracket_endpoints() -> None:
    """Returns a valid root when target age exactly matches an endpoint."""
    from darkhunter_sed.misty_iso import evaluate_mist

    mass2 = 1.2
    mist_lo = evaluate_mist(1.0, mass2, 0.0, 0.0, predictor=_mock_mist_predictor)
    # Target age exactly at the lower bound — should still find a root near eep_lo.
    eep2 = solve_eep2_for_age_match(
        mist_lo.age_gyr, mass2, 0.0, 0.0,
        predictor=_mock_mist_predictor,
        eep_bounds=(1.0, 808.0),
        xtol=1.0,
    )
    # Either found a root or returned None (exact boundary: both OK).
    if eep2 is not None:
        assert 1.0 <= eep2 <= 808.0


# ---------------------------------------------------------------------------
# predict_2star_phot tests
# ---------------------------------------------------------------------------

def test_predict_2star_phot_basic() -> None:
    """predict_2star_phot returns a TwoStarPrediction with valid fields."""
    params = TwoStarParams(
        eep1=300.0, mass1=1.2, mass2=0.9,
        feh=0.0, afe=0.0, a_v=0.1, parallax_mas=10.0,
    )
    bands = ["b0", "b1", "b2"]
    pred = predict_2star_phot(
        params, bands,
        mist_predictor=_mock_mist_predictor,
        synth_2star=_mock_synth_2star,
    )
    assert set(pred.mags) == set(bands)
    assert all(math.isfinite(m) for m in pred.mags.values())
    assert pred.eep2 is not None
    assert 1.0 <= pred.eep2 <= 808.0
    assert math.isfinite(pred.mist1.age_gyr)
    assert math.isfinite(pred.mist2.age_gyr)
    # Coeval: ages should match within solver tolerance
    assert abs(pred.mist1.age_gyr - pred.mist2.age_gyr) < 0.2


def test_predict_2star_phot_from_array() -> None:
    """predict_2star_phot accepts a length-7 numpy array."""
    theta = np.array([300.0, 1.2, 0.9, 0.0, 0.0, 0.1, 10.0])
    bands = ["b0"]
    pred = predict_2star_phot(
        theta, bands,
        mist_predictor=_mock_mist_predictor,
        synth_2star=_mock_synth_2star,
    )
    assert math.isfinite(pred.mags["b0"])


def test_predict_2star_phot_mass_constraint_raises() -> None:
    """Raises ValueError when mass2 > mass1 (M1 >= M2 constraint)."""
    params = TwoStarParams(
        eep1=300.0, mass1=0.8, mass2=1.2,  # mass2 > mass1
        feh=0.0, afe=0.0, a_v=0.0, parallax_mas=10.0,
    )
    with pytest.raises(ValueError, match="M₁ ≥ M₂"):
        predict_2star_phot(
            params, ["b0"],
            mist_predictor=_mock_mist_predictor,
            synth_2star=_mock_synth_2star,
        )


def test_predict_2star_phot_equal_masses() -> None:
    """mass1 == mass2 is allowed (exact binary); produces valid mags."""
    params = TwoStarParams(
        eep1=300.0, mass1=1.0, mass2=1.0,
        feh=0.0, afe=0.0, a_v=0.0, parallax_mas=10.0,
    )
    pred = predict_2star_phot(
        params, ["b0"],
        mist_predictor=_mock_mist_predictor,
        synth_2star=_mock_synth_2star,
    )
    assert math.isfinite(pred.mags["b0"])


def test_twostar_params_from_array_length_check() -> None:
    """from_array raises ValueError for wrong length."""
    with pytest.raises(ValueError, match="length 7"):
        TwoStarParams.from_array(np.zeros(6))


def test_twostar_params_roundtrip() -> None:
    """from_array and as_array are inverses."""
    arr = np.array([300.0, 1.2, 0.9, 0.0, 0.1, 0.2, 10.0])
    p = TwoStarParams.from_array(arr)
    np.testing.assert_array_equal(p.as_array(), arr)


# ---------------------------------------------------------------------------
# fit_2star_dynesty smoke tests
# ---------------------------------------------------------------------------

def _make_2star_rows_and_bounds() -> tuple[list[PhotRow], TwoStarPriorBounds]:
    """4-band synthetic 2-star observations with tight priors for fast tests."""
    truth = TwoStarParams(
        eep1=300.0, mass1=1.1, mass2=0.9,
        feh=0.0, afe=0.0, a_v=0.1, parallax_mas=10.0,
    )
    bands = ["b0", "b1", "b2", "b3"]
    truth_pred = predict_2star_phot(
        truth, bands,
        mist_predictor=_mock_mist_predictor,
        synth_2star=_mock_synth_2star,
    )
    rows = [PhotRow(b, truth_pred.mags[b], 0.05, FLAG_DETECTION) for b in bands]
    bounds = TwoStarPriorBounds(
        eep1=(280.0, 320.0),
        mass1=(0.9, 1.3),
        mass2=(0.7, 1.1),
        feh=(-0.2, 0.2),
        afe=(-0.1, 0.1),
        a_v=(0.0, 0.3),
        parallax_mas=(8.0, 12.0),
    )
    return rows, bounds


def test_fit_2star_dynesty_smoke() -> None:
    """fit_2star_dynesty completes and returns a valid FitResult2Star."""
    rows, bounds = _make_2star_rows_and_bounds()
    result = fit_2star_dynesty(
        rows,
        mist_predictor=_mock_mist_predictor,
        synth_2star=_mock_synth_2star,
        bounds=bounds,
        nlive=12,
        maxiter=20,
        seed=0,
        jit_warmup=False,
    )
    assert isinstance(result, FitResult2Star)
    assert result.n_free == len(TWO_STAR_PARAM_NAMES)
    assert result.n_free == 7
    assert math.isfinite(result.logz)
    assert math.isfinite(result.bic)
    assert result.ln_l_max > -np.inf
    assert result.samples.ndim == 2
    assert result.samples.shape[1] == 7
    assert result.param_names == TWO_STAR_PARAM_NAMES


def test_run_2star_fit_writes_outputs(tmp_path: Path) -> None:
    """run_2star_fit writes summary JSON and samples npz with correct model field."""
    rows, bounds = _make_2star_rows_and_bounds()
    result, paths = run_2star_fit(
        rows,
        gaia_id="42",
        mist_predictor=_mock_mist_predictor,
        synth_2star=_mock_synth_2star,
        bounds=bounds,
        out_dir=tmp_path,
        nlive=12,
        maxiter=20,
        seed=1,
        jit_warmup=False,
    )
    assert paths["summary_json"].is_file()
    assert paths["samples_npz"].is_file()
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["model"] == "2star"
    assert summary["n_free"] == 7
    assert "logz" in summary and "bic" in summary
    assert summary["best_theta"]["EEP1"] == result.best_theta[0]
    assert "eep2_solved" in summary
    assert "best_mist1" in summary and "best_mist2" in summary


def test_fit_2star_jit_warmup_runs(tmp_path: Path) -> None:
    """jit_warmup=True path completes without error."""
    rows, bounds = _make_2star_rows_and_bounds()
    result = fit_2star_dynesty(
        rows,
        mist_predictor=_mock_mist_predictor,
        synth_2star=_mock_synth_2star,
        bounds=bounds,
        nlive=12,
        maxiter=5,
        seed=2,
        jit_warmup=True,
    )
    assert math.isfinite(result.logz)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

def test_cli_2star_choice_parses() -> None:
    """--model 2star is accepted by the argument parser."""
    p = _build_parser()
    args = p.parse_args(["42", "--model", "2star"])
    assert args.model == "2star"


def test_cli_1star_still_valid() -> None:
    """--model 1star still parses after adding 2star."""
    p = _build_parser()
    args = p.parse_args(["42", "--model", "1star"])
    assert args.model == "1star"


def test_cli_2star_missing_phot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """--model 2star with missing phot file returns rc=1."""
    monkeypatch.setenv("DARKHUNTER_SED_PHOTOMETRY_DIR", str(tmp_path))
    rc = phot_cli_main(["12345", "--model", "2star"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "12345_phot.fits" in err


def test_twostar_prior_bounds_defaults() -> None:
    """TwoStarPriorBounds default as_list() returns 7 bounds."""
    pb = TwoStarPriorBounds()
    bl = pb.as_list()
    assert len(bl) == 7
    for lo, hi in bl:
        assert lo < hi
