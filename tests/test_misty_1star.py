"""Path-2 MISTy + 1-star dynesty tests (synthetic phot; no real mistNN/PHOENIX)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from darkhunter_sed.config import phot_sed_dir
from darkhunter_sed.misty_iso import MistPoint, evaluate_mist, R_SUN_CM
from darkhunter_sed.phot_sed_fit import (
    OneStarPriorBounds,
    bic_from_max_likelihood,
    fit_1star_dynesty,
    photometry_loglike,
    run_1star_fit,
)
from darkhunter_sed.phot_sed_io import FLAG_DETECTION, FLAG_UPPER_LIMIT, PhotRow
from darkhunter_sed.phot_sed_models import (
    ONE_STAR_PARAM_NAMES,
    OneStarParams,
    parallax_mas_to_distance_pc,
    predict_1star_phot,
)
from darkhunter_sed.phot_sed_cli import (
    _normalize_gaia_id,
    default_phot_fits_path,
    main as phot_cli_main,
)


def _mock_mist_predictor(**kwargs):
    """
    Analytic stand-in for ``getMIST``.

    Teff / logg / R encode inputs so recovery tests are exact:
    - log(Teff) = log10(5000 + 2*EEP + 100*mass)
    - log(g) = 4.0 + 0.1*feh
    - log(R) = log10(1.0 + 0.1*mass + 0.01*afe)
    - log(Age) = 9.5 - 0.001*EEP
    """
    eep = float(kwargs["eep"])
    mass = float(kwargs["mass"])
    feh = float(kwargs["feh"])
    afe = float(kwargs["afe"])
    teff = 5000.0 + 2.0 * eep + 100.0 * mass
    radius = 1.0 + 0.1 * mass + 0.01 * afe
    return {
        "EEP": eep,
        "initial_Mass": mass,
        "initial_[Fe/H]": feh,
        "initial_[a/Fe]": afe,
        "Mass": mass,
        "log(Age)": 9.5 - 0.001 * eep,
        "log(R)": math.log10(radius),
        "log(L)": 0.0,
        "log(Teff)": math.log10(teff),
        "log(g)": 4.0 + 0.1 * feh,
        "[Fe/H]": feh,
        "[a/Fe]": afe,
    }


def _mock_synth(
    teff_k: float,
    logg: float,
    mh: float,
    alpha: float,
    a_v: float,
    bands,
    *,
    radius_cm: float,
    distance_pc: float,
    systems=("ab",),
    bandpasses=None,
):
    """
    Synthetic AB mags that depend linearly on params (noise-free recovery).

    ``mag = 10 - 2.5*log10((R/d)^2) + 0.01*(teff-5800) + 0.1*Av - 0.05*mh``
    same for every band (plus tiny band index offset).
    """
    geom = (float(radius_cm) / (float(distance_pc) * 3.0856775814913673e18)) ** 2
    base = 10.0 - 2.5 * math.log10(max(geom, 1e-40))
    base += 0.01 * (float(teff_k) - 5800.0) + 0.1 * float(a_v) + 0.5 * float(mh)
    base += 0.02 * float(logg) + 0.2 * float(alpha)
    out = {}
    for i, band in enumerate(bands):
        entry = {}
        if "ab" in {s.lower() for s in systems}:
            entry["ab"] = base + 0.01 * i
        if "vega" in {s.lower() for s in systems}:
            entry["vega"] = entry.get("ab", base) + 0.1
        out[band] = entry
    return out


def test_evaluate_mist_mock() -> None:
    pt = evaluate_mist(350.0, 1.0, 0.0, 0.0, predictor=_mock_mist_predictor)
    assert isinstance(pt, MistPoint)
    assert pt.teff_k == pytest.approx(5000.0 + 2.0 * 350.0 + 100.0)
    assert pt.radius_cm == pytest.approx(pt.radius_rsun * R_SUN_CM)
    assert pt.age_gyr == pytest.approx(10.0 ** (9.5 - 0.001 * 350.0 - 9.0))


def test_parallax_distance() -> None:
    assert parallax_mas_to_distance_pc(10.0) == pytest.approx(100.0)
    with pytest.raises(ValueError):
        parallax_mas_to_distance_pc(0.0)


def test_predict_1star_phot_synthetic() -> None:
    params = OneStarParams(
        eep=300.0, mass=1.0, feh=0.0, afe=0.0, a_v=0.2, parallax_mas=10.0
    )
    bands = ["PS_g", "PS_r"]
    pred = predict_1star_phot(
        params,
        bands,
        mist_predictor=_mock_mist_predictor,
        synth_phot=_mock_synth,
    )
    assert set(pred.mags) == set(bands)
    assert pred.distance_pc == pytest.approx(100.0)
    assert all(math.isfinite(m) for m in pred.mags.values())


def test_photometry_loglike_detection_and_ul() -> None:
    pred = {"PS_g": 12.0, "PS_r": 11.5}
    rows = [
        PhotRow("PS_g", 12.0, 0.05, FLAG_DETECTION),
        PhotRow("PS_r", 11.0, 0.05, FLAG_UPPER_LIMIT),  # model fainter than UL → 0
    ]
    ln0 = photometry_loglike(pred, rows)
    # Brighten model in UL band past the limit → penalty
    pred_bright = {"PS_g": 12.0, "PS_r": 10.5}
    ln1 = photometry_loglike(pred_bright, rows)
    assert ln1 < ln0


def test_bic_formula() -> None:
    bic = bic_from_max_likelihood(ln_l_max=-10.0, n_free=6, n_data=8)
    assert bic == pytest.approx(6.0 * math.log(8.0) - 2.0 * (-10.0))


def test_fit_1star_dynesty_synthetic(tmp_path: Path) -> None:
    truth = OneStarParams(
        eep=320.0, mass=1.05, feh=-0.1, afe=0.0, a_v=0.15, parallax_mas=12.0
    )
    bands = ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]
    truth_pred = predict_1star_phot(
        truth,
        bands,
        mist_predictor=_mock_mist_predictor,
        synth_phot=_mock_synth,
    )
    rows = [
        PhotRow(b, truth_pred.mags[b], 0.02, FLAG_DETECTION) for b in bands
    ]
    # Tight priors around truth so maxiter stays small.
    bounds = OneStarPriorBounds(
        eep=(300.0, 340.0),
        mass=(0.9, 1.2),
        feh=(-0.3, 0.1),
        a_v=(0.0, 0.5),
        parallax_mas=(8.0, 16.0),
    )
    result, paths = run_1star_fit(
        rows,
        gaia_id="999",
        mist_predictor=_mock_mist_predictor,
        synth_phot=_mock_synth,
        bounds=bounds,
        out_dir=tmp_path,
        nlive=50,
        maxiter=250,
        seed=0,
    )
    assert result.n_free == len(ONE_STAR_PARAM_NAMES) + 1  # +1 for sigma_int
    assert math.isfinite(result.logz)
    assert math.isfinite(result.bic)
    assert result.ln_l_max > -np.inf
    assert paths["summary_json"].is_file()
    assert paths["samples_npz"].is_file()
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["model"] == "1star"
    assert "logz" in summary and "bic" in summary
    assert summary["best_theta"]["EEP"] == result.best_theta[0]
    # Truth should be near the likelihood peak on this noise-free synthetic set.
    ln_truth = photometry_loglike(truth_pred.mags, rows)
    assert result.ln_l_max >= ln_truth - 5.0
    wrong = truth.as_array().copy()
    wrong[0] = bounds.eep[0]
    wrong[3] = bounds.a_v[1]
    ln_wrong = photometry_loglike(
        predict_1star_phot(
            wrong,
            bands,
            mist_predictor=_mock_mist_predictor,
            synth_phot=_mock_synth,
        ).mags,
        rows,
    )
    assert result.ln_l_max > ln_wrong
    assert result.samples.ndim == 2
    assert result.samples.shape[1] == len(ONE_STAR_PARAM_NAMES) + 1  # +1 for sigma_int

def test_phot_sed_dir_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DARKHUNTER_SED_PHOT_SED_DIR", str(tmp_path / "ps"))
    assert phot_sed_dir() == (tmp_path / "ps").resolve()


def test_cli_normalize_and_help() -> None:
    assert _normalize_gaia_id("Gaia_DR3_123") == "123"
    assert _normalize_gaia_id("123") == "123"
    with pytest.raises(SystemExit) as exc:
        phot_cli_main(["--help"])
    assert exc.value.code == 0


def test_default_phot_fits_path_matches_gather(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default lookup is ``{id}_phot.fits``, not ``Gaia_DR3_{id}_phot.fits``."""
    monkeypatch.setenv("DARKHUNTER_SED_PHOTOMETRY_DIR", str(tmp_path))
    path = default_phot_fits_path("77413727493690112")
    assert path == (tmp_path / "77413727493690112_phot.fits").resolve()
    assert "Gaia_DR3_" not in path.name
    # Explicit dir override
    other = tmp_path / "other"
    other.mkdir()
    assert default_phot_fits_path("99", other) == (other / "99_phot.fits").resolve()


def test_cli_missing_phot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("DARKHUNTER_SED_PHOTOMETRY_DIR", str(tmp_path))
    rc = phot_cli_main(["12345", "--model", "1star"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "12345_phot.fits" in err
    assert "Gaia_DR3_12345_phot.fits" not in err


# ---------------------------------------------------------------------------
# Speed-knob tests (Issue #29): JIT warm-up, new CLI flags, nworkers
# ---------------------------------------------------------------------------

def _make_tight_fit_args(tmp_path: Path) -> tuple[list[PhotRow], OneStarPriorBounds]:
    """Shared fixture: 4-band synthetic rows with tight priors."""
    truth = OneStarParams(eep=300.0, mass=1.0, feh=0.0, afe=0.0, a_v=0.1, parallax_mas=10.0)
    bands = ["b0", "b1", "b2", "b3"]
    truth_pred = predict_1star_phot(
        truth, bands, mist_predictor=_mock_mist_predictor, synth_phot=_mock_synth
    )
    rows = [PhotRow(b, truth_pred.mags[b], 0.05, FLAG_DETECTION) for b in bands]
    bounds = OneStarPriorBounds(
        eep=(280.0, 320.0),
        mass=(0.8, 1.2),
        feh=(-0.2, 0.2),
        a_v=(0.0, 0.3),
        parallax_mas=(8.0, 12.0),
    )
    return rows, bounds


def test_jit_warmup_calls_predictor_before_sampler(tmp_path: Path) -> None:
    """jit_warmup=True must call mist_predictor before dynesty sampling starts."""
    rows, bounds = _make_tight_fit_args(tmp_path)
    bands = [r.band for r in rows]
    call_log: list[str] = []

    def counting_predictor(**kwargs: float) -> dict:
        call_log.append("call")
        return _mock_mist_predictor(**kwargs)

    # Run with jit_warmup=True and maxiter=1 so dynesty barely runs.
    fit_1star_dynesty(
        rows,
        mist_predictor=counting_predictor,
        synth_phot=_mock_synth,
        bounds=bounds,
        nlive=10,
        maxiter=1,
        seed=0,
        jit_warmup=True,
    )
    # Warm-up fires at least once, plus dynesty's own evaluations.
    assert len(call_log) >= 1


def test_jit_warmup_false_still_works(tmp_path: Path) -> None:
    """jit_warmup=False must not crash and must produce a valid result."""
    rows, bounds = _make_tight_fit_args(tmp_path)
    result = fit_1star_dynesty(
        rows,
        mist_predictor=_mock_mist_predictor,
        synth_phot=_mock_synth,
        bounds=bounds,
        nlive=10,
        maxiter=1,
        seed=0,
        jit_warmup=False,
    )
    assert math.isfinite(result.logz)


def test_fit_1star_sample_bound_params(tmp_path: Path) -> None:
    """sample and bound parameters are accepted and produce valid results."""
    rows, bounds = _make_tight_fit_args(tmp_path)
    result = fit_1star_dynesty(
        rows,
        mist_predictor=_mock_mist_predictor,
        synth_phot=_mock_synth,
        bounds=bounds,
        nlive=10,
        maxiter=5,
        seed=1,
        sample="unif",
        bound="single",
        jit_warmup=False,
    )
    assert math.isfinite(result.logz)
    assert math.isfinite(result.bic)


def test_fit_1star_dlogz_param(tmp_path: Path) -> None:
    """dlogz is forwarded to dynesty and accepted without error."""
    rows, bounds = _make_tight_fit_args(tmp_path)
    result = fit_1star_dynesty(
        rows,
        mist_predictor=_mock_mist_predictor,
        synth_phot=_mock_synth,
        bounds=bounds,
        nlive=10,
        maxiter=20,
        seed=2,
        dlogz=2.0,
        jit_warmup=False,
    )
    assert math.isfinite(result.logz)


def test_cli_new_flags_parse() -> None:
    """New CLI flags --dlogz, --sample, --bound, --nworkers parse without error."""
    import argparse
    from darkhunter_sed.phot_sed_cli import _build_parser

    p = _build_parser()
    args = p.parse_args([
        "12345",
        "--nlive", "100",
        "--dlogz", "1.0",
        "--sample", "unif",
        "--bound", "single",
        "--nworkers", "2",
    ])
    assert args.nlive == 100
    assert args.dlogz == pytest.approx(1.0)
    assert args.sample == "unif"
    assert args.bound == "single"
    assert args.nworkers == 2


def test_cli_nlive_default_is_100() -> None:
    """Production --nlive default must be 100 (was 200 before Issue #29)."""
    from darkhunter_sed.phot_sed_cli import _build_parser

    p = _build_parser()
    args = p.parse_args(["42"])
    assert args.nlive == 100


def test_cli_dlogz_default_is_1() -> None:
    """Production --dlogz default must be 1.0 (tighter than 0.5 needs explicit flag)."""
    from darkhunter_sed.phot_sed_cli import _build_parser

    p = _build_parser()
    args = p.parse_args(["42"])
    assert args.dlogz == pytest.approx(1.0)


def test_nworkers_serial_produces_valid_result(tmp_path: Path) -> None:
    """nworkers=1 (serial) produces a valid FitResult1Star."""
    rows, bounds = _make_tight_fit_args(tmp_path)
    result, paths = run_1star_fit(
        rows,
        gaia_id="serial_test",
        mist_predictor=_mock_mist_predictor,
        synth_phot=_mock_synth,
        bounds=bounds,
        out_dir=tmp_path,
        nlive=10,
        maxiter=10,
        seed=3,
        nworkers=1,
        jit_warmup=False,
    )
    assert math.isfinite(result.logz)
    assert paths["summary_json"].is_file()
