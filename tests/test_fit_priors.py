"""UMS per-epoch spectral nuisance priors."""

from __future__ import annotations

from darkhunter_sed import fit


def test_ums_default_rigid_continuum_priors() -> None:
    priors_dict: dict = {}
    fit._apply_per_epoch_spec_priors(priors_dict, 2, fix_specjitter=False, rigid_continuum=True)
    assert priors_dict["pc0_0"] == ["fixed", 1.0]
    assert priors_dict["pc1_1"] == ["fixed", 0.0]
    assert priors_dict["lsf_0"] == ["fixed", 60000.0]
    assert priors_dict["specjitter_0"] == ["uniform", [1e-6, 1e-2]]


def test_ums_flexible_continuum_priors_opt_in() -> None:
    priors_dict: dict = {}
    fit._apply_per_epoch_spec_priors(priors_dict, 2, fix_specjitter=False, rigid_continuum=False)
    assert priors_dict["pc0_0"] == ["uniform", [0.95, 1.05]]
    assert priors_dict["pc1_1"] == ["uniform", [-0.1, 0.1]]
    assert priors_dict["lsf_0"][0] == "tnormal"
    assert priors_dict["specjitter_0"] == ["uniform", [1e-6, 1e-2]]


def test_ums_rigid_continuum_with_fixed_specjitter() -> None:
    priors_dict: dict = {}
    fit._apply_per_epoch_spec_priors(priors_dict, 1, fix_specjitter=True, rigid_continuum=True)
    assert priors_dict["pc0_0"] == ["fixed", 1.0]
    assert priors_dict["pc2_0"] == ["fixed", 0.0]
    assert priors_dict["lsf_0"] == ["fixed", 60000.0]
    assert priors_dict["specjitter_0"] == ["fixed", 0.0]
