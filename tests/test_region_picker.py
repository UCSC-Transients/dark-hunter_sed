"""Region picker JSON helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from darkhunter_sed.region_picker import (
    FIT_MODEL_SINC2,
    FIT_MODEL_SINC_POLY,
    blaze_dict_from_model,
    build_manual_base_mask,
    default_order_view,
    delete_region_at_x,
    empty_order_regions,
    find_region_at_x,
    fit_model_from_regions,
    fit_order_from_regions,
    fit_shared_sinc_blaze,
    gaia_id_from_spec_path,
    init_orders_state,
    load_regions_json,
    make_document,
    merge_regions,
    normalize_fit_model,
    normalize_poly_order,
    normalize_region,
    order_blaze_model_from_regions,
    order_edge_exclusion_mask,
    poly_order_from_regions,
    regions_to_pixel_mask,
    regions_to_spans,
    resolve_order_regions,
    resolve_regions_json_for_star,
    save_regions_json,
    saved_view_usable,
    shade_mask_runs,
    subtract_spans,
)
from darkhunter_rv.blaze import OrderBlazeModel, eval_blaze_sinc2


def test_normalize_region_swaps() -> None:
    assert normalize_region(5200.0, 5180.0) == (5180.0, 5200.0)


def test_regions_to_spans_sorts() -> None:
    spans = regions_to_spans([[5250, 5260], [5180, 5190], [5200, 5210]])
    assert spans == [(5180.0, 5190.0), (5200.0, 5210.0), (5250.0, 5260.0)]


def test_find_region_at_x() -> None:
    regions = [[5180.0, 5190.0], [5250.0, 5260.0]]
    assert find_region_at_x(regions, 5185.0) == 0
    assert find_region_at_x(regions, 5255.0) == 1
    assert find_region_at_x(regions, 5220.0) is None


def test_delete_region_at_x() -> None:
    regions = [[5180.0, 5190.0], [5250.0, 5260.0]]
    new_list, removed = delete_region_at_x(regions, 5255.0)
    assert removed == [5250.0, 5260.0]
    assert new_list == [[5180.0, 5190.0]]


def test_init_orders_state_merges_existing() -> None:
    existing = {
        "35": {"continuum_regions": [[5180, 5190]], "line_regions": []},
    }
    state = init_orders_state([34, 35, 36], existing)
    assert state["35"]["continuum_regions"] == [[5180, 5190]]
    assert state["34"] == empty_order_regions()
    assert state["36"] == empty_order_regions()


def test_json_round_trip(tmp_path: Path) -> None:
    doc = make_document(
        "/data/Gaia_DR3_99_epoch_0.txt",
        "99",
        {
            "35": {
                "continuum_regions": [[5250, 5260], [5180, 5190]],
                "line_regions": [[5200, 5210]],
            }
        },
    )
    out = tmp_path / "regions.json"
    save_regions_json(out, doc)
    loaded = load_regions_json(out)
    assert loaded["gaia_id"] == "99"
    assert loaded["wave_system"] == "air"
    orders = loaded["orders"]
    assert orders["35"]["continuum_regions"] == [[5180.0, 5190.0], [5250.0, 5260.0]]
    assert orders["35"]["line_regions"] == [[5200.0, 5210.0]]


def test_load_regions_json_invalid_root(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        load_regions_json(path)


def test_gaia_id_from_spec_path() -> None:
    assert gaia_id_from_spec_path("Gaia_DR3_77413727493690112_epoch_0.txt") == "77413727493690112"
    assert gaia_id_from_spec_path("custom.txt") == "custom"


def test_default_order_view_no_zero_x() -> None:
    wave = np.linspace(5195.0, 5278.0, 200)
    flux = np.ones_like(wave) * 100.0
    (xlo, xhi), _ = default_order_view(wave, flux)
    assert xlo == pytest.approx(5195.0)
    assert xhi == pytest.approx(5278.0)
    assert xlo > 0.0


def test_saved_view_usable_rejects_empty_axes() -> None:
    wave = np.linspace(5195.0, 5278.0, 50)
    assert not saved_view_usable((0.0, 1.0), (0.0, 1.0), wave)
    assert saved_view_usable((5200.0, 5250.0), (90.0, 110.0), wave)


def test_saved_view_usable_rejects_other_order_xlim() -> None:
    order0 = np.linspace(3730.0, 3790.0, 50)
    order35_xlim = (5200.0, 5250.0)
    assert not saved_view_usable(order35_xlim, (0.0, 100.0), order0)


def test_merge_regions_abutting() -> None:
    merged = merge_regions([[5180.0, 5190.0], [5190.01, 5200.0]])
    assert merged == [(5180.0, 5200.0)]


def test_subtract_spans_line_from_continuum() -> None:
    cont = [(5180.0, 5220.0)]
    line = [(5195.0, 5205.0)]
    out = subtract_spans(cont, line)
    assert out == [(5180.0, 5195.0), (5205.0, 5220.0)]


def test_resolve_order_regions_line_wins() -> None:
    cont, line = resolve_order_regions(
        [[5180.0, 5220.0]],
        [[5195.0, 5205.0]],
    )
    assert line == [(5195.0, 5205.0)]
    assert cont == [(5180.0, 5195.0), (5205.0, 5220.0)]


def test_order_edge_exclusion_mask() -> None:
    edge = order_edge_exclusion_mask(20, edge_pixels=5)
    assert edge[5:15].all()
    assert not edge[:5].any()
    assert not edge[15:].any()


def test_build_manual_base_mask_fixed_cont_and_line() -> None:
    rng = np.random.default_rng(0)
    wave = np.linspace(5180.0, 5220.0, 41)
    flux = 100.0 + rng.normal(0, 0.5, size=wave.size)
    bundle = build_manual_base_mask(
        wave,
        flux,
        continuum_regions=[[5185.0, 5215.0]],
        line_regions=[[5198.0, 5202.0]],
        edge_pixels=5,
    )
    assert bundle.fixed_line_mask[20]
    assert bundle.fixed_cont_mask[10]
    assert not bundle.fixed_cont_mask[0]
    assert not bundle.fixed_cont_mask[20]

    flux_cr = flux.copy()
    flux_cr[20] = 1e6
    bundle_cr = build_manual_base_mask(wave, flux_cr, continuum_regions=[[5185.0, 5215.0]])
    assert not bundle_cr.cr_mask[20]


def test_manual_continuum_supplements_auto_base() -> None:
    rng = np.random.default_rng(9)
    wave = np.linspace(5180.0, 5220.0, 41)
    flux = 100.0 + rng.normal(0, 0.5, size=wave.size)
    bundle = build_manual_base_mask(
        wave,
        flux,
        continuum_regions=[[5190.0, 5195.0]],
        rest_lines=[],
        half_width_angstrom=22.0,
        edge_pixels=0,
    )
    manual_mask = regions_to_pixel_mask(wave, [[5190.0, 5195.0]])
    outside_manual = ~manual_mask
    assert bundle.base_mask[manual_mask].any()
    assert bundle.base_mask[outside_manual].any()
    assert bundle.fixed_cont_mask[manual_mask].any()


def test_manual_line_supplements_auto_exclusion() -> None:
    rng = np.random.default_rng(11)
    wave = np.linspace(5180.0, 5220.0, 81)
    center = 5200.0
    shape = eval_blaze_sinc2(wave, center, 80.0, power=2.0, amplitude=100.0)
    flux = shape + rng.normal(0, 0.3, size=wave.size)
    fallback = OrderBlazeModel(
        echelle_order=35,
        model="sinc2",
        center_angstrom=center,
        width_angstrom=80.0,
        power=2.0,
        n_spectra_fit=1,
        wavelength_min=5180.0,
        wavelength_max=5220.0,
    )
    regions = {
        "continuum_regions": [],
        "line_regions": [[5198.0, 5202.0]],
        "fit_model": FIT_MODEL_SINC2,
    }
    preview = fit_order_from_regions(
        wave,
        flux,
        regions,
        fallback,
        echelle_order=35,
    )
    assert preview is not None
    line_mask = regions_to_pixel_mask(wave, [[5198.0, 5202.0]])
    assert not preview.continuum_mask[line_mask].any()
    assert int(np.sum(preview.continuum_mask)) >= 18


def test_blaze_json_round_trip(tmp_path: Path) -> None:
    doc = make_document(
        "/data/spec.txt",
        "42",
        {
            "35": {
                "continuum_regions": [[5180, 5220]],
                "line_regions": [],
                "blaze": {
                    "center_angstrom": 5200.5,
                    "width_angstrom": 42.0,
                    "power": 2.1,
                },
            }
        },
    )
    out = tmp_path / "regions_blaze.json"
    save_regions_json(out, doc)
    loaded = load_regions_json(out)
    blaze = loaded["orders"]["35"]["blaze"]
    assert blaze["center_angstrom"] == pytest.approx(5200.5)
    assert blaze["width_angstrom"] == pytest.approx(42.0)
    assert blaze["power"] == pytest.approx(2.1)


def test_fit_shared_sinc_blaze_stores_params() -> None:
    rng = np.random.default_rng(12)
    wave = np.linspace(5180.0, 5220.0, 81)
    true_center = 5205.0
    shape = eval_blaze_sinc2(wave, true_center, 70.0, power=2.0, amplitude=130.0)
    flux = shape + rng.normal(0, 0.4, size=wave.size)
    regions = {
        "continuum_regions": [[5185.0, 5215.0]],
        "line_regions": [],
    }
    fitted = fit_shared_sinc_blaze(wave, flux, regions, 35, half_width_angstrom=22.0)
    assert fitted is not None
    blaze_d = blaze_dict_from_model(fitted)
    assert abs(blaze_d["center_angstrom"] - true_center) < 8.0
    assert blaze_d["width_angstrom"] > 0
    assert blaze_d["power"] >= 2.0


def test_order_blaze_model_prefers_stored_over_fallback() -> None:
    wave = np.linspace(5180.0, 5220.0, 41)
    fallback = OrderBlazeModel(
        echelle_order=35,
        model="sinc2",
        center_angstrom=5000.0,
        width_angstrom=10.0,
        power=4.0,
        n_spectra_fit=77,
        wavelength_min=5180.0,
        wavelength_max=5220.0,
    )
    regions = {
        "blaze": {
            "center_angstrom": 5200.0,
            "width_angstrom": 55.0,
            "power": 2.0,
        }
    }
    resolved = order_blaze_model_from_regions(regions, 35, wave, fallback)
    assert resolved is not None
    assert resolved.center_angstrom == pytest.approx(5200.0)
    assert resolved.width_angstrom == pytest.approx(55.0)
    assert resolved.power == pytest.approx(2.0)


def test_fit_model_defaults() -> None:
    assert normalize_fit_model(None) == FIT_MODEL_SINC2
    assert normalize_fit_model("sinc_poly") == FIT_MODEL_SINC_POLY
    assert normalize_poly_order(None) == 2
    assert normalize_poly_order(5) == 4
    assert fit_model_from_regions(None) == FIT_MODEL_SINC2
    regions = {"continuum_regions": [[5180, 5190]], "line_regions": [], "fit_model": "sinc_poly"}
    assert fit_model_from_regions(regions) == FIT_MODEL_SINC_POLY
    assert poly_order_from_regions(regions) == 2


def test_json_round_trip_fit_model(tmp_path: Path) -> None:
    doc = make_document(
        "/data/spec.txt",
        "42",
        {
            "35": {
                "continuum_regions": [[5180, 5220]],
                "line_regions": [],
                "fit_model": "sinc_poly",
                "poly_order": 3,
            }
        },
    )
    out = tmp_path / "regions_fit.json"
    save_regions_json(out, doc)
    loaded = load_regions_json(out)
    order = loaded["orders"]["35"]
    assert order["fit_model"] == FIT_MODEL_SINC_POLY
    assert order["poly_order"] == 3


def test_fit_order_from_regions_flat_spectrum() -> None:
    from darkhunter_rv.blaze import OrderBlazeModel

    rng = np.random.default_rng(7)
    wave = np.linspace(5180.0, 5220.0, 81)
    blaze_model = OrderBlazeModel(
        echelle_order=35,
        model="sinc2",
        center_angstrom=5200.0,
        width_angstrom=120.0,
        power=2.0,
        n_spectra_fit=1,
        wavelength_min=5180.0,
        wavelength_max=5220.0,
    )
    shape = blaze_model.blaze_on_grid(wave)
    flux = 120.0 * shape + rng.normal(0, 0.4, size=wave.size)
    regions = {
        "continuum_regions": [[5185.0, 5215.0]],
        "line_regions": [],
        "fit_model": FIT_MODEL_SINC2,
    }
    preview = fit_order_from_regions(wave, flux, regions, blaze_model)
    assert preview is not None
    assert preview.envelope.shape == wave.shape
    assert int(np.sum(preview.continuum_mask)) >= 18
    ok = preview.continuum_mask & np.isfinite(flux) & (preview.envelope > 0)
    norm = flux[ok] / preview.envelope[ok]
    assert abs(float(np.nanmedian(norm)) - 1.0) < 0.08


def test_resolve_regions_json_for_star_no_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("darkhunter_sed.region_picker.masks_dir", lambda: tmp_path)
    assert resolve_regions_json_for_star("missing_id") is None
    assert resolve_regions_json_for_star("missing_id", auto_resolve=False) is None


def test_shade_mask_runs_merges_contiguous() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = np.array([True, True, False, True, True])
    artists = shade_mask_runs(ax, w, mask, color="green", alpha=0.1)
    assert len(artists) == 2
    plt.close(fig)
