"""Per-order blaze normalization helpers."""

from __future__ import annotations

import numpy as np
import pytest

from darkhunter_rv.blaze import OrderBlazeModel, eval_blaze_sinc2


def test_blaze_scale_overlay() -> None:
    w = np.linspace(5180, 5280, 100)
    model = OrderBlazeModel(
        echelle_order=35,
        model="sinc2",
        center_angstrom=5230.0,
        width_angstrom=80.0,
        power=2.0,
        n_spectra_fit=10,
        wavelength_min=5180.0,
        wavelength_max=5280.0,
    )
    blaze = model.blaze_on_grid(w)
    flux = blaze * 1500.0 + 10.0
    good = np.isfinite(flux) & (blaze > 0)
    scale = float(np.nanmedian(flux[good]) / np.nanmedian(blaze[good]))
    assert scale == pytest.approx(1500.0, rel=0.01)


def test_eval_blaze_sinc2_peak() -> None:
    w = np.array([5200.0, 5200.0])
    b = eval_blaze_sinc2(w, center=5200.0, width=50.0, power=2.0, amplitude=1.0)
    assert b[0] == pytest.approx(1.0)
