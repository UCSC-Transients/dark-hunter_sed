"""Continuum fractional residual metrics."""

from __future__ import annotations

import numpy as np
import pytest

from darkhunter_sed.continuum_diagnostics import continuum_fractional_residual


def test_continuum_fractional_residual_flat_model() -> None:
    wave = np.linspace(5150, 5300, 200)
    data = np.ones_like(wave)
    model = np.ones_like(wave) * 1.02
    m = continuum_fractional_residual(wave, data, model)
    assert m["median_ratio"] == pytest.approx(1.02, rel=1e-6)
    assert m["continuum_rms"] == pytest.approx(0.02, abs=0.01)
