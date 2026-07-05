"""Inverse-variance coalescing of overlapping order segments."""

from __future__ import annotations

import numpy as np
import pytest

from darkhunter_sed import spectrum


def test_coalesce_weights_low_error_pixel_more():
    # Two segments overlap at one wavelength; low-error pixel should dominate.
    lam = 5200.0
    seg_a = (np.array([lam]), np.array([1.0]), np.array([0.01]))
    seg_b = (np.array([lam]), np.array([2.0]), np.array([1.0]))
    w, f, e = spectrum._coalesce_segments([seg_a, seg_b], dedup_tol=1e-2)
    assert w.size == 1
    # Weighted mean pulled toward the 1.0 (tiny error) value, not plain 1.5.
    assert f[0] == pytest.approx(1.0, abs=0.01)
    # Combined error smaller than either input? At least <= smallest input.
    assert e[0] <= 0.01 + 1e-9


def test_coalesce_distinct_wavelengths_preserved():
    seg = (np.array([5200.0, 5201.0]), np.array([1.0, 1.1]), np.array([0.1, 0.1]))
    w, _f, _e = spectrum._coalesce_segments([seg], dedup_tol=1e-3)
    assert w.size == 2
