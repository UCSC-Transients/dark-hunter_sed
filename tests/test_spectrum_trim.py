"""Order-end trimming of noisy blaze-normalized overlap wings."""

from __future__ import annotations

import numpy as np

from darkhunter_sed import spectrum


def _flat_order(n: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = np.linspace(5200.0, 5230.0, n)
    f = np.ones(n)
    e = np.full(n, 0.02)
    return w, f, e


def test_trim_removes_rising_wing_ends():
    w, f, e = _flat_order()
    f[:5] = 1.06  # rising blue wing
    f[-4:] = 1.07  # rising red wing
    tw, tf, _te = spectrum._trim_normalized_order_ends(w, f, e, dev_tol=0.04, min_pixels=18)
    assert tw.size == w.size - 9
    assert np.all(np.abs(tf - 1.0) <= 0.04)


def test_trim_keeps_min_pixels():
    w, f, e = _flat_order(n=40)
    f[:] = 1.2  # everything out of tolerance
    tw, _tf, _te = spectrum._trim_normalized_order_ends(w, f, e, dev_tol=0.04, min_pixels=18)
    # Cannot satisfy tolerance without dropping below min; keep full order.
    assert tw.size == w.size


def test_trim_noop_when_clean():
    w, f, e = _flat_order()
    tw, tf, te = spectrum._trim_normalized_order_ends(w, f, e, dev_tol=0.04, min_pixels=18)
    assert tw.size == w.size
    assert np.array_equal(tf, f)
