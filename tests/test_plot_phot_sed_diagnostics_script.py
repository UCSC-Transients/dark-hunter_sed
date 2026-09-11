"""
Regression tests for scripts/plot_phot_sed_diagnostics.py (Issue #56).

``_plot_sed_panels`` / ``_plot_sed_2star`` must not draw a residual/"pull" for
a detection the fit likelihood skipped as a sentinel-model band (predicted
magnitude >= SENTINEL_MAG_THRESHOLD, see photometry_loglike). Doing so makes a
point that had zero influence on the posterior look like a catastrophic
outlier.

All tests use ``matplotlib.use("Agg")`` for headless operation in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede any other matplotlib import

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from darkhunter_sed.phot_sed_fit import SENTINEL_MAG_THRESHOLD
from darkhunter_sed.phot_sed_io import FLAG_DETECTION, PhotRow
from scripts.plot_phot_sed_diagnostics import _plot_sed_panels


def test_plot_sed_panels_skips_pull_for_sentinel_band() -> None:
    """A detection whose best-fit model magnitude is a sentinel is excluded
    from the residual panel, even though it is plotted in the SED panel."""
    best_mags = {"GaiaDR3_G": 12.0, "GALEX_NUV": SENTINEL_MAG_THRESHOLD + 5.0}
    dets = [
        PhotRow("GaiaDR3_G", 12.05, 0.02, FLAG_DETECTION),
        # Fit never used this detection (model predicts ~zero UV flux); a
        # naive residual would be a huge, meaningless outlier.
        PhotRow("GALEX_NUV", 16.0, 0.05, FLAG_DETECTION),
    ]
    fig, (ax_sed, ax_res) = plt.subplots(2, 1)
    _plot_sed_panels(ax_sed, ax_res, best_mags, dets, uls=[], title="test")

    scatter_children = [c for c in ax_res.collections]
    assert len(scatter_children) == 1
    offsets = scatter_children[0].get_offsets()
    assert offsets.shape[0] == 1, "only the non-sentinel band should get a pull"
    plt.close(fig)


def test_plot_sed_panels_no_detections_survive_sentinel_filter() -> None:
    """When every detection is sentinel-skipped, the panel reports no
    detections rather than plotting spurious pulls for all of them."""
    best_mags = {"GALEX_NUV": SENTINEL_MAG_THRESHOLD + 1.0}
    dets = [PhotRow("GALEX_NUV", 16.0, 0.05, FLAG_DETECTION)]
    fig, (ax_sed, ax_res) = plt.subplots(2, 1)
    _plot_sed_panels(ax_sed, ax_res, best_mags, dets, uls=[], title="test")

    assert len(ax_res.collections) == 0
    plt.close(fig)
