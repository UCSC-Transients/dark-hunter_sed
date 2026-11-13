"""WD+star prior-transform tests (synthetic Bergeron grid; no real tables)."""

from __future__ import annotations

import numpy as np
import pytest

from darkhunter_sed.bergeron_wd import _LOGG_GRID, BergeronGrid
from darkhunter_sed.cummings_ifmr import CummingsIFMR
from darkhunter_sed.phot_sed_io import FLAG_DETECTION, PhotRow
from darkhunter_sed.wd_model import (
    _WDS_TEFF_IDX,
    WDStarPriorBounds,
    _make_wdstar_loglike,
)

_N_COL = 42  # Bergeron global table columns minus Teff and log g


def _synthetic_grid(teff_min: float, teff_max: float) -> BergeronGrid:
    teffs = np.linspace(teff_min, teff_max, 6)
    data = np.full((len(_LOGG_GRID), teffs.size, _N_COL), 12.0)
    data[:, :, 0] = np.linspace(0.4, 1.2, len(_LOGG_GRID))[:, None]
    data[:, :, -1] = 1.0e9
    return BergeronGrid("DB", teffs, data)


def test_wdstar_teff_prior_clamped_to_grid() -> None:
    grid = _synthetic_grid(3250.0, 150_000.0)
    rows = [PhotRow("GaiaDR3_G", 15.0, 0.01, FLAG_DETECTION)]
    _, ptform = _make_wdstar_loglike(
        grid,
        CummingsIFMR("MIST"),
        rows,
        WDStarPriorBounds(teff_wd=(1500.0, 200_000.0)),
        mist_predictor=None,
        phoenix_grid=None,
        bandpasses=None,
    )
    ndim = len(WDStarPriorBounds().as_list())
    assert ptform(np.zeros(ndim))[_WDS_TEFF_IDX] == pytest.approx(3250.0)
    assert ptform(np.ones(ndim))[_WDS_TEFF_IDX] == pytest.approx(150_000.0)


def test_wdstar_teff_prior_disjoint_from_grid_raises() -> None:
    grid = _synthetic_grid(3250.0, 150_000.0)
    rows = [PhotRow("GaiaDR3_G", 15.0, 0.01, FLAG_DETECTION)]
    with pytest.raises(ValueError, match="does not overlap"):
        _make_wdstar_loglike(
            grid,
            CummingsIFMR("MIST"),
            rows,
            WDStarPriorBounds(teff_wd=(1500.0, 3000.0)),
            mist_predictor=None,
            phoenix_grid=None,
            bandpasses=None,
        )
