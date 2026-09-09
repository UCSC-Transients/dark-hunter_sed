"""
Tests for darkhunter_sed.bergeron_wd, darkhunter_sed.cummings_ifmr,
and darkhunter_sed.wd_model (smoke test with real Bergeron tables).

Fixture strategy
----------------
``_MINI_TABLE_DA`` / ``_MINI_TABLE_DB`` are trimmed excerpts of the actual
Bergeron tables (3 log g values × 5 Teff points each).  They are written to
tmp directories so the parser and interpolator can be exercised without the
full ~300-row file.  The last section tests a handful of real system photometry
examples using the actual WD table directory (skipped if not present).
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np
import pytest

from darkhunter_sed.bergeron_wd import BergeronGrid, _parse_global_table, _LOGG_GRID
from darkhunter_sed.cummings_ifmr import CummingsIFMR, ms_lifetime_yr

# ---------------------------------------------------------------------------
# Fixture: minimal Bergeron Table_DA subset
# ---------------------------------------------------------------------------
# Real header line 1 pattern; columns after M/Mo: Mbol BC U B V R I J H Ks Y
# J H K W1 W2 W3 W4 S3.6 S4.5 S5.8 S8.0 u g r i z g r i z y
# G2 G2_BP G2_RP G3 G3_BP G3_RP FUV NUV Age
# We include only 5 log g values (required) and 5 Teff values per log g.
# Magnitudes are synthetic values; only structure matters for parser tests.

# header line 1: "n_logg  n_teff  label"
# header line 2: column names
# data: space-delimited, one row per (Teff, log g) pair.
# We use 5 log g values to match _LOGG_GRID = [7.0, 7.5, 8.0, 8.5, 9.0].

_DA_HEADER = """\
    5    5  Pure-hydrogen grid (test fixture)              Test
 Teff  log g  M/Mo  Mbol     BC    U      B      V      R      I      J      H      Ks     Y      J      H      K      W1     W2     W3     W4    S3.6   S4.5   S5.8   S8.0    u      g      r      i      z      g      r      i      z      y      G2    G2_BP  G2_RP   G3    G3_BP  G3_RP  FUV    NUV      Age
"""

# 5 log g × 5 Teff = 25 rows.  Values are rough but self-consistent.
# Format: Teff logg M/Mo Mbol BC U B V R I J H Ks Y J H K W1 W2 W3 W4 S3.6 S4.5 S5.8 S8.0 u g r i z g r i z y G2 G2_BP G2_RP G3 G3_BP G3_RP FUV NUV Age
def _da_rows() -> str:
    lines = []
    teffs = [5000.0, 8000.0, 12000.0, 20000.0, 40000.0]
    loggs = [7.0, 7.5, 8.0, 8.5, 9.0]
    masses = [0.40, 0.55, 0.70, 0.90, 1.20]  # approximate per log g
    ages = [2e9, 1e9, 5e8, 1e8, 1e7]
    for i_g, (lg, mwo) in enumerate(zip(loggs, masses)):
        for i_t, (tf, age) in enumerate(zip(teffs, ages)):
            mbol = 12.0 - 2.5 * math.log10(max(1e-10, tf / 10000.0) ** 4)
            bc = -0.3
            mags = [12.0 + 0.1 * k for k in range(38)]  # 38 magnitude columns
            vals = [tf, lg, mwo, mbol, bc] + mags + [age]
            lines.append("  " + "  ".join(f"{v:.3E}" if abs(v) > 1e5 else f"{v:.4f}" for v in vals))
    return "\n".join(lines) + "\n"


_MINI_TABLE_DA_CONTENT = _DA_HEADER + _da_rows()


@pytest.fixture()
def mini_wd_dir(tmp_path: Path) -> Path:
    """Write minimal Table_DA and Table_DB fixture files; return dir."""
    (tmp_path / "Table_DA").write_text(_MINI_TABLE_DA_CONTENT)
    # DB has same structure as DA (pure-helium); reuse same fixture data.
    db_header = _DA_HEADER.replace("Pure-hydrogen", "Pure-helium")
    (tmp_path / "Table_DB").write_text(db_header + _da_rows())
    return tmp_path


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------
class TestParseGlobalTable:
    def test_teff_grid_shape(self, mini_wd_dir: Path) -> None:
        teff_grid, data = _parse_global_table(mini_wd_dir / "Table_DA")
        assert teff_grid.shape == (5,)
        assert teff_grid[0] == pytest.approx(5000.0)
        assert teff_grid[-1] == pytest.approx(40000.0)

    def test_data_shape(self, mini_wd_dir: Path) -> None:
        teff_grid, data = _parse_global_table(mini_wd_dir / "Table_DA")
        # shape: (5 log g, 5 teff, n_col)
        # n_col = 44 cols total − 2 (Teff, logg) = 42
        assert data.shape == (5, 5, 42)

    def test_mass_column_positive(self, mini_wd_dir: Path) -> None:
        _, data = _parse_global_table(mini_wd_dir / "Table_DA")
        # M/Mo is data[:, :, 0] — must be positive
        assert np.all(data[:, :, 0] > 0)

    def test_wrong_logg_raises(self, tmp_path: Path) -> None:
        # Build a table with only 3 log g values (wrong)
        header = "    3    3  Bad grid\n Teff  log g  M/Mo  Mbol  BC  " + "  U " * 38 + "Age\n"
        rows = ""
        for lg in [7.0, 8.0, 9.0]:
            for tf in [5000.0, 8000.0, 12000.0]:
                vals = [tf, lg, 0.6, 12.0, -0.3] + [12.0] * 38 + [1e9]
                rows += "  " + "  ".join(f"{v:.4f}" for v in vals) + "\n"
        (tmp_path / "Table_DA").write_text(header + rows)
        with pytest.raises(ValueError, match="log g"):
            _parse_global_table(tmp_path / "Table_DA")


# ---------------------------------------------------------------------------
# BergeronGrid tests
# ---------------------------------------------------------------------------
class TestBergeronGrid:
    def test_from_dir_da(self, mini_wd_dir: Path) -> None:
        grid = BergeronGrid.from_dir(mini_wd_dir, "DA")
        assert grid.atm_type == "DA"

    def test_from_dir_db(self, mini_wd_dir: Path) -> None:
        grid = BergeronGrid.from_dir(mini_wd_dir, "DB")
        assert grid.atm_type == "DB"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            BergeronGrid.from_dir(tmp_path, "DA")

    def test_teff_bounds(self, mini_wd_dir: Path) -> None:
        grid = BergeronGrid.from_dir(mini_wd_dir, "DA")
        assert grid.teff_min == pytest.approx(5000.0)
        assert grid.teff_max == pytest.approx(40000.0)

    def test_m_wd_interpolation_on_grid(self, mini_wd_dir: Path) -> None:
        grid = BergeronGrid.from_dir(mini_wd_dir, "DA")
        # Evaluate at a grid point (logg=8.0 → masses[2]=0.70)
        mw = grid.m_wd(8000.0, 8.0)
        assert 0.0 < mw < 2.0  # physical mass range

    def test_synth_phot_returns_dict(self, mini_wd_dir: Path) -> None:
        grid = BergeronGrid.from_dir(mini_wd_dir, "DA")
        result = grid.synth_phot(10000.0, 8.0, a_v=0.1, distance_pc=100.0)
        assert "mags" in result
        assert "m_wd" in result
        assert "extrap_mass" in result
        assert isinstance(result["mags"], dict)

    def test_synth_phot_distance_scales(self, mini_wd_dir: Path) -> None:
        grid = BergeronGrid.from_dir(mini_wd_dir, "DA")
        r10  = grid.synth_phot(10000.0, 8.0, a_v=0.0, distance_pc=10.0)
        r100 = grid.synth_phot(10000.0, 8.0, a_v=0.0, distance_pc=100.0)
        # Distance modulus difference: 5*log10(100/10) = 5
        for band in set(r10["mags"]) & set(r100["mags"]):
            assert r100["mags"][band] == pytest.approx(r10["mags"][band] + 5.0, abs=1e-6)

    def test_synth_phot_extinction_increases_mag(self, mini_wd_dir: Path) -> None:
        grid = BergeronGrid.from_dir(mini_wd_dir, "DA")
        r0 = grid.synth_phot(10000.0, 8.0, a_v=0.0, distance_pc=100.0)
        r1 = grid.synth_phot(10000.0, 8.0, a_v=1.0, distance_pc=100.0)
        # All optical bands should be fainter (higher magnitude) with extinction.
        for band in ("GaiaDR3_G", "PS_r"):
            if band in r0["mags"] and band in r1["mags"]:
                assert r1["mags"][band] > r0["mags"][band]

    def test_extrap_mass_flag_high_logg(self, mini_wd_dir: Path) -> None:
        grid = BergeronGrid.from_dir(mini_wd_dir, "DA")
        # logg = 9.5 is above the grid max (9.0) → extrap_mass must be True
        result = grid.synth_phot(10000.0, 9.5, a_v=0.0, distance_pc=100.0)
        assert result["extrap_mass"] is True

    def test_no_extrap_flag_inside_grid(self, mini_wd_dir: Path) -> None:
        grid = BergeronGrid.from_dir(mini_wd_dir, "DA")
        result = grid.synth_phot(8000.0, 8.0, a_v=0.0, distance_pc=100.0)
        # M_WD for logg=8 in the fixture is 0.70 < 1.3 → no extrap flag
        assert result["extrap_mass"] is False

    def test_band_subset_filter(self, mini_wd_dir: Path) -> None:
        grid = BergeronGrid.from_dir(mini_wd_dir, "DA")
        result = grid.synth_phot(
            10000.0, 8.0, a_v=0.0, distance_pc=100.0, bands=["GaiaDR3_G", "PS_r"]
        )
        assert set(result["mags"].keys()) == {"GaiaDR3_G", "PS_r"}

    def test_available_bands_nonempty(self, mini_wd_dir: Path) -> None:
        grid = BergeronGrid.from_dir(mini_wd_dir, "DA")
        assert len(grid.available_bands) > 10


# ---------------------------------------------------------------------------
# CummingsIFMR tests
# ---------------------------------------------------------------------------
class TestCummingsIFMR:
    @pytest.mark.parametrize("variant", ["MIST", "PARSEC"])
    def test_invalid_variant_raises(self, variant: str) -> None:
        CummingsIFMR(variant)  # should not raise

    def test_invalid_variant_raises_bad(self) -> None:
        with pytest.raises(ValueError):
            CummingsIFMR("BAD")  # type: ignore[arg-type]

    @pytest.mark.parametrize("variant", ["MIST", "PARSEC"])
    def test_forward_inside_grid(self, variant: str) -> None:
        ifmr = CummingsIFMR(variant)  # type: ignore[arg-type]
        mf = ifmr.final_mass(2.0)
        # Expect ~0.6–0.7 M_sun for Mi = 2.0
        assert 0.4 < mf < 1.0

    @pytest.mark.parametrize("variant", ["MIST", "PARSEC"])
    def test_forward_outside_grid_nan(self, variant: str) -> None:
        ifmr = CummingsIFMR(variant)  # type: ignore[arg-type]
        assert math.isnan(ifmr.final_mass(0.5))    # below both Mi_MIN values
        assert math.isnan(ifmr.final_mass(100.0))  # above both Mi_MAX values

    @pytest.mark.parametrize("variant", ["MIST", "PARSEC"])
    def test_final_mass_unc_positive(self, variant: str) -> None:
        ifmr = CummingsIFMR(variant)  # type: ignore[arg-type]
        for mi in [1.5, 2.0, 4.0, 6.0]:
            unc = ifmr.final_mass_unc(mi)
            if not math.isnan(unc):
                assert unc > 0.0

    @pytest.mark.parametrize("variant", ["MIST", "PARSEC"])
    def test_final_mass_unc_outside_range_nan(self, variant: str) -> None:
        ifmr = CummingsIFMR(variant)  # type: ignore[arg-type]
        assert math.isnan(ifmr.final_mass_unc(0.1))
        assert math.isnan(ifmr.final_mass_unc(99.0))

    @pytest.mark.parametrize("variant", ["MIST", "PARSEC"])
    def test_initial_mass_unc_structure(self, variant: str) -> None:
        ifmr = CummingsIFMR(variant)  # type: ignore[arg-type]
        mf = ifmr.final_mass(2.0)
        mi, sigma_mi = ifmr.initial_mass_unc(mf)
        assert not math.isnan(mi)
        assert sigma_mi > 0.0
        # σ_Mi should scale with 1/slope — verify it's in a plausible range (< 1 M_sun).
        assert sigma_mi < 1.0

    @pytest.mark.parametrize("variant", ["MIST", "PARSEC"])
    def test_initial_mass_unc_outside_range(self, variant: str) -> None:
        ifmr = CummingsIFMR(variant)  # type: ignore[arg-type]
        mi, sigma_mi = ifmr.initial_mass_unc(0.01)  # below any segment's Mf range
        assert math.isnan(mi)
        assert math.isnan(sigma_mi)

    @pytest.mark.parametrize("variant", ["MIST", "PARSEC"])
    def test_round_trip(self, variant: str) -> None:
        ifmr = CummingsIFMR(variant)  # type: ignore[arg-type]
        for mi in [1.5, 2.5, 3.0, 4.0, 6.0]:
            mf = ifmr.final_mass(mi)
            mi_back = ifmr.initial_mass(mf)
            if not math.isnan(mf) and not math.isnan(mi_back):
                assert mi_back == pytest.approx(mi, rel=1e-6)

    @pytest.mark.parametrize("variant", ["MIST", "PARSEC"])
    def test_monotone_increasing(self, variant: str) -> None:
        ifmr = CummingsIFMR(variant)  # type: ignore[arg-type]
        mi_arr = np.linspace(1.2, 7.0, 50)
        mf_arr = ifmr.final_mass_vec(mi_arr)
        valid = ~np.isnan(mf_arr)
        assert np.all(np.diff(mf_arr[valid]) > 0)

    def test_ms_lifetime_solar(self) -> None:
        # 1 M_sun → ~10 Gyr
        t = ms_lifetime_yr(1.0)
        assert 8e9 < t < 12e9

    def test_ms_lifetime_heavy(self) -> None:
        # 8 M_sun → much shorter
        t = ms_lifetime_yr(8.0)
        assert t < 1e8  # < 100 Myr

    @pytest.mark.parametrize("variant", ["MIST", "PARSEC"])
    def test_age_gate_young_system(self, variant: str) -> None:
        ifmr = CummingsIFMR(variant)  # type: ignore[arg-type]
        # Mi=2 M_sun → t_MS ~ 1–2 Gyr.  System age 100 Myr → gate should fail.
        mf = ifmr.final_mass(2.0)
        assert ifmr.age_gate_ok(mf, 1e8) is False

    @pytest.mark.parametrize("variant", ["MIST", "PARSEC"])
    def test_age_gate_old_system(self, variant: str) -> None:
        ifmr = CummingsIFMR(variant)  # type: ignore[arg-type]
        # Mi=2 M_sun → t_MS ~ 1–2 Gyr.  System age 10 Gyr → gate should pass.
        mf = ifmr.final_mass(2.0)
        assert ifmr.age_gate_ok(mf, 1e10) is True

    @pytest.mark.parametrize("variant", ["MIST", "PARSEC"])
    def test_age_gate_nan_mf(self, variant: str) -> None:
        ifmr = CummingsIFMR(variant)  # type: ignore[arg-type]
        # mf out of inverse range → False, not exception
        assert ifmr.age_gate_ok(0.01, 1e10) is False


# ---------------------------------------------------------------------------
# wd_model smoke test (minimal dynesty run with fixture grids)
# ---------------------------------------------------------------------------
class TestWDModelSmoke:
    def test_run_wd_fit_smoke(self, mini_wd_dir: Path) -> None:
        """Tiny dynesty run (maxiter=20) to verify wd_model integration."""
        pytest.importorskip("dynesty")
        from darkhunter_sed.phot_sed_io import PhotRow, FLAG_DETECTION
        from darkhunter_sed.wd_model import WDPriorBounds, run_wd_fit

        rows = [
            PhotRow("GaiaDR3_G",  14.5, 0.02, FLAG_DETECTION),
            PhotRow("GaiaDR3_BP", 14.8, 0.03, FLAG_DETECTION),
            PhotRow("GaiaDR3_RP", 14.2, 0.03, FLAG_DETECTION),
            PhotRow("PS_g",       14.9, 0.02, FLAG_DETECTION),
            PhotRow("PS_r",       14.5, 0.02, FLAG_DETECTION),
            PhotRow("2MASS_J",    14.1, 0.04, FLAG_DETECTION),
        ]
        bounds = WDPriorBounds(
            teff_wd=(5000.0, 40000.0),
            logg_wd=(7.0, 9.0),
            av=(0.0, 1.0),
            parallax_mas=(5.0, 50.0),
        )
        results = run_wd_fit(
            rows,
            wd_dir=mini_wd_dir,
            atm_types=("DA",),
            ifmr_variants=("MIST",),
            system_age_yr=0.0,
            prior_bounds=bounds,
            nlive=10,
            maxiter=20,
            seed=7,
        )
        assert len(results) == 1
        res = results[0]
        assert res.atm_type == "DA"
        assert res.ifmr == "MIST"
        assert np.isfinite(res.logevidence)
        assert res.samples.shape[1] == 4
        assert res.m_wd_samples.shape == (res.samples.shape[0],)

    def test_run_wd_fit_summary_keys(self, mini_wd_dir: Path) -> None:
        """Summary dict has all required keys."""
        pytest.importorskip("dynesty")
        from darkhunter_sed.phot_sed_io import PhotRow, FLAG_DETECTION
        from darkhunter_sed.wd_model import WDPriorBounds, run_wd_fit

        rows = [
            PhotRow("GaiaDR3_G",  14.5, 0.02, FLAG_DETECTION),
            PhotRow("GaiaDR3_BP", 14.8, 0.03, FLAG_DETECTION),
        ]
        results = run_wd_fit(
            rows,
            wd_dir=mini_wd_dir,
            atm_types=("DA",),
            ifmr_variants=("MIST",),
            prior_bounds=WDPriorBounds(teff_wd=(5000.0, 40000.0)),
            nlive=10,
            maxiter=15,
            seed=42,
        )
        s = results[0].summary()
        required = {
            "atm_type", "ifmr", "logevidence",
            "m_wd_median", "m_wd_lo", "m_wd_hi",
            "m_i_median", "m_i_ifmr_unc_median",
            "extrap_mass_frac", "extrap_mass",
            "teff_median", "logg_median", "av_median", "parallax_median",
        }
        assert required <= s.keys()


# ---------------------------------------------------------------------------
# Real-data system tests (skipped when WD table directory absent)
# ---------------------------------------------------------------------------
_REAL_WD_DIR = Path("/Users/rfoley/stellar/wd")

@pytest.mark.skipif(
    not _REAL_WD_DIR.is_dir(),
    reason="Bergeron WD tables not found at /Users/rfoley/stellar/wd",
)
class TestRealBergeronGrid:
    """Sanity checks against the actual Bergeron tables."""

    @pytest.fixture(scope="class")
    def da_grid(self) -> BergeronGrid:
        return BergeronGrid.from_dir(_REAL_WD_DIR, "DA")

    @pytest.fixture(scope="class")
    def db_grid(self) -> BergeronGrid:
        return BergeronGrid.from_dir(_REAL_WD_DIR, "DB")

    def test_da_grid_bands(self, da_grid: BergeronGrid) -> None:
        assert "GaiaDR3_G" in da_grid.available_bands
        assert "GALEX_FUV" in da_grid.available_bands

    def test_da_mass_at_logg8(self, da_grid: BergeronGrid) -> None:
        # log g=8.0 → M_WD ~ 0.6 M_sun (canonical value)
        mw = da_grid.m_wd(12000.0, 8.0)
        assert 0.50 < mw < 0.75

    def test_db_mass_at_logg8(self, db_grid: BergeronGrid) -> None:
        mw = db_grid.m_wd(12000.0, 8.0)
        assert 0.50 < mw < 0.75

    def test_da_extrap_flag_logg_9p5(self, da_grid: BergeronGrid) -> None:
        result = da_grid.synth_phot(10000.0, 9.5, a_v=0.0, distance_pc=100.0)
        assert result["extrap_mass"] is True

    def test_da_extrap_flag_inside(self, da_grid: BergeronGrid) -> None:
        result = da_grid.synth_phot(10000.0, 8.0, a_v=0.0, distance_pc=100.0)
        assert result["extrap_mass"] is False


@pytest.mark.skipif(
    not _REAL_WD_DIR.is_dir(),
    reason="Bergeron WD tables not found at /Users/rfoley/stellar/wd",
)
class TestRealSystemsFewExamples:
    """
    End-to-end WD dynesty fits for a handful of representative photometry
    examples.  These use the real Bergeron tables and a small nlive so the
    test suite completes in reasonable time (~60 s per combination).
    """

    @pytest.mark.parametrize(
        "description, rows_spec, expected_m_wd_range",
        [
            (
                "hot_DA_approx_0p6",
                [
                    # Approximate photometry of a 15000 K, logg=8, d=100 pc WD (synthetic)
                    ("GaiaDR3_G",  12.9, 0.02),
                    ("GaiaDR3_BP", 12.7, 0.03),
                    ("GaiaDR3_RP", 13.2, 0.03),
                    ("PS_g",       12.8, 0.02),
                    ("PS_r",       13.0, 0.02),
                    ("SDSS_u",     12.6, 0.05),
                    ("2MASS_J",    14.0, 0.04),
                ],
                (0.4, 1.1),
            ),
            (
                "cool_DA_approx_0p8",
                [
                    # Approximate photometry for a cooler WD
                    ("GaiaDR3_G",  14.5, 0.02),
                    ("GaiaDR3_BP", 14.9, 0.03),
                    ("GaiaDR3_RP", 14.2, 0.03),
                    ("PS_r",       14.5, 0.02),
                    ("2MASS_J",    14.8, 0.05),
                ],
                (0.3, 1.2),
            ),
        ],
    )
    def test_real_system(
        self,
        description: str,
        rows_spec: list[tuple[str, float, float]],
        expected_m_wd_range: tuple[float, float],
    ) -> None:
        pytest.importorskip("dynesty")
        from darkhunter_sed.phot_sed_io import PhotRow, FLAG_DETECTION
        from darkhunter_sed.wd_model import WDPriorBounds, run_wd_fit

        rows = [PhotRow(b, m, e, FLAG_DETECTION) for b, m, e in rows_spec]
        results = run_wd_fit(
            rows,
            wd_dir=_REAL_WD_DIR,
            atm_types=("DA", "DB"),
            ifmr_variants=("MIST", "PARSEC"),
            system_age_yr=0.0,
            prior_bounds=WDPriorBounds(
                teff_wd=(3000.0, 80000.0),
                logg_wd=(7.0, 9.0),
                av=(0.0, 2.0),
                parallax_mas=(1.0, 200.0),
            ),
            nlive=50,
            seed=0,
        )
        # Four combinations should run
        assert len(results) == 4
        for res in results:
            s = res.summary()
            lo, hi = expected_m_wd_range
            assert lo < s["m_wd_median"] < hi, (
                f"{description} {res.atm_type}×{res.ifmr}: "
                f"M_WD={s['m_wd_median']:.3f} outside [{lo},{hi}]"
            )
