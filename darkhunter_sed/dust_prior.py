"""3D dust-map Av priors for uberMS (dustmaps chain + fallbacks)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

logger = logging.getLogger(__name__)

# Conversions use R_V=3.32 unless noted; Payne likelihood uses R_V=3.1 (documented mismatch).
RV_DUST = 3.32
LEGACY_AV_PRIOR: list = ["tnormal", [0.0, 0.1, 0.0, 0.5]]

# GSF19 Table 1 (bayestar2019): A_V ≈ coeff * map_reddening at R_V=3.1.
BAYESTAR2019_AV_COEFF = 2.271

# ZGR 2023 integrated E → A_V at V (zenodo.6674521); approximate until curve file loaded.
ZGR_E_TO_AV = 2.27

# Fitzpatrick 1999, R_V=3.32: A_V / A_r at SDSS r ~ 622 nm.
AR_TO_AV_RATIO = 1.337


@dataclass(frozen=True)
class AvPriorResult:
    """uberMS Av prior plus provenance for sed_summary."""

    prior: list
    init_av: float
    map_used: str
    prior_kind: str  # informative_3d | upper_limit | legacy
    a_v_med: float | None = None
    sigma: float | None = None
    a_v_ul: float | None = None
    distance_pc: float | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "map_used": self.map_used,
            "prior_kind": self.prior_kind,
            "prior": self.prior,
            "init_av": self.init_av,
            "a_v_med": self.a_v_med,
            "sigma": self.sigma,
            "a_v_ul": self.a_v_ul,
            "distance_pc": self.distance_pc,
            "notes": self.notes,
            **self.extra,
        }


class DustQueryBackend(Protocol):
    """Injectable backend for unit tests."""

    def query_bayestar(self, l_deg: float, b_deg: float, d_pc: float) -> tuple[float, float] | None: ...

    def query_decaps(self, l_deg: float, b_deg: float, d_pc: float) -> tuple[float, float] | None: ...

    def query_edenhofer(self, l_deg: float, b_deg: float, d_pc: float) -> tuple[float, float] | None: ...

    def query_chen_3d(self, l_deg: float, b_deg: float, d_pc: float) -> tuple[float, float] | None: ...

    def query_chen_los(self, l_deg: float, b_deg: float) -> float | None: ...

    def query_csfd(self, l_deg: float, b_deg: float) -> float | None: ...


def _tnormal_informative(loc: float, sigma: float) -> list:
    """Truncated normal: center=loc, scale=sigma, bounds at ±3σ (clipped at 0)."""
    sigma = max(float(sigma), 1e-6)
    loc = max(float(loc), 0.0)
    low = max(0.0, loc - 3.0 * sigma)
    high = loc + 3.0 * sigma
    return ["tnormal", [loc, sigma, low, high]]


def _tnormal_upper_limit(a_v_ul: float) -> list:
    """Upper-limit prior: loc=0, scale=ul/3, high=ul."""
    ul = max(float(a_v_ul), 1e-6)
    return ["tnormal", [0.0, ul / 3.0, 0.0, ul]]


def _init_from_prior(prior: list) -> float:
    loc = float(prior[1][0])
    low, high = float(prior[1][2]), float(prior[1][3])
    init = float(np.clip(loc, low, high))
    # Av=0 exactly at the truncated-normal lower bound breaks uberMS SVI init for some stars.
    if high > low and init <= low + 1e-8:
        init = min(max(low + 0.01, 0.01), high)
    return init


def ebv_to_av(ebv: float) -> float:
    return float(RV_DUST) * float(ebv)


def bayestar_reddening_to_av(reddening: float) -> float:
    return float(BAYESTAR2019_AV_COEFF) * float(reddening)


def ar_to_av(a_r: float) -> float:
    return float(AR_TO_AV_RATIO) * float(a_r)


def edenhofer_e_to_av(e_integrated: float) -> float:
    return float(ZGR_E_TO_AV) * float(e_integrated)


def galactic_lb(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    g = c.galactic
    return float(g.l.deg), float(g.b.deg)


def in_bayestar_footprint(dec_deg: float) -> bool:
    return float(dec_deg) > -30.0


def in_decaps_footprint(l_deg: float, b_deg: float) -> bool:
    l = float(l_deg) % 360.0
    b = float(b_deg)
    in_l = (l > 239.0) or (l < 6.0)
    return in_l and abs(b) < 10.0


def in_chen_footprint(l_deg: float, b_deg: float) -> bool:
    l = float(l_deg) % 360.0
    b = float(b_deg)
    return (140.0 < l < 240.0) and (-60.0 < b < 40.0)


def in_edenhofer_distance_range(d_pc: float) -> bool:
    return 69.0 <= float(d_pc) <= 1250.0


def _as_scalar_positive(x: Any) -> float | None:
    try:
        v = float(np.asarray(x).ravel()[0])
    except (TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(v) or v < 0.0:
        return None
    return v


def _percentile_pair(query_fn: Callable[..., Any], coords: Any) -> tuple[float, float] | None:
    """p16, p50, p84 → (p50, sigma) with sigma = (p84-p16)/2."""
    try:
        p16, p50, p84 = query_fn(coords, mode="percentile", pct=[16, 50, 84])
    except TypeError:
        try:
            p50 = query_fn(coords, mode="median")
            p16 = p84 = p50
        except Exception:
            return None
    except Exception:
        return None
    med = _as_scalar_positive(p50)
    if med is None:
        return None
    lo = _as_scalar_positive(p16)
    hi = _as_scalar_positive(p84)
    if lo is not None and hi is not None and hi > lo:
        sigma = (hi - lo) / 2.0
    else:
        sigma = max(med * 0.1, 0.01)
    return med, sigma


class _DustmapsBackend:
    """Live dustmaps queries (lazy map init)."""

    def __init__(self) -> None:
        self._bayestar = None
        self._decaps = None
        self._edenhofer = None
        self._chen = None
        self._csfd = None

    def _skycoord(self, l_deg: float, b_deg: float, d_pc: float | None) -> Any:
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        if d_pc is not None and math.isfinite(d_pc) and d_pc > 0:
            return SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, distance=d_pc * u.pc, frame="galactic")
        return SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic")

    def query_bayestar(self, l_deg: float, b_deg: float, d_pc: float) -> tuple[float, float] | None:
        try:
            from dustmaps.bayestar import BayestarQuery
        except ImportError:
            return None
        if self._bayestar is None:
            try:
                self._bayestar = BayestarQuery(version="bayestar2019")
            except Exception as exc:
                logger.debug("Bayestar init failed: %s", exc)
                return None
        coords = self._skycoord(l_deg, b_deg, d_pc)
        pair = _percentile_pair(self._bayestar, coords)
        if pair is None:
            return None
        med_r, sig_r = pair
        return bayestar_reddening_to_av(med_r), bayestar_reddening_to_av(sig_r)

    def query_decaps(self, l_deg: float, b_deg: float, d_pc: float) -> tuple[float, float] | None:
        try:
            from dustmaps.decaps import DECaPSQueryLite
        except ImportError:
            return None
        if self._decaps is None:
            try:
                self._decaps = DECaPSQueryLite()
            except Exception as exc:
                logger.debug("DECaPS init failed: %s", exc)
                return None
        coords = self._skycoord(l_deg, b_deg, d_pc)
        pair = _percentile_pair(self._decaps.query, coords)
        if pair is None:
            return None
        return ebv_to_av(pair[0]), ebv_to_av(pair[1])

    def query_edenhofer(self, l_deg: float, b_deg: float, d_pc: float) -> tuple[float, float] | None:
        try:
            from dustmaps.edenhofer2023 import Edenhofer2023Query
        except ImportError:
            return None
        if self._edenhofer is None:
            try:
                self._edenhofer = Edenhofer2023Query(integrated=True)
            except Exception as exc:
                logger.debug("Edenhofer init failed: %s", exc)
                return None
        coords = self._skycoord(l_deg, b_deg, d_pc)
        try:
            mean_e = float(self._edenhofer.query(coords, mode="mean"))
            std_e = float(self._edenhofer.query(coords, mode="std"))
        except Exception:
            return None
        if not math.isfinite(mean_e) or mean_e < 0:
            return None
        if not math.isfinite(std_e) or std_e <= 0:
            std_e = max(mean_e * 0.1, 0.01)
        return edenhofer_e_to_av(mean_e), edenhofer_e_to_av(std_e)

    def query_chen_3d(self, l_deg: float, b_deg: float, d_pc: float) -> tuple[float, float] | None:
        try:
            from dustmaps.chen2014 import Chen2014Query
        except ImportError:
            return None
        if self._chen is None:
            try:
                self._chen = Chen2014Query()
            except Exception as exc:
                logger.debug("Chen init failed: %s", exc)
                return None
        coords = self._skycoord(l_deg, b_deg, d_pc)
        try:
            out = self._chen.query(coords, return_sigma=True)
        except Exception:
            return None
        if isinstance(out, tuple) and len(out) >= 2:
            a_r, sig = out[0], out[1]
        else:
            a_r, sig = out, None
        med = _as_scalar_positive(a_r)
        if med is None:
            return None
        sig_v = _as_scalar_positive(sig)
        if sig_v is None:
            sig_v = max(med * 0.1, 0.01)
        return ar_to_av(med), ar_to_av(sig_v)

    def query_chen_los(self, l_deg: float, b_deg: float) -> float | None:
        try:
            from dustmaps.chen2014 import Chen2014Query
        except ImportError:
            return None
        if self._chen is None:
            try:
                self._chen = Chen2014Query()
            except Exception:
                return None
        coords = self._skycoord(l_deg, b_deg, None)
        try:
            out = self._chen.query(coords, return_sigma=False)
        except Exception:
            return None
        med = _as_scalar_positive(out)
        if med is None:
            return None
        return ar_to_av(med)

    def query_csfd(self, l_deg: float, b_deg: float) -> float | None:
        try:
            from dustmaps.csfd import CSFDQuery
        except ImportError:
            return None
        if self._csfd is None:
            try:
                self._csfd = CSFDQuery()
            except Exception as exc:
                logger.debug("CSFD init failed: %s", exc)
                return None
        coords = self._skycoord(l_deg, b_deg, None)
        try:
            ebv = float(self._csfd(coords))
        except Exception:
            return None
        if not math.isfinite(ebv) or ebv < 0:
            return None
        return ebv_to_av(ebv)


def build_av_prior(
    *,
    ra_deg: float,
    dec_deg: float,
    parallax_mas: float,
    parallax_err_mas: float | None = None,
    backend: DustQueryBackend | None = None,
    use_dustmaps: bool = True,
) -> AvPriorResult:
    """
    Build uberMS ``Av`` prior from 3D dust maps + parallax distance.

    Chain: Bayestar2019 → DECaPS → Edenhofer → Chen (3D). Fallbacks: Chen LOS
    upper limit → CSFD upper limit → legacy ``tnormal(0, 0.1, 0, 0.5)``.
    """
    del parallax_err_mas  # reserved for distance uncertainty propagation
    if not use_dustmaps:
        prior = list(LEGACY_AV_PRIOR)
        return AvPriorResult(
            prior=prior,
            init_av=_init_from_prior(prior),
            map_used="legacy",
            prior_kind="legacy",
            notes="dustmaps disabled",
        )

    if not (math.isfinite(ra_deg) and math.isfinite(dec_deg) and math.isfinite(parallax_mas) and parallax_mas > 0):
        logger.warning("Invalid coords/parallax for dust prior; using legacy Av prior")
        prior = list(LEGACY_AV_PRIOR)
        return AvPriorResult(
            prior=prior,
            init_av=_init_from_prior(prior),
            map_used="legacy",
            prior_kind="legacy",
            notes="invalid coordinates or parallax",
        )

    l_deg, b_deg = galactic_lb(ra_deg, dec_deg)
    d_pc = 1000.0 / float(parallax_mas)
    be = backend if backend is not None else _DustmapsBackend()

    attempts: list[tuple[str, Callable[[], tuple[float, float] | None]]] = []
    if in_bayestar_footprint(dec_deg):
        attempts.append(("bayestar2019", lambda: be.query_bayestar(l_deg, b_deg, d_pc)))
    if in_decaps_footprint(l_deg, b_deg):
        attempts.append(("decaps", lambda: be.query_decaps(l_deg, b_deg, d_pc)))
    if in_edenhofer_distance_range(d_pc):
        attempts.append(("edenhofer2023", lambda: be.query_edenhofer(l_deg, b_deg, d_pc)))
    if in_chen_footprint(l_deg, b_deg):
        attempts.append(("chen2014_3d", lambda: be.query_chen_3d(l_deg, b_deg, d_pc)))

    for map_name, fn in attempts:
        pair = fn()
        if pair is None:
            continue
        med, sigma = pair
        if med is None or not math.isfinite(med):
            continue
        prior = _tnormal_informative(med, sigma)
        return AvPriorResult(
            prior=prior,
            init_av=_init_from_prior(prior),
            map_used=map_name,
            prior_kind="informative_3d",
            a_v_med=med,
            sigma=sigma,
            distance_pc=d_pc,
            notes=f"3D dust prior from {map_name}",
            extra={"l_deg": l_deg, "b_deg": b_deg},
        )

    if in_chen_footprint(l_deg, b_deg):
        a_ul = be.query_chen_los(l_deg, b_deg)
        if a_ul is not None and math.isfinite(a_ul) and a_ul > 0:
            prior = _tnormal_upper_limit(a_ul)
            return AvPriorResult(
                prior=prior,
                init_av=_init_from_prior(prior),
                map_used="chen2014_los",
                prior_kind="upper_limit",
                a_v_ul=a_ul,
                distance_pc=d_pc,
                notes="Chen LOS upper limit",
                extra={"l_deg": l_deg, "b_deg": b_deg},
            )

    a_ul = be.query_csfd(l_deg, b_deg)
    if a_ul is not None and math.isfinite(a_ul) and a_ul > 0:
        prior = _tnormal_upper_limit(a_ul)
        return AvPriorResult(
            prior=prior,
            init_av=_init_from_prior(prior),
            map_used="csfd",
            prior_kind="upper_limit",
            a_v_ul=a_ul,
            distance_pc=d_pc,
            notes="CSFD (Chiang 2023) upper limit",
            extra={"l_deg": l_deg, "b_deg": b_deg},
        )

    logger.warning("No dust map value for (l,b)=(%.2f,%.2f); legacy Av prior", l_deg, b_deg)
    prior = list(LEGACY_AV_PRIOR)
    return AvPriorResult(
        prior=prior,
        init_av=_init_from_prior(prior),
        map_used="legacy",
        prior_kind="legacy",
        distance_pc=d_pc,
        notes="no dust map match",
        extra={"l_deg": l_deg, "b_deg": b_deg},
    )


def build_av_prior_from_fit_data(fit_data: dict, *, use_dustmaps: bool = True) -> AvPriorResult:
    """Convenience: RA/Dec/parallax from uberMS ``getdata`` dict."""
    ra = fit_data.get("RA")
    dec = fit_data.get("Dec")
    plx = fit_data.get("parallax")
    if ra is None or dec is None or plx is None:
        prior = list(LEGACY_AV_PRIOR)
        return AvPriorResult(
            prior=prior,
            init_av=_init_from_prior(prior),
            map_used="legacy",
            prior_kind="legacy",
            notes="missing RA/Dec/parallax in fit_data",
        )
    plx_err = float(plx[1]) if isinstance(plx, (list, tuple)) and len(plx) > 1 else None
    return build_av_prior(
        ra_deg=float(ra),
        dec_deg=float(dec),
        parallax_mas=float(plx[0]),
        parallax_err_mas=plx_err,
        use_dustmaps=use_dustmaps,
    )
