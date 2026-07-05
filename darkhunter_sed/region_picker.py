"""Helpers for manual continuum/line wavelength region selection."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

import numpy as np

from darkhunter_sed.config import masks_dir
from darkhunter_rv.blaze import (
    FIT_MODEL_SINC2,
    FIT_MODEL_SINC_POLY,
    HB_REST_A,
    IterativeBlazeFitResult,
    OrderBlazeModel,
    blaze_line_mask,
    fit_order_blaze_iterative,
    shared_blaze_order_envelope,
    strong_lines_in_span,
)


class StoredBlazeParams(TypedDict):
    center_angstrom: float
    width_angstrom: float
    power: float


class OrderRegions(TypedDict, total=False):
    continuum_regions: list[list[float]]
    line_regions: list[list[float]]
    fit_model: str
    poly_order: int
    blaze: StoredBlazeParams


DEFAULT_POLY_ORDER = 2

logger = logging.getLogger(__name__)


def resolve_regions_json_for_star(
    gaia_id: str,
    explicit: str | Path | None = None,
    *,
    auto_resolve: bool = True,
) -> Path | None:
    """
    Resolve picker regions JSON for one Gaia source.

    Priority: explicit path (if file) → newest ``regions_Gaia_DR3_<id>_*.json`` in
    masks_dir → None (calibrated blaze only).
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path.resolve()
    if not auto_resolve:
        return None
    pattern = f"regions_Gaia_DR3_{str(gaia_id).strip()}_*.json"
    candidates = sorted(masks_dir().glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


class ManualMaskBundle(NamedTuple):
    """Per-order masks for blaze continuum fitting."""

    base_mask: np.ndarray
    fixed_line_mask: np.ndarray
    fixed_cont_mask: np.ndarray
    cr_mask: np.ndarray


class FitPreviewResult(NamedTuple):
    """Continuum fit preview for one echelle order."""

    envelope: np.ndarray
    continuum_mask: np.ndarray
    stage_counts: tuple[tuple[float, int], ...]
    fit_result: IterativeBlazeFitResult | None
    mask_bundle: ManualMaskBundle | None
    fit_model: str


def normalize_fit_model(value: str | None) -> str:
    """Return a supported per-order fit model name."""
    if value == FIT_MODEL_SINC_POLY:
        return FIT_MODEL_SINC_POLY
    return FIT_MODEL_SINC2


def normalize_poly_order(value: int | float | None) -> int:
    """Clamp polynomial order to 1–4."""
    if value is None:
        return DEFAULT_POLY_ORDER
    return max(1, min(int(value), 4))


def fit_model_from_regions(order_regions: OrderRegions | None) -> str:
    if order_regions is None:
        return FIT_MODEL_SINC2
    return normalize_fit_model(order_regions.get("fit_model"))


def poly_order_from_regions(order_regions: OrderRegions | None) -> int:
    if order_regions is None:
        return DEFAULT_POLY_ORDER
    return normalize_poly_order(order_regions.get("poly_order"))


def _coerce_blaze_dict(value: Any) -> StoredBlazeParams | None:
    """Return blaze params dict when all required keys are present."""
    if not isinstance(value, dict):
        return None
    keys = ("center_angstrom", "width_angstrom", "power")
    if not all(k in value for k in keys):
        return None
    try:
        return StoredBlazeParams(
            center_angstrom=float(value["center_angstrom"]),
            width_angstrom=float(value["width_angstrom"]),
            power=float(value["power"]),
        )
    except (TypeError, ValueError):
        return None


def blaze_dict_from_model(model: OrderBlazeModel) -> StoredBlazeParams:
    """Serialize shared sinc² shape for regions JSON."""
    return StoredBlazeParams(
        center_angstrom=float(model.center_angstrom),
        width_angstrom=float(model.width_angstrom),
        power=float(model.power),
    )


def order_has_stored_blaze(order_regions: OrderRegions | None) -> bool:
    if order_regions is None:
        return False
    return _coerce_blaze_dict(order_regions.get("blaze")) is not None


def order_blaze_model_from_regions(
    order_regions: OrderRegions | None,
    echelle_order: int,
    wavelength: np.ndarray | list[float],
    fallback: OrderBlazeModel | None,
) -> OrderBlazeModel | None:
    """Resolve blaze shape: picker-stored sinc² overrides calibrated fallback."""
    w = np.asarray(wavelength, dtype=float)
    blaze_d = _coerce_blaze_dict(order_regions.get("blaze") if order_regions else None)
    if blaze_d is not None:
        half_width = 22.0
        if fallback is not None:
            half_width = float(fallback.line_mask_half_width_angstrom)
        return OrderBlazeModel(
            echelle_order=int(echelle_order),
            model="sinc2",
            center_angstrom=blaze_d["center_angstrom"],
            width_angstrom=blaze_d["width_angstrom"],
            power=blaze_d["power"],
            n_spectra_fit=1,
            wavelength_min=float(np.min(w)),
            wavelength_max=float(np.max(w)),
            rest_line_angstrom=HB_REST_A,
            line_mask_half_width_angstrom=half_width,
        )
    return fallback


def normalize_region(lo: float, hi: float) -> tuple[float, float]:
    """Return (lo, hi) with lo <= hi."""
    lo_f, hi_f = float(lo), float(hi)
    if lo_f <= hi_f:
        return lo_f, hi_f
    return hi_f, lo_f


def regions_to_spans(
    regions: list[list[float]] | list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Normalize and sort wavelength intervals by lower edge."""
    spans = [normalize_region(r[0], r[1]) for r in regions]
    spans.sort(key=lambda pair: pair[0])
    return spans


def merge_regions(
    regions: list[list[float]] | list[tuple[float, float]],
    *,
    gap_tol: float = 0.01,
) -> list[tuple[float, float]]:
    """Normalize, sort, and merge overlapping or abutting intervals."""
    spans = regions_to_spans(regions)
    if not spans:
        return []
    merged: list[tuple[float, float]] = [spans[0]]
    for lo, hi in spans[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi + float(gap_tol):
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def subtract_spans(
    minuend: list[tuple[float, float]],
    subtrahend: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Remove subtrahend intervals from minuend (interval arithmetic)."""
    if not minuend:
        return []
    if not subtrahend:
        return list(minuend)
    out: list[tuple[float, float]] = []
    for lo, hi in minuend:
        parts = [(lo, hi)]
        for s_lo, s_hi in subtrahend:
            next_parts: list[tuple[float, float]] = []
            for p_lo, p_hi in parts:
                if s_hi < p_lo or s_lo > p_hi:
                    next_parts.append((p_lo, p_hi))
                    continue
                if s_lo > p_lo:
                    next_parts.append((p_lo, s_lo))
                if s_hi < p_hi:
                    next_parts.append((s_hi, p_hi))
            parts = next_parts
        out.extend(parts)
    return merge_regions(out)


def resolve_order_regions(
    continuum_regions: list[list[float]] | list[tuple[float, float]],
    line_regions: list[list[float]] | list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Merge spans within each type; subtract line from continuum (line wins)."""
    cont = merge_regions(continuum_regions)
    line = merge_regions(line_regions)
    cont = subtract_spans(cont, line)
    return cont, line


def resolve_order_regions_dict(regions: OrderRegions) -> OrderRegions:
    """Return copy with merged spans and line-over-continuum resolved."""
    cont, line = resolve_order_regions(
        regions.get("continuum_regions", []),
        regions.get("line_regions", []),
    )
    out: OrderRegions = {
        "continuum_regions": [list(pair) for pair in cont],
        "line_regions": [list(pair) for pair in line],
    }
    if "fit_model" in regions:
        out["fit_model"] = normalize_fit_model(regions.get("fit_model"))
    if "poly_order" in regions:
        out["poly_order"] = normalize_poly_order(regions.get("poly_order"))
    blaze_d = _coerce_blaze_dict(regions.get("blaze"))
    if blaze_d is not None:
        out["blaze"] = blaze_d
    return out


def find_region_at_x(
    regions: list[list[float]] | list[tuple[float, float]],
    x: float,
) -> int | None:
    """Index of first region containing x, or None."""
    x_f = float(x)
    for i, (lo, hi) in enumerate(regions):
        lo_n, hi_n = normalize_region(lo, hi)
        if lo_n <= x_f <= hi_n:
            return i
    return None


def empty_order_regions() -> OrderRegions:
    return {"continuum_regions": [], "line_regions": []}


def init_orders_state(
    order_numbers: list[int],
    existing: dict[str, OrderRegions] | None = None,
    *,
    resolve: bool = True,
) -> dict[str, OrderRegions]:
    """Build per-order region dict; copy from existing JSON where present."""
    src = existing or {}
    out: dict[str, OrderRegions] = {}
    for order_num in order_numbers:
        key = str(int(order_num))
        if key in src:
            raw: OrderRegions = {
                "continuum_regions": [list(r) for r in src[key].get("continuum_regions", [])],
                "line_regions": [list(r) for r in src[key].get("line_regions", [])],
            }
            if "fit_model" in src[key]:
                raw["fit_model"] = normalize_fit_model(src[key].get("fit_model"))
            if "poly_order" in src[key]:
                raw["poly_order"] = normalize_poly_order(src[key].get("poly_order"))
            blaze_d = _coerce_blaze_dict(src[key].get("blaze"))
            if blaze_d is not None:
                raw["blaze"] = blaze_d
            out[key] = resolve_order_regions_dict(raw) if resolve else raw
        else:
            out[key] = empty_order_regions()
    return out


def make_document(
    spec_path: str | Path,
    gaia_id: str,
    orders: dict[str, OrderRegions],
) -> dict[str, Any]:
    return {
        "spec_path": str(Path(spec_path).resolve()),
        "gaia_id": str(gaia_id),
        "wave_system": "air",
        "orders": orders,
    }


def _resolved_orders_copy(orders: dict[str, OrderRegions]) -> dict[str, OrderRegions]:
    out: dict[str, OrderRegions] = {}
    for key, regions in orders.items():
        out[key] = resolve_order_regions_dict(regions)
    return out


def load_regions_json(path: str | Path, *, resolve: bool = True) -> dict[str, Any]:
    """Load region picker JSON; raises FileNotFoundError or json.JSONDecodeError."""
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("region JSON root must be an object")
    orders = data.get("orders")
    if orders is not None and not isinstance(orders, dict):
        raise ValueError("orders must be an object")
    if resolve and isinstance(orders, dict):
        data = dict(data)
        data["orders"] = _resolved_orders_copy(orders)
    return data


def save_regions_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write region JSON with merged, resolved wavelength intervals per order."""
    out = dict(data)
    orders = out.get("orders")
    if isinstance(orders, dict):
        out["orders"] = _resolved_orders_copy(orders)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def gaia_id_from_spec_path(spec_path: str | Path) -> str:
    """Extract Gaia DR3 id from APF filename stem when possible."""
    stem = Path(spec_path).stem
    if stem.startswith("Gaia_DR3_"):
        parts = stem.split("_")
        if len(parts) >= 3:
            return parts[2]
    return stem


def delete_region_at_x(
    regions: list[list[float]],
    x: float,
) -> tuple[list[list[float]], list[float] | None]:
    """Remove region containing x; return (new_list, removed_region_or_None)."""
    idx = find_region_at_x(regions, x)
    if idx is None:
        return regions, None
    removed = list(regions[idx])
    new_list = [list(r) for i, r in enumerate(regions) if i != idx]
    return new_list, removed


def order_edge_exclusion_mask(
    n_pixels: int,
    *,
    edge_pixels: int = 5,
) -> np.ndarray:
    """True for interior pixels; False for first/last ``edge_pixels`` indices."""
    n = int(n_pixels)
    ok = np.ones(n, dtype=bool)
    ep = max(0, int(edge_pixels))
    if n <= 2 * ep:
        return np.zeros(n, dtype=bool)
    ok[:ep] = False
    ok[n - ep :] = False
    return ok


def regions_to_pixel_mask(
    wavelength: np.ndarray | list[float],
    spans: list[tuple[float, float]] | list[list[float]],
) -> np.ndarray:
    """True where wavelength falls inside any span."""
    w = np.asarray(wavelength, dtype=float)
    mask = np.zeros(w.shape, dtype=bool)
    for lo, hi in merge_regions(spans):
        mask |= (w >= lo) & (w <= hi)
    return mask


def build_manual_base_mask(
    wavelength: np.ndarray | list[float],
    flux: np.ndarray | list[float],
    *,
    continuum_regions: list[list[float]] | None = None,
    line_regions: list[list[float]] | None = None,
    edge_pixels: int = 5,
    rest_lines: list[float] | None = None,
    half_width_angstrom: float = 22.0,
) -> ManualMaskBundle:
    """
    Build blaze continuum masks from manual regions.

    Auto ``blaze_line_mask`` always defines the candidate pool. Manual continuum
    spans supplement (union) auto selection and are forced on via ``fixed_cont_mask``.
    Manual line spans supplement exclusions via ``fixed_line_mask``.
    """
    from darkhunter_rv.continuum import outlier_mask

    w = np.asarray(wavelength, dtype=float)
    f = np.asarray(flux, dtype=float)
    n = w.size
    valid = np.isfinite(w) & np.isfinite(f) & (f > 0)
    cr_mask = outlier_mask(w, f)
    edge_ok = order_edge_exclusion_mask(n, edge_pixels=edge_pixels)

    cont_spans, line_spans = resolve_order_regions(
        continuum_regions or [],
        line_regions or [],
    )
    fixed_line = regions_to_pixel_mask(w, line_spans)
    in_cont = regions_to_pixel_mask(w, cont_spans)

    manual_cont = in_cont & cr_mask & edge_ok & valid & ~fixed_line
    fixed_cont = manual_cont

    if rest_lines:
        auto = blaze_line_mask(
            w,
            rest_lines=rest_lines,
            half_width_angstrom=half_width_angstrom,
        )
    else:
        auto = valid.copy()
    auto_base = auto & cr_mask & edge_ok & valid
    base = auto_base | manual_cont if cont_spans else auto_base

    return ManualMaskBundle(
        base_mask=base,
        fixed_line_mask=fixed_line,
        fixed_cont_mask=fixed_cont,
        cr_mask=cr_mask & valid,
    )


def fit_shared_sinc_blaze(
    wavelength: np.ndarray | list[float],
    flux: np.ndarray | list[float],
    order_regions: OrderRegions | None,
    echelle_order: int,
    *,
    half_width_angstrom: float = 22.0,
) -> OrderBlazeModel | None:
    """
    Fit shared sinc² blaze (center, width, power) on one reference spectrum.

    Uses supplement manual masks when regions are present; amplitude is not stored.
    """
    w = np.asarray(wavelength, dtype=float)
    f = np.asarray(flux, dtype=float)
    rests = strong_lines_in_span(float(np.min(w)), float(np.max(w)))
    continuum_regions = order_regions.get("continuum_regions") if order_regions else None
    line_regions = order_regions.get("line_regions") if order_regions else None
    bundle = build_manual_base_mask(
        w,
        f,
        continuum_regions=continuum_regions,
        line_regions=line_regions,
        rest_lines=rests if rests else None,
        half_width_angstrom=half_width_angstrom,
    )
    result = fit_order_blaze_iterative(
        w,
        f,
        initial_mask=bundle.base_mask,
        fixed_line_mask=bundle.fixed_line_mask,
        fixed_cont_mask=bundle.fixed_cont_mask,
        cr_mask=bundle.cr_mask,
        fit_model=FIT_MODEL_SINC2,
    )
    if result is None:
        return None
    return OrderBlazeModel(
        echelle_order=int(echelle_order),
        model="sinc2",
        center_angstrom=float(result.center_angstrom),
        width_angstrom=float(result.width_angstrom),
        power=float(result.power),
        n_spectra_fit=1,
        wavelength_min=float(np.min(w)),
        wavelength_max=float(np.max(w)),
        rest_line_angstrom=HB_REST_A,
        line_mask_half_width_angstrom=float(half_width_angstrom),
    )


def fit_order_from_regions(
    wavelength: np.ndarray | list[float],
    flux: np.ndarray | list[float],
    order_regions: OrderRegions | None,
    blaze_model: OrderBlazeModel | None,
    *,
    echelle_order: int | None = None,
    fit_model: str | None = None,
    poly_order: int | None = None,
) -> FitPreviewResult | None:
    """
    Shared blaze shape with per-spectrum scale (optional log-poly multiplier).

    Blaze shape resolves from picker-stored ``blaze`` in ``order_regions``, then
    ``blaze_model`` fallback. Returns envelope curve and continuum mask.
    """
    w = np.asarray(wavelength, dtype=float)
    f = np.asarray(flux, dtype=float)
    order_num = int(echelle_order if echelle_order is not None else (blaze_model.echelle_order if blaze_model else 0))
    resolved = order_blaze_model_from_regions(order_regions, order_num, w, blaze_model)
    if resolved is None:
        return None
    blaze_model = resolved
    model = normalize_fit_model(fit_model or fit_model_from_regions(order_regions))
    p_order = normalize_poly_order(
        poly_order if poly_order is not None else poly_order_from_regions(order_regions)
    )
    half_width = float(blaze_model.line_mask_half_width_angstrom)
    rests = strong_lines_in_span(float(np.min(w)), float(np.max(w)))
    bundle: ManualMaskBundle | None = None
    base_mask = None
    fixed_line = None
    fixed_cont = None
    cr_mask = None
    if order_regions is not None and (
        order_regions.get("continuum_regions") or order_regions.get("line_regions")
    ):
        bundle = build_manual_base_mask(
            w,
            f,
            continuum_regions=order_regions.get("continuum_regions"),
            line_regions=order_regions.get("line_regions"),
            rest_lines=rests if rests else None,
            half_width_angstrom=half_width,
        )
        base_mask = bundle.base_mask
        fixed_line = bundle.fixed_line_mask
        fixed_cont = bundle.fixed_cont_mask
        cr_mask = bundle.cr_mask

    result = shared_blaze_order_envelope(
        w,
        f,
        blaze_model,
        rest_lines=rests if rests else None,
        base_mask=base_mask,
        fixed_line_mask=fixed_line,
        fixed_cont_mask=fixed_cont,
        cr_mask=cr_mask,
        fit_model=model,
        poly_order=p_order,
    )
    if result is None:
        return None
    return FitPreviewResult(
        envelope=result.envelope,
        continuum_mask=result.continuum_mask,
        stage_counts=(),
        fit_result=None,
        mask_bundle=bundle,
        fit_model=result.fit_model,
    )


def shade_mask_gaps(
    ax,
    wavelength: np.ndarray | list[float],
    mask: np.ndarray,
    *,
    color: str,
    alpha: float,
    zorder: int = 1,
) -> list:
    """Shade wavelength intervals where mask is False; return axvspan artists."""
    w = np.asarray(wavelength, dtype=float)
    use = np.asarray(mask, bool)
    artists: list = []
    if w.size == 0:
        return artists
    in_gap = False
    gap_lo = 0.0
    for i in range(w.size):
        if not use[i] and not in_gap:
            in_gap = True
            gap_lo = float(w[i])
        elif use[i] and in_gap:
            in_gap = False
            patch = ax.axvspan(
                gap_lo,
                float(w[i - 1]),
                color=color,
                alpha=alpha,
                lw=0,
                zorder=zorder,
            )
            artists.append(patch)
    if in_gap:
        patch = ax.axvspan(gap_lo, float(w[-1]), color=color, alpha=alpha, lw=0, zorder=zorder)
        artists.append(patch)
    return artists


def shade_mask_runs(
    ax,
    wavelength: np.ndarray | list[float],
    mask: np.ndarray,
    *,
    color: str,
    alpha: float,
    zorder: int = 1,
) -> list:
    """Shade wavelength intervals where mask is True; return axvspan artists."""
    w = np.asarray(wavelength, dtype=float)
    use = np.asarray(mask, bool)
    artists: list = []
    if w.size == 0:
        return artists
    in_run = False
    run_lo = 0.0
    for i in range(w.size):
        if use[i] and not in_run:
            in_run = True
            run_lo = float(w[i])
        elif not use[i] and in_run:
            in_run = False
            patch = ax.axvspan(
                run_lo,
                float(w[i - 1]),
                color=color,
                alpha=alpha,
                lw=0,
                zorder=zorder,
            )
            artists.append(patch)
    if in_run:
        patch = ax.axvspan(run_lo, float(w[-1]), color=color, alpha=alpha, lw=0, zorder=zorder)
        artists.append(patch)
    return artists


def order_regions_from_doc(
    doc: dict[str, Any],
    order_num: int,
) -> OrderRegions | None:
    """Lookup resolved regions for one echelle order."""
    orders = doc.get("orders")
    if not isinstance(orders, dict):
        return None
    key = str(int(order_num))
    if key not in orders:
        return None
    return resolve_order_regions_dict(orders[key])


def default_order_view(
    wave: np.ndarray | list[float],
    flux: np.ndarray | list[float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Full-order x/y limits from wavelength and flux arrays."""
    w = np.asarray(wave, dtype=float)
    f = np.asarray(flux, dtype=float)
    wmin, wmax = float(np.min(w)), float(np.max(w))
    finite = f[np.isfinite(f)]
    if finite.size:
        ymin, ymax = float(np.min(finite)), float(np.max(finite))
    else:
        ymin, ymax = 0.0, 1.0
    pad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
    return (wmin, wmax), (ymin - pad, ymax + pad)


def saved_view_usable(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    wave: np.ndarray | list[float],
) -> bool:
    """True if saved axis limits fit this order's wavelength span."""
    w = np.asarray(wave, dtype=float)
    wmin, wmax = float(np.min(w)), float(np.max(w))
    span = wmax - wmin
    if xlim[1] - xlim[0] < 1.0:
        return False
    if xlim[1] < wmin or xlim[0] > wmax:
        return False
    if xlim[0] < wmin - 0.05 * span:
        return False
    if ylim[1] - ylim[0] <= 0.0:
        return False
    return True
