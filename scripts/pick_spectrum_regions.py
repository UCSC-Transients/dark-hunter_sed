#!/usr/bin/env python3
"""
Interactive per-order continuum/line region picker for APF spectra.

Toolbar zoom/pan stays active until c/l/Esc; then drag to add regions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from darkhunter_rv import io_utils
from darkhunter_rv.summary_paths import discover_primary_epoch_files

from darkhunter_sed.config import spec_root
from darkhunter_sed.region_picker import (
    DEFAULT_POLY_ORDER,
    FIT_MODEL_SINC2,
    FIT_MODEL_SINC_POLY,
    FitPreviewResult,
    blaze_dict_from_model,
    default_order_view,
    delete_region_at_x,
    fit_model_from_regions,
    fit_order_from_regions,
    fit_shared_sinc_blaze,
    gaia_id_from_spec_path,
    init_orders_state,
    load_regions_json,
    make_document,
    normalize_fit_model,
    normalize_region,
    order_blaze_model_from_regions,
    order_has_stored_blaze,
    save_regions_json,
    saved_view_usable,
    shade_mask_runs,
)
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RadioButtons

from darkhunter_sed import spectrum


def _pick_interactive_backend() -> str:
    import matplotlib

    for backend in ("MacOSX", "TkAgg", "Qt5Agg"):
        try:
            matplotlib.use(backend, force=True)
            return backend
        except ImportError:
            continue
    return matplotlib.get_backend()


def _disable_matplotlib_keymaps() -> None:
    import matplotlib as mpl

    for key in (
        "keymap.yscale",
        "keymap.xscale",
        "keymap.grid",
        "keymap.home",
        "keymap.back",
        "keymap.forward",
    ):
        mpl.rcParams[key] = []


class RegionPickerApp:
    """Matplotlib UI for manual wavelength region selection."""

    def __init__(
        self,
        spectrum_data: dict,
        orders_state: dict,
        spec_path: Path,
        gaia_id: str,
        output_path: Path,
        blaze_calibration=None,
    ) -> None:
        self.spectrum_data = spectrum_data
        self.orders = sorted(int(o) for o in spectrum_data.keys())
        self.state = orders_state
        self.spec_path = spec_path
        self.gaia_id = gaia_id
        self.output_path = output_path
        self.blaze_calibration = blaze_calibration
        self.order_idx = 0
        self.mode = "continuum"
        self.delete_mode = False
        self.history: list[tuple[str, str, str, list[float]]] = []
        self.dirty = False
        self._select_mode = True
        self._nav_lock: str | None = None
        self._flux_line = None
        self._region_spans: list = []
        self._drag_x0: float | None = None
        self._drag_patch = None
        self._view_xlim: tuple[float, float] | None = None
        self._view_ylim: tuple[float, float] | None = None
        self._fit_line = None
        self._fit_scatter = None
        self._fit_shade_spans: list = []
        self._fit_preview: FitPreviewResult | None = None
        self._fit_note = ""
        self._fit_live = False

        self.fig, self.ax = plt.subplots(figsize=(12, 5))
        self.fig.subplots_adjust(bottom=0.16)
        self.fig.canvas.manager.set_window_title("Spectrum region picker")
        self.ax.set_autoscale_on(False)
        _disable_matplotlib_keymaps()

        ax_fit = self.fig.add_axes([0.72, 0.02, 0.14, 0.06])
        self._fit_button = Button(ax_fit, "Fit order")
        self._fit_button.on_clicked(self._on_fit_button)

        ax_radio = self.fig.add_axes([0.02, 0.02, 0.24, 0.07])
        self._model_radio = RadioButtons(
            ax_radio,
            (FIT_MODEL_SINC2, FIT_MODEL_SINC_POLY),
            active=0,
        )
        self._model_radio.on_clicked(self._on_model_radio)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self._hook_toolbar()
        self._redraw(reset_view=True)

    def _order_key(self) -> str:
        return str(self.orders[self.order_idx])

    def _region_key(self) -> str:
        return "continuum_regions" if self.mode == "continuum" else "line_regions"

    def _order_regions(self) -> dict:
        return self.state[self._order_key()]

    def _current_order_regions(self) -> dict:
        return self.state[self._order_key()]

    def _order_has_regions(self) -> bool:
        regions = self._current_order_regions()
        return bool(regions.get("continuum_regions") or regions.get("line_regions"))

    def _should_auto_fit(self) -> bool:
        """Refit after region edits when live fit is on or this order has regions."""
        return self._fit_live or self._order_has_regions()

    def _sync_model_radio(self) -> None:
        model = fit_model_from_regions(self._current_order_regions())
        labels = (FIT_MODEL_SINC2, FIT_MODEL_SINC_POLY)
        idx = labels.index(model) if model in labels else 0
        if self._model_radio.active != idx:
            self._model_radio.set_active(idx)

    def _on_model_radio(self, label: str) -> None:
        regions = self._current_order_regions()
        regions["fit_model"] = normalize_fit_model(label)
        if regions["fit_model"] == FIT_MODEL_SINC_POLY:
            regions.setdefault("poly_order", DEFAULT_POLY_ORDER)
        self.dirty = True
        self._clear_fit_overlay()
        self._fit_note = ""
        self._update_title()
        if self._should_auto_fit():
            self._fit_order()

    def _on_fit_button(self, _event) -> None:
        self._fit_live = True
        self._fit_order()

    def _toggle_fit_model(self) -> None:
        regions = self._current_order_regions()
        cur = fit_model_from_regions(regions)
        new = FIT_MODEL_SINC_POLY if cur == FIT_MODEL_SINC2 else FIT_MODEL_SINC2
        regions["fit_model"] = new
        if new == FIT_MODEL_SINC_POLY:
            regions.setdefault("poly_order", DEFAULT_POLY_ORDER)
        self.dirty = True
        self._sync_model_radio()
        self._fit_note = ""
        self._update_title()
        logging.info("Fit model for order %s: %s", self._order_key(), new)
        if self._should_auto_fit():
            self._fit_order()

    def _fit_order(self) -> None:
        wave, flux = self._current_arrays()
        order_num = self.orders[self.order_idx]
        regions = self._current_order_regions()
        cal_model = None
        if self.blaze_calibration is not None:
            cal_model = self.blaze_calibration.model_for_order(order_num)
        half_width = (
            float(cal_model.line_mask_half_width_angstrom) if cal_model is not None else 22.0
        )
        fitted = fit_shared_sinc_blaze(
            wave,
            flux,
            regions,
            order_num,
            half_width_angstrom=half_width,
        )
        if fitted is not None:
            regions["blaze"] = blaze_dict_from_model(fitted)
            self.dirty = True
        resolved = order_blaze_model_from_regions(regions, order_num, wave, cal_model)
        preview = fit_order_from_regions(
            wave,
            flux,
            regions,
            resolved,
            echelle_order=order_num,
        )
        if preview is None:
            logging.warning("Fit failed for order %s", order_num)
            self._fit_note = "fit failed"
            self._clear_fit_overlay()
            self._update_title()
            return
        n_cont = int(np.sum(preview.continuum_mask))
        final_thr = preview.stage_counts[-1][0] if preview.stage_counts else 0.0
        model_label = "sinc²×poly" if preview.fit_model == FIT_MODEL_SINC_POLY else "sinc²"
        blaze_src = "stored" if order_has_stored_blaze(regions) else "cal"
        self._fit_note = f"{blaze_src} {model_label}: {n_cont}/{wave.size} px @ {final_thr:.2f}"
        self._draw_fit_overlay(wave, flux, preview)
        self._update_title()
        logging.info("Fit order %s: %s", order_num, self._fit_note)

    def _clear_fit_overlay(self) -> None:
        if self._fit_line is not None:
            self._fit_line.remove()
            self._fit_line = None
        if self._fit_scatter is not None:
            self._fit_scatter.remove()
            self._fit_scatter = None
        for patch in self._fit_shade_spans:
            patch.remove()
        self._fit_shade_spans.clear()
        self._fit_preview = None

    def _draw_fit_overlay(
        self,
        wave: np.ndarray,
        flux: np.ndarray,
        preview: FitPreviewResult,
    ) -> None:
        self._clear_fit_overlay()
        self._fit_preview = preview
        cont_mask = preview.continuum_mask
        if preview.mask_bundle is not None:
            excluded = preview.mask_bundle.base_mask & ~cont_mask
        else:
            excluded = ~cont_mask
        self._fit_shade_spans.extend(
            shade_mask_runs(
                self.ax,
                wave,
                cont_mask,
                color="green",
                alpha=0.08,
                zorder=1,
            )
        )
        if np.any(excluded):
            self._fit_shade_spans.extend(
                shade_mask_runs(
                    self.ax,
                    wave,
                    excluded,
                    color="red",
                    alpha=0.10,
                    zorder=1,
                )
            )
        (self._fit_line,) = self.ax.plot(
            wave,
            preview.envelope,
            color="C1",
            lw=1.2,
            ls="--",
            label="fit envelope",
            zorder=4,
        )
        self._fit_scatter = self.ax.scatter(
            wave[cont_mask],
            flux[cont_mask],
            s=10,
            c="C2",
            alpha=0.55,
            zorder=5,
        )
        self.fig.canvas.draw_idle()

    def _toolbar(self):
        return getattr(self.fig.canvas, "toolbar", None)

    def _toolbar_nav_active(self) -> bool:
        if self._nav_lock is not None:
            return True
        toolbar = self._toolbar()
        return bool(toolbar and getattr(toolbar, "mode", ""))

    def _current_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        order_num = self.orders[self.order_idx]
        chunk = self.spectrum_data[order_num]
        wave = np.asarray(chunk["wavelength"], dtype=float)
        flux = np.asarray(chunk["flux"], dtype=float)
        return wave, flux

    def _reset_order_view(self) -> None:
        self._view_xlim = None
        self._view_ylim = None
        wave, flux = self._current_arrays()
        self._apply_view(wave, flux, reset_view=True)

    def _hook_toolbar(self) -> None:
        toolbar = self._toolbar()
        if toolbar is None:
            return

        if hasattr(toolbar, "zoom"):
            orig_zoom = toolbar.zoom

            def zoom(*args, **kwargs):
                self._nav_lock = "zoom"
                self._leave_select_mode()
                return orig_zoom(*args, **kwargs)

            toolbar.zoom = zoom

        if hasattr(toolbar, "pan"):
            orig_pan = toolbar.pan

            def pan(*args, **kwargs):
                self._nav_lock = "pan"
                self._leave_select_mode()
                return orig_pan(*args, **kwargs)

            toolbar.pan = pan

        if hasattr(toolbar, "release_zoom"):
            orig_release_zoom = toolbar.release_zoom

            def release_zoom(event):
                orig_release_zoom(event)
                self._capture_view()
                if self._nav_lock == "zoom":
                    toolbar.mode = "zoom rect"
                    if hasattr(toolbar, "set_message"):
                        toolbar.set_message("Zoom to rectangle")

            toolbar.release_zoom = release_zoom

        if hasattr(toolbar, "release_pan"):
            orig_release_pan = toolbar.release_pan

            def release_pan(event):
                orig_release_pan(event)
                self._capture_view()
                if self._nav_lock == "pan":
                    toolbar.mode = "pan/zoom"
                    if hasattr(toolbar, "set_message"):
                        toolbar.set_message("Pan/zoom with mouse")

            toolbar.release_pan = release_pan

        if hasattr(toolbar, "home"):
            orig_home = toolbar.home

            def home(*args, **kwargs):
                out = orig_home(*args, **kwargs)
                self._reset_order_view()
                self.fig.canvas.draw_idle()
                return out

            toolbar.home = home

    def _capture_view(self) -> None:
        self._view_xlim = tuple(float(v) for v in self.ax.get_xlim())
        self._view_ylim = tuple(float(v) for v in self.ax.get_ylim())

    def _leave_select_mode(self) -> None:
        self._select_mode = False
        self._clear_drag_patch()

    def _clear_toolbar_mode(self) -> None:
        toolbar = self._toolbar()
        if toolbar is None:
            return
        self._nav_lock = None
        if not getattr(toolbar, "mode", ""):
            return
        toolbar.mode = ""
        for attr in ("_active", "_idPress", "_idRelease", "_zoom_info", "_pan_info"):
            if hasattr(toolbar, attr):
                setattr(toolbar, attr, None)
        if hasattr(toolbar, "_update_buttons"):
            toolbar._update_buttons()

    def _enter_select_mode(self) -> None:
        self._select_mode = True
        self._nav_lock = None
        self._clear_toolbar_mode()
        self._clear_drag_patch()
        logging.info("Select mode: %s — drag to add region", self.mode.upper())

    def _title_text(self) -> str:
        order_num = self.orders[self.order_idx]
        if self._toolbar_nav_active():
            mode_label = "ZOOM — press c/l/Esc to select"
        elif self.delete_mode:
            mode_label = f"DELETE {self._region_key()}"
        elif self.mode == "continuum":
            mode_label = "CONTINUUM — drag to add"
        else:
            mode_label = "LINE — drag to add"
        model = fit_model_from_regions(self._current_order_regions())
        model_short = "sinc²×poly" if model == FIT_MODEL_SINC_POLY else "sinc²"
        title = (
            f"Order {order_num} ({self.order_idx + 1}/{len(self.orders)}) | "
            f"{mode_label} | model={model_short} | "
            "u undo | f fit | m model | d delete | n/p order | s save | q quit"
        )
        if self._fit_live or self._order_has_regions():
            title += " | live refit"
        if self._fit_note:
            title += f" | {self._fit_note}"
        return title

    def _update_title(self) -> None:
        self.ax.set_title(self._title_text(), fontsize=10)
        self.fig.canvas.draw_idle()

    def _add_region(self, lo: float, hi: float) -> None:
        region = [lo, hi]
        key = self._region_key()
        self._order_regions()[key].append(region)
        self.history.append(("add", self._order_key(), key, region))
        self.dirty = True
        self._capture_view()
        self._redraw(reset_view=False)

    def _clear_drag_patch(self) -> None:
        if self._drag_patch is not None:
            self._drag_patch.remove()
            self._drag_patch = None
        self._drag_x0 = None

    def _on_press(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if event.button != 1:
            return
        if self._toolbar_nav_active():
            return
        if self.delete_mode:
            key = self._region_key()
            regions = self._order_regions()[key]
            new_list, removed = delete_region_at_x(regions, float(event.xdata))
            if removed is None:
                logging.info("No %s region at %.2f A", key, event.xdata)
                return
            self._order_regions()[key] = new_list
            self.history.append(("del", self._order_key(), key, removed))
            self.dirty = True
            self.delete_mode = False
            self._capture_view()
            self._redraw(reset_view=False)
            self._enter_select_mode()
            return
        if not self._select_mode:
            return
        self._drag_x0 = float(event.xdata)

    def _on_motion(self, event) -> None:
        if self._drag_x0 is None or not self._select_mode or self._toolbar_nav_active():
            return
        if event.inaxes is not self.ax or event.xdata is None:
            return
        lo, hi = normalize_region(self._drag_x0, float(event.xdata))
        if self._drag_patch is not None:
            self._drag_patch.remove()
        ymin, ymax = self.ax.get_ylim()
        drag_color = "green" if self.mode == "continuum" else "red"
        self._drag_patch = self.ax.add_patch(
            Rectangle(
                (lo, ymin),
                hi - lo,
                ymax - ymin,
                facecolor=drag_color,
                alpha=0.25,
                edgecolor=drag_color,
                linewidth=0.8,
            )
        )
        self.fig.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self._drag_x0 is None:
            return
        if event.inaxes is not self.ax or event.xdata is None:
            self._clear_drag_patch()
            return
        if not self._select_mode or self._toolbar_nav_active():
            self._clear_drag_patch()
            return
        lo, hi = normalize_region(self._drag_x0, float(event.xdata))
        self._clear_drag_patch()
        if lo == hi:
            return
        self._add_region(lo, hi)

    def _on_key(self, event) -> None:
        if event.key is None:
            return
        key = event.key.lower()
        if key in ("escape", "enter"):
            self.delete_mode = False
            self._enter_select_mode()
            self._update_title()
        elif key == "c":
            self.mode = "continuum"
            self.delete_mode = False
            self._enter_select_mode()
            self._update_title()
        elif key == "l":
            self.mode = "line"
            self.delete_mode = False
            self._enter_select_mode()
            self._update_title()
        elif key == "d":
            self.delete_mode = True
            self._enter_select_mode()
            logging.info("Delete mode: click inside a %s region", self._region_key())
            self._update_title()
        elif key == "u":
            self._undo()
        elif key == "n":
            self.order_idx = min(self.order_idx + 1, len(self.orders) - 1)
            self.delete_mode = False
            self._view_xlim = None
            self._view_ylim = None
            self._redraw(reset_view=True)
            self._enter_select_mode()
        elif key == "p":
            self.order_idx = max(self.order_idx - 1, 0)
            self.delete_mode = False
            self._view_xlim = None
            self._view_ylim = None
            self._redraw(reset_view=True)
            self._enter_select_mode()
        elif key == "s":
            self._save()
        elif key == "f":
            self._fit_live = True
            self._fit_order()
        elif key == "m":
            self._toggle_fit_model()
        elif key == "q":
            self._quit()

    def _undo(self) -> None:
        if not self.history:
            logging.info("Nothing to undo")
            return
        action, order_key, region_key, region = self.history.pop()
        regions = self.state[order_key][region_key]
        lo, hi = normalize_region(region[0], region[1])
        if action == "add":
            for i, r in enumerate(regions):
                rlo, rhi = normalize_region(r[0], r[1])
                if rlo == lo and rhi == hi:
                    del regions[i]
                    self.dirty = True
                    if order_key != self._order_key():
                        self.order_idx = self.orders.index(int(order_key))
                    self.delete_mode = False
                    self._capture_view()
                    self._redraw(reset_view=False)
                    self._enter_select_mode()
                    return
            logging.warning("Could not undo add %s in order %s", region, order_key)
            return
        regions.append([lo, hi])
        self.dirty = True
        if order_key != self._order_key():
            self.order_idx = self.orders.index(int(order_key))
        self.delete_mode = False
        self._capture_view()
        self._redraw(reset_view=False)
        self._enter_select_mode()

    def _save(self) -> None:
        doc = make_document(self.spec_path, self.gaia_id, self.state)
        save_regions_json(self.output_path, doc)
        self.dirty = False
        logging.info("Saved %s", self.output_path)

    def _quit(self) -> None:
        if self.dirty:
            try:
                ans = input("Unsaved changes. Save before quit? [y/N]: ").strip().lower()
            except EOFError:
                ans = "n"
            if ans in ("y", "yes"):
                self._save()
        import matplotlib.pyplot as plt

        plt.close(self.fig)

    def _clear_artists(self) -> None:
        self._clear_fit_overlay()
        if self._flux_line is not None:
            self._flux_line.remove()
            self._flux_line = None
        for span in self._region_spans:
            span.remove()
        self._region_spans.clear()

    def _apply_view(self, wave: np.ndarray, flux: np.ndarray, *, reset_view: bool) -> None:
        if reset_view:
            self._view_xlim = None
            self._view_ylim = None
        elif self._view_xlim is not None and self._view_ylim is not None:
            if not saved_view_usable(self._view_xlim, self._view_ylim, wave):
                self._view_xlim = None
                self._view_ylim = None
        if self._view_xlim is None or self._view_ylim is None:
            (xlo, xhi), (ylo, yhi) = default_order_view(wave, flux)
        else:
            xlo, xhi = self._view_xlim
            ylo, yhi = self._view_ylim
        self.ax.set_xlim(xlo, xhi)
        self.ax.set_ylim(ylo, yhi)
        self._view_xlim = (float(xlo), float(xhi))
        self._view_ylim = (float(ylo), float(yhi))

    def _redraw(self, *, reset_view: bool) -> None:
        order_num = self.orders[self.order_idx]
        chunk = self.spectrum_data[order_num]
        wave = np.asarray(chunk["wavelength"], dtype=float)
        flux = np.asarray(chunk["flux"], dtype=float)

        self._clear_artists()
        (self._flux_line,) = self.ax.plot(wave, flux, color="0.15", lw=0.8, label="flux")

        order_key = self._order_key()
        regions = self.state[order_key]
        for lo, hi in regions["continuum_regions"]:
            lo_n, hi_n = normalize_region(lo, hi)
            patch = self.ax.axvspan(lo_n, hi_n, color="green", alpha=0.25, lw=0)
            patch.set_in_layout(False)
            self._region_spans.append(patch)
        for lo, hi in regions["line_regions"]:
            lo_n, hi_n = normalize_region(lo, hi)
            patch = self.ax.axvspan(lo_n, hi_n, color="red", alpha=0.25, lw=0)
            patch.set_in_layout(False)
            self._region_spans.append(patch)

        self.ax.set_title(self._title_text(), fontsize=10)
        self.ax.set_xlabel("Wavelength (A, air)")
        self.ax.set_ylabel("Flux")
        self.ax.grid(True, alpha=0.3)
        self._sync_model_radio()
        self._apply_view(wave, flux, reset_view=reset_view)
        if self._should_auto_fit():
            self._fit_order()
        else:
            self.fig.canvas.draw_idle()

    def run(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()


def _default_output_path(spec_path: Path) -> Path:
    return Path("output/masks") / f"regions_{spec_path.stem}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gaia_id", nargs="?", default=None, help="Gaia DR3 id (with --from-spec-root)")
    parser.add_argument("--from-spec-root", action="store_true")
    parser.add_argument("--spec-path", type=Path, default=None, help="APF epoch .txt path")
    parser.add_argument("--epoch", type=int, default=0, help="0-based epoch if using --from-spec-root")
    parser.add_argument("--input", type=Path, default=None, help="Existing regions JSON to resume/edit")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.spec_path is not None:
        spec_path = Path(args.spec_path)
    elif args.from_spec_root:
        if not args.gaia_id:
            parser.error("gaia_id required with --from-spec-root")
        gaia_id = str(args.gaia_id).strip()
        paths = spectrum.sort_spectrum_paths(discover_primary_epoch_files(spec_root(), gaia_id))
        if not paths:
            print("No spectrum files found.", file=sys.stderr)
            return 1
        spec_path = paths[int(np.clip(args.epoch, 0, len(paths) - 1))]
    else:
        parser.error("Provide --spec-path or --from-spec-root with gaia_id")

    if not spec_path.is_file():
        print(f"Spectrum not found: {spec_path}", file=sys.stderr)
        return 1

    gaia_id = str(args.gaia_id).strip() if args.gaia_id else gaia_id_from_spec_path(spec_path)
    if args.input is not None and args.output is None:
        output_path = Path(args.input)
    else:
        output_path = args.output or _default_output_path(spec_path)

    existing_orders: dict | None = None
    if args.input is not None:
        if not args.input.is_file():
            print(f"Input JSON not found: {args.input}", file=sys.stderr)
            return 1
        loaded = load_regions_json(args.input)
        existing_orders = loaded.get("orders")
        json_spec = loaded.get("spec_path")
        json_gaia = loaded.get("gaia_id")
        if json_spec and Path(json_spec).resolve() != spec_path.resolve():
            logging.warning(
                "Input JSON spec_path differs: json=%s cli=%s",
                json_spec,
                spec_path,
            )
        if json_gaia and str(json_gaia) != str(gaia_id):
            logging.warning(
                "Input JSON gaia_id differs: json=%s cli=%s",
                json_gaia,
                gaia_id,
            )
        if isinstance(existing_orders, dict):
            n_with = sum(
                1
                for v in existing_orders.values()
                if v.get("continuum_regions") or v.get("line_regions")
            )
            logging.info(
                "Loaded regions from %s (%d orders with regions)",
                args.input,
                n_with,
            )
        else:
            logging.info("Loaded regions from %s", args.input)

    _header, spectrum_data = io_utils.read_spectrum(str(spec_path))
    if not spectrum_data:
        print("No orders in spectrum file.", file=sys.stderr)
        return 1

    order_numbers = sorted(int(o) for o in spectrum_data.keys())
    orders_state = init_orders_state(order_numbers, existing_orders)
    blaze_calibration = spectrum.load_blaze_calibration()

    _pick_interactive_backend()
    app = RegionPickerApp(
        spectrum_data=spectrum_data,
        orders_state=orders_state,
        spec_path=spec_path,
        gaia_id=gaia_id,
        output_path=output_path,
        blaze_calibration=blaze_calibration,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
