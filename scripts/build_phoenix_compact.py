#!/usr/bin/env python3
"""
Build a compact, rebinned PHOENIX ACES grid from the HiRes archive (Path 2).

Streams the HiRes grid one file at a time — download to a temp file → rebin
(flux-conserving, constant-R log-λ) → write compact file → delete temp — so the
~6 MB HiRes spectra never accumulate on disk. Output layout and file format:
:mod:`darkhunter_sed.phoenix_compact`; read back by
:class:`darkhunter_sed.phoenix_grid.PhoenixGrid` (auto-detected).

Use cases
---------
Full grid into the default ``PHOENIX_DIR`` (resumable; rerun after failures)::

    python3 scripts/build_phoenix_compact.py --workers 4

Solar-α subset typical of Path-2 targets, preview first::

    python3 scripts/build_phoenix_compact.py --subgrids Z-0.0,Z-0.5,Z+0.5,Z-1.0 \\
        --teff-min 2300 --teff-max 12000 --logg-min 0 --logg-max 6 --dry-run

Rebuild from an existing local HiRes tree; check Δmag vs HiRes for 5 models::

    python3 scripts/build_phoenix_compact.py --src-local /path/HiResFITS --validate 5

Limitations
-----------
- Network access via stdlib ``urllib``; directory listings are parsed from the
  mirror's HTML index (``%2B``/``%3D`` decoded to ``+``/``=``).
- ``--validate`` needs the HiRes spectra again (downloaded to temp when remote)
  and synphot bandpasses (``PYSYN_CDBS``) for every registered Path-2 band.
- A destination holds one ``R``; a different ``--resolution`` needs another ``--dest``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from http.client import HTTPException, HTTPResponse
from pathlib import Path
from typing import TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from darkhunter_sed.config import phoenix_dir
from darkhunter_sed.filters_synphot import load_bandpass, registered_bands
from darkhunter_sed.phoenix_compact import (
    DEFAULT_HIRES_URL,
    DEFAULT_RESOLUTION,
    GRID_SUBDIR,
    HIRES_WAVE_NAME,
    NOMINAL_HIRES_N_WAVE,
    CompactJob,
    bin_centers,
    build_compact_from_hires,
    compact_flux_is_valid,
    compare_band_mags,
    expected_n_bins,
    fits_image_bytes,
    log_lambda_edges,
    make_jobs,
    parse_directory_listing,
    read_compact_wave,
    rebin_flux_conserving,
    split_done,
    write_compact_wave,
)
from darkhunter_sed.phoenix_grid import compact_wave_name, parse_subgrid_name

_CHUNK = 1 << 20
_USER_AGENT = "darkhunter-sed-phoenix-compact/1"
# erg/s/cm^2/cm → erg/s/cm^2/Å
_CM_TO_AA = 1.0e8

_T = TypeVar("_T")


class DiskSpaceError(RuntimeError):
    """Free space on the destination fell below ``--min-free-gb``."""


@dataclass(frozen=True, slots=True)
class Fetcher:
    """
    HiRes source: remote mirror (``base_url``) or local tree (``local_root``).

    Parameters
    ----------
    base_url :
        Mirror root containing ``WAVE_…fits`` and ``PHOENIX-ACES-AGSS-COND-2011/``.
    local_root :
        Local HiRes root; when set, nothing is downloaded or deleted.
    retries, timeout_s :
        Network retry count (exponential backoff) and per-request timeout.
    """

    base_url: str
    local_root: Path | None
    retries: int
    timeout_s: float

    def url_for(self, *parts: str) -> str:
        """Mirror URL for path components (``+``/``=`` percent-encoded)."""
        base = self.base_url if self.base_url.endswith("/") else self.base_url + "/"
        return base + "/".join(quote(p, safe="") for p in parts)

    def _open(self, url: str) -> HTTPResponse:
        return urlopen(Request(url, headers={"User-Agent": _USER_AGENT}), timeout=self.timeout_s)

    def _retry(self, what: str, fn: Callable[[], _T]) -> _T:
        delay = 2.0
        for attempt in range(1, self.retries + 2):
            try:
                return fn()
            except HTTPError as exc:
                if exc.code < 500 and exc.code != 429:
                    raise
                err: Exception = exc
            except (URLError, HTTPException, TimeoutError, ConnectionError, OSError) as exc:
                err = exc
            if attempt > self.retries:
                raise RuntimeError(f"{what}: giving up after {attempt} attempts: {err}") from err
            print(f"  retry {attempt}/{self.retries} {what}: {err}", flush=True)
            time.sleep(delay)
            delay = min(delay * 2.0, 60.0)
        raise AssertionError("unreachable")

    def list_dir(self, *parts: str) -> list[tuple[str, bool]]:
        """``(name, is_dir)`` children of a source directory."""
        if self.local_root is not None:
            d = self.local_root.joinpath(*parts)
            return [(p.name, p.is_dir()) for p in sorted(d.iterdir())]
        url = self.url_for(*parts) + "/"

        def get() -> str:
            with self._open(url) as resp:
                return resp.read().decode("utf-8", errors="replace")

        page = self._retry(f"list {url}", get)
        return [(e.name, e.is_dir) for e in parse_directory_listing(page)]

    def fetch(self, parts: Sequence[str], tmp_dir: Path) -> tuple[Path, int]:
        """
        Local path to a source file and bytes downloaded (``0`` for local).

        Remote files go to a unique temp file in ``tmp_dir``; the caller deletes it.
        """
        if self.local_root is not None:
            return self.local_root.joinpath(*parts), 0
        url = self.url_for(*parts)

        def get() -> tuple[Path, int]:
            fd, tmp_name = tempfile.mkstemp(dir=tmp_dir, suffix=".fits.part")
            tmp = Path(tmp_name)
            n = 0
            try:
                with self._open(url) as resp, open(fd, "wb") as fh:
                    expected = resp.headers.get("Content-Length")
                    while chunk := resp.read(_CHUNK):
                        fh.write(chunk)
                        n += len(chunk)
                if expected is not None and int(expected) != n:
                    raise ConnectionError(f"short read {n}/{expected} bytes")
                return tmp, n
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise

        return self._retry(f"download {url}", get)

    @property
    def is_remote(self) -> bool:
        return self.local_root is None


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a compact rebinned PHOENIX ACES grid from HiRes (streams one file at a time).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dest", type=Path, default=None,
                   help="Compact grid root (default: PHOENIX_DIR or config DEFAULT_PHOENIX_DIR)")
    p.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                   help="Constant resolving power R = 1/dln(lambda) of the compact grid")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--src-url", default=DEFAULT_HIRES_URL, help="HiRes mirror root URL")
    src.add_argument("--src-local", type=Path, default=None,
                     help="Existing local HiRes root (no download)")
    p.add_argument("--subgrids", default=None,
                   help="Comma list of subgrid dirs, e.g. Z-0.0,Z-0.5,Z-0.0.Alpha=+0.40 (default: all)")
    p.add_argument("--teff-min", type=float, default=None)
    p.add_argument("--teff-max", type=float, default=None)
    p.add_argument("--logg-min", type=float, default=None)
    p.add_argument("--logg-max", type=float, default=None)
    p.add_argument("--workers", type=int, default=1, help="Parallel download/rebin workers")
    p.add_argument("--retries", type=int, default=5, help="Network retries per request")
    p.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout (s)")
    p.add_argument("--min-free-gb", type=float, default=2.0,
                   help="Abort when free space on --dest drops below this")
    p.add_argument("--dry-run", action="store_true",
                   help="List selected models and estimated sizes; download nothing")
    p.add_argument("--validate", type=int, default=0, metavar="N",
                   help="Compare HiRes vs compact AB mags for N sample models in all registered bands")
    p.add_argument("--validate-only", action="store_true",
                   help="Skip building; only run --validate")
    args = p.parse_args(argv)
    if args.resolution <= 0:
        p.error("--resolution must be positive")
    if args.workers < 1:
        p.error("--workers must be ≥ 1")
    if args.validate_only and args.validate <= 0:
        p.error("--validate-only needs --validate N")
    return args


def _select_subgrids(fetcher: Fetcher, wanted: str | None) -> list[str]:
    available = [n for n, is_dir in fetcher.list_dir(GRID_SUBDIR) if is_dir and parse_subgrid_name(n)]
    if wanted is None:
        return available
    req = [s.strip() for s in wanted.split(",") if s.strip()]
    missing = [s for s in req if s not in available]
    if missing:
        raise SystemExit(f"Unknown subgrid(s) {missing}; available: {available}")
    return req


def _collect_jobs(fetcher: Fetcher, args: argparse.Namespace, dest: Path) -> list[CompactJob]:
    jobs: list[CompactJob] = []
    for sub in _select_subgrids(fetcher, args.subgrids):
        names = [n for n, is_dir in fetcher.list_dir(GRID_SUBDIR, sub) if not is_dir]
        sub_jobs = make_jobs(
            dest, sub, names, resolution=args.resolution,
            teff_min=args.teff_min, teff_max=args.teff_max,
            logg_min=args.logg_min, logg_max=args.logg_max,
        )
        print(f"  {sub:22s} {len(sub_jobs):6d} models", flush=True)
        jobs.extend(sub_jobs)
    return jobs


def _ensure_compact_wave(
    fetcher: Fetcher, dest: Path, tmp_dir: Path, resolution: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Native HiRes wavelengths and compact edges; writes/validates the compact WAVE file."""
    path, _ = fetcher.fetch([HIRES_WAVE_NAME], tmp_dir)
    try:
        native = np.asarray(fits.getdata(path), dtype=np.float64).reshape(-1)
    finally:
        if fetcher.is_remote:
            path.unlink(missing_ok=True)
    edges = log_lambda_edges(float(native[0]), float(native[-1]), resolution)
    wave_path = dest / compact_wave_name(resolution)
    if wave_path.is_file():
        _, old_edges, old_res = read_compact_wave(wave_path)
        if old_res != resolution or old_edges.shape != edges.shape or not np.allclose(
            old_edges, edges, rtol=1e-12, atol=0.0
        ):
            raise SystemExit(f"{wave_path} exists with a different grid; use another --dest")
    else:
        others = sorted(dest.glob("WAVE_PHOENIX-ACES-AGSS-COND-2011-R*.fits"))
        if others:
            raise SystemExit(f"{dest} already holds {[o.name for o in others]}; use another --dest")
        source = (fetcher.url_for(HIRES_WAVE_NAME) if fetcher.local_root is None
                  else str(fetcher.local_root / HIRES_WAVE_NAME))
        write_compact_wave(wave_path, edges, resolution=resolution, source_wave=source)
    return native, edges


def _dest_usage_bytes(dest: Path, resolution: int) -> int:
    total = 0
    wave = dest / compact_wave_name(resolution)
    if wave.is_file():
        total += wave.stat().st_size
    grid = dest / GRID_SUBDIR
    if grid.is_dir():
        total += sum(p.stat().st_size for p in grid.rglob(f"*-R{resolution}.fits"))
    return total


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    raise AssertionError("unreachable")


def _build_one(
    job: CompactJob,
    fetcher: Fetcher,
    tmp_dir: Path,
    native: NDArray[np.float64],
    edges: NDArray[np.float64],
    resolution: int,
    min_free_bytes: int,
) -> int:
    if shutil.disk_usage(tmp_dir).free < min_free_bytes:
        raise DiskSpaceError(f"free space below {_fmt_bytes(min_free_bytes)}")
    src, n_bytes = fetcher.fetch([GRID_SUBDIR, job.subgrid, job.hires_name], tmp_dir)
    try:
        build_compact_from_hires(
            src, job.compact_path, native_wave=native, edges=edges, resolution=resolution,
            source_name=f"{job.subgrid}/{job.hires_name}",
        )
    finally:
        if fetcher.is_remote:
            src.unlink(missing_ok=True)
    return n_bytes


def _run_build(
    todo: Sequence[CompactJob],
    fetcher: Fetcher,
    tmp_dir: Path,
    native: NDArray[np.float64],
    edges: NDArray[np.float64],
    args: argparse.Namespace,
) -> tuple[int, int, list[str]]:
    """Returns ``(n_built, bytes_downloaded, failures)``."""
    min_free = int(args.min_free_gb * 1024**3)
    built = 0
    downloaded = 0
    failures: list[str] = []
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures: dict[Future[int], CompactJob] = {
            pool.submit(_build_one, j, fetcher, tmp_dir, native, edges, args.resolution, min_free): j
            for j in todo
        }
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_EXCEPTION)
            for fut in done:
                job = futures[fut]
                try:
                    downloaded += fut.result()
                    built += 1
                except DiskSpaceError as exc:
                    for f in pending:
                        f.cancel()
                    raise SystemExit(f"ABORT: {exc} on {tmp_dir}; free space and rerun (resumable)")
                except Exception as exc:  # noqa: BLE001 - report and continue; rerun resumes
                    failures.append(f"{job.subgrid}/{job.hires_name}: {exc}")
                    print(f"  FAILED {job.subgrid}/{job.hires_name}: {exc}", flush=True)
                n_done = built + len(failures)
                if n_done % 25 == 0 or n_done == len(todo):
                    rate = n_done / max(time.monotonic() - t0, 1e-9)
                    eta = (len(todo) - n_done) / rate if rate > 0 else float("inf")
                    print(f"  [{n_done}/{len(todo)}] built={built} failed={len(failures)} "
                          f"downloaded={_fmt_bytes(downloaded)} eta={eta / 60:.1f} min", flush=True)
    return built, downloaded, failures


def _run_validate(
    jobs: Sequence[CompactJob],
    n: int,
    fetcher: Fetcher,
    tmp_dir: Path,
    native: NDArray[np.float64],
    edges: NDArray[np.float64],
    resolution: int,
) -> None:
    if not jobs:
        print("validate: no models selected")
        return
    picks = [jobs[int(i)] for i in np.unique(np.linspace(0, len(jobs) - 1, min(n, len(jobs))).round())]
    bandpasses = {}
    for band in registered_bands():
        try:
            bandpasses[band] = load_bandpass(band)
        except (OSError, ValueError) as exc:
            print(f"  validate: skip band {band}: {exc}")
    centers = bin_centers(edges)
    n_bins = centers.size
    worst: dict[str, np.ndarray] = {b: np.zeros(3) for b in bandpasses}
    for job in picks:
        src, _ = fetcher.fetch([GRID_SUBDIR, job.subgrid, job.hires_name], tmp_dir)
        try:
            native_flux = np.asarray(fits.getdata(src), dtype=np.float64).reshape(-1)
        finally:
            if fetcher.is_remote:
                src.unlink(missing_ok=True)
        if compact_flux_is_valid(job.compact_path, n_bins=n_bins, resolution=resolution):
            compact_flux = np.asarray(fits.getdata(job.compact_path), dtype=np.float64)
        else:
            compact_flux = rebin_flux_conserving(native, native_flux, edges).astype(np.float32)
        cmp = compare_band_mags(native, native_flux / _CM_TO_AA, centers,
                                np.asarray(compact_flux, dtype=np.float64) / _CM_TO_AA, bandpasses)
        print(f"  validated {job.subgrid}/{job.hires_name}", flush=True)
        for band, c in cmp.items():
            d = np.abs([c.compact_exact - c.reference, c.compact_pipeline - c.reference,
                        c.native_pipeline - c.reference])
            worst[band] = np.maximum(worst[band], d)
    print(f"\nmax |Δmag| over {len(picks)} models (reference = HiRes, throughput on native grid)")
    print(f"  {'band':12s} {'compact':>9s} {'compact':>9s} {'HiRes':>9s}")
    print(f"  {'':12s} {'(exact)':>9s} {'(fit)':>9s} {'(fit)':>9s}")
    for band in sorted(worst):
        a, b, c = worst[band]
        print(f"  {band:12s} {a:9.5f} {b:9.5f} {c:9.5f}")
    allmax = max(float(v[0]) for v in worst.values()) if worst else float("nan")
    print(f"  rebinning error (compact exact) max over bands: {allmax:.5f} mag")
    print("  '(fit)' columns sample the spectrum on the stock bandpass wavesets, as Path-2 fits do.")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    dest = phoenix_dir(args.dest)
    local = args.src_local.expanduser().resolve() if args.src_local is not None else None
    fetcher = Fetcher(base_url=args.src_url, local_root=local, retries=args.retries,
                      timeout_s=args.timeout)
    print(f"source: {local or args.src_url}")
    print(f"dest:   {dest}  (R={args.resolution})")
    print("selecting models:")
    jobs = _collect_jobs(fetcher, args, dest)
    wave_path = dest / compact_wave_name(args.resolution)
    n_bins = read_compact_wave(wave_path)[0].size if wave_path.is_file() else None
    todo, done = split_done(jobs, n_bins=n_bins, resolution=args.resolution)

    nb = n_bins if n_bins is not None else expected_n_bins(args.resolution)
    est_hires = fits_image_bytes(NOMINAL_HIRES_N_WAVE, bytes_per_value=4, header_blocks=4)
    est_compact = fits_image_bytes(nb, bytes_per_value=4, header_blocks=4)
    print(f"selected {len(jobs)} models: {len(done)} already valid, {len(todo)} to build")
    print(f"estimated download: {_fmt_bytes(len(todo) * est_hires)} "
          f"(~{_fmt_bytes(est_hires)} per HiRes file, deleted after rebinning)")
    print(f"estimated compact footprint: {_fmt_bytes(len(jobs) * est_compact)} "
          f"(~{_fmt_bytes(est_compact)} per model, {nb} bins)")
    if args.dry_run:
        for job in todo[:20]:
            print(f"  would build {job.subgrid}/{job.hires_name}")
        if len(todo) > 20:
            print(f"  ... and {len(todo) - 20} more")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=dest, prefix=".tmp-build-"))
    t0 = time.monotonic()
    built, downloaded, failures = 0, 0, []
    try:
        native, edges = _ensure_compact_wave(fetcher, dest, tmp_dir, args.resolution)
        if not args.validate_only and todo:
            built, downloaded, failures = _run_build(todo, fetcher, tmp_dir, native, edges, args)
        if args.validate > 0:
            _run_validate(jobs, args.validate, fetcher, tmp_dir, native, edges, args.resolution)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\nsummary")
    print(f"  models selected: {len(jobs)}  built: {built}  skipped(valid): {len(done)}  "
          f"failed: {len(failures)}")
    print(f"  downloaded: {_fmt_bytes(downloaded)}  elapsed: {(time.monotonic() - t0) / 60:.1f} min")
    print(f"  disk used by compact grid: {_fmt_bytes(_dest_usage_bytes(dest, args.resolution))}")
    print(f"  set: export PHOENIX_DIR={dest}")
    if failures:
        print("  failures (rerun the same command to retry):")
        for f in failures[:50]:
            print(f"    {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
