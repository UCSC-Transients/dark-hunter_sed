# dark-hunter_sed operations

Batch spectrophotometric fitting with uberMS. **Run the RV pipeline first** so
`Gaia_DR3_<id>_summary.txt` files exist (mask-CCF RVs + Gaia GSP-Phot priors).

## Environment (local macOS)

| Variable | Typical value |
|----------|----------------|
| `PY` | `/opt/local/bin/python` (has **dustmaps**) |
| `RV_REPO` | `/Users/rfoley/darkhunter/rvs/dark-hunter_rv` |
| `SED_REPO` | this repo |
| `STELLAR_ROOT` | `/Users/rfoley/stellar` |
| `DARKHUNTER_SED_MODELS_DIR` | `$STELLAR_ROOT/gaia/models` |
| `SPEC_ROOT` | `/Users/rfoley/darkhunter/rvs/data` |
| `DARKHUNTER_OUTPUT_DIR` | `/Users/rfoley/darkhunter/rvs/output` |

```bash
export SED_REPO=/Users/rfoley/darkhunter/seds/dark-hunter_sed
export RV_REPO=/Users/rfoley/darkhunter/rvs/dark-hunter_rv
export STELLAR_ROOT=/Users/rfoley/stellar
export DARKHUNTER_SED_MODELS_DIR=$STELLAR_ROOT/gaia/models
export SPEC_ROOT=/Users/rfoley/darkhunter/rvs/data
export DARKHUNTER_OUTPUT_DIR=/Users/rfoley/darkhunter/rvs/output
export DARKHUNTER_SED_OUTPUT_DIR=$SED_REPO/output
export DARKHUNTER_SED_PHOTOMETRY_DIR=$SED_REPO/output/photometry
export PYTHONPATH=$RV_REPO:$SED_REPO
```

Use MacPorts Python for dust Av priors (default anaconda `python3` may lack dustmaps):

```bash
bash scripts/run_local.sh -m darkhunter_sed.cli <gaia_id> --from-spec-root --plot
# or explicitly:
PY=/opt/local/bin/python bash scripts/run_local.sh -m darkhunter_sed.cli <gaia_id> --from-spec-root
```

First Bayestar map load can take ~1 minute; later queries reuse cached HDF5.

### JAX / XLA_FLAGS on macOS

Conda `(base)` and older JAX CPU builds may set `XLA_FLAGS=--xla_cpu_use_thunk_runtime=false`. Apple Metal XLA aborts on that flag (`Unknown flags in XLA_FLAGS`). `scripts/run_local.sh` unsets shell `XLA_FLAGS` before launch; legacy uberMS `runSVI` re-sets it on import, and `darkhunter_sed.fit` calls `sanitize_xla_flags_env()` immediately after importing uberMS so UMS/UTP SVI can run.

`run_local.sh` also sets `JAX_PLATFORMS=cpu` (uberMS SVI is CPU-only). If Metal still initializes on MacPorts Python, run `pip uninstall jax-metal` on that interpreter, or use conda after `pip install dustmaps matplotlib`:

```bash
PY=/Users/rfoley/anaconda3/bin/python3 bash scripts/run_local.sh -m darkhunter_sed.cli <gaia_id> --from-spec-root --plot
```

Pinned JAX stack for uberMS: see [`requirements-svi.txt`](../requirements-svi.txt) (`jax==0.5.2`, `numpyro==0.15.0`). Do not upgrade MacPorts JAX to 0.7+ for SVI.

### Gaia-informed UMS init (priors unchanged)

Gaia GSP-Phot / FLAME values seed **SVI starting points only** (`init_to_value`); sampling priors stay the legacy uberMS widths (uniform EEP, IMF mass, uniform `[Fe/H]`, etc.):

| Init param | Source |
|------------|--------|
| `initial_[Fe/H]` | Summary `MH` |
| `initial_Mass` | Summary `Mass_FLAME` when present |
| `EEP` | Summary `Age_FLAME` (Gyr) → MIST age→EEP interpolation |

Re-query Gaia after RV pipeline updates to populate FLAME fields:

```bash
python3 $RV_REPO/scripts/ensure_pipeline_summaries.py \
  --gaia-id <id> --spec-root "$SPEC_ROOT" --output-dir "$DARKHUNTER_OUTPUT_DIR" --force-gaia
```

Diagnose init without a full SVI run:

```bash
bash scripts/run_local.sh scripts/diagnose_ums_init.py <gaia_id> --from-spec-root
# optional 1-step NumPyro init probe (loads NNs):
bash scripts/run_local.sh scripts/diagnose_ums_init.py <gaia_id> --from-spec-root --probe-svi
```

SB1 systems with large epoch-to-epoch RV drift may need `--vrad-prior uniform` or higher `--vrad-err-inflate` (see fit log warning when span > 10 km/s).

### RV summary before SED fit

SED reads canonical `Gaia_DR3_<id>_summary.txt` from `DARKHUNTER_OUTPUT_DIR` (`[GAIA METADATA]` + `[PIPELINE RESULTS]`). Legacy `<id>_summary.txt` files are not enough. Backfill with the RV repo (17-digit Gaia ids are supported):

```bash
export PYTHONPATH=$RV_REPO
python3 $RV_REPO/scripts/ensure_pipeline_summaries.py \
  --gaia-id <id> --spec-root "$SPEC_ROOT" --output-dir "$DARKHUNTER_OUTPUT_DIR" --force-gaia
```

## Environment (ziggy)

| Variable | Typical value |
|----------|----------------|
| `REPO` | `/data2/darkhunter/dark-hunter_sed` |
| `RV_REPO` | `/data2/darkhunter/dark-hunter_rv` |
| `STELLAR_ROOT` | `/data2/stellar` (uberMS, ThePayne, MISTy, models) |
| `SPEC_ROOT` | `/data2/gaia_stars/apf_reductions` |
| `DARKHUNTER_OUTPUT_DIR` | `$RV_REPO/output` |
| `DARKHUNTER_SED_OUTPUT_DIR` | `$REPO/output` |
| `DARKHUNTER_SED_MODELS_DIR` | optional override for NN weights |
| `PY` | `gaia-env` python |

```bash
export PYTHONPATH=$RV_REPO:$REPO
export STELLAR_ROOT=/data2/stellar
export SPEC_ROOT=/data2/gaia_stars/apf_reductions
export DARKHUNTER_OUTPUT_DIR=$RV_REPO/output
```

## One-time photometry per star

Photometry is **auto-gathered** on first `cli` / `batch` run when `{gaia_id}_phot.fits` is absent. To gather or refresh manually:

```bash
cd $REPO
python -m darkhunter_sed.photometry_gather <gaia_id> -d output/photometry
```

Use `--no-auto-gather-phot` on `cli` or `batch` to require an existing FITS file.

## Blaze regions (picker → fit / plot)

1. Pick continuum/line regions and fit blaze on a reference star:
   ```bash
   bash scripts/run_local.sh scripts/pick_spectrum_regions.py <ref_gaia_id> --from-spec-root
   ```
   **Fit** refits sinc² per order; **Save** writes `output/masks/regions_Gaia_DR3_<ref_id>_*.json` with per-order `blaze`.

2. Apply shared blaze to another star (auto-resolve newest per-star file, or explicit path):
   ```bash
   bash scripts/run_local.sh scripts/plot_order_blaze.py <gaia_id> --from-spec-root --order 35 \
     --regions-json output/masks/regions_Gaia_DR3_<ref_id>_epoch_1.json --blaze-only
   ```

3. Run uberMS with the same regions (auto per star, or shared `--regions-json` on batch):
   ```bash
   bash scripts/run_local.sh -m darkhunter_sed.cli <gaia_id> --from-spec-root --plot
   bash scripts/run_local.sh -m darkhunter_sed.batch --update --no-progress \
     --regions-json output/masks/regions_Gaia_DR3_<ref_id>_epoch_1.json
   ```

Regions resolution order: explicit `--regions-json` → newest `regions_Gaia_DR3_<id>_*.json` in `output/masks/` → calibrated `blaze_orders_apf.json` only.

## Single-star fit

```bash
python -m darkhunter_sed.cli <gaia_id> --from-spec-root -D output/photometry
```

Writes:

- `output/samples/Gaia_DR3_<id>_ums.fits`
- `output/samples/Gaia_DR3_<id>_utp.fits`
- `output/sed_summaries/Gaia_DR3_<id>_sed_summary.json`

## Production batch

```bash
# Incremental: skip when inputs are older than outputs
python -m darkhunter_sed.batch --update --no-progress

# One star, force refit
python -m darkhunter_sed.batch --gaia-id <id> --force --no-progress
```

**Skip rules (`--update`):**

- Refit when any epoch spectrum, photometry FITS, RV summary, or regions JSON is newer than
  `sed_summary.json` / sample FITS.
- Skip stars with fewer than 2 epoch spectra (UMS requirement).
- Stars without `{gaia_id}_phot.fits` are **refit candidates** when auto-gather is enabled (default); use `--no-auto-gather-phot` to skip them.

## Cron

After RV cron (`dark-hunter_rv/scripts/cron_update_rv_website.sh`):

```bash
bash scripts/cron_update_sed.sh
```

Log: `$REPO/logs/cron_sed.log`

## Priors

- **RV:** per-epoch `vrad_i` normal priors from `[PIPELINE RESULTS]` (2× error inflation, 2 km/s floor by default). Legacy `# File Summary` rows are parsed when `[PIPELINE RESULTS]` is absent.
- **Gaia:** from summary `[GAIA METADATA]`; `--force-redownload` on CLI/batch re-queries TAP.
- **Av (extinction):** default dustmaps chain at parallax distance — Bayestar2019 → DECaPS → Edenhofer → Chen (3D); fallbacks Chen LOS upper limit → CSFD upper limit → legacy `tnormal(0,0.1,0,0.5)`. Disable with `--no-dust-av-prior`. Provenance stored in `sed_summary.json` as `av_prior`.

### Dust map downloads (one-time per machine)

```bash
python -c "import dustmaps.bayestar; dustmaps.bayestar.fetch()"
python -c "import dustmaps.decaps; dustmaps.decaps.fetch()"
python -c "import dustmaps.edenhofer2023; dustmaps.edenhofer2023.fetch()"
python -c "import dustmaps.chen2014; dustmaps.chen2014.fetch()"
python -c "import dustmaps.csfd; dustmaps.csfd.fetch()"
```

Install: `pip install 'dark-hunter-sed[dust,plot]'` or `pip install dustmaps matplotlib`.

Note: dust prior conversions use `R_V=3.32`; Payne photometry likelihood uses `R_V=3.1`.

## Diagnostic plots

```bash
python -m darkhunter_sed.cli <gaia_id> --from-spec-root --plot --plot-epoch 0
```

Per-order blaze diagnostic (two panels; no UMS samples):

```bash
bash scripts/run_local.sh scripts/plot_order_blaze.py <gaia_id> --from-spec-root \
  --order 35 --blaze-only --regions-json output/masks/regions_Gaia_DR3_<ref_id>_epoch_1.json
```

Writes `output/plots/Gaia_DR3_<id>_ums.pdf` (and `_utp.pdf` if UTP ran) using the **same** spectrum data as the fit.

## Outputs (JSON)

`Gaia_DR3_<id>_sed_summary.json` includes:

- `m1_msun` — luminous primary mass from UMS `initial_Mass` (median, p16, p84)
- `fits.ums` / `fits.utp` — full parameter blocks
- `vrad_epochs` — per-epoch posterior RVs
- `av_prior` — dust-map prior used at fit time (map name, bounds, distance)

Phase 4 will push `m1_msun` into the RV website `tables/data.csv`.

## Model files

Not in git. Install under `STELLAR_ROOT/models/` or set `DARKHUNTER_SED_MODELS_DIR`:

- `specNN/modV0_spec_LinNet_R65K_WL515_530_wvt2.h5`
- `photNN/nnMIST_*.h5`
- `mistNN/mistyNN_2.3_v256_v0.h5`

## Blaze calibration

Uses `dark-hunter_rv` APF blaze JSON (default `calibration/blaze_orders_apf.json` in RV repo).
