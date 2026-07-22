# dark-hunter_sed

Spectrophotometric fitting of Gaia/APF stellar spectra and archival photometry using [uberMS](https://github.com/pacargile/uberMS) (MIST evolution + Payne neural nets).

## Dependencies

- **dark-hunter_rv** — spectrum I/O, blaze calibration, RV summary format, Gaia metadata cache
- **uberMS**, **ThePayne**, **MISTy** — editable installs under `STELLAR_ROOT` (default `~/stellar`)
- Neural-net weights under `DARKHUNTER_SED_MODELS_DIR` or `STELLAR_ROOT/models/`

```bash
cd dark-hunter_sed
pip install -e ../rvs/dark-hunter_rv
pip install -e .
pip install -e ~/stellar/uberMS ~/stellar/ThePayne ~/stellar/MISTy
```

## Workflow

1. **RV pipeline** (prerequisite): `dark-hunter_rv` produces `Gaia_DR3_<id>_summary.txt` with mask-CCF RVs and Gaia GSP-Phot priors.

2. **Photometry** — gathered automatically on first `cli` / `batch` run when `{gaia_id}_phot.fits` is missing (network query). To gather manually or refresh:
   ```bash
   python -m darkhunter_sed.photometry_gather <gaia_id>
   # default: output/photometry (override with -d)
   ```
   Disable auto-gather with `--no-auto-gather-phot` on `cli` or `batch`.

3. **Optional blaze regions** (picker on a reference star, then reuse for others):
   ```bash
   scripts/run_local.sh scripts/pick_spectrum_regions.py <reference_gaia_id> --from-spec-root
   # Fit + Save writes per-order blaze to output/masks/regions_Gaia_DR3_<id>_*.json
   ```
   `cli`, `batch`, `convert_spectra`, and `plot_order_blaze.py` auto-resolve the newest matching regions file per star, or pass `--regions-json` explicitly (shared blaze across stars).

4. **Fit** (UMS + UTP; UMS gives luminous `initial_Mass` → M1):
   ```bash
   export STELLAR_ROOT=~/stellar
   export DARKHUNTER_OUTPUT_DIR=../rvs/dark-hunter_rv/output
   export SPEC_ROOT=../rvs/data

   python -m darkhunter_sed.cli <gaia_id> --from-spec-root
   ```

   Per-epoch `vrad_i` uses **normal priors** from RV `[PIPELINE RESULTS]` (errors inflated 2× by default, floor 2 km/s). Gaia priors read from summary unless `--force-redownload`.

5. **Spectrum prep** uses dark-hunter_rv **sinc² blaze** (`sinc_blaze_only`: calibrated blaze + iterative S/N continuum mask, median-scaled; no modpoly pc in UMS), coalesced to 5150–5300 Å for uberMS. Stored per-order blaze from the picker overrides the calibrated shape when regions JSON is present. Set `DARKHUNTER_BLAZE_CALIBRATION` to the rebuilt `blaze_orders_apf.json` (see `scripts/run_local.sh`).

6. **Blaze diagnostic** (no UMS samples required):
   ```bash
   scripts/run_local.sh scripts/plot_order_blaze.py <gaia_id> --from-spec-root --order 35 \
     --regions-json output/masks/regions_Gaia_DR3_<ref_id>_epoch_1.json --blaze-only
   ```

## Environment

| Variable | Purpose |
|----------|---------|
| `STELLAR_ROOT` | uberMS / Payne / MISTy / models |
| `DARKHUNTER_SED_OUTPUT_DIR` | Default `output/` |
| `DARKHUNTER_SED_SAMPLES_DIR` | Posterior FITS (`output/samples/`) |
| `DARKHUNTER_OUTPUT_DIR` | RV summaries |
| `SPEC_ROOT` | APF reduced spectra tree |
| `DARKHUNTER_SED_MODELS_DIR` | NN weights override |
| `DARKHUNTER_BLAZE_CALIBRATION` | Per-order sinc² blaze JSON (default: dark-hunter_rv `calibration/blaze_orders_apf.json`) |
| `DARKHUNTER_SED_MASKS_DIR` | Picker regions JSON directory (default: `output/masks/`) |

## Outputs

| File | Content |
|------|---------|
| `output/samples/Gaia_DR3_<id>_ums.fits` | UMS posterior (primary; `initial_Mass` → M1) |
| `output/samples/Gaia_DR3_<id>_utp.fits` | UTP posterior (atmospheric cross-check) |
| `output/sed_summaries/Gaia_DR3_<id>_sed_summary.json` | Medians, credible intervals, `m1_msun` |

## Ziggy commands

Source once per shell (sets `PY`, `PYTHONPATH`, `STELLAR_ROOT`, `SPEC_ROOT`, `DARKHUNTER_OUTPUT_DIR`, `DATA_CSV`, and `cd` to the SED repo):

```bash
source /data2/darkhunter/dark-hunter_sed/scripts/activate_ziggy.sh
export GAIA_ID=<gaia_id>
```

Override before source if needed, e.g. `PY=/other/bin/python source …/activate_ziggy.sh`.
Set `DARKHUNTER_SED_MODELS_DIR` only when models are not under `$STELLAR_ROOT/{models,gaia/models}`.

Commands below assume activate is already sourced.

### Shared keywords

| Flag | Where | Effect |
|------|--------|--------|
| `-v` / `--verbose` | most entry points | DEBUG logs |
| `--no-progress` | `cli`, `batch` | no progress bar (cron/batch) |
| `--plot` | `cli`, `diagnose_continuum` | write/open diagnostic plot |
| `--plot-epoch N` | `cli` | epoch for `--plot` (0-based) |
| `--plot-simple` | `cli` | lightweight PDF instead of ppUMS-style |
| `--from-spec-root` | `cli` + scripts | discover epochs under `SPEC_ROOT` |
| `--spec-glob` / `-f` | `cli`, `plot_sed_posterior` | explicit spectrum glob |
| `--from-fits` | `cli`, `plot_sed_posterior` | input already FITS |
| `-D` / `--photometry-dir` | `cli` + scripts | phot FITS dir (default `output/photometry`) |
| `--ums-only` / `--utp-only` | `cli`, `batch` | skip the other fit |
| `--force-redownload` | `cli`, `batch` | Gaia TAP instead of RV summary cache |
| `--no-auto-gather-phot` | `cli`, `batch` | require existing `{id}_phot.fits` |
| `--regions-json PATH` | `cli`, `batch`, convert, blaze | shared/explicit blaze regions |
| `--no-auto-regions` | same | calibrated blaze only |
| `--vrad-prior {normal,uniform,fixed}` | `cli` | default `normal` |
| `--vrad-err-inflate` / `--vrad-err-floor` | `cli` | default 2 / 2 km/s |
| `--flexible-continuum` (+ pc*/jitter bounds) | `cli` | free UMS continuum (not recommended for APF) |
| `--no-dust-av-prior` | `cli`, `diagnose_ums_init` | legacy Av prior |
| `--phot-err-floor` / `--gaia-phot-err-floor` / `--phot-outlier-sigma` | `cli` | phot cleaning |
| `--force` / `--update` / `--limit N` / `--gaia-id` | `batch` | refit policy |
| `--no-show` / `--save` | `plot_order_blaze` | headless PDF |

### Use cases

**0. Prerequisite: RV summary exists**

```bash
export PYTHONPATH=$RV_REPO
$PY $RV_REPO/scripts/ensure_pipeline_summaries.py \
  --gaia-id $GAIA_ID --spec-root "$SPEC_ROOT" \
  --output-dir "$DARKHUNTER_OUTPUT_DIR" --force-gaia
```

**1. Manual photometry gather / refresh**

```bash
$PY -m darkhunter_sed.photometry_gather $GAIA_ID
# optional: --phot-outlier-sigma 3
```

**2. Single-star SED → samples + summary + push M1 (website `data.csv` + RV summary)**

`cli` calls `push_m1` after a successful UMS fit.

```bash
$PY -m darkhunter_sed.cli $GAIA_ID --from-spec-root
```

**3. Same + diagnostic PDFs**

```bash
$PY -m darkhunter_sed.cli $GAIA_ID --from-spec-root --plot
```

**4. Batch incremental (skip up-to-date)**

```bash
$PY -m darkhunter_sed.batch --update --no-progress
```

**5. Force refit one / few stars**

```bash
$PY -m darkhunter_sed.batch --gaia-id $GAIA_ID --force --no-progress
# repeat --gaia-id for more
```

**6. Force refit all stars**

```bash
$PY -m darkhunter_sed.batch --force --no-progress
```

**7. Push M1 only (recover website / summary without refit)**

```bash
$PY -m darkhunter_sed.push_m1 $GAIA_ID
# or all summaries:
$PY -m darkhunter_sed.push_m1 --all
```

**8. Convert epoch `.txt` → uberMS FITS**

```bash
$PY -m darkhunter_sed.convert_spectra \
  --source-id $GAIA_ID \
  -g "$SPEC_ROOT/**/Gaia_DR3_${GAIA_ID}_epoch_*.txt" \
  -o output/fits/$GAIA_ID
# then fit with: --from-fits --spec-glob 'output/fits/.../*.fits'
```

**9. Pick continuum/line regions + blaze (interactive)**

```bash
$PY scripts/pick_spectrum_regions.py $GAIA_ID --from-spec-root
# Fit → Save → output/masks/regions_Gaia_DR3_<id>_*.json
# resume: --input PATH [--output PATH]
```

**10. Order blaze diagnostic**

```bash
$PY scripts/plot_order_blaze.py $GAIA_ID --from-spec-root --order 35 \
  --blaze-only --no-show --save output/plots/order_blaze_${GAIA_ID}_o35.pdf
# shared blaze: --regions-json output/masks/regions_Gaia_DR3_<ref>_epoch_1.json
# all orders: --all-orders
# with UMS model panel: drop --blaze-only; optional --samples PATH --refit
```

**11. Regenerate SED posterior PDF from existing samples**

```bash
$PY scripts/plot_sed_posterior.py $GAIA_ID --from-spec-root \
  --ums-samples output/samples/Gaia_DR3_${GAIA_ID}_ums.fits \
  --utp-samples output/samples/Gaia_DR3_${GAIA_ID}_utp.fits \
  --sed-summary output/sed_summaries/Gaia_DR3_${GAIA_ID}_sed_summary.json
```

**12. Continuum diagnostics (JSON; optional plot)**

```bash
$PY scripts/diagnose_continuum.py $GAIA_ID --from-spec-root
# --plot  |  --use-posterior-pc  |  --samples PATH  |  --epoch N
```

**13. UMS init diagnose (no full SVI)**

```bash
$PY scripts/diagnose_ums_init.py $GAIA_ID --from-spec-root
# optional NN probe: --probe-svi
```

**14. Daily cron (batch `--update` + `push_m1 --all`)**

```bash
bash /data2/darkhunter/dark-hunter_sed/scripts/cron_update_sed.sh
# crontab: 30 10 * * * /bin/bash /data2/darkhunter/dark-hunter_sed/scripts/cron_update_sed.sh
```

**15. Monthly: new spectra → SED force + push_m1 + RV Keplerian refit**

```bash
bash /data2/darkhunter/dark-hunter_sed/scripts/cron_monthly_new_spectra.sh
# DAYS=7 bash ...   # optional window
```

**16. Screen: all-star SED force → push_m1 → parallel RV refit**

```bash
screen -dmS darkhunter_sed_m1_refit \
  bash /data2/darkhunter/dark-hunter_sed/scripts/screen_sed_then_refit_rvs.sh
# screen -r darkhunter_sed_m1_refit
```

**17. One-time dust map download (Av priors)**

```bash
$PY -c "import dustmaps.bayestar; dustmaps.bayestar.fetch()"
$PY -c "import dustmaps.decaps; dustmaps.decaps.fetch()"
$PY -c "import dustmaps.edenhofer2023; dustmaps.edenhofer2023.fetch()"
$PY -c "import dustmaps.chen2014; dustmaps.chen2014.fetch()"
$PY -c "import dustmaps.csfd; dustmaps.csfd.fetch()"
```

**18. Fit with shared picker blaze (batch)**

```bash
$PY -m darkhunter_sed.batch --update --no-progress \
  --regions-json output/masks/regions_Gaia_DR3_<REF>_epoch_1.json
```

**19. SB1 / bad RV priors**

```bash
$PY -m darkhunter_sed.cli $GAIA_ID --from-spec-root --vrad-prior uniform
# or: --vrad-err-inflate 5
```

More detail: [docs/operations.md](docs/operations.md).

## Batch / cron

```bash
python -m darkhunter_sed.batch --update --no-progress
bash scripts/cron_update_sed.sh   # after RV cron
```

## Local macOS

`dustmaps` is installed for **`/opt/local/bin/python`** (MacPorts), not necessarily for other interpreters. Use:

```bash
bash scripts/run_local.sh -m darkhunter_sed.cli <gaia_id> --from-spec-root --plot
```

See [docs/operations.md](docs/operations.md) for full env block.

## Tests

```bash
PYTHONPATH=/path/to/dark-hunter_rv:. pytest -q
```

## License

BSD-3-Clause (see LICENSE).
