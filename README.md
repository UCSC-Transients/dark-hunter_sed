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
   python -m darkhunter_sed.photometry_gather <gaia_id> -d output/photometry
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

4. **Spectrum prep** uses dark-hunter_rv **sinc² blaze** (`sinc_blaze_only`: calibrated blaze + iterative S/N continuum mask, median-scaled; no modpoly pc in UMS), coalesced to 5150–5300 Å for uberMS. Stored per-order blaze from the picker overrides the calibrated shape when regions JSON is present. Set `DARKHUNTER_BLAZE_CALIBRATION` to the rebuilt `blaze_orders_apf.json` (see `scripts/run_local.sh`).

5. **Blaze diagnostic** (no UMS samples required):
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

## Batch / cron

```bash
python -m darkhunter_sed.batch --update --no-progress
bash scripts/cron_update_sed.sh   # after RV cron
```

See [docs/operations.md](docs/operations.md).

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
