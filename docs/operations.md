# dark-hunter_sed operations

Batch spectrophotometric fitting with uberMS. **Run the RV pipeline first** so
`Gaia_DR3_<id>_summary.txt` files exist (mask-CCF RVs + Gaia GSP-Phot priors).

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

```bash
cd $REPO
python -m darkhunter_sed.photometry_gather <gaia_id> -d output/photometry
```

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

- Refit when any epoch spectrum, photometry FITS, or RV summary is newer than
  `sed_summary.json` / sample FITS.
- Skip stars with fewer than 2 epoch spectra (UMS requirement).
- Skip stars without `{gaia_id}_phot.fits` in `DARKHUNTER_SED_PHOTOMETRY_DIR`.

## Cron

After RV cron (`dark-hunter_rv/scripts/cron_update_rv_website.sh`):

```bash
bash scripts/cron_update_sed.sh
```

Log: `$REPO/logs/cron_sed.log`

## Priors

- **RV:** per-epoch `vrad_i` normal priors from `[PIPELINE RESULTS]` (2× error inflation, 2 km/s floor by default).
- **Gaia:** from summary `[GAIA METADATA]`; `--force-redownload` on CLI/batch re-queries TAP.

## Outputs (JSON)

`Gaia_DR3_<id>_sed_summary.json` includes:

- `m1_msun` — luminous primary mass from UMS `initial_Mass` (median, p16, p84)
- `fits.ums` / `fits.utp` — full parameter blocks
- `vrad_epochs` — per-epoch posterior RVs

Phase 4 will push `m1_msun` into the RV website `tables/data.csv`.

## Model files

Not in git. Install under `STELLAR_ROOT/models/` or set `DARKHUNTER_SED_MODELS_DIR`:

- `specNN/modV0_spec_LinNet_R65K_WL515_530_wvt2.h5`
- `photNN/nnMIST_*.h5`
- `mistNN/mistyNN_2.3_v256_v0.h5`

## Blaze calibration

Uses `dark-hunter_rv` APF blaze JSON (default `calibration/blaze_orders_apf.json` in RV repo).
