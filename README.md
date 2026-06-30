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

2. **Photometry** (if needed):
   ```bash
   python -m darkhunter_sed.photometry_gather <gaia_id> -d output/photometry
   ```

3. **Fit** (UMS + UTP; UMS gives luminous `initial_Mass` → M1):
   ```bash
   export STELLAR_ROOT=~/stellar
   export DARKHUNTER_OUTPUT_DIR=../rvs/dark-hunter_rv/output
   export SPEC_ROOT=../rvs/data

   python -m darkhunter_sed.cli <gaia_id> --from-spec-root
   ```

   Per-epoch `vrad_i` uses **normal priors** from RV `[PIPELINE RESULTS]` (errors inflated 2× by default, floor 2 km/s). Gaia priors read from summary unless `--force-redownload`.

4. **Spectrum prep** uses dark-hunter_rv **sinc² blaze** + CR rejection (`calibration/blaze_orders_apf.json`), coalesced to 5150–5300 Å for uberMS.

## Environment

| Variable | Purpose |
|----------|---------|
| `STELLAR_ROOT` | uberMS / Payne / MISTy / models |
| `DARKHUNTER_SED_OUTPUT_DIR` | Default `output/` |
| `DARKHUNTER_SED_SAMPLES_DIR` | Posterior FITS (`output/samples/`) |
| `DARKHUNTER_OUTPUT_DIR` | RV summaries |
| `SPEC_ROOT` | APF reduced spectra tree |
| `DARKHUNTER_SED_MODELS_DIR` | NN weights override |

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

## Tests

```bash
PYTHONPATH=/path/to/dark-hunter_rv:. pytest -q
```

## License

BSD-3-Clause (see LICENSE).
