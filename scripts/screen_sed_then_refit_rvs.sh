#!/usr/bin/env bash
# Screen-friendly: force SED for all stars → push_m1 → parallel RV Keplerian refit.
#
# Ziggy example:
#   screen -dmS darkhunter_sed_m1_refit bash /data2/darkhunter/dark-hunter_sed/scripts/screen_sed_then_refit_rvs.sh
#
# Env: SED_REPO, RV_REPO, JOBS, NICE_LEVEL, PIPELINE_FORCE, FIT_FORCE, SKIP_SINGLE_EPOCH_FIT

set -euo pipefail

SED_REPO="${SED_REPO:-/data2/darkhunter/dark-hunter_sed}"
RV_REPO="${RV_REPO:-/data2/darkhunter/dark-hunter_rv}"
PY="${PY:-/home/marley/anaconda2/envs/gaia-env/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="${PY_FALLBACK:-python3}"
fi
WEB_ROOT="${WEB_ROOT:-/var/www/html/darkhunter/rv}"
DATA_CSV="${DATA_CSV:-$WEB_ROOT/tables/data.csv}"
SPEC_ROOT="${SPEC_ROOT:-/data2/gaia_stars/apf_reductions}"
STELLAR_ROOT="${STELLAR_ROOT:-/data2/stellar}"
JOBS="${JOBS:-4}"
NICE_LEVEL="${NICE_LEVEL:-10}"
PIPELINE_FORCE="${PIPELINE_FORCE:-0}"
PIPELINE_UPDATE="${PIPELINE_UPDATE:-1}"
FIT_FORCE="${FIT_FORCE:-1}"
FIT_JITTER="${FIT_JITTER:-1}"
SKIP_SINGLE_EPOCH_FIT="${SKIP_SINGLE_EPOCH_FIT:-1}"
RUN_SED="${RUN_SED:-0}"  # SED already done in this script; skip inside refit workers

export PY SED_REPO RV_REPO WEB_ROOT DATA_CSV SPEC_ROOT STELLAR_ROOT
export PYTHONPATH="$RV_REPO:$SED_REPO${PYTHONPATH:+:$PYTHONPATH}"
export DARKHUNTER_OUTPUT_DIR="${DARKHUNTER_OUTPUT_DIR:-$RV_REPO/output}"
export DARKHUNTER_SED_OUTPUT_DIR="${DARKHUNTER_SED_OUTPUT_DIR:-$SED_REPO/output}"
export DARKHUNTER_SED_SAMPLES_DIR="${DARKHUNTER_SED_SAMPLES_DIR:-$SED_REPO/output/samples}"
export DARKHUNTER_SED_SUMMARIES_DIR="${DARKHUNTER_SED_SUMMARIES_DIR:-$SED_REPO/output/sed_summaries}"
export DARKHUNTER_SED_PHOTOMETRY_DIR="${DARKHUNTER_SED_PHOTOMETRY_DIR:-$SED_REPO/output/photometry}"
export JOBS NICE_LEVEL PIPELINE_FORCE PIPELINE_UPDATE FIT_FORCE FIT_JITTER
export SKIP_SINGLE_EPOCH_FIT RUN_SED REPO="$RV_REPO" OUT="$DARKHUNTER_OUTPUT_DIR"

echo "=== $(date -Is) screen_sed_then_refit_rvs start ==="
cd "$SED_REPO"
echo "=== SED batch --force ==="
"$PY" -m darkhunter_sed.batch --force --no-progress \
  || echo "[WARN] SED batch had errors"

echo "=== push_m1 --all ==="
"$PY" -m darkhunter_sed.push_m1 --all \
  --rv-output-dir "$DARKHUNTER_OUTPUT_DIR" \
  --data-csv "$DATA_CSV" \
  || echo "[WARN] push_m1 had errors"

echo "=== RV parallel refit (SKIP_SINGLE_EPOCH_FIT=$SKIP_SINGLE_EPOCH_FIT) ==="
cd "$RV_REPO"
bash scripts/refit_all_per_object_parallel.sh

echo "=== $(date -Is) screen_sed_then_refit_rvs done ==="
