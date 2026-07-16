#!/usr/bin/env bash
# Cron-friendly: incremental SED fits for stars with APF spectra + RV summaries.
#
# Run after dark-hunter_rv cron (RV summaries must exist for best priors).
# Daily crontab example (ziggy, after RV cron):
#
#   30 10 * * * /bin/bash /data2/darkhunter/dark-hunter_sed/scripts/cron_update_sed.sh
#
# Env: REPO, RV_REPO, SPEC_ROOT, STELLAR_ROOT, PY, LOG

set -euo pipefail

REPO="${REPO:-/data2/darkhunter/dark-hunter_sed}"
RV_REPO="${RV_REPO:-/data2/darkhunter/dark-hunter_rv}"
SPEC_ROOT="${SPEC_ROOT:-/data2/gaia_stars/apf_reductions}"
STELLAR_ROOT="${STELLAR_ROOT:-/data2/stellar}"
PY="${PY:-/home/marley/anaconda2/envs/gaia-env/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="${PY_FALLBACK:-python3}"
fi
LOG="${LOG:-$REPO/logs/cron_sed.log}"

mkdir -p "$(dirname "$LOG")" "$REPO/output/sed_summaries" "$REPO/output/samples"
cd "$REPO"

export PYTHONPATH="$RV_REPO:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export STELLAR_ROOT
export SPEC_ROOT
export DARKHUNTER_OUTPUT_DIR="${DARKHUNTER_OUTPUT_DIR:-$RV_REPO/output}"
export DARKHUNTER_SED_OUTPUT_DIR="${DARKHUNTER_SED_OUTPUT_DIR:-$REPO/output}"
export DARKHUNTER_SED_SAMPLES_DIR="${DARKHUNTER_SED_SAMPLES_DIR:-$REPO/output/samples}"
export DARKHUNTER_SED_SUMMARIES_DIR="${DARKHUNTER_SED_SUMMARIES_DIR:-$REPO/output/sed_summaries}"
export DARKHUNTER_SED_PHOTOMETRY_DIR="${DARKHUNTER_SED_PHOTOMETRY_DIR:-$REPO/output/photometry}"

exec >>"$LOG" 2>&1
echo "=== $(date -Is) cron_update_sed start (pid $$) ==="

if [[ ! -d "$SPEC_ROOT" ]]; then
  echo "[WARN] SPEC_ROOT missing: $SPEC_ROOT"
  exit 0
fi

echo "=== SED batch --update ==="
"$PY" -m darkhunter_sed.batch --update --no-progress \
  || echo "[WARN] SED batch had errors (see log)"

echo "=== push_m1 --all ==="
DATA_CSV="${DATA_CSV:-/var/www/html/darkhunter/rv/tables/data.csv}"
"$PY" -m darkhunter_sed.push_m1 --all \
  --rv-output-dir "$DARKHUNTER_OUTPUT_DIR" \
  --data-csv "$DATA_CSV" \
  || echo "[WARN] push_m1 had errors (see log)"

echo "=== $(date -Is) cron_update_sed done ==="
