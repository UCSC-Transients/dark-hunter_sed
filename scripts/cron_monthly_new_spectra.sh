#!/usr/bin/env bash
# Monthly: SED + push_m1 for stars with new spectra in the last 30 days;
# Keplerian RV refit only when ≥2 epochs.
#
# Crontab (ziggy, after daily RV/SED crons):
#   0 11 1 * * /bin/bash /data2/darkhunter/dark-hunter_sed/scripts/cron_monthly_new_spectra.sh
#
# Env: REPO, RV_REPO, SPEC_ROOT, STELLAR_ROOT, PY, LOG, DAYS, WEB_ROOT, DATA_CSV

set -euo pipefail

REPO="${REPO:-/data2/darkhunter/dark-hunter_sed}"
RV_REPO="${RV_REPO:-/data2/darkhunter/dark-hunter_rv}"
SPEC_ROOT="${SPEC_ROOT:-/data2/gaia_stars/apf_reductions}"
STELLAR_ROOT="${STELLAR_ROOT:-/data2/stellar}"
WEB_ROOT="${WEB_ROOT:-/var/www/html/darkhunter/rv}"
DATA_CSV="${DATA_CSV:-$WEB_ROOT/tables/data.csv}"
PY="${PY:-/home/marley/anaconda2/envs/gaia-env/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="${PY_FALLBACK:-python3}"
fi
LOG="${LOG:-$REPO/logs/cron_monthly_sed_rv.log}"
DAYS="${DAYS:-30}"

mkdir -p "$(dirname "$LOG")" "$REPO/output/sed_summaries" "$REPO/output/samples"
cd "$REPO"

export PYTHONPATH="$RV_REPO:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export STELLAR_ROOT SPEC_ROOT DATA_CSV
export DARKHUNTER_OUTPUT_DIR="${DARKHUNTER_OUTPUT_DIR:-$RV_REPO/output}"
export DARKHUNTER_SED_OUTPUT_DIR="${DARKHUNTER_SED_OUTPUT_DIR:-$REPO/output}"
export DARKHUNTER_SED_SAMPLES_DIR="${DARKHUNTER_SED_SAMPLES_DIR:-$REPO/output/samples}"
export DARKHUNTER_SED_SUMMARIES_DIR="${DARKHUNTER_SED_SUMMARIES_DIR:-$REPO/output/sed_summaries}"
export DARKHUNTER_SED_PHOTOMETRY_DIR="${DARKHUNTER_SED_PHOTOMETRY_DIR:-$REPO/output/photometry}"
export REPO RV_REPO OUT="${OUT:-$RV_REPO/output}" WEB_ROOT
export SKIP_SINGLE_EPOCH_FIT="${SKIP_SINGLE_EPOCH_FIT:-1}"
export FIT_FORCE="${FIT_FORCE:-1}"
export FIT_JITTER="${FIT_JITTER:-1}"
export PIPELINE_UPDATE="${PIPELINE_UPDATE:-1}"
export PIPELINE_FORCE="${PIPELINE_FORCE:-0}"
# SED already run above; workers only pipeline/fit/website
export RUN_SED="${RUN_SED:-0}"
export SED_REPO="$REPO"

exec >>"$LOG" 2>&1
echo "=== $(date -Is) cron_monthly_new_spectra start (pid $$) DAYS=$DAYS ==="

if [[ ! -d "$SPEC_ROOT" ]]; then
  echo "[WARN] SPEC_ROOT missing: $SPEC_ROOT"
  exit 0
fi

# Gaia ids with any primary epoch file mtime within DAYS.
mapfile -t GIDS < <(
  find "$SPEC_ROOT" -type f \( \
      \( -name 'Gaia_DR3_*_epoch_*.txt' ! -name '*_order_*' \) -o \
      -name 'Gaia_DR3_*_ap1.flm' -o \
      -name 'Gaia_DR3_*_ap1.txt' \
    \) -mtime "-$DAYS" -print 2>/dev/null \
    | sed -n 's/.*Gaia_DR3_\([0-9][0-9]*\).*/\1/p' \
    | sort -u
)

if [[ "${#GIDS[@]}" -eq 0 ]]; then
  echo "[INFO] no spectra newer than ${DAYS}d; exit"
  echo "=== $(date -Is) cron_monthly_new_spectra done ==="
  exit 0
fi

echo "=== SED batch for ${#GIDS[@]} star(s) with new spectra ==="
gaia_args=()
for gid in "${GIDS[@]}"; do
  gaia_args+=(--gaia-id "$gid")
done
"$PY" -m darkhunter_sed.batch --force --no-progress "${gaia_args[@]}" \
  || echo "[WARN] SED batch had errors (see log)"

echo "=== push_m1 --all (touched stars may already be pushed by batch) ==="
"$PY" -m darkhunter_sed.push_m1 --all \
  --rv-output-dir "$DARKHUNTER_OUTPUT_DIR" \
  --data-csv "$DATA_CSV" \
  || echo "[WARN] push_m1 had errors"

echo "=== RV refit_one_object (Keplerian skipped when n_spec<2) ==="
for gid in "${GIDS[@]}"; do
  echo "--- refit Gaia_DR3_${gid} ---"
  bash "$RV_REPO/scripts/lib/refit_one_object.sh" "$gid" \
    || echo "[WARN] refit_one_object failed for $gid"
done

echo "=== $(date -Is) cron_monthly_new_spectra done ==="
