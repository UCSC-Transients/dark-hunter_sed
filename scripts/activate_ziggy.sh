# Ziggy interactive env for dark-hunter_sed.
# Usage:  source /data2/darkhunter/dark-hunter_sed/scripts/activate_ziggy.sh
#
# After sourcing, run e.g.:
#   $PY -m darkhunter_sed.cli $GAIA_ID --from-spec-root
#
# Override any var before source, e.g. PY=/other/python source .../activate_ziggy.sh

# Do not use set -e / set -u: this file is sourced into the interactive shell.

_SED_ACTIVATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SED_REPO="${SED_REPO:-$(cd "${_SED_ACTIVATE_DIR}/.." && pwd)}"
RV_REPO="${RV_REPO:-/data2/darkhunter/dark-hunter_rv}"
PY="${PY:-/data2/darkhunter/.venv/bin/python}"
STELLAR_ROOT="${STELLAR_ROOT:-/data2/darkhunter/stellar}"
SPEC_ROOT="${SPEC_ROOT:-/data2/gaia_stars/apf_reductions}"
DARKHUNTER_OUTPUT_DIR="${DARKHUNTER_OUTPUT_DIR:-${RV_REPO}/output}"
DATA_CSV="${DATA_CSV:-/var/www/html/darkhunter/rv/tables/data.csv}"

# Optional SED output overrides (defaults in config.py already use $SED_REPO/output
# when the package lives in this tree — leave unset unless relocating outputs).
# DARKHUNTER_SED_OUTPUT_DIR / SAMPLES / PLOTS / PHOTOMETRY / MASKS / SUMMARIES

# Models: set only if not under $STELLAR_ROOT/models or $STELLAR_ROOT/gaia/models
if [[ -z "${DARKHUNTER_SED_MODELS_DIR:-}" ]]; then
  if [[ -d "${STELLAR_ROOT}/models" ]]; then
    : # resolve_models_dir finds STELLAR_ROOT/models
  elif [[ -d "${STELLAR_ROOT}/gaia/models" ]]; then
    : # resolve_models_dir finds STELLAR_ROOT/gaia/models
  else
    echo "WARNING: no models dir under ${STELLAR_ROOT}/{models,gaia/models}; set DARKHUNTER_SED_MODELS_DIR" >&2
  fi
fi

export SED_REPO RV_REPO PY STELLAR_ROOT SPEC_ROOT DARKHUNTER_OUTPUT_DIR DATA_CSV
export PYTHONPATH="${SED_REPO}:${RV_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export REPO="${REPO:-$SED_REPO}"

if [[ -n "${DARKHUNTER_SED_MODELS_DIR:-}" ]]; then
  export DARKHUNTER_SED_MODELS_DIR
fi

if [[ ! -x "$PY" ]]; then
  echo "WARNING: PY not executable: $PY (set PY=... before source)" >&2
fi

cd "$SED_REPO" || echo "WARNING: cannot cd to SED_REPO=$SED_REPO" >&2

echo "dark-hunter_sed ziggy env active:"
echo "  SED_REPO=$SED_REPO"
echo "  RV_REPO=$RV_REPO"
echo "  PY=$PY"
echo "  STELLAR_ROOT=$STELLAR_ROOT"
echo "  SPEC_ROOT=$SPEC_ROOT"
echo "  DARKHUNTER_OUTPUT_DIR=$DARKHUNTER_OUTPUT_DIR"
echo "  DATA_CSV=$DATA_CSV"
echo "  PYTHONPATH=$PYTHONPATH"
unset _SED_ACTIVATE_DIR
