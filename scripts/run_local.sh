#!/usr/bin/env bash
# Local macOS harness: MacPorts Python has dustmaps; anaconda may not.
set -euo pipefail

SED_REPO="$(cd "$(dirname "$0")/.." && pwd)"
RV_REPO="${RV_REPO:-/Users/rfoley/darkhunter/rvs/dark-hunter_rv}"
STELLAR_ROOT="${STELLAR_ROOT:-/Users/rfoley/stellar}"

export PYTHONPATH="${RV_REPO}:${SED_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export STELLAR_ROOT
export DARKHUNTER_SED_MODELS_DIR="${DARKHUNTER_SED_MODELS_DIR:-${STELLAR_ROOT}/gaia/models}"
export SPEC_ROOT="${SPEC_ROOT:-/Users/rfoley/darkhunter/rvs/data}"
export DARKHUNTER_OUTPUT_DIR="${DARKHUNTER_OUTPUT_DIR:-/Users/rfoley/darkhunter/rvs/output}"
export DARKHUNTER_SED_OUTPUT_DIR="${DARKHUNTER_SED_OUTPUT_DIR:-${SED_REPO}/output}"
export DARKHUNTER_SED_PHOTOMETRY_DIR="${DARKHUNTER_SED_PHOTOMETRY_DIR:-${SED_REPO}/output/photometry}"
export DARKHUNTER_BLAZE_CALIBRATION="${DARKHUNTER_BLAZE_CALIBRATION:-${RV_REPO}/calibration/blaze_orders_apf.json}"

PY="${PY:-/opt/local/bin/python}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python not found: $PY (set PY=...)" >&2
  exit 1
fi

# conda/base may set a stale CPU-only flag; Metal XLA aborts on it. fit.py re-sanitizes after uberMS import.
unset XLA_FLAGS 2>/dev/null || true

# uberMS SVI expects CPU JAX; MacPorts jax-metal otherwise initializes Metal (experimental).
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

# Preflight: warn if JAX backend is not CPU (uberMS SVI is not validated on Metal).
if "$PY" -c "import jax; import sys; sys.exit(0 if jax.default_backend()=='cpu' else 1)" 2>/dev/null; then
  :
else
  echo "WARNING: JAX default_backend is not 'cpu' (set JAX_PLATFORMS=cpu or use conda python3)." >&2
  echo "  PY=$PY" >&2
  "$PY" -c "import jax; print('  jax', jax.__version__, 'backend=', jax.default_backend())" 2>/dev/null || true
fi

exec "$PY" "$@"
