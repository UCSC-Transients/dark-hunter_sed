#!/usr/bin/env bash
# Run after: gh auth login -h github.com
# Creates issues #1-#3, pushes feat/sed-pipeline-v0.1, opens PR.
set -euo pipefail

REPO="UCSC-Transients/dark-hunter_sed"
BRANCH="feat/sed-pipeline-v0.1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

gh auth status -h github.com

ISSUE1="$(gh issue create --repo "$REPO" \
  --title "SED pipeline: dust Av priors, UMS/UTP fitting, and Gaia-informed init" \
  --body "$(cat <<'EOF'
## Problem
uberMS SED fitting needs dust extinction priors, aligned UMS/UTP SVI configuration, and sensible SVI starting points from Gaia metadata.

## Solution
- Add dustmaps-based Av prior chain with provenance in sed_summary
- Wire UMS/UTP SVI with relaxed continuum/LSF priors and fixed pc* defaults for rigid UMS
- Gaia FLAME mass/age for UMS init; photometry error floors; continuum diagnostics
- ppUMS-style posterior plotting and local macOS run harness
EOF
)")"

ISSUE2="$(gh issue create --repo "$REPO" \
  --title "Interactive blaze region picker and shared per-order sinc² normalization" \
  --body "$(cat <<'EOF'
## Problem
Per-order blaze continuum fitting needs manual region control and a reusable sinc² shape across epochs and stars.

## Solution
- Interactive `pick_spectrum_regions` with continuum/line regions and per-order Fit/Save
- Shared sinc² blaze stored in regions JSON; manual regions supplement auto line masks
- `plot_order_blaze` diagnostics and spectrum ingest use stored blaze
EOF
)")"

ISSUE3="$(gh issue create --repo "$REPO" \
  --title "uberMS workflow: regions auto-resolve, photometry auto-gather, blaze-only diagnostics" \
  --body "$(cat <<'EOF'
## Problem
Running uberMS on new stars requires manual photometry gather and regions JSON wiring through CLI, batch, and convert.

## Solution
- `resolve_regions_json_for_star` and `ensure_photometry_fits` in getdata/cli/batch/convert
- Batch `--update` refits when phot missing (auto-gather) or regions JSON is newer
- `--blaze-only` plot mode without UMS samples; CI pytest with dark-hunter_rv checkout
EOF
)")"

git push -u origin "$BRANCH"

gh pr create --repo "$REPO" --head "$BRANCH" \
  --title "SED pipeline: dust priors, blaze picker, and uberMS workflow" \
  --body "$(cat <<EOF
## Summary
- Resolves ${ISSUE1##*/} — dust Av prior chain, UMS/UTP SVI wiring, Gaia FLAME-informed init, continuum/posterior diagnostics.
- Resolves ${ISSUE2##*/} — interactive \`pick_spectrum_regions\` with per-order shared sinc² blaze stored in regions JSON.
- Resolves ${ISSUE3##*/} — auto-resolve regions JSON, auto-gather photometry, \`--blaze-only\` plot, batch/cli/convert plumbing.

## Test plan
- [x] \`PYTHONPATH=\$RV_REPO:. pytest -q\` (80 passed locally)
- [ ] CI pytest workflow green on PR
- [ ] Manual: \`bash scripts/run_local.sh -m darkhunter_sed.cli <gaia_id> --from-spec-root --plot\`
- [ ] Manual: \`bash scripts/run_local.sh scripts/plot_order_blaze.py <gaia_id> --from-spec-root --blaze-only --order 35\`
EOF
)"

echo "Done. Issues: $ISSUE1 $ISSUE2 $ISSUE3"
