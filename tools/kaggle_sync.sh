#!/usr/bin/env bash
# Copy the code needed on Kaggle into the staging folder ~/kaggle_up/p4aids-code/p4aids/
# and (optionally) upload it as the private dataset bathanhphan/p4aids-code.
#
#   tools/kaggle_sync.sh                    # only refresh the staging folder
#   tools/kaggle_sync.sh create             # first upload (private dataset)
#   tools/kaggle_sync.sh version "message"  # upload a new version
#
# Safety: writes ONLY inside ~/kaggle_up/p4aids-code (other folders in ~/kaggle_up are other datasets).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$HOME/kaggle_up/p4aids-code"
DEST="$STAGE/p4aids"
[[ "$(basename "$STAGE")" == "p4aids-code" ]] || { echo "refusing: unexpected staging path $STAGE"; exit 1; }

mkdir -p "$DEST"
if [[ ! -f "$STAGE/dataset-metadata.json" ]]; then
  cat > "$STAGE/dataset-metadata.json" <<'JSON'
{
  "title": "p4aids-code",
  "id": "bathanhphan/p4aids-code",
  "licenses": [{"name": "other"}],
  "description": "Private. Code for the P4AIDS assignment (day-ahead load forecasting, Spain): src/, scripts/, holiday table and frozen hyper-parameters. Attach together with nicholasjhana/energy-consumption-generation-prices-and-weather."
}
JSON
fi

# --delete is limited to $DEST, so stale files of THIS dataset are removed and nothing else is touched.
rsync -a --delete --exclude "__pycache__" --exclude ".DS_Store" \
  --include "/src/***" --include "/scripts/***" \
  --include "/data/" --include "/data/external/***" \
  --include "/results/" --include "/results/frozen/***" \
  --include "/results/val_metrics.csv" --include "/results/test_run_log.json" \
  --include "/results/frozen_v2/***" --include "/results/val_predictions.parquet" \
  --include "/results/test_predictions.parquet" --include "/results/test_ablation_run_log.json" \
  --include "/requirements.txt" --include "/README.md" \
  --exclude "*" \
  "$REPO/" "$DEST/"
echo "staged:"; (cd "$DEST" && find . -type f | sort)

case "${1:-}" in
  create)  kaggle datasets create -p "$STAGE" --dir-mode zip ;;
  version) kaggle datasets version -p "$STAGE" --dir-mode zip -m "${2:?version message required}" ;;
  "") ;;
  *) echo "unknown command: $1"; exit 1 ;;
esac
