#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

device="${DEVICE:-cpu}"
data_root="${DATA_ROOT:-$script_dir/../data}"
if [[ ! -f "$data_root/nebDFT2k/nebDFT2k_index.csv" ]]; then
  echo "LiTraj dataset not found at: $data_root/nebDFT2k/nebDFT2k_index.csv" >&2
  echo "Set DATA_ROOT to the directory containing the nebDFT2k folder." >&2
  exit 2
fi

args=(
  run_pathway_hyperparameter_sensitivity.py
  --data-root "$data_root"
  --device "$device"
  --seeds 7 17 29 43 71 89 101 131 167 197
  --cutoffs 2.5 3.0 3.5
  --sigmas 1.2 1.5 1.8
  --epochs 100
  --early-stopping-patience 20
  --checkpoint-dir checkpoints/pathway_sensitivity
  --output results/pathway_sensitivity_10seed/metrics.csv
)

# Reuse the completed seeds 7, 17, and 29 where possible and resume any
# interrupted new fits. Set RESUME=0 only to deliberately start fresh.
if [[ "${RESUME:-1}" == "1" ]]; then
  python "${args[@]}" --resume
else
  python "${args[@]}"
fi

python plot_pathway_sensitivity.py \
  --metrics results/pathway_sensitivity_10seed/metrics.csv \
  --output ../results/dissertation_figures/pathway_incidence_sensitivity_10seed.pdf

echo "Ten-seed sensitivity experiment complete."
