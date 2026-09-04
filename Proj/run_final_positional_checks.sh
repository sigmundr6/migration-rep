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

# macOS ships Bash 3.2, where expanding an empty array under `set -u` raises
# "unbound variable". Add --resume through a function instead of an optional
# array so the runner remains compatible with the system shell.
run_training_script() {
  if [[ "${RESUME:-0}" == "1" ]]; then
    python "$@" --resume
  else
    python "$@"
  fi
}

# No retraining: seed-split reliability and individual-versus-ensemble summary.
python analyze_seed_reliability_and_ensembles.py

# Ten-seed central controls: midpoint, unordered, shuffled-s, directional
# positional, and event-level reversal-invariant positional representations.
run_training_script run_positional_controls.py \
  --data-root "$data_root" \
  --device "$device" \
  --epochs 100 \
  --early-stopping-patience 20

python analyze_positional_controls.py

# The 3x3 sensitivity grid is deliberately optional because it adds 54 fits.
# Enable it with RUN_SENSITIVITY=1 after the two central controls complete.
if [[ "${RUN_SENSITIVITY:-0}" == "1" ]]; then
  run_training_script run_pathway_hyperparameter_sensitivity.py \
    --data-root "$data_root" \
    --device "$device" \
    --epochs 100 \
    --early-stopping-patience 20
fi
