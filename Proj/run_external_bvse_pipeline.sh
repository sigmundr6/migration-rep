#!/usr/bin/env bash
# Generate an independent BVSE set, audit it, then run the frozen four-model test.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
EXTERNAL_DIR="${EXTERNAL_DIR:-${DATA_ROOT}/bvse_external}"
MANIFEST="${MANIFEST:-${EXTERNAL_DIR}/manifest.csv}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results/external_bvse_generalization}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRIPT_DIR}/checkpoints/external_bvse_generalization}"
DEVICE="${DEVICE:-cpu}"
N_IMAGES="${N_IMAGES:-5}"
FMAX="${FMAX:-0.1}"
GENERATOR_STEPS="${GENERATOR_STEPS:-100}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"

pilot=0
audit_only=0
check_only=0
skip_generation=0
evaluation_scope="independent"
inputs=()

usage() {
  printf '%s\n' \
    "Usage: $(basename "$0") [options] STRUCTURE_FILE [STRUCTURE_FILE ...]" \
    "" \
    "Options:" \
    "  --pilot       Generate at most 5 structures x 3 hops, audit, then stop." \
    "  --audit-only  Generate/resume and audit, but do not train." \
    "  --existing-manifest FILE  Skip generation and consume an existing manifest." \
    "  --regenerated-test FILE  Use only official-test rows from an existing regenerated manifest." \
    "  --check       Check dependencies and scripts without requiring inputs." \
    "  --device DEV  cpu, mps, or auto (default: ${DEVICE})." \
    "  --help        Show this message." \
    "" \
    "Useful environment overrides:" \
    "  PYTHON_BIN, DATA_ROOT, EXTERNAL_DIR, MANIFEST, RESULT_DIR," \
    "  CHECKPOINT_DIR, DEVICE, N_IMAGES, FMAX, GENERATOR_STEPS, EPOCHS, PATIENCE" \
    "" \
    "Example:" \
    "  $(basename "$0") /independent/source/*.cif"
}

while (($#)); do
  case "$1" in
    --pilot)
      pilot=1
      shift
      ;;
    --audit-only)
      audit_only=1
      shift
      ;;
    --check)
      check_only=1
      shift
      ;;
    --existing-manifest)
      if (($# < 2)); then
        printf 'error: --existing-manifest requires a value\n' >&2
        exit 2
      fi
      MANIFEST="$2"
      skip_generation=1
      shift 2
      ;;
    --regenerated-test)
      if (($# < 2)); then
        printf 'error: --regenerated-test requires a manifest path\n' >&2
        exit 2
      fi
      MANIFEST="$2"
      skip_generation=1
      evaluation_scope="regenerated-test"
      shift 2
      ;;
    --device)
      if (($# < 2)); then
        printf 'error: --device requires a value\n' >&2
        exit 2
      fi
      DEVICE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      inputs+=("$@")
      break
      ;;
    -* )
      printf 'error: unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
    *)
      inputs+=("$1")
      shift
      ;;
  esac
done

case "${DEVICE}" in
  cpu|mps|auto) ;;
  *)
    printf 'error: DEVICE must be cpu, mps, or auto; got %s\n' "${DEVICE}" >&2
    exit 2
    ;;
esac

printf 'Checking Python dependencies...\n'
"${PYTHON_BIN}" -c 'import ase, ions, litraj, numpy, sklearn, torch'
"${PYTHON_BIN}" -m py_compile \
  "${SCRIPT_DIR}/generate_bvse_trajectories.py" \
  "${SCRIPT_DIR}/run_external_bvse_generalization.py"

if ((check_only)); then
  printf 'Pipeline check passed.\n'
  exit 0
fi

if ((!skip_generation && ${#inputs[@]} == 0)); then
  printf 'error: supply independently sourced ASE-readable structure files\n' >&2
  usage >&2
  exit 2
fi

if ((!skip_generation)); then
  for input in "${inputs[@]}"; do
    if [[ ! -f "${input}" ]]; then
      printf 'error: input is not a readable file: %s\n' "${input}" >&2
      exit 2
    fi
  done
fi

mkdir -p "${EXTERNAL_DIR}" "${RESULT_DIR}" "${CHECKPOINT_DIR}"

if ((skip_generation)); then
  printf 'Stage 1/3: using existing manifest: %s\n' "${MANIFEST}"
else
  generator_args=(
    "${SCRIPT_DIR}/generate_bvse_trajectories.py"
    "${inputs[@]}"
    --output-dir "${EXTERNAL_DIR}"
    --manifest "${MANIFEST}"
    --n-images "${N_IMAGES}"
    --fmax "${FMAX}"
    --steps "${GENERATOR_STEPS}"
    --skip-errors
  )
  if ((pilot)); then
    generator_args+=(--max-structures 5 --max-edges-per-structure 3)
  fi
  printf 'Stage 1/3: generating or resuming BVSE trajectories...\n'
  "${PYTHON_BIN}" "${generator_args[@]}"
fi

if [[ ! -s "${MANIFEST}" ]]; then
  printf 'error: generator did not produce a non-empty manifest: %s\n' "${MANIFEST}" >&2
  exit 1
fi

minimum_hops=50
if ((pilot)); then
  minimum_hops=1
fi

evaluation_args=(
  "${SCRIPT_DIR}/run_external_bvse_generalization.py"
  --data-root "${DATA_ROOT}"
  --external-manifest "${MANIFEST}"
  --evaluation-scope "${evaluation_scope}"
  --external-cache-dir "${SCRIPT_DIR}/cache/external_bvse_preprocessed"
  --output-dir "${RESULT_DIR}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --early-stopping-patience "${PATIENCE}"
  --min-external-hops "${minimum_hops}"
)

printf 'Stage 2/3: auditing convergence, provenance, and overlap...\n'
"${PYTHON_BIN}" "${evaluation_args[@]}" --audit-only

if ((pilot || audit_only)); then
  printf 'Stopped after the audit as requested. Review: %s\n' \
    "${RESULT_DIR}/dataset_audit.json"
  exit 0
fi

printf 'Stage 3/3: training on LiTraj BVSE and evaluating the frozen external set...\n'
"${PYTHON_BIN}" "${evaluation_args[@]}" --resume

printf 'Pipeline complete. Primary outputs:\n'
printf '  %s\n' \
  "${RESULT_DIR}/dataset_audit.json" \
  "${RESULT_DIR}/ensemble_metrics.csv" \
  "${RESULT_DIR}/paired_bootstrap.csv"
