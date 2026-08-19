#!/usr/bin/env python3
"""Compare DFT and BVSE paths for DFT-barrier prediction on nebDFT2k."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


METHODS = ("crystal", "midpoint", "trajectory", "positional")
RESULT = re.compile(
    r"^(?P<method>\S+)\s+(?P<params>\d+)\s+"
    r"(?P<train>[0-9.]+) eV\s+(?P<val>[0-9.]+) eV\s+(?P<test>[0-9.]+) eV$"
)
FIELDS = (
    "experiment", "trajectory_source", "target_source", "method", "params",
    "train", "val", "test",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-root", type=Path, default=Path("data"))
    result.add_argument("--comparison-csv", type=Path,
                        default=Path("results/nebdft2k_bvse_dft.csv"))
    result.add_argument("--output", type=Path,
                        default=Path("results/nebdft2k_cross_fidelity.csv"))
    result.add_argument("--epochs", type=int, default=25)
    result.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    result.add_argument("--hidden-dim", type=int, default=32)
    result.add_argument("--layers", type=int, default=2)
    result.add_argument("--learning-rate", type=float, default=2e-3)
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--checkpoint-every", type=int, default=5)
    result.add_argument("--checkpoint-dir", type=Path,
                        default=Path("checkpoints/nebdft2k_cross_fidelity"))
    result.add_argument("--resume", action="store_true")
    result.add_argument("--mps-memory-report", action="store_true")
    return result


def prior_dft_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"comparison CSV not found: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        if (row.get("trajectory_source") == "relaxed"
                and row.get("target_source") == "em_dft"
                and row.get("method") in METHODS):
            selected.append({
                "experiment": "dft_path_to_dft_barrier",
                "trajectory_source": "relaxed", "target_source": "em_dft",
                "method": row["method"], "params": row["params"],
                "train": row["train"], "val": row["val"], "test": row["test"],
            })
    if len(selected) != len(METHODS):
        found = [row["method"] for row in selected]
        raise SystemExit(f"comparison CSV lacks the four DFT reference rows; found {found}")
    return selected


def scalar_baselines(data_root: Path) -> list[dict[str, str]]:
    try:
        from litraj.data import load_data
    except ImportError as exc:
        raise SystemExit("scalar baseline requires the installed litraj package") from exc
    table = load_data("nebDFT2k", str(data_root))
    required = {"em_bvse", "em_dft", "_split"}
    missing = required - set(table.columns)
    if missing:
        raise SystemExit(f"nebDFT2k is missing scalar baseline columns: {sorted(missing)}")
    values = {}
    for split in ("train", "val", "test"):
        part = table[table["_split"].astype(str).str.lower() == split]
        x = np.asarray(part["em_bvse"], dtype=float)
        y = np.asarray(part["em_dft"], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        values[split] = (x[valid], y[valid])
    train_x, train_y = values["train"]
    slope, intercept = np.polyfit(train_x, train_y, 1)

    def row(name: str, predictor) -> dict[str, str]:
        metrics = {
            split: f"{np.mean(np.abs(predictor(x) - y)):.4f}"
            for split, (x, y) in values.items()
        }
        return {
            "experiment": "bvse_scalar_to_dft_barrier", "trajectory_source": "none",
            "target_source": "em_dft", "method": name, "params": "0", **metrics,
        }

    return [
        row("raw_em_bvse", lambda x: x),
        row(f"affine_em_bvse(a={slope:.4f},b={intercept:.4f})",
            lambda x: slope * x + intercept),
    ]


def cross_path_rows(args, auxiliary_mode: str = "none") -> list[dict[str, str]]:
    runner = Path(__file__).with_name("run_all_hyperedge_methods.py")
    command = [
        sys.executable, "-u", str(runner), "--litraj-data", str(args.data_root),
        "--litraj-dataset", "nebDFT2k", "--trajectory-source", "init",
        "--target-source", "em_dft", "--epochs", str(args.epochs),
        "--device", args.device, "--hidden-dim", str(args.hidden_dim),
        "--layers", str(args.layers), "--learning-rate", str(args.learning_rate),
        "--seed", str(args.seed), "--checkpoint-every", str(args.checkpoint_every),
        "--checkpoint-dir", str(args.checkpoint_dir / auxiliary_mode),
        "--auxiliary-mode", auxiliary_mode, "--methods", *METHODS,
    ]
    if args.resume:
        command.append("--resume")
    if args.mps_memory_report:
        command.append("--mps-memory-report")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    rows = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        match = RESULT.match(line.strip())
        if match:
            rows.append({
                "experiment": {
                    "none": "bvse_path_to_dft_barrier",
                    "fusion": "bvse_path_plus_scalar_to_dft_barrier",
                    "delta": "calibrated_bvse_plus_path_delta_to_dft_barrier",
                }[auxiliary_mode],
                "trajectory_source": "init", "target_source": "em_dft",
                **match.groupdict(),
            })
    returncode = process.wait()
    if returncode != 0:
        raise SystemExit(f"cross-fidelity training failed with exit code {returncode}")
    if len(rows) != len(METHODS):
        raise SystemExit(f"expected {len(METHODS)} cross-fidelity rows, captured {len(rows)}")
    return rows


def main() -> None:
    args = parser().parse_args()
    rows = prior_dft_rows(args.comparison_csv)
    existing_cross = []
    if args.output.exists():
        with args.output.open(newline="") as handle:
            existing_cross = [
                row for row in csv.DictReader(handle)
                if row.get("experiment") == "bvse_path_to_dft_barrier"
                and row.get("method") in METHODS
            ]
    if len(existing_cross) == len(METHODS):
        print(f"reusing completed path-only rows from {args.output}", flush=True)
        rows.extend(existing_cross)
    else:
        rows.extend(cross_path_rows(args, "none"))
    rows.extend(cross_path_rows(args, "fusion"))
    rows.extend(cross_path_rows(args, "delta"))
    rows.extend(scalar_baselines(args.data_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote cross-fidelity comparison: {args.output}")


if __name__ == "__main__":
    main()
