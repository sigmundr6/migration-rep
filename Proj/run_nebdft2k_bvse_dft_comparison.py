#!/usr/bin/env python3
"""Run identical representations on official BVSE and DFT nebDFT2k paths."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import subprocess
import sys


RESULT = re.compile(
    r"^(?P<method>\S+)\s+(?P<params>\d+)\s+"
    r"(?P<train>[0-9.]+) eV\s+(?P<val>[0-9.]+) eV\s+(?P<test>[0-9.]+) eV$"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-root", type=Path, default=Path("data"))
    result.add_argument("--epochs", type=int, default=25)
    result.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    result.add_argument("--mps-data", choices=["stream", "resident"], default="stream")
    result.add_argument("--mps-memory-report", action="store_true")
    result.add_argument(
        "--no-mps-isolate-methods", action="store_true",
        help="run all methods in one MPS process (diagnostic only; may accumulate Metal memory)",
    )
    result.add_argument("--hidden-dim", type=int, default=32)
    result.add_argument("--layers", type=int, default=2)
    result.add_argument("--learning-rate", type=float, default=2e-3)
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--checkpoint-every", type=int, default=5)
    result.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/nebdft2k_theory"))
    result.add_argument("--output", type=Path, default=Path("results/nebdft2k_bvse_dft.csv"))
    result.add_argument("--resume", action="store_true")
    result.add_argument("--reversal-invariant", action="store_true")
    result.add_argument(
        "--no-cpu-fallback", action="store_true",
        help="do not resume on CPU if the MPS child process terminates by signal",
    )
    return result


def run_theory(args, theory: str) -> list[dict[str, str]]:
    source, target = ("init", "em_bvse") if theory == "bvse" else ("relaxed", "em_dft")
    runner = Path(__file__).with_name("run_all_hyperedge_methods.py")
    command = [
        sys.executable, "-u", str(runner), "--litraj-data", str(args.data_root),
        "--litraj-dataset", "nebDFT2k", "--trajectory-source", source,
        "--target-source", target, "--epochs", str(args.epochs),
        "--device", args.device, "--hidden-dim", str(args.hidden_dim),
        "--mps-data", args.mps_data,
        "--layers", str(args.layers), "--learning-rate", str(args.learning_rate),
        "--seed", str(args.seed), "--checkpoint-every", str(args.checkpoint_every),
        "--checkpoint-dir", str(args.checkpoint_dir / theory),
    ]
    if args.resume:
        command.append("--resume")
    if args.reversal_invariant:
        command.append("--reversal-invariant")
    if args.mps_memory_report:
        command.append("--mps-memory-report")
    if args.no_mps_isolate_methods:
        command.append("--no-mps-isolate-methods")
    print(f"\n=== {theory.upper()}: {source}.xyz -> {target} ===", flush=True)
    def execute(values):
        process = subprocess.Popen(
            values, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        captured = []
        mps_out_of_memory = False
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            if "MPS backend out of memory" in line:
                mps_out_of_memory = True
            if match := RESULT.match(line.strip()):
                captured.append({"theory": theory, "trajectory_source": source,
                                 "target_source": target, **match.groupdict()})
        return process.wait(), captured, mps_out_of_memory

    returncode, records, mps_out_of_memory = execute(command)
    if (returncode < 0 or mps_out_of_memory) and args.device != "cpu" and not args.no_cpu_fallback:
        reason = f"signal {-returncode}" if returncode < 0 else "MPS out-of-memory"
        print(
            f"\n{theory} GPU process ended from {reason}; "
            "resuming its latest checkpoint on CPU.", flush=True,
        )
        device_index = command.index("--device") + 1
        command[device_index] = "cpu"
        if "--resume" not in command:
            command.append("--resume")
        returncode, resumed_records, _ = execute(command)
        # Resumed output supersedes any earlier final row for the same method.
        by_method = {record["method"]: record for record in records}
        by_method.update({record["method"]: record for record in resumed_records})
        records = list(by_method.values())
    if returncode != 0:
        raise SystemExit(f"{theory} comparison failed with exit code {returncode}")
    return records


def main() -> None:
    args = parser().parse_args()
    records = run_theory(args, "bvse") + run_theory(args, "dft")
    if not records:
        raise SystemExit("no result rows were captured")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nwrote paired comparison: {args.output}")


if __name__ == "__main__":
    main()
