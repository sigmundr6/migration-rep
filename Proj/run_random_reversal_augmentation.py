#!/usr/bin/env python3
"""Train positional models with random s <-> 1-s augmentation and test both orientations."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch

from litraj_hypergraphs.data import materialize_prepared, prepare_litraj
from litraj_hypergraphs.experiment import move_splits, release_device_cache, resolve_device, train
from run_positional_controls import make_args, reversed_sample


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--early-stopping-patience", type=int, default=20)
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    p.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--neighbour-cutoff", type=float, default=3.5)
    p.add_argument("--hyperedge-cutoff", type=float, default=3.0)
    p.add_argument("--hyperedge-sigma", type=float, default=1.5)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--checkpoint-dir", type=Path,
                   default=Path("checkpoints/random_reversal_augmentation"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/random_reversal_augmentation"))
    p.add_argument("--preprocessed-cache-dir", type=Path,
                   default=Path("cache/litraj_preprocessed"))
    p.add_argument("--rebuild-preprocessed-cache", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def summaries(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    metric_rows, sensitivity_rows = [], []
    for seed in sorted({int(r["seed"]) for r in rows}):
        seed_rows = [r for r in rows if int(r["seed"]) == seed]
        for orientation in ("forward", "reverse", "inference_symmetrised"):
            subset = [r for r in seed_rows if r["orientation"] == orientation]
            target = np.asarray([r["target"] for r in subset], dtype=float)
            prediction = np.asarray([r["prediction"] for r in subset], dtype=float)
            error = target - prediction
            metric_rows.append({"seed": seed, "orientation": orientation,
                                "n": len(subset), "mae": np.mean(np.abs(error)),
                                "rmse": np.sqrt(np.mean(error ** 2)),
                                "median_ae": np.median(np.abs(error))})
        differences = np.asarray([r["forward_reverse_abs_difference"] for r in seed_rows
                                  if r["orientation"] == "forward"], dtype=float)
        sensitivity_rows.append({"seed": seed,
                                 "mean_abs_forward_reverse_difference": differences.mean(),
                                 "median_abs_forward_reverse_difference": np.median(differences)})
    return metric_rows, sensitivity_rows


def main() -> None:
    cli = parser().parse_args()
    # ``make_args`` centralises the exact architecture and preprocessing values
    # used by the completed positional-control family.
    cli.variants = ["positional"]
    device = resolve_device(cli.device)
    template = make_args(cli, cli.seeds[0], "positional")
    template.random_reversal_augmentation = True
    prepared = prepare_litraj(cli.data_root, "nebDFT2k", template)
    rows: list[dict] = []
    for seed in cli.seeds:
        args = make_args(cli, seed, "positional")
        args.random_reversal_augmentation = True
        args.checkpoint_dir = cli.checkpoint_dir
        splits = materialize_prepared(prepared, "positional", args)
        storage = torch.device("cpu") if device.type == "mps" else device
        splits = move_splits(splits, storage)
        model, _ = train(splits, args, device, "random_reversal_augmented_positional")
        model.eval()
        with torch.no_grad():
            for stored in splits["test"]:
                sample = stored if stored.target.device == device else stored.to(device)
                reverse = reversed_sample(sample)
                forward_value = float(model(sample).item())
                reverse_value = float(model(reverse).item())
                target = float(sample.target.item())
                difference = abs(forward_value - reverse_value)
                for orientation, prediction in (
                    ("forward", forward_value), ("reverse", reverse_value),
                    ("inference_symmetrised", 0.5 * (forward_value + reverse_value)),
                ):
                    rows.append({"seed": seed, "source": sample.source,
                                 "orientation": orientation, "target": target,
                                 "prediction": prediction,
                                 "absolute_error": abs(target - prediction),
                                 "forward_reverse_abs_difference": difference})
        write(cli.output_dir / "per_sample_predictions.csv", rows)
        del model, splits
        release_device_cache(device, f"random_reversal_seed{seed}")
    metric_rows, sensitivity_rows = summaries(rows)
    write(cli.output_dir / "seed_metrics.csv", metric_rows)
    write(cli.output_dir / "reversal_sensitivity.csv", sensitivity_rows)
    print(f"wrote random-reversal augmentation results to {cli.output_dir}")


if __name__ == "__main__":
    main()
