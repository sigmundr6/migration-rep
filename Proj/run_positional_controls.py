#!/usr/bin/env python3
"""Focused reversal and shuffled-reaction-coordinate controls for nebDFT2k."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from litraj_hypergraphs.data import materialize_prepared, prepare_litraj
from litraj_hypergraphs.experiment import move_splits, release_device_cache, resolve_device, train


VARIANTS = {
    "midpoint": ("midpoint", False),
    "unordered": ("trajectory", False),
    "shuffled_s": ("positional_shuffled_s", False),
    "positional": ("positional", False),
    "positional_invariant": ("positional", True),
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[7, 17, 29, 43, 71, 89, 101, 131, 167, 197])
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
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/positional_controls"))
    p.add_argument("--output-dir", type=Path, default=Path("results/positional_controls"))
    p.add_argument("--preprocessed-cache-dir", type=Path, default=Path("cache/litraj_preprocessed"))
    p.add_argument("--rebuild-preprocessed-cache", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def make_args(cli, seed: int, variant: str) -> SimpleNamespace:
    _, invariant = VARIANTS[variant]
    return SimpleNamespace(
        litraj_data=cli.data_root, litraj_dataset="nebDFT2k",
        trajectory_source="relaxed", target_source="em_dft",
        manifest=None, trajectory=None, neighbour_cutoff=cli.neighbour_cutoff,
        hyperedge_cutoff=cli.hyperedge_cutoff, hyperedge_sigma=cli.hyperedge_sigma,
        bottleneck_neighbours=4, change_top_k=8, num_segments=5, segment_sigma=0.18,
        hidden_dim=cli.hidden_dim, layers=cli.layers, epochs=cli.epochs,
        learning_rate=cli.learning_rate, seed=seed, reversal_invariant=invariant,
        presence_threshold=1e-6, auxiliary_mode="none",
        checkpoint_dir=cli.checkpoint_dir / variant,
        checkpoint_every=cli.checkpoint_every, resume=cli.resume,
        early_stopping_patience=cli.early_stopping_patience,
        early_stopping_min_delta=cli.early_stopping_min_delta,
        mps_memory_report=False, mps_data="stream",
        preprocessed_cache_dir=cli.preprocessed_cache_dir,
        rebuild_preprocessed_cache=cli.rebuild_preprocessed_cache,
    )


def reversed_sample(sample):
    """Reverse only the stored reaction coordinate; geometry and weights are unchanged."""
    return replace(sample, path_position=1.0 - sample.path_position)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def metric_rows(predictions: list[dict]) -> list[dict]:
    output = []
    groups = sorted({(r["variant"], r["seed"], r["orientation"]) for r in predictions})
    for variant, seed, orientation in groups:
        rows = [r for r in predictions if (r["variant"], r["seed"], r["orientation"])
                == (variant, seed, orientation)]
        y = np.asarray([r["target"] for r in rows], dtype=float)
        p = np.asarray([r["prediction"] for r in rows], dtype=float)
        error = y - p
        denominator = np.sum((y - y.mean()) ** 2)
        output.append({
            "variant": variant, "seed": seed, "orientation": orientation,
            "n": len(rows), "mae": np.mean(np.abs(error)),
            "rmse": np.sqrt(np.mean(error ** 2)),
            "median_ae": np.median(np.abs(error)),
            "r2": 1 - np.sum(error ** 2) / denominator if denominator else np.nan,
        })
    return output


def main() -> None:
    cli = parser().parse_args()
    device = resolve_device(cli.device)
    template = make_args(cli, cli.seeds[0], cli.variants[0])
    prepared = prepare_litraj(cli.data_root, "nebDFT2k", template)
    predictions: list[dict] = []
    for variant in cli.variants:
        candidate, _ = VARIANTS[variant]
        for seed in cli.seeds:
            args = make_args(cli, seed, variant)
            splits = materialize_prepared(prepared, candidate, args)
            storage = torch.device("cpu") if device.type == "mps" else device
            splits = move_splits(splits, storage)
            model, _ = train(splits, args, device, f"positional_control_{variant}")
            model.eval()
            with torch.no_grad():
                for stored in splits["test"]:
                    sample = stored if stored.target.device == device else stored.to(device)
                    reverse = reversed_sample(sample)
                    forward_value = float(model(sample).item())
                    reverse_value = float(model(reverse).item())
                    target = float(sample.target.item())
                    for orientation, value in (
                        ("forward", forward_value), ("reverse", reverse_value),
                        ("inference_symmetrised", 0.5 * (forward_value + reverse_value)),
                    ):
                        predictions.append({
                            "variant": variant, "candidate": candidate, "seed": seed,
                            "source": sample.source, "orientation": orientation,
                            "target": target, "prediction": value,
                            "absolute_error": abs(target - value),
                            "forward_reverse_abs_difference": abs(forward_value - reverse_value),
                        })
            write_csv(cli.output_dir / "per_sample_predictions.csv", predictions)
            del model, splits
            release_device_cache(device, f"{variant}_seed{seed}")

    metrics = metric_rows(predictions)
    write_csv(cli.output_dir / "seed_metrics.csv", metrics)
    sensitivity = []
    for variant in cli.variants:
        rows = [r for r in predictions if r["variant"] == variant and r["orientation"] == "forward"]
        by_seed = []
        for seed in cli.seeds:
            values = [r["forward_reverse_abs_difference"] for r in rows if r["seed"] == seed]
            if values:
                by_seed.append((seed, np.mean(values), np.median(values)))
        for seed, mean_difference, median_difference in by_seed:
            sensitivity.append({"variant": variant, "seed": seed,
                                "mean_abs_forward_reverse_difference": mean_difference,
                                "median_abs_forward_reverse_difference": median_difference})
    write_csv(cli.output_dir / "reversal_sensitivity.csv", sensitivity)
    print(f"wrote positional controls to {cli.output_dir}")


if __name__ == "__main__":
    main()
