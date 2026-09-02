#!/usr/bin/env python3
"""Repeated-seed evaluation of all exploratory LiTraj trajectory representations."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from litraj_hypergraphs.data import materialize_prepared, prepare_litraj
from litraj_hypergraphs.experiment import (
    checkpoint_path,
    move_splits,
    release_device_cache,
    resolve_device,
    train,
)


ALL_METHODS = (
    "crystal",
    "midpoint",
    "trajectory",
    "positional",
    "positional_shuffled_s",
    "segmented",
    "randomized",
    "image_coordination",
    "geometric_bottleneck",
    "coordination_change",
    "species_image",
    "position_aware_trajectory_graph",
    "position_aware_trajectory_graph_shuffled",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 17, 29, 43, 71, 89, 101, 131, 167, 197],
    )
    p.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    p.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--early-stopping-patience", type=int, default=20)
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/exploratory_statistical_evaluation"),
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("results/exploratory_statistics")
    )
    p.add_argument(
        "--preprocessed-cache-dir", type=Path, default=Path("cache/litraj_preprocessed")
    )
    p.add_argument("--rebuild-preprocessed-cache", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=2026)
    p.add_argument("--neighbour-cutoff", type=float, default=3.5)
    p.add_argument("--hyperedge-cutoff", type=float, default=3.0)
    p.add_argument("--hyperedge-sigma", type=float, default=1.5)
    p.add_argument("--bottleneck-neighbours", type=int, default=4)
    p.add_argument("--change-top-k", type=int, default=8)
    p.add_argument("--num-segments", type=int, default=5)
    p.add_argument("--segment-sigma", type=float, default=0.18)
    p.add_argument("--presence-threshold", type=float, default=1e-6)
    p.add_argument("--reversal-invariant", action="store_true")
    return p


def experiment_args(cli: argparse.Namespace, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        litraj_data=cli.data_root,
        litraj_dataset="nebDFT2k",
        trajectory_source="relaxed",
        target_source="em_dft",
        manifest=None,
        trajectory=None,
        neighbour_cutoff=cli.neighbour_cutoff,
        hyperedge_cutoff=cli.hyperedge_cutoff,
        hyperedge_sigma=cli.hyperedge_sigma,
        bottleneck_neighbours=cli.bottleneck_neighbours,
        change_top_k=cli.change_top_k,
        num_segments=cli.num_segments,
        segment_sigma=cli.segment_sigma,
        hidden_dim=cli.hidden_dim,
        layers=cli.layers,
        epochs=cli.epochs,
        learning_rate=cli.learning_rate,
        seed=seed,
        reversal_invariant=cli.reversal_invariant,
        presence_threshold=cli.presence_threshold,
        auxiliary_mode="none",
        checkpoint_dir=cli.checkpoint_dir,
        checkpoint_every=cli.checkpoint_every,
        resume=cli.resume,
        early_stopping_patience=cli.early_stopping_patience,
        early_stopping_min_delta=cli.early_stopping_min_delta,
        mps_memory_report=False,
        mps_data="stream",
        preprocessed_cache_dir=cli.preprocessed_cache_dir,
        rebuild_preprocessed_cache=cli.rebuild_preprocessed_cache,
    )


def write_rows(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return result


def metrics(rows: list[dict]) -> dict[str, float]:
    target = np.asarray([row["target"] for row in rows], dtype=float)
    prediction = np.asarray([row["prediction"] for row in rows], dtype=float)
    error = target - prediction
    denominator = np.sum((target - target.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_ae": float(np.median(np.abs(error))),
        "r2": float(1 - np.sum(error**2) / denominator) if denominator else float("nan"),
        "spearman": float(np.corrcoef(rank(target), rank(prediction))[0, 1]),
    }


def predict(model, samples, device: torch.device, method: str, seed: int) -> list[dict]:
    model.eval()
    rows = []
    with torch.no_grad():
        for stored in samples:
            sample = stored if stored.target.device == device else stored.to(device)
            prediction = float(model(sample).item())
            target = float(sample.target.item())
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "source": sample.source,
                    "split": "test",
                    "target": target,
                    "prediction": prediction,
                    "absolute_error": abs(target - prediction),
                }
            )
    return rows


def paired_bootstrap(
    rows: list[dict], methods: list[str], draws: int, seed: int
) -> list[dict]:
    """Compare each method with positional after averaging each hop over seeds."""
    lookup = {
        (row["method"], int(row["seed"]), row["source"]): float(row["absolute_error"])
        for row in rows
    }
    comparisons = [(method, "positional") for method in methods if method != "positional"]
    ordered = "position_aware_trajectory_graph"
    shuffled = "position_aware_trajectory_graph_shuffled"
    if ordered in methods and shuffled in methods:
        comparisons.append((shuffled, ordered))

    rng = np.random.default_rng(seed)
    output = []
    for left, right in comparisons:
        by_source: dict[str, list[float]] = {}
        for (method, run_seed, source), left_error in lookup.items():
            if method != left:
                continue
            right_key = (right, run_seed, source)
            if right_key in lookup:
                by_source.setdefault(source, []).append(left_error - lookup[right_key])
        differences = np.asarray([np.mean(values) for values in by_source.values()])
        if not len(differences):
            continue
        bootstrap = rng.choice(
            differences, size=(draws, len(differences)), replace=True
        ).mean(axis=1)
        output.append(
            {
                "comparison": f"{left}_minus_{right}",
                "left_method": left,
                "right_method": right,
                "n_test_hops": len(differences),
                "mean_paired_error_difference": float(differences.mean()),
                "median_paired_error_difference": float(np.median(differences)),
                "ci95_low": float(np.quantile(bootstrap, 0.025)),
                "ci95_high": float(np.quantile(bootstrap, 0.975)),
                "probability_left_error_gt_right": float(np.mean(bootstrap > 0)),
            }
        )
    return output


def plot_curves(rows: list[dict], path: Path) -> None:
    matplotlib_cache = path.parent / ".matplotlib_cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache.resolve()))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping validation plot", flush=True)
        return
    figure, axis = plt.subplots(figsize=(11, 7))
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        epochs = sorted({int(row["epoch"]) for row in subset})
        means = [
            np.mean([float(row["val_mae"]) for row in subset if int(row["epoch"]) == epoch])
            for epoch in epochs
        ]
        axis.plot(epochs, means, label=method)
    axis.set(
        xlabel="Epoch",
        ylabel="Mean validation MAE (eV)",
        title="Exploratory representations: validation convergence across seeds",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    cli = parser().parse_args()
    methods = list(dict.fromkeys(cli.methods))
    device = resolve_device(cli.device)
    if device.type == "mps":
        print(
            "warning: CPU is recommended for stable repeated-seed statistics; "
            "MPS streams samples but this runner does not isolate every run in a child process",
            flush=True,
        )

    template = experiment_args(cli, cli.seeds[0])
    prepared = prepare_litraj(cli.data_root, "nebDFT2k", template)
    predictions: list[dict] = []
    histories: list[dict] = []
    run_info: dict[tuple[str, int], dict[str, int]] = {}

    for method in methods:
        for seed in cli.seeds:
            args = experiment_args(cli, seed)
            splits = materialize_prepared(prepared, method, args)
            storage = torch.device("cpu") if device.type == "mps" else device
            splits = move_splits(splits, storage)
            model, _ = train(splits, args, device, f"exploratory_{method}")
            saved = torch.load(
                checkpoint_path(args, f"exploratory_{method}"),
                map_location="cpu",
                weights_only=False,
            )
            best_epoch = int(saved.get("best_epoch", 0))
            run_info[(method, seed)] = {
                "best_epoch": best_epoch,
                "trained_epochs": int(saved["epoch"]),
            }
            for item in saved.get("history", []):
                histories.append(
                    {
                        "method": method,
                        "seed": seed,
                        "epoch": int(item["epoch"]),
                        "train_mae": item["train_mae"],
                        "val_mae": item["val_mae"],
                        "best_epoch": best_epoch,
                        "selected": int(int(item["epoch"]) == best_epoch),
                    }
                )
            predictions.extend(predict(model, splits["test"], device, method, seed))
            write_rows(
                cli.output_dir / "per_sample_predictions.csv",
                predictions,
                tuple(predictions[0]),
            )
            write_rows(
                cli.output_dir / "validation_curves.csv", histories, tuple(histories[0])
            )
            del model, splits
            release_device_cache(device, f"exploratory_{method}_seed{seed}")

    seed_rows = []
    for method in methods:
        for seed in cli.seeds:
            subset = [
                row
                for row in predictions
                if row["method"] == method and int(row["seed"]) == seed
            ]
            if not subset:
                continue
            seed_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    **run_info[(method, seed)],
                    **metrics(subset),
                }
            )
    write_rows(cli.output_dir / "seed_metrics.csv", seed_rows, tuple(seed_rows[0]))

    summary = []
    for method in methods:
        subset = [row for row in seed_rows if row["method"] == method]
        if not subset:
            continue
        row = {"method": method, "seeds": len(subset)}
        for metric in ("mae", "rmse", "median_ae", "r2", "spearman"):
            values = np.asarray([item[metric] for item in subset], dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        epochs = np.asarray([item["best_epoch"] for item in subset], dtype=float)
        row["best_epoch_mean"] = float(epochs.mean())
        row["best_epoch_std"] = float(epochs.std(ddof=1)) if len(epochs) > 1 else 0.0
        row["early_stopped_fraction"] = float(
            np.mean([item["trained_epochs"] < cli.epochs for item in subset])
        )
        summary.append(row)
    write_rows(cli.output_dir / "summary.csv", summary, tuple(summary[0]))

    paired = paired_bootstrap(
        predictions, methods, cli.bootstrap_samples, cli.bootstrap_seed
    )
    if paired:
        write_rows(cli.output_dir / "paired_bootstrap.csv", paired, tuple(paired[0]))
    plot_curves(histories, cli.output_dir / "validation_curves.png")
    print(f"wrote exploratory repeated-seed evaluation to {cli.output_dir}", flush=True)


if __name__ == "__main__":
    main()
