#!/usr/bin/env python3
"""Repeated-seed, per-hop statistical evaluation for the final LiTraj models."""

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
    checkpoint_path, move_splits, release_device_cache, resolve_device, train,
)


CORE = ("crystal", "midpoint", "trajectory", "positional")
SCENARIOS = (
    ("dft_path", "relaxed", "none", CORE),
    ("bvse_path", "init", "none", ("positional",)),
    ("bvse_scalar_path_fusion", "init", "fusion", ("positional",)),
    ("bvse_calibrated_delta", "init", "delta", ("positional",)),
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[7, 17, 29, 43, 71, 89, 101, 131, 167, 197])
    p.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--early-stopping-patience", type=int, default=20)
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    p.add_argument("--include-randomized", action="store_true")
    p.add_argument("--checkpoint-dir", type=Path,
                   default=Path("checkpoints/final_statistical_evaluation"))
    p.add_argument("--output-dir", type=Path, default=Path("results/final_statistics"))
    p.add_argument("--preprocessed-cache-dir", type=Path,
                   default=Path("cache/litraj_preprocessed"))
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


def experiment_args(cli, source: str, mode: str, seed: int, scenario: str):
    return SimpleNamespace(
        litraj_data=cli.data_root, litraj_dataset="nebDFT2k",
        trajectory_source=source, target_source="em_dft", manifest=None, trajectory=None,
        neighbour_cutoff=cli.neighbour_cutoff, hyperedge_cutoff=cli.hyperedge_cutoff,
        hyperedge_sigma=cli.hyperedge_sigma,
        bottleneck_neighbours=cli.bottleneck_neighbours, change_top_k=cli.change_top_k,
        num_segments=cli.num_segments, segment_sigma=cli.segment_sigma,
        hidden_dim=cli.hidden_dim, layers=cli.layers, epochs=cli.epochs,
        learning_rate=cli.learning_rate, seed=seed,
        reversal_invariant=cli.reversal_invariant,
        presence_threshold=cli.presence_threshold, auxiliary_mode=mode,
        checkpoint_dir=cli.checkpoint_dir / scenario,
        checkpoint_every=cli.checkpoint_every, resume=cli.resume,
        early_stopping_patience=cli.early_stopping_patience,
        early_stopping_min_delta=cli.early_stopping_min_delta,
        mps_memory_report=False, mps_data="stream",
        preprocessed_cache_dir=cli.preprocessed_cache_dir,
        rebuild_preprocessed_cache=cli.rebuild_preprocessed_cache,
    )


def fit_calibration(prepared, args) -> tuple[float, float]:
    pairs = [(item.cheap_barrier, item.target) for item in prepared["train"]]
    if any(x is None for x, _ in pairs):
        raise ValueError("em_bvse is missing from at least one training sample")
    x, y = np.asarray(pairs, dtype=float).T
    slope, intercept = np.polyfit(x, y, 1)
    args.delta_slope, args.delta_intercept = float(slope), float(intercept)
    return args.delta_slope, args.delta_intercept


def predictions(model, samples, device, scenario, method, seed):
    model.eval()
    rows = []
    with torch.no_grad():
        for stored in samples:
            sample = stored if stored.target.device == device else stored.to(device)
            predicted = float(model(sample).item())
            target = float(sample.target.item())
            rows.append({
                "scenario": scenario, "method": method, "seed": seed,
                "source": sample.source, "split": "test", "target": target,
                "prediction": predicted, "absolute_error": abs(target - predicted),
            })
    return rows


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


def metrics(rows):
    y = np.asarray([row["target"] for row in rows])
    p = np.asarray([row["prediction"] for row in rows])
    error = y - p
    denominator = np.sum((y - y.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "median_ae": float(np.median(np.abs(error))),
        "r2": float(1 - np.sum(error ** 2) / denominator) if denominator else float("nan"),
        "spearman": float(np.corrcoef(rank(y), rank(p))[0, 1]),
    }


def write_rows(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_validation_curves(rows, path: Path) -> None:
    matplotlib_cache = path.parent / ".matplotlib_cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache.resolve()))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping validation-curve plot", flush=True)
        return
    figure, axis = plt.subplots(figsize=(9, 6))
    groups = sorted({(r["scenario"], r["method"]) for r in rows})
    for scenario, method in groups:
        subset = [r for r in rows if r["scenario"] == scenario and r["method"] == method]
        epochs = sorted({int(r["epoch"]) for r in subset})
        means = []
        for epoch in epochs:
            values = [float(r["val_mae"]) for r in subset if int(r["epoch"]) == epoch]
            means.append(float(np.mean(values)))
        axis.plot(epochs, means, label=f"{scenario}: {method}")
    axis.axvline(25, color="black", linestyle="--", linewidth=1, alpha=0.5,
                 label="previous 25-epoch horizon")
    axis.set(xlabel="Epoch", ylabel="Mean validation MAE (eV)",
             title="Validation convergence across seeds")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def add_affine_rows(all_predictions, prepared_init, seeds, slope, intercept):
    for seed in seeds:
        for item in prepared_init["test"]:
            prediction = slope * float(item.cheap_barrier) + intercept
            all_predictions.append({
                "scenario": "affine_bvse", "method": "affine_em_bvse", "seed": seed,
                "source": item.source, "split": "test", "target": item.target,
                "prediction": prediction, "absolute_error": abs(item.target - prediction),
            })


def paired_bootstrap(rows, bootstrap_samples: int, seed: int):
    comparisons = (
        ("dft_midpoint_minus_positional", ("dft_path", "midpoint"),
         ("dft_path", "positional")),
        ("affine_minus_bvse_fusion", ("affine_bvse", "affine_em_bvse"),
         ("bvse_scalar_path_fusion", "positional")),
        ("affine_minus_bvse_delta", ("affine_bvse", "affine_em_bvse"),
         ("bvse_calibrated_delta", "positional")),
        ("dft_path_positional_minus_bvse_fusion", ("dft_path", "positional"),
         ("bvse_scalar_path_fusion", "positional")),
    )
    lookup = {(r["scenario"], r["method"], int(r["seed"]), r["source"]):
              float(r["absolute_error"]) for r in rows}
    output, rng = [], np.random.default_rng(seed)
    for name, left, right in comparisons:
        deltas_by_source = {}
        for key, left_error in lookup.items():
            scenario, method, run_seed, source = key
            if (scenario, method) != left:
                continue
            right_key = (right[0], right[1], run_seed, source)
            if right_key in lookup:
                deltas_by_source.setdefault(source, []).append(left_error - lookup[right_key])
        deltas = np.asarray([np.mean(values) for values in deltas_by_source.values()])
        if not len(deltas):
            continue
        draws = rng.choice(deltas, size=(bootstrap_samples, len(deltas)), replace=True).mean(1)
        output.append({
            "comparison": name, "n_test_hops": len(deltas),
            "mean_paired_improvement": float(deltas.mean()),
            "median_paired_improvement": float(np.median(deltas)),
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "probability_improvement_gt_zero": float(np.mean(draws > 0)),
        })
    return output


def main() -> None:
    cli = parser().parse_args()
    device = resolve_device(cli.device)
    if device.type == "mps":
        print("warning: repeated-seed statistics are most stable with --device cpu", flush=True)
    prepared_by_source = {}
    all_predictions = []
    validation_curves = []
    training_runs = {}
    prediction_path = cli.output_dir / "per_sample_predictions.csv"

    scenarios = list(SCENARIOS)
    if cli.include_randomized:
        scenarios[0] = ("dft_path", "relaxed", "none", CORE + ("randomized",))
    for scenario, source, mode, methods in scenarios:
        template = experiment_args(cli, source, mode, cli.seeds[0], scenario)
        prepared = prepared_by_source.get(source)
        if prepared is None:
            prepared = prepare_litraj(cli.data_root, "nebDFT2k", template)
            prepared_by_source[source] = prepared
        for seed in cli.seeds:
            args = experiment_args(cli, source, mode, seed, scenario)
            if mode in {"fusion", "delta"}:
                slope, intercept = fit_calibration(prepared, args)
                print(f"[{scenario}] train calibration: {slope:.6f}*BVSE{intercept:+.6f}")
            for method in methods:
                splits = materialize_prepared(prepared, method, args)
                storage = torch.device("cpu") if device.type == "mps" else device
                splits = move_splits(splits, storage)
                run_name = f"{scenario}_{method}"
                model, _ = train(splits, args, device, run_name)
                saved = torch.load(
                    checkpoint_path(args, run_name), map_location="cpu", weights_only=False
                )
                saved_best_epoch = int(saved.get("best_epoch", 0))
                training_runs[(scenario, method, seed)] = {
                    "best_epoch": saved_best_epoch,
                    "trained_epochs": int(saved["epoch"]),
                }
                for item in saved.get("history", []):
                    validation_curves.append({
                        "scenario": scenario, "method": method, "seed": seed,
                        "epoch": int(item["epoch"]), "train_mae": item["train_mae"],
                        "val_mae": item["val_mae"], "best_epoch": saved_best_epoch,
                        "selected": int(int(item["epoch"]) == saved_best_epoch),
                    })
                all_predictions.extend(
                    predictions(model, splits["test"], device, scenario, method, seed)
                )
                write_rows(prediction_path, all_predictions, tuple(all_predictions[0]))
                write_rows(cli.output_dir / "validation_curves.csv", validation_curves,
                           tuple(validation_curves[0]))
                del model, splits
                release_device_cache(device, run_name)

    calibration_args = experiment_args(cli, "init", "delta", cli.seeds[0], "affine")
    slope, intercept = fit_calibration(prepared_by_source["init"], calibration_args)
    add_affine_rows(all_predictions, prepared_by_source["init"], cli.seeds, slope, intercept)
    write_rows(prediction_path, all_predictions, tuple(all_predictions[0]))

    seed_metrics = []
    groups = sorted({(r["scenario"], r["method"], int(r["seed"])) for r in all_predictions})
    for scenario, method, seed in groups:
        subset = [r for r in all_predictions if r["scenario"] == scenario
                  and r["method"] == method and int(r["seed"]) == seed]
        run = training_runs.get((scenario, method, seed), {"best_epoch": 0, "trained_epochs": 0})
        seed_metrics.append({"scenario": scenario, "method": method, "seed": seed,
                             **run, **metrics(subset)})
    write_rows(cli.output_dir / "seed_metrics.csv", seed_metrics, tuple(seed_metrics[0]))

    summary = []
    for scenario, method in sorted({(r["scenario"], r["method"]) for r in seed_metrics}):
        subset = [r for r in seed_metrics if r["scenario"] == scenario and r["method"] == method]
        row = {"scenario": scenario, "method": method, "seeds": len(subset)}
        for metric in ("mae", "rmse", "median_ae", "r2", "spearman"):
            values = np.asarray([r[metric] for r in subset])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        epochs = np.asarray([r["best_epoch"] for r in subset], dtype=float)
        row["best_epoch_mean"] = float(epochs.mean())
        row["best_epoch_std"] = float(epochs.std(ddof=1)) if len(epochs) > 1 else 0.0
        row["fraction_best_epoch_at_or_after_25"] = float(np.mean(epochs >= 25))
        summary.append(row)
    write_rows(cli.output_dir / "summary.csv", summary, tuple(summary[0]))

    bootstrap = paired_bootstrap(all_predictions, cli.bootstrap_samples, cli.bootstrap_seed)
    write_rows(cli.output_dir / "paired_bootstrap.csv", bootstrap, tuple(bootstrap[0]))
    plot_validation_curves(validation_curves, cli.output_dir / "validation_curves.png")
    print(f"wrote final statistical evaluation to {cli.output_dir}")


if __name__ == "__main__":
    main()
