#!/usr/bin/env python3
"""Material-disjoint cross-validation for the headline LiTraj GNN experiments.

This experiment complements the official hop-level split.  All hops belonging to
one Materials Project parent structure are assigned together.  Each outer test
fold is therefore an unseen-material transfer test, and early stopping uses a
separate group-disjoint validation split made only from the outer training data.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from litraj_hypergraphs.data import PreparedTrajectory, materialize_prepared, prepare_litraj
from litraj_hypergraphs.experiment import (
    checkpoint_path, move_splits, release_device_cache, resolve_device, train,
)


CORE_METHODS = ("crystal", "midpoint", "trajectory", "positional")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--validation-fraction", type=float, default=0.125,
                   help="fraction of outer-training material groups used for validation")
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--early-stopping-patience", type=int, default=20)
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    p.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--checkpoint-dir", type=Path,
                   default=Path("checkpoints/material_disjoint_evaluation"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/material_disjoint_evaluation"))
    p.add_argument("--preprocessed-cache-dir", type=Path,
                   default=Path("cache/litraj_preprocessed"))
    p.add_argument("--rebuild-preprocessed-cache", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--core-methods", nargs="+", choices=CORE_METHODS,
                   default=list(CORE_METHODS))
    p.add_argument("--skip-core", action="store_true")
    p.add_argument("--skip-q4", action="store_true",
                   help="skip affine BVSE and BVSE positional delta experiments")
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


def material_id(source: str) -> str:
    match = re.search(r"(mp-\d+)", str(source))
    if not match:
        raise ValueError(f"cannot extract parent material from source {source!r}")
    return match.group(1)


def experiment_args(cli, source: str, mode: str, seed: int, fold: int, scenario: str):
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
        checkpoint_dir=(cli.checkpoint_dir / f"split_seed_{cli.split_seed}"
                        / f"fold_{fold}" / scenario),
        checkpoint_every=cli.checkpoint_every, resume=cli.resume,
        early_stopping_patience=cli.early_stopping_patience,
        early_stopping_min_delta=cli.early_stopping_min_delta,
        mps_memory_report=False, mps_data="stream",
        preprocessed_cache_dir=cli.preprocessed_cache_dir,
        rebuild_preprocessed_cache=cli.rebuild_preprocessed_cache,
    )


def flatten(prepared: dict[str, list[PreparedTrajectory]]) -> list[PreparedTrajectory]:
    values = [item for split in ("train", "val", "test") for item in prepared[split]]
    sources = [item.source for item in values]
    if len(sources) != len(set(sources)):
        raise ValueError("LiTraj source identifiers are not unique")
    return values


def make_fold_indices(items: list[PreparedTrajectory], cli):
    groups = np.asarray([material_id(item.source) for item in items])
    if len(set(groups)) < cli.folds:
        raise ValueError(f"need at least {cli.folds} parent materials")
    outer = GroupKFold(n_splits=cli.folds, shuffle=True, random_state=cli.split_seed)
    dummy = np.zeros(len(items))
    folds = []
    for fold, (outer_train, test) in enumerate(outer.split(dummy, groups=groups)):
        inner = GroupShuffleSplit(
            n_splits=1, test_size=cli.validation_fraction,
            random_state=cli.split_seed + 10_000 + fold,
        )
        inner_train, inner_val = next(inner.split(
            np.zeros(len(outer_train)), groups=groups[outer_train]
        ))
        train = outer_train[inner_train]
        val = outer_train[inner_val]
        group_sets = [set(groups[index]) for index in (train, val, test)]
        if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
            raise AssertionError("material leakage between train, validation, and test")
        folds.append((train, val, test))
    test_indices = np.concatenate([fold[2] for fold in folds])
    if sorted(test_indices.tolist()) != list(range(len(items))):
        raise AssertionError("outer folds do not form exactly one test partition per hop")
    return folds


def align_source(items: list[PreparedTrajectory], canonical: list[PreparedTrajectory]):
    lookup = {item.source: item for item in items}
    missing = [item.source for item in canonical if item.source not in lookup]
    extra = sorted(set(lookup) - {item.source for item in canonical})
    if missing or extra:
        raise ValueError(
            f"relaxed/init trajectory sources do not align: missing={missing[:3]}, extra={extra[:3]}"
        )
    aligned = [lookup[item.source] for item in canonical]
    for left, right in zip(canonical, aligned):
        if not np.isclose(left.target, right.target):
            raise ValueError(f"DFT target mismatch for {left.source}")
    return aligned


def subset(items, indices) -> list[PreparedTrajectory]:
    return [items[int(index)] for index in indices]


def prepared_fold(items, indices):
    train, val, test = indices
    return {"train": subset(items, train), "val": subset(items, val), "test": subset(items, test)}


def fit_calibration(prepared, args) -> tuple[float, float]:
    pairs = [(item.cheap_barrier, item.target) for item in prepared["train"]]
    if any(x is None or not np.isfinite(float(x)) for x, _ in pairs):
        raise ValueError("em_bvse is missing from a fold's training data")
    x, y = np.asarray(pairs, dtype=float).T
    slope, intercept = np.polyfit(x, y, 1)
    args.delta_slope, args.delta_intercept = float(slope), float(intercept)
    return args.delta_slope, args.delta_intercept


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


def metrics(rows) -> dict[str, float]:
    y = np.asarray([float(row["target"]) for row in rows])
    p = np.asarray([float(row["prediction"]) for row in rows])
    error = y - p
    denominator = np.sum((y - y.mean()) ** 2)
    spearman = np.corrcoef(rank(y), rank(p))[0, 1] if len(y) > 1 else np.nan
    return {
        "n_test": len(y), "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "median_ae": float(np.median(np.abs(error))),
        "r2": float(1 - np.sum(error ** 2) / denominator) if denominator else np.nan,
        "spearman": float(spearman),
    }


def predict(model, samples, device, fold, scenario, method, seed):
    rows = []
    model.eval()
    with torch.no_grad():
        for stored in samples:
            sample = stored if stored.target.device == device else stored.to(device)
            prediction = float(model(sample).item())
            target = float(sample.target.item())
            rows.append({
                "fold": fold, "scenario": scenario, "method": method, "seed": seed,
                "source": sample.source, "material_id": material_id(sample.source),
                "target": target, "prediction": prediction,
                "absolute_error": abs(target - prediction),
            })
    return rows


def write_rows(path: Path, rows, fields=None):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fields or tuple(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def averaged_prediction_rows(predictions):
    grouped = {}
    for row in predictions:
        key = (int(row["fold"]), row["scenario"], row["method"], row["source"])
        grouped.setdefault(key, []).append(row)
    output = []
    for (fold, scenario, method, source), rows in sorted(grouped.items()):
        target = float(rows[0]["target"])
        prediction = float(np.mean([float(row["prediction"]) for row in rows]))
        output.append({
            "fold": fold, "scenario": scenario, "method": method, "source": source,
            "material_id": rows[0]["material_id"], "target": target,
            "prediction": prediction, "absolute_error": abs(target - prediction),
            "seeds": len(rows),
        })
    return output


def aggregate_results(predictions, run_metadata):
    seed_metrics = []
    keys = sorted({(int(r["fold"]), r["scenario"], r["method"], int(r["seed"]))
                   for r in predictions})
    for key in keys:
        fold, scenario, method, seed = key
        rows = [r for r in predictions if
                (int(r["fold"]), r["scenario"], r["method"], int(r["seed"])) == key]
        seed_metrics.append({"fold": fold, "scenario": scenario, "method": method,
                             "seed": seed, **run_metadata.get(key, {}), **metrics(rows)})

    averaged = averaged_prediction_rows(predictions)
    fold_metrics = []
    fold_keys = sorted({(int(r["fold"]), r["scenario"], r["method"]) for r in averaged})
    for key in fold_keys:
        fold, scenario, method = key
        rows = [r for r in averaged if (int(r["fold"]), r["scenario"], r["method"]) == key]
        fold_metrics.append({"fold": fold, "scenario": scenario, "method": method,
                             **metrics(rows)})

    summary = []
    for scenario, method in sorted({(r["scenario"], r["method"]) for r in fold_metrics}):
        rows = [r for r in fold_metrics if r["scenario"] == scenario and r["method"] == method]
        combined = [r for r in averaged if r["scenario"] == scenario and r["method"] == method]
        entry = {"scenario": scenario, "method": method, "folds": len(rows),
                 "oof_hops": len(combined), "pooled_oof_mae": metrics(combined)["mae"]}
        for name in ("mae", "rmse", "median_ae", "r2", "spearman"):
            values = np.asarray([float(row[name]) for row in rows])
            entry[f"fold_{name}_mean"] = float(np.nanmean(values))
            entry[f"fold_{name}_std"] = (
                float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            )
        summary.append(entry)
    return seed_metrics, averaged, fold_metrics, summary


def paired_fold_effects(fold_metrics):
    comparisons = (
        ("crystal_minus_midpoint", ("dft_path", "crystal"), ("dft_path", "midpoint")),
        ("midpoint_minus_trajectory", ("dft_path", "midpoint"), ("dft_path", "trajectory")),
        ("midpoint_minus_positional", ("dft_path", "midpoint"), ("dft_path", "positional")),
        ("trajectory_minus_positional", ("dft_path", "trajectory"), ("dft_path", "positional")),
        ("affine_minus_delta", ("affine_bvse", "affine_em_bvse"),
         ("bvse_calibrated_delta", "positional")),
    )
    lookup = {(int(r["fold"]), r["scenario"], r["method"]): float(r["mae"])
              for r in fold_metrics}
    rows = []
    for name, left, right in comparisons:
        for fold in sorted({key[0] for key in lookup}):
            left_key, right_key = (fold, *left), (fold, *right)
            if left_key in lookup and right_key in lookup:
                rows.append({
                    "comparison": name, "fold": fold,
                    "left_mae": lookup[left_key], "right_mae": lookup[right_key],
                    "mae_improvement": lookup[left_key] - lookup[right_key],
                })
    summaries = []
    for name in sorted({row["comparison"] for row in rows}):
        values = np.asarray([row["mae_improvement"] for row in rows
                             if row["comparison"] == name])
        summaries.append({
            "comparison": name, "folds": len(values),
            "mean_mae_improvement": float(values.mean()),
            "std_mae_improvement": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "folds_improved": int(np.sum(values > 0)),
            "fraction_folds_improved": float(np.mean(values > 0)),
        })
    return rows, summaries


def main() -> None:
    cli = parser().parse_args()
    if not 0 < cli.validation_fraction < 1:
        raise SystemExit("--validation-fraction must lie strictly between 0 and 1")
    if cli.skip_core and cli.skip_q4:
        raise SystemExit("nothing to run: both --skip-core and --skip-q4 were supplied")
    device = resolve_device(cli.device)
    if device.type == "mps":
        print("warning: CPU is recommended for long repeated-fit runs; MPS remains enabled", flush=True)

    relaxed_args = experiment_args(cli, "relaxed", "none", cli.seeds[0], 0, "prepare")
    prepared_relaxed = prepare_litraj(cli.data_root, "nebDFT2k", relaxed_args)
    relaxed = flatten(prepared_relaxed)
    folds = make_fold_indices(relaxed, cli)

    init = None
    if not cli.skip_q4:
        init_args = experiment_args(cli, "init", "delta", cli.seeds[0], 0, "prepare")
        init = align_source(flatten(prepare_litraj(cli.data_root, "nebDFT2k", init_args)), relaxed)

    split_rows = []
    for fold, indices in enumerate(folds):
        for role, selected in zip(("train", "val", "test"), indices):
            for index in selected:
                item = relaxed[int(index)]
                split_rows.append({"fold": fold, "split": role, "source": item.source,
                                   "material_id": material_id(item.source)})
        counts = {role: len(selected) for role, selected in zip(("train", "val", "test"), indices)}
        group_counts = {role: len({material_id(relaxed[int(i)].source) for i in selected})
                        for role, selected in zip(("train", "val", "test"), indices)}
        print(f"[fold {fold}] hops={counts}; parent_materials={group_counts}", flush=True)
    write_rows(cli.output_dir / "split_assignments.csv", split_rows)

    predictions, validation_rows, run_metadata = [], [], {}
    prediction_path = cli.output_dir / "per_sample_predictions.csv"
    for fold, indices in enumerate(folds):
        jobs = []
        if not cli.skip_core:
            jobs.extend(("dft_path", "relaxed", "none", method, relaxed)
                        for method in cli.core_methods)
        if not cli.skip_q4:
            jobs.append(("bvse_calibrated_delta", "init", "delta", "positional", init))

        for scenario, source, mode, method, items in jobs:
            fold_prepared = prepared_fold(items, indices)
            for seed in cli.seeds:
                args = experiment_args(cli, source, mode, seed, fold, scenario)
                if mode == "delta":
                    slope, intercept = fit_calibration(fold_prepared, args)
                    print(f"[fold {fold} {scenario}] calibration={slope:.6f}*BVSE{intercept:+.6f}",
                          flush=True)
                splits = materialize_prepared(fold_prepared, method, args)
                storage = torch.device("cpu") if device.type == "mps" else device
                splits = move_splits(splits, storage)
                run_name = f"fold{fold}_{scenario}_{method}"
                model, _ = train(splits, args, device, run_name)
                saved = torch.load(checkpoint_path(args, run_name), map_location="cpu",
                                   weights_only=False)
                key = (fold, scenario, method, seed)
                run_metadata[key] = {
                    "best_epoch": int(saved.get("best_epoch", 0)),
                    "trained_epochs": int(saved["epoch"]),
                }
                for history in saved.get("history", []):
                    validation_rows.append({
                        "fold": fold, "scenario": scenario, "method": method, "seed": seed,
                        "epoch": int(history["epoch"]), "train_mae": history["train_mae"],
                        "val_mae": history["val_mae"],
                    })
                predictions.extend(predict(model, splits["test"], device, fold,
                                           scenario, method, seed))
                write_rows(prediction_path, predictions)
                write_rows(cli.output_dir / "validation_curves.csv", validation_rows)
                del model, splits
                release_device_cache(device, run_name)

        if not cli.skip_q4:
            fold_init = prepared_fold(init, indices)
            calibration_args = experiment_args(cli, "init", "delta", cli.seeds[0],
                                               fold, "affine_bvse")
            slope, intercept = fit_calibration(fold_init, calibration_args)
            for seed in cli.seeds:
                key = (fold, "affine_bvse", "affine_em_bvse", seed)
                run_metadata[key] = {"best_epoch": 0, "trained_epochs": 0}
                for item in fold_init["test"]:
                    prediction = slope * float(item.cheap_barrier) + intercept
                    predictions.append({
                        "fold": fold, "scenario": "affine_bvse", "method": "affine_em_bvse",
                        "seed": seed, "source": item.source,
                        "material_id": material_id(item.source), "target": item.target,
                        "prediction": prediction,
                        "absolute_error": abs(item.target - prediction),
                    })
            write_rows(prediction_path, predictions)

    seed_metrics, averaged, fold_metrics, summary = aggregate_results(predictions, run_metadata)
    effects, effect_summary = paired_fold_effects(fold_metrics)
    write_rows(cli.output_dir / "fold_seed_metrics.csv", seed_metrics)
    write_rows(cli.output_dir / "seed_averaged_oof_predictions.csv", averaged)
    write_rows(cli.output_dir / "fold_metrics.csv", fold_metrics)
    write_rows(cli.output_dir / "summary.csv", summary)
    write_rows(cli.output_dir / "paired_fold_effects.csv", effects)
    write_rows(cli.output_dir / "paired_fold_effect_summary.csv", effect_summary)
    print(f"wrote material-disjoint evaluation to {cli.output_dir}", flush=True)


if __name__ == "__main__":
    main()
