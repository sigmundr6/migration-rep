#!/usr/bin/env python3
"""Frozen external-domain BVSE test for the four core pathway representations.

Models are trained and early-stopped exclusively on the prescribed nebDFT2k
BVSE-labelled train/validation partitions.  A separately generated BVSE
manifest is used once as an external test set; its labels never select models.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
from types import SimpleNamespace

import numpy as np
import torch

from litraj_hypergraphs.data import (
    PreparedTrajectory, materialize_prepared, prepare_litraj, read_extxyz,
)
from litraj_hypergraphs.experiment import (
    checkpoint_path, move_splits, release_device_cache, resolve_device, train,
)
from litraj_hypergraphs.geometry import atom_path_geometry, periodic_radius_graph, unwrap_points


METHODS = ("crystal", "midpoint", "trajectory", "positional")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--external-manifest", type=Path, required=True,
                   help="manifest emitted by generate_bvse_trajectories.py")
    p.add_argument("--evaluation-scope", choices=["independent", "regenerated-test"],
                   default="independent", help=(
                       "independent rejects LiTraj material overlap; regenerated-test uses only "
                       "manifest rows belonging to the official LiTraj test split"
                   ))
    p.add_argument("--data-root", type=Path, default=Path("data"),
                   help="root consumed by litraj.data.load_data for nebDFT2k")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
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
                   default=Path("checkpoints/external_bvse_generalization"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/external_bvse_generalization"))
    p.add_argument("--preprocessed-cache-dir", type=Path,
                   default=Path("cache/litraj_preprocessed"))
    p.add_argument("--external-cache-dir", type=Path,
                   default=Path("cache/external_bvse_preprocessed"))
    p.add_argument("--rebuild-preprocessed-cache", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--require-converged", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--max-force", type=float,
                   help="optional maximum recorded final force in eV/A")
    p.add_argument("--min-external-hops", type=int, default=50)
    p.add_argument("--allow-material-overlap", action="store_true",
                   help="allow exact mp-* parent IDs shared with nebDFT2k (not recommended)")
    p.add_argument("--audit-only", action="store_true",
                   help="validate provenance/quality and write the cohort without training")
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=2026)
    p.add_argument("--bootstrap-unit", choices=["material", "hop"], default="material")
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


def experiment_args(cli, seed: int):
    return SimpleNamespace(
        litraj_data=cli.data_root, litraj_dataset="nebDFT2k",
        trajectory_source="init", target_source="em_bvse", manifest=None, trajectory=None,
        neighbour_cutoff=cli.neighbour_cutoff, hyperedge_cutoff=cli.hyperedge_cutoff,
        hyperedge_sigma=cli.hyperedge_sigma,
        bottleneck_neighbours=cli.bottleneck_neighbours, change_top_k=cli.change_top_k,
        num_segments=cli.num_segments, segment_sigma=cli.segment_sigma,
        hidden_dim=cli.hidden_dim, layers=cli.layers, epochs=cli.epochs,
        learning_rate=cli.learning_rate, seed=seed,
        reversal_invariant=cli.reversal_invariant,
        presence_threshold=cli.presence_threshold, auxiliary_mode="none",
        checkpoint_dir=cli.checkpoint_dir, checkpoint_every=cli.checkpoint_every,
        resume=cli.resume, early_stopping_patience=cli.early_stopping_patience,
        early_stopping_min_delta=cli.early_stopping_min_delta,
        mps_memory_report=False, mps_data="stream",
        preprocessed_cache_dir=cli.preprocessed_cache_dir,
        rebuild_preprocessed_cache=cli.rebuild_preprocessed_cache,
    )


def parse_bool(value: str) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def mp_id(value: str) -> str | None:
    match = re.search(r"(mp-\d+)", str(value))
    return match.group(1) if match else None


def manifest_mp_ids(path: Path) -> set[str]:
    """Read explicit or filename-derived MP identifiers without preprocessing graphs."""
    identifiers = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get("source") or row.get("event_id") or Path(row.get("path", "")).name
            if identifier := mp_id(value):
                identifiers.add(identifier)
    return identifiers


def external_cache_path(cli) -> Path:
    manifest = cli.external_manifest.expanduser().resolve()
    stat = manifest.stat()
    metadata = {
        "schema": 1, "manifest": str(manifest), "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "neighbour_cutoff": cli.neighbour_cutoff,
        "require_converged": cli.require_converged, "max_force": cli.max_force,
    }
    digest = hashlib.sha256(repr(sorted(metadata.items())).encode()).hexdigest()[:16]
    return cli.external_cache_dir / f"external_bvse_{digest}.pkl"


def load_external(cli) -> tuple[list[PreparedTrajectory], list[dict], dict]:
    cache = external_cache_path(cli)
    if cache.exists() and not cli.rebuild_preprocessed_cache:
        with cache.open("rb") as handle:
            payload = pickle.load(handle)
        print(f"[external] loaded {len(payload['items'])} trajectories from {cache}", flush=True)
        return payload["items"], payload["rows"], payload["audit"]

    items, accepted_rows = [], []
    rejected = {"not_converged": 0, "force": 0, "invalid_target": 0, "duplicate": 0}
    seen = set()
    missing_diagnostics = 0
    with cli.external_manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "target", "migrating_index"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"external manifest is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, 2):
            if cli.evaluation_scope == "regenerated-test" and row.get("split") != "test":
                continue
            event_id = (row.get("event_id") or Path(row["path"]).stem).strip()
            if event_id in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(event_id)
            try:
                target = float(row["target"])
            except ValueError:
                target = np.nan
            if not np.isfinite(target) or target < 0:
                rejected["invalid_target"] += 1
                continue
            path = Path(row["path"])
            if not path.is_absolute():
                path = cli.external_manifest.parent / path
            sidecar = path.with_name(path.name.replace("_regenerated.xyz", "_metrics.json"))
            diagnostics = {}
            if sidecar.exists():
                with sidecar.open() as sidecar_handle:
                    diagnostics = json.load(sidecar_handle)
            converged = parse_bool(row.get("converged", diagnostics.get("converged", "")))
            if converged is None:
                missing_diagnostics += 1
            if cli.require_converged and converged is not True:
                rejected["not_converged"] += 1
                continue
            raw_force = str(row.get("max_force") or diagnostics.get("max_force", "")).strip()
            force = float(raw_force) if raw_force else np.nan
            if cli.max_force is not None and (not np.isfinite(force) or force > cli.max_force):
                rejected["force"] += 1
                continue
            index = int(row["migrating_index"])
            numbers, frames, cell, index = read_extxyz(path, index)
            frame_graphs = [periodic_radius_graph(frame, cell, cli.neighbour_cutoff)
                            for frame in frames]
            trajectory = unwrap_points(frames[:, index], cell)
            item_source = f"external:{event_id}"
            items.append(PreparedTrajectory(
                numbers=numbers, frames=frames, cell=cell, migrating_index=index,
                target=target, cheap_barrier=None, split="test", source=item_source,
                frame_graphs=frame_graphs,
                path_geometry=atom_path_geometry(frames[0], trajectory, cell),
            ))
            material = (row.get("source") or diagnostics.get("edge_id") or event_id).strip()
            accepted_rows.append({
                "source": item_source, "event_id": event_id,
                "material_id": mp_id(material) or material, "path": str(path.resolve()),
                "target": target, "converged": converged, "max_force": force,
                "n_images": len(frames), "elements": " ".join(map(str, sorted(set(numbers)))),
            })
            if len(items) % 100 == 0:
                print(f"[external] preprocessed {len(items)} accepted trajectories", flush=True)
    if missing_diagnostics and cli.require_converged:
        print(
            "[external] convergence fields are missing; rerun the generator with the same "
            "inputs to refresh its manifest from progress.jsonl", flush=True,
        )
    audit = {
        "manifest": str(cli.external_manifest.resolve()), "accepted_hops": len(items),
        "evaluation_scope": cli.evaluation_scope,
        "accepted_materials": len({row["material_id"] for row in accepted_rows}),
        "rejected": rejected, "rows_missing_convergence_diagnostics": missing_diagnostics,
        "target_min": float(min((item.target for item in items), default=np.nan)),
        "target_max": float(max((item.target for item in items), default=np.nan)),
        "target_mean": float(np.mean([item.target for item in items])) if items else np.nan,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(cache.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump({"items": items, "rows": accepted_rows, "audit": audit}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, cache)
    print(f"[external] saved preprocessing cache to {cache}", flush=True)
    return items, accepted_rows, audit


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
    y = np.asarray([float(row["target"]) for row in rows])
    p = np.asarray([float(row["prediction"]) for row in rows])
    error = y - p
    denominator = np.sum((y - y.mean()) ** 2)
    return {
        "n": len(y), "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "median_ae": float(np.median(np.abs(error))),
        "r2": float(1 - np.sum(error ** 2) / denominator) if denominator else np.nan,
        "spearman": float(np.corrcoef(rank(y), rank(p))[0, 1]) if len(y) > 1 else np.nan,
    }


def write_rows(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def predict(model, samples, metadata, device, method, seed):
    by_source = {row["source"]: row for row in metadata}
    output = []
    model.eval()
    with torch.no_grad():
        for stored in samples:
            sample = stored if stored.target.device == device else stored.to(device)
            prediction = float(model(sample).item())
            target = float(sample.target.item())
            meta = by_source[sample.source]
            output.append({
                "method": method, "seed": seed, "source": sample.source,
                "event_id": meta["event_id"], "material_id": meta["material_id"],
                "target": target, "prediction": prediction,
                "absolute_error": abs(target - prediction),
            })
    return output


def ensemble_rows(predictions):
    grouped = {}
    for row in predictions:
        grouped.setdefault((row["method"], row["source"]), []).append(row)
    output = []
    for (method, source), rows in sorted(grouped.items()):
        prediction = float(np.mean([float(row["prediction"]) for row in rows]))
        target = float(rows[0]["target"])
        output.append({
            "method": method, "source": source, "event_id": rows[0]["event_id"],
            "material_id": rows[0]["material_id"], "target": target,
            "prediction": prediction, "absolute_error": abs(target - prediction),
            "seeds": len(rows),
        })
    return output


def paired_bootstrap(ensemble, cli):
    comparisons = (
        ("crystal_minus_midpoint", "crystal", "midpoint"),
        ("midpoint_minus_positional", "midpoint", "positional"),
        ("trajectory_minus_positional", "trajectory", "positional"),
        ("midpoint_minus_trajectory", "midpoint", "trajectory"),
    )
    lookup = {(row["method"], row["source"]): row for row in ensemble}
    rng, output = np.random.default_rng(cli.bootstrap_seed), []
    for name, left, right in comparisons:
        records = []
        for (method, source), row in lookup.items():
            if method == left and (right, source) in lookup:
                records.append((
                    row["material_id"], float(row["absolute_error"])
                    - float(lookup[(right, source)]["absolute_error"]),
                ))
        if not records:
            continue
        observed = float(np.mean([value for _, value in records]))
        if cli.bootstrap_unit == "hop":
            values = np.asarray([value for _, value in records])
            draws = rng.choice(values, size=(cli.bootstrap_samples, len(values)), replace=True).mean(1)
            units = len(values)
        else:
            clusters = {}
            for material, value in records:
                clusters.setdefault(material, []).append(value)
            names = list(clusters)
            draws = []
            for _ in range(cli.bootstrap_samples):
                sampled = rng.choice(names, size=len(names), replace=True)
                values = [value for material in sampled for value in clusters[material]]
                draws.append(float(np.mean(values)))
            draws, units = np.asarray(draws), len(names)
        output.append({
            "comparison": name, "n_hops": len(records), "bootstrap_unit": cli.bootstrap_unit,
            "n_units": units, "mean_paired_improvement": observed,
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "probability_improvement_gt_zero": float(np.mean(draws > 0)),
        })
    return output


def main() -> None:
    cli = parser().parse_args()
    if cli.min_external_hops < 1:
        raise SystemExit("--min-external-hops must be positive")
    device = resolve_device(cli.device)
    if device.type == "mps":
        print("warning: CPU is recommended for long repeated-fit runs; MPS remains enabled", flush=True)

    declared_external_ids = manifest_mp_ids(cli.external_manifest)
    index_path = cli.data_root / "nebDFT2k" / "nebDFT2k_index.csv"
    indexed_litraj_ids = set()
    if index_path.exists():
        with index_path.open(newline="") as handle:
            indexed_litraj_ids = {
                identifier for row in csv.DictReader(handle)
                if (identifier := mp_id(row.get("edge_id", "")))
            }
    declared_overlap = sorted(indexed_litraj_ids & declared_external_ids)
    if (cli.evaluation_scope == "independent" and declared_overlap
            and not cli.allow_material_overlap):
        raise SystemExit(
            f"external manifest shares {len(declared_overlap)} exact mp-* parent IDs with "
            f"nebDFT2k ({declared_overlap[:5]}). This is not an external-domain cohort; "
            f"do not use --allow-material-overlap for an external-generalization claim."
        )
    if cli.evaluation_scope == "regenerated-test":
        if not index_path.exists():
            raise SystemExit("regenerated-test scope requires nebDFT2k_index.csv")
        with index_path.open(newline="") as handle:
            official_splits = {row["edge_id"]: row["_split"] for row in csv.DictReader(handle)}
        selected, invalid = 0, []
        with cli.external_manifest.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") != "test":
                    continue
                selected += 1
                event_id = row.get("event_id") or Path(row["path"]).name.replace(
                    "_regenerated.xyz", ""
                )
                if official_splits.get(event_id) != "test":
                    invalid.append(event_id)
        if not selected or invalid:
            raise SystemExit(
                f"regenerated-test cohort validation failed: selected={selected}, "
                f"non-test-or-unknown={invalid[:5]}"
            )
    template = experiment_args(cli, cli.seeds[0])
    # Geometry preprocessing is identical to the existing init/em_dft cache,
    # which already retains cheap_barrier. Reuse it and switch only the label.
    template.target_source = "em_dft"
    cached_litraj = prepare_litraj(cli.data_root, "nebDFT2k", template)
    if any(item.cheap_barrier is None for values in cached_litraj.values() for item in values):
        raise SystemExit("cached LiTraj inputs are missing em_bvse labels")
    litraj = {
        split: [replace(item, target=float(item.cheap_barrier)) for item in values]
        for split, values in cached_litraj.items()
    }
    litraj_ids = {mp_id(item.source) for values in litraj.values() for item in values}
    external, external_metadata, audit = load_external(cli)
    if len(external) < cli.min_external_hops:
        raise SystemExit(
            f"only {len(external)} external hops passed quality filters; "
            f"need at least {cli.min_external_hops}"
        )
    external_ids = {mp_id(row["material_id"]) for row in external_metadata}
    overlap = sorted((litraj_ids & external_ids) - {None})
    train_targets = np.asarray([item.target for item in litraj["train"]], dtype=float)
    external_targets = np.asarray([item.target for item in external], dtype=float)
    train_elements = {int(number) for item in litraj["train"] for number in item.numbers}
    external_elements = {int(number) for item in external for number in item.numbers}
    audit.update({
        "litraj_training_hops": len(litraj["train"]),
        "litraj_validation_hops": len(litraj["val"]),
        "litraj_training_target_mean": float(train_targets.mean()),
        "litraj_training_target_std": float(train_targets.std()),
        "external_target_std": float(external_targets.std()),
        "external_materials_with_mp_id": len(external_ids - {None}),
        "external_atomic_numbers": sorted(external_elements),
        "atomic_numbers_absent_from_litraj_training": sorted(external_elements - train_elements),
    })
    audit["exact_mp_id_overlap_count"] = len(overlap)
    audit["exact_mp_id_overlap"] = overlap
    if (cli.evaluation_scope == "independent" and overlap
            and not cli.allow_material_overlap):
        raise SystemExit(
            f"external set shares {len(overlap)} exact mp-* parent IDs with nebDFT2k "
            f"({overlap[:5]}); use independent inputs or explicitly pass --allow-material-overlap"
        )
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    with (cli.output_dir / "dataset_audit.json").open("w") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True, allow_nan=True)
    write_rows(cli.output_dir / "external_manifest_accepted.csv", external_metadata)
    print(f"[external] audit={audit}", flush=True)
    if cli.audit_only:
        print(f"external dataset audit passed; wrote {cli.output_dir}", flush=True)
        return

    predictions, histories, run_metadata = [], [], {}
    for seed in cli.seeds:
        args = experiment_args(cli, seed)
        for method in cli.methods:
            # The external test objects are materialised separately and never enter train().
            training_prepared = {"train": litraj["train"], "val": litraj["val"], "test": []}
            training = materialize_prepared(training_prepared, method, args)
            external_samples = materialize_prepared(
                {"train": [], "val": [], "test": external}, method, args
            )["test"]
            storage = torch.device("cpu") if device.type == "mps" else device
            training = move_splits(training, storage)
            external_samples = [sample.to(storage) for sample in external_samples]
            run_name = f"litraj_bvse_to_external_bvse_{method}"
            model, _ = train(training, args, device, run_name)
            saved = torch.load(checkpoint_path(args, run_name), map_location="cpu",
                               weights_only=False)
            run_metadata[(method, seed)] = {
                "best_epoch": int(saved.get("best_epoch", 0)),
                "trained_epochs": int(saved["epoch"]),
            }
            for row in saved.get("history", []):
                histories.append({
                    "method": method, "seed": seed, "epoch": int(row["epoch"]),
                    "train_mae": row["train_mae"], "validation_mae": row["val_mae"],
                })
            predictions.extend(predict(model, external_samples, external_metadata,
                                       device, method, seed))
            write_rows(cli.output_dir / "per_sample_predictions.csv", predictions)
            write_rows(cli.output_dir / "validation_curves.csv", histories)
            del model, training, external_samples
            release_device_cache(device, run_name)

    seed_metrics = []
    for method, seed in sorted({(r["method"], int(r["seed"])) for r in predictions}):
        rows = [r for r in predictions if r["method"] == method and int(r["seed"]) == seed]
        seed_metrics.append({"method": method, "seed": seed,
                             **run_metadata[(method, seed)], **metrics(rows)})
    ensemble = ensemble_rows(predictions)
    ensemble_metrics = []
    for method in sorted({row["method"] for row in ensemble}):
        rows = [row for row in ensemble if row["method"] == method]
        ensemble_metrics.append({"method": method, "seeds": len(cli.seeds), **metrics(rows)})
    bootstrap = paired_bootstrap(ensemble, cli)
    write_rows(cli.output_dir / "seed_metrics.csv", seed_metrics)
    write_rows(cli.output_dir / "ensemble_predictions.csv", ensemble)
    write_rows(cli.output_dir / "ensemble_metrics.csv", ensemble_metrics)
    write_rows(cli.output_dir / "paired_bootstrap.csv", bootstrap)
    print(f"wrote frozen external BVSE evaluation to {cli.output_dir}", flush=True)


if __name__ == "__main__":
    main()
