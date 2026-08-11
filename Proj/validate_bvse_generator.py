#!/usr/bin/env python3
"""Validate regenerated BVSE-NEB bands against LiTraj nebDFT2k BVSE references."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read, write

from litraj_hypergraphs.bvse import evaluate_bvse_band, optimize_bvse_band
from litraj_hypergraphs.data import identify_migrating_li
from litraj_hypergraphs.geometry import minimum_image_vector, unwrap_points


REQUIRED_BVSE_ARRAYS = {
    "oxi_states", "r0", "d0", "alpha", "r_min", "n2", "rc2", "mask", "freezed"
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-root", type=Path, default=Path("data"))
    result.add_argument(
        "--split", choices=["train", "val", "test", "all", "proportional"], default="test",
        help="proportional samples every official split using its dataset ratio",
    )
    result.add_argument("--samples", type=int, default=5)
    result.add_argument("--edge-id", action="append", help="validate exact edge; repeatable")
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--fmax", type=float, default=0.1)
    result.add_argument("--steps", type=int, default=100)
    result.add_argument("--no-distort", action="store_true")
    result.add_argument("--output-dir", type=Path, default=Path("validation/bvse_generator"))
    result.add_argument("--write-trajectories", action="store_true")
    result.add_argument("--barrier-tolerance", type=float, default=0.10)
    result.add_argument("--li-path-tolerance", type=float, default=0.50)
    result.add_argument("--strict", action="store_true", help="exit nonzero if tolerances fail")
    return result


def fresh_linear_band(reference):
    """Build a new band from reference endpoints without using intermediate images."""
    count = len(reference)
    start = reference[0]
    end = reference[-1]
    moving = identify_migrating_li(
        np.asarray(start.numbers),
        np.stack([np.asarray(image.positions) for image in reference]),
        np.asarray(start.cell),
    )
    displacement = np.stack([
        minimum_image_vector(target - source, np.asarray(start.cell))
        for source, target in zip(start.positions, end.positions)
    ])
    # Preserve the actual unwrapped endpoint displacement for the mobile Li.
    official_li = unwrap_points(
        np.stack([image.positions[moving] for image in reference]), np.asarray(start.cell)
    )
    displacement[moving] = official_li[-1] - official_li[0]
    images = []
    for fraction in np.linspace(0.0, 1.0, count):
        image = start.copy()
        image.calc = None
        image.positions = start.positions + fraction * displacement
        images.append(image)
    return images, moving


def periodic_errors(reference, generated, moving: int):
    cell = np.asarray(reference[0].cell)
    all_errors, li_errors, framework_errors = [], [], []
    framework = np.arange(len(reference[0])) != moving
    for expected, actual in zip(reference, generated):
        distances = np.asarray([
            np.linalg.norm(minimum_image_vector(b - a, cell))
            for a, b in zip(expected.positions, actual.positions)
        ])
        all_errors.extend(distances.tolist())
        li_errors.append(float(distances[moving]))
        framework_errors.extend(distances[framework].tolist())
    return {
        "geometry_mae": float(np.mean(all_errors)),
        "li_path_mae": float(np.mean(li_errors)),
        "framework_mae": float(np.mean(framework_errors)),
    }


def normalized_profile_rmse(first: np.ndarray, second: np.ndarray) -> float:
    first = first - first.min()
    second = second - second.min()
    return float(np.sqrt(np.mean((first - second) ** 2)))


def select_rows(table: pd.DataFrame, args) -> pd.DataFrame:
    if args.edge_id:
        selected = table[table.edge_id.astype(str).isin(args.edge_id)]
        missing = sorted(set(args.edge_id) - set(selected.edge_id.astype(str)))
        if missing:
            raise SystemExit(f"unknown edge ids: {missing}")
        return selected
    if args.split == "proportional":
        counts = table._split.value_counts()
        raw = args.samples * counts / len(table)
        allocated = np.floor(raw).astype(int)
        for split in (raw - allocated).sort_values(ascending=False).index:
            if allocated.sum() >= args.samples:
                break
            allocated[split] += 1
        return pd.concat([
            table[table._split == split].sample(
                n=int(allocated[split]), random_state=args.seed
            )
            for split in ("train", "val", "test")
        ])
    eligible = table if args.split == "all" else table[table._split == args.split]
    if args.samples > len(eligible):
        raise SystemExit(f"requested {args.samples} samples but only {len(eligible)} are eligible")
    return eligible.sample(n=args.samples, random_state=args.seed)


def main() -> None:
    args = parser().parse_args()
    dataset = args.data_root / "nebDFT2k"
    table = pd.read_csv(dataset / "nebDFT2k_index.csv")
    required_columns = {"edge_id", "em_bvse", "_split"}
    if missing := required_columns - set(table.columns):
        raise SystemExit(f"index lacks required columns: {sorted(missing)}")
    rows = select_rows(table, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    print("edge_id                         published  stored-eval  regenerated  |dE|    Li-MAE  fmax  pass")
    print("------------------------------  ---------  -----------  -----------  ------  ------  -----  ----")
    for _, row in rows.iterrows():
        edge_id = str(row.edge_id)
        record_path = args.output_dir / f"{edge_id}_metrics.json"
        trajectory_path = args.output_dir / f"{edge_id}_regenerated.xyz"
        if args.write_trajectories and record_path.exists() and trajectory_path.exists():
            record = json.loads(record_path.read_text())
            records.append(record)
            print(f"{edge_id:<30}  {record['published_barrier']:>9.4f}  "
                  f"{record['stored_geometry_barrier']:>11.4f}  "
                  f"{record['regenerated_barrier']:>11.4f}  {record['barrier_error']:>6.4f}  "
                  f"{record['li_path_mae']:>6.3f}  {record['max_force']:>5.2f}  cached", flush=True)
            continue
        reference = read(dataset / f"{edge_id}_init.xyz", index=":")
        if len(reference) < 2:
            raise ValueError(f"{edge_id}: reference has fewer than two images")
        missing_arrays = REQUIRED_BVSE_ARRAYS - set(reference[0].arrays)
        if missing_arrays:
            raise ValueError(f"{edge_id}: reference lacks BVSE arrays {sorted(missing_arrays)}")
        official = [image.copy() for image in reference]
        for image in official:
            image.calc = None
        stored = evaluate_bvse_band(official)
        generated, moving = fresh_linear_band(reference)
        result = optimize_bvse_band(
            generated, fmax=args.fmax, steps=args.steps, distort=not args.no_distort
        )
        errors = periodic_errors(reference, generated, moving)
        published = float(row.em_bvse)
        barrier_error = abs(result["barrier"] - published)
        stored_error = abs(stored["barrier"] - published)
        passed = barrier_error <= args.barrier_tolerance and errors["li_path_mae"] <= args.li_path_tolerance
        record = {
            "edge_id": edge_id, "split": str(row._split), "images": len(reference),
            "published_barrier": published, "stored_geometry_barrier": stored["barrier"],
            "stored_barrier_error": stored_error, "regenerated_barrier": result["barrier"],
            "barrier_error": barrier_error, **errors,
            "profile_rmse": normalized_profile_rmse(stored["profile"], result["profile"]),
            "max_force": result["max_force"], "converged": result["converged"],
            "passed": passed,
        }
        records.append(record)
        if args.write_trajectories:
            write(trajectory_path, generated, format="extxyz")
            record["path"] = str(trajectory_path.resolve())
            record["target"] = result["barrier"]
            record["migrating_index"] = moving
            record_path.write_text(json.dumps(record, sort_keys=True) + "\n")
        print(f"{edge_id:<30}  {published:>9.4f}  {stored['barrier']:>11.4f}  "
              f"{result['barrier']:>11.4f}  {barrier_error:>6.4f}  "
              f"{errors['li_path_mae']:>6.3f}  {result['max_force']:>5.2f}  "
              f"{'yes' if passed else 'NO':>4}", flush=True)

    csv_path = args.output_dir / "per_edge.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if args.write_trajectories:
        manifest_path = args.output_dir / "manifest.csv"
        with manifest_path.open("w", newline="") as handle:
            fields = ["path", "target", "split", "migrating_index"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: record[key] for key in fields} for record in records)
    numeric = ("stored_barrier_error", "barrier_error", "geometry_mae", "li_path_mae",
               "framework_mae", "profile_rmse", "max_force")
    summary = {
        "samples": len(records), "passed": sum(record["passed"] for record in records),
        "converged": sum(record["converged"] for record in records),
        **{f"mean_{key}": float(np.mean([record[key] for record in records])) for key in numeric},
        **{f"max_{key}": float(np.max([record[key] for record in records])) for key in numeric},
        "barrier_tolerance": args.barrier_tolerance,
        "li_path_tolerance": args.li_path_tolerance,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nresults: {csv_path}\nsummary: {summary_path}\n{json.dumps(summary, indent=2)}")
    if args.strict and summary["passed"] != summary["samples"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
