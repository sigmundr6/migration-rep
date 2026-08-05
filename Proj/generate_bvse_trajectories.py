#!/usr/bin/env python3
"""Generate a manifest dataset of inexpensive BVSE-NEB Li trajectories.

This is a script form of ``LiTraj-main/notebooks/bvse_neb.ipynb``. Input is a
CIF or any ASE-readable file containing one or more Li-bearing structures.
Splits are assigned per source material, never per hop, to avoid leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from ase.io import iread, write
from litraj_hypergraphs.bvse import optimize_bvse_band


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "structure"


def material_key(atoms, input_path: Path, index: int) -> str:
    for key in ("material_id", "mp_id"):
        if atoms.info.get(key):
            return str(atoms.info[key])
    # Formula is not a material identifier: polymorphs and independent input
    # structures can share it.  Keep the fallback unique to the source file.
    digest = hashlib.sha256(str(input_path.resolve()).encode()).hexdigest()[:10]
    return f"{input_path.stem}-{digest}-{index}"


def split_for(key: str, seed: int, train_fraction: float, val_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_fraction:
        return "train"
    if value < train_fraction + val_fraction:
        return "val"
    return "test"


def load_completed(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    completed = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
                completed[record["id"]] = record
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid progress record") from exc
    return completed


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def write_manifest(path: Path, records: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = [
        "path", "target", "split", "migrating_index", "source", "event_id",
        "converged", "max_force", "percolation_dimension", "cutoff",
        "n_images", "requested_fmax", "optimizer_steps",
    ]
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records.values(), key=lambda item: item["id"]):
            row = {key: record.get(key, "") for key in fields}
            row["event_id"] = record["id"]
            writer.writerow(row)
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("inputs", nargs="+", type=Path, help="ASE-readable CIF/XYZ inputs")
    result.add_argument("--output-dir", type=Path, default=Path("data/bvse_generated"))
    result.add_argument("--manifest", type=Path, default=Path("data/bvse_generated/manifest.csv"))
    result.add_argument("--n-images", type=int, default=5)
    result.add_argument("--upper-bound", type=float, default=8.0)
    result.add_argument("--framework-clearance", type=float, default=0.5)
    result.add_argument("--fmax", type=float, default=0.1)
    result.add_argument("--steps", type=int, default=100)
    result.add_argument("--max-structures", type=int)
    result.add_argument("--max-edges-per-structure", type=int)
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--train-fraction", type=float, default=0.8)
    result.add_argument("--val-fraction", type=float, default=0.1)
    result.add_argument(
        "--remove-centroid-x", action="store_true",
        help="remove LiTraj X markers before edge enumeration; use only if inputs are centroid structures",
    )
    result.add_argument(
        "--skip-errors", action="store_true", help="report failures and continue with later structures"
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if args.n_images < 2:
        raise SystemExit("--n-images must be at least 2")
    if not (0 < args.train_fraction < 1 and 0 <= args.val_fraction < 1
            and args.train_fraction + args.val_fraction < 1):
        raise SystemExit("split fractions must leave nonzero train and test ranges")
    try:
        from ions import Decorator
        from ions.tools import Percolator
        from ions.utils import collect_bvse_params
    except ImportError as exc:
        raise SystemExit(
            "BVSE trajectory generation requires the ions package used by the "
            "LiTraj notebook. Install the compatible API with: "
            "python3 -m pip install 'ions==0.4.1'"
        ) from exc

    from litraj_hypergraphs.data import identify_migrating_li

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.jsonl"
    completed = load_completed(progress_path)
    processed_materials: set[str] = set()
    structure_count = success_count = failure_count = 0

    for input_path in args.inputs:
        for input_index, original in enumerate(iread(str(input_path), index=":")):
            key = material_key(original, input_path, input_index)
            if key in processed_materials:
                continue
            processed_materials.add(key)
            if args.max_structures is not None and structure_count >= args.max_structures:
                break
            structure_count += 1
            try:
                atoms = original.copy()
                dummy = np.flatnonzero(np.asarray(atoms.numbers) == 0)
                if len(dummy):
                    if not args.remove_centroid_x:
                        raise ValueError(
                            "structure contains X centroid marker(s); pass --remove-centroid-x "
                            "to enumerate fresh paths from the remaining host"
                        )
                    del atoms[dummy.tolist()]
                if not np.any(np.asarray(atoms.numbers) == 3):
                    raise ValueError("structure contains no Li")

                decorated = Decorator().decorate(atoms)
                decorated = collect_bvse_params(decorated, "Li", 1)
                percolator = Percolator(decorated, 3, args.upper_bound)
                cutoff, dimension = percolator.mincut_maxdim(tr=args.framework_clearance)
                edges, _ = percolator.unique_edges(cutoff, args.framework_clearance)
                if args.max_edges_per_structure is not None:
                    edges = edges[:args.max_edges_per_structure]

                for edge_index, edge in enumerate(edges):
                    event_id = f"{safe_name(key)}__s{input_index:05d}__e{edge_index:04d}"
                    if event_id in completed and completed[event_id].get("status") == "complete":
                        continue
                    initial = edge.superedge(args.upper_bound).interpolate(n_images=args.n_images)
                    images = [image.copy() for image in initial]
                    result = optimize_bvse_band(
                        images, fmax=args.fmax, steps=args.steps, distort=True
                    )
                    barrier = result["barrier"]
                    max_force = result["max_force"]
                    numbers = np.asarray(images[0].numbers)
                    frames = np.stack([np.asarray(image.positions) for image in images])
                    migrating_index = identify_migrating_li(numbers, frames, np.asarray(images[0].cell))
                    trajectory_path = args.output_dir / f"{event_id}.xyz"
                    output_images = [image.copy() for image in images]
                    for image_number, image in enumerate(output_images):
                        image.calc = None
                        image.info.update({
                            "event_id": event_id, "material_key": key,
                            "image": image_number, "em_bvse": barrier,
                        })
                    write(trajectory_path, output_images, format="extxyz")
                    record = {
                        "id": event_id, "status": "complete",
                        "path": str(trajectory_path.resolve()), "target": barrier,
                        "split": split_for(
                            key, args.seed, args.train_fraction, args.val_fraction
                        ),
                        "migrating_index": migrating_index, "source": key,
                        "max_force": max_force, "converged": result["converged"],
                        "percolation_dimension": int(dimension), "cutoff": float(cutoff),
                        "n_images": args.n_images, "requested_fmax": args.fmax,
                        "optimizer_steps": args.steps,
                    }
                    append_record(progress_path, record)
                    completed[event_id] = record
                    success_count += 1
                    print(
                        f"{event_id}: barrier={barrier:.4f} eV  "
                        f"fmax={max_force:.3f} eV/A  split={record['split']}", flush=True
                    )
            except Exception as exc:
                failure_count += 1
                print(f"ERROR {key}: {type(exc).__name__}: {exc}", flush=True)
                if not args.skip_errors:
                    raise

    successful = {key: value for key, value in completed.items()
                  if value.get("status") == "complete"}
    write_manifest(args.manifest, successful)
    print(
        f"wrote {len(successful)} trajectories to {args.manifest} "
        f"({success_count} new; {failure_count} failed structures)", flush=True
    )


if __name__ == "__main__":
    main()
