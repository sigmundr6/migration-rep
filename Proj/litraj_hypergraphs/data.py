"""Input adapters and a small self-contained synthetic smoke-test dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import pickle
import numpy as np

from .candidates import HypergraphSample, build_candidate_sample
from .geometry import atom_path_geometry, periodic_distances, periodic_radius_graph, unwrap_points


@dataclass
class PreparedTrajectory:
    numbers: np.ndarray
    frames: np.ndarray
    cell: np.ndarray
    migrating_index: int
    target: float
    cheap_barrier: float | None
    split: str
    source: str
    frame_graphs: list[tuple[np.ndarray, np.ndarray]]
    path_geometry: tuple[np.ndarray, np.ndarray]


def identify_migrating_li(numbers: np.ndarray, frames: np.ndarray, cell: np.ndarray) -> int:
    li_indices = np.flatnonzero(np.asarray(numbers) == 3)
    if not len(li_indices):
        raise ValueError("structure contains no Li atom")
    displacements = [
        np.linalg.norm(unwrap_points(frames[:, index], cell)[-1] - unwrap_points(frames[:, index], cell)[0])
        for index in li_indices
    ]
    return int(li_indices[int(np.argmax(displacements))])


def read_extxyz(path: Path, migrating_index: int | None = None):
    try:
        from ase.io import read
    except ImportError as exc:
        raise SystemExit("trajectory input requires ASE: pip install ase") from exc
    atoms = read(str(path), index=":")
    if len(atoms) < 2:
        raise ValueError(f"{path}: expected at least two trajectory images")
    numbers = np.asarray(atoms[0].numbers)
    cell = np.asarray(atoms[0].cell)
    frames = np.stack([np.asarray(image.positions) for image in atoms])
    index = identify_migrating_li(numbers, frames, cell) if migrating_index is None else migrating_index
    if numbers[index] != 3:
        raise ValueError(f"{path}: atom {index} is not Li")
    return numbers, frames, cell, index


def build_from_arrays(candidate: str, numbers, frames, cell, index, target, source, args):
    return build_candidate_sample(
        candidate, numbers, frames, cell, index, target,
        neighbour_cutoff=args.neighbour_cutoff,
        hyperedge_cutoff=args.hyperedge_cutoff,
        hyperedge_sigma=args.hyperedge_sigma,
        bottleneck_neighbours=args.bottleneck_neighbours,
        change_top_k=args.change_top_k,
        num_segments=args.num_segments,
        segment_sigma=args.segment_sigma,
        random_seed=args.seed,
        source=source,
    )


def prepare_manifest(path: Path, args) -> dict[str, list[PreparedTrajectory]]:
    splits: dict[str, list[PreparedTrajectory]] = {"train": [], "val": [], "test": []}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"path", "target", "split"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, 1):
            split = row["split"].strip().lower()
            if split not in splits:
                raise ValueError(f"invalid split {split!r}")
            trajectory_path = Path(row["path"])
            if not trajectory_path.is_absolute():
                trajectory_path = path.parent / trajectory_path
            raw_index = (row.get("migrating_index") or "").strip()
            numbers, frames, cell, index = read_extxyz(
                trajectory_path, int(raw_index) if raw_index else None
            )
            frame_graphs = [
                periodic_radius_graph(frame, cell, args.neighbour_cutoff) for frame in frames
            ]
            trajectory = unwrap_points(frames[:, index], cell)
            path_geometry = atom_path_geometry(frames[0], trajectory, cell)
            splits[split].append(PreparedTrajectory(
                numbers=numbers, frames=frames, cell=cell, migrating_index=index,
                target=float(row["target"]), split=split, source=str(trajectory_path),
                cheap_barrier=(float(row["em_bvse"]) if row.get("em_bvse") else None),
                frame_graphs=frame_graphs, path_geometry=path_geometry,
            ))
            if row_number % 100 == 0:
                print(f"[shared] preprocessed {row_number} manifest trajectories", flush=True)
    if not all(splits.values()):
        raise ValueError(f"manifest must contain every split; got { {k: len(v) for k, v in splits.items()} }")
    return splits


def load_manifest(path: Path, candidate: str, args) -> dict[str, list[HypergraphSample]]:
    return materialize_prepared(prepare_manifest(path, args), candidate, args)


def prepare_litraj(root: Path, dataset: str, args) -> dict[str, list[PreparedTrajectory]]:
    trajectory_source = getattr(args, "trajectory_source", "relaxed")
    requested_target = getattr(args, "target_source", "auto")
    cache_directory = getattr(args, "preprocessed_cache_dir", None)
    cache_path = None
    cache_metadata = {
        "schema": 1,
        "dataset": dataset,
        "data_root": str(Path(root).expanduser().resolve()),
        "trajectory_source": trajectory_source,
        "target_source": requested_target,
        "neighbour_cutoff": float(args.neighbour_cutoff),
    }
    if cache_directory is not None:
        digest = hashlib.sha256(repr(sorted(cache_metadata.items())).encode()).hexdigest()[:16]
        cache_path = Path(cache_directory) / f"{dataset}_{trajectory_source}_{requested_target}_{digest}.pkl"
        if cache_path.exists() and not getattr(args, "rebuild_preprocessed_cache", False):
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if cached.get("metadata") == cache_metadata:
                splits = cached["splits"]
                print(
                    f"[shared] loaded {sum(map(len, splits.values()))} preprocessed "
                    f"trajectories from {cache_path}", flush=True,
                )
                return splits

    try:
        from litraj.data import load_data
    except ImportError as exc:
        raise SystemExit("official dataset loading requires: pip install litraj") from exc
    if dataset != "nebDFT2k":
        raise ValueError(
            "nebBVSE122k is distributed as centroid structures rather than full "
            "NEB frame sequences in the documented LiTraj interface. These "
            "trajectory-hyperedge experiments require nebDFT2k or an explicit "
            "multi-frame --manifest."
        )
    table = load_data(dataset, str(root))
    if not hasattr(table, "columns") or not hasattr(table, "iterrows"):
        raise TypeError(
            "this installed litraj version returned an unsupported load_data result; "
            "expected the nebDFT2k pandas table documented by LiTraj"
        )
    frame_key = f"trajectory_{trajectory_source}"
    if frame_key not in table.columns:
        raise KeyError(
            f"LiTraj table lacks {frame_key!r}; available columns are {list(table.columns)}"
        )
    if requested_target == "auto":
        target_candidates = ("em_bvse", "em") if trajectory_source == "init" else ("em_dft", "em")
    else:
        target_candidates = (requested_target,)
    target_key = next((name for name in target_candidates if name in table.columns), None)
    if target_key is None:
        raise KeyError(
            f"LiTraj table has no migration-barrier column; available columns are {list(table.columns)}"
        )
    if "_split" not in table.columns:
        raise KeyError("LiTraj table lacks the required '_split' benchmark column")
    splits: dict[str, list[PreparedTrajectory]] = {"train": [], "val": [], "test": []}
    accepted = 0
    for row_number, (_, row) in enumerate(table.iterrows(), start=1):
        split = str(row["_split"]).lower().strip()
        if split not in splits:
            continue
        images = row[frame_key]
        if len(images) < 2:
            continue
        numbers = np.asarray(images[0].numbers)
        cell = np.asarray(images[0].cell)
        if any(not np.array_equal(np.asarray(image.numbers), numbers) for image in images):
            continue
        frames = np.stack([np.asarray(image.positions) for image in images])
        target = float(row[target_key])
        if not np.isfinite(target):
            continue
        try:
            index = identify_migrating_li(numbers, frames, cell)
            frame_graphs = [
                periodic_radius_graph(frame, cell, args.neighbour_cutoff) for frame in frames
            ]
            path = unwrap_points(frames[:, index], cell)
            path_geometry = atom_path_geometry(frames[0], path, cell)
            splits[split].append(PreparedTrajectory(
                numbers=numbers,
                frames=frames,
                cell=cell,
                migrating_index=index,
                target=target,
                cheap_barrier=(
                    float(row["em_bvse"])
                    if "em_bvse" in row.index and np.isfinite(float(row["em_bvse"])) else None
                ),
                split=split,
                source=f"{dataset}:{row.get('edge_id', 'unknown')}",
                frame_graphs=frame_graphs,
                path_geometry=path_geometry,
            ))
            accepted += 1
            if accepted % 100 == 0:
                print(
                    f"[shared] preprocessed {accepted}/{len(table)} trajectories",
                    flush=True,
                )
        except (ValueError, IndexError):
            continue
    if not all(splits.values()):
        raise ValueError(f"LiTraj did not yield every split; got { {k: len(v) for k, v in splits.items()} }")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump({"metadata": cache_metadata, "splits": splits}, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, cache_path)
        print(f"[shared] saved preprocessed trajectories to {cache_path}", flush=True)
    return splits


def materialize_prepared(
    prepared: dict[str, list[PreparedTrajectory]], candidate: str, args
) -> dict[str, list[HypergraphSample]]:
    return {
        split: [
            build_candidate_sample(
                candidate,
                item.numbers,
                item.frames,
                item.cell,
                item.migrating_index,
                item.target,
                neighbour_cutoff=args.neighbour_cutoff,
                hyperedge_cutoff=args.hyperedge_cutoff,
                hyperedge_sigma=args.hyperedge_sigma,
                bottleneck_neighbours=args.bottleneck_neighbours,
                change_top_k=args.change_top_k,
                num_segments=args.num_segments,
                segment_sigma=args.segment_sigma,
                random_seed=args.seed,
                source=item.source,
                frame_graphs=item.frame_graphs,
                path_geometry=item.path_geometry,
                cheap_barrier=item.cheap_barrier,
            ) for item in items
        ]
        for split, items in prepared.items()
    }


def load_litraj(root: Path, dataset: str, candidate: str, args):
    return materialize_prepared(prepare_litraj(root, dataset, args), candidate, args)


def synthetic_dataset(candidate: str, count: int, seed: int, args):
    rng = np.random.default_rng(seed)
    samples = []
    cell = np.diag([10.0, 8.0, 8.0])
    framework = np.asarray([
        [3.0, 2.7, 4.0], [3.0, 5.3, 4.0], [5.0, 3.0, 4.0],
        [5.0, 5.0, 4.0], [7.0, 2.7, 4.0], [7.0, 5.3, 4.0],
        [5.0, 4.0, 2.3], [5.0, 4.0, 5.7],
    ])
    numbers = np.asarray([3, 8, 8, 8, 8, 15, 15, 3, 14])
    for sample_index in range(count):
        fixed = framework + rng.normal(0, 0.18, framework.shape)
        bend = rng.uniform(-1.2, 1.2)
        frames = []
        for step, u in enumerate(np.linspace(0, 1, 7)):
            li = np.asarray([1.2 + 7.6 * u, 4 + bend * np.sin(np.pi * u), 4.0])
            relaxation = 0.08 * np.sin(np.pi * u) * np.sign(fixed - li)
            frames.append(np.vstack([li, fixed + relaxation]))
        frames = np.asarray(frames)
        closest = min(
            periodic_distances(frame[1:], frame[0], cell).min() for frame in frames
        )
        barrier = 0.15 + 0.22 / max(closest, 0.5) + 0.04 * abs(bend) + rng.normal(0, 0.008)
        samples.append(build_from_arrays(
            candidate, numbers, frames, cell, 0, barrier, f"synthetic-{sample_index}", args
        ))
    rng.shuffle(samples)
    train_end, val_end = int(0.7 * count), int(0.85 * count)
    return {"train": samples[:train_end], "val": samples[train_end:val_end], "test": samples[val_end:]}
