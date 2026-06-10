#!/usr/bin/env python3
"""Representation proof of concept for the revised pathway-aware LiTraj proposal.

This script tests the *representation question* rather than claiming to replace
DFT-NEB.  A migration sample is treated as (crystal X, trajectory T, barrier y)
and the underlying sample/target is held fixed while only the representation of
T changes.

Compared representations
------------------------
1. crystal      : periodic crystal graph only;
2. midpoint     : same crystal encoder + a local migration marker at the hop midpoint;
3. trajectory   : same crystal encoder + a trajectory-derived pathway hyperedge;
4. randomized   : same as (3), but atom-to-pathway memberships are permuted.

The ordinary crystal graph is actually used by all models.  The trajectory
hyperedge is therefore an additional coarse-graining operation over GNN node
embeddings, not a replacement for atomic message passing.

Without arguments the script runs a small synthetic controlled experiment in
which different curved migration paths share the same endpoints/midpoint.  The
synthetic barrier depends on the environment along the *whole* path, so this is
only a sanity check that the representation comparison can detect a known
trajectory-level signal.

For a real LiTraj-style experiment, provide a CSV manifest with columns:

    path,target,split[,migrating_index]

where ``path`` is a multi-frame extended-XYZ trajectory readable by ASE,
``target`` is the migration barrier in eV, and ``split`` is train/val/test.
This explicit manifest avoids assuming an undocumented LiTraj on-disk schema.

Examples
--------
Synthetic representation sanity check::

    python pathway_hypergraph_representation_poc.py

Single real trajectory smoke test::

    python pathway_hypergraph_representation_poc.py --trajectory edge_relaxed.xyz \
        --target 0.42

Small manifest experiment::

    python pathway_hypergraph_representation_poc.py --manifest litraj_manifest.csv \
        --epochs 60

This is intentionally a proof of concept, not a benchmark architecture.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch import nn


# ------------------------------- data model ---------------------------------

@dataclass
class MigrationSample:
    atomic_numbers: torch.Tensor       # [N]
    positions: torch.Tensor            # [N, 3]
    edge_index: torch.Tensor           # [2, E], directed periodic radius graph
    edge_vectors: torch.Tensor         # [E, 3], minimum-image displacements
    trajectory: torch.Tensor           # [T, 3], unwrapped Cartesian trajectory
    path_distance: torch.Tensor        # [N], minimum atom-to-trajectory distance
    pathway_weight: torch.Tensor       # [N], weighted hyperedge incidence
    midpoint_weight: torch.Tensor      # [N], local marker pooling weights
    target: torch.Tensor               # [1], migration barrier in eV
    migrating_index: Optional[int] = None
    source: str = ""


# --------------------------- periodic geometry -------------------------------

def minimum_image_vector(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Shortest periodic image of one or more Cartesian row vectors."""
    inv_cell = np.linalg.inv(cell)
    frac = np.asarray(delta, dtype=float) @ inv_cell
    frac -= np.round(frac)
    return frac @ cell


def unwrap_path(points: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Make a sequence of periodically wrapped path points spatially continuous."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        raise ValueError("Trajectory must contain at least one point")
    result = [points[0]]
    for point in points[1:]:
        step = minimum_image_vector(point - result[-1], cell)
        result.append(result[-1] + step)
    return np.asarray(result)


def periodic_radius_graph(
    positions: np.ndarray, cell: np.ndarray, cutoff: float
) -> tuple[np.ndarray, np.ndarray]:
    """Small-system periodic radius graph built by exhaustive search."""
    senders, receivers, vectors = [], [], []
    for i in range(len(positions)):
        for j in range(len(positions)):
            if i == j:
                continue
            vec = minimum_image_vector(positions[j] - positions[i], cell)
            distance = np.linalg.norm(vec)
            if 1e-8 < distance <= cutoff:
                senders.append(i)
                receivers.append(j)
                vectors.append(vec)
    return (
        np.asarray([senders, receivers], dtype=np.int64),
        np.asarray(vectors, dtype=np.float32).reshape(-1, 3),
    )


def point_segment_projection(
    point: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[float, float]:
    """Return distance to segment and clamped segment coordinate t in [0,1]."""
    direction = b - a
    length2 = float(direction @ direction)
    if length2 < 1e-14:
        return float(np.linalg.norm(point - a)), 0.0
    t = np.clip(float((point - a) @ direction) / length2, 0.0, 1.0)
    closest = a + t * direction
    return float(np.linalg.norm(point - closest)), float(t)


def distances_to_periodic_path(
    positions: np.ndarray, path: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    """Minimum atom-to-polyline distances over the 27 nearest periodic images."""
    translations = np.asarray(list(product((-1, 0, 1), repeat=3))) @ cell
    distances = []
    for atom in positions:
        best = np.inf
        for translation in translations:
            image = atom + translation
            if len(path) == 1:
                best = min(best, float(np.linalg.norm(image - path[0])))
            else:
                for a, b in zip(path[:-1], path[1:]):
                    dist, _ = point_segment_projection(image, a, b)
                    best = min(best, dist)
        distances.append(best)
    return np.asarray(distances, dtype=np.float32)


def distances_to_periodic_point(
    positions: np.ndarray, point: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    delta = minimum_image_vector(positions - point[None, :], cell)
    return np.linalg.norm(delta, axis=1).astype(np.float32)


# ------------------------- representation builder ----------------------------

def build_sample(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    trajectory: np.ndarray,
    target: float,
    *,
    neighbour_cutoff: float = 3.5,
    pathway_cutoff: float = 2.4,
    pathway_sigma: float = 1.2,
    midpoint_sigma: float = 1.2,
    weighted_pathway: bool = True,
    migrating_index: Optional[int] = None,
    exclude_migrating_ion: bool = True,
    source: str = "",
) -> MigrationSample:
    """Build the common crystal graph and trajectory-conditioned representations."""
    path = unwrap_path(trajectory, cell)
    edge_index, edge_vectors = periodic_radius_graph(positions, cell, neighbour_cutoff)

    path_distance = distances_to_periodic_path(positions, path, cell)
    if weighted_pathway:
        pathway_weight = np.exp(-((path_distance / pathway_sigma) ** 2))
        pathway_weight[path_distance > pathway_cutoff] = 0.0
    else:
        pathway_weight = (path_distance < pathway_cutoff).astype(np.float32)

    # The hyperedge is intended to represent the framework surrounding the hop,
    # so do not let the migrating Li trivially identify the path by default.
    if exclude_migrating_ion and migrating_index is not None:
        pathway_weight[migrating_index] = 0.0

    if pathway_weight.sum() <= 1e-8:
        # Real LiTraj trajectories can occasionally be sparse or weakly coupled to
        # the local framework under a strict cutoff.  In that case keep the
        # physically closest atoms to the trajectory instead of failing the whole
        # sample, while preserving the same trajectory-conditioned representation.
        n_fallback = max(1, min(len(path_distance), max(3, int(math.ceil(0.05 * len(path_distance))))))
        fallback_idx = np.argsort(path_distance)[:n_fallback]
        pathway_weight = np.zeros(len(path_distance), dtype=np.float32)
        pathway_weight[fallback_idx] = np.exp(-((path_distance[fallback_idx] / max(pathway_sigma, 1e-6)) ** 2))
        if exclude_migrating_ion and migrating_index is not None:
            pathway_weight[migrating_index] = 0.0
        if pathway_weight.sum() <= 1e-8:
            pathway_weight = np.ones(len(path_distance), dtype=np.float32)
            if exclude_migrating_ion and migrating_index is not None:
                pathway_weight[migrating_index] = 0.0

    midpoint = 0.5 * (path[0] + path[-1])
    midpoint_distance = distances_to_periodic_point(positions, midpoint, cell)
    midpoint_weight = np.exp(-((midpoint_distance / midpoint_sigma) ** 2))
    if exclude_migrating_ion and migrating_index is not None:
        midpoint_weight[migrating_index] = 0.0
    if midpoint_weight.sum() <= 1e-8:
        midpoint_weight = np.ones(len(positions), dtype=np.float32)

    return MigrationSample(
        atomic_numbers=torch.as_tensor(atomic_numbers, dtype=torch.long),
        positions=torch.as_tensor(positions, dtype=torch.float32),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        edge_vectors=torch.as_tensor(edge_vectors, dtype=torch.float32),
        trajectory=torch.as_tensor(path, dtype=torch.float32),
        path_distance=torch.as_tensor(path_distance, dtype=torch.float32),
        pathway_weight=torch.as_tensor(pathway_weight, dtype=torch.float32),
        midpoint_weight=torch.as_tensor(midpoint_weight, dtype=torch.float32),
        target=torch.tensor([float(target)], dtype=torch.float32),
        migrating_index=migrating_index,
        source=source,
    )


# ------------------------------- model ---------------------------------------

class CrystalEncoder(nn.Module):
    """Tiny periodic message-passing encoder shared by every representation."""

    def __init__(self, hidden_dim: int = 48, layers: int = 2):
        super().__init__()
        self.element_embedding = nn.Embedding(119, hidden_dim)
        self.message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(layers)
        ])
        self.update_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(layers)
        ])

    def forward(self, sample: MigrationSample) -> torch.Tensor:
        h = self.element_embedding(sample.atomic_numbers.clamp(0, 118))
        if sample.edge_index.numel() == 0:
            return h
        send, recv = sample.edge_index
        distance = sample.edge_vectors.norm(dim=-1, keepdim=True)
        for msg_mlp, upd_mlp in zip(self.message_mlps, self.update_mlps):
            msg = msg_mlp(torch.cat([h[send], h[recv], distance], dim=-1))
            agg = torch.zeros_like(h)
            agg.index_add_(0, recv, msg)
            degree = torch.zeros(len(h), device=h.device, dtype=h.dtype)
            degree.index_add_(0, recv, torch.ones(len(recv), device=h.device, dtype=h.dtype))
            agg = agg / degree.clamp_min(1.0)[:, None]
            h = h + upd_mlp(torch.cat([h, agg], dim=-1))
        return h


class BarrierModel(nn.Module):
    """Same encoder/readout capacity; only migration-event representation changes."""

    def __init__(
        self,
        mode: str,
        hidden_dim: int = 48,
        layers: int = 2,
        random_seed: int = 0,
    ):
        super().__init__()
        if mode not in {"crystal", "midpoint", "trajectory", "randomized"}:
            raise ValueError(f"Unknown representation mode: {mode}")
        self.mode = mode
        self.random_seed = random_seed
        self.encoder = CrystalEncoder(hidden_dim=hidden_dim, layers=layers)
        self.null_context = nn.Parameter(torch.zeros(hidden_dim))
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def weighted_pool(h: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.clamp_min(0.0)
        return (weights[:, None] * h).sum(dim=0) / weights.sum().clamp_min(1e-8)

    def event_context(self, h: torch.Tensor, sample: MigrationSample) -> torch.Tensor:
        if self.mode == "crystal":
            return self.null_context
        if self.mode == "midpoint":
            return self.weighted_pool(h, sample.midpoint_weight)
        if self.mode == "trajectory":
            return self.weighted_pool(h, sample.pathway_weight)

        # Negative control: preserve the pathway weight distribution/hyperedge
        # size but destroy the physical atom-to-path assignment deterministically.
        generator = torch.Generator(device="cpu")
        key = sum(ord(c) for c in sample.source) + self.random_seed
        generator.manual_seed(key)
        permutation = torch.randperm(len(sample.pathway_weight), generator=generator)
        randomized = sample.pathway_weight[permutation.to(sample.pathway_weight.device)]
        return self.weighted_pool(h, randomized)

    def forward(self, sample: MigrationSample) -> torch.Tensor:
        h = self.encoder(sample)
        crystal = h.mean(dim=0)
        event = self.event_context(h, sample)
        return self.readout(torch.cat([crystal, event], dim=-1)).squeeze(-1)


# ---------------------------- real trajectory IO -----------------------------

def load_extxyz_trajectory(
    filename: Path, migrating_index: Optional[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    try:
        from ase.io import read
    except ImportError as exc:
        raise SystemExit("Real trajectory input requires ASE: pip install ase") from exc

    frames = read(str(filename), index=":")
    if len(frames) < 2:
        raise ValueError(f"{filename}: expected a multi-frame NEB trajectory")
    first = frames[0]
    numbers = np.asarray(first.numbers)
    cell = np.asarray(first.cell)

    if migrating_index is None:
        li_indices = np.flatnonzero(numbers == 3)
        if not len(li_indices):
            raise ValueError(f"{filename}: no Li atoms found")
        displacements = []
        for index in li_indices:
            points = np.asarray([frame.positions[index] for frame in frames])
            unwrapped = unwrap_path(points, cell)
            displacements.append(np.linalg.norm(unwrapped[-1] - unwrapped[0]))
        migrating_index = int(li_indices[int(np.argmax(displacements))])

    if numbers[migrating_index] != 3:
        raise ValueError(f"{filename}: atom {migrating_index} is not Li")

    trajectory = np.asarray([frame.positions[migrating_index] for frame in frames])
    return numbers, np.asarray(first.positions), cell, trajectory, migrating_index


def load_manifest(
    manifest: Path,
    args: argparse.Namespace,
) -> dict[str, list[MigrationSample]]:
    groups: dict[str, list[MigrationSample]] = {"train": [], "val": [], "test": []}
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "target", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for row in reader:
            split = row["split"].strip().lower()
            if split not in groups:
                raise ValueError(f"Unknown split {split!r}; expected train/val/test")
            trajectory_path = Path(row["path"])
            if not trajectory_path.is_absolute():
                trajectory_path = manifest.parent / trajectory_path
            idx_text = (row.get("migrating_index") or "").strip()
            idx = int(idx_text) if idx_text else None
            numbers, positions, cell, trajectory, idx = load_extxyz_trajectory(
                trajectory_path, idx
            )
            groups[split].append(build_sample(
                numbers, positions, cell, trajectory, float(row["target"]),
                neighbour_cutoff=args.neighbour_cutoff,
                pathway_cutoff=args.pathway_cutoff,
                pathway_sigma=args.pathway_sigma,
                midpoint_sigma=args.midpoint_sigma,
                weighted_pathway=not args.binary_pathway,
                migrating_index=idx,
                exclude_migrating_ion=not args.include_migrating_li,
                source=str(trajectory_path),
            ))
    if not all(groups.values()):
        counts = {k: len(v) for k, v in groups.items()}
        raise ValueError(f"Manifest must contain train/val/test rows; got {counts}")
    return groups


def load_litraj_dataset(
    dataset_root: Path,
    dataset_name: str,
    args: argparse.Namespace,
) -> dict[str, list[MigrationSample]]:
    """Load the official LiTraj benchmark data using the package's dataset API."""
    try:
        from litraj.data import load_data
    except ImportError as exc:
        raise SystemExit(
            "LiTraj dataset loading requires the official litraj package: "
            "pip install litraj"
        ) from exc

    dataset_root = Path(dataset_root)
    data = load_data(dataset_name, str(dataset_root))
    groups: dict[str, list[MigrationSample]] = {"train": [], "val": [], "test": []}

    if dataset_name == "nebDFT2k":
        frame_column = "trajectory_relaxed"
        target_column = "em_dft"
    elif dataset_name == "nebBVSE122k":
        frame_column = "trajectory"
        target_column = "em"
    else:
        raise ValueError(f"Unsupported LiTraj dataset for trajectory comparison: {dataset_name!r}")

    for _, row in data.iterrows():
        split = str(row["_split"]).strip().lower()
        if split not in groups:
            continue
        frames = row[frame_column]
        if len(frames) < 2:
            continue
        first = frames[0]
        numbers = np.asarray(first.numbers)
        positions = np.asarray(first.positions)
        cell = np.asarray(first.cell)
        li_indices = np.flatnonzero(numbers == 3)
        if len(li_indices) == 0:
            continue
        displacements = []
        for idx in li_indices:
            points = np.asarray([frame.positions[idx] for frame in frames], dtype=np.float64)
            unwrapped = unwrap_path(points, cell)
            displacements.append(float(np.linalg.norm(unwrapped[-1] - unwrapped[0])))
        migrating_index = int(li_indices[int(np.argmax(displacements))])
        trajectory = np.asarray([frame.positions[migrating_index] for frame in frames], dtype=np.float64)
        target = float(row[target_column])
        groups[split].append(build_sample(
            numbers,
            positions,
            cell,
            trajectory,
            target,
            neighbour_cutoff=args.neighbour_cutoff,
            pathway_cutoff=args.pathway_cutoff,
            pathway_sigma=args.pathway_sigma,
            midpoint_sigma=args.midpoint_sigma,
            weighted_pathway=not args.binary_pathway,
            migrating_index=migrating_index,
            exclude_migrating_ion=not args.include_migrating_li,
            source=f"{dataset_name}:{row.get('edge_id', 'unknown')}",
        ))

    if not all(groups.values()):
        counts = {k: len(v) for k, v in groups.items()}
        raise ValueError(f"LiTraj dataset {dataset_name!r} did not yield all splits; got {counts}")
    return groups


# -------------------------- synthetic PoC dataset ----------------------------

def synthetic_dataset(
    n: int,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, list[MigrationSample]]:
    """Controlled dataset where full-path environment, not midpoint, sets y.

    Each sample has the same hop endpoints (hence the same midpoint marker), but
    the path bends differently through a perturbed framework.  The target is a
    deterministic function of framework proximity to the entire trajectory.
    This deliberately gives the trajectory representation a real signal and is
    only intended to validate the experimental harness.
    """
    rng = np.random.default_rng(seed)
    samples: list[MigrationSample] = []
    cell = np.diag([10.0, 8.0, 8.0])
    start = np.array([1.2, 4.0, 4.0])
    end = np.array([8.8, 4.0, 4.0])

    base_framework = np.asarray([
        [2.4, 2.5, 4.0], [2.4, 5.5, 4.0],
        [4.0, 3.0, 4.0], [4.0, 5.0, 4.0],
        [5.8, 2.7, 4.0], [5.8, 5.3, 4.0],
        [7.4, 2.5, 4.0], [7.4, 5.5, 4.0],
        [4.9, 4.0, 2.2], [4.9, 4.0, 5.8],
    ])
    framework_z = np.asarray([8, 8, 8, 8, 8, 8, 8, 8, 15, 15])

    for i in range(n):
        framework = base_framework + rng.normal(0.0, 0.22, base_framework.shape)
        positions = np.vstack([start, end, framework])
        atomic_numbers = np.concatenate([[3, 3], framework_z])

        # Same endpoints and midpoint for every sample; trajectory curvature varies.
        bend = rng.uniform(-1.7, 1.7)
        skew = rng.uniform(-0.55, 0.55)
        xs = np.linspace(start[0], end[0], 7)
        u = np.linspace(0.0, 1.0, 7)
        ys = 4.0 + bend * np.sin(np.pi * u)
        zs = 4.0 + skew * np.sin(2.0 * np.pi * u)
        trajectory = np.column_stack([xs, ys, zs])

        d = distances_to_periodic_path(framework, unwrap_path(trajectory, cell), cell)
        # Synthetic "barrier": close O atoms and especially close P atoms penalize
        # the whole corridor.  Add small noise to prevent a trivial exact mapping.
        z_factor = np.where(framework_z == 15, 1.35, 1.0)
        crowding = np.sum(z_factor * np.exp(-((d / 1.15) ** 2))) / len(framework)
        barrier = 0.12 + 0.62 * crowding + rng.normal(0.0, 0.012)

        samples.append(build_sample(
            atomic_numbers, positions, cell, trajectory, barrier,
            neighbour_cutoff=args.neighbour_cutoff,
            pathway_cutoff=args.pathway_cutoff,
            pathway_sigma=args.pathway_sigma,
            midpoint_sigma=args.midpoint_sigma,
            weighted_pathway=not args.binary_pathway,
            migrating_index=0,
            exclude_migrating_ion=not args.include_migrating_li,
            source=f"synthetic-{i}",
        ))

    rng.shuffle(samples)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    return {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }


# ------------------------------ training -------------------------------------

def mae(model: nn.Module, samples: Iterable[MigrationSample]) -> float:
    model.eval()
    errors = []
    with torch.no_grad():
        for sample in samples:
            errors.append(abs(float(model(sample)) - float(sample.target.item())))
    return float(np.mean(errors)) if errors else math.nan


def train_one(
    mode: str,
    splits: dict[str, list[MigrationSample]],
    *,
    hidden_dim: int,
    layers: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[BarrierModel, dict[str, float]]:
    torch.manual_seed(seed)
    model = BarrierModel(mode, hidden_dim=hidden_dim, layers=layers, random_seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = None
    best_val = float("inf")

    order_rng = random.Random(seed)
    for _ in range(epochs):
        model.train()
        order = list(range(len(splits["train"])))
        order_rng.shuffle(order)
        for idx in order:
            sample = splits["train"][idx]
            optimizer.zero_grad()
            pred = model(sample)
            loss = (pred - sample.target.squeeze(0)).square()
            loss.backward()
            optimizer.step()
        val = mae(model, splits["val"])
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = {
        "train_mae": mae(model, splits["train"]),
        "val_mae": mae(model, splits["val"]),
        "test_mae": mae(model, splits["test"]),
    }
    return model, metrics


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_comparison(splits: dict[str, list[MigrationSample]], args: argparse.Namespace) -> None:
    counts = {k: len(v) for k, v in splits.items()}
    print(f"split sizes: {counts}")
    print("representation comparison (same samples / targets / atomic encoder family)")
    print("mode         params      train_MAE   val_MAE    test_MAE")
    print("-----------  ----------  ----------  ---------  ---------")
    for mode in ("crystal", "midpoint", "trajectory", "randomized"):
        model, metrics = train_one(
            mode, splits,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        print(
            f"{mode:<11}  {parameter_count(model):>10d}  "
            f"{metrics['train_mae']:.4f} eV  {metrics['val_mae']:.4f} eV  "
            f"{metrics['test_mae']:.4f} eV"
        )
    print("\nInterpretation: this PoC tests whether trajectory-conditioned grouping carries")
    print("useful signal beyond crystal-only and midpoint conditioning. It does not claim")
    print("that a DFT-NEB trajectory is free or that this model replaces DFT-NEB.")


def run_single(args: argparse.Namespace) -> None:
    numbers, positions, cell, trajectory, idx = load_extxyz_trajectory(
        args.trajectory, args.migrating_index
    )
    if args.target is None:
        raise SystemExit("--trajectory mode requires --target because metadata schema is not assumed")
    sample = build_sample(
        numbers, positions, cell, trajectory, args.target,
        neighbour_cutoff=args.neighbour_cutoff,
        pathway_cutoff=args.pathway_cutoff,
        pathway_sigma=args.pathway_sigma,
        midpoint_sigma=args.midpoint_sigma,
        weighted_pathway=not args.binary_pathway,
        migrating_index=idx,
        exclude_migrating_ion=not args.include_migrating_li,
        source=str(args.trajectory),
    )
    print(f"source: {sample.source}")
    print(f"atoms: {len(sample.atomic_numbers)}")
    print(f"directed neighbour edges: {sample.edge_index.shape[1]}")
    print(f"trajectory images: {len(sample.trajectory)}")
    print(f"migrating Li index: {sample.migrating_index}")
    print(f"nonzero pathway memberships: {(sample.pathway_weight > 0).sum().item()}")
    print(f"target: {sample.target.item():.4f} eV")
    for mode in ("crystal", "midpoint", "trajectory", "randomized"):
        torch.manual_seed(args.seed)
        model = BarrierModel(mode, args.hidden_dim, args.layers, args.seed)
        pred = model(sample)
        loss = (pred - sample.target.squeeze(0)).square()
        loss.backward()
        print(f"{mode:<11}: forward/backward PASS, untrained prediction={pred.item():.4f} eV")


# ---------------------------------- CLI --------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--trajectory", type=Path, help="one multi-frame extxyz trajectory")
    source.add_argument("--manifest", type=Path, help="CSV: path,target,split[,migrating_index]")
    source.add_argument(
        "--litraj-data",
        type=Path,
        help="directory containing LiTraj dataset folders such as /path/to/litraj_root",
    )
    parser.add_argument(
        "--litraj-dataset",
        choices=["nebDFT2k", "nebBVSE122k"],
        default="nebDFT2k",
        help="LiTraj benchmark to use when --litraj-data is provided",
    )
    parser.add_argument("--target", type=float, help="barrier for --trajectory smoke test")
    parser.add_argument("--migrating-index", type=int, help="0-based Li index for one trajectory")
    parser.add_argument("--neighbour-cutoff", type=float, default=3.5)
    parser.add_argument("--pathway-cutoff", type=float, default=2.4)
    parser.add_argument("--pathway-sigma", type=float, default=1.2)
    parser.add_argument("--midpoint-sigma", type=float, default=1.2)
    parser.add_argument("--binary-pathway", action="store_true", help="use hard cutoff rather than weighted incidence")
    parser.add_argument("--include-migrating-li", action="store_true", help="include migrating Li in event pooling")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--synthetic-samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.trajectory is not None:
        run_single(args)
        return

    if args.manifest is not None:
        splits = load_manifest(args.manifest, args)
        run_comparison(splits, args)
        return

    if args.litraj_data is not None:
        splits = load_litraj_dataset(args.litraj_data, args.litraj_dataset, args)
        run_comparison(splits, args)
        return

    print("Running controlled synthetic representation sanity check.")
    print("Different trajectories share the same endpoints/midpoint; the synthetic target")
    print("depends on the framework environment along the full trajectory.\n")
    splits = synthetic_dataset(args.synthetic_samples, args, args.seed)
    run_comparison(splits, args)


if __name__ == "__main__":
    main()
