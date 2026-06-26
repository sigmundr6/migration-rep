#!/usr/bin/env python3
"""Construction and forward-pass sanity checks on real LiTraj nebDFT2k files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase.io import read

from litraj_hypergraphs.candidates import build_candidate_sample
from litraj_hypergraphs.data import identify_migrating_li
from litraj_hypergraphs.geometry import atom_path_geometry, periodic_radius_graph, unwrap_points
from litraj_hypergraphs.model import CandidateBarrierModel


METHODS = (
    "crystal",
    "midpoint",
    "trajectory",
    "positional",
    "segmented",
    "randomized",
    "image_coordination",
    "geometric_bottleneck",
    "coordination_change",
    "species_image",
    "position_aware_trajectory_graph",
    "position_aware_trajectory_graph_shuffled",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="directory containing nebDFT2k/ (default: data)",
    )
    parser.add_argument("--samples-per-split", type=int, default=1)
    parser.add_argument("--neighbour-cutoff", type=float, default=3.5)
    parser.add_argument("--hyperedge-cutoff", type=float, default=3.0)
    parser.add_argument("--hyperedge-sigma", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-forward", action="store_true")
    return parser.parse_args()


def load_row(dataset: Path, row: pd.Series):
    path = dataset / f"{row.edge_id}_relaxed.xyz"
    images = read(str(path), index=":")
    numbers = np.asarray(images[0].numbers)
    frames = np.stack([np.asarray(image.positions) for image in images])
    cell = np.asarray(images[0].cell)
    if any(not np.array_equal(np.asarray(image.numbers), numbers) for image in images):
        raise AssertionError(f"{row.edge_id}: atom ordering changes across images")
    moving = identify_migrating_li(numbers, frames, cell)
    return numbers, frames, cell, moving


def check_sample(sample, method: str) -> None:
    atom_count, edge_count = sample.incidence.shape
    assert atom_count == len(sample.atomic_numbers)
    assert edge_count == len(sample.hyperedge_steps) == len(sample.hyperedge_types)
    assert edge_count == len(sample.hyperedge_frames) == len(sample.hyperedge_positions)
    assert sample.hyperedge_distances.shape == sample.incidence.shape
    assert edge_count > 0
    assert torch.isfinite(sample.incidence).all()
    assert (sample.incidence >= 0).all()
    assert float(sample.incidence.sum()) > 0
    assert float(sample.incidence[sample.migrating_index].sum()) == 0.0
    assert len(sample.frame_edge_indices) == len(sample.frames)
    assert all(index.shape[0] == 2 for index in sample.frame_edge_indices)
    assert all(torch.isfinite(vector).all() for vector in sample.frame_edge_vectors)
    if method == "image_coordination":
        assert edge_count == len(sample.frames)
    if method == "segmented":
        assert edge_count == 5
    if method == "geometric_bottleneck":
        image = int(sample.metadata["bottleneck_image"])
        assert 0 <= image < len(sample.frames)
    if method == "species_image":
        assert set(sample.hyperedge_types.tolist()).issubset({0, 1, 2})
    if method.startswith("position_aware_trajectory_graph"):
        assert edge_count == len(sample.frames)
        assert torch.all(sample.hyperedge_positions[1:] > sample.hyperedge_positions[:-1])


def check_cross_method(samples: dict[str, object]) -> dict[str, float | int]:
    trajectory = samples["trajectory"]
    randomized = samples["randomized"]
    framework = np.arange(len(trajectory.atomic_numbers)) != trajectory.migrating_index
    assert np.allclose(
        np.sort(trajectory.incidence[framework, 0].numpy()),
        np.sort(randomized.incidence[framework, 0].numpy()),
    )
    randomized_difference = float(
        torch.max(torch.abs(trajectory.incidence[:, 0] - randomized.incidence[:, 0]))
    )
    assert randomized_difference > 0

    image = samples["image_coordination"]
    bottleneck = samples["geometric_bottleneck"]
    bottleneck_image = int(bottleneck.metadata["bottleneck_image"])
    assert torch.allclose(
        bottleneck.incidence[:, 0], image.incidence[:, bottleneck_image]
    )

    species = samples["species_image"]
    for step in range(len(image.frames)):
        columns = species.hyperedge_steps == step
        reconstructed = species.incidence[:, columns].sum(dim=1)
        assert torch.allclose(reconstructed, image.incidence[:, step], atol=1e-6)

    change = samples["coordination_change"]
    changed_atoms = int((change.incidence[:, 0] > 0).sum())
    assert 0 < changed_atoms <= 8
    framework_motion = image.frames[:, framework] - image.frames[0, framework]
    maximum_relaxation = float(torch.linalg.vector_norm(framework_motion, dim=-1).max())
    assert maximum_relaxation > 0
    image_variation = float(torch.max(torch.abs(image.incidence[:, 1:] - image.incidence[:, :-1])))
    assert image_variation > 0
    hierarchical = samples["position_aware_trajectory_graph"]
    shuffled = samples["position_aware_trajectory_graph_shuffled"]
    assert torch.allclose(hierarchical.incidence, shuffled.incidence)
    assert torch.allclose(hierarchical.hyperedge_positions, shuffled.hyperedge_positions)
    assert torch.equal(hierarchical.hyperedge_frames, shuffled.hyperedge_frames)
    assert not torch.equal(hierarchical.hyperedge_steps, shuffled.hyperedge_steps)
    return {
        "bottleneck_image": bottleneck_image,
        "changed_atoms": changed_atoms,
        "max_framework_relaxation": maximum_relaxation,
        "max_image_incidence_change": image_variation,
        "randomized_assignment_change": randomized_difference,
    }


def main() -> None:
    args = parse_args()
    dataset = args.data_root / "nebDFT2k"
    index_path = dataset / "nebDFT2k_index.csv"
    if not index_path.exists():
        raise SystemExit(f"missing {index_path}; download the official nebDFT2k dataset first")
    table = pd.read_csv(index_path)
    required = {"edge_id", "em_dft", "_split"}
    if missing := required - set(table.columns):
        raise SystemExit(f"index is missing columns: {sorted(missing)}")
    rows = pd.concat([
        table[table._split == split].head(args.samples_per_split)
        for split in ("train", "val", "test")
    ])
    torch.manual_seed(args.seed)
    model = CandidateBarrierModel(hidden_dim=16, layers=1)
    checked = 0
    print("split  edge_id                         method                                  atoms images edges members prediction")
    print("-----  ------------------------------  --------------------------------------  ----- ------ ----- ------- ----------")
    for _, row in rows.iterrows():
        numbers, frames, cell, moving = load_row(dataset, row)
        frame_graphs = [
            periodic_radius_graph(frame, cell, args.neighbour_cutoff) for frame in frames
        ]
        path = unwrap_points(frames[:, moving], cell)
        path_geometry = atom_path_geometry(frames[0], path, cell)
        method_samples = {}
        for method in METHODS:
            sample = build_candidate_sample(
                method, numbers, frames, cell, moving, float(row.em_dft),
                neighbour_cutoff=args.neighbour_cutoff,
                hyperedge_cutoff=args.hyperedge_cutoff,
                hyperedge_sigma=args.hyperedge_sigma,
                random_seed=args.seed,
                source=f"nebDFT2k:{row.edge_id}",
                frame_graphs=frame_graphs,
                path_geometry=path_geometry,
            )
            check_sample(sample, method)
            method_samples[method] = sample
            prediction = "skipped"
            if not args.skip_forward:
                with torch.no_grad():
                    value = float(model(sample))
                assert np.isfinite(value)
                prediction = f"{value: .4f}"
            members = int((sample.incidence > 0).sum())
            print(
                f"{row._split:<5}  {row.edge_id:<30}  {method:<38}  "
                f"{len(numbers):>5} {len(frames):>6} {sample.incidence.shape[1]:>5} "
                f"{members:>7} {prediction:>10}"
            )
            checked += 1
        diagnostics = check_cross_method(method_samples)
        print(f"cross-method diagnostics for {row.edge_id}: {diagnostics}")
    print(f"PASS: {checked} real LiTraj sample/method constructions satisfied all invariants")


if __name__ == "__main__":
    main()
