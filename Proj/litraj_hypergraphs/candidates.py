"""Four physically motivated trajectory-to-hyperedge constructions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable

import numpy as np
import torch

from .geometry import atom_path_geometry, periodic_distances, periodic_radius_graph, unwrap_points


ANION_ELEMENTS = frozenset({7, 8, 9, 16, 17, 34, 35, 52, 53})


@dataclass
class HypergraphSample:
    atomic_numbers: torch.Tensor       # [N]
    positions: torch.Tensor            # [N, 3], first image
    edge_index: torch.Tensor           # [2, E]
    edge_vectors: torch.Tensor         # [E, 3]
    frame_edge_indices: tuple[torch.Tensor, ...]  # one periodic graph per image
    frame_edge_vectors: tuple[torch.Tensor, ...]
    frames: torch.Tensor               # [T, N, 3]
    trajectory: torch.Tensor           # [T, 3], unwrapped migrating-Li path
    incidence: torch.Tensor            # [N, H], soft atom-to-hyperedge incidence
    hyperedge_steps: torch.Tensor      # [H], temporal/image index
    hyperedge_types: torch.Tensor      # [H], construction-specific type
    hyperedge_frames: torch.Tensor     # [H], source NEB image (-1 if not image-based)
    hyperedge_positions: torch.Tensor  # [H], normalized physical path position
    hyperedge_distances: torch.Tensor  # [N, H], local radial distance proxy
    path_distance: torch.Tensor        # [N], first-image atom distance to Li path
    path_position: torch.Tensor        # [N], closest normalized path coordinate
    target: torch.Tensor               # scalar barrier in eV
    cheap_barrier: torch.Tensor        # scalar BVSE barrier, NaN when unavailable
    migrating_index: int
    source: str = ""
    metadata: dict[str, float | int | str] | None = None

    def to(self, device: torch.device | str) -> "HypergraphSample":
        """Return a copy with every tensor moved to the requested device."""
        return HypergraphSample(
            atomic_numbers=self.atomic_numbers.to(device),
            positions=self.positions.to(device),
            edge_index=self.edge_index.to(device),
            edge_vectors=self.edge_vectors.to(device),
            frame_edge_indices=tuple(value.to(device) for value in self.frame_edge_indices),
            frame_edge_vectors=tuple(value.to(device) for value in self.frame_edge_vectors),
            frames=self.frames.to(device),
            trajectory=self.trajectory.to(device),
            incidence=self.incidence.to(device),
            hyperedge_steps=self.hyperedge_steps.to(device),
            hyperedge_types=self.hyperedge_types.to(device),
            hyperedge_frames=self.hyperedge_frames.to(device),
            hyperedge_positions=self.hyperedge_positions.to(device),
            hyperedge_distances=self.hyperedge_distances.to(device),
            path_distance=self.path_distance.to(device),
            path_position=self.path_position.to(device),
            target=self.target.to(device),
            cheap_barrier=self.cheap_barrier.to(device),
            migrating_index=self.migrating_index,
            source=self.source,
            metadata=self.metadata,
        )


def _environment_weights(
    positions: np.ndarray,
    li_position: np.ndarray,
    cell: np.ndarray,
    migrating_index: int,
    cutoff: float,
    sigma: float,
) -> np.ndarray:
    distances = periodic_distances(positions, li_position, cell)
    weights = np.exp(-((distances / max(sigma, 1e-6)) ** 2)).astype(np.float32)
    weights[distances > cutoff] = 0.0
    weights[migrating_index] = 0.0
    if weights.sum() <= 1e-8:
        candidates = np.argsort(distances + (np.arange(len(distances)) == migrating_index) * 1e6)
        weights[candidates[: min(4, len(candidates) - 1)]] = 1.0
    return weights


def image_coordination(
    numbers: np.ndarray, frames: np.ndarray, cell: np.ndarray, migrating_index: int,
    cutoff: float, sigma: float, **_: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """One local coordination-environment hyperedge per trajectory image."""
    columns = [
        _environment_weights(frame, frame[migrating_index], cell, migrating_index, cutoff, sigma)
        for frame in frames
    ]
    return (
        np.stack(columns, axis=1),
        np.arange(len(columns), dtype=np.int64),
        np.zeros(len(columns), dtype=np.int64),
        {"num_images": len(columns)},
    )


def position_aware_trajectory_graph(
    numbers: np.ndarray, frames: np.ndarray, cell: np.ndarray, migrating_index: int,
    cutoff: float, sigma: float, **kwargs: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Actual NEB-image environments used as nodes of an ordered path graph."""
    incidence, steps, types, _ = image_coordination(
        numbers, frames, cell, migrating_index, cutoff, sigma, **kwargs
    )
    return incidence, steps, types, {
        "construction": "position_aware_neb_image_trajectory_graph",
        "num_trajectory_nodes": len(steps),
    }


def shuffled_position_aware_trajectory_graph(
    numbers: np.ndarray, frames: np.ndarray, cell: np.ndarray, migrating_index: int,
    cutoff: float, sigma: float, random_seed: int = 0, source: str = "", **kwargs: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Same image nodes and absolute positions, with path adjacency permuted."""
    incidence, _, types, _ = image_coordination(
        numbers, frames, cell, migrating_index, cutoff, sigma, **kwargs
    )
    sample_seed = random_seed + sum((i + 1) * ord(char) for i, char in enumerate(source))
    order = np.random.default_rng(sample_seed).permutation(incidence.shape[1])
    if len(order) > 1 and np.array_equal(order, np.arange(len(order))):
        order = np.roll(order, 1)
    # Column t retains its physical features, but receives a shuffled graph rank.
    steps = np.empty(len(order), dtype=np.int64)
    steps[order] = np.arange(len(order), dtype=np.int64)
    return incidence, steps, types, {
        "construction": "shuffled_position_aware_neb_image_trajectory_graph",
        "num_trajectory_nodes": len(steps),
        "shuffle_order": ",".join(map(str, order.tolist())),
    }


def bottleneck(
    numbers: np.ndarray, frames: np.ndarray, cell: np.ndarray, migrating_index: int,
    cutoff: float, sigma: float, bottleneck_neighbours: int = 4, **_: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """One hyperedge at the geometrically narrowest trajectory image.

    The bottleneck score is the mean distance to the k closest framework atoms.
    It uses geometry only and therefore does not leak the NEB energy profile.
    """
    distance_rows = []
    for frame in frames:
        distances = periodic_distances(frame, frame[migrating_index], cell)
        distances = np.delete(distances, migrating_index)
        k = min(max(1, bottleneck_neighbours), len(distances))
        distance_rows.append(float(np.partition(distances, k - 1)[:k].mean()))
    image = int(np.argmin(distance_rows))
    weights = _environment_weights(
        frames[image], frames[image, migrating_index], cell, migrating_index, cutoff, sigma
    )
    return (
        weights[:, None], np.asarray([image]), np.asarray([0]),
        {
            "construction": "geometric_crowding_bottleneck",
            "bottleneck_image": image,
            "mean_k_neighbour_distance": distance_rows[image],
        },
    )


def coordination_change(
    numbers: np.ndarray, frames: np.ndarray, cell: np.ndarray, migrating_index: int,
    cutoff: float, sigma: float, change_top_k: int = 8, **_: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """One hyperedge containing atoms whose smooth Li coordination changes most."""
    distances = np.stack([
        periodic_distances(frame, frame[migrating_index], cell) for frame in frames
    ])
    coordination = np.exp(-((distances / max(sigma, 1e-6)) ** 2))
    change = coordination.max(axis=0) - coordination.min(axis=0)
    near_path = distances.min(axis=0) <= cutoff
    change[~near_path] = 0.0
    change[migrating_index] = 0.0
    candidates = np.flatnonzero(change > 1e-8)
    if len(candidates):
        keep = candidates[np.argsort(change[candidates])[-change_top_k:]]
        weights = np.zeros_like(change, dtype=np.float32)
        weights[keep] = change[keep]
    else:
        weights = _environment_weights(
            frames[len(frames) // 2], frames[len(frames) // 2, migrating_index],
            cell, migrating_index, cutoff, sigma,
        )
    return (
        weights[:, None], np.asarray([0]), np.asarray([0]),
        {
            "construction": "smooth_radial_interaction_change",
            "selected_atoms": int(np.count_nonzero(weights)),
        },
    )


def species_image(
    numbers: np.ndarray, frames: np.ndarray, cell: np.ndarray, migrating_index: int,
    cutoff: float, sigma: float, **_: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Per-image environments split into anion, framework-cation and other-Li edges."""
    numbers = np.asarray(numbers)
    masks = [
        np.isin(numbers, list(ANION_ELEMENTS)),
        (~np.isin(numbers, list(ANION_ELEMENTS))) & (numbers != 3),
        numbers == 3,
    ]
    columns, steps, types = [], [], []
    for step, frame in enumerate(frames):
        base = _environment_weights(frame, frame[migrating_index], cell, migrating_index, cutoff, sigma)
        for edge_type, mask in enumerate(masks):
            typed = base * mask
            if typed.sum() > 1e-8:
                columns.append(typed)
                steps.append(step)
                types.append(edge_type)
    if not columns:
        columns = [np.ones(len(numbers), dtype=np.float32)]
        columns[0][migrating_index] = 0.0
        steps, types = [0], [0]
    return (
        np.stack(columns, axis=1), np.asarray(steps), np.asarray(types),
        {
            "construction": "heuristic_element_role_partition",
            "num_typed_hyperedges": len(columns),
        },
    )


def _path_weights(
    frames: np.ndarray, cell: np.ndarray, migrating_index: int,
    cutoff: float, sigma: float,
    path_geometry: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path_geometry is None:
        path = unwrap_points(frames[:, migrating_index], cell)
        distance, position = atom_path_geometry(frames[0], path, cell)
    else:
        distance, position = path_geometry
    weights = np.exp(-((distance / max(sigma, 1e-6)) ** 2)).astype(np.float32)
    weights[distance > cutoff] = 0.0
    weights[migrating_index] = 0.0
    if weights.sum() <= 1e-8:
        order = np.argsort(distance + (np.arange(len(distance)) == migrating_index) * 1e6)
        weights[order[: min(5, len(order) - 1)]] = 1.0
    return weights, distance, position


def crystal_only(
    numbers: np.ndarray, frames: np.ndarray, cell: np.ndarray, migrating_index: int,
    cutoff: float, sigma: float, **_: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    # A placeholder incidence keeps the common sample schema valid; the model
    # deliberately substitutes a learned null event context for this candidate.
    weights = np.ones((len(numbers), 1), dtype=np.float32)
    weights[migrating_index] = 0.0
    return weights, np.asarray([0]), np.asarray([0]), {}


def midpoint(
    numbers: np.ndarray, frames: np.ndarray, cell: np.ndarray, migrating_index: int,
    cutoff: float, sigma: float, **_: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    path = unwrap_points(frames[:, migrating_index], cell)
    weights = _environment_weights(frames[0], 0.5 * (path[0] + path[-1]), cell,
                                   migrating_index, cutoff, sigma)
    return weights[:, None], np.asarray([0]), np.asarray([0]), {}


def whole_trajectory(
    numbers: np.ndarray, frames: np.ndarray, cell: np.ndarray, migrating_index: int,
    cutoff: float, sigma: float,
    path_geometry: tuple[np.ndarray, np.ndarray] | None = None, **_: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    weights, _, _ = _path_weights(
        frames, cell, migrating_index, cutoff, sigma, path_geometry
    )
    return weights[:, None], np.asarray([0]), np.asarray([0]), {}


def positional_trajectory(
    numbers: np.ndarray, frames: np.ndarray, cell: np.ndarray, migrating_index: int,
    cutoff: float, sigma: float,
    path_geometry: tuple[np.ndarray, np.ndarray] | None = None, **_: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    return whole_trajectory(
        numbers, frames, cell, migrating_index, cutoff, sigma,
        path_geometry=path_geometry,
    )


def shuffled_s_trajectory(
    numbers: np.ndarray, frames: np.ndarray, cell: np.ndarray, migrating_index: int,
    cutoff: float, sigma: float,
    path_geometry: tuple[np.ndarray, np.ndarray] | None = None, **_: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Positional incidence with atom-to-reaction-coordinate association shuffled.

    The actual coordinate permutation is applied in ``build_candidate_sample``
    because path positions are part of the shared sample schema rather than the
    incidence returned by a candidate constructor.
    """
    incidence, steps, types, _ = whole_trajectory(
        numbers, frames, cell, migrating_index, cutoff, sigma,
        path_geometry=path_geometry,
    )
    return incidence, steps, types, {"control": "within_hop_supported_atom_s_shuffle"}


def segmented_trajectory(
    numbers: np.ndarray, frames: np.ndarray, cell: np.ndarray, migrating_index: int,
    cutoff: float, sigma: float, num_segments: int = 5, segment_sigma: float = 0.18,
    path_geometry: tuple[np.ndarray, np.ndarray] | None = None,
    **_: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    weights, _, position = _path_weights(
        frames, cell, migrating_index, cutoff, sigma, path_geometry
    )
    centres = (np.arange(num_segments, dtype=np.float32) + 0.5) / num_segments
    along_path = np.exp(-(((position[:, None] - centres[None, :]) / segment_sigma) ** 2))
    incidence = weights[:, None] * along_path
    return incidence, np.arange(num_segments), np.zeros(num_segments, dtype=np.int64), {
        "num_segments": num_segments
    }


def randomized_trajectory(
    numbers: np.ndarray, frames: np.ndarray, cell: np.ndarray, migrating_index: int,
    cutoff: float, sigma: float, random_seed: int = 0, source: str = "",
    path_geometry: tuple[np.ndarray, np.ndarray] | None = None, **_: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    weights, _, _ = _path_weights(
        frames, cell, migrating_index, cutoff, sigma, path_geometry
    )
    sample_seed = random_seed + sum((index + 1) * ord(char) for index, char in enumerate(source))
    framework = np.flatnonzero(np.arange(len(weights)) != migrating_index)
    randomized = np.zeros_like(weights)
    randomized[framework] = np.random.default_rng(sample_seed).permutation(weights[framework])
    randomized[migrating_index] = 0.0
    return randomized[:, None], np.asarray([0]), np.asarray([0]), {
        "random_seed": random_seed
    }


CANDIDATES: dict[str, Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, dict]]] = {
    "crystal": crystal_only,
    "midpoint": midpoint,
    "trajectory": whole_trajectory,
    "positional": positional_trajectory,
    "positional_shuffled_s": shuffled_s_trajectory,
    "segmented": segmented_trajectory,
    "randomized": randomized_trajectory,
    "image_coordination": image_coordination,
    "bottleneck": bottleneck,
    "coordination_change": coordination_change,
    "species_image": species_image,
    "geometric_bottleneck": bottleneck,
    "position_aware_trajectory_graph": position_aware_trajectory_graph,
    "position_aware_trajectory_graph_shuffled": shuffled_position_aware_trajectory_graph,
}


def build_candidate_sample(
    candidate: str,
    numbers: np.ndarray,
    frames: np.ndarray,
    cell: np.ndarray,
    migrating_index: int,
    target: float,
    *,
    neighbour_cutoff: float = 3.5,
    hyperedge_cutoff: float = 3.0,
    hyperedge_sigma: float = 1.5,
    bottleneck_neighbours: int = 4,
    change_top_k: int = 8,
    num_segments: int = 5,
    segment_sigma: float = 0.18,
    random_seed: int = 0,
    source: str = "",
    frame_graphs: list[tuple[np.ndarray, np.ndarray]] | None = None,
    path_geometry: tuple[np.ndarray, np.ndarray] | None = None,
    cheap_barrier: float | None = None,
) -> HypergraphSample:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate!r}; choose from {sorted(CANDIDATES)}")
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 3 or len(frames) < 2:
        raise ValueError("frames must have shape [T, N, 3] with at least two images")
    if any(len(frame) != len(numbers) for frame in frames):
        raise ValueError("all images must contain the same atoms")
    if frame_graphs is None:
        frame_graphs = [periodic_radius_graph(frame, cell, neighbour_cutoff) for frame in frames]
    if len(frame_graphs) != len(frames):
        raise ValueError("frame_graphs must contain one graph per trajectory image")
    edge_index, edge_vectors = frame_graphs[0]
    trajectory = unwrap_points(frames[:, migrating_index], cell)
    image_lengths = np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)
    image_cumulative = np.concatenate([[0.0], np.cumsum(image_lengths)])
    if image_cumulative[-1] > 1e-12:
        image_positions = image_cumulative / image_cumulative[-1]
    else:
        image_positions = np.linspace(0.0, 1.0, len(trajectory))
    if path_geometry is None:
        path_geometry = atom_path_geometry(frames[0], trajectory, cell)
    incidence, steps, types, metadata = CANDIDATES[candidate](
        np.asarray(numbers), frames, np.asarray(cell), migrating_index,
        hyperedge_cutoff, hyperedge_sigma,
        bottleneck_neighbours=bottleneck_neighbours,
        change_top_k=change_top_k,
        num_segments=num_segments,
        segment_sigma=segment_sigma,
        random_seed=random_seed,
        source=source,
        path_geometry=path_geometry,
    )
    path_distance, path_position = path_geometry
    if candidate == "positional_shuffled_s":
        # Shuffle only atoms with nonzero pathway incidence, excluding the
        # migrating ion. A source-derived seed makes the control invariant to
        # training seed, data-loader order, and epoch.
        supported = np.flatnonzero(
            (np.asarray(incidence).sum(axis=1) > 0)
            & (np.arange(len(numbers)) != migrating_index)
        )
        path_position = np.asarray(path_position, dtype=np.float32).copy()
        if len(supported) > 1:
            digest = hashlib.sha256(str(source).encode("utf-8")).digest()
            shuffle_seed = int.from_bytes(digest[:8], "little", signed=False)
            permutation = np.random.default_rng(shuffle_seed).permutation(supported)
            if np.array_equal(permutation, supported):
                permutation = np.roll(permutation, 1)
            original = path_position.copy()
            path_position[supported] = original[permutation]
        metadata["shuffled_s_atoms"] = int(len(supported))
    if candidate == "segmented":
        hyperedge_positions = (np.arange(len(steps), dtype=np.float32) + 0.5) / len(steps)
        hyperedge_frames = np.full(len(steps), -1, dtype=np.int64)
    elif candidate == "position_aware_trajectory_graph_shuffled":
        hyperedge_positions = image_positions[:len(steps)]
        hyperedge_frames = np.arange(len(steps), dtype=np.int64)
    elif len(steps) and int(np.max(steps)) < len(image_positions):
        hyperedge_positions = image_positions[np.asarray(steps, dtype=int)]
        hyperedge_frames = np.asarray(steps, dtype=np.int64)
    else:
        hyperedge_positions = np.full(len(steps), 0.5, dtype=np.float32)
        hyperedge_frames = np.full(len(steps), -1, dtype=np.int64)
    clipped = np.clip(incidence, 1e-30, 1.0)
    hyperedge_distances = hyperedge_sigma * np.sqrt(-np.log(clipped))
    hyperedge_distances[incidence <= 0] = hyperedge_cutoff
    return HypergraphSample(
        atomic_numbers=torch.as_tensor(numbers, dtype=torch.long),
        positions=torch.as_tensor(frames[0], dtype=torch.float32),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        edge_vectors=torch.as_tensor(edge_vectors, dtype=torch.float32),
        frame_edge_indices=tuple(torch.as_tensor(graph[0], dtype=torch.long) for graph in frame_graphs),
        frame_edge_vectors=tuple(torch.as_tensor(graph[1], dtype=torch.float32) for graph in frame_graphs),
        frames=torch.as_tensor(frames, dtype=torch.float32),
        trajectory=torch.as_tensor(trajectory, dtype=torch.float32),
        incidence=torch.as_tensor(incidence, dtype=torch.float32),
        hyperedge_steps=torch.as_tensor(steps, dtype=torch.long),
        hyperedge_types=torch.as_tensor(types, dtype=torch.long),
        hyperedge_frames=torch.as_tensor(hyperedge_frames, dtype=torch.long),
        hyperedge_positions=torch.as_tensor(hyperedge_positions, dtype=torch.float32),
        hyperedge_distances=torch.as_tensor(hyperedge_distances, dtype=torch.float32),
        path_distance=torch.as_tensor(path_distance, dtype=torch.float32),
        path_position=torch.as_tensor(path_position, dtype=torch.float32),
        target=torch.tensor(float(target), dtype=torch.float32),
        cheap_barrier=torch.tensor(
            float("nan") if cheap_barrier is None else float(cheap_barrier), dtype=torch.float32
        ),
        migrating_index=int(migrating_index), source=source,
        metadata={"candidate": candidate, **metadata},
    )
