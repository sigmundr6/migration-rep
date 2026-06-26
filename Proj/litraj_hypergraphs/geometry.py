"""Periodic geometry helpers used by all hyperedge constructions."""

from __future__ import annotations

from itertools import product

import numpy as np


def minimum_image_vector(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Return exact closest lattice images, including for skewed cells.

    Fractional rounding is only exact for orthogonal cells.  Here it supplies an
    initial upper bound; reciprocal-vector norms then give a finite search box
    guaranteed to contain every lattice translation capable of improving it.
    """
    cell = np.asarray(cell, dtype=float)
    inverse = np.linalg.inv(cell)
    values = np.asarray(delta, dtype=float)
    original_shape = values.shape
    rows = values.reshape(-1, 3)
    result = np.empty_like(rows)
    reciprocal_norms = np.linalg.norm(inverse, axis=0)
    for row_index, vector in enumerate(rows):
        fractional = vector @ inverse
        centre = np.rint(-fractional).astype(int)
        initial = vector + centre @ cell
        radius = float(np.linalg.norm(initial))
        bounds = np.ceil(radius * reciprocal_norms + 1e-12).astype(int) + 1
        best, best_squared = initial, float(initial @ initial)
        ranges = [range(centre[i] - bounds[i], centre[i] + bounds[i] + 1) for i in range(3)]
        for translation_index in product(*ranges):
            candidate = vector + np.asarray(translation_index) @ cell
            squared = float(candidate @ candidate)
            if squared < best_squared:
                best, best_squared = candidate, squared
        result[row_index] = best
    return result.reshape(original_shape)


def unwrap_points(points: np.ndarray, cell: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if not len(points):
        raise ValueError("a trajectory must contain at least one point")
    unwrapped = [points[0]]
    for point in points[1:]:
        unwrapped.append(unwrapped[-1] + minimum_image_vector(point - unwrapped[-1], cell))
    return np.asarray(unwrapped)


def periodic_distances(points: np.ndarray, centre: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Minimum-image distances from all points to one centre."""
    return np.linalg.norm(minimum_image_vector(np.asarray(points) - centre, cell), axis=1)


def point_segment_projection(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> tuple[float, float]:
    direction = end - start
    length_squared = float(direction @ direction)
    if length_squared < 1e-14:
        return float(np.linalg.norm(point - start)), 0.0
    coordinate = np.clip(float((point - start) @ direction) / length_squared, 0.0, 1.0)
    return float(np.linalg.norm(point - (start + coordinate * direction))), float(coordinate)


def periodic_point_segment_projection(
    point: np.ndarray, start: np.ndarray, end: np.ndarray, cell: np.ndarray
) -> tuple[float, float]:
    """Exact closest distance from a periodic point lattice to one segment."""
    inverse = np.linalg.inv(cell)
    reciprocal_norms = np.linalg.norm(inverse, axis=0)
    desired_start = (start - point) @ inverse
    desired_end = (end - point) @ inverse
    centre = np.rint(0.5 * (desired_start + desired_end)).astype(int)
    best_distance, best_coordinate = point_segment_projection(point + centre @ cell, start, end)
    bounds = np.ceil(best_distance * reciprocal_norms + 1e-12).astype(int) + 1
    lower = np.floor(np.minimum(desired_start, desired_end)).astype(int) - bounds
    upper = np.ceil(np.maximum(desired_start, desired_end)).astype(int) + bounds
    ranges = [range(lower[i], upper[i] + 1) for i in range(3)]
    for translation in product(*ranges):
        distance, coordinate = point_segment_projection(
            point + np.asarray(translation) @ cell, start, end
        )
        if distance < best_distance:
            best_distance, best_coordinate = distance, coordinate
    return best_distance, best_coordinate


def atom_path_geometry(
    positions: np.ndarray, path: np.ndarray, cell: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Periodic atom-to-polyline distances and normalized closest arc positions."""
    positions = np.asarray(positions, dtype=float)
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        distances = periodic_distances(positions, path[0], cell)
        return distances.astype(np.float32), np.zeros(len(positions), dtype=np.float32)
    lengths = np.linalg.norm(path[1:] - path[:-1], axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    if total <= 1e-12:
        distances = periodic_distances(positions, path[0], cell)
        return distances.astype(np.float32), np.zeros(len(positions), dtype=np.float32)
    distances = np.full(len(positions), np.inf)
    coordinates = np.zeros(len(positions))
    for atom_index, atom in enumerate(positions):
        for segment_index, (start, end) in enumerate(zip(path[:-1], path[1:])):
            distance, coordinate = periodic_point_segment_projection(atom, start, end, cell)
            if distance < distances[atom_index]:
                distances[atom_index] = distance
                coordinates[atom_index] = (
                    cumulative[segment_index] + coordinate * lengths[segment_index]
                ) / total
    return distances.astype(np.float32), coordinates.astype(np.float32)


def periodic_radius_graph(
    positions: np.ndarray, cell: np.ndarray, cutoff: float
) -> tuple[np.ndarray, np.ndarray]:
    """Directed graph containing every periodic atom image inside ``cutoff``.

    Unlike a minimum-image graph, this includes multiple images of an atom pair
    and nonzero periodic self-images when the cell is sufficiently small.
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    try:
        from ase.neighborlist import primitive_neighbor_list

        senders, receivers, vectors = primitive_neighbor_list(
            "ijD",
            pbc=np.ones(3, dtype=bool),
            cell=cell,
            positions=positions,
            cutoff=cutoff,
            self_interaction=False,
        )
        return (
            np.asarray([senders, receivers], dtype=np.int64),
            np.asarray(vectors, dtype=np.float32).reshape(-1, 3),
        )
    except ImportError:
        pass

    # Dependency-free exact fallback for environments without ASE.
    senders: list[int] = []
    receivers: list[int] = []
    vectors: list[np.ndarray] = []
    inverse = np.linalg.inv(cell)
    reciprocal_norms = np.linalg.norm(inverse, axis=0)
    # If |r| <= cutoff, each fractional component is bounded by
    # cutoff * ||corresponding reciprocal vector||.
    bounds = np.ceil(cutoff * reciprocal_norms).astype(int) + 1
    for i, source in enumerate(positions):
        for j, target in enumerate(positions):
            fractional = (target - source) @ inverse
            centre = np.rint(-fractional).astype(int)
            ranges = [range(centre[k] - bounds[k], centre[k] + bounds[k] + 1) for k in range(3)]
            for offset_tuple in product(*ranges):
                offset = np.asarray(offset_tuple, dtype=int)
                if i == j and np.all(offset == 0):
                    continue
                vector = target + offset @ cell - source
                norm = float(np.linalg.norm(vector))
                if 1e-8 < norm <= cutoff:
                    senders.append(i)
                    receivers.append(j)
                    vectors.append(vector)
    edge_index = np.asarray([senders, receivers], dtype=np.int64)
    edge_vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, 3)
    return edge_index, edge_vectors
