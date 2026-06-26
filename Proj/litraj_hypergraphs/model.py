"""Shared capacity-controlled model for candidate comparisons."""

from __future__ import annotations

import math

import torch
from torch import nn

from .candidates import HypergraphSample


class CrystalEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 48, layers: int = 2):
        super().__init__()
        self.elements = nn.Embedding(119, hidden_dim)
        self.messages = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(layers)
        ])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(layers)
        ])

    def encode_graph(
        self,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        h = self.elements(atomic_numbers.clamp(0, 118))
        if edge_index.numel() == 0:
            return h
        send, receive = edge_index
        distance = edge_vectors.norm(dim=-1, keepdim=True)
        for message, update in zip(self.messages, self.updates):
            # index_select uses the dedicated gather path on MPS. Generic
            # advanced indexing (h[send]) repeatedly entered index_Tensor and
            # triggered an allocator/libdispatch crash in long image runs.
            sent = torch.index_select(h, 0, send)
            received = torch.index_select(h, 0, receive)
            m = message(torch.cat([sent, received, distance], dim=-1))
            aggregate = torch.zeros_like(h)
            aggregate.index_add_(0, receive, m)
            degree = torch.zeros(len(h), dtype=h.dtype, device=h.device)
            degree.index_add_(0, receive, torch.ones(len(receive), dtype=h.dtype, device=h.device))
            h = h + update(torch.cat([h, aggregate / degree.clamp_min(1)[:, None]], dim=-1))
        return h

    def forward(self, sample: HypergraphSample) -> torch.Tensor:
        return self.encode_graph(sample.atomic_numbers, sample.edge_index, sample.edge_vectors)


class CandidateBarrierModel(nn.Module):
    """Pools arbitrary typed hyperedges, then processes their temporal steps."""

    def __init__(
        self,
        hidden_dim: int = 48,
        layers: int = 2,
        *,
        reversal_invariant: bool = False,
        presence_threshold: float = 1e-6,
        auxiliary_mode: str = "none",
        delta_slope: float = 1.0,
        delta_intercept: float = 0.0,
    ):
        super().__init__()
        self.reversal_invariant = reversal_invariant
        self.presence_threshold = presence_threshold
        if auxiliary_mode not in {"none", "fusion", "delta"}:
            raise ValueError(f"unknown auxiliary mode: {auxiliary_mode}")
        self.auxiliary_mode = auxiliary_mode
        self.delta_slope = float(delta_slope)
        self.delta_intercept = float(delta_intercept)
        self.encoder = CrystalEncoder(hidden_dim, layers)
        self.null_context = nn.Parameter(torch.zeros(hidden_dim))
        self.edge_types = nn.Embedding(3, hidden_dim)
        self.edge_update = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.positional_update = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.trajectory_graph_updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(max(1, layers))
        ])
        readout_inputs = 2 * hidden_dim + (1 if auxiliary_mode == "fusion" else 0)
        self.readout = nn.Sequential(
            nn.Linear(readout_inputs, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    @staticmethod
    def weighted_pool(h: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.clamp_min(0)
        return (weights[:, None] * h).sum(0) / weights.sum().clamp_min(1e-8)

    def event_context(self, h: torch.Tensor, sample: HypergraphSample) -> torch.Tensor:
        candidate = str(sample.metadata["candidate"])
        if candidate == "crystal":
            return self.null_context
        if candidate in {"positional", "positional_shuffled_s"}:
            def encode(position: torch.Tensor) -> torch.Tensor:
                features = torch.stack([
                    sample.path_distance,
                    torch.sin(math.pi * position),
                    torch.cos(math.pi * position),
                    torch.sin(2 * math.pi * position),
                    torch.cos(2 * math.pi * position),
                ], dim=-1)
                encoded = self.positional_update(torch.cat([h, features], dim=-1))
                return self.weighted_pool(encoded, sample.incidence[:, 0])

            forward = encode(sample.path_position)
            if not self.reversal_invariant:
                return forward
            return 0.5 * (forward + encode(1 - sample.path_position))

        if candidate in {
            "position_aware_trajectory_graph",
            "position_aware_trajectory_graph_shuffled",
        }:
            return self.position_aware_trajectory_context(sample)

        present = sample.incidence.sum(dim=0) > self.presence_threshold
        if not bool(present.any()):
            present[torch.argmax(sample.incidence.sum(dim=0))] = True
        selected = torch.nonzero(present, as_tuple=False).flatten()
        incidence = torch.index_select(sample.incidence, 1, selected)
        steps = torch.index_select(sample.hyperedge_steps, 0, selected)
        types = torch.index_select(sample.hyperedge_types, 0, selected)
        dynamic = candidate in {
            "image_coordination", "species_image", "bottleneck", "geometric_bottleneck"
        }
        frame_embeddings: dict[int, torch.Tensor] = {}
        pooled = []
        for index in range(incidence.shape[1]):
            step = int(steps[index].item())
            if dynamic:
                if step not in frame_embeddings:
                    frame_embeddings[step] = self.encoder.encode_graph(
                        sample.atomic_numbers,
                        sample.frame_edge_indices[step],
                        sample.frame_edge_vectors[step],
                    )
                atom_embeddings = frame_embeddings[step]
            else:
                atom_embeddings = h
            pooled.append(self.weighted_pool(atom_embeddings, incidence[:, index]))
        edges = torch.stack(pooled)
        edges = self.edge_update(edges + self.edge_types(types.clamp(0, 2)))
        unique_steps = torch.unique(steps, sorted=True)
        sequence = torch.stack([
            edges[steps == step].mean(0) for step in unique_steps
        ])[None, :, :]

        def encode(values: torch.Tensor) -> torch.Tensor:
            _, state = self.temporal(values)
            return state[-1, 0]

        forward = encode(sequence)
        if not self.reversal_invariant:
            return forward
        return 0.5 * (forward + encode(torch.flip(sequence, dims=[1])))

    def position_aware_trajectory_context(self, sample: HypergraphSample) -> torch.Tensor:
        """Atom -> image hyperedge -> chain-graph hierarchy with absolute position."""
        present = sample.incidence.sum(dim=0) > self.presence_threshold
        if not bool(present.any()):
            present[torch.argmax(sample.incidence.sum(dim=0))] = True
        columns = torch.nonzero(present, as_tuple=False).flatten()

        def build_nodes(positions: torch.Tensor) -> torch.Tensor:
            nodes = []
            for column in columns:
                column_index = int(column.item())
                frame_index = int(sample.hyperedge_frames[column_index].item())
                atoms = self.encoder.encode_graph(
                    sample.atomic_numbers,
                    sample.frame_edge_indices[frame_index],
                    sample.frame_edge_vectors[frame_index],
                )
                position = positions[column_index]
                distance = sample.hyperedge_distances[:, column_index]
                features = torch.stack([
                    distance,
                    torch.sin(math.pi * position).expand_as(distance),
                    torch.cos(math.pi * position).expand_as(distance),
                    torch.sin(2 * math.pi * position).expand_as(distance),
                    torch.cos(2 * math.pi * position).expand_as(distance),
                ], dim=-1)
                local_atoms = self.positional_update(torch.cat([atoms, features], dim=-1))
                node = self.weighted_pool(local_atoms, sample.incidence[:, column_index])
                nodes.append(node)
            nodes = torch.stack(nodes)
            steps = sample.hyperedge_steps[columns]
            order = torch.argsort(steps)
            nodes = nodes[order]
            for update in self.trajectory_graph_updates:
                neighbours = torch.zeros_like(nodes)
                degree = torch.zeros(len(nodes), dtype=nodes.dtype, device=nodes.device)
                if len(nodes) > 1:
                    neighbours[1:] += nodes[:-1]
                    neighbours[:-1] += nodes[1:]
                    degree[1:] += 1
                    degree[:-1] += 1
                aggregated = neighbours / degree.clamp_min(1)[:, None]
                nodes = nodes + update(torch.cat([nodes, aggregated], dim=-1))
            return nodes.mean(dim=0)

        forward = build_nodes(sample.hyperedge_positions)
        if not self.reversal_invariant:
            return forward
        reverse = build_nodes(1 - sample.hyperedge_positions)
        return 0.5 * (forward + reverse)

    def representation(self, sample: HypergraphSample) -> torch.Tensor:
        atoms = self.encoder(sample)
        candidate = str(sample.metadata["candidate"])
        # The focused positional control symmetrises only s <-> 1-s. Temporal
        # image candidates additionally need endpoint crystal averaging because
        # reversing their stored frames changes which image supplies the global
        # crystal embedding.
        if self.reversal_invariant and candidate not in {
            "positional", "positional_shuffled_s"
        }:
            endpoint_atoms = self.encoder.encode_graph(
                sample.atomic_numbers,
                sample.frame_edge_indices[-1],
                sample.frame_edge_vectors[-1],
            )
            atoms = 0.5 * (atoms + endpoint_atoms)
        event = self.event_context(atoms, sample)
        return torch.cat([atoms.mean(0), event])

    def forward(self, sample: HypergraphSample) -> torch.Tensor:
        representation = self.representation(sample)
        if self.auxiliary_mode == "none":
            return self.readout(representation).squeeze(-1)
        if not bool(torch.isfinite(sample.cheap_barrier)):
            raise ValueError(f"{sample.source}: BVSE barrier is required for {self.auxiliary_mode}")
        if self.auxiliary_mode == "fusion":
            features = torch.cat([representation, sample.cheap_barrier.reshape(1)])
            return self.readout(features).squeeze(-1)
        calibrated = self.delta_slope * sample.cheap_barrier + self.delta_intercept
        return calibrated + self.readout(representation).squeeze(-1)
