"""Fast construction-level regression tests; run with ``python -m unittest``."""

from dataclasses import replace
import unittest

import numpy as np

import torch

from .candidates import CANDIDATES, build_candidate_sample
from .geometry import minimum_image_vector, periodic_radius_graph
from .model import CandidateBarrierModel


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.numbers = np.asarray([3, 8, 15, 3])
        self.cell = np.diag([8.0, 8.0, 8.0])
        self.frames = np.asarray([
            [[1.0, 4.0, 4.0], [3.0, 4.8, 4.0], [5.0, 3.0, 4.0], [7.0, 7.0, 7.0]],
            [[3.0, 4.0, 4.0], [3.0, 4.7, 4.0], [5.0, 3.0, 4.0], [7.0, 7.0, 7.0]],
            [[5.0, 4.0, 4.0], [3.0, 4.8, 4.0], [5.0, 3.0, 4.0], [7.0, 7.0, 7.0]],
        ])

    def build(self, candidate):
        return build_candidate_sample(
            candidate, self.numbers, self.frames, self.cell, 0, 0.4,
            hyperedge_cutoff=3.0, hyperedge_sigma=1.5,
        )

    def test_all_candidates_produce_valid_incidence(self):
        for name in CANDIDATES:
            with self.subTest(candidate=name):
                sample = self.build(name)
                self.assertEqual(sample.incidence.shape[0], len(self.numbers))
                self.assertEqual(sample.incidence.shape[1], len(sample.hyperedge_steps))
                self.assertGreater(sample.incidence.sum().item(), 0)
                self.assertTrue(np.allclose(sample.incidence[0].numpy(), 0.0))

    def test_image_candidate_has_one_edge_per_image(self):
        self.assertEqual(self.build("image_coordination").incidence.shape[1], len(self.frames))

    def test_bottleneck_uses_no_energy_information(self):
        first = self.build("bottleneck")
        second = build_candidate_sample(
            "bottleneck", self.numbers, self.frames, self.cell, 0, 9.9,
            hyperedge_cutoff=3.0, hyperedge_sigma=1.5,
        )
        self.assertTrue(np.allclose(first.incidence.numpy(), second.incidence.numpy()))
        self.assertEqual(first.metadata["bottleneck_image"], second.metadata["bottleneck_image"])

    def test_species_edges_are_typed(self):
        types = set(self.build("species_image").hyperedge_types.tolist())
        self.assertTrue(types.issubset({0, 1, 2}))
        self.assertIn(0, types)
        self.assertIn(1, types)

    def test_sample_moves_all_tensors(self):
        sample = self.build("image_coordination").to("cpu")
        direct = [
            sample.atomic_numbers, sample.positions, sample.edge_index,
            sample.edge_vectors, sample.frames, sample.trajectory,
            sample.incidence, sample.hyperedge_steps, sample.hyperedge_types,
            sample.hyperedge_frames, sample.hyperedge_positions, sample.hyperedge_distances,
            sample.path_distance, sample.path_position, sample.target,
        ]
        nested = [*sample.frame_edge_indices, *sample.frame_edge_vectors]
        self.assertTrue(all(value.device.type == "cpu" for value in direct + nested))

    def test_legacy_path_candidates_share_path_geometry(self):
        trajectory = self.build("trajectory")
        positional = self.build("positional")
        self.assertTrue(np.allclose(trajectory.incidence.numpy(), positional.incidence.numpy()))
        self.assertTrue(np.allclose(trajectory.path_position.numpy(), positional.path_position.numpy()))

    def test_shuffled_s_preserves_supported_values_but_changes_association(self):
        positional = build_candidate_sample(
            "positional", self.numbers, self.frames, self.cell, 0, 0.4,
            hyperedge_cutoff=8.0, hyperedge_sigma=1.5, source="fixed-hop",
        )
        shuffled = build_candidate_sample(
            "positional_shuffled_s", self.numbers, self.frames, self.cell, 0, 0.4,
            hyperedge_cutoff=8.0, hyperedge_sigma=1.5, source="fixed-hop",
        )
        supported = torch.nonzero(shuffled.incidence[:, 0] > 0).flatten()
        self.assertTrue(torch.allclose(positional.incidence, shuffled.incidence))
        self.assertTrue(torch.allclose(
            torch.sort(positional.path_position[supported]).values,
            torch.sort(shuffled.path_position[supported]).values,
        ))
        self.assertFalse(torch.allclose(
            positional.path_position[supported], shuffled.path_position[supported]
        ))

    def test_positional_invariance_changes_only_event_orientation(self):
        sample = self.build("positional")
        reversed_sample = replace(sample, path_position=1 - sample.path_position)
        torch.manual_seed(22)
        model = CandidateBarrierModel(8, 1, reversal_invariant=True)
        self.assertTrue(torch.allclose(model(sample), model(reversed_sample), atol=1e-6))

    def test_segment_count_is_configurable(self):
        sample = build_candidate_sample(
            "segmented", self.numbers, self.frames, self.cell, 0, 0.4,
            hyperedge_cutoff=3.0, hyperedge_sigma=1.5, num_segments=3,
        )
        self.assertEqual(sample.incidence.shape[1], 3)

    def test_position_aware_graph_uses_physical_image_coordinates(self):
        frames = self.frames.copy()
        frames[1, 0, 0] = 2.0
        sample = build_candidate_sample(
            "position_aware_trajectory_graph", self.numbers, frames, self.cell, 0, 0.4,
            hyperedge_cutoff=3.0, hyperedge_sigma=1.5,
        )
        self.assertTrue(torch.equal(sample.hyperedge_frames, torch.arange(len(frames))))
        self.assertTrue(torch.allclose(
            sample.hyperedge_positions, torch.tensor([0.0, 0.25, 1.0])
        ))

    def test_shuffled_control_preserves_nodes_but_changes_chain_order(self):
        ordered = self.build("position_aware_trajectory_graph")
        shuffled = self.build("position_aware_trajectory_graph_shuffled")
        self.assertTrue(torch.allclose(ordered.incidence, shuffled.incidence))
        self.assertTrue(torch.equal(ordered.hyperedge_frames, shuffled.hyperedge_frames))
        self.assertTrue(torch.allclose(
            ordered.hyperedge_positions, shuffled.hyperedge_positions
        ))
        self.assertFalse(torch.equal(ordered.hyperedge_steps, shuffled.hyperedge_steps))

    def test_position_aware_graph_forward_and_backward(self):
        torch.manual_seed(12)
        model = CandidateBarrierModel(8, 1)
        for name in (
            "position_aware_trajectory_graph",
            "position_aware_trajectory_graph_shuffled",
        ):
            with self.subTest(candidate=name):
                model.zero_grad(set_to_none=True)
                prediction = model(self.build(name))
                self.assertTrue(torch.isfinite(prediction))
                prediction.backward()
                self.assertTrue(any(
                    parameter.grad is not None and torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ))

    def test_minimum_image_is_correct_for_skewed_cell(self):
        cell = np.asarray([[3.0, 0.0, 0.0], [2.8, 3.0, 0.0], [0.0, 0.0, 4.0]])
        displacement = np.asarray([-4.08084058, 1.00100526, 2.28560527])
        closest = minimum_image_vector(displacement, cell)
        self.assertTrue(np.allclose(closest, [-1.08084058, 1.00100526, -1.71439473]))

    def test_periodic_graph_includes_nonzero_self_images(self):
        edges, vectors = periodic_radius_graph(
            np.asarray([[0.0, 0.0, 0.0]]), np.eye(3) * 2.0, 2.1
        )
        self.assertEqual(edges.shape[1], 6)
        self.assertTrue(np.all(edges == 0))
        self.assertTrue(np.allclose(np.linalg.norm(vectors, axis=1), 2.0))

    def test_randomization_preserves_framework_distribution_and_li_exclusion(self):
        trajectory = self.build("trajectory")
        randomized = self.build("randomized")
        framework = np.arange(len(self.numbers)) != 0
        self.assertEqual(float(randomized.incidence[0, 0]), 0.0)
        self.assertTrue(np.allclose(
            np.sort(trajectory.incidence[framework, 0].numpy()),
            np.sort(randomized.incidence[framework, 0].numpy()),
        ))

    def test_empty_segments_are_removed_before_gru(self):
        sample = build_candidate_sample(
            "segmented", self.numbers, self.frames, self.cell, 0, 0.4,
            hyperedge_cutoff=3.0, hyperedge_sigma=1.5, num_segments=3,
        )
        incidence = sample.incidence.clone()
        incidence[:, 1] = 1e-12
        sample = replace(sample, incidence=incidence)
        seen = []
        model = CandidateBarrierModel(8, 1, presence_threshold=1e-6)
        handle = model.temporal.register_forward_pre_hook(lambda _module, args: seen.append(args[0].shape[1]))
        model(sample)
        handle.remove()
        self.assertEqual(seen, [2])

    def test_reversal_invariance_is_explicit_opt_in(self):
        sample = self.build("image_coordination")
        reverse = build_candidate_sample(
            "image_coordination", self.numbers, self.frames[::-1].copy(), self.cell, 0, 0.4,
            hyperedge_cutoff=3.0, hyperedge_sigma=1.5,
        )
        torch.manual_seed(4)
        model = CandidateBarrierModel(8, 1, reversal_invariant=True)
        self.assertAlmostEqual(float(model(sample)), float(model(reverse)), places=6)

    def test_image_model_uses_frame_specific_geometry(self):
        sample = self.build("image_coordination")
        altered_vectors = list(sample.frame_edge_vectors)
        altered_vectors[1] = altered_vectors[1] * 1.7
        altered = replace(sample, frame_edge_vectors=tuple(altered_vectors))
        torch.manual_seed(9)
        model = CandidateBarrierModel(8, 1)
        self.assertNotAlmostEqual(float(model(sample)), float(model(altered)), places=7)

    def test_scalar_fusion_and_delta_learning_backward(self):
        sample = replace(self.build("positional"), cheap_barrier=torch.tensor(0.7))
        for mode in ("fusion", "delta"):
            with self.subTest(mode=mode):
                model = CandidateBarrierModel(
                    8, 1, auxiliary_mode=mode, delta_slope=0.5, delta_intercept=0.1
                )
                prediction = model(sample)
                self.assertTrue(torch.isfinite(prediction))
                prediction.backward()
                self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
