#!/usr/bin/env python3
"""Compare all legacy and new LiTraj hyperedge constructions consistently."""

from litraj_hypergraphs.experiment import comparison_main


ALL_CANDIDATES = (
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


if __name__ == "__main__":
    comparison_main(ALL_CANDIDATES)
