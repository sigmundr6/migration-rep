#!/usr/bin/env python3
"""Compare the original LiTraj representations using the shared experiment stack.

This compatibility entry point now delegates geometry, data loading, sample
construction, modelling, training and evaluation to ``litraj_hypergraphs``.
Consequently its results can be compared directly with the four new scripts.
"""

from litraj_hypergraphs.experiment import comparison_main


LEGACY_CANDIDATES = (
    "crystal",
    "midpoint",
    "trajectory",
    "positional",
    "segmented",
    "randomized",
)


if __name__ == "__main__":
    comparison_main(LEGACY_CANDIDATES)
