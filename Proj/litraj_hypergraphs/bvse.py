"""Shared helpers for BVSE-NEB generation and validation."""

from __future__ import annotations

import numpy as np
from ase.optimize import FIRE


def optimize_bvse_band(images, *, fmax: float, steps: int, distort: bool = True):
    """Run the same SaddleFinder/FIRE operation used by the LiTraj notebook."""
    try:
        from ions.tools import SaddleFinder
    except ImportError as exc:
        raise RuntimeError("install the LiTraj-compatible dependency: ions==0.4.1") from exc
    saddle = SaddleFinder()
    neb = saddle.bvse_neb(images, distort=distort)
    converged = bool(FIRE(neb, logfile=None).run(fmax=fmax, steps=steps))
    forces = np.asarray(neb.get_forces())
    max_force = float(np.abs(forces).max()) if forces.size else 0.0
    profile = np.asarray(saddle.get_profile(images), dtype=float)
    return {
        "barrier": float(saddle.get_barrier(images)),
        "max_force": max_force,
        "converged": converged or max_force <= fmax,
        "profile": profile,
    }


def evaluate_bvse_band(images):
    """Evaluate a stored band without changing its geometry."""
    try:
        from ions.tools import SaddleFinder
    except ImportError as exc:
        raise RuntimeError("install the LiTraj-compatible dependency: ions==0.4.1") from exc
    saddle = SaddleFinder()
    saddle.bvse_neb(images, distort=False)
    profile = np.asarray(saddle.get_profile(images), dtype=float)
    return {"barrier": float(saddle.get_barrier(images)), "profile": profile}
